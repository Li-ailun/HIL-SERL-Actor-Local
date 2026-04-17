import os
import time
import queue
import threading
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from serl_robot_infra.Galaxea_env.communication.ros2_bridge import Ros2Bridge
from serl_robot_infra.Galaxea_env.camera.rs_capture import RSCapture
from serl_robot_infra.Galaxea_env.camera.video_capture import VideoCapture
from serl_robot_infra.Galaxea_env.camera.multi_video_capture import MultiVideoCapture
from serl_robot_infra.Galaxea_env.envs.utils.rotations import (
    apply_delta_rotation,
    clip_rotation,
)


class GalaxeaImageDisplayer(threading.Thread):
    """
    异步图像显示线程，避免阻塞主控制循环。

    改进点：
    1) 与原版相比，推荐 env 侧用有界队列（maxsize=1），避免显示线程跟不上时无限堆积。
    2) 这里只负责显示，不参与控制逻辑。
    """

    def __init__(
        self,
        queue_obj,
        image_keys,
        obs_image_size,
        window_name,
        window_size,
        display_frame_size,
    ):
        super().__init__()
        self.queue = queue_obj
        self.daemon = True

        self.image_keys = list(image_keys)
        self.obs_image_size = tuple(obs_image_size)
        self.window_name = window_name
        self.window_size = tuple(window_size)
        self.display_frame_size = tuple(display_frame_size)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

        h, w = self.obs_image_size
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        while True:
            img_dict = self.queue.get()
            if img_dict is None:
                break

            frame = np.concatenate(
                [img_dict.get(k, blank) for k in self.image_keys],
                axis=1,
            )

            display_frame = cv2.resize(
                frame,
                self.display_frame_size,
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(1)


class GalaxeaArmEnv(gym.Env):
    """
    Galaxea 统一底层环境：同时支持双臂与单臂。

    设计目标
    ----------
    1. 兼容你当前的双臂任务，不要求立刻把上层 rlpd / wrapper / config 全部重写。
    2. 通过 config.ARM_MODE 来控制：
       - "dual"   : 双臂模式
       - "single" : 单臂模式
    3. 单臂模式下通过 config.ARM_SIDE 指定控制哪一侧：
       - "left"
       - "right"

    与旧版 dual env 的兼容策略
    --------------------------
    1. 类名改成更通用的 GalaxeaArmEnv。
    2. 文件末尾保留：
         GalaxeaDualArmEnv = GalaxeaArmEnv
       这样旧代码如果还 import GalaxeaDualArmEnv，不会立刻炸。
    3. 对 reset pose 做兼容：
       - dual 模式优先读 RESET_L / RESET_R
       - single 模式优先读 RESET_POSE
       - 如果 single 模式没给 RESET_POSE，则尝试从 RESET_L/RESET_R 里按 ARM_SIDE 回退读取
    """

    metadata = {"render_modes": []}

    def __init__(self, config, cfg, save_video=False):
        super().__init__()

        self.config = config

        # ==========================================================
        # 0. 模式配置
        # ==========================================================
        self.arm_mode = str(getattr(self.config, "ARM_MODE", "dual")).lower()
        if self.arm_mode not in ("dual", "single"):
            raise ValueError(f"config.ARM_MODE 必须是 'dual' 或 'single'，当前为 {self.arm_mode!r}")

        # 单臂时指定 left / right；双臂时这个字段可忽略
        self.arm_side = str(getattr(self.config, "ARM_SIDE", "right")).lower()
        if self.arm_mode == "single" and self.arm_side not in ("left", "right"):
            raise ValueError(f"单臂模式下 config.ARM_SIDE 必须是 'left' 或 'right'，当前为 {self.arm_side!r}")

        # ==========================================================
        # 1. 通用运行参数（必须从 config 读取）
        # ==========================================================
        self.display_images = self._require_config_attr("DISPLAY_IMAGES")
        self.save_video = save_video
        self.hz = self._require_config_attr("HZ")
        self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
        self.curr_path_length = 0

        # 图像配置
        self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
        self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
        self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

        # 显示窗口配置
        self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
        self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
        self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

        # 头部相机 key
        self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

        # ==========================================================
        # 2. 复位坐标
        # ==========================================================
        if self.arm_mode == "dual":
            self.reset_l = np.array(self._require_config_attr("RESET_L"), dtype=np.float32)
            self.reset_r = np.array(self._require_config_attr("RESET_R"), dtype=np.float32)
        else:
            if hasattr(self.config, "RESET_POSE"):
                self.reset_pose = np.array(self.config.RESET_POSE, dtype=np.float32)
            else:
                # 兼容旧双臂配置：单臂模式下如果没有 RESET_POSE，就从 RESET_L/RESET_R 兜底
                fallback_key = "RESET_L" if self.arm_side == "left" else "RESET_R"
                self.reset_pose = np.array(self._require_config_attr(fallback_key), dtype=np.float32)

        # ==========================================================
        # 3. 动作缩放
        # ==========================================================
        self.pos_scale = float(self._require_config_attr("POS_SCALE"))
        self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

        # ==========================================================
        # 4. 工作空间限位
        # ==========================================================
        xyz_low = np.asarray(self._require_config_attr("XYZ_LIMIT_LOW"), dtype=np.float64)
        xyz_high = np.asarray(self._require_config_attr("XYZ_LIMIT_HIGH"), dtype=np.float64)
        rpy_low = np.asarray(self._require_config_attr("RPY_LIMIT_LOW"), dtype=np.float64)
        rpy_high = np.asarray(self._require_config_attr("RPY_LIMIT_HIGH"), dtype=np.float64)

        self.xyz_bounding_box = gym.spaces.Box(xyz_low, xyz_high, dtype=np.float64)
        self.rpy_bounding_box = gym.spaces.Box(rpy_low, rpy_high, dtype=np.float64)

        # ==========================================================
        # 5. 录像缓冲区
        # ==========================================================
        self.recording_frames = []
        if self.save_video:
            print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")

        # ==========================================================
        # 6. 全局 ESC 急停
        # ==========================================================
        self.terminate = False
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.esc:
                    print("🛑 检测到 ESC，触发全局紧急终止！")
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            print("⚠️ 未安装 pynput，ESC 紧急停止功能不可用。(pip install pynput)")

        # ==========================================================
        # 7. ROS2 通信
        # ==========================================================
        self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

        # ==========================================================
        # 8. 相机初始化
        # ==========================================================
        print("正在启动 USB 直连相机阵列...")
        self.multi_cap = self._build_multi_camera_capture()
        print("相机阵列启动完毕！")

        # ==========================================================
        # 9. 显示线程
        # ==========================================================
        if self.display_images:
            # 官方风格完善点：使用有界队列，避免显示堆积
            self.img_queue = queue.Queue(maxsize=1)
            self.displayer = GalaxeaImageDisplayer(
                self.img_queue,
                image_keys=self.display_image_keys,
                obs_image_size=self.obs_image_size,
                window_name=self.display_window_name,
                window_size=self.display_window_size,
                display_frame_size=self.display_frame_size,
            )
            self.displayer.start()

        # ==========================================================
        # 10. Gym 空间定义
        # ==========================================================
        h, w = self.obs_image_size
        image_space = {
            key: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
            for key in self.image_keys
        }

        if self.arm_mode == "dual":
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(14,),
                dtype=np.float32,
            )
            state_space = {
                "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            )
            ee_key = f"{self.arm_side}_ee_pose"
            gripper_key = f"{self.arm_side}_gripper"
            state_space = {
                ee_key: gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                gripper_key: gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_space),
                "images": gym.spaces.Dict(image_space),
            }
        )

        self._last_valid_state = None
        self._printed_missing_state_warning = False

    def _require_config_attr(self, name: str):
        """
        强制要求 config 提供字段。
        缺少任何任务相关配置时，直接报错。
        """
        if not hasattr(self.config, name):
            raise AttributeError(f"GalaxeaArmEnv 初始化失败：缺少必需配置项 config.{name}")
        return getattr(self.config, name)

    # ==========================================================
    # 相机构建
    # ==========================================================
    def _build_multi_camera_capture(self):
        realsense_cameras = self._require_config_attr("REALSENSE_CAMERAS")
        head_camera_cfg = self._require_config_attr("HEAD_CAMERA")

        caps = {}

        for cam_name, cam_cfg in realsense_cameras.items():
            rs_kwargs = dict(cam_cfg)
            rs_kwargs.setdefault("name", cam_name)
            caps[cam_name] = RSCapture(**rs_kwargs)

        required_head_keys = ["device_index", "api", "name", "split_left_half"]
        for key in required_head_keys:
            if key not in head_camera_cfg:
                raise AttributeError(f"GalaxeaArmEnv 初始化失败：HEAD_CAMERA 缺少必需字段 '{key}'")

        device_index = head_camera_cfg["device_index"]
        api = head_camera_cfg["api"]

        head_cv2 = cv2.VideoCapture(device_index, api)

        fourcc = head_camera_cfg.get("fourcc")
        if fourcc:
            head_cv2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        if "frame_width" in head_camera_cfg:
            head_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, head_camera_cfg["frame_width"])
        if "frame_height" in head_camera_cfg:
            head_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, head_camera_cfg["frame_height"])
        if "fps" in head_camera_cfg:
            head_cv2.set(cv2.CAP_PROP_FPS, head_camera_cfg["fps"])

        if not head_cv2.isOpened():
            raise RuntimeError(f"无法通过 /dev/video{device_index} 打开头部相机")

        head_camera_name = head_camera_cfg["name"]
        caps[head_camera_name] = VideoCapture(head_cv2, name=head_camera_name)

        self.head_camera_cfg = head_camera_cfg
        return MultiVideoCapture(caps)

    # ==========================================================
    # 安全 / 复位
    # ==========================================================
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """检查并裁剪 7 维位姿，确保其在安全软限位内。"""
        pose[:3] = np.clip(
            pose[:3],
            self.xyz_bounding_box.low,
            self.xyz_bounding_box.high,
        )
        pose[3:] = clip_rotation(
            pose[3:],
            self.rpy_bounding_box.low,
            self.rpy_bounding_box.high,
        )
        return pose

    def interpolate_move_dual(self, goal_l: np.ndarray, goal_r: np.ndarray, timeout: float):
        """双臂线性插值平滑复位。"""
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        curr_l = self._last_valid_state["left_ee_pose"].copy()
        curr_r = self._last_valid_state["right_ee_pose"].copy()

        path_l = np.linspace(curr_l, goal_l, steps)
        path_r = np.linspace(curr_r, goal_r, steps)

        print(f"🤖 正在执行双臂平滑复位 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            self._send_ros_poses(path_l[i], path_r[i], g_l=1.0, g_r=1.0)
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def interpolate_move_single(self, goal_pose: np.ndarray, timeout: float, gripper: float = 1.0):
        """单臂线性插值平滑复位。"""
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        ee_key = f"{self.arm_side}_ee_pose"
        curr_pose = self._last_valid_state[ee_key].copy()

        path = np.linspace(curr_pose, goal_pose, steps)

        print(f"🤖 正在执行单臂平滑复位 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            self._send_ros_pose(self.arm_side, path[i], gripper)
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def go_to_reset(self):
        """
        默认复位流程。
        任务子类可以继续重写，但底层默认已经支持单双臂。
        """
        print("开始复位")
        time.sleep(0.3)

        if self.arm_mode == "dual":
            self.interpolate_move_dual(self.reset_l, self.reset_r, timeout=1.5)
        else:
            self.interpolate_move_single(self.reset_pose, timeout=1.5, gripper=1.0)

        time.sleep(0.5)

    def reset(self, **kwargs):
        """Gym reset 接口。"""
        if self.save_video:
            self.save_video_recording()

        self.curr_path_length = 0
        self.terminate = False

        self.go_to_reset()
        self._last_valid_state = None
        obs = self._get_sync_obs()
        return obs, {"succeed": False}
    
    

    # ==========================================================
    # 主 step / close
    # ==========================================================
    def step(self, action: np.ndarray):
        """
        Gymnasium step。

        官方风格完善点：
        - 先 clip action
        - 区分 terminated / truncated
        """
        start_time = time.time()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = 14 if self.arm_mode == "dual" else 7

        if action.shape[0] != expected_dim:
            raise ValueError(
                f"动作维度错误：当前 arm_mode={self.arm_mode}, "
                f"期望 {expected_dim} 维，实际收到 {action.shape}"
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._publish_action(action)

        self.curr_path_length += 1

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        obs = self._get_sync_obs()
        reward = 0

        terminated = self.terminate
        truncated = self.curr_path_length >= self.max_episode_length

        return obs, reward, terminated, truncated, {}

    def close(self):
        """优雅关闭所有硬件资源。"""
        if hasattr(self, "listener"):
            self.listener.stop()

        self.bridge.destroy()
        self.multi_cap.close()

        if self.display_images:
            try:
                self.img_queue.put_nowait(None)
            except queue.Full:
                pass
            cv2.destroyAllWindows()
            self.displayer.join()

    # ==========================================================
    # 视频保存
    # ==========================================================
    def save_video_recording(self):
        """将内存中的图像序列编码并写入硬盘文件。"""
        try:
            if len(self.recording_frames) > 0:
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    video_path = f"./videos/{camera_key}_{timestamp}.mp4"

                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.hz,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"🎥 视频已保存: {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"⚠️ 视频保存失败: {e}")

    # ==========================================================
    # 动作发布
    # ==========================================================
    def _publish_action(self, action: np.ndarray):
        """
        将归一化动作映射为真实物理增量。

        dual 模式:
            [left(6)+left_gripper(1)+right(6)+right_gripper(1)] = 14
        single 模式:
            [arm(6)+gripper(1)] = 7
        """
        if self._last_valid_state is None:
            if not self._printed_missing_state_warning:
                print("⚠️ _last_valid_state 还未准备好，本次动作发布被跳过。")
                self._printed_missing_state_warning = True
            return

        if self.arm_mode == "dual":
            left_arm_action = action[0:6]
            left_gripper_action = action[6]
            right_arm_action = action[7:13]
            right_gripper_action = action[13]

            l_pose = self._last_valid_state["left_ee_pose"].copy()
            next_xyz_l = l_pose[:3] + left_arm_action[0:3] * self.pos_scale
            next_quat_l = apply_delta_rotation(
                l_pose[3:7],
                left_arm_action[3:6] * self.rot_scale,
            )
            clipped_pose_l = self.clip_safety_box(np.concatenate([next_xyz_l, next_quat_l]))

            r_pose = self._last_valid_state["right_ee_pose"].copy()
            next_xyz_r = r_pose[:3] + right_arm_action[0:3] * self.pos_scale
            next_quat_r = apply_delta_rotation(
                r_pose[3:7],
                right_arm_action[3:6] * self.rot_scale,
            )
            clipped_pose_r = self.clip_safety_box(np.concatenate([next_xyz_r, next_quat_r]))

            self._send_ros_poses(
                clipped_pose_l,
                clipped_pose_r,
                left_gripper_action,
                right_gripper_action,
            )
        else:
            arm_action = action[0:6]
            gripper_action = action[6]

            ee_key = f"{self.arm_side}_ee_pose"
            pose = self._last_valid_state[ee_key].copy()

            next_xyz = pose[:3] + arm_action[0:3] * self.pos_scale
            next_quat = apply_delta_rotation(
                pose[3:7],
                arm_action[3:6] * self.rot_scale,
            )
            clipped_pose = self.clip_safety_box(np.concatenate([next_xyz, next_quat]))

            self._send_ros_pose(self.arm_side, clipped_pose, gripper_action)

    def _send_ros_pose(self, arm_side: str, pose, gripper):
        """单臂发布标准 ROS2 PoseStamped / JointState。"""
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)

        msg_pose = PoseStamped()
        msg_pose.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_pose.header.frame_id = "base_link"
        msg_pose.pose.position.x = float(pose[0])
        msg_pose.pose.position.y = float(pose[1])
        msg_pose.pose.position.z = float(pose[2])
        msg_pose.pose.orientation.x = float(pose[3])
        msg_pose.pose.orientation.y = float(pose[4])
        msg_pose.pose.orientation.z = float(pose[5])
        msg_pose.pose.orientation.w = float(pose[6])

        msg_gripper = JointState()
        msg_gripper.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_gripper.name = [f"R1PRO_{arm_side}_gripper_joint"]
        msg_gripper.position = [float(gripper)]

        ee_key = f"{arm_side}_ee_pose"
        gripper_key = f"{arm_side}_gripper"

        try:
            self.bridge.publishers[self.bridge.topics_config.action[ee_key]].publish(msg_pose)
            self.bridge.publishers[self.bridge.topics_config.action[gripper_key]].publish(msg_gripper)
        except KeyError as e:
            print(f"⚠️ 发布单臂动作时出现异常: {e}")

    def _send_ros_poses(self, p_l, p_r, g_l, g_r):
        """双臂发布标准 ROS2 PoseStamped / JointState。"""
        p_l = np.asarray(p_l, dtype=np.float64).reshape(-1)
        p_r = np.asarray(p_r, dtype=np.float64).reshape(-1)

        msg_left = PoseStamped()
        msg_left.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_left.header.frame_id = "base_link"
        msg_left.pose.position.x = float(p_l[0])
        msg_left.pose.position.y = float(p_l[1])
        msg_left.pose.position.z = float(p_l[2])
        msg_left.pose.orientation.x = float(p_l[3])
        msg_left.pose.orientation.y = float(p_l[4])
        msg_left.pose.orientation.z = float(p_l[5])
        msg_left.pose.orientation.w = float(p_l[6])

        msg_right = PoseStamped()
        msg_right.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_right.header.frame_id = "base_link"
        msg_right.pose.position.x = float(p_r[0])
        msg_right.pose.position.y = float(p_r[1])
        msg_right.pose.position.z = float(p_r[2])
        msg_right.pose.orientation.x = float(p_r[3])
        msg_right.pose.orientation.y = float(p_r[4])
        msg_right.pose.orientation.z = float(p_r[5])
        msg_right.pose.orientation.w = float(p_r[6])

        msg_gripper_left = JointState()
        msg_gripper_left.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_gripper_left.name = ["R1PRO_left_gripper_joint"]
        msg_gripper_left.position = [float(g_l)]

        msg_gripper_right = JointState()
        msg_gripper_right.header.stamp = self.bridge.node.get_clock().now().to_msg()
        msg_gripper_right.name = ["R1PRO_right_gripper_joint"]
        msg_gripper_right.position = [float(g_r)]

        try:
            self.bridge.publishers[self.bridge.topics_config.action["left_ee_pose"]].publish(msg_left)
            self.bridge.publishers[self.bridge.topics_config.action["right_ee_pose"]].publish(msg_right)
            self.bridge.publishers[self.bridge.topics_config.action["left_gripper"]].publish(msg_gripper_left)
            self.bridge.publishers[self.bridge.topics_config.action["right_gripper"]].publish(msg_gripper_right)
        except KeyError as e:
            print(f"⚠️ 发布双臂动作时出现异常: {e}")

    # ==========================================================
    # 观测抓取
    # ==========================================================
    def _get_sync_obs(self):
        """
        强制从硬件抓取最新画面和 ROS 状态，并对齐返回。

        官方风格完善点：
        1) 对 ROS 状态等待加超时，避免永久卡死。
        2) 显示队列只保留最新一帧，避免堆积。
        """
        img_dict = self.multi_cap.read()
        if img_dict is None:
            img_dict = {}

        raw_state = None
        start_wait = time.time()
        timeout_s = 5.0

        while raw_state is None:
            raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
            if raw_state is None:
                if time.time() - start_wait > timeout_s:
                    raise TimeoutError("等待 ROS 状态超时（5 秒内未拿到有效状态）")
                time.sleep(0.01)

        formatted_obs = {"state": {}, "images": {}}
        full_res_images = {}
        display_images = {}

        state_src = raw_state.get("state", {}) if raw_state is not None else {}

        if self.arm_mode == "dual":
            state_keys = ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
        else:
            state_keys = [f"{self.arm_side}_ee_pose", f"{self.arm_side}_gripper"]

        for key in state_keys:
            if key in state_src:
                formatted_obs["state"][key] = state_src[key].numpy().flatten()
            else:
                dim = 7 if "pose" in key else 1
                formatted_obs["state"][key] = np.zeros(dim, dtype=np.float32)

        self._last_valid_state = formatted_obs["state"]

        h, w = self.obs_image_size

        for key in self.image_keys:
            if key in img_dict:
                img_bgr = img_dict[key]

                # 头部相机如果是左右拼接图，是否取左半边完全由 config 决定
                if key == self.head_camera_key and self.head_camera_cfg["split_left_half"]:
                    if img_bgr.shape[1] > img_bgr.shape[0] * 2:
                        width = img_bgr.shape[1]
                        img_bgr = img_bgr[:, : width // 2, :]

                full_res_images[key] = img_bgr.copy()

                try:
                    img_resized_bgr = cv2.resize(img_bgr, (w, h))
                    img_rgb = img_resized_bgr[..., ::-1]

                    formatted_obs["images"][key] = img_rgb
                    display_images[key] = img_resized_bgr
                except cv2.error as e:
                    print(f"图像处理失败 ({key}): {e}")
                    formatted_obs["images"][key] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                raise RuntimeError(f"相机 {key} 离线，无法继续采样。")

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_images:
            if self.img_queue.full():
                try:
                    self.img_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.img_queue.put_nowait(display_images)
            except queue.Full:
                pass

        return formatted_obs




##################only dual arm 
# import os
# import time
# import queue
# import threading
# from datetime import datetime

# import cv2
# import gymnasium as gym
# import numpy as np
# import torch
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import JointState

# from serl_robot_infra.Galaxea_env.communication.ros2_bridge import Ros2Bridge
# from serl_robot_infra.Galaxea_env.camera.rs_capture import RSCapture
# from serl_robot_infra.Galaxea_env.camera.video_capture import VideoCapture
# from serl_robot_infra.Galaxea_env.camera.multi_video_capture import MultiVideoCapture
# from serl_robot_infra.Galaxea_env.envs.utils.rotations import (
#     apply_delta_rotation,
#     clip_rotation,
# )


# class GalaxeaImageDisplayer(threading.Thread):
#     """
#     异步图像显示线程，避免阻塞主控制循环。
#     """

#     def __init__(
#         self,
#         queue_obj,
#         image_keys,
#         obs_image_size,
#         window_name,
#         window_size,
#         display_frame_size,
#     ):
#         super().__init__()
#         self.queue = queue_obj
#         self.daemon = True

#         self.image_keys = list(image_keys)
#         self.obs_image_size = tuple(obs_image_size)
#         self.window_name = window_name
#         self.window_size = tuple(window_size)
#         self.display_frame_size = tuple(display_frame_size)

#     def run(self):
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.window_name, *self.window_size)

#         h, w = self.obs_image_size
#         blank = np.zeros((h, w, 3), dtype=np.uint8)

#         while True:
#             img_dict = self.queue.get()
#             if img_dict is None:
#                 break

#             frame = np.concatenate(
#                 [img_dict.get(k, blank) for k in self.image_keys],
#                 axis=1,
#             )

#             display_frame = cv2.resize(
#                 frame,
#                 self.display_frame_size,
#                 interpolation=cv2.INTER_NEAREST,
#             )

#             cv2.imshow(self.window_name, display_frame)
#             cv2.waitKey(1)


# class GalaxeaDualArmEnv(gym.Env):
#     """
#     Galaxea 双臂通用底层环境。

#     设计原则：
#     - 任务专属参数必须从 config 读取
#     - env 只负责通用双臂控制、观测抓取、视频保存、显示与安全裁剪
#     """

#     def __init__(
#         self,
#         config,
#         cfg,
#         save_video=False,
#     ):
#         super().__init__()

#         self.config = config

#         # ==========================================================
#         # 0. 配置检查辅助函数
#         # ==========================================================
#         # 所有任务相关字段都必须显式在 config 中提供，不再保留默认兜底。
#         # 这样 dual_galaxea_env.py 就不会再偷偷带 USB 任务残留。
#         # ==========================================================
#         # 注意：save_video 仍然保留为运行时参数，而不是任务静态配置。
#         # display_images / hz / max_episode_length 这里强制从 config 读取。
#         # ==========================================================

#         # ==========================================================
#         # 1. 通用运行参数（必须从 config 读取）
#         # ==========================================================
#         self.display_images = self._require_config_attr("DISPLAY_IMAGES")
#         self.save_video = save_video
#         self.hz = self._require_config_attr("HZ")
#         self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
#         self.curr_path_length = 0

#         # 图像配置（必须从 config 读取）
#         self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
#         self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
#         self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

#         # 显示窗口配置（必须从 config 读取）
#         self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
#         self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
#         self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

#         # 头部相机 key（必须从 config 读取）
#         self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

#         # ==========================================================
#         # 2. 复位坐标（必须从 config 读取）
#         # ==========================================================
#         self.reset_l = np.array(self._require_config_attr("RESET_L"), dtype=np.float32)
#         self.reset_r = np.array(self._require_config_attr("RESET_R"), dtype=np.float32)

#         # ==========================================================
#         # 3. 动作缩放（必须从 config 读取）
#         # ==========================================================
#         self.pos_scale = float(self._require_config_attr("POS_SCALE"))
#         self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

#         # ==========================================================
#         # 4. 工作空间限位（必须从 config 读取）
#         # ==========================================================
#         xyz_low = np.asarray(self._require_config_attr("XYZ_LIMIT_LOW"), dtype=np.float64)
#         xyz_high = np.asarray(self._require_config_attr("XYZ_LIMIT_HIGH"), dtype=np.float64)
#         rpy_low = np.asarray(self._require_config_attr("RPY_LIMIT_LOW"), dtype=np.float64)
#         rpy_high = np.asarray(self._require_config_attr("RPY_LIMIT_HIGH"), dtype=np.float64)

#         self.xyz_bounding_box = gym.spaces.Box(xyz_low, xyz_high, dtype=np.float64)
#         self.rpy_bounding_box = gym.spaces.Box(rpy_low, rpy_high, dtype=np.float64)

#         # ==========================================================
#         # 5. 录像缓冲区
#         # ==========================================================
#         if self.save_video:
#             print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")
#             self.recording_frames = []

#         # ==========================================================
#         # 6. 全局 ESC 急停
#         # ==========================================================
#         self.terminate = False
#         try:
#             from pynput import keyboard

#             def on_press(key):
#                 if key == keyboard.Key.esc:
#                     print("🛑 检测到 ESC，触发全局紧急终止！")
#                     self.terminate = True

#             self.listener = keyboard.Listener(on_press=on_press)
#             self.listener.start()
#         except ImportError:
#             print("⚠️ 未安装 pynput，ESC 紧急停止功能不可用。(pip install pynput)")

#         # ==========================================================
#         # 7. ROS2 通信
#         # ==========================================================
#         self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

#         # ==========================================================
#         # 8. 相机初始化（必须从 config 读取）
#         # ==========================================================
#         print("正在启动 USB 直连相机阵列...")
#         self.multi_cap = self._build_multi_camera_capture()
#         print("相机阵列启动完毕！")

#         # ==========================================================
#         # 9. 显示线程
#         # ==========================================================
#         if self.display_images:
#             self.img_queue = queue.Queue()
#             self.displayer = GalaxeaImageDisplayer(
#                 self.img_queue,
#                 image_keys=self.display_image_keys,
#                 obs_image_size=self.obs_image_size,
#                 window_name=self.display_window_name,
#                 window_size=self.display_window_size,
#                 display_frame_size=self.display_frame_size,
#             )
#             self.displayer.start()

#         # ==========================================================
#         # 10. Gym 空间定义
#         # ==========================================================
#         h, w = self.obs_image_size
#         image_space = {
#             key: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
#             for key in self.image_keys
#         }

#         self.action_space = gym.spaces.Box(
#             low=-1.0,
#             high=1.0,
#             shape=(14,),
#             dtype=np.float32,
#         )

#         self.observation_space = gym.spaces.Dict(
#             {
#                 "state": gym.spaces.Dict(
#                     {
#                         "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
#                         "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
#                         "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
#                         "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
#                     }
#                 ),
#                 "images": gym.spaces.Dict(image_space),
#             }
#         )

#         self._last_valid_state = None

#     def _require_config_attr(self, name: str):
#         """
#         强制要求 config 提供字段。
#         缺少任何任务相关配置时，直接报错。
#         """
#         if not hasattr(self.config, name):
#             raise AttributeError(f"GalaxeaDualArmEnv 初始化失败：缺少必需配置项 config.{name}")
#         return getattr(self.config, name)

#     # ==========================================================
#     # 相机构建
#     # ==========================================================
#     def _build_multi_camera_capture(self):
#         """
#         从 config 中构建多路相机。
#         不再保留任何 USB 任务默认值。
#         """
#         realsense_cameras = self._require_config_attr("REALSENSE_CAMERAS")
#         head_camera_cfg = self._require_config_attr("HEAD_CAMERA")

#         caps = {}

#         # RealSense cameras
#         for cam_name, cam_cfg in realsense_cameras.items():
#             rs_kwargs = dict(cam_cfg)
#             rs_kwargs.setdefault("name", cam_name)
#             caps[cam_name] = RSCapture(**rs_kwargs)

#         # Head camera / ZED
#         required_head_keys = ["device_index", "api", "name", "split_left_half"]
#         for key in required_head_keys:
#             if key not in head_camera_cfg:
#                 raise AttributeError(f"GalaxeaDualArmEnv 初始化失败：HEAD_CAMERA 缺少必需字段 '{key}'")

#         device_index = head_camera_cfg["device_index"]
#         api = head_camera_cfg["api"]

#         head_cv2 = cv2.VideoCapture(device_index, api)

#         fourcc = head_camera_cfg.get("fourcc")
#         if fourcc:
#             head_cv2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

#         if "frame_width" in head_camera_cfg:
#             head_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, head_camera_cfg["frame_width"])
#         if "frame_height" in head_camera_cfg:
#             head_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, head_camera_cfg["frame_height"])
#         if "fps" in head_camera_cfg:
#             head_cv2.set(cv2.CAP_PROP_FPS, head_camera_cfg["fps"])

#         if not head_cv2.isOpened():
#             raise RuntimeError(f"无法通过 /dev/video{device_index} 打开头部相机")

#         head_camera_name = head_camera_cfg["name"]
#         caps[head_camera_name] = VideoCapture(head_cv2, name=head_camera_name)

#         self.head_camera_cfg = head_camera_cfg
#         return MultiVideoCapture(caps)

#     # ==========================================================
#     # 安全 / 复位
#     # ==========================================================
#     def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
#         """检查并裁剪 7 维位姿，确保其在安全软限位内。"""
#         pose[:3] = np.clip(
#             pose[:3],
#             self.xyz_bounding_box.low,
#             self.xyz_bounding_box.high,
#         )
#         pose[3:] = clip_rotation(
#             pose[3:],
#             self.rpy_bounding_box.low,
#             self.rpy_bounding_box.high,
#         )
#         return pose

#     def interpolate_move(self, goal_l: np.ndarray, goal_r: np.ndarray, timeout: float):
#         """将机械臂从当前位置线性插值平稳移动到目标位置。"""
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()
#         curr_l = self._last_valid_state["left_ee_pose"].copy()
#         curr_r = self._last_valid_state["right_ee_pose"].copy()

#         path_l = np.linspace(curr_l, goal_l, steps)
#         path_r = np.linspace(curr_r, goal_r, steps)

#         print(f"🤖 正在执行平滑复位 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             self._send_ros_poses(path_l[i], path_r[i], g_l=1.0, g_r=1.0)
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def go_to_reset(self):
#         """默认复位流程。任务子类可重写。"""
#         print("开始复位")
#         time.sleep(0.3)
#         self.interpolate_move(self.reset_l, self.reset_r, timeout=1.5)
#         time.sleep(0.5)

#     def reset(self, **kwargs):
#         """Gym reset 接口。"""
#         if self.save_video:
#             self.save_video_recording()

#         self.curr_path_length = 0
#         self.terminate = False

#         self.go_to_reset()
#         self._last_valid_state = None
#         obs = self._get_sync_obs()
#         return obs, {"succeed": False}

#     # ==========================================================
#     # 主 step / close
#     # ==========================================================
#     def step(self, action: np.ndarray) -> tuple:
#         start_time = time.time()
#         self._publish_action(action)

#         self.curr_path_length += 1

#         dt = time.time() - start_time
#         time.sleep(max(0, (1.0 / self.hz) - dt))

#         obs = self._get_sync_obs()
#         reward = 0
#         done = self.curr_path_length >= self.max_episode_length or self.terminate

#         return obs, reward, done, False, {}

#     def close(self):
#         """优雅关闭所有硬件资源。"""
#         if hasattr(self, "listener"):
#             self.listener.stop()

#         self.bridge.destroy()
#         self.multi_cap.close()

#         if self.display_images:
#             self.img_queue.put(None)
#             cv2.destroyAllWindows()
#             self.displayer.join()

#     # ==========================================================
#     # 视频保存
#     # ==========================================================
#     def save_video_recording(self):
#         """将内存中的图像序列编码并写入硬盘文件。"""
#         try:
#             if len(self.recording_frames) > 0:
#                 if not os.path.exists("./videos"):
#                     os.makedirs("./videos")

#                 timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#                 for camera_key in self.recording_frames[0].keys():
#                     video_path = f"./videos/{camera_key}_{timestamp}.mp4"

#                     first_frame = self.recording_frames[0][camera_key]
#                     height, width = first_frame.shape[:2]

#                     video_writer = cv2.VideoWriter(
#                         video_path,
#                         cv2.VideoWriter_fourcc(*"mp4v"),
#                         self.hz,
#                         (width, height),
#                     )

#                     for frame_dict in self.recording_frames:
#                         video_writer.write(frame_dict[camera_key])

#                     video_writer.release()
#                     print(f"🎥 视频已保存: {video_path}")

#             self.recording_frames.clear()
#         except Exception as e:
#             print(f"⚠️ 视频保存失败: {e}")

#     # ==========================================================
#     # 动作发布
#     # ==========================================================
#     def _publish_action(self, action: np.ndarray):
#         """将归一化动作映射为真实物理增量。"""
#         if self._last_valid_state is None:
#             return

#         left_arm_action = action[0:6]
#         left_gripper_action = action[6]
#         right_arm_action = action[7:13]
#         right_gripper_action = action[13]

#         l_pose = self._last_valid_state["left_ee_pose"].copy()
#         next_xyz_l = l_pose[:3] + left_arm_action[0:3] * self.pos_scale
#         next_quat_l = apply_delta_rotation(
#             l_pose[3:7],
#             left_arm_action[3:6] * self.rot_scale,
#         )
#         clipped_pose_l = self.clip_safety_box(np.concatenate([next_xyz_l, next_quat_l]))

#         r_pose = self._last_valid_state["right_ee_pose"].copy()
#         next_xyz_r = r_pose[:3] + right_arm_action[0:3] * self.pos_scale
#         next_quat_r = apply_delta_rotation(
#             r_pose[3:7],
#             right_arm_action[3:6] * self.rot_scale,
#         )
#         clipped_pose_r = self.clip_safety_box(np.concatenate([next_xyz_r, next_quat_r]))

#         self._send_ros_poses(
#             clipped_pose_l,
#             clipped_pose_r,
#             left_gripper_action,
#             right_gripper_action,
#         )

#     def _send_ros_poses(self, p_l, p_r, g_l, g_r):
#         """构建标准 ROS2 PoseStamped / JointState 并发布。"""
#         p_l = np.asarray(p_l, dtype=np.float64).reshape(-1)
#         p_r = np.asarray(p_r, dtype=np.float64).reshape(-1)

#         msg_left = PoseStamped()
#         msg_left.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_left.header.frame_id = "base_link"
#         msg_left.pose.position.x = float(p_l[0])
#         msg_left.pose.position.y = float(p_l[1])
#         msg_left.pose.position.z = float(p_l[2])
#         msg_left.pose.orientation.x = float(p_l[3])
#         msg_left.pose.orientation.y = float(p_l[4])
#         msg_left.pose.orientation.z = float(p_l[5])
#         msg_left.pose.orientation.w = float(p_l[6])

#         msg_right = PoseStamped()
#         msg_right.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_right.header.frame_id = "base_link"
#         msg_right.pose.position.x = float(p_r[0])
#         msg_right.pose.position.y = float(p_r[1])
#         msg_right.pose.position.z = float(p_r[2])
#         msg_right.pose.orientation.x = float(p_r[3])
#         msg_right.pose.orientation.y = float(p_r[4])
#         msg_right.pose.orientation.z = float(p_r[5])
#         msg_right.pose.orientation.w = float(p_r[6])

#         msg_gripper_left = JointState()
#         msg_gripper_left.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_gripper_left.name = ["R1PRO_left_gripper_joint"]
#         msg_gripper_left.position = [float(g_l)]

#         msg_gripper_right = JointState()
#         msg_gripper_right.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_gripper_right.name = ["R1PRO_right_gripper_joint"]
#         msg_gripper_right.position = [float(g_r)]

#         try:
#             self.bridge.publishers[self.bridge.topics_config.action["left_ee_pose"]].publish(msg_left)
#             self.bridge.publishers[self.bridge.topics_config.action["right_ee_pose"]].publish(msg_right)
#             self.bridge.publishers[self.bridge.topics_config.action["left_gripper"]].publish(msg_gripper_left)
#             self.bridge.publishers[self.bridge.topics_config.action["right_gripper"]].publish(msg_gripper_right)
#         except KeyError as e:
#             print(f"⚠️ 发布动作时出现异常: {e}")
#     # ==========================================================
#     # 观测抓取
#     # ==========================================================
#     def _get_sync_obs(self):
#         """强制从硬件抓取最新画面和 ROS 状态，并对齐返回。"""
#         img_dict = self.multi_cap.read()
#         if img_dict is None:
#             img_dict = {}

#         raw_state = None
#         while raw_state is None:
#             raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
#             if raw_state is None:
#                 time.sleep(0.01)

#         formatted_obs = {"state": {}, "images": {}}
#         full_res_images = {}
#         display_images = {}

#         for key in ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]:
#             if raw_state and "state" in raw_state and key in raw_state["state"]:
#                 formatted_obs["state"][key] = raw_state["state"][key].numpy().flatten()
#             else:
#                 dim = 7 if "pose" in key else 1
#                 formatted_obs["state"][key] = np.zeros(dim, dtype=np.float32)

#         self._last_valid_state = formatted_obs["state"]

#         h, w = self.obs_image_size

#         for key in self.image_keys:
#             if key in img_dict:
#                 img_bgr = img_dict[key]

#                 # 头部相机如果是左右拼接图，是否取左半边完全由 config 决定
#                 if key == self.head_camera_key and self.head_camera_cfg["split_left_half"]:
#                     if img_bgr.shape[1] > img_bgr.shape[0] * 2:
#                         width = img_bgr.shape[1]
#                         img_bgr = img_bgr[:, : width // 2, :]

#                 full_res_images[key] = img_bgr.copy()

#                 try:
#                     img_resized_bgr = cv2.resize(img_bgr, (w, h))
#                     img_rgb = img_resized_bgr[..., ::-1]

#                     formatted_obs["images"][key] = img_rgb
#                     display_images[key] = img_resized_bgr
#                 except cv2.error as e:
#                     print(f"图像处理失败 ({key}): {e}")
#                     formatted_obs["images"][key] = np.zeros((h, w, 3), dtype=np.uint8)
#             else:
#                 print(f"警告: 相机 {key} 离线！！！")
#                 input("请检查硬件连接后按【回车键】重试...")
#                 return self._get_sync_obs()

#         if self.save_video:
#             self.recording_frames.append(full_res_images)

#         if self.display_images:
#             self.img_queue.put(display_images)

#         return formatted_obs

