######HIL-SERL 算法大脑只认识 OpenAI Gym 的标准格式。

# HIL-SERL 的视角：它不知道什么是 ROS 2，什么是 DDS，什么是 PoseStamped。
# 它只知道调用 env.step(action)，然后等着拿 (obs, reward, done)。

# 这就是这个文件的使命：它继承了 gym.Env，把 HIL-SERL 输出的 Numpy 数组（你的 14 维动作），打包成了 ROS 2 的消息交给 Bridge 发出去；
# 同时把 Bridge 抓回来的乱七八糟的图像和关节数据，打包成漂亮、规整的 Dict 喂回给算法。

#Franka 的架构：由于 Franka 机械臂是独立的单臂工业臂，作者写了单臂环境（franka_env.py），
# 然后在 dual_franka_env.py 里用多线程把两个单臂环境“强行”拼在了一起。

#Galaxea (星海图) 的架构：你的 R1 PRO 是一台完整的人形机器人。底层的 Ros2Bridge 已经天然地把双臂、腰部、夹爪的数据打包在了一起。
# 所以，我们刚才写好的 dual_galaxea_env.py，其实在地位上等同于 Franka 的 dual_franka_env.py 和 franka_env.py 的究极结合体！



######################################################
#还差官方的：
# （1）边界限制（完成，需要调试）：防止大模型乱发指令导致机械臂撞桌子。由于你的底层可能有自己的安全策略，这一步可以后期加。
# （2）物理复位（不光写函数，还得修改配置文件config.py）：我们目前在 reset() 里只是读取数据，还没写控制机械臂回原点的逻辑。这是采数据前的必经之路。
# （3）复位点 (Reset Pose)	确定两只手在任务开始前的“黄金起点”坐标。	DefaultEnvConfig.RESET_POSE
# （4）目标点 (Target Pose)（不写在这个代码里，写在视觉奖励分类器文件）(如果用目标点，那就是离目标点越近则给奖励，如果用视觉不用目标点，则成功或者失败标签的画面给对应奖励，论文用后者)	任务成功的坐标（虽然有分类器，但用于调试报错）。	DefaultEnvConfig.TARGET_POSE
# （5）激活逻辑 (Intervention)（写在wrappers.py中）	确定你的信号如何触发 intervened = True。	DualSpacemouseIntervention.action
# （6）动作缩放（完成，需要调试） (Scale)	pos_scale（0.01）和 rot_scale（0.05）是否太快或太慢？	FrankaEnv.action_scale
########################################################
##还差：
#（1）画面收不到，可能信号爆了

#暂时忽略官网的两个函数
#(1)compute_reward（基于视觉，所以不需要compute_reward）
#(2)save_video_recording(录制视频1,用于成功数据全程强化学习（过程中人类在环）；2,用于拍摄成功和失败画面训练奖励函数；3,用于回顾强化学习的效果)（借鉴官网的）

##目前dual_galaxea_env.py写死了：
# （1）limit_low = np.array([0.1, -0.5, -0.1, -np.pi, -np.pi, -np.pi])（限位）
# （2）pos_scale, rot_scale = 0.01, 0.05（缩放）
# （3）相机序列号
# 最终可以都写进config.py中

#查询详细相机接口，修改配置的相机接口（realsense相机序列号固定，不会出问题，但是zed相机的dev/video会变）



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


class GalaxeaDualArmEnv(gym.Env):
    """
    Galaxea 双臂通用底层环境。

    设计原则：
    - 任务专属参数必须从 config 读取
    - env 只负责通用双臂控制、观测抓取、视频保存、显示与安全裁剪
    """

    def __init__(
        self,
        config,
        cfg,
        save_video=False,
    ):
        super().__init__()

        self.config = config

        # ==========================================================
        # 0. 配置检查辅助函数
        # ==========================================================
        # 所有任务相关字段都必须显式在 config 中提供，不再保留默认兜底。
        # 这样 dual_galaxea_env.py 就不会再偷偷带 USB 任务残留。
        # ==========================================================
        # 注意：save_video 仍然保留为运行时参数，而不是任务静态配置。
        # display_images / hz / max_episode_length 这里强制从 config 读取。
        # ==========================================================

        # ==========================================================
        # 1. 通用运行参数（必须从 config 读取）
        # ==========================================================
        self.display_images = self._require_config_attr("DISPLAY_IMAGES")
        self.save_video = save_video
        self.hz = self._require_config_attr("HZ")
        self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
        self.curr_path_length = 0

        # 图像配置（必须从 config 读取）
        self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
        self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
        self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

        # 显示窗口配置（必须从 config 读取）
        self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
        self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
        self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

        # 头部相机 key（必须从 config 读取）
        self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

        # ==========================================================
        # 2. 复位坐标（必须从 config 读取）
        # ==========================================================
        self.reset_l = np.array(self._require_config_attr("RESET_L"), dtype=np.float32)
        self.reset_r = np.array(self._require_config_attr("RESET_R"), dtype=np.float32)

        # ==========================================================
        # 3. 动作缩放（必须从 config 读取）
        # ==========================================================
        self.pos_scale = float(self._require_config_attr("POS_SCALE"))
        self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

        # ==========================================================
        # 4. 工作空间限位（必须从 config 读取）
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
        if self.save_video:
            print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")
            self.recording_frames = []

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
        # 8. 相机初始化（必须从 config 读取）
        # ==========================================================
        print("正在启动 USB 直连相机阵列...")
        self.multi_cap = self._build_multi_camera_capture()
        print("相机阵列启动完毕！")

        # ==========================================================
        # 9. 显示线程
        # ==========================================================
        if self.display_images:
            self.img_queue = queue.Queue()
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

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                        "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                    }
                ),
                "images": gym.spaces.Dict(image_space),
            }
        )

        self._last_valid_state = None

    def _require_config_attr(self, name: str):
        """
        强制要求 config 提供字段。
        缺少任何任务相关配置时，直接报错。
        """
        if not hasattr(self.config, name):
            raise AttributeError(f"GalaxeaDualArmEnv 初始化失败：缺少必需配置项 config.{name}")
        return getattr(self.config, name)

    # ==========================================================
    # 相机构建
    # ==========================================================
    def _build_multi_camera_capture(self):
        """
        从 config 中构建多路相机。
        不再保留任何 USB 任务默认值。
        """
        realsense_cameras = self._require_config_attr("REALSENSE_CAMERAS")
        head_camera_cfg = self._require_config_attr("HEAD_CAMERA")

        caps = {}

        # RealSense cameras
        for cam_name, cam_cfg in realsense_cameras.items():
            rs_kwargs = dict(cam_cfg)
            rs_kwargs.setdefault("name", cam_name)
            caps[cam_name] = RSCapture(**rs_kwargs)

        # Head camera / ZED
        required_head_keys = ["device_index", "api", "name", "split_left_half"]
        for key in required_head_keys:
            if key not in head_camera_cfg:
                raise AttributeError(f"GalaxeaDualArmEnv 初始化失败：HEAD_CAMERA 缺少必需字段 '{key}'")

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

    def interpolate_move(self, goal_l: np.ndarray, goal_r: np.ndarray, timeout: float):
        """将机械臂从当前位置线性插值平稳移动到目标位置。"""
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        curr_l = self._last_valid_state["left_ee_pose"].copy()
        curr_r = self._last_valid_state["right_ee_pose"].copy()

        path_l = np.linspace(curr_l, goal_l, steps)
        path_r = np.linspace(curr_r, goal_r, steps)

        print(f"🤖 正在执行平滑复位 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            self._send_ros_poses(path_l[i], path_r[i], g_l=1.0, g_r=1.0)
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def go_to_reset(self):
        """默认复位流程。任务子类可重写。"""
        print("开始复位")
        time.sleep(0.3)
        self.interpolate_move(self.reset_l, self.reset_r, timeout=1.5)
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
    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        self._publish_action(action)

        self.curr_path_length += 1

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        obs = self._get_sync_obs()
        reward = 0
        done = self.curr_path_length >= self.max_episode_length or self.terminate

        return obs, reward, done, False, {}

    def close(self):
        """优雅关闭所有硬件资源。"""
        if hasattr(self, "listener"):
            self.listener.stop()

        self.bridge.destroy()
        self.multi_cap.close()

        if self.display_images:
            self.img_queue.put(None)
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
        """将归一化动作映射为真实物理增量。"""
        if self._last_valid_state is None:
            return

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

    def _send_ros_poses(self, p_l, p_r, g_l, g_r):
        """构建标准 ROS2 PoseStamped / JointState 并发布。"""
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
            print(f"⚠️ 发布动作时出现异常: {e}")
    # ==========================================================
    # 观测抓取
    # ==========================================================
    def _get_sync_obs(self):
        """强制从硬件抓取最新画面和 ROS 状态，并对齐返回。"""
        img_dict = self.multi_cap.read()
        if img_dict is None:
            img_dict = {}

        raw_state = None
        while raw_state is None:
            raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
            if raw_state is None:
                time.sleep(0.01)

        formatted_obs = {"state": {}, "images": {}}
        full_res_images = {}
        display_images = {}

        for key in ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]:
            if raw_state and "state" in raw_state and key in raw_state["state"]:
                formatted_obs["state"][key] = raw_state["state"][key].numpy().flatten()
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
                print(f"警告: 相机 {key} 离线！！！")
                input("请检查硬件连接后按【回车键】重试...")
                return self._get_sync_obs()

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_images:
            self.img_queue.put(display_images)

        return formatted_obs

# """
# Galaxea Dual Arm Environment for HIL-SERL
# Author: Eren
# """

# import os
# import time
# import queue          
# import threading      
# from datetime import datetime
# import numpy as np
# import gymnasium as gym
# import cv2
# import torch
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import JointState

# # 导入星海图通信库（通信位姿）
# # Ros2Bridge: 负责与星海图底层 DDS 通讯，收发位姿数据。
# #from communication.ros2_bridge import Ros2Bridge
# from serl_robot_infra.Galaxea_env.communication.ros2_bridge import Ros2Bridge

# # 导入 HIL 相机通讯库 (USB直连相机)（通信相机）
# # RSCapture/VideoCapture: 封装好的硬件驱动，绕过 ROS 消息大幅降低画面延迟。
# # from camera.rs_capture import RSCapture
# # from camera.video_capture import VideoCapture
# # from camera.multi_video_capture import MultiVideoCapture
# from serl_robot_infra.Galaxea_env.camera.rs_capture import RSCapture
# from serl_robot_infra.Galaxea_env.camera.video_capture import VideoCapture
# from serl_robot_infra.Galaxea_env.camera.multi_video_capture import MultiVideoCapture

# #导入rotations.py，替代from scipy.spatial.transform import Rotation as R
# # apply_delta_rotation: 数学工具，处理四元数的相对旋转计算。
# # from envs.utils.rotations import apply_delta_rotation, clip_rotation
# from serl_robot_infra.Galaxea_env.envs.utils.rotations import apply_delta_rotation, clip_rotation


# # ==========================================
# # 📺 官方同款：异步后台显示线程 (绝不卡顿主循环)
# # ==========================================
# #（测试脚本可知“执行动作时是实时视频，平常是一帧固定的画面”）完全印证了我们之前讲过的多线程架构逻辑。
# #硬件读入：BGR
# #屏幕显示：BGR（无转换，快！）
# #视频录制：BGR（无转换，快！）
# #模型训练/推理：只有在这里，才通过 [..., ::-1] 瞬间切出一份 RGB 给大模型吃。

# #该模块作用（该模块有多个视频相关功能，通过是否TRUE的逻辑还启用其中的功能，不会出现不使用却一直执行功能的情况）：
# #（1）record_demos.py 的main引用该模块，用于录制成功全流程数据用于全程强化学习模板
# #（2）record_success_fail.py 的main引用该模块，用于拍摄成功和失败画面来训练视觉奖励分类器
# #（3）用于监视实时画面
# #（4）用于保存训练过程的mp4视频用于回顾
# class GalaxeaImageDisplayer(threading.Thread):
#     """
#     【架构】: 
#     为了保证预先设置的 15Hz 的控制频率不波动，我们将 OpenCV 的渲染放在独立线程中。
#     主线程只需将图像“丢进”队列，不负责等待渲染完成。
#     """
#     def __init__(self, queue):
#         threading.Thread.__init__(self)
#         self.queue = queue
#         self.daemon = True  # 设置为守护线程，# 守护线程：主程序结束时，显示窗口会自动关闭

#     def run(self):
#         window_name = "Galaxea Vision Monitor"
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(window_name, 1400, 500)

#         while True:
#             # 阻塞式获取队列中的图像字典
#             img_dict = self.queue.get()
#             if img_dict is None:  # 收到 None 信号，说明环境正在 close()，退出循环
#                 break
            
#             # 把左腕、头部、右腕的图横向拼在一起 (注意尺寸对齐)
#             # 【画面拼接】: 将三路 128x128 的图横向拼接，形成 384x128 的监控长条
#             frame = np.concatenate([
#                 img_dict.get("left_wrist_rgb", np.zeros((128,128,3), dtype=np.uint8)),
#                 img_dict.get("head_rgb", np.zeros((128,128,3), dtype=np.uint8)),
#                 img_dict.get("right_wrist_rgb", np.zeros((128,128,3), dtype=np.uint8))
#             ], axis=1)

#             # 把 384x128 放大成更适合人眼观察的尺寸
#             display_frame = cv2.resize(
#                 frame,
#                 (1152, 384),   # 3倍放大
#                 interpolation=cv2.INTER_NEAREST
#             )

#             # OpenCV 显示需要 BGR 格式
#             # OpenCV 在 Linux 环境下默认使用 BGR，而模型和存储使用 RGB，此处需转换
#             #(zhuan rgb de jian kong hua mian) cv2.imshow('Galaxea Dual Arm Vision', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#             cv2.imshow(window_name, display_frame) # 直接写入，因为 full_res_images 存的是 BGR
#             cv2.waitKey(1)  # 维持窗口响应的最小等待时间


# # ==============================================================================
# #  主环境类：GalaxeaDualArmEnv
# # ==============================================================================
# class GalaxeaDualArmEnv(gym.Env):
#     def __init__(self, config, cfg, display_images=False, save_video=False, hz=15, max_episode_length=10000):
#         super().__init__()


#         # --- 1. 核心运行参数 ---
#         self.display_images = display_images   # 是否开启实时监控窗口
#         self.save_video = save_video     # 是否开启高清视频录制 (消耗性能较多)

#         self.hz = hz  #控制频率15hz      # 控制频率，HIL-SERL ，目前推荐 10-15Hz
        
#         ##############
#         #这个env脚本和录制demo脚本谁设置的步长短，谁就决定终止录制的最长步数
#         ##############
#         self.max_episode_length = max_episode_length  # 单局最大步数 (100步大约6.6秒)   # 单局最大步数，超时将强制 Reset
#         self.curr_path_length = 0  # 当前局已经走了多少步   # 当前回合步数计数器
 
#         # --- 2. 复位坐标提取 ---
#         # 坐标格式均为 7 维绝对坐标: [x, y, z, qx, qy, qz, qw]
#         #  提取 Config 中的双臂复位点坐标
#         self.reset_l = np.array(config.RESET_L)
#         self.reset_r = np.array(config.RESET_R)

#         # 录像存储列表
#         # 录像缓冲区：存放高清原图，直到回合结束时写入 MP4
#         if self.save_video:
#             print(" 视频录制模式已激活 (视频将保存在 ./videos 目录)")
#             self.recording_frames = []

#         # --- 3.  安全急停监听 ---
#         #  官方同款：全局紧急停止监听 (ESC 键)
#         # 允许人类在任何时候按下键盘 ESC 键强行中止当前的 AI 运行
#         self.terminate = False
#         try:
#             from pynput import keyboard
#             def on_press(key):
#                 if key == keyboard.Key.esc:
#                     print(" 检测到 ESC，触发全局紧急终止！")
#                     self.terminate = True
#             self.listener = keyboard.Listener(on_press=on_press)
#             self.listener.start()
#         except ImportError:
#             print("⚠️ 未安装 pynput，ESC 紧急停止功能不可用。(pip install pynput)")

#         # --- 4. 硬件初始化 (ROS2通信位姿，夹爪 & usb直连相机python通信) ---
#         # 建立与星海图 R1 PRO 的通讯链路
#         self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

#         # 定义工作空间限位 (Bounding Box)，防止机械臂撞击桌面或自身
#         # 格式: [X, Y, Z, Roll, Pitch, Yaw] (旋转限位用于防止手腕过度拧麻花)
#         limit_low = np.array([0.1, -0.5, -0.1, -np.pi, -np.pi, -np.pi]) 
#         limit_high = np.array([0.7,  0.5,  0.5,  np.pi,  np.pi,  np.pi])
#         self.xyz_bounding_box = gym.spaces.Box(limit_low[:3], limit_high[:3], dtype=np.float64)
#         self.rpy_bounding_box = gym.spaces.Box(limit_low[3:], limit_high[3:], dtype=np.float64)

#         # 启动 USB 直连相机阵列 (绕过 ROS 以追求极致低延迟)
#         print("正在启动 USB 直连相机阵列...")
#         # 手腕相机使用 RealSense 序列号直连
#         left_wrist_cap = RSCapture(name="left_wrist_rgb", serial_number="230322270950", dim=(640, 480), fps=15)
#         right_wrist_cap = RSCapture(name="right_wrist_rgb", serial_number="230322271216", dim=(640, 480), fps=15)
        

#         # 头部 ZED 使用 V4L2 驱动，并强制开启硬件 MJPG 压缩以节省 USB 带宽，防止绿屏
#         ##################
#         #zed2相机接口修改处v4l2-ctl --list-devices
#         ##################
#         zed_cv2 = cv2.VideoCapture(8, cv2.CAP_V4L2)
#         zed_cv2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#         zed_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)  # ZED 双目原始宽度
#         zed_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)  # ZED 原始高度
#         zed_cv2.set(cv2.CAP_PROP_FPS, 15)
        
#         if not zed_cv2.isOpened():
#             print("⚠️ 警告: 无法通过 /dev/video14 打开 ZED 相机！")
            
#         head_cap = VideoCapture(zed_cv2, name="head_rgb")

#         # 使用同步器管理多路相机，确保每一帧拿到的图片在时间上是对齐的
#         self.multi_cap = MultiVideoCapture({
#             "head_rgb": head_cap,
#             "left_wrist_rgb": left_wrist_cap,
#             "right_wrist_rgb": right_wrist_cap
#         })
#         print("相机阵列启动完毕！")

#         # --- 5. 启动显示线程 ---
#         if self.display_images:
#             self.img_queue = queue.Queue()
#             self.displayer = GalaxeaImageDisplayer(self.img_queue)
#             self.displayer.start()

#         # --- 6. 定义动作与观测空间 (Gym 标准) ---
#         # 14 维动作: [L_XYZ, L_RPY, L_Gripper, R_XYZ, R_RPY, R_Gripper] (均为归一化 [-1, 1])
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
#         self.observation_space = gym.spaces.Dict({
#             "state": gym.spaces.Dict({
#                 "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)), 
#                 "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
#                 "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
#                 "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
#             }),
#             "images": gym.spaces.Dict({
#                 "head_rgb": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
#                 "left_wrist_rgb": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
#                 "right_wrist_rgb": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
#             })
#         })

#         self._last_valid_state = None
    
#     def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
#         """【安全核心】: 检查并裁剪 7 维位姿，确保其在安全软限位内"""
#         pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
#         pose[3:] = clip_rotation(pose[3:], self.rpy_bounding_box.low, self.rpy_bounding_box.high)
#         return pose     
    
#     def interpolate_move(self, goal_l: np.ndarray, goal_r: np.ndarray, timeout: float):
#         """【平滑驱动】: 将机械臂从当前位置线性插值平稳移动到目标位置，防止电机过载或抽搐"""
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()   # 刷新当前位置作为起点
#         curr_l = self._last_valid_state["left_ee_pose"].copy()
#         curr_r = self._last_valid_state["right_ee_pose"].copy()

#         # 生成平滑路径点阵列
#         path_l = np.linspace(curr_l, goal_l, steps)
#         path_r = np.linspace(curr_r, goal_r, steps)

#         print(f"🤖 正在执行平滑复位 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             self._send_ros_poses(path_l[i], path_r[i], g_l=1.0, g_r=1.0)
#             # 严格锁频，保证复位速度恒定
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def go_to_reset(self):
#         """执行标准的复位流程"""
#         print("开始复位")
#         time.sleep(0.3)
#         self.interpolate_move(self.reset_l, self.reset_r, timeout=1.5)
#         time.sleep(0.5)

#     def reset(self, **kwargs):
#         # 官方同款：如果开启了录像，在每次回合重置前把上一局的视频存盘
#         """【Gym 重置接口】: 每一局游戏结束时调用"""
#         if self.save_video:
#             self.save_video_recording()  # 如果录像开启，存盘

#         self.curr_path_length = 0  # 步数清零
#         self.terminate = False     # 终止状态清零

#         self.go_to_reset()
#         self._last_valid_state = None
#         obs = self._get_sync_obs()
#         return obs, {"succeed": False}

#     def step(self, action: np.ndarray) -> tuple:
#         """【Gym 核心循环】: 接收动作，执行动作，返回观测值"""
#         start_time = time.time() 
#         self._publish_action(action)  # 下发动作到 ROS2
        
#         self.curr_path_length += 1 # 计数器 +1  # 步数累计
        
#         # ⚠️ 关键：强制锁频到设置的 15Hz。如果计算太快，程序会休眠等待；如果太慢，会跳过休眠。
#         dt = time.time() - start_time
#         time.sleep(max(0, (1.0 / self.hz) - dt))
        
#         obs = self._get_sync_obs()  # 获取执行动作后的最新感官数据
#         reward = 0   # 奖励由外部视觉分类器判定，此处默认为 0
        
#         # 🛡️ 官方同款：到达最大步数，或者你按了 ESC，都会触发 done
#         # 判定当前回合是否结束 (超时或按了 ESC)
#         done = self.curr_path_length >= self.max_episode_length or self.terminate
        
#         return obs, reward, done, False, {}

#     def close(self):
#         """【销毁接口】: 优雅关闭所有硬件资源"""
#         if hasattr(self, 'listener'):
#             self.listener.stop()
#         self.bridge.destroy()
#         self.multi_cap.close()
#         # 优雅关闭显示线程
#         if self.display_images:
#             self.img_queue.put(None)  # 发送结束信号给后台线程
#             cv2.destroyAllWindows()
#             self.displayer.join()

#     #  官方同款：存盘逻辑 (将高分辨率图写入 MP4)
#     def save_video_recording(self):
#         """【存盘逻辑】: 将内存中的图像序列编码并写入硬盘文件"""
#         try:
#             if len(self.recording_frames) > 0:
#                 if not os.path.exists('./videos'):
#                     os.makedirs('./videos')
                
#                 timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
#                 for camera_key in self.recording_frames[0].keys():
#                     video_path = f'./videos/{camera_key}_{timestamp}.mp4'
                    
#                     first_frame = self.recording_frames[0][camera_key]
#                     height, width = first_frame.shape[:2]
                    
#                     # 使用 mp4v 编码器，帧率与环境控制频率 HZ 保持一致
#                     video_writer = cv2.VideoWriter(
#                         video_path,
#                         cv2.VideoWriter_fourcc(*"mp4v"),
#                         self.hz, # 用环境的真实控制频率作为视频帧率
#                         (width, height),
#                     )
                    
#                     for frame_dict in self.recording_frames:
#                         # # OpenCV 写入视频需要 BGR 格式
#                         # bgr_frame = cv2.cvtColor(frame_dict[camera_key], cv2.COLOR_RGB2BGR)
#                         # video_writer.write(bgr_frame)

#                         # 不需要再进行 cvtColor 转换
#                         video_writer.write(frame_dict[camera_key])
                    
#                     video_writer.release()
#                     print(f" 视频已保存: {video_path}")
                
#             self.recording_frames.clear() # 清空内存，准备录制下一局
#         except Exception as e:
#             print(f"⚠️ 视频保存失败: {e}")

#     def _publish_action(self, action: np.ndarray):
#         """【动作解析】: 将归一化的 [-1, 1] 动作映射为真实的物理增量坐标"""
#         if self._last_valid_state is None: return

#         # 1. 拆分 14 维动作向量
#         left_arm_action = action[0:6]
#         left_gripper_action = action[6]
#         right_arm_action = action[7:13]
#         right_gripper_action = action[13]
        
#         # 动作缩放：将 [-1, 1] 的网络输出映射为 [1cm 移动, 0.05rad 旋转]
#         pos_scale = 0.01
#         rot_scale = 0.05
        
#         # --- 左臂增量计算 ---
#         l_pose = self._last_valid_state["left_ee_pose"].copy()
#         next_xyz_l = l_pose[:3] + left_arm_action[0:3] * pos_scale 
#         next_quat_l = apply_delta_rotation(l_pose[3:7], left_arm_action[3:6] * rot_scale)
#         clipped_pose_l = self.clip_safety_box(np.concatenate([next_xyz_l, next_quat_l]))
        
#         # --- 右臂增量计算 ---
#         r_pose = self._last_valid_state["right_ee_pose"].copy()
#         next_xyz_r = r_pose[:3] + right_arm_action[0:3] * pos_scale
#         next_quat_r = apply_delta_rotation(r_pose[3:7], right_arm_action[3:6] * rot_scale)
#         clipped_pose_r = self.clip_safety_box(np.concatenate([next_xyz_r, next_quat_r]))
        
#         # 调用 ROS 发布函数
#         self._send_ros_poses(clipped_pose_l, clipped_pose_r, left_gripper_action, right_gripper_action)

#     def _send_ros_poses(self, p_l, p_r, g_l, g_r):
#         """【协议封装】: 构建标准的 ROS2 PoseStamped 消息并 Publish"""
#         msg_left = PoseStamped()
#         msg_left.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_left.header.frame_id = 'base_link'
#         msg_left.pose.position.x, msg_left.pose.position.y, msg_left.pose.position.z = p_l[0], p_l[1], p_l[2]
#         msg_left.pose.orientation.x, msg_left.pose.orientation.y, msg_left.pose.orientation.z, msg_left.pose.orientation.w = p_l[3], p_l[4], p_l[5], p_l[6]

#         msg_right = PoseStamped()
#         msg_right.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_right.header.frame_id = 'base_link'
#         msg_right.pose.position.x, msg_right.pose.position.y, msg_right.pose.position.z = p_r[0], p_r[1], p_r[2]
#         msg_right.pose.orientation.x, msg_right.pose.orientation.y, msg_right.pose.orientation.z, msg_right.pose.orientation.w = p_r[3], p_r[4], p_r[5], p_r[6]

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

#     def _get_sync_obs(self):
#         """【感官中枢】: 强制从硬件抓取最新画面和 ROS 状态，并对齐返回"""
#         img_dict = self.multi_cap.read()  # 硬件级图像抓取
#         if img_dict is None: img_dict = {}
        
#         # 强制轮询直到拿到最新的本体位姿
#         raw_state = None
#         while raw_state is None:
#             raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
#             if raw_state is None: time.sleep(0.01)

#         formatted_obs = {"state": {}, "images": {}}
        
#         # 官方同款：准备高清原图和显示图容器
#         full_res_images = {}
#         display_images = {}
        
#         # --- 处理本体状态 ---
#         for key in ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]:
#             if raw_state and "state" in raw_state and key in raw_state["state"]:
#                 formatted_obs["state"][key] = raw_state["state"][key].numpy().flatten()
#             else:
#                 dim = 7 if "pose" in key else 1
#                 formatted_obs["state"][key] = np.zeros(dim, dtype=np.float32)

#         self._last_valid_state = formatted_obs["state"]
        
#         # --- 处理多路视觉图像 ---
#         for key in ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]:
#             if key in img_dict:
#                 img_bgr = img_dict[key] # 保持原生 BGR 命名，提醒自己还没转色

#                 # ZED 裁剪逻辑: ZED 输出的是超宽双目拼接图，我们只需左半边
#                 if key == "head_rgb" and img_bgr.shape[1] > img_bgr.shape[0] * 2:
#                     w = img_bgr.shape[1]
#                     img_bgr = img_bgr[:, :w//2, :]

#                 # 1. 存高清原图用于录像 (官方通常存 BGR，我们这里为了录像对齐也用 BGR)
#                 # 录像函数 save_video_recording 里就不用再 cvtColor 了
#                 full_res_images[key] = img_bgr.copy()
                
#                 try:
#                     # 2. 缩放画面 (此时依然是 BGR)
#                     # 缩放为模型标准的 128x128
#                     img_resized_bgr = cv2.resize(img_bgr, (128, 128))
                    
#                     # 🌟 官方精髓：就在这一步进行切片转换 BGR -> RGB
#                     img_rgb = img_resized_bgr[..., ::-1] 

#                     formatted_obs["images"][key] = img_rgb # 给模型吃 RGB
#                     display_images[key] = img_resized_bgr  # 给显示器吃 BGR
#                 except cv2.error as e:
#                     print(f"图像处理失败 ({key}): {e}")
#                     formatted_obs["images"][key] = np.zeros((128, 128, 3), dtype=np.uint8)

#             else:
#                 # 参考官方做法：不要返回黑图等来强行让进程进行下去，直接报警并等待人工介入
#                 print(f"警告: 相机 {key} 离线！！！!")
#                 input("请检查硬件连接后按【回车键】重试...")
#                 # 这里可以尝试调用你环境里的相机重启逻辑
#                 return self._get_sync_obs()
        
#         # 数据分发
#         #  如果开启了录像，把这一帧高清图加入列表
#         if self.save_video:
#             self.recording_frames.append(full_res_images)
            
#         #  如果开启了显示，把这一帧的小图塞进异步队列
#         if self.display_images:
#             self.img_queue.put(display_images)

#         return formatted_obs



