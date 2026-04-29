#少一个位姿是否到达来判断奖励（星海图难以复现，因为vr权限最大，难以保持终点一直，旋转一致），目前只有视觉奖励分类器判断成功
#
#





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

        self.arm_side = str(getattr(self.config, "ARM_SIDE", "right")).lower()
        if self.arm_mode == "single" and self.arm_side not in ("left", "right"):
            raise ValueError(f"单臂模式下 config.ARM_SIDE 必须是 'left' 或 'right'，当前为 {self.arm_side!r}")

        # ==========================================================
        # 1. 通用运行参数
        # ==========================================================
        self.display_images = self._require_config_attr("DISPLAY_IMAGES")
        self.save_video = save_video
        self.hz = self._require_config_attr("HZ")
        self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
        self.curr_path_length = 0

        self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
        self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
        self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

        self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
        self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
        self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

        self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

        # ==========================================================
        # 2. 动作缩放
        # ==========================================================
        self.pos_scale = float(self._require_config_attr("POS_SCALE"))
        self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

        # ==========================================================
        # 2.1 step 等待 / 状态同步配置
        # ----------------------------------------------------------
        # 借鉴官方 FrankaEnv 的思想：
        # - actor / replay 外层只调用 env.step(action)
        # - env.step 内部负责发布动作、按控制频率等待、再读取真实状态
        #
        # Galaxea 的 ROS2 bridge / 底层跟随可能比 1/HZ 更慢，
        # 因此允许额外 ACTION_SETTLE_SEC，让返回的 next_obs 更接近动作真正执行后的状态。
        #
        # 推荐：
        #   ACTION_SETTLE_SEC = 0.00  # 最接近原始 15Hz，可能仍有抖动
        #   ACTION_SETTLE_SEC = 0.05  # 轻量稳定
        #   ACTION_SETTLE_SEC = 0.08~0.10  # 更稳，采样更慢
        # ==========================================================
        self.action_settle_sec = float(getattr(self.config, "ACTION_SETTLE_SEC", 0.0))
        if self.action_settle_sec < 0:
            raise ValueError("ACTION_SETTLE_SEC 必须 >= 0")

        self.debug_step_timing = bool(getattr(self.config, "DEBUG_STEP_TIMING", False))

        # ==========================================================
        # 3. 工作空间限位
        # ==========================================================
        xyz_low = np.asarray(self._require_config_attr("XYZ_LIMIT_LOW"), dtype=np.float64)
        xyz_high = np.asarray(self._require_config_attr("XYZ_LIMIT_HIGH"), dtype=np.float64)
        rpy_low = np.asarray(self._require_config_attr("RPY_LIMIT_LOW"), dtype=np.float64)
        rpy_high = np.asarray(self._require_config_attr("RPY_LIMIT_HIGH"), dtype=np.float64)

        self.xyz_bounding_box = gym.spaces.Box(xyz_low, xyz_high, dtype=np.float64)
        self.rpy_bounding_box = gym.spaces.Box(rpy_low, rpy_high, dtype=np.float64)

        # ==========================================================
        # 4. 录像缓冲区
        # ==========================================================
        self.recording_frames = []
        if self.save_video:
            print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")

        # ==========================================================
        # 5. 全局 ESC 急停
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
        # 6. ROS2 通信
        # ==========================================================
        self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

        # ==========================================================
        # 7. 相机初始化
        # ==========================================================
        print("正在启动 USB 直连相机阵列...")
        self.multi_cap = self._build_multi_camera_capture()
        print("相机阵列启动完毕！")

        # ==========================================================
        # 8. 显示线程
        # ==========================================================
        if self.display_images:
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
        # 9. Gym 空间定义
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

        # ==========================================================
        # 10. 夹爪命令/反馈配置、hold 记忆与事件 latch
        # ----------------------------------------------------------
        # 关键设计：
        # - 训练 / demo / actor 存储层永远使用三值标签：
        #     -1 = close event, 0 = hold, +1 = open event
        # - 硬件执行层才把三值标签映射到 R1 PRO 夹爪量程：
        #     close_cmd = 10, open_cmd = 80
        # - 一帧 close/open 标签对真实夹爪可能太短，因此增加 latch：
        #     收到 close(-1) 后，后续若干帧 hold(0) 仍继续发送 close_cmd
        #     收到 open(+1)  后，后续若干帧 hold(0) 仍继续发送 open_cmd
        # - feedback 只用于诊断/确认，不再覆盖已经明确下发的目标命令。
        #
        # 推荐在任务 config 中显式加入：
        #   GRIPPER_CLOSE_CMD = 10.0
        #   GRIPPER_OPEN_CMD = 80.0
        #   GRIPPER_FEEDBACK_CLOSE_MAX = 30.0
        #   GRIPPER_FEEDBACK_OPEN_MIN = 70.0
        #   GRIPPER_CLOSE_LATCH_STEPS = 5
        #   GRIPPER_OPEN_LATCH_STEPS = 5
        #   GRIPPER_FEEDBACK_SYNC_MEMORY = False
        # ==========================================================
        self.gripper_close_cmd = float(getattr(self.config, "GRIPPER_CLOSE_CMD", 10.0))
        self.gripper_open_cmd = float(getattr(self.config, "GRIPPER_OPEN_CMD", 80.0))

        self.gripper_feedback_close_max = float(
            getattr(self.config, "GRIPPER_FEEDBACK_CLOSE_MAX", 30.0)
        )
        self.gripper_feedback_open_min = float(
            getattr(self.config, "GRIPPER_FEEDBACK_OPEN_MIN", 70.0)
        )

        self.gripper_close_latch_steps = int(
            getattr(self.config, "GRIPPER_CLOSE_LATCH_STEPS", 0)
        )
        self.gripper_open_latch_steps = int(
            getattr(self.config, "GRIPPER_OPEN_LATCH_STEPS", 0)
        )
        if self.gripper_close_latch_steps < 0:
            raise ValueError("GRIPPER_CLOSE_LATCH_STEPS 必须 >= 0")
        if self.gripper_open_latch_steps < 0:
            raise ValueError("GRIPPER_OPEN_LATCH_STEPS 必须 >= 0")

        # True 时，feedback 允许在每次 _get_sync_obs 后同步 memory。
        # 默认 False，因为 close 刚触发时 feedback 仍可能处于 open 区间，
        # 如果允许同步，会把 close memory 立刻覆盖回 open，导致一帧 close 失效。
        self.gripper_feedback_sync_memory = bool(
            getattr(self.config, "GRIPPER_FEEDBACK_SYNC_MEMORY", False)
        )

        # True 时，如果 latch 期间 feedback 已经确认到达目标状态，则 latch 提前结束。
        # 注意：提前结束只清空 latch counter，不改变 desired command；
        # close 后 desired 仍保持 close_cmd，open 后 desired 仍保持 open_cmd。
        self.gripper_latch_release_on_feedback = bool(
            getattr(self.config, "GRIPPER_LATCH_RELEASE_ON_FEEDBACK", True)
        )

        # 当历史 memory 被污染成 middle 值时，hold 如何吸附。
        # USB 插入任务里，闭合后保持更重要，所以默认 middle -> close。
        self.gripper_hold_middle_default = str(
            getattr(self.config, "GRIPPER_HOLD_MIDDLE_DEFAULT", "close")
        ).lower()
        if self.gripper_hold_middle_default not in ("close", "open"):
            raise ValueError(
                "GRIPPER_HOLD_MIDDLE_DEFAULT 必须是 'close' 或 'open'，"
                f"当前为 {self.gripper_hold_middle_default!r}"
            )

        # 上一次明确希望发给硬件的目标命令。只应为 close_cmd / open_cmd，
        # 不应被 17/22/32/60 这类连续 feedback 污染。
        self._last_hw_gripper_cmd = {
            "left": self.gripper_open_cmd,
            "right": self.gripper_open_cmd,
        }

        # 是否已经通过标签/硬件命令显式设置过 desired command。
        # reset 后先设为 False，允许用真实 feedback 初始化一次；
        # 一旦收到 close/open 事件，就设为 True，之后 feedback 默认不再覆盖 memory。
        self._gripper_has_explicit_command = {
            "left": False,
            "right": False,
        }

        # latch 状态：用于把一帧 close/open event 扩展成连续硬件命令。
        self._gripper_latch_kind = {
            "left": None,
            "right": None,
        }
        self._gripper_latch_remaining = {
            "left": 0,
            "right": 0,
        }
        self._gripper_latch_cmd = {
            "left": self.gripper_open_cmd,
            "right": self.gripper_open_cmd,
        }

    def _require_config_attr(self, name: str):
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
    # gripper 反馈 / 保持状态辅助
    # ==========================================================
    def _extract_gripper_feedback_from_state(self, state_dict, arm_side: str):
        key = f"{arm_side}_gripper"
        if state_dict is None or key not in state_dict:
            return None
        arr = np.asarray(state_dict[key]).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[-1])

    def _sync_last_hw_gripper_cmd_from_state(self, state_dict):
        """
        用真实 gripper feedback 做状态确认，但默认不覆盖显式命令记忆。

        这一步的核心是解决“一帧 close 太短”的问题：
        - 收到 close(-1) 后，_last_hw_gripper_cmd 会被设为 close_cmd=10；
        - 随后的 _get_sync_obs 可能还看到 feedback=72/open；
        - 如果此时把 memory 覆盖回 open_cmd=80，后续 hold(0) 就会重新张开；
        - 因此，默认只有在尚未收到过显式命令时，才用 feedback 初始化 memory。

        如果 config.GRIPPER_FEEDBACK_SYNC_MEMORY=True，则恢复为允许 feedback 同步 memory，
        但不建议在当前 USB 插入任务中打开。
        """
        for arm_side in ("left", "right"):
            val = self._extract_gripper_feedback_from_state(state_dict, arm_side)
            if val is None or not np.isfinite(val):
                continue

            val = float(val)

            # latch 期间如果 feedback 已经到达目标区间，可以提前结束 latch。
            # 但不会改变 _last_hw_gripper_cmd；close 仍保持 10，open 仍保持 80。
            if self.gripper_latch_release_on_feedback:
                kind = self._gripper_latch_kind.get(arm_side, None)
                if kind == "close" and val <= self.gripper_feedback_close_max:
                    self._gripper_latch_remaining[arm_side] = 0
                elif kind == "open" and val >= self.gripper_feedback_open_min:
                    self._gripper_latch_remaining[arm_side] = 0

            # 默认：只在没有显式命令时，使用 feedback 初始化 memory。
            # 这样 reset 后可以根据真实状态初始化 open/close；
            # 但 close/open event 之后，feedback 不会把 desired command 覆盖回相反方向。
            allow_feedback_to_set_memory = (
                self.gripper_feedback_sync_memory
                or not self._gripper_has_explicit_command.get(arm_side, False)
            )

            if not allow_feedback_to_set_memory:
                continue

            if val <= self.gripper_feedback_close_max:
                self._last_hw_gripper_cmd[arm_side] = self.gripper_close_cmd
            elif val >= self.gripper_feedback_open_min:
                self._last_hw_gripper_cmd[arm_side] = self.gripper_open_cmd
            else:
                # middle/unclear: 不更新，避免 30~70 中间反馈污染 hold memory。
                pass

    def _resolve_hold_gripper_cmd(self, arm_side: str) -> float:
        """
        hold(0) 时返回应该继续发送的硬件目标命令。

        优先级：
        1. 如果 latch 仍在生效，继续发送 latch 对应命令，并消耗 1 个 latch step；
        2. 否则发送上一条明确目标命令；
        3. 如果 memory 由于历史代码被污染成中间值，则按配置吸附到 close/open。
        """
        remaining = int(self._gripper_latch_remaining.get(arm_side, 0))
        if remaining > 0:
            hw = float(self._gripper_latch_cmd.get(arm_side, self.gripper_open_cmd))
            self._gripper_latch_remaining[arm_side] = max(0, remaining - 1)
            self._last_hw_gripper_cmd[arm_side] = hw
            self._gripper_has_explicit_command[arm_side] = True
            return hw

        cmd = float(self._last_hw_gripper_cmd.get(arm_side, self.gripper_open_cmd))

        if cmd <= self.gripper_feedback_close_max:
            return self.gripper_close_cmd
        if cmd >= self.gripper_feedback_open_min:
            return self.gripper_open_cmd

        # 历史污染防御：不允许 middle 值成为 hold 目标。
        if self.gripper_hold_middle_default == "open":
            return self.gripper_open_cmd
        return self.gripper_close_cmd

    def _set_gripper_desired_cmd(
        self,
        arm_side: str,
        hw: float,
        *,
        kind: str = None,
        latch_steps: int = 0,
    ):
        """
        设置上一条明确硬件目标命令，并可启动 latch。

        kind:
          "close" -> close latch
          "open"  -> open latch
          None    -> 只设置 memory，不启动 latch
        """
        hw = float(hw)
        self._last_hw_gripper_cmd[arm_side] = hw
        self._gripper_has_explicit_command[arm_side] = True

        if kind in ("close", "open") and latch_steps > 0:
            self._gripper_latch_kind[arm_side] = kind
            self._gripper_latch_remaining[arm_side] = int(latch_steps)
            self._gripper_latch_cmd[arm_side] = hw
        else:
            self._gripper_latch_kind[arm_side] = None
            self._gripper_latch_remaining[arm_side] = 0
            self._gripper_latch_cmd[arm_side] = hw

    def _remember_explicit_gripper_cmd(self, arm_side: str, hw: float):
        """
        记录明确 close/open 的硬件目标。
        reset / interpolate 可能直接给 10 或 80；这类命令应该更新 memory。
        30~70 中间硬件值可以发送，但不允许污染 hold memory。
        """
        hw = float(hw)
        if hw <= self.gripper_feedback_close_max:
            self._set_gripper_desired_cmd(
                arm_side,
                self.gripper_close_cmd,
                kind=None,
                latch_steps=0,
            )
        elif hw >= self.gripper_feedback_open_min:
            self._set_gripper_desired_cmd(
                arm_side,
                self.gripper_open_cmd,
                kind=None,
                latch_steps=0,
            )

    def _get_gripper_debug_info(self):
        """
        返回当前夹爪执行层状态，方便 replay / actor 调试。
        """
        return {
            "last_hw_gripper_cmd": dict(self._last_hw_gripper_cmd),
            "has_explicit_command": dict(self._gripper_has_explicit_command),
            "latch_kind": dict(self._gripper_latch_kind),
            "latch_remaining": dict(self._gripper_latch_remaining),
            "latch_cmd": dict(self._gripper_latch_cmd),
            "close_cmd": self.gripper_close_cmd,
            "open_cmd": self.gripper_open_cmd,
            "feedback_close_max": self.gripper_feedback_close_max,
            "feedback_open_min": self.gripper_feedback_open_min,
            "feedback_sync_memory": self.gripper_feedback_sync_memory,
        }

    # ==========================================================
    # gripper 语义 -> 硬件量程 映射
    # ==========================================================
    def _map_gripper_cmd_to_hardware(self, cmd: float, arm_side: str) -> float:
        """
        将上层 gripper 三值标签 / 硬件直传命令映射为真实硬件量程。

        标签空间：
          cmd <= -0.5 -> close event
          cmd >=  0.5 -> open event
          otherwise   -> hold

        执行层 latch：
          收到 close event 后，设置 desired=close_cmd，并让后续若干帧 hold 继续发 close_cmd；
          收到 open event 后，设置 desired=open_cmd，并让后续若干帧 hold 继续发 open_cmd；
          收到 hold 时，不改变标签语义，只继续发送当前 desired/latch 命令。

        这样 demo / actor buffer 中仍然是一帧 -1/+1 事件标签，
        但真实硬件能得到足够长的 close/open 指令。
        """
        cmd = float(cmd)

        # 硬件量程直传：reset / interpolate_move_single 等路径可能使用。
        # step() 的归一化动作会先被 clip 到 [-1,1]，正常不会误入这里。
        if 0.0 <= cmd <= 100.0 and abs(cmd) > 5.0:
            hw = float(cmd)
            self._remember_explicit_gripper_cmd(arm_side, hw)
            return hw

        # open event
        if cmd >= 0.5:
            hw = self.gripper_open_cmd
            self._set_gripper_desired_cmd(
                arm_side,
                hw,
                kind="open",
                latch_steps=self.gripper_open_latch_steps,
            )
            return hw

        # close event
        if cmd <= -0.5:
            hw = self.gripper_close_cmd
            self._set_gripper_desired_cmd(
                arm_side,
                hw,
                kind="close",
                latch_steps=self.gripper_close_latch_steps,
            )
            return hw

        # hold：继续发送 latch / 上一条明确目标命令。
        hw = self._resolve_hold_gripper_cmd(arm_side)
        self._last_hw_gripper_cmd[arm_side] = hw
        return float(hw)

    # ==========================================================
    # 安全 / 复位基础工具
    # ==========================================================
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
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

    def interpolate_move_dual(
        self,
        goal_l: np.ndarray,
        goal_r: np.ndarray,
        timeout: float,
        g_l: float = None,
        g_r: float = None,
    ):
        """
        双臂平滑移动基础工具。
        如果 g_l / g_r 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
        """
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        curr_l = self._last_valid_state["left_ee_pose"].copy()
        curr_r = self._last_valid_state["right_ee_pose"].copy()

        if g_l is None:
            g_l = self._resolve_hold_gripper_cmd("left")
        if g_r is None:
            g_r = self._resolve_hold_gripper_cmd("right")

        path_l = np.linspace(curr_l, goal_l, steps)
        path_r = np.linspace(curr_r, goal_r, steps)

        print(f"🤖 正在执行双臂平滑复位/移动 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            self._send_ros_poses(path_l[i], path_r[i], g_l=float(g_l), g_r=float(g_r))
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def interpolate_move_single(
        self,
        goal_pose: np.ndarray,
        timeout: float,
        gripper: float = None,
    ):
        """
        单臂平滑移动基础工具。
        如果 gripper 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
        """
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        ee_key = f"{self.arm_side}_ee_pose"
        curr_pose = self._last_valid_state[ee_key].copy()

        if gripper is None:
            gripper = self._resolve_hold_gripper_cmd(self.arm_side)

        path = np.linspace(curr_pose, goal_pose, steps)

        print(f"🤖 正在执行单臂平滑复位/移动 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            # 这里传的是硬件量程；若 gripper=None，上面已解析为“保持当前夹爪状态”
            self._send_ros_pose(self.arm_side, path[i], gripper)
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def go_to_reset(self):
        """
        通用 env 不再持有任务级 reset 逻辑。
        具体任务请在对应 wrapper / task env 中覆写这个方法，
        并在那里决定：
        - reset 目标位姿
        - 是否随机扰动
        - 是否改变夹爪
        - 用什么 gripper 硬件值
        """
        raise NotImplementedError(
            "GalaxeaArmEnv.go_to_reset() 已改为任务级接口。"
            "请在具体任务 wrapper / env 中实现 reset 逻辑。"
        )

    def reset(self, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.curr_path_length = 0
        self.terminate = False

        # reset 后允许用真实 feedback 重新初始化一次 hold memory。
        # 注意：这不会改变标签空间，只是为下一轮 episode 的 hold 初始化硬件目标。
        self._gripper_has_explicit_command = {
            "left": False,
            "right": False,
        }
        self._gripper_latch_kind = {
            "left": None,
            "right": None,
        }
        self._gripper_latch_remaining = {
            "left": 0,
            "right": 0,
        }
        self._gripper_latch_cmd = {
            "left": self._last_hw_gripper_cmd.get("left", self.gripper_open_cmd),
            "right": self._last_hw_gripper_cmd.get("right", self.gripper_open_cmd),
        }

        # 不再在通用 env 里强制把夹爪 reset 成张开
        self.go_to_reset()

        self._last_valid_state = None
        obs = self._get_sync_obs()

        # 用真实反馈同步“上一夹爪状态”，让后续 hold 更可信
        self._sync_last_hw_gripper_cmd_from_state(obs["state"])

        return obs, {"succeed": False}
    

    def observe_only_step(self):
        """
        只读取当前 observation，不发布任何 arm / gripper 命令。

        用途：
          - VR 接管 Mode0 期间记录真实机器人运动
          - mode 切换等待期间刷新 obs
          - 绝对禁止 _publish_action()
          - 避免 env.step(zero/hold) 造成末端慢慢移动或夹爪轻微张开

        注意：
          这个函数会保持和 step() 类似的时间节奏：
            1) 按 1/HZ 等待
            2) 可选 ACTION_SETTLE_SEC
            3) 读取 _get_sync_obs()
          但不会调用 _publish_action()
        """
        step_start = time.time()

        # 关键：observe_only 绝对不 publish。
        publish_dt = 0.0

        self.curr_path_length += 1

        # 保持和 step() 类似的时间节奏，避免 observe loop 过快。
        target_period = 1.0 / float(self.hz)
        elapsed_after_publish = time.time() - step_start
        hz_sleep = max(0.0, target_period - elapsed_after_publish)
        if hz_sleep > 0:
            time.sleep(hz_sleep)

        settle_sleep = float(self.action_settle_sec)
        if settle_sleep > 0:
            time.sleep(settle_sleep)

        obs_start = time.time()
        obs = self._get_sync_obs()
        obs_dt = time.time() - obs_start

        reward = 0
        terminated = self.terminate
        truncated = self.curr_path_length >= self.max_episode_length

        total_dt = time.time() - step_start
        info = {
            "observe_only": True,
            "gripper_debug": self._get_gripper_debug_info(),
            "step_timing": {
                "hz": float(self.hz),
                "target_period_sec": float(target_period),
                "publish_dt_sec": float(publish_dt),
                "hz_sleep_sec": float(hz_sleep),
                "action_settle_sec": float(settle_sleep),
                "obs_dt_sec": float(obs_dt),
                "total_step_dt_sec": float(total_dt),
                "effective_hz": float(1.0 / total_dt) if total_dt > 1e-9 else 0.0,
            },
        }

        if self.debug_step_timing:
            print(
                "[GalaxeaArmEnv.observe_only_step timing] "
                f"publish={publish_dt:.4f}s, "
                f"hz_sleep={hz_sleep:.4f}s, "
                f"settle={settle_sleep:.4f}s, "
                f"obs={obs_dt:.4f}s, "
                f"total={total_dt:.4f}s, "
                f"effective_hz={info['step_timing']['effective_hz']:.2f}"
            )

        return obs, reward, terminated, truncated, info      

    # ==========================================================
    # 主 step / close
    # ==========================================================
    def step(self, action: np.ndarray):
        """
        标准 Gym step。

        关键设计与官方 FrankaEnv 对齐：
        - 外层 actor / replay 不应该各自实现不同的 sleep/refresh；
        - env.step 内部统一负责：
            1) 接收 normalized action；
            2) clip 到 action_space；
            3) 发布机器人目标；
            4) 按 1/HZ 等待一个控制周期；
            5) 可选额外等待 ACTION_SETTLE_SEC，让 Galaxea 底层更充分跟随；
            6) 读取最新真实 obs，并更新 _last_valid_state；
            7) 返回 next_obs。

        这样 replay 和 actor 都只需要调用 env.step(action)，行为一致。
        """
        step_start = time.time()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = 14 if self.arm_mode == "dual" else 7

        if action.shape[0] != expected_dim:
            raise ValueError(
                f"动作维度错误：当前 arm_mode={self.arm_mode}, "
                f"期望 {expected_dim} 维，实际收到 {action.shape}"
            )

        # 官方风格：step 接收归一化动作，统一 clip 到 [-1, 1]。
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 发布动作。_publish_action 内部会用当前 _last_valid_state 计算增量目标。
        publish_start = time.time()
        self._publish_action(action)
        publish_dt = time.time() - publish_start

        self.curr_path_length += 1

        # 先按控制频率等待一个周期。
        elapsed_after_publish = time.time() - step_start
        hz_sleep = max(0.0, (1.0 / float(self.hz)) - elapsed_after_publish)
        if hz_sleep > 0:
            time.sleep(hz_sleep)

        # Galaxea 额外 settle：
        # 官方 FrankaEnv 在 1/HZ 后读取状态；你的硬件链路实测需要更久时，
        # 在 env 内加这段，确保 actor/replay/eval 统一。
        settle_sleep = float(self.action_settle_sec)
        if settle_sleep > 0:
            time.sleep(settle_sleep)

        obs_start = time.time()
        obs = self._get_sync_obs()
        obs_dt = time.time() - obs_start

        reward = 0
        terminated = self.terminate
        truncated = self.curr_path_length >= self.max_episode_length

        total_dt = time.time() - step_start
        info = {
            "gripper_debug": self._get_gripper_debug_info(),
            "step_timing": {
                "hz": float(self.hz),
                "target_period_sec": 1.0 / float(self.hz),
                "publish_dt_sec": float(publish_dt),
                "hz_sleep_sec": float(hz_sleep),
                "action_settle_sec": float(settle_sleep),
                "obs_dt_sec": float(obs_dt),
                "total_step_dt_sec": float(total_dt),
                "effective_hz": float(1.0 / total_dt) if total_dt > 1e-9 else 0.0,
            },
        }

        if self.debug_step_timing:
            print(
                "[GalaxeaArmEnv.step timing] "
                f"publish={publish_dt:.4f}s, "
                f"hz_sleep={hz_sleep:.4f}s, "
                f"settle={settle_sleep:.4f}s, "
                f"obs={obs_dt:.4f}s, "
                f"total={total_dt:.4f}s, "
                f"effective_hz={info['step_timing']['effective_hz']:.2f}"
            )

        return obs, reward, terminated, truncated, info

    def close(self):
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
            left_gripper_action = float(action[6])
            right_arm_action = action[7:13]
            right_gripper_action = float(action[13])

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

            hw_left = self._map_gripper_cmd_to_hardware(left_gripper_action, "left")
            hw_right = self._map_gripper_cmd_to_hardware(right_gripper_action, "right")

            self._send_ros_poses(
                clipped_pose_l,
                clipped_pose_r,
                hw_left,
                hw_right,
            )
        else:
            arm_action = action[0:6]
            gripper_action = float(action[6])

            ee_key = f"{self.arm_side}_ee_pose"
            pose = self._last_valid_state[ee_key].copy()

            next_xyz = pose[:3] + arm_action[0:3] * self.pos_scale
            next_quat = apply_delta_rotation(
                pose[3:7],
                arm_action[3:6] * self.rot_scale,
            )
            clipped_pose = self.clip_safety_box(np.concatenate([next_xyz, next_quat]))

            hw_gripper = self._map_gripper_cmd_to_hardware(
                gripper_action,
                arm_side=self.arm_side,
            )

            self._send_ros_pose(self.arm_side, clipped_pose, hw_gripper)

    def _send_ros_pose(self, arm_side: str, pose, gripper):
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
        self._sync_last_hw_gripper_cmd_from_state(formatted_obs["state"])

        h, w = self.obs_image_size

        for key in self.image_keys:
            if key in img_dict:
                img_bgr = img_dict[key]

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

        self.arm_side = str(getattr(self.config, "ARM_SIDE", "right")).lower()
        if self.arm_mode == "single" and self.arm_side not in ("left", "right"):
            raise ValueError(f"单臂模式下 config.ARM_SIDE 必须是 'left' 或 'right'，当前为 {self.arm_side!r}")

        # ==========================================================
        # 1. 通用运行参数
        # ==========================================================
        self.display_images = self._require_config_attr("DISPLAY_IMAGES")
        self.save_video = save_video
        self.hz = self._require_config_attr("HZ")
        self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
        self.curr_path_length = 0

        self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
        self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
        self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

        self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
        self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
        self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

        self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

        # ==========================================================
        # 2. 动作缩放
        # ==========================================================
        self.pos_scale = float(self._require_config_attr("POS_SCALE"))
        self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

        # ==========================================================
        # 3. 工作空间限位
        # ==========================================================
        xyz_low = np.asarray(self._require_config_attr("XYZ_LIMIT_LOW"), dtype=np.float64)
        xyz_high = np.asarray(self._require_config_attr("XYZ_LIMIT_HIGH"), dtype=np.float64)
        rpy_low = np.asarray(self._require_config_attr("RPY_LIMIT_LOW"), dtype=np.float64)
        rpy_high = np.asarray(self._require_config_attr("RPY_LIMIT_HIGH"), dtype=np.float64)

        self.xyz_bounding_box = gym.spaces.Box(xyz_low, xyz_high, dtype=np.float64)
        self.rpy_bounding_box = gym.spaces.Box(rpy_low, rpy_high, dtype=np.float64)

        # ==========================================================
        # 4. 录像缓冲区
        # ==========================================================
        self.recording_frames = []
        if self.save_video:
            print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")

        # ==========================================================
        # 5. 全局 ESC 急停
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
        # 6. ROS2 通信
        # ==========================================================
        self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

        # ==========================================================
        # 7. 相机初始化
        # ==========================================================
        print("正在启动 USB 直连相机阵列...")
        self.multi_cap = self._build_multi_camera_capture()
        print("相机阵列启动完毕！")

        # ==========================================================
        # 8. 显示线程
        # ==========================================================
        if self.display_images:
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
        # 9. Gym 空间定义
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

        # ==========================================================
        # 10. 夹爪命令/反馈配置与 hold 记忆
        # ----------------------------------------------------------
        # 关键设计：
        # - _last_hw_gripper_cmd 只记录“上一条明确发给硬件的目标命令”。
        # - 不再把连续真实反馈值直接写入 _last_hw_gripper_cmd。
        # - 否则 close 后反馈从 17 慢慢回弹到 30+ 时，hold 会被污染成 30+，
        #   导致夹爪越 hold 越张开。
        #
        # 推荐在任务 config 中显式加入：
        #   GRIPPER_CLOSE_CMD = 10.0
        #   GRIPPER_OPEN_CMD = 80.0
        #   GRIPPER_FEEDBACK_CLOSE_MAX = 30.0
        #   GRIPPER_FEEDBACK_OPEN_MIN = 70.0
        # ==========================================================
        self.gripper_close_cmd = float(getattr(self.config, "GRIPPER_CLOSE_CMD", 10.0))
        self.gripper_open_cmd = float(getattr(self.config, "GRIPPER_OPEN_CMD", 80.0))
        self.gripper_feedback_close_max = float(
            getattr(self.config, "GRIPPER_FEEDBACK_CLOSE_MAX", 30.0)
        )
        self.gripper_feedback_open_min = float(
            getattr(self.config, "GRIPPER_FEEDBACK_OPEN_MIN", 70.0)
        )

        # 当历史 memory 被污染成 middle 值时，hold 如何吸附。
        # USB 插入任务里，闭合后保持更重要，所以默认 middle -> close。
        # 可在 config 中设为 "open"。
        self.gripper_hold_middle_default = str(
            getattr(self.config, "GRIPPER_HOLD_MIDDLE_DEFAULT", "close")
        ).lower()

        self._last_hw_gripper_cmd = {
            "left": self.gripper_open_cmd,
            "right": self.gripper_open_cmd,
        }

    def _require_config_attr(self, name: str):
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
    # gripper 反馈 / 保持状态辅助
    # ==========================================================
    def _extract_gripper_feedback_from_state(self, state_dict, arm_side: str):
        key = f"{arm_side}_gripper"
        if state_dict is None or key not in state_dict:
            return None
        arr = np.asarray(state_dict[key]).reshape(-1)
        if arr.size == 0:
            return None
        return float(arr[-1])

    def _sync_last_hw_gripper_cmd_from_state(self, state_dict):
        """
        用真实 feedback 只做“离散状态同步”，不要把连续 feedback
        直接写成 hold 命令。

        为什么不能直接同步连续反馈：
        - close 后真实反馈可能从 17 慢慢回弹到 22、26、30、32；
        - 如果每次把 feedback 写入 _last_hw_gripper_cmd，hold(0) 就会继续发 22/26/32；
        - 结果就是夹爪越 hold 越张开。

        新规则：
        - feedback <= close_max: 确认真实处于闭合区，hold 目标吸附为 close_cmd；
        - feedback >= open_min : 确认真实处于张开区，hold 目标吸附为 open_cmd；
        - middle/unclear       : 不更新 memory，保持上一条明确命令。
        """
        for arm_side in ("left", "right"):
            val = self._extract_gripper_feedback_from_state(state_dict, arm_side)
            if val is None or not np.isfinite(val):
                continue

            val = float(val)
            if val <= self.gripper_feedback_close_max:
                self._last_hw_gripper_cmd[arm_side] = self.gripper_close_cmd
            elif val >= self.gripper_feedback_open_min:
                self._last_hw_gripper_cmd[arm_side] = self.gripper_open_cmd
            else:
                # middle/unclear: 不更新，避免机械回弹污染 hold 目标。
                pass

    def _resolve_hold_gripper_cmd(self, arm_side: str) -> float:
        """
        hold(0) 时返回上一条明确硬件目标命令。

        防御逻辑：如果历史版本曾把 memory 污染成 30~60 的中间值，
        这里会把中间值吸附到 close/open 中一个稳定目标。
        默认 middle -> close，因为 USB 插入任务里闭合保持更关键。
        """
        cmd = float(self._last_hw_gripper_cmd.get(arm_side, self.gripper_open_cmd))

        if cmd <= self.gripper_feedback_close_max:
            return self.gripper_close_cmd
        if cmd >= self.gripper_feedback_open_min:
            return self.gripper_open_cmd

        if self.gripper_hold_middle_default == "open":
            return self.gripper_open_cmd
        return self.gripper_close_cmd

    def _remember_explicit_gripper_cmd(self, arm_side: str, hw: float):
        """
        只把明确 close/open 的硬件目标写入 hold memory。
        中间硬件值可以被发送，但不污染 hold memory。
        """
        hw = float(hw)
        if hw <= self.gripper_feedback_close_max:
            self._last_hw_gripper_cmd[arm_side] = self.gripper_close_cmd
        elif hw >= self.gripper_feedback_open_min:
            self._last_hw_gripper_cmd[arm_side] = self.gripper_open_cmd

    # ==========================================================
    # gripper 语义 -> 硬件量程 映射
    # ==========================================================
    def _map_gripper_cmd_to_hardware(self, cmd: float, arm_side: str) -> float:
        """
        将上层 gripper 语义映射为硬件量程。

        上层训练/回放动作语义：
        - cmd <= -0.5 -> close event，发送 GRIPPER_CLOSE_CMD，例如 10；
        - cmd >=  0.5 -> open event，发送 GRIPPER_OPEN_CMD，例如 80；
        - 中间区      -> hold，继续发送上一条明确目标命令 10/80。

        硬件量程直传语义：
        - reset / interpolate_move_single 可能直接传 80 或 10；
        - 这类命令直接发送；
        - 只有明确 close/open 的硬件量程才更新 hold memory；
        - 30~60 这类 middle 值不会更新 hold memory。
        """
        cmd = float(cmd)

        # 硬件量程直传：reset / interpolate_move_single 等路径使用。
        # 注意：step() 的归一化动作已被 clip 到 [-1, 1]，不会误入这里。
        if 0.0 <= cmd <= 100.0 and abs(cmd) > 5.0:
            hw = cmd
            self._remember_explicit_gripper_cmd(arm_side, hw)
            return hw

        # 归一化事件命令。
        if cmd >= 0.5:
            hw = self.gripper_open_cmd
            self._last_hw_gripper_cmd[arm_side] = hw
            return hw

        if cmd <= -0.5:
            hw = self.gripper_close_cmd
            self._last_hw_gripper_cmd[arm_side] = hw
            return hw

        # hold：持续发送上一条明确目标命令。
        hw = self._resolve_hold_gripper_cmd(arm_side)
        self._last_hw_gripper_cmd[arm_side] = hw
        return hw

    # ==========================================================
    # 安全 / 复位基础工具
    # ==========================================================
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
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

    def interpolate_move_dual(
        self,
        goal_l: np.ndarray,
        goal_r: np.ndarray,
        timeout: float,
        g_l: float = None,
        g_r: float = None,
    ):
        """
        双臂平滑移动基础工具。
        如果 g_l / g_r 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
        """
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        curr_l = self._last_valid_state["left_ee_pose"].copy()
        curr_r = self._last_valid_state["right_ee_pose"].copy()

        if g_l is None:
            g_l = self._resolve_hold_gripper_cmd("left")
        if g_r is None:
            g_r = self._resolve_hold_gripper_cmd("right")

        path_l = np.linspace(curr_l, goal_l, steps)
        path_r = np.linspace(curr_r, goal_r, steps)

        print(f"🤖 正在执行双臂平滑复位/移动 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            self._send_ros_poses(path_l[i], path_r[i], g_l=float(g_l), g_r=float(g_r))
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def interpolate_move_single(
        self,
        goal_pose: np.ndarray,
        timeout: float,
        gripper: float = None,
    ):
        """
        单臂平滑移动基础工具。
        如果 gripper 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
        """
        steps = int(timeout * self.hz)
        self._get_sync_obs()
        ee_key = f"{self.arm_side}_ee_pose"
        curr_pose = self._last_valid_state[ee_key].copy()

        if gripper is None:
            gripper = self._resolve_hold_gripper_cmd(self.arm_side)

        path = np.linspace(curr_pose, goal_pose, steps)

        print(f"🤖 正在执行单臂平滑复位/移动 (共 {steps} 步)...")
        for i in range(steps):
            step_start = time.time()
            # 这里传的是硬件量程；若 gripper=None，上面已解析为“保持当前夹爪状态”
            self._send_ros_pose(self.arm_side, path[i], gripper)
            dt = time.time() - step_start
            time.sleep(max(0, (1.0 / self.hz) - dt))
        self._get_sync_obs()

    def go_to_reset(self):
        """
        通用 env 不再持有任务级 reset 逻辑。
        具体任务请在对应 wrapper / task env 中覆写这个方法，
        并在那里决定：
        - reset 目标位姿
        - 是否随机扰动
        - 是否改变夹爪
        - 用什么 gripper 硬件值
        """
        raise NotImplementedError(
            "GalaxeaArmEnv.go_to_reset() 已改为任务级接口。"
            "请在具体任务 wrapper / env 中实现 reset 逻辑。"
        )

    def reset(self, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.curr_path_length = 0
        self.terminate = False

        # 不再在通用 env 里强制把夹爪 reset 成张开
        self.go_to_reset()

        self._last_valid_state = None
        obs = self._get_sync_obs()

        # 用真实反馈同步“上一夹爪状态”，让后续 hold 更可信
        self._sync_last_hw_gripper_cmd_from_state(obs["state"])

        return obs, {"succeed": False}

    # ==========================================================
    # 主 step / close
    # ==========================================================
    def step(self, action: np.ndarray):
        start_time = time.time()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = 14 if self.arm_mode == "dual" else 7

        if action.shape[0] != expected_dim:
            raise ValueError(
                f"动作维度错误：当前 arm_mode={self.arm_mode}, "
                f"期望 {expected_dim} 维，实际收到 {action.shape}"
            )

        # 这里仍然保持官方风格：step 接收归一化动作，统一 clip 到 [-1,1]
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
            left_gripper_action = float(action[6])
            right_arm_action = action[7:13]
            right_gripper_action = float(action[13])

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

            hw_left = self._map_gripper_cmd_to_hardware(left_gripper_action, "left")
            hw_right = self._map_gripper_cmd_to_hardware(right_gripper_action, "right")

            self._send_ros_poses(
                clipped_pose_l,
                clipped_pose_r,
                hw_left,
                hw_right,
            )
        else:
            arm_action = action[0:6]
            gripper_action = float(action[6])

            ee_key = f"{self.arm_side}_ee_pose"
            pose = self._last_valid_state[ee_key].copy()

            next_xyz = pose[:3] + arm_action[0:3] * self.pos_scale
            next_quat = apply_delta_rotation(
                pose[3:7],
                arm_action[3:6] * self.rot_scale,
            )
            clipped_pose = self.clip_safety_box(np.concatenate([next_xyz, next_quat]))

            hw_gripper = self._map_gripper_cmd_to_hardware(
                gripper_action,
                arm_side=self.arm_side,
            )

            self._send_ros_pose(self.arm_side, clipped_pose, hw_gripper)

    def _send_ros_pose(self, arm_side: str, pose, gripper):
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
        self._sync_last_hw_gripper_cmd_from_state(formatted_obs["state"])

        h, w = self.obs_image_size

        for key in self.image_keys:
            if key in img_dict:
                img_bgr = img_dict[key]

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


# class GalaxeaArmEnv(gym.Env):
#     metadata = {"render_modes": []}

#     def __init__(self, config, cfg, save_video=False):
#         super().__init__()

#         self.config = config

#         # ==========================================================
#         # 0. 模式配置
#         # ==========================================================
#         self.arm_mode = str(getattr(self.config, "ARM_MODE", "dual")).lower()
#         if self.arm_mode not in ("dual", "single"):
#             raise ValueError(f"config.ARM_MODE 必须是 'dual' 或 'single'，当前为 {self.arm_mode!r}")

#         self.arm_side = str(getattr(self.config, "ARM_SIDE", "right")).lower()
#         if self.arm_mode == "single" and self.arm_side not in ("left", "right"):
#             raise ValueError(f"单臂模式下 config.ARM_SIDE 必须是 'left' 或 'right'，当前为 {self.arm_side!r}")

#         # ==========================================================
#         # 1. 通用运行参数
#         # ==========================================================
#         self.display_images = self._require_config_attr("DISPLAY_IMAGES")
#         self.save_video = save_video
#         self.hz = self._require_config_attr("HZ")
#         self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
#         self.curr_path_length = 0

#         self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
#         self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
#         self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

#         self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
#         self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
#         self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

#         self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

#         # ==========================================================
#         # 2. 动作缩放
#         # ==========================================================
#         self.pos_scale = float(self._require_config_attr("POS_SCALE"))
#         self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

#         # ==========================================================
#         # 3. 工作空间限位
#         # ==========================================================
#         xyz_low = np.asarray(self._require_config_attr("XYZ_LIMIT_LOW"), dtype=np.float64)
#         xyz_high = np.asarray(self._require_config_attr("XYZ_LIMIT_HIGH"), dtype=np.float64)
#         rpy_low = np.asarray(self._require_config_attr("RPY_LIMIT_LOW"), dtype=np.float64)
#         rpy_high = np.asarray(self._require_config_attr("RPY_LIMIT_HIGH"), dtype=np.float64)

#         self.xyz_bounding_box = gym.spaces.Box(xyz_low, xyz_high, dtype=np.float64)
#         self.rpy_bounding_box = gym.spaces.Box(rpy_low, rpy_high, dtype=np.float64)

#         # ==========================================================
#         # 4. 录像缓冲区
#         # ==========================================================
#         self.recording_frames = []
#         if self.save_video:
#             print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")

#         # ==========================================================
#         # 5. 全局 ESC 急停
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
#         # 6. ROS2 通信
#         # ==========================================================
#         self.bridge = Ros2Bridge(config, cfg, use_recv_time=True)

#         # ==========================================================
#         # 7. 相机初始化
#         # ==========================================================
#         print("正在启动 USB 直连相机阵列...")
#         self.multi_cap = self._build_multi_camera_capture()
#         print("相机阵列启动完毕！")

#         # ==========================================================
#         # 8. 显示线程
#         # ==========================================================
#         if self.display_images:
#             self.img_queue = queue.Queue(maxsize=1)
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
#         # 9. Gym 空间定义
#         # ==========================================================
#         h, w = self.obs_image_size
#         image_space = {
#             key: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
#             for key in self.image_keys
#         }

#         if self.arm_mode == "dual":
#             self.action_space = gym.spaces.Box(
#                 low=-1.0,
#                 high=1.0,
#                 shape=(14,),
#                 dtype=np.float32,
#             )
#             state_space = {
#                 "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#                 "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#             }
#         else:
#             self.action_space = gym.spaces.Box(
#                 low=-1.0,
#                 high=1.0,
#                 shape=(7,),
#                 dtype=np.float32,
#             )
#             ee_key = f"{self.arm_side}_ee_pose"
#             gripper_key = f"{self.arm_side}_gripper"
#             state_space = {
#                 ee_key: gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 gripper_key: gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#             }

#         self.observation_space = gym.spaces.Dict(
#             {
#                 "state": gym.spaces.Dict(state_space),
#                 "images": gym.spaces.Dict(image_space),
#             }
#         )

#         self._last_valid_state = None
#         self._printed_missing_state_warning = False

#         # ==========================================================
#         # 10. 记录上一帧真正发给硬件的夹爪量程
#         # ----------------------------------------------------------
#         # 注意：
#         # - 这里只是“保持上一状态”所需的内部记忆，不再代表 env 要强制 reset 到张开。
#         # - 每次获取最新观测时，会尽量从真实反馈同步更新。
#         # ==========================================================
#         self._last_hw_gripper_cmd = {
#             "left": 80.0,
#             "right": 80.0,
#         }

#     def _require_config_attr(self, name: str):
#         if not hasattr(self.config, name):
#             raise AttributeError(f"GalaxeaArmEnv 初始化失败：缺少必需配置项 config.{name}")
#         return getattr(self.config, name)

#     # ==========================================================
#     # 相机构建
#     # ==========================================================
#     def _build_multi_camera_capture(self):
#         realsense_cameras = self._require_config_attr("REALSENSE_CAMERAS")
#         head_camera_cfg = self._require_config_attr("HEAD_CAMERA")

#         caps = {}

#         for cam_name, cam_cfg in realsense_cameras.items():
#             rs_kwargs = dict(cam_cfg)
#             rs_kwargs.setdefault("name", cam_name)
#             caps[cam_name] = RSCapture(**rs_kwargs)

#         required_head_keys = ["device_index", "api", "name", "split_left_half"]
#         for key in required_head_keys:
#             if key not in head_camera_cfg:
#                 raise AttributeError(f"GalaxeaArmEnv 初始化失败：HEAD_CAMERA 缺少必需字段 '{key}'")

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
#     # gripper 反馈 / 保持状态辅助
#     # ==========================================================
#     def _extract_gripper_feedback_from_state(self, state_dict, arm_side: str):
#         key = f"{arm_side}_gripper"
#         if state_dict is None or key not in state_dict:
#             return None
#         arr = np.asarray(state_dict[key]).reshape(-1)
#         if arr.size == 0:
#             return None
#         return float(arr[-1])

#     def _sync_last_hw_gripper_cmd_from_state(self, state_dict):
#         """
#         尽量用最新真实反馈同步内部“上一夹爪命令”。
#         这样当 action 落在 hold 死区时，更接近真实当前夹爪状态。
#         """
#         for arm_side in ("left", "right"):
#             val = self._extract_gripper_feedback_from_state(state_dict, arm_side)
#             if val is None:
#                 continue
#             if np.isfinite(val):
#                 self._last_hw_gripper_cmd[arm_side] = float(val)

#     def _resolve_hold_gripper_cmd(self, arm_side: str) -> float:
#         """
#         当 reset / hold 不想改变夹爪时，用当前记忆到的硬件量程继续保持。
#         """
#         return float(self._last_hw_gripper_cmd[arm_side])

#     # ==========================================================
#     # gripper 语义 -> 硬件量程 映射
#     # ==========================================================
#     def _map_gripper_cmd_to_hardware(self, cmd: float, arm_side: str) -> float:
#         """
#         将上层 gripper 语义映射为硬件量程。
#         规则：
#         - cmd <= -0.5  -> 闭合 10
#         - cmd >=  0.5  -> 张开 80
#         - 中间区       -> 保持上一状态
#         - 如果本身已经是 0~100 且明显不是归一化值，则直接透传
#           （给任务级 reset / interpolate_move_single 这种路径用）
#         """
#         cmd = float(cmd)

#         # 已经是硬件量程值：直接透传
#         if 0.0 <= cmd <= 100.0 and abs(cmd) > 5.0:
#             self._last_hw_gripper_cmd[arm_side] = cmd
#             return cmd

#         if cmd >= 0.5:
#             hw = 80.0
#         elif cmd <= -0.5:
#             hw = 10.0
#         else:
#             hw = self._last_hw_gripper_cmd[arm_side]

#         self._last_hw_gripper_cmd[arm_side] = hw
#         return hw

#     # ==========================================================
#     # 安全 / 复位基础工具
#     # ==========================================================
#     def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
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

#     def interpolate_move_dual(
#         self,
#         goal_l: np.ndarray,
#         goal_r: np.ndarray,
#         timeout: float,
#         g_l: float = None,
#         g_r: float = None,
#     ):
#         """
#         双臂平滑移动基础工具。
#         如果 g_l / g_r 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
#         """
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()
#         curr_l = self._last_valid_state["left_ee_pose"].copy()
#         curr_r = self._last_valid_state["right_ee_pose"].copy()

#         if g_l is None:
#             g_l = self._resolve_hold_gripper_cmd("left")
#         if g_r is None:
#             g_r = self._resolve_hold_gripper_cmd("right")

#         path_l = np.linspace(curr_l, goal_l, steps)
#         path_r = np.linspace(curr_r, goal_r, steps)

#         print(f"🤖 正在执行双臂平滑复位/移动 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             self._send_ros_poses(path_l[i], path_r[i], g_l=float(g_l), g_r=float(g_r))
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def interpolate_move_single(
#         self,
#         goal_pose: np.ndarray,
#         timeout: float,
#         gripper: float = None,
#     ):
#         """
#         单臂平滑移动基础工具。
#         如果 gripper 不传，则保持当前夹爪状态，不在通用 env 内强制 reset 夹爪。
#         """
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()
#         ee_key = f"{self.arm_side}_ee_pose"
#         curr_pose = self._last_valid_state[ee_key].copy()

#         if gripper is None:
#             gripper = self._resolve_hold_gripper_cmd(self.arm_side)

#         path = np.linspace(curr_pose, goal_pose, steps)

#         print(f"🤖 正在执行单臂平滑复位/移动 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             # 这里传的是硬件量程；若 gripper=None，上面已解析为“保持当前夹爪状态”
#             self._send_ros_pose(self.arm_side, path[i], gripper)
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def go_to_reset(self):
#         """
#         通用 env 不再持有任务级 reset 逻辑。
#         具体任务请在对应 wrapper / task env 中覆写这个方法，
#         并在那里决定：
#         - reset 目标位姿
#         - 是否随机扰动
#         - 是否改变夹爪
#         - 用什么 gripper 硬件值
#         """
#         raise NotImplementedError(
#             "GalaxeaArmEnv.go_to_reset() 已改为任务级接口。"
#             "请在具体任务 wrapper / env 中实现 reset 逻辑。"
#         )

#     def reset(self, **kwargs):
#         if self.save_video:
#             self.save_video_recording()

#         self.curr_path_length = 0
#         self.terminate = False

#         # 不再在通用 env 里强制把夹爪 reset 成张开
#         self.go_to_reset()

#         self._last_valid_state = None
#         obs = self._get_sync_obs()

#         # 用真实反馈同步“上一夹爪状态”，让后续 hold 更可信
#         self._sync_last_hw_gripper_cmd_from_state(obs["state"])

#         return obs, {"succeed": False}

#     # ==========================================================
#     # 主 step / close
#     # ==========================================================
#     def step(self, action: np.ndarray):
#         start_time = time.time()

#         action = np.asarray(action, dtype=np.float32).reshape(-1)
#         expected_dim = 14 if self.arm_mode == "dual" else 7

#         if action.shape[0] != expected_dim:
#             raise ValueError(
#                 f"动作维度错误：当前 arm_mode={self.arm_mode}, "
#                 f"期望 {expected_dim} 维，实际收到 {action.shape}"
#             )

#         # 这里仍然保持官方风格：step 接收归一化动作，统一 clip 到 [-1,1]
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         self._publish_action(action)

#         self.curr_path_length += 1

#         dt = time.time() - start_time
#         time.sleep(max(0, (1.0 / self.hz) - dt))

#         obs = self._get_sync_obs()
#         reward = 0

#         terminated = self.terminate
#         truncated = self.curr_path_length >= self.max_episode_length

#         return obs, reward, terminated, truncated, {}

#     def close(self):
#         if hasattr(self, "listener"):
#             self.listener.stop()

#         self.bridge.destroy()
#         self.multi_cap.close()

#         if self.display_images:
#             try:
#                 self.img_queue.put_nowait(None)
#             except queue.Full:
#                 pass
#             cv2.destroyAllWindows()
#             self.displayer.join()

#     # ==========================================================
#     # 视频保存
#     # ==========================================================
#     def save_video_recording(self):
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
#         """
#         将归一化动作映射为真实物理增量。

#         dual 模式:
#             [left(6)+left_gripper(1)+right(6)+right_gripper(1)] = 14
#         single 模式:
#             [arm(6)+gripper(1)] = 7
#         """
#         if self._last_valid_state is None:
#             if not self._printed_missing_state_warning:
#                 print("⚠️ _last_valid_state 还未准备好，本次动作发布被跳过。")
#                 self._printed_missing_state_warning = True
#             return

#         if self.arm_mode == "dual":
#             left_arm_action = action[0:6]
#             left_gripper_action = float(action[6])
#             right_arm_action = action[7:13]
#             right_gripper_action = float(action[13])

#             l_pose = self._last_valid_state["left_ee_pose"].copy()
#             next_xyz_l = l_pose[:3] + left_arm_action[0:3] * self.pos_scale
#             next_quat_l = apply_delta_rotation(
#                 l_pose[3:7],
#                 left_arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose_l = self.clip_safety_box(np.concatenate([next_xyz_l, next_quat_l]))

#             r_pose = self._last_valid_state["right_ee_pose"].copy()
#             next_xyz_r = r_pose[:3] + right_arm_action[0:3] * self.pos_scale
#             next_quat_r = apply_delta_rotation(
#                 r_pose[3:7],
#                 right_arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose_r = self.clip_safety_box(np.concatenate([next_xyz_r, next_quat_r]))

#             hw_left = self._map_gripper_cmd_to_hardware(left_gripper_action, "left")
#             hw_right = self._map_gripper_cmd_to_hardware(right_gripper_action, "right")

#             self._send_ros_poses(
#                 clipped_pose_l,
#                 clipped_pose_r,
#                 hw_left,
#                 hw_right,
#             )
#         else:
#             arm_action = action[0:6]
#             gripper_action = float(action[6])

#             ee_key = f"{self.arm_side}_ee_pose"
#             pose = self._last_valid_state[ee_key].copy()

#             next_xyz = pose[:3] + arm_action[0:3] * self.pos_scale
#             next_quat = apply_delta_rotation(
#                 pose[3:7],
#                 arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose = self.clip_safety_box(np.concatenate([next_xyz, next_quat]))

#             hw_gripper = self._map_gripper_cmd_to_hardware(
#                 gripper_action,
#                 arm_side=self.arm_side,
#             )

#             self._send_ros_pose(self.arm_side, clipped_pose, hw_gripper)

#     def _send_ros_pose(self, arm_side: str, pose, gripper):
#         pose = np.asarray(pose, dtype=np.float64).reshape(-1)

#         msg_pose = PoseStamped()
#         msg_pose.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_pose.header.frame_id = "base_link"
#         msg_pose.pose.position.x = float(pose[0])
#         msg_pose.pose.position.y = float(pose[1])
#         msg_pose.pose.position.z = float(pose[2])
#         msg_pose.pose.orientation.x = float(pose[3])
#         msg_pose.pose.orientation.y = float(pose[4])
#         msg_pose.pose.orientation.z = float(pose[5])
#         msg_pose.pose.orientation.w = float(pose[6])

#         msg_gripper = JointState()
#         msg_gripper.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_gripper.name = [f"R1PRO_{arm_side}_gripper_joint"]
#         msg_gripper.position = [float(gripper)]

#         ee_key = f"{arm_side}_ee_pose"
#         gripper_key = f"{arm_side}_gripper"

#         try:
#             self.bridge.publishers[self.bridge.topics_config.action[ee_key]].publish(msg_pose)
#             self.bridge.publishers[self.bridge.topics_config.action[gripper_key]].publish(msg_gripper)
#         except KeyError as e:
#             print(f"⚠️ 发布单臂动作时出现异常: {e}")

#     def _send_ros_poses(self, p_l, p_r, g_l, g_r):
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
#             print(f"⚠️ 发布双臂动作时出现异常: {e}")

#     # ==========================================================
#     # 观测抓取
#     # ==========================================================
#     def _get_sync_obs(self):
#         img_dict = self.multi_cap.read()
#         if img_dict is None:
#             img_dict = {}

#         raw_state = None
#         start_wait = time.time()
#         timeout_s = 5.0

#         while raw_state is None:
#             raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
#             if raw_state is None:
#                 if time.time() - start_wait > timeout_s:
#                     raise TimeoutError("等待 ROS 状态超时（5 秒内未拿到有效状态）")
#                 time.sleep(0.01)

#         formatted_obs = {"state": {}, "images": {}}
#         full_res_images = {}
#         display_images = {}

#         state_src = raw_state.get("state", {}) if raw_state is not None else {}

#         if self.arm_mode == "dual":
#             state_keys = ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
#         else:
#             state_keys = [f"{self.arm_side}_ee_pose", f"{self.arm_side}_gripper"]

#         for key in state_keys:
#             if key in state_src:
#                 formatted_obs["state"][key] = state_src[key].numpy().flatten()
#             else:
#                 dim = 7 if "pose" in key else 1
#                 formatted_obs["state"][key] = np.zeros(dim, dtype=np.float32)

#         self._last_valid_state = formatted_obs["state"]
#         self._sync_last_hw_gripper_cmd_from_state(formatted_obs["state"])

#         h, w = self.obs_image_size

#         for key in self.image_keys:
#             if key in img_dict:
#                 img_bgr = img_dict[key]

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
#                 raise RuntimeError(f"相机 {key} 离线，无法继续采样。")

#         if self.save_video:
#             self.recording_frames.append(full_res_images)

#         if self.display_images:
#             if self.img_queue.full():
#                 try:
#                     self.img_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             try:
#                 self.img_queue.put_nowait(display_images)
#             except queue.Full:
#                 pass

#         return formatted_obs


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


# class GalaxeaArmEnv(gym.Env):
#     metadata = {"render_modes": []}

#     def __init__(self, config, cfg, save_video=False):
#         super().__init__()

#         self.config = config

#         # ==========================================================
#         # 0. 模式配置
#         # ==========================================================
#         self.arm_mode = str(getattr(self.config, "ARM_MODE", "dual")).lower()
#         if self.arm_mode not in ("dual", "single"):
#             raise ValueError(f"config.ARM_MODE 必须是 'dual' 或 'single'，当前为 {self.arm_mode!r}")

#         self.arm_side = str(getattr(self.config, "ARM_SIDE", "right")).lower()
#         if self.arm_mode == "single" and self.arm_side not in ("left", "right"):
#             raise ValueError(f"单臂模式下 config.ARM_SIDE 必须是 'left' 或 'right'，当前为 {self.arm_side!r}")

#         # ==========================================================
#         # 1. 通用运行参数
#         # ==========================================================
#         self.display_images = self._require_config_attr("DISPLAY_IMAGES")
#         self.save_video = save_video
#         self.hz = self._require_config_attr("HZ")
#         self.max_episode_length = self._require_config_attr("MAX_EPISODE_LENGTH")
#         self.curr_path_length = 0

#         self.image_keys = list(self._require_config_attr("ENV_IMAGE_KEYS"))
#         self.display_image_keys = list(self._require_config_attr("DISPLAY_IMAGE_KEYS"))
#         self.obs_image_size = tuple(self._require_config_attr("IMAGE_OBS_SIZE"))

#         self.display_window_name = self._require_config_attr("DISPLAY_WINDOW_NAME")
#         self.display_window_size = tuple(self._require_config_attr("DISPLAY_WINDOW_SIZE"))
#         self.display_frame_size = tuple(self._require_config_attr("DISPLAY_FRAME_SIZE"))

#         self.head_camera_key = self._require_config_attr("HEAD_CAMERA_KEY")

#         # ==========================================================
#         # 2. 复位坐标
#         # ==========================================================
#         if self.arm_mode == "dual":
#             self.reset_l = np.array(self._require_config_attr("RESET_L"), dtype=np.float32)
#             self.reset_r = np.array(self._require_config_attr("RESET_R"), dtype=np.float32)
#         else:
#             if hasattr(self.config, "RESET_POSE"):
#                 self.reset_pose = np.array(self.config.RESET_POSE, dtype=np.float32)
#             else:
#                 fallback_key = "RESET_L" if self.arm_side == "left" else "RESET_R"
#                 self.reset_pose = np.array(self._require_config_attr(fallback_key), dtype=np.float32)

#         # ==========================================================
#         # 3. 动作缩放
#         # ==========================================================
#         self.pos_scale = float(self._require_config_attr("POS_SCALE"))
#         self.rot_scale = float(self._require_config_attr("ROT_SCALE"))

#         # ==========================================================
#         # 4. 工作空间限位
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
#         self.recording_frames = []
#         if self.save_video:
#             print("🎥 视频录制模式已激活 (视频将保存在 ./videos 目录)")

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
#         # 8. 相机初始化
#         # ==========================================================
#         print("正在启动 USB 直连相机阵列...")
#         self.multi_cap = self._build_multi_camera_capture()
#         print("相机阵列启动完毕！")

#         # ==========================================================
#         # 9. 显示线程
#         # ==========================================================
#         if self.display_images:
#             self.img_queue = queue.Queue(maxsize=1)
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

#         if self.arm_mode == "dual":
#             self.action_space = gym.spaces.Box(
#                 low=-1.0,
#                 high=1.0,
#                 shape=(14,),
#                 dtype=np.float32,
#             )
#             state_space = {
#                 "left_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 "right_ee_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 "left_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#                 "right_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#             }
#         else:
#             self.action_space = gym.spaces.Box(
#                 low=-1.0,
#                 high=1.0,
#                 shape=(7,),
#                 dtype=np.float32,
#             )
#             ee_key = f"{self.arm_side}_ee_pose"
#             gripper_key = f"{self.arm_side}_gripper"
#             state_space = {
#                 ee_key: gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                 gripper_key: gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
#             }

#         self.observation_space = gym.spaces.Dict(
#             {
#                 "state": gym.spaces.Dict(state_space),
#                 "images": gym.spaces.Dict(image_space),
#             }
#         )

#         self._last_valid_state = None
#         self._printed_missing_state_warning = False

#         # ==========================================================
#         # 11. 记录上一帧真正发给硬件的夹爪量程
#         # 这样 action[6] 在死区内时，可以“保持上一状态”
#         # ==========================================================
#         self._last_hw_gripper_cmd = {
#             "left": 80.0,
#             "right": 80.0,
#         }

#     def _require_config_attr(self, name: str):
#         if not hasattr(self.config, name):
#             raise AttributeError(f"GalaxeaArmEnv 初始化失败：缺少必需配置项 config.{name}")
#         return getattr(self.config, name)

#     # ==========================================================
#     # 相机构建
#     # ==========================================================
#     def _build_multi_camera_capture(self):
#         realsense_cameras = self._require_config_attr("REALSENSE_CAMERAS")
#         head_camera_cfg = self._require_config_attr("HEAD_CAMERA")

#         caps = {}

#         for cam_name, cam_cfg in realsense_cameras.items():
#             rs_kwargs = dict(cam_cfg)
#             rs_kwargs.setdefault("name", cam_name)
#             caps[cam_name] = RSCapture(**rs_kwargs)

#         required_head_keys = ["device_index", "api", "name", "split_left_half"]
#         for key in required_head_keys:
#             if key not in head_camera_cfg:
#                 raise AttributeError(f"GalaxeaArmEnv 初始化失败：HEAD_CAMERA 缺少必需字段 '{key}'")

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
#     # gripper 语义 -> 硬件量程 映射
#     # ==========================================================
#     def _map_gripper_cmd_to_hardware(self, cmd: float, arm_side: str) -> float:
#         """
#         将上层 gripper 语义映射为硬件量程。
#         规则：
#         - cmd <= -0.5  -> 闭合 20
#         - cmd >=  0.5  -> 张开 80
#         - 中间区       -> 保持上一状态
#         - 如果本身已经是 0~100 且明显不是归一化值，则直接透传
#           （给 reset / interpolate_move_single 这种路径用）
#         """
#         cmd = float(cmd)

#         # reset 这类直接量程命令，直接透传
#         if 0.0 <= cmd <= 100.0 and abs(cmd) > 5.0:
#             self._last_hw_gripper_cmd[arm_side] = cmd
#             return cmd

#         if cmd >= 0.5:
#             hw = 80.0
#         elif cmd <= -0.5:
#             hw = 20.0
#         else:
#             hw = self._last_hw_gripper_cmd[arm_side]

#         self._last_hw_gripper_cmd[arm_side] = hw
#         return hw

#     # ==========================================================
#     # 安全 / 复位
#     # ==========================================================
#     def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
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

#     def interpolate_move_dual(self, goal_l: np.ndarray, goal_r: np.ndarray, timeout: float):
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()
#         curr_l = self._last_valid_state["left_ee_pose"].copy()
#         curr_r = self._last_valid_state["right_ee_pose"].copy()

#         path_l = np.linspace(curr_l, goal_l, steps)
#         path_r = np.linspace(curr_r, goal_r, steps)

#         print(f"🤖 正在执行双臂平滑复位 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             self._send_ros_poses(path_l[i], path_r[i], g_l=1.0, g_r=1.0)
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def interpolate_move_single(self, goal_pose: np.ndarray, timeout: float, gripper: float = 100.0):
#         steps = int(timeout * self.hz)
#         self._get_sync_obs()
#         ee_key = f"{self.arm_side}_ee_pose"
#         curr_pose = self._last_valid_state[ee_key].copy()

#         path = np.linspace(curr_pose, goal_pose, steps)

#         print(f"🤖 正在执行单臂平滑复位 (共 {steps} 步)...")
#         for i in range(steps):
#             step_start = time.time()
#             # 注意：这里直接给的是硬件量程，不走 step/action_space clip
#             self._send_ros_pose(self.arm_side, path[i], gripper)
#             dt = time.time() - step_start
#             time.sleep(max(0, (1.0 / self.hz) - dt))
#         self._get_sync_obs()

#     def go_to_reset(self):
#         print("开始复位")
#         time.sleep(0.3)

#         if self.arm_mode == "dual":
#             self.interpolate_move_dual(self.reset_l, self.reset_r, timeout=1.5)
#         else:
#             self.interpolate_move_single(self.reset_pose, timeout=1.5, gripper=100.0)

#         time.sleep(0.5)

#     def reset(self, **kwargs):
#         if self.save_video:
#             self.save_video_recording()

#         self.curr_path_length = 0
#         self.terminate = False

#         # 每个 episode 开始时默认认为夹爪保持张开
#         self._last_hw_gripper_cmd["left"] = 80.0
#         self._last_hw_gripper_cmd["right"] = 80.0

#         self.go_to_reset()
#         self._last_valid_state = None
#         obs = self._get_sync_obs()
#         return obs, {"succeed": False}

#     # ==========================================================
#     # 主 step / close
#     # ==========================================================
#     def step(self, action: np.ndarray):
#         start_time = time.time()

#         action = np.asarray(action, dtype=np.float32).reshape(-1)
#         expected_dim = 14 if self.arm_mode == "dual" else 7

#         if action.shape[0] != expected_dim:
#             raise ValueError(
#                 f"动作维度错误：当前 arm_mode={self.arm_mode}, "
#                 f"期望 {expected_dim} 维，实际收到 {action.shape}"
#             )

#         # 这里仍然保持官方风格：step 接收归一化动作，统一 clip 到 [-1,1]
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         self._publish_action(action)

#         self.curr_path_length += 1

#         dt = time.time() - start_time
#         time.sleep(max(0, (1.0 / self.hz) - dt))

#         obs = self._get_sync_obs()
#         reward = 0

#         terminated = self.terminate
#         truncated = self.curr_path_length >= self.max_episode_length

#         return obs, reward, terminated, truncated, {}

#     def close(self):
#         if hasattr(self, "listener"):
#             self.listener.stop()

#         self.bridge.destroy()
#         self.multi_cap.close()

#         if self.display_images:
#             try:
#                 self.img_queue.put_nowait(None)
#             except queue.Full:
#                 pass
#             cv2.destroyAllWindows()
#             self.displayer.join()

#     # ==========================================================
#     # 视频保存
#     # ==========================================================
#     def save_video_recording(self):
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
#         """
#         将归一化动作映射为真实物理增量。

#         dual 模式:
#             [left(6)+left_gripper(1)+right(6)+right_gripper(1)] = 14
#         single 模式:
#             [arm(6)+gripper(1)] = 7
#         """
#         if self._last_valid_state is None:
#             if not self._printed_missing_state_warning:
#                 print("⚠️ _last_valid_state 还未准备好，本次动作发布被跳过。")
#                 self._printed_missing_state_warning = True
#             return

#         if self.arm_mode == "dual":
#             left_arm_action = action[0:6]
#             left_gripper_action = float(action[6])
#             right_arm_action = action[7:13]
#             right_gripper_action = float(action[13])

#             l_pose = self._last_valid_state["left_ee_pose"].copy()
#             next_xyz_l = l_pose[:3] + left_arm_action[0:3] * self.pos_scale
#             next_quat_l = apply_delta_rotation(
#                 l_pose[3:7],
#                 left_arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose_l = self.clip_safety_box(np.concatenate([next_xyz_l, next_quat_l]))

#             r_pose = self._last_valid_state["right_ee_pose"].copy()
#             next_xyz_r = r_pose[:3] + right_arm_action[0:3] * self.pos_scale
#             next_quat_r = apply_delta_rotation(
#                 r_pose[3:7],
#                 right_arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose_r = self.clip_safety_box(np.concatenate([next_xyz_r, next_quat_r]))

#             # 双臂这里也统一适配成硬件量程
#             hw_left = self._map_gripper_cmd_to_hardware(left_gripper_action, "left")
#             hw_right = self._map_gripper_cmd_to_hardware(right_gripper_action, "right")

#             self._send_ros_poses(
#                 clipped_pose_l,
#                 clipped_pose_r,
#                 hw_left,
#                 hw_right,
#             )
#         else:
#             arm_action = action[0:6]
#             gripper_action = float(action[6])

#             ee_key = f"{self.arm_side}_ee_pose"
#             pose = self._last_valid_state[ee_key].copy()

#             next_xyz = pose[:3] + arm_action[0:3] * self.pos_scale
#             next_quat = apply_delta_rotation(
#                 pose[3:7],
#                 arm_action[3:6] * self.rot_scale,
#             )
#             clipped_pose = self.clip_safety_box(np.concatenate([next_xyz, next_quat]))

#             # 单臂关键修复：在底层发布前才把 ±1 语义映射成硬件量程
#             hw_gripper = self._map_gripper_cmd_to_hardware(
#                 gripper_action,
#                 arm_side=self.arm_side,
#             )

#             self._send_ros_pose(self.arm_side, clipped_pose, hw_gripper)

#     def _send_ros_pose(self, arm_side: str, pose, gripper):
#         pose = np.asarray(pose, dtype=np.float64).reshape(-1)

#         msg_pose = PoseStamped()
#         msg_pose.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_pose.header.frame_id = "base_link"
#         msg_pose.pose.position.x = float(pose[0])
#         msg_pose.pose.position.y = float(pose[1])
#         msg_pose.pose.position.z = float(pose[2])
#         msg_pose.pose.orientation.x = float(pose[3])
#         msg_pose.pose.orientation.y = float(pose[4])
#         msg_pose.pose.orientation.z = float(pose[5])
#         msg_pose.pose.orientation.w = float(pose[6])

#         msg_gripper = JointState()
#         msg_gripper.header.stamp = self.bridge.node.get_clock().now().to_msg()
#         msg_gripper.name = [f"R1PRO_{arm_side}_gripper_joint"]
#         msg_gripper.position = [float(gripper)]

#         ee_key = f"{arm_side}_ee_pose"
#         gripper_key = f"{arm_side}_gripper"

#         try:
#             self.bridge.publishers[self.bridge.topics_config.action[ee_key]].publish(msg_pose)
#             self.bridge.publishers[self.bridge.topics_config.action[gripper_key]].publish(msg_gripper)
#         except KeyError as e:
#             print(f"⚠️ 发布单臂动作时出现异常: {e}")

#     def _send_ros_poses(self, p_l, p_r, g_l, g_r):
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
#             print(f"⚠️ 发布双臂动作时出现异常: {e}")

#     # ==========================================================
#     # 观测抓取
#     # ==========================================================
#     def _get_sync_obs(self):
#         img_dict = self.multi_cap.read()
#         if img_dict is None:
#             img_dict = {}

#         raw_state = None
#         start_wait = time.time()
#         timeout_s = 5.0

#         while raw_state is None:
#             raw_state = self.bridge.get_latest_obs_forcely(device=torch.device("cpu"))
#             if raw_state is None:
#                 if time.time() - start_wait > timeout_s:
#                     raise TimeoutError("等待 ROS 状态超时（5 秒内未拿到有效状态）")
#                 time.sleep(0.01)

#         formatted_obs = {"state": {}, "images": {}}
#         full_res_images = {}
#         display_images = {}

#         state_src = raw_state.get("state", {}) if raw_state is not None else {}

#         if self.arm_mode == "dual":
#             state_keys = ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
#         else:
#             state_keys = [f"{self.arm_side}_ee_pose", f"{self.arm_side}_gripper"]

#         for key in state_keys:
#             if key in state_src:
#                 formatted_obs["state"][key] = state_src[key].numpy().flatten()
#             else:
#                 dim = 7 if "pose" in key else 1
#                 formatted_obs["state"][key] = np.zeros(dim, dtype=np.float32)

#         self._last_valid_state = formatted_obs["state"]

#         h, w = self.obs_image_size

#         for key in self.image_keys:
#             if key in img_dict:
#                 img_bgr = img_dict[key]

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
#                 raise RuntimeError(f"相机 {key} 离线，无法继续采样。")

#         if self.save_video:
#             self.recording_frames.append(full_res_images)

#         if self.display_images:
#             if self.img_queue.full():
#                 try:
#                     self.img_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             try:
#                 self.img_queue.put_nowait(display_images)
#             except queue.Full:
#                 pass

#         return formatted_obs




