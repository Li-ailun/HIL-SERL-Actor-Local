# #可直接实例化的具体总配置类：
#官方 config.py 是“任务装配层”

# 官方 config.py 里其实分成两类东西：

# EnvConfig

# 放的是这个任务的静态配置参数：

# 相机序列号
# 图像裁剪
# TARGET_POSE
# RESET_POSE
# ACTION_SCALE
# RANDOM_RESET
# ABS_POSE_LIMIT_HIGH/LOW
# COMPLIANCE_PARAM
# PRECISION_PARAM
# MAX_EPISODE_LENGTH

# 这些都不是“执行逻辑”，而是“任务常量”。

# TrainConfig

# 放的是这个任务的训练侧配置和环境拼装逻辑：

# image_keys
# classifier_keys
# proprio_keys
# checkpoint_period
# discount
# encoder_type
# setup_mode
# get_environment(...)

# 尤其是 get_environment(...)，这里负责把整个环境一层层包起来：

# 先建 USBEnv(config=EnvConfig())
# 再加 SpacemouseIntervention
# 再加 RelativeFrame
# 再加 Quat2EulerWrapper
# 再加 SERLObsWrapper
# 再加 ChunkingWrapper
# 再按需加分类器 wrapper
# 最后再加 GripperPenaltyWrapper

# 所以官方 config.py 的职责是：

# “这个任务训练时，应该怎么组装环境和超参数”


# 这个文件应该保留两类内容：

# 1. EnvConfig

# 放任务常量：

# RESET_L / RESET_R
# RANDOM_RESET
# RANDOM_XY_RANGE
# 相机参数
# 限位
# 动作缩放
# MAX_EPISODE_LENGTH
# 任何任务静态常量
# 2. TrainConfig

# 放训练侧配置和环境拼装：

# image_keys
# classifier_keys
# proprio_keys
# discount
# checkpoint_period
# get_environment(...)




#环境配置
#(1)config的get_environment()：config起默认作用，谁引用它谁就可以在自己脚本里修改配置，不引用就使用config的默认配置
#              如果其他脚本引用了env = env_config.get_environment(fake_env=False,等）则可以在其他脚本里修改配置
#              如果其他脚本仅是 env = env_config.get_environment()  ，括号内无配置引用，则统一使用config里的配置
#（2）config的  GalaxeaUSBEnvConfig 里的：HZ = 15，DISPLAY_IMAGES = True等：
#              写死的，必须在config的GalaxeaUSBEnvConfig 里修改配置


import os
from typing import List
import cv2

import jax
import jax.numpy as jnp
import numpy as np

from examples.galaxea_task.config import DefaultTrainingConfig


class GalaxeaUSBEnvConfig:
    """USB 任务的专属硬件与物理配置。"""

    # ==============================
    # 1. 任务物理参数
    # ==============================
    RESET_L = np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    RESET_R = np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01

    # ==============================
    # 2. Env 基础运行参数
    # ==============================
    HZ = 15
    DISPLAY_IMAGES = True
    MAX_EPISODE_LENGTH = 10000

    # ==============================
    # 3. 图像 / 显示配置
    # ==============================
    ENV_IMAGE_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    DISPLAY_IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]
    IMAGE_OBS_SIZE = (128, 128)

    DISPLAY_WINDOW_NAME = "Galaxea Vision Monitor"
    DISPLAY_WINDOW_SIZE = (1400, 500)
    DISPLAY_FRAME_SIZE = (1152, 384)

    HEAD_CAMERA_KEY = "head_rgb"

    # ==============================
    # 4. 相机硬件配置
    # ==============================
    REALSENSE_CAMERAS = {
        "left_wrist_rgb": {
            "serial_number": "230322270950",
            "dim": (640, 480),
            "fps": 15,
        },
        "right_wrist_rgb": {
            "serial_number": "230322271216",
            "dim": (640, 480),
            "fps": 15,
        },
    }

    HEAD_CAMERA = {
        "device_index": 14,  #v4l2-ctl --list-devices查询
        "api": cv2.CAP_V4L2,  # # 显式指定使用 Linux V4L2 后端打开 /dev/videoX
        "fourcc": "MJPG",
        "frame_width": 1344,
        "frame_height": 376,
        "fps": 15,
        "name": "head_rgb",
        "split_left_half": True,
    }

    # ==============================
    # 5. 动作缩放
    # ==============================
    POS_SCALE = 0.01
    ROT_SCALE = 0.05

    # ==============================
    # 6. 安全工作空间限位
    # ==============================
    XYZ_LIMIT_LOW = np.array([0.1, -0.5, -0.1], dtype=np.float64)
    XYZ_LIMIT_HIGH = np.array([0.7, 0.5, 0.5], dtype=np.float64)

    RPY_LIMIT_LOW = np.array([-np.pi, -np.pi, -np.pi], dtype=np.float64)
    RPY_LIMIT_HIGH = np.array([np.pi, np.pi, np.pi], dtype=np.float64)

    # ==============================
    # 7. ROS2 发布配置
    # ==============================
    robot_config = {
        "hardware": "R1_PRO",
        "enable_publish": [
            "left_ee_pose",
            "right_ee_pose",
            "left_gripper",
            "right_gripper",
        ],
    }

    def __getitem__(self, key):
        if key == "robot":
            return self.robot_config
        raise KeyError(key)


class GalaxeaUSBTrainConfig(DefaultTrainingConfig):
    """USB 任务的训练配置与环境装配入口。"""

    # ==============================
    # 8. 训练超参数
    # ==============================
    agent: str = "sac"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.98

    max_steps: int = 1_000_000
    replay_buffer_capacity: int = 200_000
    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000
    checkpoint_period: int = 2000
    buffer_period: int = 1000

    # ==============================
    # 9. 观测 / 编码配置
    # ==============================
    image_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    classifier_keys: List[str] = ["head_rgb", "left_wrist_rgb"]
    proprio_keys: List[str] = [
        "left_ee_pose",
        "right_ee_pose",
        "left_gripper",
        "right_gripper",
    ]

    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "dual-arm-learned-gripper"

    # ==============================
    # 10. 环境装配
    # ==============================
    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,   #该系列任务默认不保存视频，指令控制保存？
        classifier: bool = False,   #什么意思
        use_vr: bool = True,
    ):
        """
        任务环境统一装配入口。
        """
        if fake_env:
            return None

        # 延迟导入，避免循环依赖
        from examples.galaxea_task.usb_pick_insertion.wrapper import (
            GalaxeaUSBEnv,
            DualGripperPenaltyWrapper,
        )
        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
        from serl_launcher.wrappers.chunking import ChunkingWrapper
        from serl_launcher.networks.reward_classifier import load_classifier_func
        from serl_robot_infra.Galaxea_env.envs.wrappers import (
            VRInterventionWrapper,
            MultiCameraBinaryRewardClassifierWrapper,
        )

        env_cfg = GalaxeaUSBEnvConfig()

        # 1) 核心真实环境
        env = GalaxeaUSBEnv(
            config=env_cfg,
            cfg={},
            save_video=save_video,
            use_vr=use_vr,
        )

        # 2) VR 接管逻辑
        if use_vr:
            env = VRInterventionWrapper(env)

        # 3) 通用观测包装
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 4) 视觉奖励分类器（仅在需要时开启）
        if classifier and self.classifier_keys:
            try:
                classifier_fn = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("classifier_ckpt/"),
                )

                def reward_func(obs):
                    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                    return int(sigmoid(classifier_fn(obs)) > 0.7)

                env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

            except Exception as e:
                print(f"⚠️ 分类器加载失败（如果你还没训练它，这属于正常现象）: {e}")

        # 5) 任务特有惩罚
        env = DualGripperPenaltyWrapper(env, penalty=-0.02)

        return env

    # ==============================
    # 11. demo 清洗
    # ==============================
    def process_demos(self, transitions):
        """
        对齐官方接口：
        这里只做最基础的零动作过滤。
        """
        processed = []
        for trans in transitions:
            if np.linalg.norm(np.asarray(trans["actions"])) > 1e-4:
                processed.append(trans)
        return processed


# 导出实例，供外部直接 import 使用
env_config = GalaxeaUSBTrainConfig()


