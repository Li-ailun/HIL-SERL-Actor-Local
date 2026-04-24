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
    """
    usb_pick_insertion_single 的硬件与物理配置。

    当前目标：
    1. 使用统一底层 env（GalaxeaArmEnv）
    2. 当前任务切到单臂模式
    3. 控制右臂
    4. 三个相机全部保留
    5. 任务级 reset 完全在这里定义：
       - RESET_POSE
       - RESET_GRIPPER
       - RESET_TIMEOUT_SEC
       - RANDOM_RESET / RANDOM_XY_RANGE
    """

    # ==============================
    # 0. 单 / 双臂模式配置
    # ==============================
    ARM_MODE = "single"
    ARM_SIDE = "right"

    # ==============================
    # 1. 任务级 reset 配置
    # 说明：
    # - 通用 GalaxeaArmEnv 不再定义任务级 reset
    # - 由具体任务 wrapper 读取这些字段执行 reset
    # ==============================
    RESET_POSE = np.array(
        [
            0.2,
            -0.3,
            -0.15,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )

    # reset 时夹爪硬件量程
    # 你要求“张开定义成 80”
    RESET_GRIPPER = 80.0

    # reset 扰动（config里生效，不是由wrapper决定）
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01


    # ==============================
    # 2. Env 基础运行参数
    # ==============================
    HZ = 15
    DISPLAY_IMAGES = True
    MAX_EPISODE_LENGTH = 250

    # ==============================
    # 3. 图像 / 显示配置
    # 三相机全部保留
    # ==============================
    #ENV_IMAGE_KEYS 是“环境里有哪些图可用”。决定demos和classifer_data录制的相机个数
    ENV_IMAGE_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]

    #ENV_IMAGE_KEYS = ["head_rgb", "right_wrist_rgb"]
    #DISPLAY_IMAGE_KEYS 只是显示顺序，改变可视化界面的相机个数
    DISPLAY_IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]

    #最终送进网络的分辨率。下面的dim等是先输入的分辨率，
    #可以尝试再往下降低分辨率96*96看看learner是否更新权重更快
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
        "device_index": 6,   # 用 v4l2-ctl --list-devices 查询
        "api": cv2.CAP_V4L2,
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
    POS_SCALE = 0.01   # 0.01 m
    ROT_SCALE = 0.05   # 0.05 rad
    #夹爪不缩放，我和官方都是手臂和夹爪先clip，然后手臂缩放，夹爪直接action【6】映射成标签对应的夹爪动作

    # ==============================
    # 6. 安全工作空间限位
    # ==============================
    # actor / VR / env.step 发出的增量动作：会被限位
    # reset_pose：如果 reset 目标本身超出限位，不一定会被这个限位挡住
    # 你手动 ros2 topic pub：完全不经过 env，也不会被这个限位挡住
    XYZ_LIMIT_LOW = np.array([0.1, -0.4, -0.23], dtype=np.float64)
    XYZ_LIMIT_HIGH = np.array([0.23, -0.05, -0.1], dtype=np.float64)

    RPY_LIMIT_LOW = np.array([-np.pi, -np.pi, -np.pi], dtype=np.float64)
    RPY_LIMIT_HIGH = np.array([np.pi, np.pi, np.pi], dtype=np.float64)

    # ==============================
    # 7. ROS2 发布配置
    # 单臂右臂任务只保留右臂发布
    # ==============================
    robot_config = {
        "hardware": "R1_PRO",
        "enable_publish": [
            "right_ee_pose",
            "right_gripper",
        ],
    }

    def __getitem__(self, key):
        if key == "robot":
            return self.robot_config
        raise KeyError(key)


class GalaxeaUSBTrainConfig(DefaultTrainingConfig):
    """USB 单臂任务训练配置与环境装配入口。"""

    # ==============================
    # 8. 训练超参数
    # ==============================
    agent: str = "sac"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.98

    max_steps: int = 1_000_000

    # Learner 侧经验回放池最大容量
    replay_buffer_capacity: int = 200_000

    # actor 前期不走纯随机，直接从策略开始
    random_steps: int = 0

    # learner 至少等到这么多 online 数据后再开始更新
    training_starts: int = 100

    # learner 每 50 step 向 actor 发布一次网络
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000
    checkpoint_period: int = 2000
    buffer_period: int = 1000

    # ==============================
    # 9. 观测 / 编码配置
    # ==============================
    #RL 训练主输入看 image_keys，但不决定demos录制用这几个录制
    image_keys: List[str] = ["head_rgb", "right_wrist_rgb"]

    # 单臂右臂任务更建议分类器关注 head + right wrist
    # 奖励分类器看 classifier_keys，奖励分类只看如下图像，但是不决定录制数据用这几个图像采集
    # classifier_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    classifier_keys: List[str] = ["left_wrist_rgb"]
    # classifier_keys: List[str] = ["left_wrist_rgb"]
    proprio_keys: List[str] = [
        "right_ee_pose",
        "right_gripper",
    ]

    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "single-arm-learned-gripper"

    # ==============================
    # 10. 环境装配
    # ==============================
    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
        use_vr: bool = True,
    ):
        if fake_env:
            return None

        from examples.galaxea_task.usb_pick_insertion_single.wrapper import (
            GalaxeaUSBEnv,
            SingleGripperPenaltyWrapper,
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
                    checkpoint_path=os.path.abspath("classifier_ckpt_single/"),
                )

                def reward_func(obs):
                    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                    prob = sigmoid(classifier_fn(obs))
                    prob = np.asarray(jax.device_get(prob)).reshape(-1)[0]
                    return int(prob > 0.7)

                env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

            except Exception as e:
                print(f"⚠️ 分类器加载失败（如果你还没训练它，这属于正常现象）: {e}")

        # 5) 单臂 gripper 惩罚
        # 如果夹爪频繁开合/抽搐，给予惩罚 0.02 分
        env = SingleGripperPenaltyWrapper(env, penalty=-0.02)

        return env

    # ==============================
    # 11. demo 清洗
    # ==============================
    def process_demos(self, transitions):
        processed = []
        for trans in transitions:
            if np.linalg.norm(np.asarray(trans["actions"])) > 1e-4:
                processed.append(trans)
        return processed


# 导出实例，供外部直接 import 使用
env_config = GalaxeaUSBTrainConfig()



