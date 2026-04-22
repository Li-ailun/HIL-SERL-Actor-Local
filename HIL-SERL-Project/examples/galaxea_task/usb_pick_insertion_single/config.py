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

    目标：
    1. 使用统一底层 env（GalaxeaArmEnv / 兼容名 GalaxeaDualArmEnv）
    2. 当前任务切到单臂模式
    3. 控制右臂
    4. 三个相机全部保留
    """

    # ==============================
    # 0. 单 / 双臂模式配置
    # ==============================
    ARM_MODE = "single"
    ARM_SIDE = "right"

    # ==============================
    # 1. 单臂 reset 位姿
    # 说明：
    # 统一 env 在 single 模式下优先读取 RESET_POSE
    # ==============================
    # RESET_POSE = np.array([0.41774907445156195, -0.25200100000000003, 0.012723447389100126, 7.697375045982823e-05,-0.7235532515928197, -7.34328254905007e-05, 0.6902685570067056], dtype=np.float32)
    RESET_POSE = np.array([0.31728635015134854, -0.3061865364134424, -0.13054431501184297, 0.013798754839754589,-0.07602081420421326, -0.006189825615894724, 0.9969915326779089], dtype=np.float32)

    # 为兼容旧逻辑，保留双臂字段也无妨
    #夹爪复位指令硬编码在env脚本的go to reset中，目前硬编码100
    RESET_L = np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    RESET_R = np.array([0.41774907445156195, -0.25200100000000003, 0.012723447389100126, 7.697375045982823e-05,-0.7235532515928197, -7.34328254905007e-05, 0.6902685570067056], dtype=np.float32)

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01

    # ==============================
    # 2. Env 基础运行参数
    # ==============================
    HZ = 15
    DISPLAY_IMAGES = True
    MAX_EPISODE_LENGTH = 1500

    # ==============================
    # 3. 图像 / 显示配置
    # 三相机全部保留
    # ==============================
    ENV_IMAGE_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    DISPLAY_IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]
    # ENV_IMAGE_KEYS = ["head_rgb", "left_wrist_rgb"]
    # DISPLAY_IMAGE_KEYS = ["left_wrist_rgb", "head_rgb"]
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
        "device_index": 2,   #v4l2-ctl --list-devices查询
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
    POS_SCALE = 0.01
    ROT_SCALE = 0.05

    # ==============================
    # 6. 安全工作空间限位
    # ==============================
    XYZ_LIMIT_LOW = np.array([0.1, -0.5, -0.5], dtype=np.float64)
    XYZ_LIMIT_HIGH = np.array([0.5, 0.5, 0.5], dtype=np.float64)

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
    
    #Learner 侧经验回放池的最大容量。
    #训练全称最大容量，保证不会太快丢失老经验，也可以让新数据保留更多
    replay_buffer_capacity: int = 200_000

    #这个参数只影响 Actor 采样动作的前期策略。
    #从 actor 的第 0 步开始，就不走纯随机动作了（官网提供了bc训练脚本，我们可以bc训练后，修改rlpd脚本先加载bc权重，官方没这个功能）。
    #一开始就让 actor 按 demo 初始化出来的策略分布去探索，而不是做完全无结构的随机探索。
    random_steps: int = 0

    #Learner 什么时候真正开始训练更新。
    #使你已经有很多 demos，learner 也不会立刻开始训练；
    # 它仍然会先等 actor 送来至少 100 条在线 transition。
    training_starts: int = 100

    #Learner 每完成 50 个 learner update step，就把最新网络参数发给 actor 一次。
    #策略更新传播到 actor 的频率
    #如果actor断了，则还是能继续做梯度更新；只是新的参数没人接收，并且不会再有新的 online 数据进来。
    # actor断了后，learner一直在训练之前的旧数据集
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000
    checkpoint_period: int = 2000
    buffer_period: int = 1000

    # ==============================
    # 9. 观测 / 编码配置
    # 三相机保留；本体状态只保留右臂
    # ==============================
    image_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]

    # 单臂右臂任务更建议分类器关注 head + right wrist
    classifier_keys: List[str] = ["head_rgb", "right_wrist_rgb"]

    proprio_keys: List[str] = [
        "right_ee_pose",
        "right_gripper",
    ]

    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "single-arm-learned-gripper"

    # ==============================
    # 10. 环境装配
    # 说明：
    # 1. 这里默认你后面会让 wrapper_single 使用统一底层 env
    # 2. 暂时只改 config，不替你改 wrapper 文件本体
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

        # 这里改成 single 目录下对应的 wrapper
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
                    ##若需要奖励，在如下目录下搜索奖励分类器ckpt
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
        #如果夹爪频繁开合/抽搐，给予惩罚0.02分
        env = SingleGripperPenaltyWrapper(env, penalty=-0.02)

        return env

    # ==============================
    # 11. demo 清洗
    # 单臂动作从 14 维变成 7 维，但这里逻辑不需要改
    # ==============================
    def process_demos(self, transitions):
        processed = []
        for trans in transitions:
            if np.linalg.norm(np.asarray(trans["actions"])) > 1e-4:
                processed.append(trans)
        return processed


# 导出实例，供外部直接 import 使用
env_config = GalaxeaUSBTrainConfig()