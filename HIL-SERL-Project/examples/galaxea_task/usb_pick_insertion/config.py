import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

# ==============================================================
# ✨ 核心修复：本地定义基类，彻底终结 ImportError
# ==============================================================
class DefaultTrainingConfig:
    def __init__(self):
        pass

class GalaxeaUSBEnvConfig:
    """USB 任务的专属硬件与物理配置"""
    RESET_L = np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0])
    RESET_R = np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0])

    # 任务专属难度参数
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.01

    robot_config = {
        "hardware": "R1_PRO",
        "enable_publish": ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
    }

    def __getitem__(self, key):
        if key == "robot": return self.robot_config
        raise KeyError(key)

class GalaxeaUSBTrainConfig(DefaultTrainingConfig):
    """USB 任务的强化学习大脑配置"""
    agent: str = "sac"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.98

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000
    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000
    checkpoint_period: int = 2000 
    buffer_period: int = 1000

    image_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    classifier_keys: List[str] = ["head_rgb", "left_wrist_rgb"]
    proprio_keys: List[str] = ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
    
    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "dual-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # ⚠️ 注意：这里为了 record_success_fail.py 能跑通，必须返回真实环境
        # 在正式训练 RL 时，这里会根据 fake_env 参数判断
        
        # 延迟导入，防止循环依赖
        from examples.galaxea_task.usb_pick_insertion.wrapper import GalaxeaUSBEnv, DualGripperPenaltyWrapper
        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
        from serl_launcher.wrappers.chunking import ChunkingWrapper
        from serl_launcher.networks.reward_classifier import load_classifier_func
        from serl_robot_infra.Galaxea_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper

        # 1. 实例化核心环境
        env_config = GalaxeaUSBEnvConfig()
        env = GalaxeaUSBEnv(
            config=env_config,  
            cfg={},
            display_images=True,
            save_video=save_video,
            hz=15
        )

        # 2. 包装器层 (Wrappers)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 3. 视觉奖励分类器加载 (仅在 RL 训练或验证时开启)
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
                print(f"⚠️ 分类器加载失败 (正常现象，如果你还没训练它): {e}")

        # 4. 夹爪乱动惩罚
        #env = DualGripperPenaltyWrapper(env, penalty=-0.02)

        return env

# 实例化导出
env_config = GalaxeaUSBTrainConfig()