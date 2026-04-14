

"""
Galaxea Dual Arm Total Configuration for HIL-SERL
Author: Eren (Li-ailun)
Description: 替代官方 DefaultTrainingConfig，作为星海图项目的唯一全局基准配置。
"""

import numpy as np
from typing import List

# 直接导入你写好的环境类
from serl_robot_infra.Galaxea_env.envs.dual_galaxea_env import GalaxeaDualArmEnv

class GalaxeaTrainingConfig:
    """星海图 R1 PRO 的全局总配置文件"""

    # ==========================================
    # 1. 算法核心参数 (RL Hyperparameters)
    # ==========================================
    agent: str = "sac"              # 采用 SAC 算法
    max_traj_length: int = 100      # 与环境中的 max_episode_length 保持一致
    batch_size: int = 256           # 训练批大小
    cta_ratio: int = 2              # Critic 与 Actor 的更新比例
    discount: float = 0.97          # 奖励折扣因子

    # ==========================================
    # 2. 训练步数与日志规划 (Training & Logging Schedule)
    # ==========================================
    max_steps: int = 1000000             # 总训练步数
    replay_buffer_capacity: int = 200000 # 经验池容量
    random_steps: int = 0                # 开始前随机采样的步数
    training_starts: int = 100           # 经验池累积到多少步开始训练
    steps_per_update: int = 50           # 每隔多少步同步一次网络参数
    
    # 🌟 以下是底层脚本必备的周期参数（已补全）
    log_period: int = 10                 # 每 10 步打印一次 wandb 日志
    eval_period: int = 2000              # 每 2000 步进行一次评估
    checkpoint_period: int = 5000        # 每 5000 步保存一次模型权重 (重要)
    buffer_period: int = 1000            # 每 1000 步保存一次缓冲区数据
    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5
    demo_path: str = None                # 留给命令行参数传入

    # ==========================================
    # 3. 视觉与状态配置 (Vision & Setup)
    # ==========================================
    image_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    classifier_keys: List[str] = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    
    # 🌟 必须补充的本体状态键（告诉网络要提取哪些特征）
    proprio_keys: List[str] = ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
    
    encoder_type: str = "resnet-pretrained" 
    setup_mode: str = "dual-arm-learned-gripper" 

    # ==========================================
    # 4. 硬件通讯与物理复位点 (Hardware & Reset)
    # ==========================================
    RESET_L = [0.4, -0.2, 0.2, 0.0, 1.0, 0.0, 0.0]
    RESET_R = [0.4,  0.2, 0.2, 0.0, 1.0, 0.0, 0.0]

    robot_config = {
        "hardware": "R1_PRO",
        "enable_publish": ["left_ee_pose", "right_ee_pose", "left_gripper", "right_gripper"]
    }

    # ==========================================
    # 5. 核心方法实现 (Core Methods)
    # ==========================================
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        实例化并返回实机环境。会被 record_demos.py 和 train_rlpd.py 调用。
        """
        if fake_env:
            return None 

        return GalaxeaDualArmEnv(
            config=self,  
            cfg={},
            display_images=True,
            save_video=save_video,
            hz=15
        )

    def __getitem__(self, key):
        """兼容底层字典调用的魔法方法"""
        if key == "robot":
            return self.robot_config
        raise KeyError(f"Key {key} not found in GalaxeaTrainingConfig")

    def process_demos(self, transitions):
        """数据清洗逻辑"""
        processed_transitions = []
        for trans in transitions:
            if np.linalg.norm(trans['actions']) > 1e-4:
                processed_transitions.append(trans)
        return processed_transitions