#只负责实现一个功能：
# （1）符合强化学习的坐标转换

# 这个文件在强化学习中极其重要。SERL 算法在训练时，为了泛化能力，通常不使用桌面的绝对坐标，而是使用相对于初始位姿的相对坐标。


# 这个文件里的代码虽然全都是矩阵乘法，但它其实是伯克利 SERL 算法能够高效、稳定训练的核心秘密武器。
# 我先为你总结它的核心作用，然后再在代码中逐行加上详尽的“官方意图”注释。
# 💡 核心作用总结：为什么我们需要“相对坐标”？
# 在强化学习中，如果你让神经网络输出绝对坐标（比如：把手移动到空间坐标 X: 0.5, Y: 0.2, Z: 0.3），模型会学得非常痛苦。
# 因为它必须死记硬背整个物理空间的长宽高等几何特征，一旦桌子稍微挪了一点位置，模型就彻底傻眼了。
# relative_env.py 的使命就是**“把世界变成以机器人自己为中心的”**：
  # 动作（Action）相对化：把策略网络输出的动作，解释为“相对于当前夹爪位置的偏移量”。
    # 比如模型输出 [0.01, 0, 0...]，不管机器人的手现在在哪，它的意思永远是“朝当前手心的前方移动 1 厘米”。

  # 观测（Observation）相对化：把原本基于机械臂底座（Base）的绝对坐标，转换成**“相对于初始复位姿态（Reset Pose）”的相对位移**。
    # 每次回合（Episode）开始时，机器人的相对位置永远是完美的 [0, 0, 0]。

import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env

# 从工具包导入齐次矩阵和变换矩阵的构造函数
from envs.utils.transformations import (
    construct_transform_matrix,
    construct_homogeneous_matrix,
)

class DualRelativeFrame(gym.Wrapper):
    """
    环境包装器：将“世界坐标系（绝对）”转换为“自我中心坐标系（相对）”。
    目的：让大模型更容易学习。模型不需要知道自己在房间的哪个绝对坐标，
    只需要知道“我的手距离本回合的起点有多远”。
    """
    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        
        # transform_matrix 是一个 6x6 的矩阵。
        # 它的作用：把大模型输出的“相对于当前手心的微小移动指令（局部）”（相对位置，相对于上一次，相对点时刻在变，方便训练），
        # 旋转映射成“底盘坐标系下的移动指令（全局）”（绝对位置，固定相对于基座，相对点不变，方便控制），这样真实的机械臂才能执行。
        self.left_transform_matrix = np.zeros((6, 6))
        self.right_transform_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # T_r_o_inv 存放的是 4x4 的逆齐次变换矩阵。
            # 这里的 "r" 代表 Reset (起点)，"o" 代表 Origin (基座绝对原点)。
            # 它的作用：死死记住每个回合开始时，那一个瞬间的绝对位姿的“逆向关系”。
            # 之后所有的绝对坐标乘以它，就能得出相对于起点的偏移量。
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # 1. 动作拦截：大模型发来的是“相对当前手的动作”，我们需要把它转成“基座绝对动作”发给底层。
        transformed_action = self.transform_action(action)
        
        # 2. 环境执行：让真实的星海图机械臂动起来，并拿到最新的绝对观测数据。
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # 3. 专家数据拦截：如果你在用手柄遥控，手柄给出的是绝对动作，
        # 我们要把它逆向转换为“相对动作”并存入 info，这样大模型模仿你的时候学到的才是对的。
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # 4. 更新基准：机械臂动完了，它的手心朝向变了。必须用最新的位姿，
        # 重新计算 6x6 的动作转换矩阵，为大模型的下一次指挥做准备。
        self.left_transform_matrix = construct_transform_matrix(obs["state"]["left_ee_pose"])
        self.right_transform_matrix = construct_transform_matrix(obs["state"]["right_ee_pose"])

        # 5. 观测拦截：把底层传上来的绝对位姿，转换为“距离起点的相对位姿”喂给大模型。
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        # 1. 回合重置：机械臂回到初始准备姿态
        obs, info = self.env.reset(**kwargs)

        # 2. 获取这个初始姿态的 6x6 动作转换矩阵
        self.left_transform_matrix = construct_transform_matrix(obs["state"]["left_ee_pose"])
        self.right_transform_matrix = construct_transform_matrix(obs["state"]["right_ee_pose"])

        if self.include_relative_pose:
            # 3. 【核心标定】：提取此时的绝对位姿，构造成 4x4 齐次矩阵，并求逆 (inv)！
            # 这就相当于在空间中打下了一个无形的木桩，定义这里就是本回合的 (0,0,0) 原点。
            self.left_T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(obs["state"]["left_ee_pose"]))
            self.right_T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(obs["state"]["right_ee_pose"]))
            
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        观测转换：绝对坐标 -> 相对坐标
        公式逻辑： T_相对 = (T_起点到基座的逆) 矩阵乘 (T_当前到基座)
        """
        if self.include_relative_pose:
            # ========== 左臂转换 ==========
            # 提取当前时刻的绝对齐次矩阵 (T_b_o: Base to Object)
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left_ee_pose"])
            # 计算相对齐次矩阵 (左乘逆矩阵，完成基系转换)
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o
            
            # 从算好的相对矩阵中，把前 3 行第 4 列提取出来，这就是相对位置 (x, y, z)
            left_p_b_r = left_T_b_r[:3, 3]
            # 从相对矩阵的左上角 3x3 旋转矩阵中，反解出四元数 (qx, qy, qz, qw)
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            # 拼合在一起，覆盖掉原来的绝对数据
            obs["state"]["left_ee_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            # ========== 右臂转换 (逻辑同上) ==========
            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right_ee_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o
            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right_ee_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))
            
        return obs

    def transform_action(self, action: np.ndarray):
        """
        动作转换：大模型的相对微调 -> 底层 ROS 的绝对微调
        利用 6x6 矩阵，将局部的平移和旋转，映射到基座坐标系下。
        """
        action = np.array(action)
        if len(action) == 14:
            # 前 6 维是左臂 (xyz + rpy)，乘以左臂的转换矩阵
            action[:6] = self.left_transform_matrix @ action[:6]
            # 7:13 是右臂 (xyz + rpy)，乘以右臂的转换矩阵
            # 注：索引 6 和 13 是夹爪，直接透传，不需要矩阵转换
            action[7:13] = self.right_transform_matrix @ action[7:13]
        else:
            raise ValueError("Galaxea action space must be 14!")
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        动作逆转换：人类专家的绝对微调 -> 相对微调
        使用 6x6 矩阵的逆矩阵 (np.linalg.inv) 进行反向映射。
        """
        action = np.array(action)
        if len(action) == 14:
            action[:6] = np.linalg.inv(self.left_transform_matrix) @ action[:6]
            action[7:13] = np.linalg.inv(self.right_transform_matrix) @ action[7:13]
        else:
            raise ValueError("Galaxea action space must be 14!")
        return action



##########官方代码

#为什么代码变短了？少了什么？
# 原作者的 DualRelativeFrame 之所以长，是因为它包含了两个我们目前不需要的“冗余功能”：

# 1. 删减了：速度的相对坐标转换 (tcp_vel)

# 原版情况：原版 Franka 环境的观测字典里，除了位姿 (tcp_pose)，还有末端速度 (tcp_vel)。因此原版代码在 transform_observation 里有一段专门把全局速度转换成局部速度的逻辑。

# 我们的情况：我们之前写的 dual_galaxea_env.py 的观测空间里，只抓取了位姿 (ee_pose) 和夹爪 (gripper)，并没有抓取机器人的实时速度。既然字典里没有 tcp_vel，如果保留那段代码就会报错（KeyError）。

# 需要你提供信息吗？：不需要，除非你想加。 强化学习算法（如 SERL）通常靠“帧堆叠 (Frame Stacking)”或者直接看相邻两帧的位姿差就能自己推测出速度。所以没有实时速度反馈，模型一样能训练。但如果你的星海图 Ros2Bridge 确实能提供精确的末端线速度和角速度，你可以告诉我，我们随时可以把这段逻辑加回来。

# 2. 删减了：12 维动作的兼容分支 (len(action) == 12)

# 原版情况：原作者为了让代码既能跑“带夹爪的双臂（14维）”，又能跑“不带夹爪的双臂（12维）”，写了 if-elif 冗余分支。

# 我们的情况：我们的星海图 R1 PRO 是固定的 14 维输出（每条胳膊 6 维位姿 + 1 维夹爪）。直接硬编码 14 维，代码运行效率更高，也更清晰。


# import copy
# from scipy.spatial.transform import Rotation as R
# import gymnasium as gym
# import numpy as np
# from gym import Env

# # 导入将位姿（XYZ + 四元数）转换为 6x6 动作变换矩阵和 4x4 齐次变换矩阵的数学工具
# from franka_env.utils.transformations import (
#     construct_transform_matrix,
#     construct_homogeneous_matrix,
# )

# class DualRelativeFrame(gym.Wrapper):
#     """
#     双臂相对坐标系包装器 (Wrapper)。
#     官方意图：将底层环境（绝对坐标系）完全包裹起来，向上层（RL算法）提供一个纯粹的相对坐标系接口。
#     """

#     def __init__(self, env: Env, include_relative_pose=True):
#         super().__init__(env)
#         # 用来将 局部动作(手部坐标系) 转换为 全局动作(基座坐标系) 的 6x6 矩阵
#         self.left_transform_matrix = np.zeros((6, 6))
#         self.right_transform_matrix = np.zeros((6, 6))

#         self.include_relative_pose = include_relative_pose
#         if self.include_relative_pose:
#             # 核心变量：存储每次 reset 时，初始位姿到基座坐标系的【逆】齐次矩阵
#             # 用大白话说：它记录了“回合起点的坐标系”。以后所有的运动，都要和这个起点做对比。
#             self.left_T_r_o_inv = np.zeros((4, 4))
#             self.right_T_r_o_inv = np.zeros((4, 4))

#     def step(self, action: np.ndarray):
#         # 1. 拦截大模型发来的 Action（此时 Action 还是相对于当前夹爪的局部指令）
#         # 将其转换为底层 ROS 2 能听懂的基座全局移动指令
#         transformed_action = self.transform_action(action)
        
#         # 2. 把转换后的全局指令发给底层的 dual_galaxea_env 执行
#         obs, reward, done, truncated, info = self.env.step(transformed_action)

#         # 3. 处理人类专家的接管数据（SpaceMouse）
#         # 如果人类介入了，要把人类遥控器发出的全局指令，逆向转换回局部指令，
#         # 这样 RL 算法记录的专家数据（Demo）才是符合相对坐标逻辑的。
#         if "intervene_action" in info:
#             info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

#         # 4. 机械臂移动完了，立刻更新当前位姿的变换矩阵，为下一次 Action 转换做准备
#         self.left_transform_matrix = construct_transform_matrix(obs["state"]["left/tcp_pose"])
#         self.right_transform_matrix = construct_transform_matrix(obs["state"]["right/tcp_pose"])

#         # 5. 拦截底层传回来的绝对坐标 Obs，将其转换为相对坐标 Obs 喂给大模型
#         transformed_obs = self.transform_observation(obs)
#         return transformed_obs, reward, done, truncated, info

#     def reset(self, **kwargs):
#         # 1. 重置底层环境，机械臂回到初始位置（Reset Pose）
#         obs, info = self.env.reset(**kwargs)

#         # 2. 拿到初始位置的变换矩阵
#         self.left_transform_matrix = construct_transform_matrix(obs["state"]["left/tcp_pose"])
#         self.right_transform_matrix = construct_transform_matrix(obs["state"]["right/tcp_pose"])

#         if self.include_relative_pose:
#             # 3. 【极度关键】：计算并保存初始位置齐次矩阵的【逆矩阵】 (T_reset -> base)
#             # 这相当于在这里插了一个虚拟的“零点标杆”。本回合之后所有的观测数据，
#             # 都要乘上这个逆矩阵，算出相对于这个“零点标杆”的偏移量。
#             self.left_T_r_o_inv = np.linalg.inv(
#                 construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
#             )
#             self.right_T_r_o_inv = np.linalg.inv(
#                 construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
#             )
            
#         return self.transform_observation(obs), info

#     def transform_observation(self, obs):
#         """
#         官方意图：把全局的基座坐标系数据，映射回我们定义的相对坐标系中。
#         """
#         # 1. 速度转换：把全局速度转换成相对于当前夹爪方向的局部速度
#         left_transform_inv = np.linalg.inv(self.left_transform_matrix)
#         obs["state"]["left/tcp_vel"] = left_transform_inv @ obs["state"]["left/tcp_vel"]

#         right_transform_inv = np.linalg.inv(self.right_transform_matrix)
#         obs["state"]["right/tcp_vel"] = right_transform_inv @ obs["state"]["right/tcp_vel"]

#         if self.include_relative_pose:
#             # 2. 位姿转换（最核心的相对坐标计算）
            
#             # 以左臂为例：获取当前绝对位姿的齐次矩阵 (T_base -> current)
#             left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            
#             # 矩阵乘法：(T_reset -> base) x (T_base -> current) = (T_reset -> current)
#             # 结果 left_T_b_r 就是当前夹爪相对于最初起点（Reset Pose）的精确位移和旋转关系
#             left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

#             # 从齐次矩阵中拆解出位置 (xyz) 和姿态 (四元数) 覆写原数据
#             left_p_b_r = left_T_b_r[:3, 3]
#             left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
#             obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

#             # 右臂同理...
#             right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
#             right_T_b_r = self.right_T_r_o_inv @ right_T_b_o
#             right_p_b_r = right_T_b_r[:3, 3]
#             right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
#             obs["state"]["right/tcp_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))

#         return obs

#     def transform_action(self, action: np.ndarray):
#         """
#         官方意图：神经网络输出的 action 是相对于夹爪当前朝向的局部微调。
#         我们要把它乘以当前位姿的转换矩阵，映射到基座的全局坐标系中，ROS 2 才能执行。
#         """
#         action = np.array(action)
#         if len(action) == 12: # 不带夹爪的 12 维动作
#             action[:6] = self.left_transform_matrix @ action[:6]
#             action[6:] = self.right_transform_matrix @ action[6:]
#         elif len(action) == 14: # 带夹爪的 14 维动作
#             action[:6] = self.left_transform_matrix @ action[:6]
#             action[7:13] = self.right_transform_matrix @ action[7:13]
#         else:
#             raise ValueError("Action dimension not supported")
#         return action

#     def transform_action_inv(self, action: np.ndarray):
#         """
#         官方意图：将人类专家的绝对全局动作，逆向转回局部动作。用于数据录制（Demo collection）。
#         """
#         action = np.array(action)
#         if len(action) == 12:
#             action[:6] = np.linalg.inv(self.left_transform_matrix) @ action[:6]
#             action[6:] = np.linalg.inv(self.right_transform_matrix) @ action[6:]
#         elif len(action) == 14:
#             action[:6] = np.linalg.inv(self.left_transform_matrix) @ action[:6]
#             action[7:13] = np.linalg.inv(self.right_transform_matrix) @ action[7:13]
#         else:
#             raise ValueError("Action dimension not supported")
#         return action