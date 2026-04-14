#这个文件主要负责：
# （1）奖励判定、
# （2）四元数转欧拉角，
# （3）以及最核心的 SpaceMouse（3D 鼠标 / VR 手柄）介入接管逻辑。

# 同样，我已经为你处理好了字典键的适配。

#这个代码定义了很多主功能(很多属于gym.Wrapper类的功能，后续按需调用)，
# 其他细小特定任务定制的功能写到examples/galaxea_task/usb_pick_insertion/wrapper.py这种wrapper.py下面对
# 然后wrappers.py(底层通用库)+wrapper.py(特定任务层)的功能加到一起就是完整的vr介入功能


import time
from typing import List

from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
import copy
import requests

# ROS 2 相关库 (用于星海图 VR 接管和硬件通信)
import rclpy
from rclpy.node import Node
from teleoperation_msg_ros2.srv import SwitchControlModeVR
# TODO: 实机部署前，务必取消注释并根据实际包名导入星海图的 VRPose 消息类型
from teleoperation_msg_ros2.msg import VrPose

# ============================================================================
# 第一部分：通用功能与辅助 Wrappers (官方提供)
# 作用：通过强化学习的 "Wrapper" 机制，对底层基础环境进行层层包装，
#       实现状态净化、打分自动化和异常动作惩罚，而无需修改基础环境的源码。
# ============================================================================

# 辅助数学函数：用于将神经网络的输出 Logit 映射到 (0, 1) 区间，方便做概率判断
sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    """
    [开发/调试专用] 手动打分器 Wrapper
    
    工作原理：
    当环境报告一个回合 (Episode) 结束时 (done=True)，此包装器会阻塞主线程，
    并在终端弹出输入框，要求操作员手动输入 1 (成功) 或 0 (失败)。
    
    使用场景：
    在视觉奖励模型 (Reward Classifier) 尚未训练好，或者需要排查视觉模型 Bug 时，
    使用此包装器进行半自动化的强化学习训练。
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # 1. 先让底层环境执行动作
        obs, rew, done, truncated, info = self.env.step(action)
        
        # 2. 如果回合结束，强制人工打分
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0): "))
                    # 确保输入合法
                    assert rew == 0 or rew == 1
                    break
                except:
                    print("Invalid input. Please enter 1 for success, 0 for failure.")
                    continue
                    
        # 3. 将打分结果注入 info 字典，供外层日志记录或算法读取
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    [正式训练主力] 视觉二分类自动打分器 Wrapper
    
    工作原理：
    每一帧动作执行后，提取当前画面的 Observation，丢给预先训练好的 ResNet (或类似视觉模型) 打分。
    还可以通过 target_hz 参数，强制控制环境的运行频率（例如让真实物理世界保持 20Hz 运行）。
    """
    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        # 传入的通常是一个加载了权重的神经网络推断函数
        self.reward_classifier_func = reward_classifier_func
        # 期望的运行频率（赫兹）
        self.target_hz = target_hz

    def compute_reward(self, obs):
        """调用神经网络计算奖励分数"""
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        
        # 1. 底层环境执行动作
        obs, rew, done, truncated, info = self.env.step(action)
        
        # 2. 视觉模型基于最新状态打分
        rew = self.compute_reward(obs)
        
        # 3. 核心逻辑：如果视觉模型认为任务已经成功 (rew==1)，则强制结束当前回合
        done = done or rew
        info['succeed'] = bool(rew)
        
        # 4. 频率控制锁：如果执行太快，强制 sleep 补足时间差，确保控制频率恒定
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    [复杂任务专用] 多阶段视觉打分器 Wrapper
    
    工作原理：
    如果一个任务包含多个子阶段（例如：1. 抓起插头 -> 2. 对准插座 -> 3. 插入），
    系统会传入多个分类器。只有当前一个阶段达成后，才开始计算下一个阶段。
    """
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        # 记录每个子任务是否已经完成的状态数组
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        """遍历所有分类器，累加奖励"""
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            # 如果该阶段已经完成过，不再重复给分
            if self.received[i]:
                continue
                
            # 获取模型输出的 Logit 值
            logit = classifier_func(obs).item()
            
            # 阈值判定：置信度大于 75% 才算成功
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1
                
        # 返回这一帧获得的新增奖励总和
        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        # 当所有子任务都达成时，整个大任务才算完成
        done = (done or all(self.received)) 
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 重置所有子任务的完成状态
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


# ----------------------------------------------------------------------------
# 状态转换类 Wrapper (ObservationWrapper)
# 作用：拦截环境的输出数据，修改后再交给外层的神经网络，降低模型的学习难度。
# ----------------------------------------------------------------------------

class Quat2EulerWrapper(gym.ObservationWrapper):
    """单臂环境：四元数转欧拉角"""
    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # 修改向外暴露的状态空间定义，从 7维 (xyz+quat) 改为 6维 (xyz+euler)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """单臂环境：四元数转旋转矩阵 (某些算法对旋转矩阵的连续性更敏感)"""
    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        # 展平矩阵并拼接
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    【双臂环境核心】双臂四元数转欧拉角 Wrapper
    
    重要性：神经网络很难直接预测和拟合四元数（因为四元数有模长为1的约束）。
    必须用此 Wrapper 将状态转换为欧拉角，并且必须确保 Action 空间也使用欧拉角，
    做到输入输出格式一致，模型才能有效收敛。
    """
    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # 转换左臂
        tcp_pose_l = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose_l[:3], R.from_quat(tcp_pose_l[3:]).as_euler("xyz"))
        )
        # 转换右臂
        tcp_pose_r = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose_r[:3], R.from_quat(tcp_pose_r[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        # 重置环境时，同样需要对初始观测进行四元数转换
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ----------------------------------------------------------------------------
# 动作修改类 Wrapper (ActionWrapper / RewardWrapper)
# 作用：拦截外层传来的动作指令，或者根据不良动作施加惩罚。
# ----------------------------------------------------------------------------

class GripperCloseEnv(gym.ActionWrapper):
    """
    强制闭合夹爪 Wrapper
    如果你的任务（如：用手背推开障碍物）根本不需要用到夹爪开合，
    加上这个 Wrapper 可以强行把夹爪指令归零，减小模型的探索空间。
    """
    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        # 如果存在干预记录，也要把多余的夹爪维度切掉
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    """单臂：夹爪状态翻转惩罚器"""
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        # 如果模型发出错误指令导致夹爪抽搐，扣除惩罚分
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info


class DualGripperPenaltyWrapper(gym.RewardWrapper):
    """
    【必须使用】双臂夹爪动作惩罚 Wrapper
    
    工作原理：
    强化学习模型在早期探索时，很容易输出高频交替的夹爪张合指令（抽搐）。
    这不仅会破坏抓取任务，还极易导致物理电机关机甚至烧毁。
    此 Wrapper 通过比对前后帧状态，如果发现模型在做无效的开关动作，
    就从总奖励中扣除 penalty 分数，强迫模型学会“保持夹爪静止”。
    """
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        # 确认是在处理 14 维的双臂动作
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        # 状态位：0 代表张开，1 代表闭合
        self.last_gripper_pos_left = 0 
        self.last_gripper_pos_right = 0 
    
    def reward(self, reward: float, action) -> float:
        # 检查左夹爪 (索引 6) 是否发生状态翻转
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
            
        # 检查右夹爪 (索引 13) 是否发生状态翻转
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
            
        return reward
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # 如果这一帧人类专家进行了接管，以人类的实际动作为准计算惩罚
        if "intervene_action" in info:
            action = info["intervene_action"]
        
        # 将扣分后的奖励返回给外层
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info


# ============================================================================
# 第二部分：星海图 R1 PRO 专属定制 (物理离合接管 Wrapper)
# ============================================================================

class VRInterventionWrapper(gym.ActionWrapper):
    """
    星海图 R1 PRO VR 物理离合接管 Wrapper
    需求语义：
    - control_mode == 0 -> VR 接管 -> use_vr_mode = True
    - control_mode == 2 -> 脚本/AI 接管 -> use_vr_mode = False
    - 只在模式发生变化时发送一次 service
    """

    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        # 当前是否由 VR 接管
        self.use_vr_mode = False

        # 上一次看到的 control_mode，用于“边沿触发”
        self.last_control_mode = None

        # 上一次已经发送过的 service 目标，防止重复发
        self.last_service_target = None

        self.action_indices = action_indices
        self.gripper_enabled = gripper_enabled

        # 14 维动作缓存
        self.latest_vr_action = np.zeros(14, dtype=np.float32)

        # 复用底层 bridge 的 node
        try:
            self.node = self.env.unwrapped.bridge.node
        except AttributeError:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node("vr_intervention_wrapper_node")

        # 持久化 service client
        self.mode_client = self.node.create_client(
            SwitchControlModeVR,
            "/switch_control_mode_vr"
        )

        print("⏳ 正在等待底层硬件服务连接 (/switch_control_mode_vr)...")
        if not self.mode_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("❌ /switch_control_mode_vr 服务未上线，无法建立 VR 模式切换通道")
        print("✅ 服务已连接！持久化通讯通道已建立，VR 物理离合器准备就绪。")

        # 订阅 VR 话题
        self.vr_sub = self.node.create_subscription(
            VrPose,
            "/vr_pose",
            self._vr_cb,
            10
        )

    def _on_switch_done(self, future):
        try:
            resp = future.result()
            if resp is not None:
                print(f"✅ 模式切换返回: success={resp.success}, message={resp.message}")
        except Exception as e:
            print(f"❌ 模式切换 service 调用失败: {e}")

    def _call_switch_service(self, use_vr_mode: bool):
        """
        异步调用模式切换服务
        use_vr_mode=True  -> 切到 VR 控制
        use_vr_mode=False -> 切到脚本/IK 控制
        """
        # 防止重复发同一个目标状态
        if self.last_service_target == use_vr_mode:
            return

        req = SwitchControlModeVR.Request()
        req.use_vr_mode = use_vr_mode

        future = self.mode_client.call_async(req)
        future.add_done_callback(self._on_switch_done)

        self.last_service_target = use_vr_mode

    def _vr_cb(self, msg):
        """
        高频回调：
        - 监听 control_mode
        - 只在 0/2 发生变化时发一次 service
        - 若当前为 VR 接管，则同步更新 latest_vr_action
        """
        current_mode = int(msg.control_mode)

        # 只关心 0 和 2
        if current_mode not in (0, 2):
            return

        # 当前系统状态始终和 mode 对齐
        # mode 0 -> VR 接管
        # mode 2 -> 脚本/AI 接管
        self.use_vr_mode = (current_mode == 0)

        # mode 没变，不重复发 service
        if current_mode == self.last_control_mode:
            pass
        else:
            self.last_control_mode = current_mode

            if current_mode == 0:
               self._call_switch_service(True)
               print("🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True")
               if hasattr(self.env.unwrapped, "notify_script_control"):
                   self.env.unwrapped.notify_script_control(False)

            elif current_mode == 2:
                self._call_switch_service(False)
                print("🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False")
                if hasattr(self.env.unwrapped, "notify_script_control"):
                    self.env.unwrapped.notify_script_control(True)
        # 只有 VR 接管时，才解析 VR 位姿为动作
        if self.use_vr_mode:
            left_pos = np.array(msg.left_position, dtype=np.float32)
            right_pos = np.array(msg.right_position, dtype=np.float32)

            left_euler = R.from_quat(msg.left_quaternion).as_euler("xyz")
            right_euler = R.from_quat(msg.right_quaternion).as_euler("xyz")

            if self.gripper_enabled:
                if msg.left_gripper_close:
                    left_grip = np.random.uniform(-1.0, -0.9, size=(1,)).astype(np.float32)
                else:
                    left_grip = np.random.uniform(0.9, 1.0, size=(1,)).astype(np.float32)

                if msg.right_gripper_close:
                    right_grip = np.random.uniform(-1.0, -0.9, size=(1,)).astype(np.float32)
                else:
                    right_grip = np.random.uniform(0.9, 1.0, size=(1,)).astype(np.float32)
            else:
                left_grip = np.zeros(1, dtype=np.float32)
                right_grip = np.zeros(1, dtype=np.float32)

            raw_vr_action = np.concatenate([
                left_pos, left_euler, left_grip,
                right_pos, right_euler, right_grip
            ])

            if self.action_indices is not None:
                filtered_action = np.zeros_like(raw_vr_action)
                filtered_action[self.action_indices] = raw_vr_action[self.action_indices]
                self.latest_vr_action = filtered_action
            else:
                self.latest_vr_action = raw_vr_action

    def action(self, action: np.ndarray):
        """
        根据当前接管状态决定到底执行谁的动作
        """
        # 注意：这里不要再 rclpy.spin_once(self.node)
        # 因为 bridge 已经在后台 executor 线程里 spin 了

        if self.use_vr_mode:
            return self.latest_vr_action, True
        else:
            return action, False

    def step(self, action):
        new_action, is_intervened = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        if is_intervened:
            info["intervene_action"] = new_action

        info["left"] = is_intervened
        info["right"] = is_intervened
        info["left1"] = is_intervened
        info["left2"] = False
        info["right1"] = is_intervened
        info["right2"] = False

        return obs, rew, done, truncated, info


# import time
# from gymnasium import Env, spaces
# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box
# import copy
# from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
# import requests
# from scipy.spatial.transform import Rotation as R
# from franka_env.envs.franka_env import FrankaEnv
# from typing import List

# sigmoid = lambda x: 1 / (1 + np.exp(-x))

# class HumanClassifierWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)
#         if done:
#             while True:
#                 try:
#                     rew = int(input("Success? (1/0)"))
#                     assert rew == 0 or rew == 1
#                     break
#                 except:
#                     continue
#         info['succeed'] = rew
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info
    
# class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     This wrapper uses the camera images to compute the reward,
#     which is not part of the observation space
#     """

#     def __init__(self, env: Env, reward_classifier_func, target_hz = None):
#         super().__init__(env)
#         self.reward_classifier_func = reward_classifier_func
#         self.target_hz = target_hz

#     def compute_reward(self, obs):
#         if self.reward_classifier_func is not None:
#             return self.reward_classifier_func(obs)
#         return 0

#     def step(self, action):
#         start_time = time.time()
#         obs, rew, done, truncated, info = self.env.step(action)
#         rew = self.compute_reward(obs)
#         done = done or rew
#         info['succeed'] = bool(rew)
#         if self.target_hz is not None:
#             time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         info['succeed'] = False
#         return obs, info
    
    
# class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
#     def __init__(self, env: Env, reward_classifier_func: List[callable]):
#         super().__init__(env)
#         self.reward_classifier_func = reward_classifier_func
#         self.received = [False] * len(reward_classifier_func)
    
#     def compute_reward(self, obs):
#         rewards = [0] * len(self.reward_classifier_func)
#         for i, classifier_func in enumerate(self.reward_classifier_func):
#             if self.received[i]:
#                 continue

#             logit = classifier_func(obs).item()
#             if sigmoid(logit) >= 0.75:
#                 self.received[i] = True
#                 rewards[i] = 1

#         reward = sum(rewards)
#         return reward

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)
#         rew = self.compute_reward(obs)
#         done = (done or all(self.received)) # either environment done or all rewards satisfied
#         info['succeed'] = all(self.received)
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.received = [False] * len(self.reward_classifier_func)
#         info['succeed'] = False
#         return obs, info


# class Quat2EulerWrapper(gym.ObservationWrapper):
#     """
#     Convert the quaternion representation of the tcp pose to euler angles
#     """

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)
#         # from xyz + quat to xyz + euler
#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )

#     def observation(self, observation):
#         # convert tcp pose from quat to euler
#         tcp_pose = observation["state"]["tcp_pose"]
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         return observation


# class Quat2R2Wrapper(gym.ObservationWrapper):
#     """
#     Convert the quaternion representation of the tcp pose to rotation matrix
#     """

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)
#         # from xyz + quat to xyz + euler
#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(9,)
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         r = R.from_quat(tcp_pose[3:]).as_matrix()
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], r[..., :2].flatten())
#         )
#         return observation


# class DualQuat2EulerWrapper(gym.ObservationWrapper):
#     """
#     Convert the quaternion representation of the tcp pose to euler angles
#     """

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
#         assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
#         # from xyz + quat to xyz + euler
#         self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )
#         self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )

#     def observation(self, observation):
#         # convert tcp pose from quat to euler
#         tcp_pose = observation["state"]["left/tcp_pose"]
#         observation["state"]["left/tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         tcp_pose = observation["state"]["right/tcp_pose"]
#         observation["state"]["right/tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         return observation
    
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return self.observation(obs), info

# class GripperCloseEnv(gym.ActionWrapper):
#     """
#     Use this wrapper to task that requires the gripper to be closed
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         ub = self.env.action_space
#         assert ub.shape == (7,)
#         self.action_space = Box(ub.low[:6], ub.high[:6])

#     def action(self, action: np.ndarray) -> np.ndarray:
#         new_action = np.zeros((7,), dtype=np.float32)
#         new_action[:6] = action.copy()
#         return new_action

#     def step(self, action):
#         new_action = self.action(action)
#         obs, rew, done, truncated, info = self.env.step(new_action)
#         if "intervene_action" in info:
#             info["intervene_action"] = info["intervene_action"][:6]
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

    
# class SpacemouseIntervention(gym.ActionWrapper):
#     def __init__(self, env, action_indices=None):
#         super().__init__(env)

#         self.gripper_enabled = True
#         if self.action_space.shape == (6,):
#             self.gripper_enabled = False

#         self.expert = SpaceMouseExpert()
#         self.left, self.right = False, False
#         self.action_indices = action_indices

#     def action(self, action: np.ndarray) -> np.ndarray:
#         """
#         Input:
#         - action: policy action
#         Output:
#         - action: spacemouse action if nonezero; else, policy action
#         """
#         expert_a, buttons = self.expert.get_action()
#         self.left, self.right = tuple(buttons)
#         intervened = False
        
#         if np.linalg.norm(expert_a) > 0.001:
#             intervened = True

#         if self.gripper_enabled:
#             if self.left:  # close gripper
#                 gripper_action = np.random.uniform(-1, -0.9, size=(1,))
#                 intervened = True
#             elif self.right:  # open gripper
#                 gripper_action = np.random.uniform(0.9, 1, size=(1,))
#                 intervened = True
#             else:
#                 gripper_action = np.zeros((1,))
#             expert_a = np.concatenate((expert_a, gripper_action), axis=0)

#         if self.action_indices is not None:
#             filtered_expert_a = np.zeros_like(expert_a)
#             filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
#             expert_a = filtered_expert_a

#         if intervened:
#             return expert_a, True

#         return action, False

#     def step(self, action):

#         new_action, replaced = self.action(action)

#         obs, rew, done, truncated, info = self.env.step(new_action)
#         if replaced:
#             info["intervene_action"] = new_action
#         info["left"] = self.left
#         info["right"] = self.right
#         return obs, rew, done, truncated, info

# class DualSpacemouseIntervention(gym.ActionWrapper):
#     def __init__(self, env, action_indices=None, gripper_enabled=True):
#         super().__init__(env)

#         self.gripper_enabled = gripper_enabled

#         self.expert = SpaceMouseExpert()
#         self.left1, self.left2, self.right1, self.right2 = False, False, False, False
#         self.action_indices = action_indices

#     def action(self, action: np.ndarray) -> np.ndarray:
#         """
#         Input:
#         - action: policy action
#         Output:
#         - action: spacemouse action if nonezero; else, policy action
#         """
#         intervened = False
#         expert_a, buttons = self.expert.get_action()
#         self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


#         if self.gripper_enabled:
#             if self.left1:  # close gripper
#                 left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
#                 intervened = True
#             elif self.left2:  # open gripper
#                 left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
#                 intervened = True
#             else:
#                 left_gripper_action = np.zeros((1,))

#             if self.right1:  # close gripper
#                 right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
#                 intervened = True
#             elif self.right2:  # open gripper
#                 right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
#                 intervened = True
#             else:
#                 right_gripper_action = np.zeros((1,))
#             expert_a = np.concatenate(
#                 (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
#                 axis=0,
#             )

#         if self.action_indices is not None:
#             filtered_expert_a = np.zeros_like(expert_a)
#             filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
#             expert_a = filtered_expert_a

#         if np.linalg.norm(expert_a) > 0.001:
#             intervened = True

#         if intervened:
#             return expert_a, True
#         return action, False

#     def step(self, action):

#         new_action, replaced = self.action(action)

#         obs, rew, done, truncated, info = self.env.step(new_action)
#         if replaced:
#             info["intervene_action"] = new_action
#         info["left1"] = self.left1
#         info["left2"] = self.left2
#         info["right1"] = self.right1
#         info["right2"] = self.right2
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


# class GripperPenaltyWrapper(gym.RewardWrapper):
#     def __init__(self, env, penalty=0.1):
#         super().__init__(env)
#         assert env.action_space.shape == (7,)
#         self.penalty = penalty
#         self.last_gripper_pos = None

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.last_gripper_pos = obs["state"][0, 0]
#         return obs, info

#     def reward(self, reward: float, action) -> float:
#         if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
#             action[6] > 0.5 and self.last_gripper_pos < 0.95
#         ):
#             return reward - self.penalty
#         else:
#             return reward

#     def step(self, action):
#         """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         if "intervene_action" in info:
#             action = info["intervene_action"]
#         reward = self.reward(reward, action)
#         self.last_gripper_pos = observation["state"][0, 0]
#         return observation, reward, terminated, truncated, info

# class DualGripperPenaltyWrapper(gym.RewardWrapper):
#     def __init__(self, env, penalty=0.1):
#         super().__init__(env)
#         assert env.action_space.shape == (14,)
#         self.penalty = penalty
#         self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
#         self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
#     def reward(self, reward: float, action) -> float:
#         if (action[6] < -0.5 and self.last_gripper_pos_left==0):
#             reward -= self.penalty
#             self.last_gripper_pos_left = 1
#         elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
#             reward -= self.penalty
#             self.last_gripper_pos_left = 0
#         if (action[13] < -0.5 and self.last_gripper_pos_right==0):
#             reward -= self.penalty
#             self.last_gripper_pos_right = 1
#         elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
#             reward -= self.penalty
#             self.last_gripper_pos_right = 0
#         return reward
    
#     def step(self, action):
#         """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         if "intervene_action" in info:
#             action = info["intervene_action"]
#         reward = self.reward(reward, action)
#         return observation, reward, terminated, truncated, info