# A. actor 自主控制时：
#    网络输出归一化增量 action
#    env 把 action 乘缩放，加到当前末端位姿上执行

# B. VR 介入记录时：
#    VR 底层真实控制机器人
#    wrapper 用末端前后位姿差反推归一化 action
#    保存到 info["intervene_action"]

#后续 inspect 要看有没有大量 ±1。少量饱和可以接受，大量饱和说明 actor step 太慢、VR 速度太快，或者 scale 太小。






import time
from typing import List

from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

# ROS 2 相关库
import rclpy
from teleoperation_msg_ros2.srv import SwitchControlModeVR
from teleoperation_msg_ros2.msg import VrPose


# ============================================================================
# 通用功能与辅助 Wrappers
# ============================================================================

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class HumanClassifierWrapper(gym.Wrapper):
    """
    手动打分器 Wrapper。
    当 done=True 时，在终端要求人工输入成功/失败。
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0): "))
                    assert rew == 0 or rew == 1
                    break
                except Exception:
                    print("Invalid input. Please enter 1 for success, 0 for failure.")

        info["succeed"] = rew
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    视觉二分类自动打分器 Wrapper。
    每一步 env.step 后，用 reward classifier 判断当前 obs 是否成功。
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()

        obs, rew, done, truncated, info = self.env.step(action)

        rew = int(self.compute_reward(obs))
        success = bool(rew)
        done = bool(done or success)

        info["succeed"] = success
        info["success"] = success
        info["is_success"] = success

        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        info["success"] = False
        info["is_success"] = False
        return obs, info


class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    多阶段视觉打分器 Wrapper。
    """

    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)

    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)

        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        return sum(rewards)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        rew = self.compute_reward(obs)
        done = bool(done or all(self.received))
        info["succeed"] = all(self.received)

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info["succeed"] = False
        return obs, info


# ============================================================================
# Observation Wrappers
# ============================================================================

class Quat2EulerWrapper(gym.ObservationWrapper):
    """单臂环境：四元数转欧拉角。"""

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)

        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(6,),
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """单臂环境：四元数转旋转矩阵前两列。"""

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)

        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(9,),
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """双臂环境：左右臂四元数转欧拉角。"""

    def __init__(self, env: Env):
        super().__init__(env)

        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)

        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(6,),
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(6,),
        )

    def observation(self, observation):
        tcp_pose_l = observation["state"]["left/tcp_pose"]
        tcp_pose_r = observation["state"]["right/tcp_pose"]

        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose_l[:3], R.from_quat(tcp_pose_l[3:]).as_euler("xyz"))
        )
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose_r[:3], R.from_quat(tcp_pose_r[3:]).as_euler("xyz"))
        )

        return observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ============================================================================
# Action / Reward Wrappers
# ============================================================================

class GripperCloseEnv(gym.ActionWrapper):
    """
    强制去掉夹爪动作维度的 wrapper。
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

        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    """
    旧版单臂夹爪惩罚 wrapper。
    当前 Galaxea 单臂任务建议使用任务 wrapper.py 中的 SingleGripperPenaltyWrapper。
    """

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
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
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
    双臂夹爪动作惩罚 wrapper。
    """

    def __init__(self, env, penalty=0.1):
        super().__init__(env)

        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0
        self.last_gripper_pos_right = 0

    def reward(self, reward: float, action) -> float:
        if action[6] < -0.5 and self.last_gripper_pos_left == 0:
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif action[6] > 0.5 and self.last_gripper_pos_left == 1:
            reward -= self.penalty
            self.last_gripper_pos_left = 0

        if action[13] < -0.5 and self.last_gripper_pos_right == 0:
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif action[13] > 0.5 and self.last_gripper_pos_right == 1:
            reward -= self.penalty
            self.last_gripper_pos_right = 0

        return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        reward = self.reward(reward, action)

        return observation, reward, terminated, truncated, info


# ============================================================================
# 星海图 R1 PRO VR 物理离合接管 Wrapper
# ============================================================================

class VRInterventionWrapper(gym.ActionWrapper):
    """
    星海图 R1 PRO VR 物理离合接管 Wrapper。

    control_mode 语义：
      0 -> VR 接管，use_vr_mode=True
      2 -> 脚本/AI 接管，use_vr_mode=False

    当前版本核心设计：
      1) /vr_pose 只用于读取：
           - control_mode
           - left/right_gripper_close
         不再用 /vr_pose 的 position/quaternion 生成前 6 维动作。

      2) VR 接管时，机器人由底层 VR 系统实际控制。
         wrapper 在 env.step 前后读取真实 EE pose：
           prev_ee_pose -> next_ee_pose

         然后反推：
           action[:3] = (next_pos - prev_pos) / POS_SCALE
           action[3:6] = relative_rotvec(prev_quat -> next_quat) / ROT_SCALE

         最后 clip 到 [-1, 1]。

      3) 夹爪 action[6] 使用三值事件标签：
           -1 = close event
            0 = hold / no-op
           +1 = open event

    注意：
      保存进 replay buffer / demo_buffer 的 intervene_action
      是“真实末端反馈反推出来的归一化动作”，不是 VR 手柄坐标。
    """

    def __init__(
        self,
        env,
        action_indices=None,
        gripper_enabled=True,
        pos_scale=None,
        rot_scale=None,
        use_ee_feedback_action=True,
    ):
        super().__init__(env)

        self.use_vr_mode = False
        self.last_control_mode = None
        self.last_service_target = None

        self.action_indices = action_indices
        self.gripper_enabled = bool(gripper_enabled)
        self.use_ee_feedback_action = bool(use_ee_feedback_action)

        # 从底层 env/config 自动读取动作缩放。
        #代码里如果没有显式传，就会从底层 env/config 读取 POS_SCALE / ROT_SCALE，读不到才用默认 0.01 / 0.05。
        self.pos_scale = float(
            pos_scale if pos_scale is not None else self._get_config_value("POS_SCALE", 0.01)
        )
        self.rot_scale = float(
            rot_scale if rot_scale is not None else self._get_config_value("ROT_SCALE", 0.05)
        )

        if self.pos_scale <= 1e-9:
            raise ValueError(f"pos_scale 必须为正数，当前 {self.pos_scale}")
        if self.rot_scale <= 1e-9:
            raise ValueError(f"rot_scale 必须为正数，当前 {self.rot_scale}")

        # 夹爪按钮上一帧状态，用于生成边沿事件。
        self.prev_left_gripper_close = None
        self.prev_right_gripper_close = None

        # 最近一次 VR gripper 事件。
        self.latest_left_gripper_event = 0.0
        self.latest_right_gripper_event = 0.0

        # 当前 env action 维度。
        self.action_dim = int(self.env.action_space.shape[0])

        # 在 VR 模式下传给 env.step 的占位动作。
        # 注意：真实机器人运动由底层 VR 模式完成；
        # 保存到 buffer 的 intervene_action 会在 step 后由 EE delta 反推覆盖。
        self.latest_vr_action = np.zeros(self.action_dim, dtype=np.float32)

        # 上一帧真实 EE pose，用于反推 delta action。
        self.last_ee_pose = None

        # 避免 fallback warning 刷屏。
        self._warned_missing_ee_pose = False

        # 复用底层 bridge 的 node
        try:
            self.node = self.env.unwrapped.bridge.node
        except AttributeError:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node("vr_intervention_wrapper_node")

        self.mode_client = self.node.create_client(
            SwitchControlModeVR,
            "/switch_control_mode_vr",
        )

        print("⏳ 正在等待底层硬件服务连接 (/switch_control_mode_vr)...")
        if not self.mode_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(
                "❌ /switch_control_mode_vr 服务未上线，无法建立 VR 模式切换通道"
            )
        print("✅ 服务已连接！持久化通讯通道已建立，VR 物理离合器准备就绪。")

        self.vr_sub = self.node.create_subscription(
            VrPose,
            "/vr_pose",
            self._vr_cb,
            10,
        )

        print(
            "✅ VRInterventionWrapper 已启用："
            f" use_ee_feedback_action={self.use_ee_feedback_action},"
            f" pos_scale={self.pos_scale},"
            f" rot_scale={self.rot_scale},"
            " gripper_event_mode=True"
        )

    def _clear_vr_runtime_cache(self):
        """
        清空 VR 接管运行时残留状态。

        目的：
          - Mode0 -> Mode2 时，不把 VR 残留 gripper event 带给下一帧 actor
          - Mode2 -> Mode0 时，不跨 mode 计算 EE delta
          - 防止 latest_vr_action 的 zero/hold 占位动作在切换瞬间误用
        """
        self.latest_vr_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_ee_pose = None
        self._reset_gripper_events()

    # ----------------------------------------------------------------------
    # Config helpers
    # ----------------------------------------------------------------------

    def _get_config_value(self, name: str, default):
        """
        从 env.unwrapped.config 里读取配置值。
        当前 GalaxeaUSBEnv 会保存 self.config，所以这里可以读到 POS_SCALE / ROT_SCALE。
        """
        base_env = self.env.unwrapped

        cfg = getattr(base_env, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)

        if hasattr(base_env, name):
            return getattr(base_env, name)

        return default

    # ----------------------------------------------------------------------
    # ROS service / mode switch
    # ----------------------------------------------------------------------

    def _on_switch_done(self, future):
        try:
            resp = future.result()
            if resp is not None:
                print(f"✅ 模式切换返回: success={resp.success}, message={resp.message}")
        except Exception as e:
            print(f"❌ 模式切换 service 调用失败: {e}")

    def _call_switch_service(self, use_vr_mode: bool):
        """
        异步调用模式切换服务。

        use_vr_mode=True  -> 切到 VR 控制
        use_vr_mode=False -> 切到脚本/IK 控制
        """
        if self.last_service_target == use_vr_mode:
            return

        req = SwitchControlModeVR.Request()
        req.use_vr_mode = use_vr_mode

        future = self.mode_client.call_async(req)
        future.add_done_callback(self._on_switch_done)

        self.last_service_target = use_vr_mode

    # ----------------------------------------------------------------------
    # Gripper helpers
    # ----------------------------------------------------------------------

    def _reset_gripper_events(self):
        self.prev_left_gripper_close = None
        self.prev_right_gripper_close = None
        self.latest_left_gripper_event = 0.0
        self.latest_right_gripper_event = 0.0

    @staticmethod
    def _gripper_button_to_event(current_close: bool, prev_close):
        """
        把 VR 夹爪按钮状态转成三值事件标签：
          -1 = close event
           0 = hold / no-op
          +1 = open event

        第一帧只初始化按钮状态，不触发事件，避免 mode 切换瞬间误触发。
        """
        current_close = bool(current_close)

        if prev_close is None:
            return 0.0, current_close

        if current_close and not prev_close:
            return -1.0, current_close

        if (not current_close) and prev_close:
            return 1.0, current_close

        return 0.0, current_close

    def _make_action_with_gripper_only(self):
        """
        构造 VR 模式下传入 env.step 的占位动作。
        前 6 维为 0，夹爪维度使用最新 gripper event。
        """
        action = np.zeros(self.action_dim, dtype=np.float32)

        if self.action_dim == 7:
            # 单臂模式
            base_env = self.env.unwrapped
            arm_side = getattr(base_env, "arm_side", "right")

            if arm_side == "left":
                action[6] = self.latest_left_gripper_event
            else:
                action[6] = self.latest_right_gripper_event

        elif self.action_dim == 14:
            # 双臂模式
            action[6] = self.latest_left_gripper_event
            action[13] = self.latest_right_gripper_event

        if self.action_indices is not None:
            filtered = np.zeros_like(action)
            filtered[self.action_indices] = action[self.action_indices]
            action = filtered

        return action.astype(np.float32)

    # ----------------------------------------------------------------------
    # EE pose extraction / action reconstruction
    # ----------------------------------------------------------------------

    @staticmethod
    def _to_1d_array(x):
        try:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return None
            return arr
        except Exception:
            return None

    def _extract_pose_from_state_dict(self, state):
        """
        从 raw env observation["state"] dict 里提取右臂/单臂 EE pose。
        兼容不同 key 命名。
        """
        base_env = self.env.unwrapped
        arm_side = getattr(base_env, "arm_side", "right")

        preferred_keys = []

        if arm_side == "right":
            preferred_keys.extend([
                "right_ee_pose",
                "right/tcp_pose",
                "right_tcp_pose",
                "state/right_ee_pose",
                "state/right/tcp_pose",
            ])
        else:
            preferred_keys.extend([
                "left_ee_pose",
                "left/tcp_pose",
                "left_tcp_pose",
                "state/left_ee_pose",
                "state/left/tcp_pose",
            ])

        preferred_keys.extend([
            "ee_pose",
            "tcp_pose",
            "pose_ee",
            "pose_ee_arm_right",
            "pose_ee_arm_left",
        ])

        for key in preferred_keys:
            if key in state:
                arr = self._to_1d_array(state[key])
                if arr is not None and arr.size >= 6:
                    return arr[:7] if arr.size >= 7 else arr[:6]

        # 兜底：找包含 pose/tcp/ee 的字段
        for key, value in state.items():
            k = str(key).lower()
            if ("pose" in k or "tcp" in k or "ee" in k) and "gripper" not in k:
                arr = self._to_1d_array(value)
                if arr is not None and arr.size >= 6:
                    return arr[:7] if arr.size >= 7 else arr[:6]

        return None

    def _extract_ee_pose(self, obs):
        """
        从 observation 中提取 EE pose。

        支持：
          1) obs["state"] 是 dict，包含 right_ee_pose / tcp_pose 等
          2) obs["state"] 是 array，shape 类似 (1,8)，前 7 维为 ee_pose，最后 1 维为 gripper
        """
        if obs is None or not isinstance(obs, dict):
            return None

        if "state" not in obs:
            return None

        state = obs["state"]

        if isinstance(state, dict):
            return self._extract_pose_from_state_dict(state)

        arr = self._to_1d_array(state)
        if arr is None:
            return None

        # 常见单臂拼接 state:
        # [x, y, z, qx, qy, qz, qw, gripper]
        if arr.size >= 7:
            return arr[:7]

        if arr.size >= 6:
            return arr[:6]

        return None

    @staticmethod
    def _pose_to_pos_quat(pose):
        """
        支持两种 pose：
          7 维: xyz + quat(xyzw)
          6 维: xyz + euler(xyz)
        """
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)

        if pose.size >= 7:
            pos = pose[:3].astype(np.float32)
            quat = pose[3:7].astype(np.float32)

            norm = float(np.linalg.norm(quat))
            if norm < 1e-8:
                return None, None

            quat = quat / norm
            return pos, quat

        if pose.size >= 6:
            pos = pose[:3].astype(np.float32)
            quat = R.from_euler("xyz", pose[3:6]).as_quat().astype(np.float32)
            return pos, quat

        return None, None

    def _compute_action_from_ee_delta(self, prev_pose, next_pose, gripper_event):
        """
        用真实 EE pose 差分反推 normalized action。

        action[:3] = pos_delta / POS_SCALE
        action[3:6] = rotvec_delta / ROT_SCALE
        action[6] = gripper_event
        """
        action = np.zeros(self.action_dim, dtype=np.float32)

        if prev_pose is None or next_pose is None:
            return None

        prev_pos, prev_quat = self._pose_to_pos_quat(prev_pose)
        next_pos, next_quat = self._pose_to_pos_quat(next_pose)

        if prev_pos is None or next_pos is None:
            return None

        try:
            pos_delta = next_pos - prev_pos

            prev_rot = R.from_quat(prev_quat)
            next_rot = R.from_quat(next_quat)

            # 假设 env 执行语义为:
            # next_rot = delta_rot * prev_rot
            # 所以 delta_rot = next_rot * prev_rot.inv()
            rot_delta = (next_rot * prev_rot.inv()).as_rotvec().astype(np.float32)

            arm_action = np.zeros(6, dtype=np.float32)
            arm_action[:3] = pos_delta / self.pos_scale
            arm_action[3:6] = rot_delta / self.rot_scale
            arm_action = np.clip(arm_action, -1.0, 1.0)

        except Exception:
            return None

        if self.action_dim == 7:
            action[:6] = arm_action
            action[6] = float(gripper_event)
            return action.astype(np.float32)

        # 当前任务是单臂；双臂暂时兜底返回 None，避免错误写入 14 维。
        return None

    def _get_current_single_gripper_event(self):
        """
        根据单臂 arm_side 返回当前 gripper event。
        """
        base_env = self.env.unwrapped
        arm_side = getattr(base_env, "arm_side", "right")

        if arm_side == "left":
            return float(self.latest_left_gripper_event)

        return float(self.latest_right_gripper_event)

    # ----------------------------------------------------------------------
    # VR callback
    # ----------------------------------------------------------------------

    def _vr_cb(self, msg):
        """
        高频回调：
        - 监听 control_mode
        - 只在 0/2 发生变化时发一次 service
        - 不再使用 /vr_pose 的 position/quaternion 生成前 6 维动作
        - 只使用 gripper_close 生成 -1/0/+1 夹爪事件

        关键修复：
        - Mode0 / Mode2 切换瞬间清空 latest_vr_action / gripper event / last_ee_pose
        - 防止切回 Mode2 后仍残留一帧 VR zero/hold action
        """
        current_mode = int(msg.control_mode)

        if current_mode not in (0, 2):
            return

        # 暴露给录制脚本读取，方便判断当前 mode。
        self.control_mode = current_mode
        self.current_control_mode = current_mode
        self.mode = current_mode

        self.use_vr_mode = current_mode == 0

        # 模式变化时，边沿触发 service
        if current_mode != self.last_control_mode:
            self.last_control_mode = current_mode

            # 关键：任何 mode 切换都清空 VR 残留状态。
            self._clear_vr_runtime_cache()

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

        # 非 VR 模式下，不更新 gripper event，也不生成 latest_vr_action。
        # 这样 Mode2 后第一帧完全由 actor/script action 决定。
        if not self.use_vr_mode:
            self.latest_vr_action = np.zeros(self.action_dim, dtype=np.float32)
            return

        if self.gripper_enabled:
            left_event, self.prev_left_gripper_close = self._gripper_button_to_event(
                msg.left_gripper_close,
                self.prev_left_gripper_close,
            )
            right_event, self.prev_right_gripper_close = self._gripper_button_to_event(
                msg.right_gripper_close,
                self.prev_right_gripper_close,
            )

            self.latest_left_gripper_event = float(left_event)
            self.latest_right_gripper_event = float(right_event)
        else:
            self.latest_left_gripper_event = 0.0
            self.latest_right_gripper_event = 0.0

        self.latest_vr_action = self._make_action_with_gripper_only()

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._clear_vr_runtime_cache()
        self.last_ee_pose = self._extract_ee_pose(obs)

        return obs, info

    def action(self, action: np.ndarray):
        """
        根据当前接管状态决定到底执行谁的动作。

        VR 模式下：
          传给底层 env.step 的只是占位 action。
          真正保存的 intervene_action 会在 step 后根据 EE feedback 反推。
        """
        if self.use_vr_mode:
            return self.latest_vr_action.copy(), True

        return np.asarray(action, dtype=np.float32), False

    def step(self, action):
        """
        VR Mode0:
          - 不发布 zero / hold
          - 先 prime 一次当前 EE pose 作为 baseline
          - 再 observe_only_step() 读取下一帧
          - 用 baseline -> next_ee_pose 反推 intervene_action

        Mode2:
          - 完全由 actor / script action 决定
        """
        prev_ee_pose = None if self.last_ee_pose is None else self.last_ee_pose.copy()

        exec_action, is_intervened = self.action(action)

        # 记录这次是否做了 baseline prime，方便调试
        vr_prime_baseline_used = False
        vr_prime_baseline_success = False

        if is_intervened:
            # VR Mode0 接管时，真实机器人由底层 VR 系统控制。
            # 这里绝对不能调用 self.env.step(exec_action)，否则会发布 zero/hold。
            base_env = self.env.unwrapped

            if not hasattr(base_env, "observe_only_step"):
                raise RuntimeError(
                    "VRInterventionWrapper 需要底层 env 实现 observe_only_step()。"
                    "否则 VR Mode0 期间会 env.step(zero/hold) 并发布错误动作。"
                )

            # ------------------------------------------------------------------
            # 关键新增：
            # 如果刚进入 Mode0，last_ee_pose 被清空，先只读取一次当前真实 EE pose
            # 作为本帧 delta 的起点。
            #
            # 注意：
            # - 这里调用的是 _get_sync_obs()
            # - 不调用 env.step()
            # - 不调用 _publish_action()
            # - 不会发布 zero / hold
            # ------------------------------------------------------------------
            if prev_ee_pose is None:
                vr_prime_baseline_used = True

                try:
                    if hasattr(base_env, "_get_sync_obs"):
                        prime_obs = base_env._get_sync_obs()
                        prime_ee_pose = self._extract_ee_pose(prime_obs)

                        if prime_ee_pose is not None:
                            prev_ee_pose = prime_ee_pose.copy()
                            self.last_ee_pose = prime_ee_pose.copy()
                            vr_prime_baseline_success = True
                        else:
                            vr_prime_baseline_success = False
                    else:
                        vr_prime_baseline_success = False

                except Exception as e:
                    vr_prime_baseline_success = False
                    print(
                        "⚠️ [VRInterventionWrapper] prime baseline pose 失败，"
                        f"将继续使用 fallback intervene_action。err={e!r}"
                    )

            obs, rew, done, truncated, info = base_env.observe_only_step()
            if info is None:
                info = {}

            info["observe_only"] = True
            info["vr_observe_only"] = True
            info["vr_prime_baseline_used"] = bool(vr_prime_baseline_used)
            info["vr_prime_baseline_success"] = bool(vr_prime_baseline_success)

        else:
            # Mode2：完全由 actor / script action 决定。
            # 不使用 latest_vr_action，不补 zero，不补 hold。
            obs, rew, done, truncated, info = self.env.step(exec_action)
            if info is None:
                info = {}

            info["observe_only"] = False
            info["vr_observe_only"] = False
            info["vr_prime_baseline_used"] = False
            info["vr_prime_baseline_success"] = False

        next_ee_pose = self._extract_ee_pose(obs)

        if is_intervened:
            gripper_event = self._get_current_single_gripper_event()

            measured_action = None
            if self.use_ee_feedback_action:
                measured_action = self._compute_action_from_ee_delta(
                    prev_pose=prev_ee_pose,
                    next_pose=next_ee_pose,
                    gripper_event=gripper_event,
                )

            if measured_action is None:
                # 兜底：如果仍然无法反推 EE delta，就保存占位动作。
                # 现在有 prime baseline 后，这种情况应该明显减少。
                if not self._warned_missing_ee_pose:
                    print(
                        "⚠️ [VRInterventionWrapper] 无法从 obs 反推 EE delta action，"
                        "暂时使用 latest_vr_action 作为 intervene_action。"
                    )
                    self._warned_missing_ee_pose = True

                measured_action = exec_action.copy()
                info["intervene_action_valid"] = False
                info["intervene_action_invalid_reason"] = "missing_or_invalid_ee_pose_delta"
            else:
                info["intervene_action_valid"] = True
                info["intervene_action_invalid_reason"] = ""

            info["intervene_action"] = measured_action.astype(np.float32).copy()

        info["left"] = bool(is_intervened)
        info["right"] = bool(is_intervened)
        info["left1"] = bool(is_intervened)
        info["left2"] = False
        info["right1"] = bool(is_intervened)
        info["right2"] = False

        self.last_ee_pose = None if next_ee_pose is None else next_ee_pose.copy()

        return obs, rew, done, truncated, info












# import time
# from typing import List

# from gymnasium import Env, spaces
# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box
# from scipy.spatial.transform import Rotation as R

# # ROS 2 相关库
# import rclpy
# from rclpy.node import Node
# from teleoperation_msg_ros2.srv import SwitchControlModeVR
# from teleoperation_msg_ros2.msg import VrPose


# # ============================================================================
# # 通用功能与辅助 Wrappers
# # ============================================================================

# sigmoid = lambda x: 1 / (1 + np.exp(-x))


# class HumanClassifierWrapper(gym.Wrapper):
#     """
#     手动打分器 Wrapper。
#     当 done=True 时，在终端要求人工输入成功/失败。
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)

#         if done:
#             while True:
#                 try:
#                     rew = int(input("Success? (1/0): "))
#                     assert rew == 0 or rew == 1
#                     break
#                 except Exception:
#                     print("Invalid input. Please enter 1 for success, 0 for failure.")

#         info["succeed"] = rew
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info


# class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     视觉二分类自动打分器 Wrapper。
#     每一步 env.step 后，用 reward classifier 判断当前 obs 是否成功。
#     """

#     def __init__(self, env: Env, reward_classifier_func, target_hz=None):
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

#         rew = int(self.compute_reward(obs))
#         success = bool(rew)
#         done = bool(done or success)

#         info["succeed"] = success
#         info["success"] = success
#         info["is_success"] = success

#         if self.target_hz is not None:
#             time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))

#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         info["succeed"] = False
#         info["success"] = False
#         info["is_success"] = False
#         return obs, info


# class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     多阶段视觉打分器 Wrapper。
#     """

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

#         return sum(rewards)

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)

#         rew = self.compute_reward(obs)
#         done = bool(done or all(self.received))
#         info["succeed"] = all(self.received)

#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.received = [False] * len(self.reward_classifier_func)
#         info["succeed"] = False
#         return obs, info


# # ============================================================================
# # Observation Wrappers
# # ============================================================================

# class Quat2EulerWrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转欧拉角。"""

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)

#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         return observation


# class Quat2R2Wrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转旋转矩阵前两列。"""

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)

#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(9,),
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         r = R.from_quat(tcp_pose[3:]).as_matrix()
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], r[..., :2].flatten())
#         )
#         return observation


# class DualQuat2EulerWrapper(gym.ObservationWrapper):
#     """双臂环境：左右臂四元数转欧拉角。"""

#     def __init__(self, env: Env):
#         super().__init__(env)

#         assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
#         assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)

#         self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )
#         self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )

#     def observation(self, observation):
#         tcp_pose_l = observation["state"]["left/tcp_pose"]
#         tcp_pose_r = observation["state"]["right/tcp_pose"]

#         observation["state"]["left/tcp_pose"] = np.concatenate(
#             (tcp_pose_l[:3], R.from_quat(tcp_pose_l[3:]).as_euler("xyz"))
#         )
#         observation["state"]["right/tcp_pose"] = np.concatenate(
#             (tcp_pose_r[:3], R.from_quat(tcp_pose_r[3:]).as_euler("xyz"))
#         )

#         return observation

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return self.observation(obs), info


# # ============================================================================
# # Action / Reward Wrappers
# # ============================================================================

# class GripperCloseEnv(gym.ActionWrapper):
#     """
#     强制去掉夹爪动作维度的 wrapper。
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


# class GripperPenaltyWrapper(gym.RewardWrapper):
#     """
#     旧版单臂夹爪惩罚 wrapper。
#     当前 Galaxea 单臂任务建议使用任务 wrapper.py 中的 SingleGripperPenaltyWrapper。
#     """

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
#         return reward

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)

#         if "intervene_action" in info:
#             action = info["intervene_action"]

#         reward = self.reward(reward, action)
#         self.last_gripper_pos = observation["state"][0, 0]

#         return observation, reward, terminated, truncated, info


# class DualGripperPenaltyWrapper(gym.RewardWrapper):
#     """
#     双臂夹爪动作惩罚 wrapper。
#     """

#     def __init__(self, env, penalty=0.1):
#         super().__init__(env)

#         assert env.action_space.shape == (14,)
#         self.penalty = penalty
#         self.last_gripper_pos_left = 0
#         self.last_gripper_pos_right = 0

#     def reward(self, reward: float, action) -> float:
#         if action[6] < -0.5 and self.last_gripper_pos_left == 0:
#             reward -= self.penalty
#             self.last_gripper_pos_left = 1
#         elif action[6] > 0.5 and self.last_gripper_pos_left == 1:
#             reward -= self.penalty
#             self.last_gripper_pos_left = 0

#         if action[13] < -0.5 and self.last_gripper_pos_right == 0:
#             reward -= self.penalty
#             self.last_gripper_pos_right = 1
#         elif action[13] > 0.5 and self.last_gripper_pos_right == 1:
#             reward -= self.penalty
#             self.last_gripper_pos_right = 0

#         return reward

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)

#         if "intervene_action" in info:
#             action = info["intervene_action"]

#         reward = self.reward(reward, action)

#         return observation, reward, terminated, truncated, info


# # ============================================================================
# # 星海图 R1 PRO VR 物理离合接管 Wrapper
# # ============================================================================

# class VRInterventionWrapper(gym.ActionWrapper):
#     """
#     星海图 R1 PRO VR 物理离合接管 Wrapper。

#     control_mode 语义：
#       0 -> VR 接管，use_vr_mode=True
#       2 -> 脚本/AI 接管，use_vr_mode=False

#     VR 前 6 维动作：
#       进入 VR 接管时，把当前 VR 手柄位姿记为 anchor。
#       后续 action = (当前 VR 位姿 - anchor) / vr_pos_range_or_rot_range。
#       最后 clip 到 [-1,1]。

#     夹爪第 7 维动作：
#       -1 = close event
#        0 = hold / no-op
#       +1 = open event

#     注意：
#       这里输出的是训练/策略标签空间。
#       硬件 10/80 映射应该只发生在底层 env 执行层，不应该进入 replay buffer。
#     """

#     def __init__(
#         self,
#         env,
#         action_indices=None,
#         gripper_enabled=True,
#         # 如果机器人变得太慢，再把对应方向调小。
#         # 最终目标是：
#         # 录制时大多数 intervene_action[:6] 在 [-0.7, 0.7] 左右，
#         # 少量接近 1.0 可以接受，但不要长时间饱和。
#         vr_pos_range=(0.25, 0.25, 0.25),
#         vr_rot_range=(0.8, 0.8, 0.8),
#     ):
#         super().__init__(env)

#         self.use_vr_mode = False
#         self.last_control_mode = None
#         self.last_service_target = None

#         self.action_indices = action_indices
#         self.gripper_enabled = gripper_enabled

#         # ------------------------------------------------------
#         # VR 源头归一化参数
#         # ------------------------------------------------------
#         self.vr_pos_range = np.asarray(vr_pos_range, dtype=np.float32)
#         self.vr_rot_range = np.asarray(vr_rot_range, dtype=np.float32)

#         if self.vr_pos_range.shape != (3,):
#             raise ValueError(f"vr_pos_range 必须是 3 维，当前 {self.vr_pos_range}")

#         if self.vr_rot_range.shape != (3,):
#             raise ValueError(f"vr_rot_range 必须是 3 维，当前 {self.vr_rot_range}")

#         if np.any(self.vr_pos_range <= 1e-6):
#             raise ValueError(f"vr_pos_range 必须全为正数，当前 {self.vr_pos_range}")

#         if np.any(self.vr_rot_range <= 1e-6):
#             raise ValueError(f"vr_rot_range 必须全为正数，当前 {self.vr_rot_range}")

#         # VR 接管瞬间的手柄零点
#         self.left_pos_anchor = None
#         self.right_pos_anchor = None
#         self.left_euler_anchor = None
#         self.right_euler_anchor = None

#         # 夹爪按钮上一帧状态，用于生成边沿事件：
#         # False -> True  : close event(-1)
#         # True  -> False : open event(+1)
#         # 其他            : hold(0)
#         self.prev_left_gripper_close = None
#         self.prev_right_gripper_close = None

#         # 动作缓存长度跟底层 env 当前 action_space 对齐
#         self.latest_vr_action = np.zeros(
#             self.env.action_space.shape[0],
#             dtype=np.float32,
#         )

#         # 复用底层 bridge 的 node
#         try:
#             self.node = self.env.unwrapped.bridge.node
#         except AttributeError:
#             if not rclpy.ok():
#                 rclpy.init()
#             self.node = rclpy.create_node("vr_intervention_wrapper_node")

#         self.mode_client = self.node.create_client(
#             SwitchControlModeVR,
#             "/switch_control_mode_vr",
#         )

#         print("⏳ 正在等待底层硬件服务连接 (/switch_control_mode_vr)...")
#         if not self.mode_client.wait_for_service(timeout_sec=15.0):
#             raise RuntimeError(
#                 "❌ /switch_control_mode_vr 服务未上线，无法建立 VR 模式切换通道"
#             )
#         print("✅ 服务已连接！持久化通讯通道已建立，VR 物理离合器准备就绪。")

#         self.vr_sub = self.node.create_subscription(
#             VrPose,
#             "/vr_pose",
#             self._vr_cb,
#             10,
#         )

#         print(
#             "✅ VRInterventionWrapper 源头归一化已启用："
#             f" vr_pos_range={self.vr_pos_range.tolist()},"
#             f" vr_rot_range={self.vr_rot_range.tolist()},"
#             " gripper_event_mode=True"
#         )

#     # ----------------------------------------------------------------------
#     # ROS service / mode switch
#     # ----------------------------------------------------------------------

#     def _on_switch_done(self, future):
#         try:
#             resp = future.result()
#             if resp is not None:
#                 print(f"✅ 模式切换返回: success={resp.success}, message={resp.message}")
#         except Exception as e:
#             print(f"❌ 模式切换 service 调用失败: {e}")

#     def _call_switch_service(self, use_vr_mode: bool):
#         """
#         异步调用模式切换服务。

#         use_vr_mode=True  -> 切到 VR 控制
#         use_vr_mode=False -> 切到脚本/IK 控制
#         """
#         if self.last_service_target == use_vr_mode:
#             return

#         req = SwitchControlModeVR.Request()
#         req.use_vr_mode = use_vr_mode

#         future = self.mode_client.call_async(req)
#         future.add_done_callback(self._on_switch_done)

#         self.last_service_target = use_vr_mode

#     # ----------------------------------------------------------------------
#     # VR normalization helpers
#     # ----------------------------------------------------------------------

#     @staticmethod
#     def _wrap_angle(x):
#         """
#         把角度差限制到 [-pi, pi]，避免欧拉角跨 pi 时突然跳变。
#         """
#         x = np.asarray(x, dtype=np.float32)
#         return (x + np.pi) % (2 * np.pi) - np.pi

#     def _reset_vr_anchors(self):
#         """
#         清空 VR anchor。
#         下一次收到 VR 接管消息时，会用当前手柄位姿重新作为零点。

#         同时重置夹爪按钮上一帧状态，避免跨 mode 误触发 open/close event。
#         """
#         self.left_pos_anchor = None
#         self.right_pos_anchor = None
#         self.left_euler_anchor = None
#         self.right_euler_anchor = None

#         self.prev_left_gripper_close = None
#         self.prev_right_gripper_close = None

#     def _ensure_vr_anchors(
#         self,
#         left_pos,
#         right_pos,
#         left_euler,
#         right_euler,
#     ):
#         """
#         如果当前没有 anchor，则把当前 VR 位姿设为零点。
#         """
#         if (
#             self.left_pos_anchor is None
#             or self.right_pos_anchor is None
#             or self.left_euler_anchor is None
#             or self.right_euler_anchor is None
#         ):
#             self.left_pos_anchor = np.asarray(left_pos, dtype=np.float32).copy()
#             self.right_pos_anchor = np.asarray(right_pos, dtype=np.float32).copy()
#             self.left_euler_anchor = np.asarray(left_euler, dtype=np.float32).copy()
#             self.right_euler_anchor = np.asarray(right_euler, dtype=np.float32).copy()

#             print(
#                 "🎯 [VR归一化] 已设置当前手柄位姿为零点："
#                 f" left_pos_anchor={np.round(self.left_pos_anchor, 4).tolist()},"
#                 f" right_pos_anchor={np.round(self.right_pos_anchor, 4).tolist()}"
#             )

#     def _normalize_vr_pose_to_action(
#         self,
#         pos,
#         euler,
#         pos_anchor,
#         euler_anchor,
#     ):
#         """
#         把 VR 原始绝对位姿转成 policy 动作空间。

#         输出：
#           pos_action ∈ [-1,1]
#           rot_action ∈ [-1,1]
#         """
#         pos = np.asarray(pos, dtype=np.float32)
#         euler = np.asarray(euler, dtype=np.float32)
#         pos_anchor = np.asarray(pos_anchor, dtype=np.float32)
#         euler_anchor = np.asarray(euler_anchor, dtype=np.float32)

#         pos_delta = pos - pos_anchor
#         euler_delta = self._wrap_angle(euler - euler_anchor)

#         pos_action = pos_delta / self.vr_pos_range
#         rot_action = euler_delta / self.vr_rot_range

#         pos_action = np.clip(pos_action, -1.0, 1.0)
#         rot_action = np.clip(rot_action, -1.0, 1.0)

#         return pos_action.astype(np.float32), rot_action.astype(np.float32)

#     @staticmethod
#     def _gripper_button_to_event(current_close: bool, prev_close):
#         """
#         把 VR 夹爪按钮状态转成三值事件标签：
#           -1 = close event
#            0 = hold / no-op
#           +1 = open event

#         设计：
#           prev_close is None 时，认为刚进入 VR 接管。
#           第一帧只初始化按钮状态，不触发 open/close，避免 mode 切换瞬间误触发。
#         """
#         current_close = bool(current_close)

#         if prev_close is None:
#             return 0.0, current_close

#         if current_close and not prev_close:
#             return -1.0, current_close

#         if (not current_close) and prev_close:
#             return 1.0, current_close

#         return 0.0, current_close

#     # ----------------------------------------------------------------------
#     # Action packing
#     # ----------------------------------------------------------------------

#     def _pack_action_for_env(self, full_action_14: np.ndarray) -> np.ndarray:
#         """
#         把 VR 原始双臂 14 维动作，打包成当前底层 env 真正需要的动作维度。

#         dual         -> 保持 14 维
#         single-left  -> 取前 7 维
#         single-right -> 取后 7 维
#         """
#         full_action_14 = np.asarray(full_action_14, dtype=np.float32).reshape(-1)

#         if full_action_14.shape[0] != 14:
#             raise ValueError(
#                 f"VR 原始动作必须是 14 维，当前收到 {full_action_14.shape}"
#             )

#         base_env = self.env.unwrapped
#         arm_mode = getattr(base_env, "arm_mode", "dual")
#         arm_side = getattr(base_env, "arm_side", "right")

#         if arm_mode == "dual":
#             return full_action_14

#         if arm_side == "left":
#             return full_action_14[:7]

#         return full_action_14[7:14]

#     # ----------------------------------------------------------------------
#     # VR callback
#     # ----------------------------------------------------------------------

#     def _vr_cb(self, msg):
#         """
#         高频回调：
#         - 监听 control_mode
#         - 只在 0/2 发生变化时发一次 service
#         - mode=0 时，把 VR 相对位姿归一化成 [-1,1] 动作
#         - gripper 输出 -1/0/+1 三值事件标签
#         """
#         current_mode = int(msg.control_mode)

#         if current_mode not in (0, 2):
#             return

#         self.use_vr_mode = current_mode == 0

#         # 模式变化时，边沿触发 service
#         if current_mode != self.last_control_mode:
#             self.last_control_mode = current_mode

#             if current_mode == 0:
#                 # 进入 VR 接管时重置 anchor。
#                 # 后面同一个 callback 中会用当前手柄位姿重新设置零点。
#                 self._reset_vr_anchors()

#                 self._call_switch_service(True)
#                 print("🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True")

#                 if hasattr(self.env.unwrapped, "notify_script_control"):
#                     self.env.unwrapped.notify_script_control(False)

#             elif current_mode == 2:
#                 # 离开 VR 接管时也重置 anchor。
#                 self._reset_vr_anchors()

#                 self._call_switch_service(False)
#                 print("🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False")

#                 if hasattr(self.env.unwrapped, "notify_script_control"):
#                     self.env.unwrapped.notify_script_control(True)

#         # 只有 VR 接管时，才解析 VR 位姿为动作
#         if not self.use_vr_mode:
#             return

#         left_pos = np.asarray(msg.left_position, dtype=np.float32)
#         right_pos = np.asarray(msg.right_position, dtype=np.float32)

#         left_euler = R.from_quat(msg.left_quaternion).as_euler("xyz").astype(np.float32)
#         right_euler = R.from_quat(msg.right_quaternion).as_euler("xyz").astype(np.float32)

#         self._ensure_vr_anchors(
#             left_pos=left_pos,
#             right_pos=right_pos,
#             left_euler=left_euler,
#             right_euler=right_euler,
#         )

#         left_pos_action, left_rot_action = self._normalize_vr_pose_to_action(
#             pos=left_pos,
#             euler=left_euler,
#             pos_anchor=self.left_pos_anchor,
#             euler_anchor=self.left_euler_anchor,
#         )

#         right_pos_action, right_rot_action = self._normalize_vr_pose_to_action(
#             pos=right_pos,
#             euler=right_euler,
#             pos_anchor=self.right_pos_anchor,
#             euler_anchor=self.right_euler_anchor,
#         )

#         if self.gripper_enabled:
#             left_event, self.prev_left_gripper_close = self._gripper_button_to_event(
#                 msg.left_gripper_close,
#                 self.prev_left_gripper_close,
#             )
#             right_event, self.prev_right_gripper_close = self._gripper_button_to_event(
#                 msg.right_gripper_close,
#                 self.prev_right_gripper_close,
#             )

#             left_grip = np.array([left_event], dtype=np.float32)
#             right_grip = np.array([right_event], dtype=np.float32)
#         else:
#             left_grip = np.zeros(1, dtype=np.float32)
#             right_grip = np.zeros(1, dtype=np.float32)

#         raw_vr_action = np.concatenate(
#             [
#                 left_pos_action,
#                 left_rot_action,
#                 left_grip,
#                 right_pos_action,
#                 right_rot_action,
#                 right_grip,
#             ]
#         ).astype(np.float32)

#         if self.action_indices is not None:
#             filtered_action = np.zeros_like(raw_vr_action)
#             filtered_action[self.action_indices] = raw_vr_action[self.action_indices]
#             raw_vr_action = filtered_action

#         self.latest_vr_action = self._pack_action_for_env(raw_vr_action)

#     # ----------------------------------------------------------------------
#     # Gym ActionWrapper API
#     # ----------------------------------------------------------------------

#     def action(self, action: np.ndarray):
#         """
#         根据当前接管状态决定到底执行谁的动作。
#         """
#         if self.use_vr_mode:
#             return self.latest_vr_action.copy(), True

#         return action, False

#     def step(self, action):
#         new_action, is_intervened = self.action(action)

#         obs, rew, done, truncated, info = self.env.step(new_action)

#         if is_intervened:
#             info["intervene_action"] = new_action.copy()

#         info["left"] = is_intervened
#         info["right"] = is_intervened
#         info["left1"] = is_intervened
#         info["left2"] = False
#         info["right1"] = is_intervened
#         info["right2"] = False

#         return obs, rew, done, truncated, info
    













    


# import time
# from typing import List

# from gymnasium import Env, spaces
# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box
# from scipy.spatial.transform import Rotation as R
# import copy
# import requests

# # ROS 2 相关库
# import rclpy
# from rclpy.node import Node
# from teleoperation_msg_ros2.srv import SwitchControlModeVR
# from teleoperation_msg_ros2.msg import VrPose


# # ============================================================================
# # 通用功能与辅助 Wrappers
# # ============================================================================

# sigmoid = lambda x: 1 / (1 + np.exp(-x))


# class HumanClassifierWrapper(gym.Wrapper):
#     """
#     手动打分器 Wrapper。
#     当 done=True 时，在终端要求人工输入成功/失败。
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)

#         if done:
#             while True:
#                 try:
#                     rew = int(input("Success? (1/0): "))
#                     assert rew == 0 or rew == 1
#                     break
#                 except Exception:
#                     print("Invalid input. Please enter 1 for success, 0 for failure.")

#         info["succeed"] = rew
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info


# class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     视觉二分类自动打分器 Wrapper。
#     每一步 env.step 后，用 reward classifier 判断当前 obs 是否成功。
#     """

#     def __init__(self, env: Env, reward_classifier_func, target_hz=None):
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

#         rew = int(self.compute_reward(obs))
#         success = bool(rew)
#         done = bool(done or success)

#         info["succeed"] = success
#         info["success"] = success
#         info["is_success"] = success

#         if self.target_hz is not None:
#             time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))

#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         info["succeed"] = False
#         info["success"] = False
#         info["is_success"] = False
#         return obs, info


# class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     多阶段视觉打分器 Wrapper。
#     """

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

#         return sum(rewards)

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)

#         rew = self.compute_reward(obs)
#         done = bool(done or all(self.received))
#         info["succeed"] = all(self.received)

#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.received = [False] * len(self.reward_classifier_func)
#         info["succeed"] = False
#         return obs, info


# # ============================================================================
# # Observation Wrappers
# # ============================================================================

# class Quat2EulerWrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转欧拉角。"""

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)

#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         return observation


# class Quat2R2Wrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转旋转矩阵前两列。"""

#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)

#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(9,),
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         r = R.from_quat(tcp_pose[3:]).as_matrix()
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], r[..., :2].flatten())
#         )
#         return observation


# class DualQuat2EulerWrapper(gym.ObservationWrapper):
#     """双臂环境：左右臂四元数转欧拉角。"""

#     def __init__(self, env: Env):
#         super().__init__(env)

#         assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
#         assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)

#         self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )
#         self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
#             -np.inf,
#             np.inf,
#             shape=(6,),
#         )

#     def observation(self, observation):
#         tcp_pose_l = observation["state"]["left/tcp_pose"]
#         tcp_pose_r = observation["state"]["right/tcp_pose"]

#         observation["state"]["left/tcp_pose"] = np.concatenate(
#             (tcp_pose_l[:3], R.from_quat(tcp_pose_l[3:]).as_euler("xyz"))
#         )
#         observation["state"]["right/tcp_pose"] = np.concatenate(
#             (tcp_pose_r[:3], R.from_quat(tcp_pose_r[3:]).as_euler("xyz"))
#         )

#         return observation

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return self.observation(obs), info


# # ============================================================================
# # Action / Reward Wrappers
# # ============================================================================

# class GripperCloseEnv(gym.ActionWrapper):
#     """
#     强制去掉夹爪动作维度的 wrapper。
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


# class GripperPenaltyWrapper(gym.RewardWrapper):
#     """
#     旧版单臂夹爪惩罚 wrapper。
#     当前 Galaxea 单臂任务建议使用任务 wrapper.py 中的 SingleGripperPenaltyWrapper。
#     """

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
#         return reward

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)

#         if "intervene_action" in info:
#             action = info["intervene_action"]

#         reward = self.reward(reward, action)
#         self.last_gripper_pos = observation["state"][0, 0]

#         return observation, reward, terminated, truncated, info


# class DualGripperPenaltyWrapper(gym.RewardWrapper):
#     """
#     双臂夹爪动作惩罚 wrapper。
#     """

#     def __init__(self, env, penalty=0.1):
#         super().__init__(env)

#         assert env.action_space.shape == (14,)
#         self.penalty = penalty
#         self.last_gripper_pos_left = 0
#         self.last_gripper_pos_right = 0

#     def reward(self, reward: float, action) -> float:
#         if action[6] < -0.5 and self.last_gripper_pos_left == 0:
#             reward -= self.penalty
#             self.last_gripper_pos_left = 1
#         elif action[6] > 0.5 and self.last_gripper_pos_left == 1:
#             reward -= self.penalty
#             self.last_gripper_pos_left = 0

#         if action[13] < -0.5 and self.last_gripper_pos_right == 0:
#             reward -= self.penalty
#             self.last_gripper_pos_right = 1
#         elif action[13] > 0.5 and self.last_gripper_pos_right == 1:
#             reward -= self.penalty
#             self.last_gripper_pos_right = 0

#         return reward

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)

#         if "intervene_action" in info:
#             action = info["intervene_action"]

#         reward = self.reward(reward, action)

#         return observation, reward, terminated, truncated, info


# # ============================================================================
# # 星海图 R1 PRO VR 物理离合接管 Wrapper
# # ============================================================================

# class VRInterventionWrapper(gym.ActionWrapper):
#     """
#     星海图 R1 PRO VR 物理离合接管 Wrapper。

#     control_mode 语义：
#       0 -> VR 接管，use_vr_mode=True
#       2 -> 脚本/AI 接管，use_vr_mode=False

#     关键修改：
#       旧逻辑：
#         直接把 VR 绝对 position/euler 当成 action。
#         这会导致 action 前 6 维出现 1.58 这类超出 [-1,1] 的值。

#       新逻辑：
#         进入 VR 接管时，把当前 VR 手柄位姿记为 anchor。
#         后续 action = (当前 VR 位姿 - anchor) / vr_pos_range_or_rot_range。
#         最后 clip 到 [-1,1]。

#     这样输出给 env.step() 的动作就是规范的策略动作空间：
#       action[:6] ∈ [-1,1]
#       action[6]  ∈ {-1,+1} 或接近这两个值
#     """

#     def __init__(
#         self,
#         env,
#         action_indices=None,
#         gripper_enabled=True,
#         # 如果机器人变得太慢，再把对应方向调小。
#         # 最终目标是：
#         # 录制时大多数 intervene_action[:6] 在 [-0.7, 0.7] 左右，
#         # 少量接近 1.0 可以接受，但不要长时间饱和。
#         vr_pos_range=(0.25, 0.25, 0.25),
#         vr_rot_range=(0.8, 0.8, 0.8),
#     ):
#         super().__init__(env)

#         self.use_vr_mode = False
#         self.last_control_mode = None
#         self.last_service_target = None

#         self.action_indices = action_indices
#         self.gripper_enabled = gripper_enabled

#         # ------------------------------------------------------
#         # VR 源头归一化参数
#         # ------------------------------------------------------
#         # VR 手柄移动多少米，对应 action=1。
#         # 如果某方向仍然容易饱和，就把对应 range 调大。
#         # 如果动作太慢，就把对应 range 调小。
#         self.vr_pos_range = np.asarray(vr_pos_range, dtype=np.float32)

#         # VR 手柄旋转多少弧度，对应 action=1。
#         # 0.8 rad 约等于 45.8 度。
#         self.vr_rot_range = np.asarray(vr_rot_range, dtype=np.float32)

#         if self.vr_pos_range.shape != (3,):
#             raise ValueError(f"vr_pos_range 必须是 3 维，当前 {self.vr_pos_range}")

#         if self.vr_rot_range.shape != (3,):
#             raise ValueError(f"vr_rot_range 必须是 3 维，当前 {self.vr_rot_range}")

#         if np.any(self.vr_pos_range <= 1e-6):
#             raise ValueError(f"vr_pos_range 必须全为正数，当前 {self.vr_pos_range}")

#         if np.any(self.vr_rot_range <= 1e-6):
#             raise ValueError(f"vr_rot_range 必须全为正数，当前 {self.vr_rot_range}")

#         # VR 接管瞬间的手柄零点
#         self.left_pos_anchor = None
#         self.right_pos_anchor = None
#         self.left_euler_anchor = None
#         self.right_euler_anchor = None

#         # 动作缓存长度跟底层 env 当前 action_space 对齐
#         self.latest_vr_action = np.zeros(
#             self.env.action_space.shape[0],
#             dtype=np.float32,
#         )

#         # 复用底层 bridge 的 node
#         try:
#             self.node = self.env.unwrapped.bridge.node
#         except AttributeError:
#             if not rclpy.ok():
#                 rclpy.init()
#             self.node = rclpy.create_node("vr_intervention_wrapper_node")

#         self.mode_client = self.node.create_client(
#             SwitchControlModeVR,
#             "/switch_control_mode_vr",
#         )

#         print("⏳ 正在等待底层硬件服务连接 (/switch_control_mode_vr)...")
#         if not self.mode_client.wait_for_service(timeout_sec=15.0):
#             raise RuntimeError(
#                 "❌ /switch_control_mode_vr 服务未上线，无法建立 VR 模式切换通道"
#             )
#         print("✅ 服务已连接！持久化通讯通道已建立，VR 物理离合器准备就绪。")

#         self.vr_sub = self.node.create_subscription(
#             VrPose,
#             "/vr_pose",
#             self._vr_cb,
#             10,
#         )

#         print(
#             "✅ VRInterventionWrapper 源头归一化已启用："
#             f" vr_pos_range={self.vr_pos_range.tolist()},"
#             f" vr_rot_range={self.vr_rot_range.tolist()}"
#         )

#     # ----------------------------------------------------------------------
#     # ROS service / mode switch
#     # ----------------------------------------------------------------------

#     def _on_switch_done(self, future):
#         try:
#             resp = future.result()
#             if resp is not None:
#                 print(f"✅ 模式切换返回: success={resp.success}, message={resp.message}")
#         except Exception as e:
#             print(f"❌ 模式切换 service 调用失败: {e}")

#     def _call_switch_service(self, use_vr_mode: bool):
#         """
#         异步调用模式切换服务。

#         use_vr_mode=True  -> 切到 VR 控制
#         use_vr_mode=False -> 切到脚本/IK 控制
#         """
#         if self.last_service_target == use_vr_mode:
#             return

#         req = SwitchControlModeVR.Request()
#         req.use_vr_mode = use_vr_mode

#         future = self.mode_client.call_async(req)
#         future.add_done_callback(self._on_switch_done)

#         self.last_service_target = use_vr_mode

#     # ----------------------------------------------------------------------
#     # VR normalization helpers
#     # ----------------------------------------------------------------------

#     @staticmethod
#     def _wrap_angle(x):
#         """
#         把角度差限制到 [-pi, pi]，避免欧拉角跨 pi 时突然跳变。
#         """
#         x = np.asarray(x, dtype=np.float32)
#         return (x + np.pi) % (2 * np.pi) - np.pi

#     def _reset_vr_anchors(self):
#         """
#         清空 VR anchor。
#         下一次收到 VR 接管消息时，会用当前手柄位姿重新作为零点。
#         """
#         self.left_pos_anchor = None
#         self.right_pos_anchor = None
#         self.left_euler_anchor = None
#         self.right_euler_anchor = None

#     def _ensure_vr_anchors(
#         self,
#         left_pos,
#         right_pos,
#         left_euler,
#         right_euler,
#     ):
#         """
#         如果当前没有 anchor，则把当前 VR 位姿设为零点。
#         """
#         if (
#             self.left_pos_anchor is None
#             or self.right_pos_anchor is None
#             or self.left_euler_anchor is None
#             or self.right_euler_anchor is None
#         ):
#             self.left_pos_anchor = np.asarray(left_pos, dtype=np.float32).copy()
#             self.right_pos_anchor = np.asarray(right_pos, dtype=np.float32).copy()
#             self.left_euler_anchor = np.asarray(left_euler, dtype=np.float32).copy()
#             self.right_euler_anchor = np.asarray(right_euler, dtype=np.float32).copy()

#             print(
#                 "🎯 [VR归一化] 已设置当前手柄位姿为零点："
#                 f" left_pos_anchor={np.round(self.left_pos_anchor, 4).tolist()},"
#                 f" right_pos_anchor={np.round(self.right_pos_anchor, 4).tolist()}"
#             )

#     def _normalize_vr_pose_to_action(
#         self,
#         pos,
#         euler,
#         pos_anchor,
#         euler_anchor,
#     ):
#         """
#         把 VR 原始绝对位姿转成 policy 动作空间。

#         输出：
#           pos_action ∈ [-1,1]
#           rot_action ∈ [-1,1]
#         """
#         pos = np.asarray(pos, dtype=np.float32)
#         euler = np.asarray(euler, dtype=np.float32)
#         pos_anchor = np.asarray(pos_anchor, dtype=np.float32)
#         euler_anchor = np.asarray(euler_anchor, dtype=np.float32)

#         pos_delta = pos - pos_anchor
#         euler_delta = self._wrap_angle(euler - euler_anchor)

#         pos_action = pos_delta / self.vr_pos_range
#         rot_action = euler_delta / self.vr_rot_range

#         pos_action = np.clip(pos_action, -1.0, 1.0)
#         rot_action = np.clip(rot_action, -1.0, 1.0)

#         return pos_action.astype(np.float32), rot_action.astype(np.float32)

#     # ----------------------------------------------------------------------
#     # Action packing
#     # ----------------------------------------------------------------------

#     def _pack_action_for_env(self, full_action_14: np.ndarray) -> np.ndarray:
#         """
#         把 VR 原始双臂 14 维动作，打包成当前底层 env 真正需要的动作维度。

#         dual         -> 保持 14 维
#         single-left  -> 取前 7 维
#         single-right -> 取后 7 维
#         """
#         full_action_14 = np.asarray(full_action_14, dtype=np.float32).reshape(-1)

#         if full_action_14.shape[0] != 14:
#             raise ValueError(
#                 f"VR 原始动作必须是 14 维，当前收到 {full_action_14.shape}"
#             )

#         base_env = self.env.unwrapped
#         arm_mode = getattr(base_env, "arm_mode", "dual")
#         arm_side = getattr(base_env, "arm_side", "right")

#         if arm_mode == "dual":
#             return full_action_14

#         if arm_side == "left":
#             return full_action_14[:7]

#         return full_action_14[7:14]

#     # ----------------------------------------------------------------------
#     # VR callback
#     # ----------------------------------------------------------------------

#     def _vr_cb(self, msg):
#         """
#         高频回调：
#         - 监听 control_mode
#         - 只在 0/2 发生变化时发一次 service
#         - mode=0 时，把 VR 相对位姿归一化成 [-1,1] 动作
#         """
#         current_mode = int(msg.control_mode)

#         if current_mode not in (0, 2):
#             return

#         self.use_vr_mode = current_mode == 0

#         # 模式变化时，边沿触发 service
#         if current_mode != self.last_control_mode:
#             self.last_control_mode = current_mode

#             if current_mode == 0:
#                 # 进入 VR 接管时重置 anchor。
#                 # 后面同一个 callback 中会用当前手柄位姿重新设置零点。
#                 self._reset_vr_anchors()

#                 self._call_switch_service(True)
#                 print("🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True")

#                 if hasattr(self.env.unwrapped, "notify_script_control"):
#                     self.env.unwrapped.notify_script_control(False)

#             elif current_mode == 2:
#                 # 离开 VR 接管时也重置 anchor。
#                 self._reset_vr_anchors()

#                 self._call_switch_service(False)
#                 print("🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False")

#                 if hasattr(self.env.unwrapped, "notify_script_control"):
#                     self.env.unwrapped.notify_script_control(True)

#         # 只有 VR 接管时，才解析 VR 位姿为动作
#         if not self.use_vr_mode:
#             return

#         left_pos = np.asarray(msg.left_position, dtype=np.float32)
#         right_pos = np.asarray(msg.right_position, dtype=np.float32)

#         left_euler = R.from_quat(msg.left_quaternion).as_euler("xyz").astype(np.float32)
#         right_euler = R.from_quat(msg.right_quaternion).as_euler("xyz").astype(np.float32)

#         self._ensure_vr_anchors(
#             left_pos=left_pos,
#             right_pos=right_pos,
#             left_euler=left_euler,
#             right_euler=right_euler,
#         )

#         left_pos_action, left_rot_action = self._normalize_vr_pose_to_action(
#             pos=left_pos,
#             euler=left_euler,
#             pos_anchor=self.left_pos_anchor,
#             euler_anchor=self.left_euler_anchor,
#         )

#         right_pos_action, right_rot_action = self._normalize_vr_pose_to_action(
#             pos=right_pos,
#             euler=right_euler,
#             pos_anchor=self.right_pos_anchor,
#             euler_anchor=self.right_euler_anchor,
#         )

#         if self.gripper_enabled:
#             if msg.left_gripper_close:
#                 left_grip = np.array([-1.0], dtype=np.float32)
#             else:
#                 left_grip = np.array([1.0], dtype=np.float32)

#             if msg.right_gripper_close:
#                 right_grip = np.array([-1.0], dtype=np.float32)
#             else:
#                 right_grip = np.array([1.0], dtype=np.float32)
#         else:
#             left_grip = np.zeros(1, dtype=np.float32)
#             right_grip = np.zeros(1, dtype=np.float32)

#         raw_vr_action = np.concatenate(
#             [
#                 left_pos_action,
#                 left_rot_action,
#                 left_grip,
#                 right_pos_action,
#                 right_rot_action,
#                 right_grip,
#             ]
#         ).astype(np.float32)

#         if self.action_indices is not None:
#             filtered_action = np.zeros_like(raw_vr_action)
#             filtered_action[self.action_indices] = raw_vr_action[self.action_indices]
#             raw_vr_action = filtered_action

#         self.latest_vr_action = self._pack_action_for_env(raw_vr_action)

#     # ----------------------------------------------------------------------
#     # Gym ActionWrapper API
#     # ----------------------------------------------------------------------

#     def action(self, action: np.ndarray):
#         """
#         根据当前接管状态决定到底执行谁的动作。
#         """
#         if self.use_vr_mode:
#             return self.latest_vr_action.copy(), True

#         return action, False

#     def step(self, action):
#         new_action, is_intervened = self.action(action)

#         obs, rew, done, truncated, info = self.env.step(new_action)

#         if is_intervened:
#             info["intervene_action"] = new_action.copy()

#         info["left"] = is_intervened
#         info["right"] = is_intervened
#         info["left1"] = is_intervened
#         info["left2"] = False
#         info["right1"] = is_intervened
#         info["right2"] = False

#         return obs, rew, done, truncated, info


# import time
# from typing import List

# from gymnasium import Env, spaces
# import gymnasium as gym
# import numpy as np
# from gymnasium.spaces import Box
# from scipy.spatial.transform import Rotation as R
# import copy
# import requests

# # ROS 2 相关库 (用于星海图 VR 接管和硬件通信)
# import rclpy
# from rclpy.node import Node
# from teleoperation_msg_ros2.srv import SwitchControlModeVR
# # TODO: 实机部署前，务必取消注释并根据实际包名导入星海图的 VRPose 消息类型
# from teleoperation_msg_ros2.msg import VrPose

# # ============================================================================
# # 第一部分：通用功能与辅助 Wrappers (官方提供)
# # 作用：通过强化学习的 "Wrapper" 机制，对底层基础环境进行层层包装，
# #       实现状态净化、打分自动化和异常动作惩罚，而无需修改基础环境的源码。
# # ============================================================================

# # 辅助数学函数：用于将神经网络的输出 Logit 映射到 (0, 1) 区间，方便做概率判断
# sigmoid = lambda x: 1 / (1 + np.exp(-x))

# class HumanClassifierWrapper(gym.Wrapper):
#     """
#     [开发/调试专用] 手动打分器 Wrapper
    
#     工作原理：
#     当环境报告一个回合 (Episode) 结束时 (done=True)，此包装器会阻塞主线程，
#     并在终端弹出输入框，要求操作员手动输入 1 (成功) 或 0 (失败)。
    
#     使用场景：
#     在视觉奖励模型 (Reward Classifier) 尚未训练好，或者需要排查视觉模型 Bug 时，
#     使用此包装器进行半自动化的强化学习训练。
#     """
#     def __init__(self, env):
#         super().__init__(env)
    
#     def step(self, action):
#         # 1. 先让底层环境执行动作
#         obs, rew, done, truncated, info = self.env.step(action)
        
#         # 2. 如果回合结束，强制人工打分
#         if done:
#             while True:
#                 try:
#                     rew = int(input("Success? (1/0): "))
#                     # 确保输入合法
#                     assert rew == 0 or rew == 1
#                     break
#                 except:
#                     print("Invalid input. Please enter 1 for success, 0 for failure.")
#                     continue
                    
#         # 3. 将打分结果注入 info 字典，供外层日志记录或算法读取
#         info['succeed'] = rew
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info


# class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     [正式训练主力] 视觉二分类自动打分器 Wrapper
    
#     工作原理：
#     每一帧动作执行后，提取当前画面的 Observation，丢给预先训练好的 ResNet (或类似视觉模型) 打分。
#     还可以通过 target_hz 参数，强制控制环境的运行频率（例如让真实物理世界保持 20Hz 运行）。
#     """
#     def __init__(self, env: Env, reward_classifier_func, target_hz = None):
#         super().__init__(env)
#         # 传入的通常是一个加载了权重的神经网络推断函数
#         self.reward_classifier_func = reward_classifier_func
#         # 期望的运行频率（赫兹）
#         self.target_hz = target_hz

#     def compute_reward(self, obs):
#         """调用神经网络计算奖励分数"""
#         if self.reward_classifier_func is not None:
#             return self.reward_classifier_func(obs)
#         return 0

#     def step(self, action):
#         start_time = time.time()
        
#         # 1. 底层环境执行动作
#         obs, rew, done, truncated, info = self.env.step(action)
        
#         # 2. 视觉模型基于最新状态打分
#         rew = int(self.compute_reward(obs))
#         success = bool(rew)

#         done = bool(done or success)

#         info["succeed"] = success
#         info["success"] = success
#         info["is_success"] = success
        
#         # 4. 频率控制锁：如果执行太快，强制 sleep 补足时间差，确保控制频率恒定
#         if self.target_hz is not None:
#             time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         info['succeed'] = False
#         info["success"] = False
#         info["is_success"] = False
#         return obs, info


# class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     [复杂任务专用] 多阶段视觉打分器 Wrapper
    
#     工作原理：
#     如果一个任务包含多个子阶段（例如：1. 抓起插头 -> 2. 对准插座 -> 3. 插入），
#     系统会传入多个分类器。只有当前一个阶段达成后，才开始计算下一个阶段。
#     """
#     def __init__(self, env: Env, reward_classifier_func: List[callable]):
#         super().__init__(env)
#         self.reward_classifier_func = reward_classifier_func
#         # 记录每个子任务是否已经完成的状态数组
#         self.received = [False] * len(reward_classifier_func)
    
#     def compute_reward(self, obs):
#         """遍历所有分类器，累加奖励"""
#         rewards = [0] * len(self.reward_classifier_func)
#         for i, classifier_func in enumerate(self.reward_classifier_func):
#             # 如果该阶段已经完成过，不再重复给分
#             if self.received[i]:
#                 continue
                
#             # 获取模型输出的 Logit 值
#             logit = classifier_func(obs).item()
            
#             # 阈值判定：置信度大于 75% 才算成功
#             if sigmoid(logit) >= 0.75:
#                 self.received[i] = True
#                 rewards[i] = 1
                
#         # 返回这一帧获得的新增奖励总和
#         reward = sum(rewards)
#         return reward

#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)
#         rew = self.compute_reward(obs)
#         # 当所有子任务都达成时，整个大任务才算完成
#         done = (done or all(self.received)) 
#         info['succeed'] = all(self.received)
#         return obs, rew, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         # 重置所有子任务的完成状态
#         self.received = [False] * len(self.reward_classifier_func)
#         info['succeed'] = False
#         return obs, info


# # ----------------------------------------------------------------------------
# # 状态转换类 Wrapper (ObservationWrapper)
# # 作用：拦截环境的输出数据，修改后再交给外层的神经网络，降低模型的学习难度。
# # ----------------------------------------------------------------------------

# class Quat2EulerWrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转欧拉角"""
#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)
#         # 修改向外暴露的状态空间定义，从 7维 (xyz+quat) 改为 6维 (xyz+euler)
#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
#         )
#         return observation


# class Quat2R2Wrapper(gym.ObservationWrapper):
#     """单臂环境：四元数转旋转矩阵 (某些算法对旋转矩阵的连续性更敏感)"""
#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["tcp_pose"].shape == (7,)
#         self.observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(9,)
#         )

#     def observation(self, observation):
#         tcp_pose = observation["state"]["tcp_pose"]
#         r = R.from_quat(tcp_pose[3:]).as_matrix()
#         # 展平矩阵并拼接
#         observation["state"]["tcp_pose"] = np.concatenate(
#             (tcp_pose[:3], r[..., :2].flatten())
#         )
#         return observation


# class DualQuat2EulerWrapper(gym.ObservationWrapper):
#     """
#     【双臂环境核心】双臂四元数转欧拉角 Wrapper
    
#     重要性：神经网络很难直接预测和拟合四元数（因为四元数有模长为1的约束）。
#     必须用此 Wrapper 将状态转换为欧拉角，并且必须确保 Action 空间也使用欧拉角，
#     做到输入输出格式一致，模型才能有效收敛。
#     """
#     def __init__(self, env: Env):
#         super().__init__(env)
#         assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
#         assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        
#         self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )
#         self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )

#     def observation(self, observation):
#         # 转换左臂
#         tcp_pose_l = observation["state"]["left/tcp_pose"]
#         observation["state"]["left/tcp_pose"] = np.concatenate(
#             (tcp_pose_l[:3], R.from_quat(tcp_pose_l[3:]).as_euler("xyz"))
#         )
#         # 转换右臂
#         tcp_pose_r = observation["state"]["right/tcp_pose"]
#         observation["state"]["right/tcp_pose"] = np.concatenate(
#             (tcp_pose_r[:3], R.from_quat(tcp_pose_r[3:]).as_euler("xyz"))
#         )
#         return observation
    
#     def reset(self, **kwargs):
#         # 重置环境时，同样需要对初始观测进行四元数转换
#         obs, info = self.env.reset(**kwargs)
#         return self.observation(obs), info


# # ----------------------------------------------------------------------------
# # 动作修改类 Wrapper (ActionWrapper / RewardWrapper)
# # 作用：拦截外层传来的动作指令，或者根据不良动作施加惩罚。
# # ----------------------------------------------------------------------------

# class GripperCloseEnv(gym.ActionWrapper):
#     """
#     强制闭合夹爪 Wrapper
#     如果你的任务（如：用手背推开障碍物）根本不需要用到夹爪开合，
#     加上这个 Wrapper 可以强行把夹爪指令归零，减小模型的探索空间。
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
#         # 如果存在干预记录，也要把多余的夹爪维度切掉
#         if "intervene_action" in info:
#             info["intervene_action"] = info["intervene_action"][:6]
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


# class GripperPenaltyWrapper(gym.RewardWrapper):
#     """单臂：夹爪状态翻转惩罚器"""
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
#         # 如果模型发出错误指令导致夹爪抽搐，扣除惩罚分
#         if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
#             action[6] > 0.5 and self.last_gripper_pos < 0.95
#         ):
#             return reward - self.penalty
#         else:
#             return reward

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         if "intervene_action" in info:
#             action = info["intervene_action"]
#         reward = self.reward(reward, action)
#         self.last_gripper_pos = observation["state"][0, 0]
#         return observation, reward, terminated, truncated, info


# class DualGripperPenaltyWrapper(gym.RewardWrapper):
#     """
#     【必须使用】双臂夹爪动作惩罚 Wrapper
    
#     工作原理：
#     强化学习模型在早期探索时，很容易输出高频交替的夹爪张合指令（抽搐）。
#     这不仅会破坏抓取任务，还极易导致物理电机关机甚至烧毁。
#     此 Wrapper 通过比对前后帧状态，如果发现模型在做无效的开关动作，
#     就从总奖励中扣除 penalty 分数，强迫模型学会“保持夹爪静止”。
#     """
#     def __init__(self, env, penalty=0.1):
#         super().__init__(env)
#         # 确认是在处理 14 维的双臂动作
#         assert env.action_space.shape == (14,)
#         self.penalty = penalty
#         # 状态位：0 代表张开，1 代表闭合
#         self.last_gripper_pos_left = 0 
#         self.last_gripper_pos_right = 0 
    
#     def reward(self, reward: float, action) -> float:
#         # 检查左夹爪 (索引 6) 是否发生状态翻转
#         if (action[6] < -0.5 and self.last_gripper_pos_left==0):
#             reward -= self.penalty
#             self.last_gripper_pos_left = 1
#         elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
#             reward -= self.penalty
#             self.last_gripper_pos_left = 0
            
#         # 检查右夹爪 (索引 13) 是否发生状态翻转
#         if (action[13] < -0.5 and self.last_gripper_pos_right==0):
#             reward -= self.penalty
#             self.last_gripper_pos_right = 1
#         elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
#             reward -= self.penalty
#             self.last_gripper_pos_right = 0
            
#         return reward
    
#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         # 如果这一帧人类专家进行了接管，以人类的实际动作为准计算惩罚
#         if "intervene_action" in info:
#             action = info["intervene_action"]
        
#         # 将扣分后的奖励返回给外层
#         reward = self.reward(reward, action)
#         return observation, reward, terminated, truncated, info


# # ============================================================================
# # 第二部分：星海图 R1 PRO 专属定制 (物理离合接管 Wrapper)
# # ============================================================================

# class VRInterventionWrapper(gym.ActionWrapper):
#     """
#     星海图 R1 PRO VR 物理离合接管 Wrapper
#     需求语义：
#     - control_mode == 0 -> VR 接管 -> use_vr_mode = True
#     - control_mode == 2 -> 脚本/AI 接管 -> use_vr_mode = False
#     - 只在模式发生变化时发送一次 service
#     """

#     def __init__(self, env, action_indices=None, gripper_enabled=True):
#         super().__init__(env)

#         # 当前是否由 VR 接管
#         self.use_vr_mode = False

#         # 上一次看到的 control_mode，用于“边沿触发”
#         self.last_control_mode = None

#         # 上一次已经发送过的 service 目标，防止重复发
#         self.last_service_target = None

#         self.action_indices = action_indices
#         self.gripper_enabled = gripper_enabled

#         # 14/7 维动作缓存
#         # 动作缓存长度跟底层 env 当前 action_space 对齐
#         self.latest_vr_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

#         # 复用底层 bridge 的 node
#         try:
#             self.node = self.env.unwrapped.bridge.node
#         except AttributeError:
#             if not rclpy.ok():
#                 rclpy.init()
#             self.node = rclpy.create_node("vr_intervention_wrapper_node")

#         # 持久化 service client
#         self.mode_client = self.node.create_client(
#             SwitchControlModeVR,
#             "/switch_control_mode_vr"
#         )

#         print("⏳ 正在等待底层硬件服务连接 (/switch_control_mode_vr)...")
#         if not self.mode_client.wait_for_service(timeout_sec=5.0):
#             raise RuntimeError("❌ /switch_control_mode_vr 服务未上线，无法建立 VR 模式切换通道")
#         print("✅ 服务已连接！持久化通讯通道已建立，VR 物理离合器准备就绪。")

#         # 订阅 VR 话题
#         self.vr_sub = self.node.create_subscription(
#             VrPose,
#             "/vr_pose",
#             self._vr_cb,
#             10
#         )

#     def _on_switch_done(self, future):
#         try:
#             resp = future.result()
#             if resp is not None:
#                 print(f"✅ 模式切换返回: success={resp.success}, message={resp.message}")
#         except Exception as e:
#             print(f"❌ 模式切换 service 调用失败: {e}")

#     def _call_switch_service(self, use_vr_mode: bool):
#         """
#         异步调用模式切换服务
#         use_vr_mode=True  -> 切到 VR 控制
#         use_vr_mode=False -> 切到脚本/IK 控制
#         """
#         # 防止重复发同一个目标状态
#         if self.last_service_target == use_vr_mode:
#             return

#         req = SwitchControlModeVR.Request()
#         req.use_vr_mode = use_vr_mode

#         future = self.mode_client.call_async(req)
#         future.add_done_callback(self._on_switch_done)

#         self.last_service_target = use_vr_mode

#     def _pack_action_for_env(self, full_action_14: np.ndarray) -> np.ndarray:
#         """
#         把 VR 原始双臂 14 维动作，打包成当前底层 env 真正需要的动作维度。
#         dual  -> 保持 14 维
#         single-left  -> 取前 7 维
#         single-right -> 取后 7 维
#         """
#         full_action_14 = np.asarray(full_action_14, dtype=np.float32).reshape(-1)

#         if full_action_14.shape[0] != 14:
#             raise ValueError(f"VR 原始动作必须是 14 维，当前收到 {full_action_14.shape}")

#         base_env = self.env.unwrapped
#         arm_mode = getattr(base_env, "arm_mode", "dual")
#         arm_side = getattr(base_env, "arm_side", "right")

#         if arm_mode == "dual":
#             return full_action_14

#         if arm_side == "left":
#            return full_action_14[:7]

#         return full_action_14[7:14]

#     def _vr_cb(self, msg):
#         """
#         高频回调：
#         - 监听 control_mode
#         - 只在 0/2 发生变化时发一次 service
#         - 若当前为 VR 接管，则同步更新 latest_vr_action
#         """
#         current_mode = int(msg.control_mode)

#         # 只关心 0 和 2
#         if current_mode not in (0, 2):
#             return

#         # 当前系统状态始终和 mode 对齐
#         # mode 0 -> VR 接管
#         # mode 2 -> 脚本/AI 接管
#         self.use_vr_mode = (current_mode == 0)

#         # mode 没变，不重复发 service
#         if current_mode == self.last_control_mode:
#             pass
#         else:
#             self.last_control_mode = current_mode

#             if current_mode == 0:
#                self._call_switch_service(True)
#                print("🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True")
#                if hasattr(self.env.unwrapped, "notify_script_control"):
#                    self.env.unwrapped.notify_script_control(False)

#             elif current_mode == 2:
#                 self._call_switch_service(False)
#                 print("🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False")
#                 if hasattr(self.env.unwrapped, "notify_script_control"):
#                     self.env.unwrapped.notify_script_control(True)
#         # 只有 VR 接管时，才解析 VR 位姿为动作
#         if self.use_vr_mode:
#             left_pos = np.array(msg.left_position, dtype=np.float32)
#             right_pos = np.array(msg.right_position, dtype=np.float32)

#             left_euler = R.from_quat(msg.left_quaternion).as_euler("xyz")
#             right_euler = R.from_quat(msg.right_quaternion).as_euler("xyz")

#             if self.gripper_enabled:
#                 if msg.left_gripper_close:
#                     left_grip = np.random.uniform(-1.0, -0.9, size=(1,)).astype(np.float32)
#                 else:
#                     left_grip = np.random.uniform(0.9, 1.0, size=(1,)).astype(np.float32)

#                 if msg.right_gripper_close:
#                     right_grip = np.random.uniform(-1.0, -0.9, size=(1,)).astype(np.float32)
#                 else:
#                     right_grip = np.random.uniform(0.9, 1.0, size=(1,)).astype(np.float32)
#             else:
#                 left_grip = np.zeros(1, dtype=np.float32)
#                 right_grip = np.zeros(1, dtype=np.float32)

#             raw_vr_action = np.concatenate([
#                 left_pos, left_euler, left_grip,
#                 right_pos, right_euler, right_grip
#             ]).astype(np.float32)

#             if self.action_indices is not None:
#                 filtered_action = np.zeros_like(raw_vr_action)
#                 filtered_action[self.action_indices] = raw_vr_action[self.action_indices]
#                 raw_vr_action = filtered_action

#             self.latest_vr_action = self._pack_action_for_env(raw_vr_action)

#     def action(self, action: np.ndarray):
#         """
#         根据当前接管状态决定到底执行谁的动作
#         """
#         # 注意：这里不要再 rclpy.spin_once(self.node)
#         # 因为 bridge 已经在后台 executor 线程里 spin 了

#         if self.use_vr_mode:
#             return self.latest_vr_action, True
#         else:
#             return action, False

#     def step(self, action):
#         new_action, is_intervened = self.action(action)
#         obs, rew, done, truncated, info = self.env.step(new_action)

#         if is_intervened:
#             info["intervene_action"] = new_action

#         info["left"] = is_intervened
#         info["right"] = is_intervened
#         info["left1"] = is_intervened
#         info["left2"] = False
#         info["right1"] = is_intervened
#         info["right2"] = False

#         return obs, rew, done, truncated, info


