

import numpy as np
import time
import gymnasium as gym

from serl_robot_infra.Galaxea_env.envs.dual_galaxea_env import GalaxeaDualArmEnv
from serl_robot_infra.Galaxea_env.envs.wrappers import VRInterventionWrapper


class GalaxeaUSBEnv(GalaxeaDualArmEnv):
    """星海图双臂 U 盘插拔专属环境"""

    def __init__(self, config=None, **kwargs):
        self.config = config

        self.reset_l = getattr(
            config,
            "RESET_L",
            np.array([0.2, -0.25, -0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.reset_r = getattr(
            config,
            "RESET_R",
            np.array([0.2, 0.25, -0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

        # 由 VRInterventionWrapper 在 mode=2 / mode=0 时更新
        self.script_control_enabled = False
        self.script_control_switch_time = None

        super().__init__(config=config, **kwargs)

    def notify_script_control(self, enabled: bool):
        """
        由 VRInterventionWrapper 回调：
        mode=2 -> enabled=True  （脚本/IK 接管）
        mode=0 -> enabled=False （VR 接管）
        """
        self.script_control_enabled = enabled
        self.script_control_switch_time = time.time() if enabled else None

    def _wait_until_script_control_ready(self, timeout=15.0):
        """
        等待用户把手柄切到 Mode 2，直到 wrapper 已经发出 use_vr_mode=False。
        """
        start = time.time()
        while not self.script_control_enabled:
            if time.time() - start > timeout:
                raise TimeoutError("等待 Mode 2 超时：一直没有进入脚本控制模式(use_vr_mode=False)")
            time.sleep(0.05)

    def _wait_extra_after_false(self, delay=2.0):
        """
        当输出 false 后，再等 2 秒，然后开始发复位轨迹。
        """
        if self.script_control_switch_time is None:
            time.sleep(delay)
            return

        elapsed = time.time() - self.script_control_switch_time
        remain = delay - elapsed
        if remain > 0:
            print(f"⏳ 已进入脚本控制，额外等待 {remain:.2f}s 后开始复位...")
            time.sleep(remain)

    def go_to_reset(self):
        """
        复位逻辑：
        1. 提示用户切到 Mode 2
        2. 等待 wrapper 检测到 mode=2，并发出 use_vr_mode=False
        3. false 发出后，额外等待 2 秒
        4. 再开始发送复位轨迹
        """
        print("🤖 [USB Task] 正在准备复位...")
        print("💡 【请按 VR 手柄的 Mode 2 键】切到脚本控制模式（会发送 use_vr_mode=False）")

        self._wait_until_script_control_ready(timeout=15.0)
        self._wait_extra_after_false(delay=2.0)

        print("🤖 [USB Task] 开始向底层发送复位坐标...")

        reset_l = self.reset_l.copy()
        reset_r = self.reset_r.copy()

        if hasattr(self, "config") and getattr(self.config, "RANDOM_RESET", False):
            xy_range = getattr(self.config, "RANDOM_XY_RANGE", 0.01)
            reset_l[:2] += np.random.uniform(-xy_range, xy_range, (2,))
            reset_r[:2] += np.random.uniform(-xy_range, xy_range, (2,))

        self.interpolate_move(reset_l, reset_r, timeout=3.0)
        time.sleep(0.5)

        print("✅ 复位坐标发送完毕！")
        print("💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！")


class DualGripperPenaltyWrapper(gym.Wrapper):
    """
    双臂夹爪惩罚器（动作版）
    不再依赖 obs['state']['left_gripper'] 这种原始字典结构，
    避免被 SERLObsWrapper / ChunkingWrapper 搞崩。
    """

    def __init__(self, env, penalty=-0.02, close_thr=-0.5, open_thr=0.5):
        super().__init__(env)
        self.penalty = penalty
        self.close_thr = close_thr
        self.open_thr = open_thr
        self.left_closed = None
        self.right_closed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.left_closed = None
        self.right_closed = None
        return obs, info

    def _update_one_side(self, cmd, prev_closed):
        """
        根据动作命令推断夹爪状态，并决定是否罚分。
        返回:
          penalty_delta, new_closed
        """
        penalty_delta = 0.0
        new_closed = prev_closed

        if cmd < self.close_thr:
            if prev_closed is False:
                penalty_delta += self.penalty
            new_closed = True

        elif cmd > self.open_thr:
            if prev_closed is True:
                penalty_delta += self.penalty
            new_closed = False

        return penalty_delta, new_closed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        real_action = info.get("intervene_action", action)
        real_action = np.asarray(real_action, dtype=np.float32)

        penalty_val = 0.0
        left_delta, self.left_closed = self._update_one_side(real_action[6], self.left_closed)
        right_delta, self.right_closed = self._update_one_side(real_action[13], self.right_closed)

        penalty_val += left_delta + right_delta
        reward += penalty_val

        info["grasp_penalty"] = penalty_val
        return obs, reward, terminated, truncated, info


def make_env(reward_classifier_model=None, use_manual_reward=False):
    from examples.galaxea_task.usb_pick_insertion.config import env_config

    env = env_config.get_environment(fake_env=False, classifier=False)
    env = VRInterventionWrapper(env)
    env = DualGripperPenaltyWrapper(env, penalty=-0.02)
    return env


