# 官方 wrapper.py 是“任务实体层”

# 你贴的官方 USBEnv(FrankaEnv) 里，放的都是和 USB 插拔这个任务本身强相关的东西，比如：

# init_cameras()：这个任务用哪些相机、怎么初始化
# reset()：这个任务开始前机械臂怎么走、夹爪怎么开合、怎么靠近目标
# interpolate_move()：这个任务用什么形式发运动命令
# go_to_reset()：这个任务的复位步骤
# GripperPenaltyWrapper：这个任务特有的惩罚逻辑

# 也就是说，官方 wrapper.py 的职责是：

# “这个任务在物理世界里到底怎么做”

# 不是训练参数，不是任务路由，而是任务动作和任务复位本身。

# 这个文件应该只保留“任务实体逻辑”：

# GalaxeaUSBEnv
# go_to_reset()
# 任务相关 reset() 特殊流程
# DualGripperPenaltyWrapper

# 不要让它再承担“总装环境”的职责。
# 也就是说，make_env() 这类函数最好最终不要在这里继续扩张。


import time
import numpy as np
import gymnasium as gym

from serl_robot_infra.Galaxea_env.envs.dual_galaxea_env import GalaxeaDualArmEnv


class GalaxeaUSBEnv(GalaxeaDualArmEnv):
    """星海图双臂 U 盘插拔专属环境"""

    def __init__(self, config=None, use_vr=True, **kwargs):
        if config is None:
            raise ValueError("GalaxeaUSBEnv 初始化失败：必须传入有效的 config")

        self.config = config
        self.use_vr = use_vr

        self.reset_l = np.array(config.RESET_L, dtype=np.float32)
        self.reset_r = np.array(config.RESET_R, dtype=np.float32)

        # 只有 use_vr=True 时才会用到这些状态
        self.script_control_enabled = False
        self.script_control_switch_time = None

        super().__init__(config=config, **kwargs)

    def notify_script_control(self, enabled: bool):
        """
        由 VRInterventionWrapper 回调：
        mode=2 -> enabled=True
        mode=0 -> enabled=False
        """
        self.script_control_enabled = enabled
        self.script_control_switch_time = time.time() if enabled else None

    def _wait_until_script_control_ready(self, timeout: float = 15.0):
        start = time.time()
        while not self.script_control_enabled:
            if time.time() - start > timeout:
                raise TimeoutError(
                    "等待 Mode 2 超时：一直没有进入脚本控制模式(use_vr_mode=False)"
                )
            time.sleep(0.05)

    def _wait_extra_after_false(self, delay: float = 2.0):
        if self.script_control_switch_time is None:
            time.sleep(delay)
            return

        elapsed = time.time() - self.script_control_switch_time
        remain = delay - elapsed
        if remain > 0:
            print(f"⏳ 已进入脚本控制，额外等待 {remain:.2f}s 后开始复位...")
            time.sleep(remain)

    def _build_reset_targets(self):
        reset_l = self.reset_l.copy()
        reset_r = self.reset_r.copy()

        if self.config.RANDOM_RESET:
            xy_range = float(self.config.RANDOM_XY_RANGE)
            reset_l[:2] += np.random.uniform(-xy_range, xy_range, (2,))
            reset_r[:2] += np.random.uniform(-xy_range, xy_range, (2,))

        return reset_l, reset_r

    def go_to_reset(self):
        """
        两种模式：
        1) use_vr=True：保留原来的 VR 模式切换复位逻辑
        2) use_vr=False：直接发送 ROS 复位轨迹，不等 VR
        """
        print("🤖 [USB Task] 正在准备复位...")

        reset_l, reset_r = self._build_reset_targets()

        if self.use_vr:
            print("💡 【请按 VR 手柄的 Mode 2 键】切到脚本控制模式（会发送 use_vr_mode=False）")
            self._wait_until_script_control_ready(timeout=15.0)
            self._wait_extra_after_false(delay=2.0)
            print("🤖 [USB Task] 开始向底层发送复位坐标...")
        else:
            print("🤖 [USB Task] 当前为无 VR 模式，直接发送复位轨迹...")

        self.interpolate_move(reset_l, reset_r, timeout=3.0)
        time.sleep(0.5)

        print("✅ 复位坐标发送完毕！")

        if self.use_vr:
            print("💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！")


class DualGripperPenaltyWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        penalty: float = -0.02,
        close_thr: float = -0.5,
        open_thr: float = 0.5,
    ):
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
        
        #默认14纬度，所以6,13对应夹爪，如果机器人变了，就不是了
        assert real_action.shape[0] >= 14, f"动作维度异常，期望至少 14，实际是 {real_action.shape}"

        penalty_val = 0.0
        left_delta, self.left_closed = self._update_one_side(real_action[6], self.left_closed)
        right_delta, self.right_closed = self._update_one_side(real_action[13], self.right_closed)

        penalty_val += left_delta + right_delta
        reward += penalty_val
        info["grasp_penalty"] = penalty_val

        return obs, reward, terminated, truncated, info