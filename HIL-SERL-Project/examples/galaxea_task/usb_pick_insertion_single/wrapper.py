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

from serl_robot_infra.Galaxea_env.envs.galaxea_arm_env import GalaxeaArmEnv


class GalaxeaUSBEnv(GalaxeaArmEnv):
    """
    星海图 USB 插拔单臂任务环境（当前按右臂单臂配置）。

    设计原则：
    1) 通用 GalaxeaArmEnv 不再持有任务级 reset 逻辑
    2) 具体任务在 config + wrapper 里定义：
       - RESET_POSE
       - RESET_GRIPPER
       - RESET_TIMEOUT_SEC
       - RANDOM_RESET / RANDOM_XY_RANGE
    3) reset 时夹爪是否张开、张开多少，由任务 config 决定
    """

    def __init__(self, config=None, use_vr=True, **kwargs):
        if config is None:
            raise ValueError("GalaxeaUSBEnv 初始化失败：必须传入有效的 config")

        self.config = config
        self.use_vr = use_vr

        # ==========================================================
        # 任务级 reset 配置：全部从 config 读取
        # ==========================================================
        if not hasattr(config, "RESET_POSE"):
            raise AttributeError("GalaxeaUSBEnv 初始化失败：config 缺少 RESET_POSE")
        self.reset_pose = np.array(config.RESET_POSE, dtype=np.float32)

        self.reset_gripper = float(getattr(config, "RESET_GRIPPER", 80.0))
        self.reset_timeout_sec = float(getattr(config, "RESET_TIMEOUT_SEC", 3.0))

        self.random_reset = bool(getattr(config, "RANDOM_RESET", False))
        self.random_xy_range = float(getattr(config, "RANDOM_XY_RANGE", 0.0))

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

    def _build_reset_target(self):
        """
        任务级 reset 目标位姿。
        目前保留 XY 随机扰动。
        后续可增加朝向扰动
        """
        reset_pose = self.reset_pose.copy()

        if self.random_reset and self.random_xy_range > 0:
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range,
                self.random_xy_range,
                (2,),
            )

        return reset_pose

    def go_to_reset(self):
        """
        两种模式：
        1) use_vr=True：保留 VR 模式切换复位逻辑
        2) use_vr=False：直接发送 ROS 复位轨迹，不等 VR

        注意：
        - reset 位姿、夹爪值、超时时长都由任务 config 决定
        - 通用 env 不再参与任务 reset 决策
        """
        print("🤖 [USB Task Single Arm] 正在准备复位...")

        reset_pose = self._build_reset_target()

        if self.use_vr:
            print("💡 【请按 VR 手柄的 Mode 2 键】切到脚本控制模式（会发送 use_vr_mode=False）")
            self._wait_until_script_control_ready(timeout=15.0)
            self._wait_extra_after_false(delay=2.0)
            print("🤖 [USB Task Single Arm] 开始向底层发送复位坐标...")
        else:
            print("🤖 [USB Task Single Arm] 当前为无 VR 模式，直接发送复位轨迹...")

        # reset 夹爪值现在完全由任务 config 控制
        self.interpolate_move_single(
            reset_pose,
            timeout=self.reset_timeout_sec,
            gripper=self.reset_gripper,
        )
        time.sleep(0.5)

        print(
            f"✅ 单臂复位完成！"
            f" pose={np.round(reset_pose, 4).tolist()}, "
            f"gripper={self.reset_gripper}, timeout={self.reset_timeout_sec}"
        )

        if self.use_vr:
            print("💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！")

    
    # def go_to_reset(self):
    #     """
    #     reset 新逻辑（固定目标重复发布版）：
    #     1) use_vr=True 时，先等手柄切到 Mode 2
    #        - VRInterventionWrapper 会在 control_mode=2 时发送 use_vr_mode=False
    #     2) 不再做插值轨迹
    #     3) 连续 2 秒重复发布同一个 reset_pose + reset_gripper
    #        - 目的：像你手动 ros2 topic pub --rate 30 那样，稳定压住 target
    #       4) use_vr=False 时，直接执行同样的固定目标重复发布
    #     """
    #     print("🤖 [USB Task Single Arm] 正在准备复位...")

    #     reset_pose = self._build_reset_target()

    # # 你要求：固定发送 2 秒
    #     hold_sec = 5.0

    #     if self.use_vr:
    #         print("💡 【请按 VR 手柄的 Mode 2 键】切到脚本控制模式（会发送 use_vr_mode=False）")
    #         self._wait_until_script_control_ready(timeout=15.0)

    #     # 保留这一步，让模式切换完全稳定下来
    #         self._wait_extra_after_false(delay=1.0)

    #         print("🤖 [USB Task Single Arm] 开始向底层连续发送固定复位目标...")
    #     else:
    #         print("🤖 [USB Task Single Arm] 当前为无 VR 模式，直接连续发送固定复位目标...")

    # # ===== 固定目标重复发布版：不插值 =====
    # # 尽量贴近你手动 `ros2 topic pub --rate 30 ...` 的思路
    #     start_t = time.time()
    #     send_count = 0
    #     period = 1.0 / float(self.hz)
   
    #     while time.time() - start_t < hold_sec:
    #         step_start = time.time()

    #         self._send_ros_pose(
    #             self.arm_side,
    #             reset_pose,
    #             self.reset_gripper,
    #         )
    #         send_count += 1

    #         dt = time.time() - step_start
    #         time.sleep(max(0.0, period - dt))

    # # 再补发一次，确保最后一个 target 是你想要的 reset_pose
    #     self._send_ros_pose(
    #         self.arm_side,
    #         reset_pose,
    #         self.reset_gripper,
    #     )

    # # 同步一次观测，避免状态滞后
    #     self._get_sync_obs()

    #     print(
    #         f"✅ 单臂固定复位目标发送完成！"
    #         f" pose={np.round(reset_pose, 4).tolist()}, "
    #         f"gripper={self.reset_gripper}, "
    #         f"hold_sec={hold_sec}, send_count={send_count + 1}"
    #     )

    #     if self.use_vr:
    #         print("💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！")    



class SingleGripperPenaltyWrapper(gym.Wrapper):
    """
    单臂夹爪惩罚 wrapper。

    兼容两种 gripper 语义：
    1) 三值/标签空间：-1 / 0 / +1
    2) 硬件空间：0~30 视为闭合，70~100 视为张开
    """

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
        self.gripper_closed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_closed = None
        return obs, info

    def _canonicalize_cmd(self, cmd: float) -> float:
        """
        统一把输入动作转成“标签语义”：
        - 硬件空间：0~30 -> -1, 70~100 -> +1
        - 三值空间：原样保留
        - 中间区：通常视为 0 / hold
        """
        cmd = float(cmd)

        # 硬件量程空间
        if 0.0 <= cmd <= 30.0:
            return -1.0
        if 70.0 <= cmd <= 100.0:
            return 1.0

        # 三值/标签空间（通常本来就在 [-1, 1]）
        return cmd

    def _update_one_side(self, cmd, prev_closed):
        penalty_delta = 0.0
        new_closed = prev_closed

        cmd = self._canonicalize_cmd(cmd)

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

        assert real_action.shape[0] == 7, f"动作维度异常，期望 7，实际是 {real_action.shape}"

        penalty_val, self.gripper_closed = self._update_one_side(
            real_action[6], self.gripper_closed
        )

        reward += penalty_val
        info["grasp_penalty"] = penalty_val

        return obs, reward, terminated, truncated, info