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

    
   



class SingleGripperPenaltyWrapper(gym.Wrapper):
    """
    单臂夹爪惩罚 wrapper，尽量对齐 HIL-SERL 官方 learned-gripper 设计。

    训练动作语义：
      action[6] = -1  -> close event
      action[6] =  0  -> hold / no-op，不改变夹爪状态
      action[6] = +1  -> open event

    硬件命令兼容：
      10 / 0~30   -> close event
      80 / 70~100 -> open event
      中间硬件值  -> hold / unclear，不触发开关事件

    惩罚语义：
      - 只惩罚夹爪状态切换事件；
      - 不直接修改环境 reward；
      - 只写入 info["grasp_penalty"]；
      - 后续由 SACAgentHybridSingleArm 的 grasp_critic/DQN 使用
        batch["rewards"] + batch["grasp_penalty"] 训练夹爪分支。

    注意：
      这里的 penalty 不是说 close/open 错了，
      而是给夹爪开合一个小动作代价，防止频繁抖动。
    """

    def __init__(
        self,
        env,
        penalty: float = -0.02,
        close_thr: float = -0.5,
        open_thr: float = 0.5,
        hw_close_max: float = 30.0,
        hw_open_min: float = 70.0,
    ):
        super().__init__(env)
        self.penalty = float(penalty)
        self.close_thr = float(close_thr)
        self.open_thr = float(open_thr)
        self.hw_close_max = float(hw_close_max)
        self.hw_open_min = float(hw_open_min)

        # True  = 当前真实夹爪稳定闭合
        # False = 当前真实夹爪稳定张开
        # None  = 不确定
        self.gripper_closed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 对齐官网：reset 后用真实 observation 初始化上一夹爪状态。
        self.gripper_closed = self._infer_closed_from_obs(obs, prev=None)

        return obs, info

    def _extract_gripper_feedback(self, obs):
        """
        从 observation 中提取真实 gripper feedback。
        当前你的单臂 state 通常是 shape=(1,8)：
          7 维 ee pose + 1 维 gripper
        所以最后一维就是夹爪反馈。
        """
        if obs is None or not isinstance(obs, dict):
            return None

        if "state" not in obs:
            return None

        state = obs["state"]

        if isinstance(state, dict):
            for key in [
                "right_gripper",
                "left_gripper",
                "gripper",
                "state/right_gripper",
                "state/left_gripper",
            ]:
                if key in state:
                    arr = np.asarray(state[key]).reshape(-1)
                    if arr.size > 0:
                        return float(arr[-1])

            for key, value in state.items():
                if "gripper" in str(key).lower():
                    arr = np.asarray(value).reshape(-1)
                    if arr.size > 0:
                        return float(arr[-1])

            return None

        arr = np.asarray(state).reshape(-1)
        if arr.size == 0:
            return None

        return float(arr[-1])

    def _infer_closed_from_obs(self, obs, prev=None):
        """
        用真实 gripper feedback 判断当前夹爪状态：
          feedback <= hw_close_max -> closed=True
          feedback >= hw_open_min  -> closed=False
          中间区                  -> 保持 prev，不确定时返回 prev
        """
        feedback = self._extract_gripper_feedback(obs)

        if feedback is None:
            return prev

        if feedback <= self.hw_close_max:
            return True

        if feedback >= self.hw_open_min:
            return False

        return prev

    def _canonicalize_action_cmd(self, cmd: float) -> float:
        """
        当前任务只接受训练/策略标签空间：
          -1 = close event
           0 = hold / no-op
          +1 = open event

        注意：
          cmd=0 是 hold 标签，不是夹爪硬件量程 0。
          硬件量程 10/80 只应该出现在最终 ROS 执行映射层。
        """
        cmd = float(cmd)

        if cmd < self.close_thr:
            return -1.0
        if cmd > self.open_thr:
            return 1.0
        return 0.0




    # def _canonicalize_action_cmd(self, cmd: float) -> float:
    #     """
    #     把 action[6] 统一转换成夹爪事件标签：
    #       -1 = close event
    #        0 = hold / no-op
    #       +1 = open event

    #     关键点：
    #       必须先判断三值空间，否则 0 会被误判成硬件 close。
    #     """
    #     cmd = float(cmd)

    #     # 1) 先处理三值/策略空间：-1 / 0 / +1
    #     #    这样 0 会正确保留为 hold。
    #     if -1.5 <= cmd <= 1.5:
    #         if cmd < self.close_thr:
    #             return -1.0
    #         if cmd > self.open_thr:
    #             return 1.0
    #         return 0.0

    #     # 2) 再处理 Galaxea 硬件量程命令，例如 10 / 80
    #     if 0.0 <= cmd <= self.hw_close_max:
    #         return -1.0

    #     if self.hw_open_min <= cmd <= 100.0:
    #         return 1.0

    #     # 3) 其他中间硬件值，视为 hold / unclear。
    #     #    注意：这不是让夹爪运动到中间值，只是不触发 penalty。
    #     return 0.0

    def _compute_penalty(self, cmd_label, prev_closed):
        """
        官网式夹爪 penalty：
          上一状态 open，当前 close -> 扣一次
          上一状态 closed，当前 open -> 扣一次
          hold -> 不扣
          prev_closed 不确定 -> 不扣
        """
        if prev_closed is None:
            return 0.0

        if cmd_label < self.close_thr:
            # 当前执行 close
            if prev_closed is False:
                return self.penalty
            return 0.0

        if cmd_label > self.open_thr:
            # 当前执行 open
            if prev_closed is True:
                return self.penalty
            return 0.0

        # hold / no-op
        return 0.0

    def step(self, action):
        # step 前保存上一真实夹爪状态
        prev_closed = self.gripper_closed

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 如果 VR 接管，则用真实执行的人类动作判断 penalty；
        # 否则用 policy action 判断。
        real_action = info.get("intervene_action", action)
        real_action = np.asarray(real_action, dtype=np.float32).reshape(-1)

        assert real_action.shape[0] == 7, (
            f"动作维度异常，期望 7，实际是 {real_action.shape}"
        )

        cmd_label = self._canonicalize_action_cmd(real_action[6])
        penalty_val = self._compute_penalty(cmd_label, prev_closed)

        # 对齐官网：不修改主 reward，只单独记录 grasp_penalty。
        info["grasp_penalty"] = float(penalty_val)

        # step 后用真实 observation 更新当前夹爪状态
        self.gripper_closed = self._infer_closed_from_obs(
            obs,
            prev=prev_closed,
        )

        return obs, reward, terminated, truncated, info