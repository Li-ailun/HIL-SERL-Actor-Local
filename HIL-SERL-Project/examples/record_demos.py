#录制专家轨迹，用于后续先bc再强化学习（人类在环监督或者无监督两种方式）

#该脚本的本质：（1）记录intervene_action（把供bc模仿学习的完美演示数据和共强化学习的人类介入数据都定义为intervene_action，这两个数据都保存仅专家池）
#           （2）所以即使这是录制脚本，也是依靠intervene_action信号记录数据的

#该脚本构成（高度模块化）
#（1）数据输入接口：无该内容，输入接口都在dual_galaxea_env.py里封装好了，只要dual_galaxea_env.py能正常收到需要的数据，则完成；
#（2）怎么录制：无该内容，录制控制都在wrappers.py里封装好了，此处录制的完整流程演示数据和wrappers.py的vr接管数据都被存入专家经验池（Demo Buffer））里
#（3）存放路径：存进 demo_data 文件夹
#（4）何时停止录制：
#       1,基于wrappers.py定义的mode0无人类介入，
                 #mode2人类介入的逻辑： 一旦切入 mode2，它就在这一帧的数据里挂上一个牌子（info["intervene_action"] = ...）。
                 #mode0时不记录，mode2后才记录
#       2,预设固定步长时间，超时自动停止录制
#       3,所以mode2提前结束任务，等待补偿结束，自动记录完美演示数据 

#总逻辑：
# dual_galaxea_env.py (底层)：负责和星海图的相机、机械臂打交道，把物理世界的声光电变成 NumPy 数组。
# wrappers.py (中层)：智能安检，负责监听你的 VR 手柄，把人类的动作替换进去，并贴上 intervene_action 的 VIP 标签。
# record_demos.py / train_rlpd.py (顶层)：数据分拣员。
#      只管看着 info 字典，看到有 VIP 标签的数据，就直接往专家经验池（Demo Buffer）里扔。
#      如果没有标签，就扔进在线探索池（Replay Buffer）。

# 默认 20 条成功 demo：
# python record_demos.py

# 限制每条最多 300 步：
# python record_demos.py --max_episode_steps=300

# 先少录几条测试：
# python record_demos.py --successes_needed=2 --max_episode_steps=200


#不过滤静止帧再保存demos（ros2 topic echo /motion_control/pose_ee_arm_right可以确认vr暂停后数值不会变化，所以可以视为静止帧 ）
#（可以回放录制的demos，看看动作是不是流畅的）

#录制的action【6】官方代表持续的张开/闭合命令，我们修改自己的情况，让其符合想闭合就显示一直闭合，想张开就一直张开的情况

##和官方一致
#不过滤静止帧，防止最后成功时被判定到静止帧范围从而demos不完整
#三值夹爪标签(可视化demos来判断是否准确实现）


import os
import sys
import time

# ==============================================================
# 像 actor 一样，先强制本地 CPU，避免 classifier=True 时本地环境炸
# ==============================================================
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

# ==============================================================
# 核心路径配置
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config


# ==============================================================
# 命令行参数配置
# ==============================================================
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "galaxea_usb_insertion_single",
    "Name of experiment corresponding to folder.",
)

flags.DEFINE_integer(
    "successes_needed",
    20,
    "Number of successful demos to collect.",
)

flags.DEFINE_integer(
    "max_episode_steps",
    400,
    "Maximum number of recorded steps per demo episode before forcing truncation.",
)

flags.DEFINE_boolean(
    "classifier",
    True,
    "Whether to use reward classifier as the success/end signal.",
)

flags.DEFINE_boolean(
    "save_video",
    False,
    "Whether to save video during recording.",
)

flags.DEFINE_boolean(
    "manual_confirm_on_success",
    False,
    "Whether to manually confirm success even when classifier says succeed=True.",
)

flags.DEFINE_string(
    "demo_image_keys",
    "__config__",
    (
        "Comma-separated image keys to save into demos. "
        "'__config__' means use env_config.image_keys. "
        "'all' means keep all observation image keys."
    ),
)

flags.DEFINE_string(
    "demo_extra_obs_keys",
    "state",
    "Comma-separated non-image observation keys to keep, usually 'state'. Use 'none' to keep none.",
)

flags.DEFINE_boolean(
    "demo_strict_obs_keys",
    True,
    "If True, raise error when requested demo image/state key is missing.",
)

flags.DEFINE_float(
    "grasp_penalty_value",
    -0.02,
    (
        "Grasp penalty written into demos after action[6] has been rewritten to "
        "-1/0/+1 event labels. This value should match the learned-gripper penalty."
    ),
)


# ==============================================================
# 夹爪反馈阈值
# ==============================================================
CLOSE_MAX = 30.0
OPEN_MIN = 70.0


# ==============================================================
# action 保存格式约定
# ==============================================================
ARM_ACTION_LOW = -1.0
ARM_ACTION_HIGH = 1.0


def parse_key_list(raw_value, default_keys=None, allow_all=False):
    """
    解析逗号分隔 key 列表。

    - "__config__" -> default_keys
    - "all"        -> None，表示不裁剪
    - "none"       -> []
    """
    if raw_value is None:
        return list(default_keys or [])

    value = str(raw_value).strip()

    if value == "__config__":
        return list(default_keys or [])

    if allow_all and value.lower() == "all":
        return None

    if value.lower() in ("none", "null", ""):
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def get_demo_storage_image_keys():
    return parse_key_list(
        FLAGS.demo_image_keys,
        default_keys=getattr(env_config, "image_keys", []),
        allow_all=True,
    )


def get_demo_storage_extra_keys():
    return parse_key_list(
        FLAGS.demo_extra_obs_keys,
        default_keys=["state"],
        allow_all=False,
    )


def ask_success_from_terminal():
    while True:
        try:
            manual_rew = int(input("Success? (1/0): ").strip())
            if manual_rew in [0, 1]:
                return bool(manual_rew)
            print("❌ 请输入 1 或 0。")
        except ValueError:
            print("❌ 输入无效，请输入 1 或 0。")


def scalar_float(x, default=0.0):
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


# ==============================================================
# observation 裁剪逻辑
# --------------------------------------------------------------
# 关键点：
# - env 内部仍然可以有三路图像，reward classifier 可以用 left_wrist_rgb。
# - 这里只裁剪“保存到 demo pkl 里的 obs/next_obs”。
# - 默认保存 env_config.image_keys，也就是 RLPD policy 真实需要的相机。
# ==============================================================

def _get_obs_value(obs, key):
    """
    支持两种 observation 结构：
    1. obs[key]
    2. obs["images"][key]
    """
    if key in obs:
        return obs[key]

    if "images" in obs and isinstance(obs["images"], dict) and key in obs["images"]:
        return obs["images"][key]

    raise KeyError(
        f"observation 中找不到 key='{key}'。当前 obs keys={list(obs.keys())}"
    )


def prune_obs_for_demo_storage(
    obs,
    image_keys,
    extra_obs_keys,
    strict=True,
):
    """
    保存 demos 前裁剪 observation。

    image_keys=None 表示保留完整 obs。
    image_keys=[...] 表示只保留这些图像 key。
    extra_obs_keys 通常保留 ["state"]。
    """
    if obs is None:
        return obs

    if not isinstance(obs, dict):
        return obs

    if image_keys is None:
        return copy.deepcopy(obs)

    keep = {}

    for key in image_keys:
        try:
            keep[key] = _get_obs_value(obs, key)
        except KeyError:
            if strict:
                raise
            print(f"⚠️ demo 保存时跳过缺失图像 key={key}")

    for key in extra_obs_keys:
        try:
            keep[key] = _get_obs_value(obs, key)
        except KeyError:
            if strict:
                raise
            print(f"⚠️ demo 保存时跳过缺失 extra obs key={key}")

    return keep


def prune_transition_obs_for_demo_storage(
    trans,
    image_keys,
    extra_obs_keys,
    strict=True,
):
    """
    保存前裁剪 transition 里的 observations / next_observations。
    """
    trans = copy.deepcopy(trans)

    if "observations" in trans:
        trans["observations"] = prune_obs_for_demo_storage(
            trans["observations"],
            image_keys=image_keys,
            extra_obs_keys=extra_obs_keys,
            strict=strict,
        )

    if "next_observations" in trans:
        trans["next_observations"] = prune_obs_for_demo_storage(
            trans["next_observations"],
            image_keys=image_keys,
            extra_obs_keys=extra_obs_keys,
            strict=strict,
        )

    return trans


def print_obs_keys_summary(transitions, name="demos"):
    if not transitions:
        print(f"⚠️ {name}: empty transitions")
        return

    obs = transitions[0].get("observations", {})
    next_obs = transitions[0].get("next_observations", {})

    print("\n===== 保存 observation keys 检查 =====")
    print(f"{name} observations keys     : {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print(f"{name} next_observations keys: {list(next_obs.keys()) if isinstance(next_obs, dict) else type(next_obs)}")


# ==============================================================
# action 清洗逻辑
# ==============================================================

def sanitize_single_arm_action_for_storage(
    action,
    quantize_gripper=True,
    source="unknown",
):
    """
    统一所有要写入 demos / replay buffer 的单臂 action 格式。

    目标格式：
      action.shape = (7,)
      action[:6]  = clip 到 [-1, 1]
      action[6]   = -1 / 0 / +1，如果 quantize_gripper=True

    注意：
    - 这里不要依赖 env.action_space.low/high。
    - 这个函数是幂等的，重复调用不会把 action 越变越小。
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    if a.shape[0] != 7:
        raise ValueError(
            f"[sanitize_single_arm_action_for_storage] 单臂任务期望 7 维 action，"
            f"但 source={source} 得到 shape={a.shape}, value={a}"
        )

    before = a.copy()

    # 前 6 维必须是 RLPD 归一化动作
    a[:6] = np.clip(a[:6], ARM_ACTION_LOW, ARM_ACTION_HIGH)

    # gripper 维保存为三值事件
    if quantize_gripper:
        g = float(a[6])
        if g <= -0.5:
            a[6] = -1.0
        elif g >= 0.5:
            a[6] = 1.0
        else:
            a[6] = 0.0
    else:
        a[6] = np.clip(a[6], ARM_ACTION_LOW, ARM_ACTION_HIGH)

    if np.any(np.abs(before[:6]) > 1.0001):
        print(
            f"⚠️ action 前6维超出 [-1,1]，已在保存前 clip。"
            f" source={source}, before={np.round(before, 4).tolist()}, "
            f"after={np.round(a, 4).tolist()}"
        )

    return a.astype(np.float32)


def sanitize_transition_for_storage(trans):
    """
    保存前最后一道动作保险。
    """
    trans = copy.deepcopy(trans)
    trans["actions"] = sanitize_single_arm_action_for_storage(
        trans["actions"],
        quantize_gripper=True,
        source="final_transition_sanitize",
    )
    return trans


# ==============================================================
# gripper 三值重写逻辑
# ==============================================================

def extract_gripper_feedback(obs):
    """
    从 obs["state"] 中提取夹爪反馈量程。
    兼容：
    1) state 是 dict，含 right_gripper / gripper
    2) state 是 ndarray，单臂常见最后一维为 gripper
    """
    if obs is None or "state" not in obs:
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

        for key, val in state.items():
            if "gripper" in str(key).lower():
                arr = np.asarray(val).reshape(-1)
                if arr.size > 0:
                    return float(arr[-1])

        return None

    arr = np.asarray(state)
    while arr.ndim > 1:
        arr = arr[-1]
    arr = arr.reshape(-1)
    if arr.size == 0:
        return None

    return float(arr[-1])


def infer_stable_gripper_state_from_feedback(
    gripper_feedback,
    prev_state,
    close_max=CLOSE_MAX,
    open_min=OPEN_MIN,
):
    """
    将反馈量程映射为稳定夹爪状态：
      -1 -> 当前稳定为闭合
      +1 -> 当前稳定为张开
      中间区 -> 保持上一稳定状态
    """
    if gripper_feedback is None:
        return prev_state

    x = float(gripper_feedback)

    if x <= close_max:
        return -1
    if x >= open_min:
        return +1

    return prev_state


def rewrite_gripper_action_to_official_style(
    action,
    obs,
    next_obs,
    prev_stable_state,
):
    """
    把夹爪最后一维改写成官网风格三值事件：
      -1 = close event
       0 = hold / no-op
      +1 = open event

    注意：
    - 这里不负责前 6 维 clip。
    - 前 6 维 clip 在 sanitize_single_arm_action_for_storage() 中完成。
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    if action.shape[0] != 7:
        return action, prev_stable_state

    prev_feedback = extract_gripper_feedback(obs)
    next_feedback = extract_gripper_feedback(next_obs)

    prev_state = infer_stable_gripper_state_from_feedback(
        prev_feedback,
        prev_stable_state,
    )
    next_state = infer_stable_gripper_state_from_feedback(
        next_feedback,
        prev_state,
    )

    gripper_event = 0.0
    if prev_state is not None and next_state is not None:
        if prev_state == +1 and next_state == -1:
            gripper_event = -1.0
        elif prev_state == -1 and next_state == +1:
            gripper_event = +1.0
        else:
            gripper_event = 0.0
    else:
        gripper_event = 0.0

    action[6] = np.float32(gripper_event)
    return action.astype(np.float32), next_state


# ==============================================================
# grasp_penalty 同步逻辑
# --------------------------------------------------------------
# 关键修改：
# env.step() 内部的 SingleGripperPenaltyWrapper 可能在 action[6]
# 被重写成三值事件之前就计算了 info["grasp_penalty"]。
#
# 因此最终保存 demos 前，必须按“最终保存的 action[6]”
# 重新计算 grasp_penalty，保证：
#   hold(0)       -> 0
#   close(-1)    -> FLAGS.grasp_penalty_value
#   open(+1)     -> FLAGS.grasp_penalty_value
#
# 这样不会再出现：
#   action[6] = hold(0)
#   grasp_penalty = -0.02
# 的错误组合。
# ==============================================================

def recompute_grasp_penalty_from_stored_action(action, penalty_value):
    """
    根据最终保存的 action[6] 三值事件标签重算 grasp_penalty。

    action[6]:
      -1 = close event -> penalty_value
       0 = hold        -> 0
      +1 = open event  -> penalty_value
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)

    if a.shape[0] != 7:
        return 0.0

    g = float(a[6])

    if g <= -0.5:
        return float(penalty_value)

    if g >= 0.5:
        return float(penalty_value)

    return 0.0


def sync_grasp_penalty_with_stored_action(trans, penalty_value):
    """
    将 grasp_penalty 与最终保存的 action[6] 三值事件标签同步。

    保存策略：
      1. trans["grasp_penalty"] 写入重算后的 penalty，方便 RLPD 直接读取。
      2. trans["infos"]["grasp_penalty"] 也写入重算后的 penalty，方便可视化脚本读取。
      3. 如果 env 原始 info 里已经有 grasp_penalty，则保留为 env_grasp_penalty_raw，方便调试。
    """
    trans = copy.deepcopy(trans)

    if "actions" not in trans:
        return trans

    expected_penalty = recompute_grasp_penalty_from_stored_action(
        trans["actions"],
        penalty_value=penalty_value,
    )

    infos = trans.get("infos", {})
    if not isinstance(infos, dict):
        infos = {}

    if "grasp_penalty" in infos:
        infos["env_grasp_penalty_raw"] = scalar_float(infos["grasp_penalty"], 0.0)

    if "grasp_penalty" in trans:
        infos["top_level_grasp_penalty_raw"] = scalar_float(trans["grasp_penalty"], 0.0)

    infos["grasp_penalty"] = float(expected_penalty)
    infos["grasp_penalty_source"] = "recomputed_from_final_stored_action"

    trans["infos"] = infos
    trans["grasp_penalty"] = float(expected_penalty)

    return trans


def extract_saved_grasp_penalty(trans):
    if "grasp_penalty" in trans:
        return scalar_float(trans["grasp_penalty"], 0.0)

    infos = trans.get("infos", {})
    if isinstance(infos, dict) and "grasp_penalty" in infos:
        return scalar_float(infos["grasp_penalty"], 0.0)

    return None


# ==============================================================
# 其他辅助
# ==============================================================

def build_safe_idle_action(action_shape):
    """
    默认发给机器人的“安全空动作”：
    - 前 6 维全 0，不继续推动末端
    - 最后一维给安全夹爪保持值

    注意：
    这个 action 是给 env.step() 执行的，不是最终保存到 demos 的 action。
    保存到 demos 的 action 会另外经过 sanitize。
    """
    safe_idle_action = np.zeros(action_shape, dtype=np.float32)
    if safe_idle_action.shape[0] == 7:
        safe_idle_action[6] = 80.0
    return safe_idle_action


def print_trajectory_action_stats(trajectory):
    if len(trajectory) == 0:
        return

    acts = np.asarray([t["actions"] for t in trajectory], dtype=np.float32)
    if acts.ndim != 2 or acts.shape[1] != 7:
        print(f"⚠️ 无法统计 action，acts.shape={acts.shape}")
        return

    arm_absmax = float(np.max(np.abs(acts[:, :6])))
    arm_min = float(np.min(acts[:, :6]))
    arm_max = float(np.max(acts[:, :6]))

    g = acts[:, 6]
    n_close = int(np.sum(g < -0.5))
    n_open = int(np.sum(g > 0.5))
    n_hold = int(np.sum(np.abs(g) <= 0.5))

    penalties = []
    for t in trajectory:
        p = extract_saved_grasp_penalty(t)
        if p is not None:
            penalties.append(float(p))

    print(f"📦 本条完整轨迹长度: {len(trajectory)}")
    print(f"🦾 arm action min={arm_min:.4f}, max={arm_max:.4f}, absmax={arm_absmax:.4f}")
    print(f"🤏 gripper统计: close={n_close}, hold={n_hold}, open={n_open}")

    if penalties:
        penalties_np = np.asarray(penalties, dtype=np.float32)
        nonzero = int(np.sum(np.abs(penalties_np) > 1e-8))
        vals, cnts = np.unique(np.round(penalties_np, 8), return_counts=True)
        dist = {float(v): int(c) for v, c in zip(vals, cnts)}
        print(f"🧲 grasp_penalty统计: nonzero={nonzero}, sum={float(np.sum(penalties_np)):.6f}, dist={dist}")

    if arm_absmax > 1.0001:
        print("❌ 警告：本条轨迹仍有 action[:6] 超出 [-1,1]，请立刻停止并检查。")
    else:
        print("✅ 本条轨迹 action[:6] 已全部在 [-1,1] 内。")


def finalize_transition_for_demo_storage(
    trans,
    demo_image_keys,
    demo_extra_obs_keys,
):
    """
    最终保存前统一处理：
    1. action 清洗
    2. grasp_penalty 按最终 action[6] 重算
    3. obs / next_obs 按 demo_image_keys 裁剪
    """
    trans = sanitize_transition_for_storage(trans)

    # 关键：必须在 action 已经是最终保存格式之后重算 penalty。
    trans = sync_grasp_penalty_with_stored_action(
        trans,
        penalty_value=FLAGS.grasp_penalty_value,
    )

    trans = prune_transition_obs_for_demo_storage(
        trans,
        image_keys=demo_image_keys,
        extra_obs_keys=demo_extra_obs_keys,
        strict=FLAGS.demo_strict_obs_keys,
    )
    return trans


# ==============================================================
# 主函数
# ==============================================================

def main(_):
    demo_image_keys = get_demo_storage_image_keys()
    demo_extra_obs_keys = get_demo_storage_extra_keys()

    print(f"🚀 开始录制专家数据：{FLAGS.exp_name}")
    print(f"🧠 reward classifier: {'开启' if FLAGS.classifier else '关闭'}")
    print("📌 当前脚本策略：")
    print("   - classifier=True 时：成功由 reward ckpt 在成功瞬间触发。")
    print("   - max_episode_steps 只统计【开始记录后】的步数，reset/等待帧不计入。")
    print("   - reset 后不会立刻记录，必须等第一次 VR 接管 intervene_action 后才开始记录。")
    print("   - 全程轨迹完整保留，不删除演示开始后的静止帧。")
    print("   - 保存到 pkl 的 action 会强制统一格式：")
    print("       action[:6] -> clip 到 [-1,1]")
    print("       action[6]  -> -1 / 0 / +1 三值事件")
    print("   - grasp_penalty 会按最终保存的 action[6] 重新计算，避免 hold 被错误处罚。")
    print("   - demos 保存的图像不直接等于 ENV_IMAGE_KEYS，而是按 image_keys 裁剪。")
    print()
    print("===== 当前相机/观测保存配置 =====")
    print(f"env_config.ENV_IMAGE_KEYS      = {getattr(env_config, 'ENV_IMAGE_KEYS', None)}")
    print(f"env_config.image_keys          = {getattr(env_config, 'image_keys', None)}")
    print(f"env_config.classifier_keys     = {getattr(env_config, 'classifier_keys', None)}")
    print(f"demo_image_keys                = {demo_image_keys if demo_image_keys is not None else 'all'}")
    print(f"demo_extra_obs_keys            = {demo_extra_obs_keys}")
    print(f"demo_strict_obs_keys           = {FLAGS.demo_strict_obs_keys}")
    print(f"grasp_penalty_value            = {FLAGS.grasp_penalty_value}")
    print("================================\n")

    env = env_config.get_environment(
        fake_env=False,
        save_video=FLAGS.save_video,
        classifier=FLAGS.classifier,
    )

    obs, info = env.reset()
    safe_idle_action = build_safe_idle_action(env.action_space.shape)

    print("✅ 环境重置完成，请戴上 VR 头显准备接管！")
    print("⏳ 当前不会记录 reset 后等待帧；检测到第一次 VR 接管后才开始记录本条 demo。")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    max_episode_steps = FLAGS.max_episode_steps

    pbar = tqdm(total=success_needed, desc="成功收集的 Demo 数量")

    trajectory = []
    returns = 0.0
    episode_step = 0

    stable_gripper_state = None
    recording_started = False

    while success_count < success_needed:
        base_env = env.unwrapped

        # 非 reset 阶段，如果还在 Mode 2，就什么都不发，只等待切回 Mode 0
        if getattr(base_env, "script_control_enabled", False):
            time.sleep(0.05)
            continue

        exec_action = safe_idle_action.copy()

        # 这里 env.step 内部仍然能看到完整 obs。
        # 所以 classifier_keys=["left_wrist_rgb"] 的 reward 判断不受 demos 裁剪影响。
        next_obs, rew, done, truncated, info = env.step(exec_action)

        had_intervention = "intervene_action" in info
        raw_episode_end = bool(done or truncated)

        # ------------------------------------------------------
        # reset 后录制门控：
        # 没有第一次 VR 接管之前，不记录 transition，不计 step，不累计 return。
        # ------------------------------------------------------
        if not recording_started:
            if not had_intervention:
                if raw_episode_end:
                    print(
                        "\n⚠️ 未开始记录时环境已经结束。"
                        f" reward={rew}, done={done}, truncated={truncated}。"
                        " 这通常是 reset/等待画面触发了 classifier 或环境终止，当前片段不会保存，重新 reset。"
                    )
                    obs, info = env.reset()
                    safe_idle_action = build_safe_idle_action(env.action_space.shape)
                    trajectory = []
                    returns = 0.0
                    episode_step = 0
                    stable_gripper_state = None
                    recording_started = False
                    print("✅ 重新 reset 完成，继续等待第一次 VR 接管。")
                    continue

                obs = next_obs
                time.sleep(0.01)
                continue

            recording_started = True
            print("🎬 检测到第一次 VR 接管，开始记录本条 demo。")

        # 只有开始记录后，才累计 return / step
        returns += float(rew)
        episode_step += 1

        # ------------------------------------------------------
        # 记录逻辑：
        # - 有 VR 接管：记录 intervene_action
        # - 已经开始记录后，如果某些帧没有 VR 接管：记录零动作/空闲帧
        # ------------------------------------------------------
        if had_intervention:
            raw_actions = np.asarray(info["intervene_action"], dtype=np.float32)
            raw_source = "info.intervene_action"
        else:
            raw_actions = np.zeros(env.action_space.shape, dtype=np.float32)
            raw_source = "zero_after_recording_started"

        raw_actions = sanitize_single_arm_action_for_storage(
            raw_actions,
            quantize_gripper=False,
            source=raw_source,
        )

        # 夹爪三值重写必须使用完整 obs / next_obs，因为 state 在完整 obs 中。
        actions, stable_gripper_state = rewrite_gripper_action_to_official_style(
            raw_actions,
            obs,
            next_obs,
            stable_gripper_state,
        )

        actions = sanitize_single_arm_action_for_storage(
            actions,
            quantize_gripper=True,
            source="after_gripper_rewrite",
        )

        forced_timeout = False
        if episode_step >= max_episode_steps and not raw_episode_end:
            forced_timeout = True
            truncated = True
            print(f"\n⏰ 达到最大录制时长：{max_episode_steps} 步，强制截断当前回合。")

        episode_end = bool(done or truncated)

        # 这里是关键：
        # 保存 transition 时裁剪 obs/next_obs，只保留 demo_image_keys + state。
        transition = copy.deepcopy(
            dict(
                observations=prune_obs_for_demo_storage(
                    obs,
                    image_keys=demo_image_keys,
                    extra_obs_keys=demo_extra_obs_keys,
                    strict=FLAGS.demo_strict_obs_keys,
                ),
                actions=actions,
                next_observations=prune_obs_for_demo_storage(
                    next_obs,
                    image_keys=demo_image_keys,
                    extra_obs_keys=demo_extra_obs_keys,
                    strict=FLAGS.demo_strict_obs_keys,
                ),
                rewards=float(rew),
                masks=1.0 - float(episode_end),
                dones=episode_end,
                infos=copy.deepcopy(info),
            )
        )

        transition = finalize_transition_for_demo_storage(
            transition,
            demo_image_keys=demo_image_keys,
            demo_extra_obs_keys=demo_extra_obs_keys,
        )

        trajectory.append(transition)

        pbar.set_description(
            f"成功 Demo 数: {success_count}/{success_needed} | "
            f"Return: {returns:.2f} | "
            f"Recorded Step: {episode_step}/{max_episode_steps}"
        )

        # 注意：obs 要保留完整 next_obs，不能用裁剪后的 obs。
        # 因为下一步 reward/gripper rewrite 还需要完整观测。
        obs = next_obs

        if episode_end:
            print("\n🔄 回合结束。")
            print(f"   reward={rew}, done={done}, truncated={truncated}, forced_timeout={forced_timeout}")
            print(f"   info.succeed={info.get('succeed', None)}")
            print(f"   recording_started={recording_started}, recorded_steps={episode_step}")

            if FLAGS.classifier:
                succeed = bool(
                    info.get(
                        "success",
                        info.get(
                            "is_success",
                            info.get("succeed", False),
                        ),
                    )
                )

                if succeed and FLAGS.manual_confirm_on_success:
                    print("📝 classifier 判定成功，请人工确认本回合是否真的成功。")
                    succeed = ask_success_from_terminal()

                if len(trajectory) > 0:
                    trajectory[-1]["infos"] = copy.deepcopy(info)
                    trajectory[-1]["infos"]["succeed"] = succeed
                    trajectory[-1]["infos"]["success"] = succeed
                    trajectory[-1]["infos"]["is_success"] = succeed
                    trajectory[-1]["dones"] = True
                    trajectory[-1]["masks"] = 0.0
                    trajectory[-1]["rewards"] = float(succeed)

                    # 重新 finalize 一次，防止上面 copy.deepcopy(info) 把旧 grasp_penalty 覆盖回来。
                    trajectory[-1] = finalize_transition_for_demo_storage(
                        trajectory[-1],
                        demo_image_keys=demo_image_keys,
                        demo_extra_obs_keys=demo_extra_obs_keys,
                    )

            else:
                print("📝 当前 classifier=False，使用人工判定 success / fail。")
                succeed = ask_success_from_terminal()

                if len(trajectory) > 0:
                    trajectory[-1]["infos"] = copy.deepcopy(info)
                    trajectory[-1]["infos"]["succeed"] = succeed
                    trajectory[-1]["infos"]["success"] = succeed
                    trajectory[-1]["infos"]["is_success"] = succeed
                    trajectory[-1]["rewards"] = float(succeed)
                    trajectory[-1]["dones"] = True
                    trajectory[-1]["masks"] = 0.0

                    # 重新 finalize 一次，防止上面 copy.deepcopy(info) 把旧 grasp_penalty 覆盖回来。
                    trajectory[-1] = finalize_transition_for_demo_storage(
                        trajectory[-1],
                        demo_image_keys=demo_image_keys,
                        demo_extra_obs_keys=demo_extra_obs_keys,
                    )

            if succeed and len(trajectory) > 0:
                clean_trajectory = []
                for trans in trajectory:
                    trans = finalize_transition_for_demo_storage(
                        trans,
                        demo_image_keys=demo_image_keys,
                        demo_extra_obs_keys=demo_extra_obs_keys,
                    )
                    clean_trajectory.append(trans)

                transitions.extend(copy.deepcopy(clean_trajectory))
                success_count += 1
                pbar.update(1)

                print(f"🎉 成功录制 1 条 Demo！当前累计成功条数: {success_count}")
                print_trajectory_action_stats(clean_trajectory)
                print_obs_keys_summary(clean_trajectory, name="clean_trajectory")

            else:
                print("❌ 当前回合失败或未成功，已丢弃该轨迹。")

            trajectory = []
            returns = 0.0
            episode_step = 0
            stable_gripper_state = None
            recording_started = False

            if success_count >= success_needed:
                break

            print("🔄 正在复位机器人...")
            obs, info = env.reset()
            safe_idle_action = build_safe_idle_action(env.action_space.shape)
            print("✅ 复位完成。等待第一次 VR 接管后才开始记录下一条 demo。\n" + "-" * 40)

    # 保存前最终全局保险
    transitions = [
        finalize_transition_for_demo_storage(
            t,
            demo_image_keys=demo_image_keys,
            demo_extra_obs_keys=demo_extra_obs_keys,
        )
        for t in transitions
    ]

    save_dir = os.path.join(os.path.dirname(__file__), "demo_data_single")
    os.makedirs(save_dir, exist_ok=True)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_{success_needed}_demos_official_style_clean_pruned_{uuid}.pkl",
    )

    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)

    print(f"\n💾 恭喜！成功保存 {success_needed} 条 Demo 数据至 {file_name}")
    print(f"📊 总 transition 数量: {len(transitions)}")

    print_obs_keys_summary(transitions, name="final_demos")

    if len(transitions) > 0:
        all_actions = np.asarray([t["actions"] for t in transitions], dtype=np.float32)
        print("\n===== 最终保存 action 检查 =====")
        print(f"actions shape: {all_actions.shape}")
        print(f"global min={float(np.min(all_actions)):.6f}")
        print(f"global max={float(np.max(all_actions)):.6f}")
        print(f"global absmax={float(np.max(np.abs(all_actions))):.6f}")
        print(f"arm absmax={float(np.max(np.abs(all_actions[:, :6]))):.6f}")

        g = all_actions[:, 6]
        print(
            "gripper 分布:",
            {
                "close(-1)": int(np.sum(g < -0.5)),
                "hold(0)": int(np.sum(np.abs(g) <= 0.5)),
                "open(+1)": int(np.sum(g > 0.5)),
            },
        )

        all_penalties = []
        for t in transitions:
            p = extract_saved_grasp_penalty(t)
            if p is not None:
                all_penalties.append(float(p))

        if all_penalties:
            all_penalties = np.asarray(all_penalties, dtype=np.float32)
            vals, cnts = np.unique(np.round(all_penalties, 8), return_counts=True)
            dist = {float(v): int(c) for v, c in zip(vals, cnts)}
            print("\n===== 最终保存 grasp_penalty 检查 =====")
            print(f"grasp_penalty shape: {all_penalties.shape}")
            print(f"grasp_penalty 分布: {dist}")
            print(f"grasp_penalty 非零数量: {int(np.sum(np.abs(all_penalties) > 1e-8))}")
            print(f"grasp_penalty sum: {float(np.sum(all_penalties)):.6f}")

            expected_nonzero = int(np.sum(np.abs(g) > 0.5))
            actual_nonzero = int(np.sum(np.abs(all_penalties) > 1e-8))
            print(f"按 gripper 事件预期非零 penalty 数量: {expected_nonzero}")
            print(f"实际非零 penalty 数量: {actual_nonzero}")

            if expected_nonzero != actual_nonzero:
                print("⚠️ grasp_penalty 非零数量和 gripper open/close 事件数量不一致，请检查。")
            else:
                print("✅ grasp_penalty 已与最终 gripper 三值事件对齐。")

        if float(np.max(np.abs(all_actions[:, :6]))) > 1.0001:
            print("❌ 保存后的 action 仍然超界，请不要用于训练。")
        else:
            print("✅ 保存后的 action[:6] 全部在 [-1,1]，可用于 RLPD demos。")


if __name__ == "__main__":
    app.run(main)



