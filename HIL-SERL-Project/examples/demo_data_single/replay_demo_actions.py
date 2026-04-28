# cd ~/HIL-SERL/HIL-SERL-Project/examples
# python demo_data_single/replay_demo_actions.py

# 你先建议用：

# DRY_RUN = True

# 跑一次，先不动机器人，只验证 demo 文件里：

# actions
# observations -> next_observations 反推 action

# 是否一致。

# 然后再改成：

# DRY_RUN = False

# 真机回放。真机回放前一定确认：

# 1. reset 后位置和录制 demo 初始位置一致；
# 2. VR 手柄不要处于 VR 接管模式；
# 3. wrapper 不要把你的 replay action 覆盖成 intervene_action；
# 4. 夹爪和物体初始状态尽量和录制时一致。
# 5,回放 demos 时，不应该把 demo 里的 action 先乘 POS_SCALE/ROT_SCALE 再传给 env.step()。
#    因为 demos / buffer 里保存的 action[:6] 已经是归一化动作，env.step(action) 
#    内部会按 config 的 POS_SCALE/ROT_SCALE 把它变成真实末端增量执行。


"""
replay_demo_actions.py

功能：
1. 读取已录制的 demo pkl。
2. 导入你的任务 config，自动读取 POS_SCALE / ROT_SCALE。
3. reset 真机环境。
4. 按 demo 中保存的归一化 action 逐步 env.step(action) 回放。
5. 打印并保存：
   - demo action 本身
   - action * POS_SCALE / ROT_SCALE 对应的理论真实增量
   - demo 文件中 obs -> next_obs 反推出来的 action
   - live replay 中真实 obs -> next_obs 反推出来的 action
   - 三者误差

重点：
- demo 里的 action[:6] 是归一化动作，不要手动乘 scale 后再传给 env.step。
- env.step(action) 内部应该负责乘 POS_SCALE / ROT_SCALE 并加到当前 EE pose。
- 本脚本只用 POS_SCALE / ROT_SCALE 做打印、验证、反推对比。
"""

import os
import sys
import glob
import time
import csv
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# 你主要改这里：脚本配置
# =============================================================================

# 项目根目录：默认假设本脚本放在 examples/ 或 examples/inspect/ 附近。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 如果你的脚本放在 examples/inspect/，上面会到 examples；
# 下面这个兜底会再向上一层找 HIL-SERL-Project。
if not os.path.exists(os.path.join(PROJECT_ROOT, "examples")):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 单臂任务 config：你当前 single 任务用这个。
from examples.galaxea_task.usb_pick_insertion_single.config import env_config


# demo 路径：可以是目录、单个 pkl、glob。
DEMO_PATH = "./demo_data_single"

# 如果 DEMO_PATH 是目录或 glob，选择第几个 pkl 文件。
DEMO_FILE_INDEX = 0

# 从 demo 的第几条 transition 开始回放。
START_TRANSITION = 0

# 回放多少步。None 表示回放到 demo 结束。
MAX_STEPS = None

# 每步之间 sleep，防止动作太快。
STEP_SLEEP_SEC = 0.10

# 是否真的执行 env.step。
# True 只离线检查 demo 文件里的 obs/action/next_obs，不动机器人。
DRY_RUN = False  #True  #False

# 回放前是否等待你按回车。
WAIT_ENTER_BEFORE_REPLAY = True

# 是否要求你确认当前机器人 reset 后状态和 demo 初始状态接近。
PRINT_INITIAL_POSE_COMPARE = True

# 如果 action[:6] 超出 [-1,1]，是否安全 clip 后再回放。
# 建议 True，避免旧数据异常值直接打给真机。
CLIP_ACTION_FOR_SAFETY = True

# 夹爪维度是否强制三值化。
# 当前你的数据应该已经是 -1/0/+1。这里保守启用。
QUANTIZE_GRIPPER_FOR_SAFETY = True

# 是否在 env 外层强制关闭 VR 接管状态。
# 注意：最好物理手柄也切到脚本/AI 模式，否则 VR callback 可能又切回去。
FORCE_SCRIPT_MODE = True

# 是否创建环境时启用 classifier。
# 回放动作检查不需要 classifier，False 更快。
USE_CLASSIFIER = False

# 是否保存 CSV 分析结果。
SAVE_CSV = True
CSV_PATH = "./demo_replay_action_check.csv"

# 打印频率。
PRINT_EVERY = 1

# 误差报警阈值：live 反推 action 与 demo action 的绝对误差。
ACTION_ERROR_WARN_THRESHOLD = 0.35

# 初始位姿误差报警阈值。
INITIAL_POS_WARN_M = 0.02       # 2 cm
INITIAL_ROT_WARN_RAD = 0.20     # 约 11.5 度


# =============================================================================
# 工具函数
# =============================================================================

def sorted_pkl_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.pkl")))
    else:
        files = sorted(glob.glob(path))

    files = [f for f in files if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError(f"没有找到 pkl 文件: {path}")

    return files


def load_transitions(path: str, file_index: int) -> Tuple[str, List[Dict[str, Any]]]:
    files = sorted_pkl_files(path)

    if file_index < 0 or file_index >= len(files):
        raise IndexError(f"DEMO_FILE_INDEX={file_index} 越界，共找到 {len(files)} 个 pkl 文件。")

    file_path = files[file_index]
    with open(file_path, "rb") as f:
        data = pkl.load(f)

    if isinstance(data, dict) and "transitions" in data:
        transitions = data["transitions"]
    elif isinstance(data, list):
        transitions = data
    else:
        raise ValueError(f"无法识别 pkl 格式: type={type(data)}")

    if len(transitions) == 0:
        raise ValueError(f"pkl 为空: {file_path}")

    return file_path, transitions


def get_config_value(env, name: str, default: float) -> float:
    """
    优先从 env.unwrapped.config 读取 POS_SCALE / ROT_SCALE。
    再从 env_config 读取。
    最后 fallback 到 default。
    """
    try:
        base_env = env.unwrapped
        cfg = getattr(base_env, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return float(getattr(cfg, name))
        if hasattr(base_env, name):
            return float(getattr(base_env, name))
    except Exception:
        pass

    if hasattr(env_config, name):
        return float(getattr(env_config, name))

    return float(default)


def to_1d_array(x) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def extract_pose_from_state_dict(state: Dict[str, Any], arm_side: str = "right") -> Optional[np.ndarray]:
    preferred_keys = []

    if arm_side == "right":
        preferred_keys.extend([
            "right_ee_pose",
            "right/tcp_pose",
            "right_tcp_pose",
            "state/right_ee_pose",
            "state/right/tcp_pose",
            "pose_ee_arm_right",
        ])
    else:
        preferred_keys.extend([
            "left_ee_pose",
            "left/tcp_pose",
            "left_tcp_pose",
            "state/left_ee_pose",
            "state/left/tcp_pose",
            "pose_ee_arm_left",
        ])

    preferred_keys.extend([
        "ee_pose",
        "tcp_pose",
        "pose_ee",
    ])

    for key in preferred_keys:
        if key in state:
            arr = to_1d_array(state[key])
            if arr is not None and arr.size >= 6:
                return arr[:7] if arr.size >= 7 else arr[:6]

    for key, value in state.items():
        k = str(key).lower()
        if ("pose" in k or "tcp" in k or "ee" in k) and "gripper" not in k:
            arr = to_1d_array(value)
            if arr is not None and arr.size >= 6:
                return arr[:7] if arr.size >= 7 else arr[:6]

    return None


def extract_ee_pose(obs: Dict[str, Any], arm_side: str = "right") -> Optional[np.ndarray]:
    """
    从 observation 中提取 EE pose。
    兼容：
      obs["state"] 是 dict
      obs["state"] 是 array: [x,y,z,qx,qy,qz,qw,gripper]
    """
    if obs is None or not isinstance(obs, dict):
        return None

    if "state" not in obs:
        return None

    state = obs["state"]

    if isinstance(state, dict):
        return extract_pose_from_state_dict(state, arm_side=arm_side)

    arr = to_1d_array(state)
    if arr is None:
        return None

    if arr.size >= 7:
        return arr[:7]
    if arr.size >= 6:
        return arr[:6]

    return None


def pose_to_pos_quat(pose: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    支持：
      7 维: xyz + quat(xyzw)
      6 维: xyz + euler(xyz)
    """
    if pose is None:
        return None, None

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


def compute_delta_action_from_poses(
    prev_pose: Optional[np.ndarray],
    next_pose: Optional[np.ndarray],
    pos_scale: float,
    rot_scale: float,
    clip: bool = False,
) -> Optional[np.ndarray]:
    """
    用真实 EE pose 差分反推 action[:6]。
    和你的 VRInterventionWrapper 逻辑一致：
      action[:3] = (next_pos - prev_pos) / POS_SCALE
      action[3:6] = rotvec(next_rot * prev_rot.inv()) / ROT_SCALE
    """
    if prev_pose is None or next_pose is None:
        return None

    prev_pos, prev_quat = pose_to_pos_quat(prev_pose)
    next_pos, next_quat = pose_to_pos_quat(next_pose)

    if prev_pos is None or next_pos is None:
        return None

    try:
        pos_delta = next_pos - prev_pos

        prev_rot = R.from_quat(prev_quat)
        next_rot = R.from_quat(next_quat)
        rot_delta = (next_rot * prev_rot.inv()).as_rotvec().astype(np.float32)

        a6 = np.zeros(6, dtype=np.float32)
        a6[:3] = pos_delta / float(pos_scale)
        a6[3:6] = rot_delta / float(rot_scale)

        if clip:
            a6 = np.clip(a6, -1.0, 1.0)

        return a6.astype(np.float32)
    except Exception:
        return None


def compute_real_delta_from_action(action: np.ndarray, pos_scale: float, rot_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    仅用于打印：
      normalized action -> 理论真实位移 / 旋转增量
    不要把这个 scaled action 再传给 env.step。
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    pos_delta_m = a[:3] * float(pos_scale)
    rot_delta_rad = a[3:6] * float(rot_scale)
    return pos_delta_m.astype(np.float32), rot_delta_rad.astype(np.float32)


def sanitize_action_for_replay(action: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    安全处理 action：
      action[:6] 可选 clip 到 [-1,1]
      action[6] 可选量化成 -1/0/+1
    返回：clean_action, changed
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    if a.shape[0] != 7:
        raise ValueError(f"当前脚本按单臂 7 维 action 写的，但收到 action shape={a.shape}")

    before = a.copy()

    if CLIP_ACTION_FOR_SAFETY:
        a[:6] = np.clip(a[:6], -1.0, 1.0)

    if QUANTIZE_GRIPPER_FOR_SAFETY:
        g = float(a[6])
        if g <= -0.5:
            a[6] = -1.0
        elif g >= 0.5:
            a[6] = 1.0
        else:
            a[6] = 0.0

    changed = bool(not np.allclose(before, a, atol=1e-6, rtol=1e-6))
    return a.astype(np.float32), changed


def describe_gripper(g: float) -> str:
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def force_script_mode_if_possible(env) -> None:
    """
    尽量让 VRInterventionWrapper 不覆盖我们传入的 demo action。
    最好你物理手柄也切到脚本/AI控制模式。
    """
    if not FORCE_SCRIPT_MODE:
        return

    print("🤖 尝试强制进入脚本/AI动作回放模式，避免 VRInterventionWrapper 覆盖 action...")

    current = env
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if hasattr(current, "use_vr_mode"):
            try:
                current.use_vr_mode = False
                print(f"  - set {type(current).__name__}.use_vr_mode=False")
            except Exception as e:
                print(f"  - warning: set use_vr_mode failed: {e!r}")

        if hasattr(current, "_call_switch_service"):
            try:
                current._call_switch_service(False)
                print(f"  - call {type(current).__name__}._call_switch_service(False)")
            except Exception as e:
                print(f"  - warning: _call_switch_service(False) failed: {e!r}")

        current = getattr(current, "env", None)

    try:
        base = env.unwrapped
        if hasattr(base, "notify_script_control"):
            base.notify_script_control(True)
            print("  - call env.unwrapped.notify_script_control(True)")
    except Exception as e:
        print(f"  - warning: notify_script_control failed: {e!r}")

    print("✅ 已请求脚本/AI模式。请确认 VR 手柄/底层控制没有处于 VR 接管模式。")


def compare_initial_pose(demo_obs: Dict[str, Any], live_obs: Dict[str, Any], pos_scale: float, rot_scale: float) -> None:
    demo_pose = extract_ee_pose(demo_obs)
    live_pose = extract_ee_pose(live_obs)

    if demo_pose is None or live_pose is None:
        print("⚠️ 无法比较初始 EE pose：demo_pose 或 live_pose 提取失败。")
        return

    demo_pos, demo_quat = pose_to_pos_quat(demo_pose)
    live_pos, live_quat = pose_to_pos_quat(live_pose)

    if demo_pos is None or live_pos is None:
        print("⚠️ 无法比较初始 EE pose：pose_to_pos_quat 失败。")
        return

    pos_err = float(np.linalg.norm(live_pos - demo_pos))

    try:
        demo_rot = R.from_quat(demo_quat)
        live_rot = R.from_quat(live_quat)
        rot_err = float((live_rot * demo_rot.inv()).magnitude())
    except Exception:
        rot_err = float("nan")

    print("=" * 100)
    print("初始 reset 位姿对比")
    print("=" * 100)
    print(f"demo_pos: {np.round(demo_pos, 6).tolist()}")
    print(f"live_pos: {np.round(live_pos, 6).tolist()}")
    print(f"pos_err : {pos_err:.6f} m")
    print(f"rot_err : {rot_err:.6f} rad")
    print(f"pos_err / POS_SCALE: {pos_err / pos_scale:.3f}")
    print(f"rot_err / ROT_SCALE: {rot_err / rot_scale:.3f}")

    if pos_err > INITIAL_POS_WARN_M or (not np.isnan(rot_err) and rot_err > INITIAL_ROT_WARN_RAD):
        print("⚠️ 初始 reset 和 demo 录制初始位姿差异较大，回放轨迹可能和录制时不一致。")
    else:
        print("✅ 初始 reset 和 demo 录制初始位姿比较接近。")


def make_env():
    env = env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=USE_CLASSIFIER,
    )
    return env


# =============================================================================
# 主逻辑
# =============================================================================

def main():
    file_path, transitions = load_transitions(DEMO_PATH, DEMO_FILE_INDEX)

    print("=" * 100)
    print("Demo Action Replay")
    print("=" * 100)
    print(f"DEMO_PATH       : {DEMO_PATH}")
    print(f"selected file   : {file_path}")
    print(f"num transitions : {len(transitions)}")
    print(f"START_TRANSITION: {START_TRANSITION}")
    print(f"MAX_STEPS       : {MAX_STEPS}")
    print(f"DRY_RUN         : {DRY_RUN}")
    print(f"CLIP_ACTION     : {CLIP_ACTION_FOR_SAFETY}")
    print(f"QUANTIZE_GRIPPER: {QUANTIZE_GRIPPER_FOR_SAFETY}")

    if START_TRANSITION < 0 or START_TRANSITION >= len(transitions):
        raise IndexError(f"START_TRANSITION={START_TRANSITION} 越界。")

    end = len(transitions) if MAX_STEPS is None else min(len(transitions), START_TRANSITION + int(MAX_STEPS))
    selected = transitions[START_TRANSITION:end]

    env = None
    obs = None

    if not DRY_RUN:
        print("🌍 正在创建真实环境...")
        env = make_env()

        pos_scale = get_config_value(env, "POS_SCALE", 0.01)
        rot_scale = get_config_value(env, "ROT_SCALE", 0.05)

        print(f"✅ 从 config/env 读取缩放：POS_SCALE={pos_scale}, ROT_SCALE={rot_scale}")

        print("🔄 正在 reset 环境...")
        obs, info = env.reset()
        force_script_mode_if_possible(env)

        if PRINT_INITIAL_POSE_COMPARE:
            compare_initial_pose(
                selected[0].get("observations", None),
                obs,
                pos_scale,
                rot_scale,
            )

        if WAIT_ENTER_BEFORE_REPLAY:
            input("确认机器人已经 reset 到和录制时一致，且处于脚本/AI控制模式后，按 Enter 开始回放...")
    else:
        # dry-run 不创建真机环境，也要从 env_config 读 scale。
        pos_scale = float(getattr(env_config, "POS_SCALE", 0.01))
        rot_scale = float(getattr(env_config, "ROT_SCALE", 0.05))
        print(f"✅ DRY_RUN：从 env_config 读取缩放：POS_SCALE={pos_scale}, ROT_SCALE={rot_scale}")

    rows = []

    print("=" * 100)
    print("开始回放 / 检查")
    print("=" * 100)

    changed_count = 0
    missing_demo_delta_count = 0
    missing_live_delta_count = 0
    warn_count = 0

    for local_i, transition in enumerate(selected):
        global_i = START_TRANSITION + local_i

        if "actions" not in transition:
            print(f"⚠️ transition {global_i} 没有 actions，跳过。")
            continue

        raw_action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)
        action, changed = sanitize_action_for_replay(raw_action)
        changed_count += int(changed)

        expected_pos_delta_m, expected_rot_delta_rad = compute_real_delta_from_action(
            action,
            pos_scale,
            rot_scale,
        )

        demo_prev_pose = extract_ee_pose(transition.get("observations", None))
        demo_next_pose = extract_ee_pose(transition.get("next_observations", None))

        demo_delta_action6 = compute_delta_action_from_poses(
            demo_prev_pose,
            demo_next_pose,
            pos_scale,
            rot_scale,
            clip=False,
        )

        if demo_delta_action6 is None:
            missing_demo_delta_count += 1

        live_delta_action6 = None
        live_reward = None
        live_done = None
        live_truncated = None

        if not DRY_RUN:
            live_prev_pose = extract_ee_pose(obs)

            next_obs, reward, done, truncated, info = env.step(action)

            live_next_pose = extract_ee_pose(next_obs)
            live_delta_action6 = compute_delta_action_from_poses(
                live_prev_pose,
                live_next_pose,
                pos_scale,
                rot_scale,
                clip=False,
            )

            if live_delta_action6 is None:
                missing_live_delta_count += 1

            live_reward = reward
            live_done = done
            live_truncated = truncated
            obs = next_obs

        # 误差统计
        demo_err = None
        live_err = None

        if demo_delta_action6 is not None:
            demo_err_vec = demo_delta_action6 - action[:6]
            demo_err = float(np.max(np.abs(demo_err_vec)))

        if live_delta_action6 is not None:
            live_err_vec = live_delta_action6 - action[:6]
            live_err = float(np.max(np.abs(live_err_vec)))
            if live_err > ACTION_ERROR_WARN_THRESHOLD:
                warn_count += 1

        if local_i % PRINT_EVERY == 0:
            print("-" * 100)
            print(f"step {global_i} / local {local_i}")
            print(f"action[:6] normalized    : {np.round(action[:6], 4).tolist()}")
            print(f"gripper action[6]        : {describe_gripper(action[6])}")
            print(f"expected pos delta (m)   : {np.round(expected_pos_delta_m, 6).tolist()}")
            print(f"expected rot delta (rad) : {np.round(expected_rot_delta_rad, 6).tolist()}")

            if demo_delta_action6 is not None:
                print(f"demo obs-delta action[:6]: {np.round(demo_delta_action6, 4).tolist()}")
                print(f"demo max_abs_error       : {demo_err:.6f}")
            else:
                print("demo obs-delta action[:6]: N/A")

            if not DRY_RUN:
                if live_delta_action6 is not None:
                    print(f"live obs-delta action[:6]: {np.round(live_delta_action6, 4).tolist()}")
                    print(f"live max_abs_error       : {live_err:.6f}")
                    if live_err is not None and live_err > ACTION_ERROR_WARN_THRESHOLD:
                        print("⚠️ live 回放反推 action 和 demo action 差异较大。")
                else:
                    print("live obs-delta action[:6]: N/A")

                print(f"reward={live_reward}, done={live_done}, truncated={live_truncated}")

        row = {
            "global_step": global_i,
            "local_step": local_i,
            "raw_action_min": float(np.min(raw_action)),
            "raw_action_max": float(np.max(raw_action)),
            "action_0": float(action[0]),
            "action_1": float(action[1]),
            "action_2": float(action[2]),
            "action_3": float(action[3]),
            "action_4": float(action[4]),
            "action_5": float(action[5]),
            "action_6": float(action[6]),
            "expected_dx_m": float(expected_pos_delta_m[0]),
            "expected_dy_m": float(expected_pos_delta_m[1]),
            "expected_dz_m": float(expected_pos_delta_m[2]),
            "expected_rx_rad": float(expected_rot_delta_rad[0]),
            "expected_ry_rad": float(expected_rot_delta_rad[1]),
            "expected_rz_rad": float(expected_rot_delta_rad[2]),
            "demo_err_max": "" if demo_err is None else float(demo_err),
            "live_err_max": "" if live_err is None else float(live_err),
            "live_reward": "" if live_reward is None else float(live_reward),
            "live_done": "" if live_done is None else bool(live_done),
            "live_truncated": "" if live_truncated is None else bool(live_truncated),
            "action_changed_by_safety": bool(changed),
        }

        if demo_delta_action6 is not None:
            for j in range(6):
                row[f"demo_delta_action_{j}"] = float(demo_delta_action6[j])
        else:
            for j in range(6):
                row[f"demo_delta_action_{j}"] = ""

        if live_delta_action6 is not None:
            for j in range(6):
                row[f"live_delta_action_{j}"] = float(live_delta_action6[j])
        else:
            for j in range(6):
                row[f"live_delta_action_{j}"] = ""

        rows.append(row)

        if not DRY_RUN:
            time.sleep(STEP_SLEEP_SEC)

            if live_done or live_truncated:
                print(f"✅ 环境在 step {global_i} 返回 done/truncated，停止回放。")
                break

    print("=" * 100)
    print("回放 / 检查总结")
    print("=" * 100)
    print(f"selected transitions        : {len(selected)}")
    print(f"executed / checked rows     : {len(rows)}")
    print(f"changed_by_safety           : {changed_count}")
    print(f"missing_demo_delta_count    : {missing_demo_delta_count}")
    print(f"missing_live_delta_count    : {missing_live_delta_count}")
    print(f"live_error_warn_count       : {warn_count}")

    if rows:
        demo_errs = [r["demo_err_max"] for r in rows if r["demo_err_max"] != ""]
        live_errs = [r["live_err_max"] for r in rows if r["live_err_max"] != ""]

        if demo_errs:
            print(f"demo_err_max mean/max       : {np.mean(demo_errs):.6f} / {np.max(demo_errs):.6f}")
        if live_errs:
            print(f"live_err_max mean/max       : {np.mean(live_errs):.6f} / {np.max(live_errs):.6f}")

        arr_actions = np.array([[r[f"action_{j}"] for j in range(7)] for r in rows], dtype=np.float32)
        print(f"replay action[:6] absmax    : {float(np.max(np.abs(arr_actions[:, :6]))):.6f}")
        print(f"gripper close/hold/open     : "
              f"{int(np.sum(arr_actions[:, 6] < -0.5))} / "
              f"{int(np.sum(np.abs(arr_actions[:, 6]) <= 0.5))} / "
              f"{int(np.sum(arr_actions[:, 6] > 0.5))}")

    if SAVE_CSV and rows:
        fieldnames = list(rows[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ CSV saved: {os.path.abspath(CSV_PATH)}")

    print("完成。")


if __name__ == "__main__":
    main()