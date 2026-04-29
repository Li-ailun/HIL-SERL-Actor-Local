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

# 修改config后必须重新录制demos，录制完demos再修改config的缩放不可行
# 



# 回到 examples 目录运行

# 推荐这样：

# cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

# python demo_data_single/replay_demo_actions.py


"""
replay_demos_episode_by_episode_reset.py

功能：
1. 自动识别 demo pkl 中一共有多少条 demos / episodes。
   - 支持 DEMO_PATH 是目录、单个 pkl、glob。
   - 支持 list[transition]、dict{"transitions": ...}、dict{"episodes": ...}。
   - 默认用 done=True / mask=0 / reward>0 作为 episode 结束边界。

2. 每次只播放一条 demo。
   - 播放 demo_i 前 env.reset()
   - reset 完成后进入脚本/IK模式
   - 等待人工确认
   - 播放 demo_i
   - demo_i 播放结束后再次 reset，再播放 demo_i+1

3. 保留原 replay 诊断逻辑：
   - action[:6] 理论物理位移/旋转
   - live obs 中 EE pose 前后差分
   - live_delta_action - demo_action 误差
   - gripper feedback 前后变化
   - env._last_hw_gripper_cmd 记忆值
   - reward / done / truncated
   - CSV 保存

使用：
  python replay_demos_episode_by_episode_reset.py

常用配置在文件顶部修改：
  DEMO_PATH
  FILE_INDEX
  DEMO_INDEX_START / DEMO_INDEX_END
  WAIT_ENTER_BEFORE_EACH_DEMO
  STOP_ON_LIVE_REWARD / STOP_ON_LIVE_DONE
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
# 0. 你主要改这里
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.exists(os.path.join(PROJECT_ROOT, "examples")):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config


# demo 路径：目录、单个 pkl、或 glob 都可以。
DEMO_PATH = "./demo_data_single"

# None 表示读取 DEMO_PATH 匹配到的全部 pkl。
# 0/1/2 表示只读取第几个 pkl。
FILE_INDEX = None

# 播放哪几条 demo。
# DEMO_INDEX_START 从 0 开始。
# DEMO_INDEX_END = None 表示播到最后；否则不包含 END。
DEMO_INDEX_START = 0
DEMO_INDEX_END = None

# 每条 demo 内最多播放多少步。
# None 表示播放该 demo 的全部 transition。
MAX_STEPS_PER_DEMO = None

DRY_RUN = False

# 统一使用 env.step() 内部等待/刷新逻辑。
# 如果 GalaxeaArmEnv.step() 里已经加入 ACTION_SETTLE_SEC，
# replay 脚本外层就不要再额外 sleep / _get_sync_obs。
STEP_SLEEP_SEC = 0.0
REFRESH_OBS_AFTER_SLEEP = False

# 每条 demo 播放前是否等待按 Enter。
WAIT_ENTER_BEFORE_EACH_DEMO = True

# 每条 demo 播放结束 reset 前是否等待按 Enter。
WAIT_ENTER_BEFORE_RESET_AFTER_DEMO = False

FORCE_SCRIPT_MODE = True
USE_CLASSIFIER = False

CLIP_ACTION_FOR_SAFETY = True
QUANTIZE_GRIPPER_FOR_SAFETY = True

# live reward/done/truncated 提前触发时是否停止当前 demo。
# 建议 True，避免真实成功后继续播放剩余动作。
STOP_ON_LIVE_REWARD = True
STOP_ON_LIVE_DONE = True
STOP_ON_LIVE_TRUNCATED = True

# demo pkl 自己的 terminal 帧一定会结束当前 demo。
STOP_ON_DEMO_TERMINAL = True

PRINT_EVERY = 1

SAVE_CSV = True
CSV_PATH = "./demo_replay_episode_by_episode_reset_diagnostics.csv"

# 夹爪 feedback 判定阈值，和 env/config 保持一致。
HW_CLOSE_MAX = 30.0
HW_OPEN_MIN = 70.0

# legacy demo 没有 metadata 时，用手动 scale 做离线提示。
# 本脚本不做动作转换，只打印检查。
MANUAL_DEMO_POS_SCALE = 0.02
MANUAL_DEMO_ROT_SCALE = 0.04
MANUAL_DEMO_HZ = 10

STRICT_SCALE_FOR_LIVE_REPLAY = True

# 手臂动作误差告警阈值。
LIVE_WARN_POS_ERR_M = 0.005       # 5 mm
LIVE_WARN_ROT_ERR_RAD = 0.05      # about 2.86 deg
LIVE_WARN_ACTION_ERR_NORM = 0.5   # normalized action error

SATURATION_THRESHOLD = 0.98


# =============================================================================
# 1. demo / metadata / episode grouping
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


def load_one_pkl(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    with open(file_path, "rb") as f:
        data = pkl.load(f)

    metadata = {}

    if isinstance(data, dict) and "episodes" in data:
        # 支持 raw/episode 格式，但如果 episode 里面已经是 standard transition，也可以直接 flatten。
        episodes_obj = data["episodes"]
        metadata = data.get("metadata", {}) or {}
        flat = []
        for ep in episodes_obj:
            if isinstance(ep, list):
                flat.extend(ep)
            else:
                raise ValueError(f"episodes 里不是 list: {type(ep)}")
        return flat, metadata, "dict_episodes_flattened"

    if isinstance(data, dict) and "transitions" in data:
        transitions = data["transitions"]
        metadata = data.get("metadata", {}) or {}
        return transitions, metadata, "dict_with_transitions"

    if isinstance(data, list):
        return data, {}, "legacy_list"

    raise ValueError(f"无法识别 pkl 格式: {file_path}, type={type(data)}")


def transition_reward(trans: Dict[str, Any]) -> float:
    for k in ("rewards", "reward"):
        if k in trans:
            try:
                return float(np.asarray(trans[k]).reshape(-1)[0])
            except Exception:
                return float(trans[k])
    return 0.0


def transition_done(trans: Dict[str, Any]) -> bool:
    for k in ("dones", "done"):
        if k in trans:
            try:
                return bool(np.asarray(trans[k]).reshape(-1)[0])
            except Exception:
                return bool(trans[k])
    return False


def transition_mask(trans: Dict[str, Any]) -> Optional[float]:
    for k in ("masks", "mask"):
        if k in trans:
            try:
                return float(np.asarray(trans[k]).reshape(-1)[0])
            except Exception:
                return float(trans[k])
    return None


def is_episode_terminal(trans: Dict[str, Any]) -> bool:
    """
    标准成功 demo 通常满足：
      reward=1, done=True, mask=0
    这里放宽：done=True 或 mask=0 或 reward>0 都认为是一条 demo 的边界。
    """
    if transition_done(trans):
        return True
    mask = transition_mask(trans)
    if mask is not None and abs(mask) < 1e-8:
        return True
    if transition_reward(trans) > 0:
        return True
    return False


def split_transitions_into_episodes(
    transitions: List[Dict[str, Any]],
    source_file: str,
    pkl_format: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    返回：
      episodes: list of {
        file_path, pkl_format, local_episode_index,
        start_index, end_index_exclusive, transitions, complete
      }
      tails: 未以 terminal 结束的残留片段
    """
    episodes = []
    tails = []

    current = []
    start_idx = 0
    local_ep_idx = 0

    for i, trans in enumerate(transitions):
        if len(current) == 0:
            start_idx = i
        current.append(trans)

        if is_episode_terminal(trans):
            episodes.append({
                "file_path": source_file,
                "pkl_format": pkl_format,
                "local_episode_index": local_ep_idx,
                "start_index": start_idx,
                "end_index_exclusive": i + 1,
                "transitions": current,
                "complete": True,
            })
            local_ep_idx += 1
            current = []

    if current:
        tails.append({
            "file_path": source_file,
            "pkl_format": pkl_format,
            "local_episode_index": local_ep_idx,
            "start_index": start_idx,
            "end_index_exclusive": len(transitions),
            "transitions": current,
            "complete": False,
        })

    return episodes, tails


def load_all_demo_episodes(path: str, file_index: Optional[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    files = sorted_pkl_files(path)

    if file_index is not None:
        if file_index < 0 or file_index >= len(files):
            raise IndexError(f"FILE_INDEX={file_index} 越界，共找到 {len(files)} 个 pkl 文件")
        files = [files[file_index]]

    all_episodes = []
    all_tails = []
    per_file = []

    for file_i, fp in enumerate(files):
        transitions, metadata, fmt = load_one_pkl(fp)
        episodes, tails = split_transitions_into_episodes(transitions, fp, fmt)

        for ep in episodes:
            ep["file_global_index"] = file_i
            ep["metadata"] = metadata

        all_episodes.extend(episodes)
        all_tails.extend(tails)

        lengths = [len(e["transitions"]) for e in episodes]
        per_file.append({
            "file_path": fp,
            "format": fmt,
            "transition_count": len(transitions),
            "episode_count": len(episodes),
            "tail_count": len(tails),
            "lengths": lengths,
            "metadata": metadata,
        })

    summary = {
        "files": files,
        "per_file": per_file,
        "total_episodes": len(all_episodes),
        "total_tails": len(all_tails),
    }
    return all_episodes, all_tails, summary


def print_episode_inventory(episodes: List[Dict[str, Any]], tails: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("=" * 100)
    print("Demo inventory / episode 边界识别")
    print("=" * 100)

    print(f"pkl files loaded: {len(summary['files'])}")
    for i, item in enumerate(summary["per_file"]):
        lengths = item["lengths"]
        if lengths:
            length_desc = f"min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}"
        else:
            length_desc = "N/A"
        print("-" * 100)
        print(f"[file {i}] {item['file_path']}")
        print(f"  format           : {item['format']}")
        print(f"  transitions      : {item['transition_count']}")
        print(f"  completed demos  : {item['episode_count']}")
        print(f"  incomplete tails : {item['tail_count']}")
        print(f"  episode length   : {length_desc}")

    print("=" * 100)
    print(f"TOTAL completed demos : {len(episodes)}")
    print(f"TOTAL incomplete tails: {len(tails)}")
    if episodes:
        lengths = [len(e["transitions"]) for e in episodes]
        print(f"TOTAL length          : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")

    print("\n前 20 条 demo：")
    for global_idx, ep in enumerate(episodes[:20]):
        print(
            f"  demo#{global_idx:03d} | file#{ep['file_global_index']} "
            f"local#{ep['local_episode_index']} | "
            f"range=[{ep['start_index']},{ep['end_index_exclusive']}) | "
            f"len={len(ep['transitions'])} | "
            f"{os.path.basename(ep['file_path'])}"
        )

    if tails:
        print("\n⚠️ 发现未以 done/mask0/reward>0 结束的 tail，默认不播放：")
        for t in tails[:10]:
            print(
                f"  tail file={os.path.basename(t['file_path'])}, "
                f"range=[{t['start_index']},{t['end_index_exclusive']}), "
                f"len={len(t['transitions'])}"
            )


def get_metadata_float(metadata: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            try:
                return float(metadata[key])
            except Exception:
                pass
    return None


def get_config_value(env, name: str, default: float) -> float:
    try:
        base = env.unwrapped
        cfg = getattr(base, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return float(getattr(cfg, name))
        if hasattr(base, name):
            return float(getattr(base, name))
    except Exception:
        pass

    if hasattr(env_config, name):
        return float(getattr(env_config, name))

    return float(default)


def resolve_demo_scales(metadata: Dict[str, Any], current_pos: float, current_rot: float, current_hz: float):
    meta_pos = get_metadata_float(metadata, "pos_scale", "POS_SCALE")
    meta_rot = get_metadata_float(metadata, "rot_scale", "ROT_SCALE")
    meta_hz = get_metadata_float(metadata, "hz", "HZ")

    if meta_pos is not None and meta_rot is not None:
        return meta_pos, meta_rot, meta_hz if meta_hz is not None else current_hz, "metadata"

    if MANUAL_DEMO_POS_SCALE is not None and MANUAL_DEMO_ROT_SCALE is not None:
        return (
            float(MANUAL_DEMO_POS_SCALE),
            float(MANUAL_DEMO_ROT_SCALE),
            float(MANUAL_DEMO_HZ) if MANUAL_DEMO_HZ is not None else current_hz,
            "manual",
        )

    return current_pos, current_rot, current_hz, "current_config_fallback"


def scales_match(a: float, b: float, atol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= atol


def check_scale_consistency(demo_pos, demo_rot, current_pos, current_rot, source, dry_run):
    print("=" * 100)
    print("Scale 检查")
    print("=" * 100)
    print(f"demo scale source : {source}")
    print(f"demo POS_SCALE    : {demo_pos}")
    print(f"demo ROT_SCALE    : {demo_rot}")
    print(f"current POS_SCALE : {current_pos}")
    print(f"current ROT_SCALE : {current_rot}")

    if scales_match(demo_pos, current_pos) and scales_match(demo_rot, current_rot):
        print("✅ demo scale 和当前 config scale 一致。")
        return

    print("⚠️ demo scale 和当前 config scale 不一致。")
    if not dry_run and STRICT_SCALE_FOR_LIVE_REPLAY:
        raise RuntimeError("真机回放被阻止：demo scale 和当前 config scale 不一致。")


# =============================================================================
# 2. observation / gripper / pose 工具
# =============================================================================

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
            "ee_pose_right",
            "tcp_pose_right",
        ])
    else:
        preferred_keys.extend([
            "left_ee_pose",
            "left/tcp_pose",
            "left_tcp_pose",
            "state/left_ee_pose",
            "state/left/tcp_pose",
            "pose_ee_arm_left",
            "ee_pose_left",
            "tcp_pose_left",
        ])

    preferred_keys.extend([
        "ee_pose",
        "tcp_pose",
        "pose_ee",
        "cartesian_pose",
        "eef_pose",
        "end_effector_pose",
    ])

    for key in preferred_keys:
        if key in state:
            arr = to_1d_array(state[key])
            if arr is not None and arr.size >= 6:
                return arr[:7] if arr.size >= 7 else arr[:6]

    for key, value in state.items():
        k = str(key).lower()
        if ("pose" in k or "tcp" in k or "ee" in k or "eef" in k) and "gripper" not in k:
            arr = to_1d_array(value)
            if arr is not None and arr.size >= 6:
                return arr[:7] if arr.size >= 7 else arr[:6]

    return None


def extract_ee_pose(obs: Dict[str, Any], arm_side: str = "right") -> Optional[np.ndarray]:
    if obs is None or not isinstance(obs, dict) or "state" not in obs:
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


def pose_to_pos_quat(pose: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
    if prev_pose is None or next_pose is None:
        return None

    prev_pos, prev_quat = pose_to_pos_quat(prev_pose)
    next_pos, next_quat = pose_to_pos_quat(next_pose)

    if prev_pos is None or next_pos is None or prev_quat is None or next_quat is None:
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
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    pos_delta_m = a[:3] * float(pos_scale)
    rot_delta_rad = a[3:6] * float(rot_scale)
    return pos_delta_m.astype(np.float32), rot_delta_rad.astype(np.float32)


def compute_error_metrics(
    observed_delta_action6: Optional[np.ndarray],
    reference_action6: Optional[np.ndarray],
    pos_scale: float,
    rot_scale: float,
) -> Optional[Dict[str, float]]:
    if observed_delta_action6 is None or reference_action6 is None:
        return None

    obs_a = np.asarray(observed_delta_action6, dtype=np.float32).reshape(-1)
    ref_a = np.asarray(reference_action6, dtype=np.float32).reshape(-1)

    if obs_a.size < 6 or ref_a.size < 6:
        return None

    err = obs_a[:6] - ref_a[:6]
    pos_err_m = np.abs(err[:3]) * float(pos_scale)
    rot_err_rad = np.abs(err[3:6]) * float(rot_scale)

    return {
        "norm_max": float(np.max(np.abs(err))),
        "norm_l2": float(np.linalg.norm(err)),
        "pos_max_m": float(np.max(pos_err_m)),
        "pos_l2_m": float(np.linalg.norm(pos_err_m)),
        "rot_max_rad": float(np.max(rot_err_rad)),
        "rot_l2_rad": float(np.linalg.norm(rot_err_rad)),
    }


def format_error_metrics(metrics: Optional[Dict[str, float]]) -> str:
    if metrics is None:
        return "N/A"
    return (
        f"norm_max={metrics['norm_max']:.4f}, "
        f"pos_max={metrics['pos_max_m'] * 1000:.2f}mm, "
        f"rot_max={np.rad2deg(metrics['rot_max_rad']):.2f}deg"
    )


def error_is_large(metrics: Optional[Dict[str, float]]) -> bool:
    if metrics is None:
        return False
    return bool(
        metrics["pos_max_m"] > LIVE_WARN_POS_ERR_M
        or metrics["rot_max_rad"] > LIVE_WARN_ROT_ERR_RAD
        or metrics["norm_max"] > LIVE_WARN_ACTION_ERR_NORM
    )


def get_transition_obs_pair(trans: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    obs = None
    next_obs = None

    for k in ("observations", "observation", "obs"):
        if k in trans:
            obs = trans[k]
            break

    for k in ("next_observations", "next_observation", "next_obs"):
        if k in trans:
            next_obs = trans[k]
            break

    return obs, next_obs


def extract_gripper_feedback(obs: Dict[str, Any], arm_side: str = "right") -> Optional[float]:
    if obs is None or not isinstance(obs, dict) or "state" not in obs:
        return None

    state = obs["state"]

    if isinstance(state, dict):
        preferred_keys = []
        if arm_side == "right":
            preferred_keys.extend([
                "right_gripper",
                "state/right_gripper",
                "right/gripper",
                "gripper_right",
            ])
        else:
            preferred_keys.extend([
                "left_gripper",
                "state/left_gripper",
                "left/gripper",
                "gripper_left",
            ])

        preferred_keys.extend(["gripper", "gripper_pos", "gripper_position"])

        for key in preferred_keys:
            if key in state:
                arr = to_1d_array(state[key])
                if arr is not None:
                    return float(arr[-1])

        for key, value in state.items():
            if "gripper" in str(key).lower():
                arr = to_1d_array(value)
                if arr is not None:
                    return float(arr[-1])
        return None

    arr = to_1d_array(state)
    if arr is None:
        return None
    return float(arr[-1])


def classify_gripper_feedback(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value <= HW_CLOSE_MAX:
        return "closed"
    if value >= HW_OPEN_MIN:
        return "open"
    return "middle/unclear"


def describe_gripper_action(g: float) -> str:
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def quantize_gripper(g: float) -> float:
    g = float(g)
    if g <= -0.5:
        return -1.0
    if g >= 0.5:
        return 1.0
    return 0.0


def get_env_gripper_memory(env) -> Optional[Dict[str, float]]:
    if env is None:
        return None
    try:
        base = env.unwrapped
        mem = getattr(base, "_last_hw_gripper_cmd", None)
        if isinstance(mem, dict):
            out = {}
            for k, v in mem.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
            return out
    except Exception:
        pass
    return None


def print_state_keys(obs: Dict[str, Any]) -> None:
    print("=" * 100)
    print("reset 后 observation state 检查")
    print("=" * 100)
    if obs is None or not isinstance(obs, dict) or "state" not in obs:
        print("⚠️ obs 中没有 state。")
        return

    state = obs["state"]
    if isinstance(state, dict):
        print(f"state keys: {list(state.keys())}")
        for k, v in state.items():
            if "gripper" in str(k).lower():
                arr = to_1d_array(v)
                print(f"  gripper-like key: {k}, value={None if arr is None else arr.tolist()}")
    else:
        arr = to_1d_array(state)
        print(f"state ndarray shape={None if arr is None else arr.shape}, value_head={None if arr is None else arr[:12].tolist()}")
        if arr is not None:
            print(f"  fallback gripper feedback = last dim = {float(arr[-1]):.3f}")


def refresh_obs_after_sleep(env, fallback_obs, sleep_sec: float):
    """
    保留接口，但不额外 sleep、不额外 _get_sync_obs。
    obs 直接使用 env.step(action) 返回值。
    """
    return fallback_obs, False, "env_step_obs_only"


def force_script_mode_if_possible(env) -> None:
    if not FORCE_SCRIPT_MODE or env is None:
        return

    print("🤖 尝试强制进入脚本/AI动作测试模式，避免 VRInterventionWrapper 覆盖 replay action...")

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


def make_env():
    return env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=USE_CLASSIFIER,
    )


# =============================================================================
# 3. transition / action
# =============================================================================

def get_transition_action(trans: Dict[str, Any]) -> np.ndarray:
    for key in ("actions", "action"):
        if key in trans:
            arr = np.asarray(trans[key], dtype=np.float32).reshape(-1)
            return arr
    raise KeyError("transition 中找不到 actions/action 字段。")


def prepare_action_for_step(raw_action: np.ndarray, action_space=None) -> Tuple[np.ndarray, bool]:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1).copy()
    before = action.copy()

    if CLIP_ACTION_FOR_SAFETY:
        if action_space is not None:
            try:
                low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
                if low.shape == action.shape and high.shape == action.shape:
                    action = np.clip(action, low, high)
                else:
                    action = np.clip(action, -1.0, 1.0)
            except Exception:
                action = np.clip(action, -1.0, 1.0)
        else:
            action = np.clip(action, -1.0, 1.0)

    if QUANTIZE_GRIPPER_FOR_SAFETY and action.size >= 7:
        action[6] = np.float32(quantize_gripper(float(action[6])))

    changed = not np.allclose(before, action, atol=1e-6)
    return action.astype(np.float32), bool(changed)


def info_has_intervention(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    keys = [
        "intervene_action",
        "intervention_action",
        "intervene",
        "intervened",
        "vr_intervention",
    ]
    for k in keys:
        if k in info:
            v = info[k]
            if isinstance(v, bool):
                return bool(v)
            if v is not None:
                return True
    return False


# =============================================================================
# 4. CSV / summary
# =============================================================================

def save_csv(rows: List[Dict[str, Any]]) -> None:
    if not SAVE_CSV or not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV saved: {os.path.abspath(CSV_PATH)}")


def summarize_rows(rows: List[Dict[str, Any]], title: str = "Replay diagnostics summary") -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"rows: {len(rows)}")

    if not rows:
        return

    close_rows = [r for r in rows if float(r["action_6"]) <= -0.5]
    hold_rows = [r for r in rows if abs(float(r["action_6"])) < 0.5]
    open_rows = [r for r in rows if float(r["action_6"]) >= 0.5]

    print(f"gripper close/hold/open: {len(close_rows)} / {len(hold_rows)} / {len(open_rows)}")

    if close_rows:
        print("close action steps:", [(r["demo_global_index"], r["demo_step"]) for r in close_rows[:20]], "..." if len(close_rows) > 20 else "")
    if open_rows:
        print("open action steps :", [(r["demo_global_index"], r["demo_step"]) for r in open_rows[:20]], "..." if len(open_rows) > 20 else "")

    vals = []
    for r in rows:
        v = r.get("next_gripper_feedback", "")
        if v != "":
            vals.append(float(v))
    if vals:
        arr = np.asarray(vals, dtype=np.float32)
        print(f"feedback min/mean/max: {float(np.min(arr)):.3f} / {float(np.mean(arr)):.3f} / {float(np.max(arr)):.3f}")
        print(f"closed count <= {HW_CLOSE_MAX}: {int(np.sum(arr <= HW_CLOSE_MAX))}")
        print(f"open count   >= {HW_OPEN_MIN}: {int(np.sum(arr >= HW_OPEN_MIN))}")
        print(f"middle count           : {int(np.sum((arr > HW_CLOSE_MAX) & (arr < HW_OPEN_MIN)))}")

    rewards = [r for r in rows if r.get("reward", "") not in ("", None) and float(r["reward"]) != 0.0]
    dones = [r for r in rows if bool(r.get("done", False))]
    truncs = [r for r in rows if bool(r.get("truncated", False))]

    if rewards:
        print(f"\nfirst live reward != 0: demo={rewards[0]['demo_global_index']}, step={rewards[0]['demo_step']}, reward={rewards[0]['reward']}")
    if dones:
        print(f"first live done=True  : demo={dones[0]['demo_global_index']}, step={dones[0]['demo_step']}")
    if truncs:
        print(f"first live truncated=True: demo={truncs[0]['demo_global_index']}, step={truncs[0]['demo_step']}")

    live_pos_errs = []
    live_rot_errs = []
    live_norm_errs = []
    saturated_rows = []
    for r in rows:
        if r.get("live_err_pos_max_m", "") != "":
            live_pos_errs.append(float(r["live_err_pos_max_m"]))
        if r.get("live_err_rot_max_rad", "") != "":
            live_rot_errs.append(float(r["live_err_rot_max_rad"]))
        if r.get("live_err_norm_max", "") != "":
            live_norm_errs.append(float(r["live_err_norm_max"]))

        vals_a = [abs(float(r[f"action_{j}"])) for j in range(6)]
        if max(vals_a) >= SATURATION_THRESHOLD:
            saturated_rows.append((r["demo_global_index"], r["demo_step"]))

    print("\n手臂动作误差 summary：")
    if live_pos_errs:
        a = np.asarray(live_pos_errs, dtype=np.float32)
        print(f"live pos err max/mean/p95: {float(np.max(a))*1000:.2f} / {float(np.mean(a))*1000:.2f} / {float(np.percentile(a,95))*1000:.2f} mm")
    else:
        print("live pos err: N/A，未能从 obs 中提取 EE pose")

    if live_rot_errs:
        a = np.asarray(live_rot_errs, dtype=np.float32)
        print(f"live rot err max/mean/p95: {np.rad2deg(float(np.max(a))):.2f} / {np.rad2deg(float(np.mean(a))):.2f} / {np.rad2deg(float(np.percentile(a,95))):.2f} deg")
    else:
        print("live rot err: N/A，未能从 obs 中提取 EE pose")

    if live_norm_errs:
        a = np.asarray(live_norm_errs, dtype=np.float32)
        print(f"live norm err max/mean/p95: {float(np.max(a)):.3f} / {float(np.mean(a)):.3f} / {float(np.percentile(a,95)):.3f}")

    print(f"action saturation rows |action_0:5| >= {SATURATION_THRESHOLD}: {len(saturated_rows)}")
    if saturated_rows:
        print("first saturated rows:", saturated_rows[:30], "..." if len(saturated_rows) > 30 else "")

    print("\n判断建议：")
    print("  A. 如果 live_delta_action[:6] 普遍小于 action[:6]，说明机器人没完全跟上，优先增加 ACTION_SETTLE_SEC 或降低 HZ。")
    print("  B. 如果 action 经常贴近 ±1，说明当前 POS_SCALE/ROT_SCALE 可能偏小。")
    print("  C. 如果 live error 很大但 action 不饱和，检查 reset 初始位姿、接触/碰撞、VR/外部 publisher、底层 IK 跟随。")
    print("  D. 本脚本每条 demo 前都会 reset；如果某条 demo 失败，不会继续把下一条 demo 接在错误状态上。")


# =============================================================================
# 5. replay one episode
# =============================================================================

def reset_before_demo(env, demo_idx: int):
    if DRY_RUN:
        return {}

    print("\n" + "=" * 100)
    print(f"🔄 准备播放 demo #{demo_idx}: 开始 reset")
    print("=" * 100)

    obs, reset_info = env.reset()

    force_script_mode_if_possible(env)

    print_state_keys(obs)
    initial_pose = extract_ee_pose(obs)
    print("=" * 100)
    print("reset 后初始 EE pose 检查")
    print("=" * 100)
    print(f"initial EE pose: {None if initial_pose is None else np.asarray(initial_pose).tolist()}")

    initial_feedback = extract_gripper_feedback(obs)
    print("=" * 100)
    print("reset 后初始 gripper feedback")
    print("=" * 100)
    print(f"initial gripper feedback: {initial_feedback} ({classify_gripper_feedback(initial_feedback)})")
    print(f"initial _last_hw_gripper_cmd: {get_env_gripper_memory(env)}")

    return obs


def replay_one_demo(
    env,
    obs,
    episode: Dict[str, Any],
    demo_global_index: int,
    current_pos: float,
    current_rot: float,
    current_hz: float,
) -> Tuple[List[Dict[str, Any]], str]:
    transitions = episode["transitions"]
    metadata = episode.get("metadata", {}) or {}
    pkl_format = episode.get("pkl_format", "")

    demo_pos, demo_rot, demo_hz, scale_source = resolve_demo_scales(
        metadata,
        current_pos,
        current_rot,
        current_hz,
    )

    check_scale_consistency(demo_pos, demo_rot, current_pos, current_rot, scale_source, DRY_RUN)

    if MAX_STEPS_PER_DEMO is not None:
        selected = transitions[:int(MAX_STEPS_PER_DEMO)]
    else:
        selected = transitions

    print("=" * 100)
    print(f"▶️ 播放 demo #{demo_global_index}")
    print("=" * 100)
    print(f"file              : {episode['file_path']}")
    print(f"file index         : {episode['file_global_index']}")
    print(f"local episode index: {episode['local_episode_index']}")
    print(f"transition range   : [{episode['start_index']}, {episode['end_index_exclusive']})")
    print(f"episode len        : {len(transitions)}")
    print(f"selected len       : {len(selected)}")
    print(f"pkl format         : {pkl_format}")

    if WAIT_ENTER_BEFORE_EACH_DEMO and not DRY_RUN:
        input(f"确认安全后按 Enter 开始播放 demo #{demo_global_index} ...")

    rows = []
    stop_reason = "episode_finished"

    for local_i, trans in enumerate(selected):
        demo_step = local_i
        raw_action = get_transition_action(trans)
        raw_min = float(np.min(raw_action)) if raw_action.size else 0.0
        raw_max = float(np.max(raw_action)) if raw_action.size else 0.0

        action_space = None if env is None else getattr(env, "action_space", None)
        action, changed_by_safety = prepare_action_for_step(raw_action, action_space)

        if action.size < 7:
            raise ValueError(f"当前脚本期望 action 至少 7 维，但 demo={demo_global_index}, step={demo_step} action.shape={action.shape}")

        g = float(action[6])
        g_desc = describe_gripper_action(g)

        demo_obs, demo_next_obs = get_transition_obs_pair(trans)

        live_prev_pose = extract_ee_pose(obs)
        demo_prev_pose = extract_ee_pose(demo_obs)
        demo_next_pose = extract_ee_pose(demo_next_obs)

        expected_pos_delta_m, expected_rot_delta_rad = compute_real_delta_from_action(
            action[:6],
            current_pos,
            current_rot,
        )

        demo_delta_action6 = compute_delta_action_from_poses(
            demo_prev_pose,
            demo_next_pose,
            demo_pos,
            demo_rot,
            clip=False,
        )

        prev_feedback = extract_gripper_feedback(obs)
        prev_state = classify_gripper_feedback(prev_feedback)
        prev_mem = get_env_gripper_memory(env)

        if DRY_RUN:
            reward = None
            done = False
            truncated = False
            info = {}
            next_obs = obs
            refresh_success = False
            refresh_source = "dry_run"
            intervention_overrode_action = False
        else:
            step_obs, reward, done, truncated, info = env.step(action)
            intervention_overrode_action = info_has_intervention(info)
            next_obs, refresh_success, refresh_source = refresh_obs_after_sleep(env, step_obs, STEP_SLEEP_SEC)

        next_feedback = extract_gripper_feedback(next_obs)
        next_state = classify_gripper_feedback(next_feedback)
        next_mem = get_env_gripper_memory(env)

        live_next_pose = extract_ee_pose(next_obs)
        live_delta_action6 = compute_delta_action_from_poses(
            live_prev_pose,
            live_next_pose,
            current_pos,
            current_rot,
            clip=False,
        )

        live_error_metrics = compute_error_metrics(
            live_delta_action6,
            action[:6],
            current_pos,
            current_rot,
        )

        demo_error_metrics = compute_error_metrics(
            demo_delta_action6,
            action[:6],
            demo_pos,
            demo_rot,
        )

        delta_feedback = ""
        if prev_feedback is not None and next_feedback is not None:
            delta_feedback = float(next_feedback - prev_feedback)

        demo_terminal = is_episode_terminal(trans)
        demo_reward = transition_reward(trans)
        demo_done = transition_done(trans)
        demo_mask = transition_mask(trans)

        row = {
            "demo_global_index": demo_global_index,
            "demo_file_index": episode["file_global_index"],
            "demo_local_episode_index": episode["local_episode_index"],
            "demo_file": os.path.basename(episode["file_path"]),
            "demo_step": demo_step,
            "demo_transition_index": episode["start_index"] + local_i,
            "pkl_format": pkl_format,
            "demo_scale_source": scale_source,
            "demo_pos_scale": demo_pos,
            "demo_rot_scale": demo_rot,
            "current_pos_scale": current_pos,
            "current_rot_scale": current_rot,
            "raw_action_min": raw_min,
            "raw_action_max": raw_max,
            "action_changed_by_safety": bool(changed_by_safety),
            "intervention_overrode_action": bool(intervention_overrode_action),

            "action_0": float(action[0]),
            "action_1": float(action[1]),
            "action_2": float(action[2]),
            "action_3": float(action[3]),
            "action_4": float(action[4]),
            "action_5": float(action[5]),
            "action_6": float(action[6]),
            "action_6_desc": g_desc,

            "expected_dx_m": float(expected_pos_delta_m[0]),
            "expected_dy_m": float(expected_pos_delta_m[1]),
            "expected_dz_m": float(expected_pos_delta_m[2]),
            "expected_rx_rad": float(expected_rot_delta_rad[0]),
            "expected_ry_rad": float(expected_rot_delta_rad[1]),
            "expected_rz_rad": float(expected_rot_delta_rad[2]),

            "live_prev_pose_found": live_prev_pose is not None,
            "live_next_pose_found": live_next_pose is not None,
            "demo_prev_pose_found": demo_prev_pose is not None,
            "demo_next_pose_found": demo_next_pose is not None,

            "live_err_norm_max": "" if live_error_metrics is None else float(live_error_metrics["norm_max"]),
            "live_err_norm_l2": "" if live_error_metrics is None else float(live_error_metrics["norm_l2"]),
            "live_err_pos_max_m": "" if live_error_metrics is None else float(live_error_metrics["pos_max_m"]),
            "live_err_pos_l2_m": "" if live_error_metrics is None else float(live_error_metrics["pos_l2_m"]),
            "live_err_rot_max_rad": "" if live_error_metrics is None else float(live_error_metrics["rot_max_rad"]),
            "live_err_rot_l2_rad": "" if live_error_metrics is None else float(live_error_metrics["rot_l2_rad"]),

            "demo_err_norm_max": "" if demo_error_metrics is None else float(demo_error_metrics["norm_max"]),
            "demo_err_pos_max_m": "" if demo_error_metrics is None else float(demo_error_metrics["pos_max_m"]),
            "demo_err_rot_max_rad": "" if demo_error_metrics is None else float(demo_error_metrics["rot_max_rad"]),

            "prev_gripper_feedback": "" if prev_feedback is None else float(prev_feedback),
            "next_gripper_feedback": "" if next_feedback is None else float(next_feedback),
            "delta_gripper_feedback": delta_feedback,
            "prev_gripper_state": prev_state,
            "next_gripper_state": next_state,
            "prev_hw_gripper_cmd_right": "" if not prev_mem or "right" not in prev_mem else float(prev_mem["right"]),
            "next_hw_gripper_cmd_right": "" if not next_mem or "right" not in next_mem else float(next_mem["right"]),
            "prev_hw_gripper_cmd_left": "" if not prev_mem or "left" not in prev_mem else float(prev_mem["left"]),
            "next_hw_gripper_cmd_left": "" if not next_mem or "left" not in next_mem else float(next_mem["left"]),

            "demo_reward": float(demo_reward),
            "demo_done": bool(demo_done),
            "demo_mask": "" if demo_mask is None else float(demo_mask),
            "demo_terminal": bool(demo_terminal),

            "refresh_success": bool(refresh_success),
            "refresh_source": refresh_source,
            "reward": "" if reward is None else float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
        }

        for j in range(6):
            row[f"live_delta_action_{j}"] = "" if live_delta_action6 is None else float(live_delta_action6[j])
            row[f"live_action_err_{j}"] = "" if live_delta_action6 is None else float(live_delta_action6[j] - action[j])
            row[f"demo_delta_action_{j}"] = "" if demo_delta_action6 is None else float(demo_delta_action6[j])

        rows.append(row)

        should_print = (
            PRINT_EVERY is not None
            and PRINT_EVERY > 0
            and (local_i % PRINT_EVERY == 0)
        )

        if g_desc != "hold(0)":
            should_print = True

        if should_print:
            print("-" * 100)
            print(f"demo {demo_global_index} step {demo_step}/{len(selected)-1}")
            print(f"action[:6]                 : {np.round(action[:6], 4).tolist()}")
            print(f"action[6]                  : {g_desc} ({g})")
            print(f"expected pos delta (mm)    : {np.round(expected_pos_delta_m * 1000.0, 3).tolist()}")
            print(f"expected rot delta (deg)   : {np.round(np.rad2deg(expected_rot_delta_rad), 3).tolist()}")
            print(f"live delta action[:6]      : {None if live_delta_action6 is None else np.round(live_delta_action6, 4).tolist()}")
            print(f"live error                 : {format_error_metrics(live_error_metrics)}")
            if error_is_large(live_error_metrics):
                print("⚠️ live error 偏大：可能是动作未完全跟随、reset 初始偏差、接触扰动或底层控制延迟。")
            print(f"demo delta action[:6]      : {None if demo_delta_action6 is None else np.round(demo_delta_action6, 4).tolist()}")
            print(f"demo error                 : {format_error_metrics(demo_error_metrics)}")
            print(f"prev gripper feedback      : {prev_feedback} ({prev_state})")
            print(f"next gripper feedback      : {next_feedback} ({next_state})")
            print(f"env _last_hw_gripper_cmd   : before={prev_mem}, after={next_mem}")
            print(f"demo reward/done/mask/term : {demo_reward} / {demo_done} / {demo_mask} / {demo_terminal}")
            print(f"live reward/done/truncated : {reward} / {done} / {truncated}")
            print(f"intervention_overrode      : {intervention_overrode_action}")

        obs = next_obs

        if not DRY_RUN:
            if STOP_ON_LIVE_REWARD and reward is not None and float(reward) != 0.0:
                stop_reason = f"live_reward_{reward}"
                print(f"✅ live reward={reward} at demo={demo_global_index}, step={demo_step}，停止当前 demo，准备 reset。")
                break
            if STOP_ON_LIVE_DONE and bool(done):
                stop_reason = "live_done"
                print(f"✅ live done=True at demo={demo_global_index}, step={demo_step}，停止当前 demo，准备 reset。")
                break
            if STOP_ON_LIVE_TRUNCATED and bool(truncated):
                stop_reason = "live_truncated"
                print(f"✅ live truncated=True at demo={demo_global_index}, step={demo_step}，停止当前 demo，准备 reset。")
                break

        if STOP_ON_DEMO_TERMINAL and demo_terminal:
            stop_reason = "demo_terminal"
            print(f"✅ demo terminal at demo={demo_global_index}, step={demo_step}，当前 demo 播放结束，准备 reset。")
            break

    print("=" * 100)
    print(f"demo #{demo_global_index} 播放结束: stop_reason={stop_reason}, played_steps={len(rows)}")
    print("=" * 100)

    summarize_rows(rows, title=f"Demo #{demo_global_index} diagnostics summary")
    return rows, stop_reason


# =============================================================================
# 6. main
# =============================================================================

def main():
    print("=" * 100)
    print("Episode-by-episode Demo Replay + Reset + Diagnostics")
    print("=" * 100)
    print(f"DEMO_PATH                    : {DEMO_PATH}")
    print(f"FILE_INDEX                   : {FILE_INDEX}")
    print(f"DEMO_INDEX_START             : {DEMO_INDEX_START}")
    print(f"DEMO_INDEX_END               : {DEMO_INDEX_END}")
    print(f"MAX_STEPS_PER_DEMO           : {MAX_STEPS_PER_DEMO}")
    print(f"DRY_RUN                      : {DRY_RUN}")
    print("TIMING_MODE                  : env.step() only; use config.ACTION_SETTLE_SEC in env")
    print(f"WAIT_ENTER_BEFORE_EACH_DEMO  : {WAIT_ENTER_BEFORE_EACH_DEMO}")
    print(f"FORCE_SCRIPT_MODE            : {FORCE_SCRIPT_MODE}")
    print(f"USE_CLASSIFIER               : {USE_CLASSIFIER}")
    print(f"STOP_ON_LIVE_REWARD/DONE/TRUNC: {STOP_ON_LIVE_REWARD} / {STOP_ON_LIVE_DONE} / {STOP_ON_LIVE_TRUNCATED}")
    print(f"STOP_ON_DEMO_TERMINAL        : {STOP_ON_DEMO_TERMINAL}")
    print(f"CSV_PATH                     : {CSV_PATH}")

    episodes, tails, inventory = load_all_demo_episodes(DEMO_PATH, FILE_INDEX)
    print_episode_inventory(episodes, tails, inventory)

    if not episodes:
        raise RuntimeError("没有识别到完整 demo。请检查 pkl 里是否有 dones=True / masks=0 / reward>0。")

    start = int(DEMO_INDEX_START)
    end = len(episodes) if DEMO_INDEX_END is None else min(int(DEMO_INDEX_END), len(episodes))
    if start < 0 or start >= len(episodes):
        raise IndexError(f"DEMO_INDEX_START={start} 越界，总 demos={len(episodes)}")
    if end <= start:
        raise ValueError(f"DEMO_INDEX_END={end} 必须大于 DEMO_INDEX_START={start}")

    selected_episodes = episodes[start:end]
    print("=" * 100)
    print("Replay demo 范围")
    print("=" * 100)
    print(f"selected demos: {len(selected_episodes)}")
    print(f"global demo index range: [{start}, {end})")

    env = None
    current_pos, current_rot, current_hz = 0.018, 0.05, 15.0

    if not DRY_RUN:
        print("\n🌍 正在创建真实环境...")
        env = make_env()
        current_pos = get_config_value(env, "POS_SCALE", 0.018)
        current_rot = get_config_value(env, "ROT_SCALE", 0.05)
        current_hz = get_config_value(env, "HZ", 15.0)
        print(f"current POS_SCALE={current_pos}, ROT_SCALE={current_rot}, HZ={current_hz}")

    all_rows = []
    stop_reasons = []

    for ep_offset, episode in enumerate(selected_episodes):
        demo_global_index = start + ep_offset

        obs = reset_before_demo(env, demo_global_index)

        rows, reason = replay_one_demo(
            env=env,
            obs=obs,
            episode=episode,
            demo_global_index=demo_global_index,
            current_pos=current_pos,
            current_rot=current_rot,
            current_hz=current_hz,
        )

        all_rows.extend(rows)
        stop_reasons.append({
            "demo_global_index": demo_global_index,
            "stop_reason": reason,
            "played_steps": len(rows),
        })

        if WAIT_ENTER_BEFORE_RESET_AFTER_DEMO and not DRY_RUN and ep_offset < len(selected_episodes) - 1:
            input(f"demo #{demo_global_index} 已结束。按 Enter reset 并播放下一条 demo...")

        # 注意：下一轮循环开头会 reset。
        if ep_offset < len(selected_episodes) - 1:
            print("\n" + "#" * 100)
            print(f"✅ demo #{demo_global_index} 完成。下一步将 reset，然后播放 demo #{demo_global_index + 1}")
            print("#" * 100)

    print("=" * 100)
    print("全部 selected demos 播放完成")
    print("=" * 100)
    print("stop reasons:")
    for item in stop_reasons:
        print(f"  demo#{item['demo_global_index']}: {item['stop_reason']}, played_steps={item['played_steps']}")

    summarize_rows(all_rows, title="All selected demos diagnostics summary")
    save_csv(all_rows)

    print("=" * 100)
    print("完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
