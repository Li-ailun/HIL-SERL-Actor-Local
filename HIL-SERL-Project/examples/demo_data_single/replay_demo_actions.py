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






"""
replay_demo_full_motion_diagnostics_env_timing.py

用途：
  在播放 demo action 的同时，完整诊断手臂动作缩放、真实执行误差和夹爪链路。

它会记录：
  1. demo 里的 action[:6]：归一化末端增量动作
  2. action[:6] 根据当前 POS_SCALE / ROT_SCALE 对应的理论真实位移/旋转
  3. env.step(action) 前后的真实 EE pose
  4. live obs delta 反推得到的真实执行 normalized action
  5. live_delta_action - action 的 normalized error / 物理量 error
  6. demo 里的 action[6]：close(-1) / hold(0) / open(+1)
  7. env.step(action) 前后的真实 gripper feedback
  8. env.unwrapped._last_hw_gripper_cmd 的 right/left 记忆值
  9. reward / done / truncated
  10. refresh_obs_after_sleep 是否成功

适用场景：
  - 你确认独立 gripper 测试正常，但播放 demo 时夹爪仍然“要闭合又张开”
  - 需要判断是动作缩放不合适、真实执行没跟上、VR/外部控制覆盖、夹爪 hold memory 异常，还是成功后脚本继续播放造成移动

注意：
  - demo 里的 action[:6] 是归一化动作，不要在脚本外部乘 POS_SCALE / ROT_SCALE。
  - 本脚本不会修改 demo、env、train_rlpd。
  - 第一次可设 DRY_RUN=True；真机诊断设 DRY_RUN=False。
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
DEMO_FILE_INDEX = 0

START_TRANSITION = 0
MAX_STEPS = None  # 建议第一次真机诊断先设 130，只覆盖 close 附近；确认后再 None。

DRY_RUN = False

# 统一使用 env.step() 内部等待/刷新逻辑。
# 重要：
# - 如果 GalaxeaArmEnv.step() 里已经加入 ACTION_SETTLE_SEC，
#   replay 脚本外层就不要再额外 sleep / _get_sync_obs。
# - 这样 demo replay 和 actor 训练/执行走完全一样的 env 时序。
STEP_SLEEP_SEC = 0.0
REFRESH_OBS_AFTER_SLEEP = False

WAIT_ENTER_BEFORE_REPLAY = True
FORCE_SCRIPT_MODE = True
USE_CLASSIFIER = False

CLIP_ACTION_FOR_SAFETY = True
QUANTIZE_GRIPPER_FOR_SAFETY = True

# 成功 / 终止后是否停止继续播放。
# replay 诊断时建议打开，避免 reward=1 / done=True 后仍然继续移动。
STOP_ON_REWARD = True
STOP_ON_DONE = True
STOP_ON_TRUNCATED = True

# 如果你只是想看完整 demo 后续动作，可临时设 False。
PRINT_EVERY = 1

SAVE_CSV = True
CSV_PATH = "./demo_replay_full_motion_diagnostics_env_timing.csv"

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
# 这些只用于打印/summary，不会改变动作。
LIVE_WARN_POS_ERR_M = 0.005       # 5 mm
LIVE_WARN_ROT_ERR_RAD = 0.05      # about 2.86 deg
LIVE_WARN_ACTION_ERR_NORM = 0.5   # normalized action error

# action 饱和统计阈值。
SATURATION_THRESHOLD = 0.98


# =============================================================================
# 1. demo / metadata
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


def load_demo_payload(path: str, file_index: int) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    files = sorted_pkl_files(path)
    if file_index < 0 or file_index >= len(files):
        raise IndexError(f"DEMO_FILE_INDEX={file_index} 越界，共找到 {len(files)} 个 pkl 文件。")

    file_path = files[file_index]
    with open(file_path, "rb") as f:
        data = pkl.load(f)

    if isinstance(data, dict) and "transitions" in data:
        transitions = data["transitions"]
        metadata = data.get("metadata", {}) or {}
        fmt = "dict_with_metadata" if metadata else "dict_without_metadata"
    elif isinstance(data, list):
        transitions = data
        metadata = {}
        fmt = "legacy_list"
    else:
        raise ValueError(f"无法识别 pkl 格式: type={type(data)}")

    if len(transitions) == 0:
        raise ValueError(f"pkl 为空: {file_path}")

    return file_path, transitions, metadata, fmt


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

    # env_config 是 TrainingConfig，实际 POS_SCALE 通常在硬件 env_cfg 里；
    # 这里仅兜底。
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
# 2. observation / gripper 工具
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
    """
    从 obs["state"] 字典中提取 EE pose。
    优先右臂 key；如果 key 名不同，会 fallback 搜索包含 pose/tcp/ee 的字段。
    返回：
      7 维: xyz + quat(x,y,z,w)
      或 6 维: xyz + euler/rpy
    """
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
    """
    从 observation 中提取 EE pose。
    兼容：
      obs["state"] 是 dict
      obs["state"] 是 array: [x,y,z,qx,qy,qz,qw,gripper...]
    """
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
    """
    支持：
      7 维: xyz + quat(x,y,z,w)
      6 维: xyz + euler/rpy(x,y,z)
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
    用真实 EE pose 差分反推 normalized action[:6]。
    语义：
      action[:3] = (next_pos - prev_pos) / POS_SCALE
      action[3:6] = rotvec(next_rot * prev_rot.inv()) / ROT_SCALE
    """
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
    """
    normalized action -> 理论真实位移 / 旋转增量。
    注意：这里只用于检测和打印，不会把 scaled action 传给 env.step。
    """
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
    """
    observed_delta_action6 和 reference_action6 都是 normalized action 语义。
    同时返回 normalized error 和物理量 error。
    """
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
    """
    尽量从 pkl transition 里取 demo obs / next_obs。
    不同脚本保存字段可能不同，所以做多 key 兼容。
    """
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


def safe_float(value):
    if value is None:
        return ""
    try:
        v = float(value)
        if not np.isfinite(v):
            return ""
        return v
    except Exception:
        return ""



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
    统一使用 env.step() 的内部时序。

    这个函数保留接口只是为了少改主循环：
    - 不额外 sleep
    - 不额外调用 env.unwrapped._get_sync_obs()
    - 直接使用 env.step(action) 返回的 step_obs

    如果需要防抖/等待/刷新，应该放到 GalaxeaArmEnv.step() 里，
    由 config.ACTION_SETTLE_SEC 控制。
    """
    return fallback_obs, False, "env_step_obs_only"


def force_script_mode_if_possible(env) -> None:
    if not FORCE_SCRIPT_MODE or env is None:
        return

    print("🤖 尝试强制进入脚本/AI动作测试模式，避免 VRInterventionWrapper 覆盖 action...")

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


def summarize_rows(rows: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print("Replay gripper diagnostics summary")
    print("=" * 100)
    print(f"rows: {len(rows)}")

    if not rows:
        return

    close_rows = [r for r in rows if float(r["action_6"]) <= -0.5]
    hold_rows = [r for r in rows if abs(float(r["action_6"])) < 0.5]
    open_rows = [r for r in rows if float(r["action_6"]) >= 0.5]

    print(f"gripper close/hold/open: {len(close_rows)} / {len(hold_rows)} / {len(open_rows)}")

    if close_rows:
        print("close action steps:", [r["global_step"] for r in close_rows[:20]], "..." if len(close_rows) > 20 else "")
    if open_rows:
        print("open action steps :", [r["global_step"] for r in open_rows[:20]], "..." if len(open_rows) > 20 else "")

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

    # 找 close 后 20 步
    if close_rows:
        first_close = close_rows[0]["global_step"]
        win = [r for r in rows if first_close <= r["global_step"] <= first_close + 25]
        print("\nfirst close 附近窗口：")
        for r in win:
            print(
                f"  step={r['global_step']:04d}, a6={r['action_6_desc']}, "
                f"fb {r['prev_gripper_feedback']} -> {r['next_gripper_feedback']} "
                f"({r['next_gripper_state']}), "
                f"hw_right {r['prev_hw_gripper_cmd_right']} -> {r['next_hw_gripper_cmd_right']}, "
                f"reward={r['reward']}, done={r['done']}"
            )

    rewards = [r for r in rows if r.get("reward", "") not in ("", None) and float(r["reward"]) != 0.0]
    dones = [r for r in rows if bool(r.get("done", False))]
    truncs = [r for r in rows if bool(r.get("truncated", False))]

    if rewards:
        print(f"\nreward != 0 first step: {rewards[0]['global_step']}, reward={rewards[0]['reward']}")
    if dones:
        print(f"done=True first step: {dones[0]['global_step']}")
    if truncs:
        print(f"truncated=True first step: {truncs[0]['global_step']}")

    # 手臂动作误差 summary
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
            saturated_rows.append(r["global_step"])

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
    print("  B. 如果 action 经常贴近 ±1，说明当前 POS_SCALE/ROT_SCALE 可能偏小；但改 scale 后要重新录 demo。")
    print("  C. 如果 live error 很大但 action 不饱和，检查 reset 初始位姿、接触/碰撞、VR/外部 publisher、底层 IK 跟随。")
    print("  D. 如果 live error 主要出现在接触后，可能是物体/夹爪/插入约束导致，不一定是 scale 问题。")
    print("  1. 如果 action_6 没有 open(+1)，但 gripper feedback 回到 open，说明不是 demo 标签打开。")
    print("  2. 如果 hold 阶段 next_hw_gripper_cmd_right 保持 10，但 feedback 升到 70/80，怀疑外部 publisher 或底层夹爪保持。")
    print("  3. 如果 hold 阶段 next_hw_gripper_cmd_right 变成 80，说明 replay 链路中 hold memory 被改回 open。")
    print("  4. 如果 reward/done 后仍有 rows，建议启用 STOP_ON_REWARD/STOP_ON_DONE，避免成功后继续移动。")


# =============================================================================
# 5. main
# =============================================================================

def main():
    print("=" * 100)
    print("Demo Replay Full Motion + Gripper Diagnostics")
    print("=" * 100)
    print(f"DEMO_PATH                  : {DEMO_PATH}")
    print(f"DEMO_FILE_INDEX            : {DEMO_FILE_INDEX}")
    print(f"START_TRANSITION           : {START_TRANSITION}")
    print(f"MAX_STEPS                  : {MAX_STEPS}")
    print(f"DRY_RUN                    : {DRY_RUN}")
    print(f"STEP_SLEEP_SEC             : {STEP_SLEEP_SEC}  # script-level extra sleep disabled")
    print(f"REFRESH_OBS_AFTER_SLEEP    : {REFRESH_OBS_AFTER_SLEEP}  # script-level refresh disabled")
    print("TIMING_MODE                : env.step() only; use config.ACTION_SETTLE_SEC in env")
    print(f"FORCE_SCRIPT_MODE          : {FORCE_SCRIPT_MODE}")
    print(f"USE_CLASSIFIER             : {USE_CLASSIFIER}")
    print(f"STOP_ON_REWARD/DONE/TRUNC  : {STOP_ON_REWARD} / {STOP_ON_DONE} / {STOP_ON_TRUNCATED}")
    print(f"CSV_PATH                   : {CSV_PATH}")

    file_path, transitions, metadata, pkl_format = load_demo_payload(DEMO_PATH, DEMO_FILE_INDEX)
    print("=" * 100)
    print("Demo 文件")
    print("=" * 100)
    print(f"file: {file_path}")
    print(f"format: {pkl_format}")
    print(f"num transitions: {len(transitions)}")
    if metadata:
        print(f"metadata keys: {list(metadata.keys())}")

    env = None
    obs = None

    if DRY_RUN:
        current_pos, current_rot, current_hz = 0.018, 0.05, 15.0
        obs = {}
    else:
        print("\n🌍 正在创建真实环境...")
        env = make_env()

        current_pos = get_config_value(env, "POS_SCALE", 0.018)
        current_rot = get_config_value(env, "ROT_SCALE", 0.05)
        current_hz = get_config_value(env, "HZ", 15.0)

        print("\n🔄 正在 reset 环境...")
        obs, reset_info = env.reset()

        force_script_mode_if_possible(env)

        print_state_keys(obs)
        initial_pose = extract_ee_pose(obs)
        print("=" * 100)
        print("初始 EE pose 检查")
        print("=" * 100)
        print(f"initial EE pose: {None if initial_pose is None else np.asarray(initial_pose).tolist()}")
        initial_feedback = extract_gripper_feedback(obs)
        print("=" * 100)
        print("初始 gripper feedback")
        print("=" * 100)
        print(f"initial gripper feedback: {initial_feedback} ({classify_gripper_feedback(initial_feedback)})")
        print(f"initial _last_hw_gripper_cmd: {get_env_gripper_memory(env)}")

    demo_pos, demo_rot, demo_hz, scale_source = resolve_demo_scales(
        metadata,
        current_pos,
        current_rot,
        current_hz,
    )
    check_scale_consistency(demo_pos, demo_rot, current_pos, current_rot, scale_source, DRY_RUN)

    end_transition = len(transitions)
    if MAX_STEPS is not None:
        end_transition = min(end_transition, START_TRANSITION + int(MAX_STEPS))

    selected = transitions[START_TRANSITION:end_transition]
    print("=" * 100)
    print("Replay 范围")
    print("=" * 100)
    print(f"selected transitions: {len(selected)}")
    print(f"global range: [{START_TRANSITION}, {end_transition})")

    if WAIT_ENTER_BEFORE_REPLAY and not DRY_RUN:
        input("确认机器人安全、夹爪附近无遮挡、VR 没有接管后，按 Enter 开始 replay 诊断...")

    rows: List[Dict[str, Any]] = []

    close_seen = 0
    open_seen = 0

    for local_i, trans in enumerate(selected):
        global_step = START_TRANSITION + local_i

        raw_action = get_transition_action(trans)
        raw_min = float(np.min(raw_action)) if raw_action.size else 0.0
        raw_max = float(np.max(raw_action)) if raw_action.size else 0.0

        action_space = None if env is None else getattr(env, "action_space", None)
        action, changed_by_safety = prepare_action_for_step(raw_action, action_space)

        if action.size < 7:
            raise ValueError(f"当前脚本期望 action 至少 7 维，但 step={global_step} action.shape={action.shape}")

        g = float(action[6])
        g_desc = describe_gripper_action(g)
        if g <= -0.5:
            close_seen += 1
        if g >= 0.5:
            open_seen += 1

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

        row = {
            "global_step": global_step,
            "local_step": local_i,
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

        # 夹爪事件附近强制打印
        if g_desc != "hold(0)":
            should_print = True

        # close 后 25 步内强制打印，方便观察闭合是否回弹
        if close_seen > 0 and close_seen <= 3:
            if len(rows) >= 2:
                first_close_step = next((r["global_step"] for r in rows if float(r["action_6"]) <= -0.5), None)
                if first_close_step is not None and global_step <= first_close_step + 25:
                    should_print = True

        if should_print:
            print("-" * 100)
            print(f"step {global_step} / local {local_i}")
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
            print(f"delta gripper feedback     : {delta_feedback}")
            print(f"env _last_hw_gripper_cmd   : before={prev_mem}, after={next_mem}")
            print(f"intervention_overrode      : {intervention_overrode_action}")
            print(f"obs_refresh_success        : {refresh_success}, source={refresh_source}")
            print(f"reward={reward}, done={done}, truncated={truncated}")
            print("说明：本版不在脚本外刷新 obs；obs 来自 env.step()。refresh_success=False 是正常的。")

        obs = next_obs

        if not DRY_RUN:
            if STOP_ON_REWARD and reward is not None and float(reward) != 0.0:
                print(f"✅ reward={reward} at step={global_step}，停止 replay，避免成功后继续移动。")
                break
            if STOP_ON_DONE and bool(done):
                print(f"✅ done=True at step={global_step}，停止 replay。")
                break
            if STOP_ON_TRUNCATED and bool(truncated):
                print(f"✅ truncated=True at step={global_step}，停止 replay。")
                break

    summarize_rows(rows)
    save_csv(rows)

    print("=" * 100)
    print("完成")
    print("=" * 100)


if __name__ == "__main__":
    main()

