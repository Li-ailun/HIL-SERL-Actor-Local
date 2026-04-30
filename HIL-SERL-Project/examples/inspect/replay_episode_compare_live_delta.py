#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_episode_compare_live_delta.py

功能：
  对 actor 保存的 episode_*.pkl 做动作增量对比：

  1) episode action:
     transition["actions"][:6]
     - actor 段：actor 直接输出的归一化增量
     - VR 段：episode 结束后由 feedback obs -> next_obs 转换得到的归一化增量

  2) saved feedback delta:
     episode pkl 内 observations["state"] -> next_observations["state"] 反推出来的归一化增量
     用来验证保存的 action 是否和 episode 内反馈位姿一致。

  3) live replay delta:
     真机 replay 时 live obs_before -> live next_obs_after 反推出来的归一化增量
     用来比较“想执行的增量”和“实机实际反馈增量”。

输出：
  - 终端逐步打印
  - CSV 明细
  - 汇总统计，按 actor / vr(intervention) 分开统计

典型用法：

  # 只做离线分析，不动机器人
  cd ~/HIL-SERL/HIL-SERL-Project/examples
  python inspect/replay_episode_compare_live_delta.py \
    --root ./rlpd_checkpoints_single \
    --which latest \
    --start 0 \
    --end 120 \
    --csv ./inspect/episode_latest_offline_compare.csv

  # 真机执行并比较 live feedback delta
  python inspect/replay_episode_compare_live_delta.py \
    --root ./rlpd_checkpoints_single \
    --which latest \
    --start 0 \
    --end 120 \
    --execute \
    --csv ./inspect/episode_latest_live_compare.csv

注意：
  - 不要把 action[:6] 手动乘 scale 后再传给 env.step。
  - env.step(action) 内部会按 POS_SCALE / ROT_SCALE 执行。
  - 本脚本只用 scale 做打印和 obs->next_obs 反推。
"""

import argparse
import csv
import glob
import math
import os
import pickle as pkl
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Path setup
# =============================================================================

THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.abspath(os.path.dirname(THIS_FILE))

# 脚本通常放 examples/inspect/，EXAMPLES_DIR = examples
if os.path.basename(THIS_DIR) == "inspect":
    EXAMPLES_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
else:
    EXAMPLES_DIR = THIS_DIR

PROJECT_ROOT = os.path.abspath(os.path.join(EXAMPLES_DIR, ".."))

for _p in [EXAMPLES_DIR, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def import_env_config():
    """兼容从 examples/ 或项目根目录运行。"""
    try:
        from galaxea_task.usb_pick_insertion_single.config import env_config
        return env_config
    except Exception:
        pass

    try:
        from examples.galaxea_task.usb_pick_insertion_single.config import env_config
        return env_config
    except Exception as e:
        raise ImportError(
            "无法导入单臂 USB task env_config。请确认脚本位于 examples/inspect/，"
            "并从 examples 目录运行，或检查 galaxea_task 路径。"
        ) from e


# =============================================================================
# General helpers
# =============================================================================

def safe_float(x, default=np.nan) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def safe_bool(x, default=False) -> bool:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return bool(default)
        return bool(arr.reshape(-1)[0])
    except Exception:
        return bool(default)


def to_1d_array(x, dtype=np.float32) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=dtype).reshape(-1)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def fmt_vec(v: Optional[np.ndarray], ndigits=3, max_len=None) -> str:
    if v is None:
        return "N/A"
    arr = np.asarray(v).reshape(-1)
    if max_len is not None:
        arr = arr[:max_len]
    return "[" + ",".join([f"{float(x):+.{ndigits}f}" for x in arr]) + "]"


def gripper_label(g: float) -> str:
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def get_reward(tr: Dict[str, Any]) -> float:
    return safe_float(tr.get("rewards", tr.get("reward", 0.0)), 0.0)


def get_done(tr: Dict[str, Any]) -> bool:
    return safe_bool(tr.get("dones", tr.get("done", False)), False)


def get_mask(tr: Dict[str, Any]) -> float:
    return safe_float(tr.get("masks", tr.get("mask", 1.0)), 1.0)


def get_grasp_penalty(tr: Dict[str, Any]) -> float:
    if "grasp_penalty" in tr:
        return safe_float(tr["grasp_penalty"], 0.0)
    info = tr.get("infos", tr.get("info", {}))
    if isinstance(info, dict) and "grasp_penalty" in info:
        return safe_float(info["grasp_penalty"], 0.0)
    return 0.0


def get_action(tr: Dict[str, Any]) -> np.ndarray:
    a = tr.get("actions", tr.get("action", None))
    if a is None:
        return np.zeros(7, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size < 7:
        out = np.zeros(7, dtype=np.float32)
        out[:a.size] = a
        return out
    return a[:7].astype(np.float32)


def sanitize_action_for_replay(action: np.ndarray, clip_action=True, quantize_gripper=True) -> Tuple[np.ndarray, bool]:
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if a.size != 7:
        raise ValueError(f"当前单臂回放要求 action shape=(7,), got {a.shape}")
    before = a.copy()
    if clip_action:
        a[:6] = np.clip(a[:6], -1.0, 1.0)
    if quantize_gripper:
        g = float(a[6])
        if g <= -0.5:
            a[6] = -1.0
        elif g >= 0.5:
            a[6] = 1.0
        else:
            a[6] = 0.0
    changed = bool(not np.allclose(a, before, atol=1e-6, rtol=1e-6))
    return a.astype(np.float32), changed


# =============================================================================
# Episode loading
# =============================================================================

def episode_sort_key(path: str):
    name = os.path.basename(path)
    m = re.search(r"episode_(\d+)_steps_(\d+)_(\d+)_len_(\d+)", name)
    if m:
        return tuple(int(x) for x in m.groups())
    return (0, 0, 0, 0, os.path.getmtime(path))


def resolve_episode_path(root: str, which: str, index: Optional[int], path: Optional[str]) -> str:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    ep_dir = os.path.join(root, "episode")
    files = sorted(glob.glob(os.path.join(ep_dir, "episode_*.pkl")), key=episode_sort_key)
    if not files:
        raise FileNotFoundError(f"没有找到 episode pkl: {ep_dir}/episode_*.pkl")

    if which == "latest":
        return files[-1]
    if which == "index":
        if index is None:
            raise ValueError("--which index 需要 --index")
        idx = int(index)
        if idx < 0:
            idx = len(files) + idx
        if idx < 0 or idx >= len(files):
            raise IndexError(f"index={index}, file_count={len(files)}")
        return files[idx]

    raise ValueError(f"未知 --which: {which}")


def load_episode(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pkl.load(f)

    if isinstance(payload, dict):
        if "transitions" not in payload:
            raise ValueError("episode pkl dict 中没有 transitions")
        return payload

    if isinstance(payload, list):
        return {
            "metadata": {},
            "transitions": payload,
            "step_records": [],
            "critic_q_values": [],
        }

    raise TypeError(f"不支持的 episode pkl 类型: {type(payload)}")


# =============================================================================
# Observation / pose extraction
# =============================================================================

def extract_obs(tr: Dict[str, Any], next_obs=False) -> Dict[str, Any]:
    key = "next_observations" if next_obs else "observations"
    obs = tr.get(key, {})
    return obs if isinstance(obs, dict) else {}


def extract_pose_from_state_dict(state: Dict[str, Any], arm_side: str = "right") -> Optional[np.ndarray]:
    preferred = []
    if arm_side == "right":
        preferred.extend([
            "right_ee_pose", "right/tcp_pose", "right_tcp_pose",
            "state/right_ee_pose", "state/right/tcp_pose", "pose_ee_arm_right",
        ])
    else:
        preferred.extend([
            "left_ee_pose", "left/tcp_pose", "left_tcp_pose",
            "state/left_ee_pose", "state/left/tcp_pose", "pose_ee_arm_left",
        ])
    preferred.extend(["ee_pose", "tcp_pose", "pose_ee"])

    for key in preferred:
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
    if not isinstance(obs, dict):
        return None
    state = obs.get("state", None)
    if state is None:
        return None
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


def extract_gripper_feedback(obs: Dict[str, Any], arm_side: str = "right") -> float:
    if not isinstance(obs, dict):
        return np.nan
    state = obs.get("state", None)
    if state is None:
        return np.nan

    if isinstance(state, dict):
        keys = [f"{arm_side}_gripper", "gripper", f"{arm_side}/gripper"]
        for k in keys:
            if k in state:
                arr = to_1d_array(state[k])
                if arr is not None:
                    return float(arr.reshape(-1)[-1])
        return np.nan

    arr = to_1d_array(state)
    if arr is not None and arr.size >= 8:
        return float(arr[7])
    return np.nan


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


def delta_norm_to_pos_mm(a6: Optional[np.ndarray], pos_scale: float) -> Optional[np.ndarray]:
    if a6 is None:
        return None
    return np.asarray(a6[:3], dtype=np.float32) * float(pos_scale) * 1000.0


def delta_norm_to_rot_deg(a6: Optional[np.ndarray], rot_scale: float) -> Optional[np.ndarray]:
    if a6 is None:
        return None
    return np.asarray(a6[3:6], dtype=np.float32) * float(rot_scale) * 180.0 / math.pi


def diff_metrics(a: Optional[np.ndarray], b: Optional[np.ndarray], pos_scale: float, rot_scale: float) -> Dict[str, float]:
    if a is None or b is None:
        return {
            "norm_absmax": np.nan,
            "norm_l2": np.nan,
            "pos_l2_mm": np.nan,
            "rot_l2_deg": np.nan,
        }
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:6]
    b = np.asarray(b, dtype=np.float32).reshape(-1)[:6]
    d = b - a
    pos_mm = d[:3] * float(pos_scale) * 1000.0
    rot_deg = d[3:6] * float(rot_scale) * 180.0 / math.pi
    return {
        "norm_absmax": float(np.max(np.abs(d))),
        "norm_l2": float(np.linalg.norm(d)),
        "pos_l2_mm": float(np.linalg.norm(pos_mm)),
        "rot_l2_deg": float(np.linalg.norm(rot_deg)),
    }


# =============================================================================
# Env helpers
# =============================================================================

def make_env(use_classifier=False):
    env_config = import_env_config()
    return env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=bool(use_classifier),
    )


def get_config_scale(env, name: str, default: float) -> float:
    # priority: env.unwrapped config / attr -> env_config -> default
    try:
        base = env.unwrapped
        cfg = getattr(base, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return float(getattr(cfg, name))
        if hasattr(base, name):
            return float(getattr(base, name))
    except Exception:
        pass
    try:
        env_config = import_env_config()
        if hasattr(env_config, name):
            return float(getattr(env_config, name))
    except Exception:
        pass
    return float(default)


def force_script_mode(env) -> None:
    print("🤖 尝试强制脚本/IK控制模式，避免 VR wrapper 覆盖 replay action...")
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
                print(f"  - warning: switch service failed: {e!r}")
        current = getattr(current, "env", None)
    try:
        base = env.unwrapped
        if hasattr(base, "notify_script_control"):
            base.notify_script_control(True)
            print("  - call env.unwrapped.notify_script_control(True)")
    except Exception as e:
        print(f"  - warning: notify_script_control failed: {e!r}")


def extract_env_step_debug(info: Any) -> Dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    for key in ["gripper_publish_debug", "gripper_debug", "last_gripper_publish_debug"]:
        if key in info and isinstance(info[key], dict):
            return info[key]
    return {}


# =============================================================================
# Source / critic helpers
# =============================================================================

def step_source(i: int, tr: Dict[str, Any], step_records: List[Dict[str, Any]]) -> str:
    # 先看 step_records
    if i < len(step_records) and isinstance(step_records[i], dict):
        rec = step_records[i]
        for key in ["had_intervene_action", "intervene", "intervention", "is_intervention"]:
            if key in rec:
                return "vr" if safe_bool(rec[key], False) else "actor"
    # 再看 transition info
    info = tr.get("infos", tr.get("info", {}))
    if isinstance(info, dict):
        for key in ["had_intervene_action", "intervene", "intervention", "is_intervention"]:
            if key in info:
                return "vr" if safe_bool(info[key], False) else "actor"
    return "unknown"


def get_critic_value(i: int, critic_values: List[Dict[str, Any]], key: str) -> float:
    if i >= len(critic_values) or not isinstance(critic_values[i], dict):
        return np.nan
    return safe_float(critic_values[i].get(key, np.nan), np.nan)


# =============================================================================
# CSV / summary
# =============================================================================

CSV_FIELDS = [
    "i", "global_step", "source", "reward", "done", "mask", "grasp_penalty",
    "action_absmax", "action_saturation", "gripper_action", "gripper_label",
    "action_dx", "action_dy", "action_dz", "action_dr", "action_dp", "action_dyaw",
    "action_dx_mm", "action_dy_mm", "action_dz_mm",
    "action_dr_deg", "action_dp_deg", "action_dyaw_deg",
    "saved_available",
    "saved_dx", "saved_dy", "saved_dz", "saved_dr", "saved_dp", "saved_dyaw",
    "saved_dx_mm", "saved_dy_mm", "saved_dz_mm",
    "saved_dr_deg", "saved_dp_deg", "saved_dyaw_deg",
    "saved_minus_action_norm_absmax", "saved_minus_action_norm_l2",
    "saved_minus_action_pos_l2_mm", "saved_minus_action_rot_l2_deg",
    "live_available",
    "live_dx", "live_dy", "live_dz", "live_dr", "live_dp", "live_dyaw",
    "live_dx_mm", "live_dy_mm", "live_dz_mm",
    "live_dr_deg", "live_dp_deg", "live_dyaw_deg",
    "live_minus_action_norm_absmax", "live_minus_action_norm_l2",
    "live_minus_action_pos_l2_mm", "live_minus_action_rot_l2_deg",
    "live_minus_saved_norm_absmax", "live_minus_saved_norm_l2",
    "live_minus_saved_pos_l2_mm", "live_minus_saved_rot_l2_deg",
    "saved_gripper_before", "saved_gripper_after",
    "live_gripper_before", "live_gripper_after",
    "env_mapped_hw", "env_reason", "env_feedback_before", "env_mem_before", "env_mem_after",
    "critic_q_mean", "critic_q_min", "grasp_q_selected",
]


def vec_to_fields(prefix: str, a6: Optional[np.ndarray], row: Dict[str, Any], pos_scale: float, rot_scale: float):
    if a6 is None:
        for k in ["dx", "dy", "dz", "dr", "dp", "dyaw"]:
            row[f"{prefix}_{k}"] = ""
        for k in ["dx_mm", "dy_mm", "dz_mm", "dr_deg", "dp_deg", "dyaw_deg"]:
            row[f"{prefix}_{k}"] = ""
        return
    a6 = np.asarray(a6, dtype=np.float32).reshape(-1)[:6]
    names = ["dx", "dy", "dz", "dr", "dp", "dyaw"]
    for j, name in enumerate(names):
        row[f"{prefix}_{name}"] = float(a6[j])
    pos_mm = delta_norm_to_pos_mm(a6, pos_scale)
    rot_deg = delta_norm_to_rot_deg(a6, rot_scale)
    row[f"{prefix}_dx_mm"] = float(pos_mm[0])
    row[f"{prefix}_dy_mm"] = float(pos_mm[1])
    row[f"{prefix}_dz_mm"] = float(pos_mm[2])
    row[f"{prefix}_dr_deg"] = float(rot_deg[0])
    row[f"{prefix}_dp_deg"] = float(rot_deg[1])
    row[f"{prefix}_dyaw_deg"] = float(rot_deg[2])


def add_metrics_to_row(prefix: str, metrics: Dict[str, float], row: Dict[str, Any]):
    row[f"{prefix}_norm_absmax"] = metrics["norm_absmax"]
    row[f"{prefix}_norm_l2"] = metrics["norm_l2"]
    row[f"{prefix}_pos_l2_mm"] = metrics["pos_l2_mm"]
    row[f"{prefix}_rot_l2_deg"] = metrics["rot_l2_deg"]


def finite_values(rows: List[Dict[str, Any]], key: str, source: Optional[str] = None) -> np.ndarray:
    vals = []
    for r in rows:
        if source is not None and r.get("source") != source:
            continue
        try:
            v = float(r.get(key, np.nan))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return np.asarray(vals, dtype=np.float64)


def print_metric_summary(rows: List[Dict[str, Any]], key: str, label: str, source: Optional[str] = None):
    vals = finite_values(rows, key, source=source)
    src = "all" if source is None else source
    if vals.size == 0:
        print(f"{label:36s} [{src:7s}]: N/A")
        return
    print(
        f"{label:36s} [{src:7s}]: "
        f"n={vals.size}, mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
        f"p90={np.percentile(vals, 90):.4f}, max={np.max(vals):.4f}"
    )


def print_summary(rows: List[Dict[str, Any]]):
    print("=" * 100)
    print("EPISODE LIVE DELTA COMPARE SUMMARY")
    print("=" * 100)
    n = len(rows)
    n_actor = sum(1 for r in rows if r.get("source") == "actor")
    n_vr = sum(1 for r in rows if r.get("source") == "vr")
    n_unknown = n - n_actor - n_vr
    n_sat = sum(1 for r in rows if bool(r.get("action_saturation", False)))
    n_saved = sum(1 for r in rows if bool(r.get("saved_available", False)))
    n_live = sum(1 for r in rows if bool(r.get("live_available", False)))
    print(f"rows                         : {n}")
    print(f"source actor/vr/unknown       : {n_actor} / {n_vr} / {n_unknown}")
    print(f"action saturation rows        : {n_sat}  (|action[:6]| >= 0.98)")
    print(f"saved feedback delta rows     : {n_saved}")
    print(f"live feedback delta rows      : {n_live}")
    print("")

    for source in [None, "actor", "vr"]:
        print_metric_summary(rows, "saved_minus_action_norm_absmax", "saved - action absmax", source)
        print_metric_summary(rows, "saved_minus_action_pos_l2_mm", "saved - action pos L2 mm", source)
        print_metric_summary(rows, "saved_minus_action_rot_l2_deg", "saved - action rot L2 deg", source)
        print_metric_summary(rows, "live_minus_action_norm_absmax", "live - action absmax", source)
        print_metric_summary(rows, "live_minus_action_pos_l2_mm", "live - action pos L2 mm", source)
        print_metric_summary(rows, "live_minus_action_rot_l2_deg", "live - action rot L2 deg", source)
        print_metric_summary(rows, "live_minus_saved_norm_absmax", "live - saved absmax", source)
        print_metric_summary(rows, "live_minus_saved_pos_l2_mm", "live - saved pos L2 mm", source)
        print_metric_summary(rows, "live_minus_saved_rot_l2_deg", "live - saved rot L2 deg", source)
        print("")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare saved episode action delta vs saved feedback delta vs live replay feedback delta.")
    parser.add_argument("--root", default="./rlpd_checkpoints_single", help="checkpoint root containing episode/")
    parser.add_argument("--path", default=None, help="direct path to episode_*.pkl")
    parser.add_argument("--which", default="latest", choices=["latest", "index"], help="episode file selection")
    parser.add_argument("--index", type=int, default=None, help="used when --which index")
    parser.add_argument("--start", type=int, default=0, help="start step, inclusive")
    parser.add_argument("--end", type=int, default=None, help="end step, exclusive")
    parser.add_argument("--execute", action="store_true", help="execute env.step(action) on real robot and compare live feedback delta")
    parser.add_argument("--no_reset", action="store_true", help="do not reset before live execution; dangerous, only for advanced debugging")
    parser.add_argument("--use_classifier", action="store_true", help="create env with classifier=True")
    parser.add_argument("--pos_scale", type=float, default=0.0, help="override POS_SCALE; 0 means config")
    parser.add_argument("--rot_scale", type=float, default=0.0, help="override ROT_SCALE; 0 means config")
    parser.add_argument("--clip_action", action="store_true", default=True, help="clip action[:6] to [-1,1] before execution")
    parser.add_argument("--no_clip_action", dest="clip_action", action="store_false")
    parser.add_argument("--quantize_gripper", action="store_true", default=True, help="quantize action[6] to -1/0/+1 before execution")
    parser.add_argument("--no_quantize_gripper", dest="quantize_gripper", action="store_false")
    parser.add_argument("--force_script_mode", action="store_true", default=True)
    parser.add_argument("--no_force_script_mode", dest="force_script_mode", action="store_false")
    parser.add_argument("--sleep", type=float, default=0.0, help="extra sleep after each env.step")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--warn_norm_absmax", type=float, default=0.35)
    parser.add_argument("--csv", default="./inspect/episode_live_delta_compare.csv")
    args = parser.parse_args()

    ep_path = resolve_episode_path(args.root, args.which, args.index, args.path)
    payload = load_episode(ep_path)
    transitions = payload.get("transitions", [])
    step_records = payload.get("step_records", []) or []
    critic_values = payload.get("critic_q_values", []) or []
    metadata = payload.get("metadata", {}) or {}

    if not transitions:
        raise ValueError(f"episode transitions 为空: {ep_path}")

    start = max(0, int(args.start))
    end = len(transitions) if args.end is None else min(len(transitions), int(args.end))
    if end <= start:
        raise ValueError(f"empty range: start={start}, end={end}")

    # scale: config if possible, but if not executing we still can import config
    pos_scale = float(args.pos_scale) if args.pos_scale > 0 else 0.02
    rot_scale = float(args.rot_scale) if args.rot_scale > 0 else 0.04

    env = None
    live_obs = None

    if args.execute:
        print("⚠️ 即将真实执行 episode action，并记录 live feedback delta。请确认机器人安全。")
        env = make_env(use_classifier=args.use_classifier)
        if args.pos_scale <= 0:
            pos_scale = get_config_scale(env, "POS_SCALE", pos_scale)
        if args.rot_scale <= 0:
            rot_scale = get_config_scale(env, "ROT_SCALE", rot_scale)
        if args.force_script_mode:
            force_script_mode(env)
        if args.no_reset:
            print("⚠️ --no_reset=True: 不调用 env.reset()，直接读取当前 obs。")
            if hasattr(env.unwrapped, "observe_only_step"):
                live_obs, _, _, _, _ = env.unwrapped.observe_only_step()
            else:
                live_obs, _ = env.reset()
        else:
            live_obs, _ = env.reset()
            if args.force_script_mode:
                force_script_mode(env)
    else:
        # 尽量从 config 读 scale，不创建 env 也可以
        try:
            env_config = import_env_config()
            if args.pos_scale <= 0 and hasattr(env_config, "POS_SCALE"):
                pos_scale = float(getattr(env_config, "POS_SCALE"))
            if args.rot_scale <= 0 and hasattr(env_config, "ROT_SCALE"):
                rot_scale = float(getattr(env_config, "ROT_SCALE"))
        except Exception:
            pass

    print("=" * 100)
    print("SAVED EPISODE ACTION / FEEDBACK / LIVE DELTA COMPARE")
    print("=" * 100)
    print(f"path        : {ep_path}")
    print(f"episode     : {metadata.get('episode_index', 'N/A')}")
    print(f"range       : [{start}, {end}) / {len(transitions)}")
    print(f"execute     : {args.execute}")
    print(f"pos_scale   : {pos_scale}")
    print(f"rot_scale   : {rot_scale}")
    print(f"csv         : {args.csv}")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    rows = []

    try:
        for i in range(start, end):
            tr = transitions[i]
            action_raw = get_action(tr)
            action_exec, changed = sanitize_action_for_replay(
                action_raw,
                clip_action=args.clip_action,
                quantize_gripper=args.quantize_gripper,
            )
            a6 = action_exec[:6]
            src = step_source(i, tr, step_records)
            global_step = metadata.get("global_step_start", None)
            if global_step is not None:
                try:
                    global_step = int(global_step) + i
                except Exception:
                    global_step = ""
            else:
                # fallback from step_records
                if i < len(step_records) and isinstance(step_records[i], dict):
                    global_step = step_records[i].get("global_step", step_records[i].get("step", ""))
                else:
                    global_step = ""

            obs = extract_obs(tr, next_obs=False)
            next_obs = extract_obs(tr, next_obs=True)
            saved_prev_pose = extract_ee_pose(obs)
            saved_next_pose = extract_ee_pose(next_obs)
            saved_delta = compute_delta_action_from_poses(saved_prev_pose, saved_next_pose, pos_scale, rot_scale, clip=False)
            saved_g_before = extract_gripper_feedback(obs)
            saved_g_after = extract_gripper_feedback(next_obs)

            live_delta = None
            live_g_before = np.nan
            live_g_after = np.nan
            env_debug = {}

            if args.execute:
                live_before_obs = live_obs
                live_g_before = extract_gripper_feedback(live_before_obs)
                live_before_pose = extract_ee_pose(live_before_obs)

                live_next_obs, live_reward, live_done, live_trunc, live_info = env.step(action_exec)
                live_after_pose = extract_ee_pose(live_next_obs)
                live_g_after = extract_gripper_feedback(live_next_obs)
                live_delta = compute_delta_action_from_poses(live_before_pose, live_after_pose, pos_scale, rot_scale, clip=False)
                env_debug = extract_env_step_debug(live_info)
                live_obs = live_next_obs
                if args.sleep > 0:
                    time.sleep(float(args.sleep))

            saved_vs_action = diff_metrics(a6, saved_delta, pos_scale, rot_scale)
            live_vs_action = diff_metrics(a6, live_delta, pos_scale, rot_scale)
            live_vs_saved = diff_metrics(saved_delta, live_delta, pos_scale, rot_scale)

            row = {
                "i": i,
                "global_step": global_step,
                "source": src,
                "reward": get_reward(tr),
                "done": get_done(tr),
                "mask": get_mask(tr),
                "grasp_penalty": get_grasp_penalty(tr),
                "action_absmax": float(np.max(np.abs(a6))),
                "action_saturation": bool(np.max(np.abs(a6)) >= 0.98),
                "gripper_action": float(action_exec[6]),
                "gripper_label": gripper_label(float(action_exec[6])),
                "saved_available": saved_delta is not None,
                "live_available": live_delta is not None,
                "saved_gripper_before": saved_g_before,
                "saved_gripper_after": saved_g_after,
                "live_gripper_before": live_g_before,
                "live_gripper_after": live_g_after,
                "env_mapped_hw": env_debug.get("mapped_hw", env_debug.get("right_mapped_hw", "")),
                "env_reason": env_debug.get("reason", env_debug.get("map_reason", "")),
                "env_feedback_before": env_debug.get("feedback_before", ""),
                "env_mem_before": env_debug.get("mem_before", ""),
                "env_mem_after": env_debug.get("mem_after", ""),
                "critic_q_mean": get_critic_value(i, critic_values, "critic_q_mean"),
                "critic_q_min": get_critic_value(i, critic_values, "critic_q_min"),
                "grasp_q_selected": get_critic_value(i, critic_values, "grasp_q_selected"),
            }

            vec_to_fields("action", a6, row, pos_scale, rot_scale)
            vec_to_fields("saved", saved_delta, row, pos_scale, rot_scale)
            vec_to_fields("live", live_delta, row, pos_scale, rot_scale)
            add_metrics_to_row("saved_minus_action", saved_vs_action, row)
            add_metrics_to_row("live_minus_action", live_vs_action, row)
            add_metrics_to_row("live_minus_saved", live_vs_saved, row)

            rows.append(row)

            warn = ""
            if np.isfinite(live_vs_action["norm_absmax"]) and live_vs_action["norm_absmax"] >= args.warn_norm_absmax:
                warn += " ⚠️live!=action"
            if np.isfinite(saved_vs_action["norm_absmax"]) and saved_vs_action["norm_absmax"] >= args.warn_norm_absmax:
                warn += " ⚠️saved!=action"
            if row["action_saturation"]:
                warn += " ⚠️sat"

            if (i - start) % max(1, args.print_every) == 0:
                print(
                    f"[compare-step] i={i:04d} global={global_step} src={src:5s} "
                    f"a={fmt_vec(a6)} delta_mm={fmt_vec(delta_norm_to_pos_mm(a6, pos_scale), 1)} "
                    f"rot_deg={fmt_vec(delta_norm_to_rot_deg(a6, rot_scale), 2)} "
                    f"grip={gripper_label(float(action_exec[6]))} "
                    f"saved={fmt_vec(saved_delta)} live={fmt_vec(live_delta)} "
                    f"saved_err_abs={saved_vs_action['norm_absmax']:.3f} "
                    f"live_err_abs={live_vs_action['norm_absmax']:.3f} "
                    f"live_pos_err_mm={live_vs_action['pos_l2_mm']:.2f} "
                    f"live_rot_err_deg={live_vs_action['rot_l2_deg']:.2f}" 
                    f"{warn}"
                )

            if args.execute and (get_done(tr) or row["reward"] > 0 or row["mask"] == 0.0):
                # 不自动 break：用户指定 end 时可能想看完整执行；这里只提示
                pass

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print_summary(rows)
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"csv saved: {args.csv}")


if __name__ == "__main__":
    main()
