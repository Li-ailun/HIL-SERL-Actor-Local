# 运行前：

# chmod +x /home/eren/HIL-SERL/HIL-SERL-Project/examples/inspect/visualize_buffer_video.py

# 看最新 buffer 的第 0 个 episode：

# cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

# python inspect/visualize_buffer_video.py \
#   --root ./rlpd_checkpoints_single \
#   --kind buffer \
#   --which latest \
#   --select episode \
#   --episode 0 \
#   --image_keys head_rgb right_wrist_rgb \
#   --pos_scale 0.02 \
#   --rot_scale 0.04 \
#   --out ./inspect/buffer_latest_ep0.mp4

# 看最新 demo_buffer 的前 300 个 transition：

# python inspect/visualize_buffer_video.py \
#   --root ./rlpd_checkpoints_single \
#   --kind demo_buffer \
#   --which latest \
#   --select global \
#   --start 0 \
#   --end 300 \
#   --image_keys head_rgb right_wrist_rgb \
#   --pos_scale 0.02 \
#   --rot_scale 0.04 \
#   --out ./inspect/demo_buffer_latest_0_300.mp4

# 看所有 buffer 文件拼接后的第 10 到 15 条 episode：

# python inspect/visualize_buffer_video.py \
#   --root ./rlpd_checkpoints_single \
#   --kind buffer \
#   --which all \
#   --select episode \
#   --episode_start 10 \
#   --episode_end 16 \
#   --image_keys head_rgb right_wrist_rgb \
#   --pos_scale 0.02 \
#   --rot_scale 0.04 \
#   --out ./inspect/buffer_all_ep10_15.mp4

# 重点看视频里的这些行：

# norm action [dx dy dz dr dp dyaw]
# physical delta = pos_mm[...] rot_deg[...]
# gripper=close/open/hold
# reward / done / mask / grasp_penalty
# observe_only / vr_observe_only / intervene_action_valid

# 如果某一帧动作接近饱和，底部会出现：

# WARNING: arm action near saturation |action[:6]| >= 0.98




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_buffer_video.py

离线可视化 HIL-SERL actor buffer / demo_buffer / demo pkl。

支持：
  1) --root + --kind buffer/demo_buffer 自动找 transitions_*.pkl
  2) --path 直接指定单个 pkl
  3) --which latest/all/index/step
  4) --select episode/global
  5) 显示：
      - 图像 head_rgb / right_wrist_rgb 等
      - action[:6] 归一化增量
      - action[:3] * POS_SCALE -> mm
      - action[3:6] * ROT_SCALE -> deg
      - action[6] 夹爪标签 close/hold/open
      - reward / done / mask / grasp_penalty
      - infos 里的 observe_only / vr_observe_only / intervention_valid / success 等

示例：

# 可视化最新 online buffer 的第 0 个 episode
python inspect/visualize_buffer_video.py \
  --root /home/eren/HIL-SERL/HIL-SERL-Project/examples/rlpd_checkpoints_single \
  --kind buffer \
  --which latest \
  --select episode \
  --episode 0 \
  --image_keys head_rgb right_wrist_rgb \
  --pos_scale 0.02 \
  --rot_scale 0.04 \
  --out buffer_ep0.mp4

# 可视化最新 demo_buffer 的 global transition 0~300
python visualize_buffer_video.py \
  --root /home/eren/HIL-SERL/HIL-SERL-Project/examples/rlpd_checkpoints_single \
  --kind demo_buffer \
  --which latest \
  --select global \
  --start 0 \
  --end 300 \
  --image_keys head_rgb right_wrist_rgb \
  --out demo_buffer_0_300.mp4

# 可视化所有 buffer 文件拼接后的 episode 10~15
python visualize_buffer_video.py \
  --root /home/eren/HIL-SERL/HIL-SERL-Project/examples/rlpd_checkpoints_single \
  --kind buffer \
  --which all \
  --select episode \
  --episode_start 10 \
  --episode_end 16 \
  --out buffer_ep10_15.mp4
"""

import argparse
import csv
import glob
import math
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Basic helpers
# =============================================================================

def as_float(x, default=0.0) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def as_int(x, default=0) -> int:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def as_bool(x) -> bool:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return bool(x)
        return bool(arr.reshape(-1)[0])
    except Exception:
        return bool(x)


def short_float(x, ndigits=3) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "nan"


def gripper_label(g: float) -> str:
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def terminal_transition(tr: Dict[str, Any]) -> bool:
    reward = as_float(tr.get("rewards", tr.get("reward", 0.0)), 0.0)
    done = as_bool(tr.get("dones", tr.get("done", False)))
    truncated = as_bool(tr.get("truncated", tr.get("truncates", False)))
    mask = as_float(tr.get("masks", tr.get("mask", 1.0)), 1.0)

    return bool(done or truncated or mask == 0.0 or reward > 0.0)


def get_info(tr: Dict[str, Any]) -> Dict[str, Any]:
    info = tr.get("infos", tr.get("info", {}))
    if isinstance(info, dict):
        return info
    return {}


def get_reward(tr: Dict[str, Any]) -> float:
    return as_float(tr.get("rewards", tr.get("reward", 0.0)), 0.0)


def get_done(tr: Dict[str, Any]) -> bool:
    return as_bool(tr.get("dones", tr.get("done", False)))


def get_mask(tr: Dict[str, Any]) -> float:
    return as_float(tr.get("masks", tr.get("mask", 1.0)), 1.0)


def get_grasp_penalty(tr: Dict[str, Any]) -> float:
    if "grasp_penalty" in tr:
        return as_float(tr["grasp_penalty"], 0.0)
    info = get_info(tr)
    if "grasp_penalty" in info:
        return as_float(info["grasp_penalty"], 0.0)
    return 0.0


def get_action(tr: Dict[str, Any]) -> np.ndarray:
    a = tr.get("actions", tr.get("action", None))
    if a is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(a, dtype=np.float32).reshape(-1)


# =============================================================================
# File selection
# =============================================================================

def parse_numeric_transition_step(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.fullmatch(r"transitions_(\d+)\.pkl", name)
    if not m:
        return None
    return int(m.group(1))


def list_numeric_transition_files(root: str, kind: str) -> List[Tuple[int, str]]:
    directory = os.path.join(root, kind)
    files = glob.glob(os.path.join(directory, "transitions_*.pkl"))

    out = []
    for p in files:
        step = parse_numeric_transition_step(p)
        if step is not None:
            out.append((step, p))

    out.sort(key=lambda x: x[0])
    return out


def resolve_files(args) -> List[str]:
    if args.path:
        if not os.path.exists(args.path):
            raise FileNotFoundError(args.path)
        return [args.path]

    numeric = list_numeric_transition_files(args.root, args.kind)
    if not numeric:
        raise FileNotFoundError(
            f"没有找到 numeric transition pkl: {os.path.join(args.root, args.kind, 'transitions_*.pkl')}"
        )

    if args.which == "latest":
        return [numeric[-1][1]]

    if args.which == "all":
        return [p for _, p in numeric]

    if args.which == "index":
        if args.file_index is None:
            raise ValueError("--which index 需要 --file_index")
        idx = int(args.file_index)
        if idx < 0:
            idx = len(numeric) + idx
        if idx < 0 or idx >= len(numeric):
            raise IndexError(f"file_index={args.file_index}, file_count={len(numeric)}")
        return [numeric[idx][1]]

    if args.which == "step":
        if args.step is None:
            raise ValueError("--which step 需要 --step")
        step = int(args.step)
        for s, p in numeric:
            if s == step:
                return [p]
        raise FileNotFoundError(f"没有找到 transitions_{step}.pkl")

    raise ValueError(f"未知 --which: {args.which}")


# =============================================================================
# PKL loading
# =============================================================================

def dict_of_arrays_to_transitions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    兼容两种格式：
      1. list[transition]
      2. dict of arrays/lists:
         observations/actions/next_observations/rewards/masks/dones/infos/grasp_penalty
    """
    if "actions" not in payload:
        raise ValueError("dict payload 中找不到 actions，无法推断 transition 数量")

    actions = payload["actions"]
    n = len(actions)

    transitions = []
    for i in range(n):
        tr = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                # observations / next_observations 常见为 dict of arrays/lists
                sub = {}
                ok = True
                for sk, sv in value.items():
                    try:
                        sub[sk] = sv[i]
                    except Exception:
                        ok = False
                        break
                if ok:
                    tr[key] = sub
                else:
                    tr[key] = value
            else:
                try:
                    tr[key] = value[i]
                except Exception:
                    tr[key] = value
        transitions.append(tr)

    return transitions


def load_transitions_from_pkl(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["transitions", "data", "demo_data"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        return dict_of_arrays_to_transitions(payload)

    raise TypeError(f"不支持的 pkl payload 类型: {type(payload)}")


def load_files(files: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_transitions = []
    inventory = []

    global_offset = 0
    for file_i, path in enumerate(files):
        transitions = load_transitions_from_pkl(path)
        inventory.append(
            {
                "file_i": file_i,
                "path": path,
                "count": len(transitions),
                "global_start": global_offset,
                "global_end": global_offset + len(transitions),
            }
        )
        for local_i, tr in enumerate(transitions):
            if isinstance(tr, dict):
                tr = dict(tr)
                tr["_viz_file_i"] = file_i
                tr["_viz_file_path"] = path
                tr["_viz_file_local_i"] = local_i
                tr["_viz_global_i"] = global_offset + local_i
            all_transitions.append(tr)
        global_offset += len(transitions)

    return all_transitions, inventory


# =============================================================================
# Episode splitting / selection
# =============================================================================

def split_episodes(transitions: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, int]]]:
    episodes = []
    meta = []

    cur = []
    start = 0

    for i, tr in enumerate(transitions):
        if not cur:
            start = i
        cur.append(tr)

        if terminal_transition(tr):
            episodes.append(cur)
            meta.append({"start": start, "end": i + 1, "length": len(cur)})
            cur = []

    if cur:
        episodes.append(cur)
        meta.append({"start": start, "end": len(transitions), "length": len(cur)})

    return episodes, meta


def select_transitions(args, transitions: List[Dict[str, Any]]):
    episodes, ep_meta = split_episodes(transitions)

    selected = []
    selected_desc = ""

    if args.select == "global":
        start = 0 if args.start is None else int(args.start)
        end = len(transitions) if args.end is None else int(args.end)

        start = max(0, start)
        end = min(len(transitions), end)

        if end <= start:
            raise ValueError(f"global segment 为空: start={start}, end={end}")

        for local_frame_i, tr in enumerate(transitions[start:end]):
            selected.append(
                {
                    "transition": tr,
                    "global_i": start + local_frame_i,
                    "episode_i": find_episode_index_for_global(ep_meta, start + local_frame_i),
                    "episode_local_i": local_index_in_episode(ep_meta, start + local_frame_i),
                    "frame_i": local_frame_i,
                }
            )

        selected_desc = f"global[{start}:{end}]"

    elif args.select == "episode":
        if args.episode is not None:
            ep_start = int(args.episode)
            ep_end = ep_start + 1
        else:
            ep_start = 0 if args.episode_start is None else int(args.episode_start)
            ep_end = len(episodes) if args.episode_end is None else int(args.episode_end)

        ep_start = max(0, ep_start)
        ep_end = min(len(episodes), ep_end)

        if ep_end <= ep_start:
            raise ValueError(f"episode segment 为空: episode_start={ep_start}, episode_end={ep_end}")

        frame_i = 0
        for ep_i in range(ep_start, ep_end):
            ep = episodes[ep_i]
            local_start = 0 if args.start is None else int(args.start)
            local_end = len(ep) if args.end is None else int(args.end)

            local_start = max(0, local_start)
            local_end = min(len(ep), local_end)

            for local_i in range(local_start, local_end):
                tr = ep[local_i]
                selected.append(
                    {
                        "transition": tr,
                        "global_i": ep_meta[ep_i]["start"] + local_i,
                        "episode_i": ep_i,
                        "episode_local_i": local_i,
                        "frame_i": frame_i,
                    }
                )
                frame_i += 1

        selected_desc = f"episode[{ep_start}:{ep_end}], local_step[{args.start}:{args.end}]"

    else:
        raise ValueError(f"未知 --select: {args.select}")

    return selected, episodes, ep_meta, selected_desc


def find_episode_index_for_global(ep_meta: List[Dict[str, int]], global_i: int) -> int:
    for ep_i, m in enumerate(ep_meta):
        if m["start"] <= global_i < m["end"]:
            return ep_i
    return -1


def local_index_in_episode(ep_meta: List[Dict[str, int]], global_i: int) -> int:
    for m in ep_meta:
        if m["start"] <= global_i < m["end"]:
            return global_i - m["start"]
    return -1


# =============================================================================
# Image / overlay
# =============================================================================

def get_observation(tr: Dict[str, Any], next_obs: bool = False) -> Dict[str, Any]:
    key = "next_observations" if next_obs else "observations"
    obs = tr.get(key, {})
    if isinstance(obs, dict):
        return obs
    return {}


def extract_image_from_obs(obs: Dict[str, Any], image_key: str) -> Optional[np.ndarray]:
    value = None

    if image_key in obs:
        value = obs[image_key]
    elif "images" in obs and isinstance(obs["images"], dict) and image_key in obs["images"]:
        value = obs["images"][image_key]

    if value is None:
        return None

    arr = np.asarray(value)

    # 常见 shape=(1,H,W,3)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        return None

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # demos 里通常是 RGB，cv2 VideoWriter 要 BGR
    arr_bgr = arr[..., ::-1].copy()
    return arr_bgr


def build_image_strip(
    tr: Dict[str, Any],
    image_keys: List[str],
    display_size: int,
    use_next_obs: bool,
) -> np.ndarray:
    obs = get_observation(tr, next_obs=use_next_obs)
    frames = []

    for key in image_keys:
        img = extract_image_from_obs(obs, key)
        if img is None:
            img = np.zeros((display_size, display_size, 3), dtype=np.uint8)
            cv2.putText(
                img,
                f"missing {key}",
                (8, display_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            img = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

        cv2.putText(
            img,
            key,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(img)

    return np.concatenate(frames, axis=1)


def put_line(panel, text, y, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(
        panel,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_info_panel(
    item: Dict[str, Any],
    width: int,
    panel_h: int,
    pos_scale: float,
    rot_scale: float,
    source_name: str,
) -> np.ndarray:
    tr = item["transition"]
    action = get_action(tr)
    reward = get_reward(tr)
    done = get_done(tr)
    mask = get_mask(tr)
    gp = get_grasp_penalty(tr)
    info = get_info(tr)

    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)

    if action.shape[0] >= 7:
        a6 = action[:6]
        grip = float(action[6])
        pos_mm = action[:3] * float(pos_scale) * 1000.0
        rot_deg = action[3:6] * float(rot_scale) * 180.0 / math.pi
    else:
        a6 = np.zeros((6,), dtype=np.float32)
        grip = 0.0
        pos_mm = np.zeros((3,), dtype=np.float32)
        rot_deg = np.zeros((3,), dtype=np.float32)

    file_name = os.path.basename(str(tr.get("_viz_file_path", "")))

    line1 = (
        f"{source_name} | file={file_name} | "
        f"global={item['global_i']} | ep={item['episode_i']} | ep_step={item['episode_local_i']} | frame={item['frame_i']}"
    )
    line2 = (
        f"reward={reward:.3f} done={done} mask={mask:.1f} gp={gp:.3f} | "
        f"gripper={gripper_label(grip)}"
    )
    line3 = (
        "norm action [dx dy dz dr dp dyaw] = "
        f"[{a6[0]:+.3f}, {a6[1]:+.3f}, {a6[2]:+.3f}, "
        f"{a6[3]:+.3f}, {a6[4]:+.3f}, {a6[5]:+.3f}]"
    )
    line4 = (
        "physical delta = "
        f"pos_mm[{pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f}] "
        f"rot_deg[{rot_deg[0]:+.2f}, {rot_deg[1]:+.2f}, {rot_deg[2]:+.2f}]"
    )

    flags = []
    for k in [
        "observe_only",
        "vr_observe_only",
        "vr_prime_baseline_used",
        "vr_prime_baseline_success",
        "intervene_action_valid",
        "success",
        "is_success",
        "succeed",
    ]:
        if k in info:
            flags.append(f"{k}={info.get(k)}")

    if "intervene_action_invalid_reason" in info and str(info.get("intervene_action_invalid_reason", "")):
        flags.append(f"invalid_reason={info.get('intervene_action_invalid_reason')}")

    if "grasp_penalty_source" in info:
        flags.append(f"gp_src={info.get('grasp_penalty_source')}")

    line5 = "info: " + (" | ".join(flags) if flags else "N/A")

    put_line(panel, line1, 22, scale=0.52)
    put_line(panel, line2, 48, scale=0.55, color=(220, 255, 220))
    put_line(panel, line3, 74, scale=0.52)
    put_line(panel, line4, 100, scale=0.52, color=(220, 220, 255))
    put_line(panel, line5[:180], 126, scale=0.45, color=(210, 210, 210))

    if action.shape[0] >= 6 and np.max(np.abs(action[:6])) >= 0.98:
        put_line(panel, "WARNING: arm action near saturation |action[:6]| >= 0.98", 152, scale=0.55, color=(0, 255, 255), thickness=2)

    return panel


def make_frame(
    item: Dict[str, Any],
    image_keys: List[str],
    display_size: int,
    panel_h: int,
    pos_scale: float,
    rot_scale: float,
    source_name: str,
    use_next_obs: bool,
) -> np.ndarray:
    tr = item["transition"]
    strip = build_image_strip(tr, image_keys, display_size, use_next_obs=use_next_obs)
    panel = make_info_panel(
        item,
        width=strip.shape[1],
        panel_h=panel_h,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        source_name=source_name,
    )
    return np.concatenate([strip, panel], axis=0)


# =============================================================================
# CSV
# =============================================================================

def write_csv(path: str, selected: List[Dict[str, Any]], pos_scale: float, rot_scale: float):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fields = [
        "frame_i",
        "global_i",
        "episode_i",
        "episode_local_i",
        "file_i",
        "file_local_i",
        "file_path",
        "reward",
        "done",
        "mask",
        "grasp_penalty",
        "action_0_dx",
        "action_1_dy",
        "action_2_dz",
        "action_3_droll",
        "action_4_dpitch",
        "action_5_dyaw",
        "action_6_gripper",
        "gripper_label",
        "pos_dx_mm",
        "pos_dy_mm",
        "pos_dz_mm",
        "rot_droll_deg",
        "rot_dpitch_deg",
        "rot_dyaw_deg",
        "observe_only",
        "vr_observe_only",
        "vr_prime_baseline_used",
        "vr_prime_baseline_success",
        "intervene_action_valid",
        "intervene_action_invalid_reason",
        "success",
        "is_success",
        "succeed",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in selected:
            tr = item["transition"]
            info = get_info(tr)
            a = get_action(tr)
            aa = np.zeros((7,), dtype=np.float32)
            aa[: min(7, a.shape[0])] = a[: min(7, a.shape[0])]

            row = {
                "frame_i": item["frame_i"],
                "global_i": item["global_i"],
                "episode_i": item["episode_i"],
                "episode_local_i": item["episode_local_i"],
                "file_i": tr.get("_viz_file_i", ""),
                "file_local_i": tr.get("_viz_file_local_i", ""),
                "file_path": tr.get("_viz_file_path", ""),
                "reward": get_reward(tr),
                "done": get_done(tr),
                "mask": get_mask(tr),
                "grasp_penalty": get_grasp_penalty(tr),
                "action_0_dx": float(aa[0]),
                "action_1_dy": float(aa[1]),
                "action_2_dz": float(aa[2]),
                "action_3_droll": float(aa[3]),
                "action_4_dpitch": float(aa[4]),
                "action_5_dyaw": float(aa[5]),
                "action_6_gripper": float(aa[6]),
                "gripper_label": gripper_label(float(aa[6])),
                "pos_dx_mm": float(aa[0] * pos_scale * 1000.0),
                "pos_dy_mm": float(aa[1] * pos_scale * 1000.0),
                "pos_dz_mm": float(aa[2] * pos_scale * 1000.0),
                "rot_droll_deg": float(aa[3] * rot_scale * 180.0 / math.pi),
                "rot_dpitch_deg": float(aa[4] * rot_scale * 180.0 / math.pi),
                "rot_dyaw_deg": float(aa[5] * rot_scale * 180.0 / math.pi),
                "observe_only": info.get("observe_only", ""),
                "vr_observe_only": info.get("vr_observe_only", ""),
                "vr_prime_baseline_used": info.get("vr_prime_baseline_used", ""),
                "vr_prime_baseline_success": info.get("vr_prime_baseline_success", ""),
                "intervene_action_valid": info.get("intervene_action_valid", ""),
                "intervene_action_invalid_reason": info.get("intervene_action_invalid_reason", ""),
                "success": info.get("success", ""),
                "is_success": info.get("is_success", ""),
                "succeed": info.get("succeed", ""),
            }
            writer.writerow(row)


# =============================================================================
# Summary
# =============================================================================

def summarize(transitions: List[Dict[str, Any]], episodes: List[List[Dict[str, Any]]]):
    n = len(transitions)
    rewards = np.array([get_reward(t) for t in transitions], dtype=np.float64)
    dones = np.array([get_done(t) for t in transitions], dtype=bool)
    masks = np.array([get_mask(t) for t in transitions], dtype=np.float64)
    gps = np.array([get_grasp_penalty(t) for t in transitions], dtype=np.float64)

    grip_counts = {"close(-1)": 0, "hold(0)": 0, "open(+1)": 0}
    sat = 0

    for t in transitions:
        a = get_action(t)
        if a.shape[0] >= 7:
            grip_counts[gripper_label(float(a[6]))] += 1
            if np.max(np.abs(a[:6])) >= 0.98:
                sat += 1

    print("=" * 100)
    print("BUFFER VISUALIZATION SUMMARY")
    print("=" * 100)
    print(f"transitions       : {n}")
    print(f"episodes split    : {len(episodes)}")
    print(f"reward_sum        : {float(rewards.sum()):.6f}")
    print(f"reward>0          : {int((rewards > 0).sum())}")
    print(f"done=True         : {int(dones.sum())}")
    print(f"mask=0            : {int((masks == 0).sum())}")
    print(f"grasp_penalty sum : {float(gps.sum()):.6f}")
    print(f"gripper dist      : {grip_counts}")
    print(f"saturation rows   : {sat}  (|action[:6]| >= 0.98)")

    if episodes:
        lens = [len(e) for e in episodes]
        print(f"episode length    : min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize HIL-SERL buffer/demo_buffer as annotated video."
    )

    parser.add_argument(
        "--root",
        default="./rlpd_checkpoints_single",
        help="checkpoint root，里面包含 buffer/ 和 demo_buffer/。如果指定 --path，则忽略 root/kind/which。",
    )
    parser.add_argument(
        "--kind",
        default="buffer",
        choices=["buffer", "demo_buffer"],
        help="选择可视化 online buffer 还是 intervention demo_buffer。",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="直接指定一个 pkl 文件。指定后忽略 --root/--kind/--which。",
    )
    parser.add_argument(
        "--which",
        default="latest",
        choices=["latest", "all", "index", "step"],
        help="选择 transitions_*.pkl 文件。",
    )
    parser.add_argument("--file_index", type=int, default=None, help="--which index 时使用。支持 -1。")
    parser.add_argument("--step", type=int, default=None, help="--which step 时使用，例如 1000。")

    parser.add_argument(
        "--select",
        default="episode",
        choices=["episode", "global"],
        help="episode=按 episode 编号选择；global=按全局 transition index 选择。",
    )
    parser.add_argument("--episode", type=int, default=None, help="选择单个 episode。")
    parser.add_argument("--episode_start", type=int, default=None, help="选择 episode 起点，左闭。")
    parser.add_argument("--episode_end", type=int, default=None, help="选择 episode 终点，右开。")
    parser.add_argument("--start", type=int, default=None, help="global 起点，或 episode 内 local step 起点。")
    parser.add_argument("--end", type=int, default=None, help="global 终点，或 episode 内 local step 终点，右开。")

    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=["head_rgb", "right_wrist_rgb"],
        help="要拼接显示的图像 key。",
    )
    parser.add_argument(
        "--use_next_obs",
        action="store_true",
        help="默认显示 observations；打开后显示 next_observations。",
    )
    parser.add_argument("--display_size", type=int, default=256, help="每个相机显示尺寸。")
    parser.add_argument("--panel_h", type=int, default=175, help="底部文字面板高度。")
    parser.add_argument("--fps", type=float, default=8.0, help="输出视频 fps。")

    parser.add_argument("--pos_scale", type=float, default=0.02, help="用于显示物理平移增量，单位 m/action。")
    parser.add_argument("--rot_scale", type=float, default=0.04, help="用于显示物理旋转增量，单位 rad/action。")

    parser.add_argument("--out", default=None, help="输出 mp4 路径。")
    parser.add_argument("--csv", default=None, help="输出 csv 路径。默认和 mp4 同名。")
    parser.add_argument("--max_frames", type=int, default=None, help="最多输出多少帧，避免视频过长。")
    parser.add_argument("--save_frames_dir", default=None, help="可选：同时保存每一帧 png。")

    args = parser.parse_args()

    files = resolve_files(args)
    transitions, inventory = load_files(files)
    selected, episodes, ep_meta, selected_desc = select_transitions(args, transitions)

    if args.max_frames is not None:
        selected = selected[: int(args.max_frames)]

    if not selected:
        raise ValueError("选择结果为空，没有可视化内容。")

    source_name = args.path if args.path else f"{args.kind}:{args.which}"

    if args.out is None:
        safe_kind = args.kind if args.path is None else os.path.splitext(os.path.basename(args.path))[0]
        args.out = f"visualize_{safe_kind}_{args.select}.mp4"

    if args.csv is None:
        base, _ = os.path.splitext(args.out)
        args.csv = base + ".csv"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("=" * 100)
    print("Input files")
    print("=" * 100)
    for item in inventory:
        print(
            f"[{item['file_i']}] count={item['count']}, "
            f"global=[{item['global_start']}, {item['global_end']}), "
            f"path={item['path']}"
        )

    summarize(transitions, episodes)

    print("=" * 100)
    print("Selection")
    print("=" * 100)
    print(f"selected         : {selected_desc}")
    print(f"selected frames  : {len(selected)}")
    print(f"image_keys       : {args.image_keys}")
    print(f"use_next_obs     : {args.use_next_obs}")
    print(f"pos_scale        : {args.pos_scale}")
    print(f"rot_scale        : {args.rot_scale}")
    print(f"out              : {args.out}")
    print(f"csv              : {args.csv}")

    first_frame = make_frame(
        selected[0],
        image_keys=args.image_keys,
        display_size=args.display_size,
        panel_h=args.panel_h,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        source_name=source_name,
        use_next_obs=args.use_next_obs,
    )

    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频 writer: {args.out}")

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    for i, item in enumerate(selected):
        frame = make_frame(
            item,
            image_keys=args.image_keys,
            display_size=args.display_size,
            panel_h=args.panel_h,
            pos_scale=args.pos_scale,
            rot_scale=args.rot_scale,
            source_name=source_name,
            use_next_obs=args.use_next_obs,
        )

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)

        writer.write(frame)

        if args.save_frames_dir:
            cv2.imwrite(os.path.join(args.save_frames_dir, f"frame_{i:06d}.png"), frame)

        if i % 100 == 0:
            print(f"writing frame {i}/{len(selected)}")

    writer.release()

    write_csv(args.csv, selected, pos_scale=args.pos_scale, rot_scale=args.rot_scale)

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"video saved: {args.out}")
    print(f"csv saved  : {args.csv}")


if __name__ == "__main__":
    main()