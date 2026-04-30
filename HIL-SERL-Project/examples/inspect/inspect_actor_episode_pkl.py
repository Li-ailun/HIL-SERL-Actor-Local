#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_episode_pkl.py

离线分析 actor 端额外保存的完整 episode 文件：
  checkpoint_path/episode/episode_*.pkl

功能：
  - 读取完整 episode payload
  - 汇总 reward/done/mask/gripper/grasp_penalty
  - 汇总 actor/intervention 比例
  - 汇总 critic Q / grasp Q
  - 汇总执行层 gripper 量程 mapped_hw / feedback / buffer 标签是否一致
  - 导出 csv
"""

import argparse
import csv
import glob
import os
import pickle
import re
from typing import Any, Dict, List, Optional

import numpy as np


def _safe_float(x, default=0.0):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def _safe_bool(x):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return bool(x)
        return bool(arr.reshape(-1)[0])
    except Exception:
        return bool(x)


def gripper_label(g):
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def action_array(t):
    return np.asarray(t.get("actions", []), dtype=np.float32).reshape(-1)


def list_episode_files(root_or_dir: str) -> List[str]:
    if os.path.isfile(root_or_dir):
        return [root_or_dir]
    if os.path.basename(root_or_dir.rstrip("/")) == "episode":
        ep_dir = root_or_dir
    else:
        ep_dir = os.path.join(root_or_dir, "episode")
    files = glob.glob(os.path.join(ep_dir, "episode_*.pkl"))
    files.sort(key=os.path.getmtime)
    return files


def resolve_path(args):
    if args.path:
        return args.path
    files = list_episode_files(args.root)
    if not files:
        raise FileNotFoundError(f"没有找到 episode_*.pkl: {args.root}")
    if args.which == "latest":
        return files[-1]
    if args.which == "index":
        idx = int(args.index)
        if idx < 0:
            idx = len(files) + idx
        return files[idx]
    if args.which == "episode_index":
        pat = re.compile(rf"episode_{int(args.episode_index):06d}_")
        hits = [p for p in files if pat.search(os.path.basename(p))]
        if not hits:
            raise FileNotFoundError(f"找不到 episode_index={args.episode_index}")
        return hits[-1]
    raise ValueError(args.which)


def load_payload(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, list):
        payload = {"metadata": {}, "transitions": payload, "step_records": [], "critic_q_values": []}
    return payload


def get_env_gripper_debug(rec: Dict[str, Any]) -> Dict[str, Any]:
    arm = rec.get("arm_side", "right")
    d = rec.get("env_gripper_publish_debug", {})
    if isinstance(d, dict):
        return d.get(arm, {}) if isinstance(d.get(arm, {}), dict) else {}
    return {}


def summarize(payload, path):
    meta = payload.get("metadata", {})
    transitions = payload.get("transitions", [])
    records = payload.get("step_records", [])
    q_values = payload.get("critic_q_values", [])

    rewards = np.array([_safe_float(t.get("rewards", 0.0)) for t in transitions], dtype=np.float64)
    dones = np.array([_safe_bool(t.get("dones", False)) for t in transitions], dtype=bool)
    masks = np.array([_safe_float(t.get("masks", 1.0), 1.0) for t in transitions], dtype=np.float64)
    gps = np.array([_safe_float(t.get("grasp_penalty", 0.0), 0.0) for t in transitions], dtype=np.float64)

    gripper_counts = {"close(-1)": 0, "hold(0)": 0, "open(+1)": 0}
    saturation = 0
    for t in transitions:
        a = action_array(t)
        if a.size >= 7:
            gripper_counts[gripper_label(a[6])] += 1
            if np.max(np.abs(a[:6])) >= 0.98:
                saturation += 1

    intervention_steps = sum(1 for r in records if r.get("had_intervene_action", False))
    mapped_hw_values = []
    feedback_before = []
    feedback_after = []
    env_open_pulse_on_hold = 0
    env_open_event = 0
    for i, r in enumerate(records):
        ed = get_env_gripper_debug(r)
        hw = ed.get("mapped_hw", ed.get("published_hw", None))
        if hw is not None and hw != "":
            mapped_hw_values.append(float(hw))
        if ed.get("feedback_before", None) not in (None, ""):
            feedback_before.append(float(ed.get("feedback_before")))
        if ed.get("feedback_after", None) not in (None, ""):
            feedback_after.append(float(ed.get("feedback_after")))
        a = action_array(transitions[i]) if i < len(transitions) else np.zeros(0)
        if a.size >= 7 and abs(float(a[6])) <= 0.5 and hw is not None and float(hw) >= 70:
            env_open_pulse_on_hold += 1
        if a.size >= 7 and float(a[6]) >= 0.5:
            env_open_event += 1

    critic_means = [q.get("critic_q_mean") for q in q_values if isinstance(q, dict) and q.get("critic_q_mean") is not None]
    critic_mins = [q.get("critic_q_min") for q in q_values if isinstance(q, dict) and q.get("critic_q_min") is not None]
    grasp_selected = [q.get("grasp_q_selected") for q in q_values if isinstance(q, dict) and q.get("grasp_q_selected") is not None]

    print("=" * 100)
    print("ACTOR SAVED EPISODE INSPECT")
    print("=" * 100)
    print(f"path                  : {path}")
    print(f"episode_index         : {meta.get('episode_index', 'N/A')}")
    print(f"global steps          : {meta.get('episode_start_step', 'N/A')} -> {meta.get('episode_end_step', 'N/A')}")
    print(f"length                : {len(transitions)}")
    print(f"return/reward_sum     : {float(rewards.sum()):.6f}")
    print(f"metadata return       : {meta.get('return', 'N/A')}")
    print(f"success               : {meta.get('success', 'N/A')}")
    print(f"reward>0              : {int((rewards > 0).sum())}")
    print(f"done=True             : {int(dones.sum())}")
    print(f"mask=0                : {int((masks == 0).sum())}")
    print(f"intervention_steps    : {intervention_steps} / {len(records)}")
    print(f"gripper dist          : {gripper_counts}")
    print(f"grasp_penalty sum     : {float(gps.sum()):.6f}")
    print(f"saturation rows       : {saturation}  (|action[:6]| >= 0.98)")
    print(f"hold but mapped_hw>=70: {env_open_pulse_on_hold}")
    print(f"buffer open events    : {env_open_event}")

    if mapped_hw_values:
        arr = np.array(mapped_hw_values, dtype=np.float64)
        unique, counts = np.unique(np.round(arr, 3), return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"mapped_hw dist        : {dist}")
    if feedback_before:
        arr = np.array(feedback_before, dtype=np.float64)
        print(f"feedback_before       : min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
    if feedback_after:
        arr = np.array(feedback_after, dtype=np.float64)
        print(f"feedback_after        : min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")

    if critic_means:
        arr = np.array(critic_means, dtype=np.float64)
        print(f"critic_q_mean         : min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    else:
        print("critic_q_mean         : N/A")
    if critic_mins:
        arr = np.array(critic_mins, dtype=np.float64)
        print(f"critic_q_min          : min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    if grasp_selected:
        arr = np.array(grasp_selected, dtype=np.float64)
        print(f"grasp_q_selected      : min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    print(f"q_summary             : {meta.get('q_summary', {})}")


def write_csv(payload, csv_path):
    transitions = payload.get("transitions", [])
    records = payload.get("step_records", [])
    q_values = payload.get("critic_q_values", [])
    meta = payload.get("metadata", {})
    pos_scale = float(meta.get("pos_scale", 0.02))
    rot_scale = float(meta.get("rot_scale", 0.04))
    fields = [
        "i", "global_step", "had_intervene_action", "action_source", "reward", "done", "mask", "grasp_penalty",
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "gripper_label",
        "pos_dx_mm", "pos_dy_mm", "pos_dz_mm", "rot_droll_deg", "rot_dpitch_deg", "rot_dyaw_deg",
        "policy_a6", "raw_intervene_a6", "stored_before_a6",
        "obs_feedback", "next_obs_feedback", "mapped_hw", "feedback_before", "feedback_after", "mem_before", "mem_after", "map_reason",
        "critic_q_mean", "critic_q_min", "critic_q_max", "critic_qs", "grasp_q_selected", "grasp_qs",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(transitions):
            rec = records[i] if i < len(records) else {}
            q = q_values[i] if i < len(q_values) else {}
            a = action_array(t)
            aa = np.zeros(7, dtype=np.float32)
            aa[: min(7, a.size)] = a[: min(7, a.size)]
            ed = get_env_gripper_debug(rec) if isinstance(rec, dict) else {}
            policy = np.asarray(rec.get("policy_action", np.zeros(7)), dtype=np.float32).reshape(-1) if isinstance(rec, dict) else np.zeros(7)
            raw_intervene = rec.get("raw_intervene_action", None) if isinstance(rec, dict) else None
            raw_intervene = np.asarray(raw_intervene, dtype=np.float32).reshape(-1) if raw_intervene is not None else np.zeros(0)
            stored_before = np.asarray(rec.get("stored_before_gripper_rewrite_action", np.zeros(7)), dtype=np.float32).reshape(-1) if isinstance(rec, dict) else np.zeros(7)
            w.writerow({
                "i": i,
                "global_step": rec.get("global_step", "") if isinstance(rec, dict) else "",
                "had_intervene_action": rec.get("had_intervene_action", "") if isinstance(rec, dict) else "",
                "action_source": rec.get("action_source", "") if isinstance(rec, dict) else "",
                "reward": _safe_float(t.get("rewards", 0.0)),
                "done": _safe_bool(t.get("dones", False)),
                "mask": _safe_float(t.get("masks", 1.0), 1.0),
                "grasp_penalty": _safe_float(t.get("grasp_penalty", 0.0), 0.0),
                "a0": float(aa[0]), "a1": float(aa[1]), "a2": float(aa[2]), "a3": float(aa[3]), "a4": float(aa[4]), "a5": float(aa[5]), "a6": float(aa[6]),
                "gripper_label": gripper_label(aa[6]),
                "pos_dx_mm": float(aa[0] * pos_scale * 1000.0), "pos_dy_mm": float(aa[1] * pos_scale * 1000.0), "pos_dz_mm": float(aa[2] * pos_scale * 1000.0),
                "rot_droll_deg": float(aa[3] * rot_scale * 180.0 / np.pi), "rot_dpitch_deg": float(aa[4] * rot_scale * 180.0 / np.pi), "rot_dyaw_deg": float(aa[5] * rot_scale * 180.0 / np.pi),
                "policy_a6": float(policy[6]) if policy.size >= 7 else "",
                "raw_intervene_a6": float(raw_intervene[6]) if raw_intervene.size >= 7 else "",
                "stored_before_a6": float(stored_before[6]) if stored_before.size >= 7 else "",
                "obs_feedback": rec.get("obs_gripper_feedback", "") if isinstance(rec, dict) else "",
                "next_obs_feedback": rec.get("next_obs_gripper_feedback", "") if isinstance(rec, dict) else "",
                "mapped_hw": ed.get("mapped_hw", ed.get("published_hw", "")),
                "feedback_before": ed.get("feedback_before", ""),
                "feedback_after": ed.get("feedback_after", ""),
                "mem_before": ed.get("mem_before", ""),
                "mem_after": ed.get("mem_after", ""),
                "map_reason": ed.get("map_reason", ""),
                "critic_q_mean": q.get("critic_q_mean", ""), "critic_q_min": q.get("critic_q_min", ""), "critic_q_max": q.get("critic_q_max", ""), "critic_qs": q.get("critic_qs", ""),
                "grasp_q_selected": q.get("grasp_q_selected", ""), "grasp_qs": q.get("grasp_qs", ""),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./rlpd_checkpoints_single", help="checkpoint root 或 episode 目录")
    ap.add_argument("--path", default=None, help="直接指定 episode_*.pkl")
    ap.add_argument("--which", choices=["latest", "index", "episode_index"], default="latest")
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--episode_index", type=int, default=None)
    ap.add_argument("--csv", default=None, help="导出 csv；默认不导出")
    args = ap.parse_args()

    path = resolve_path(args)
    payload = load_payload(path)
    summarize(payload, path)
    if args.csv:
        write_csv(payload, args.csv)
        print(f"CSV saved: {args.csv}")


if __name__ == "__main__":
    main()
