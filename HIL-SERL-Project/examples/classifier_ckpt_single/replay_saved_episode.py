#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_saved_episode.py

回放 actor 端额外保存的完整 episode。默认 dry-run，只打印每一步的：
  - 增量 action
  - 物理增量 mm/deg
  - buffer 夹爪标签
  - 记录时的硬件夹爪量程 mapped_hw/feedback/memory

加 --execute 才会真的创建 Galaxea env 并 env.step(action)。
"""

import argparse
import glob
import os
import pickle
import time
import re
from typing import Any, Dict, List

import numpy as np


def gripper_label(g):
    g = float(g)
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5:
        return "open(+1)"
    return "hold(0)"


def action_array(t):
    return np.asarray(t.get("actions", []), dtype=np.float32).reshape(-1)


def _safe_float(x, default=0.0):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def list_episode_files(root_or_dir):
    if os.path.isfile(root_or_dir):
        return [root_or_dir]
    ep_dir = root_or_dir if os.path.basename(root_or_dir.rstrip("/")) == "episode" else os.path.join(root_or_dir, "episode")
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


def get_env_gripper_debug(rec):
    if not isinstance(rec, dict):
        return {}
    arm = rec.get("arm_side", "right")
    d = rec.get("env_gripper_publish_debug", {})
    if isinstance(d, dict):
        return d.get(arm, {}) if isinstance(d.get(arm, {}), dict) else {}
    return {}


def print_step(i, t, rec, q, pos_scale, rot_scale):
    a = action_array(t)
    aa = np.zeros(7, dtype=np.float32)
    aa[: min(7, a.size)] = a[: min(7, a.size)]
    pos_mm = aa[:3] * pos_scale * 1000.0
    rot_deg = aa[3:6] * rot_scale * 180.0 / np.pi
    ed = get_env_gripper_debug(rec)
    print(
        f"[episode-replay-step] i={i:04d} global={rec.get('global_step','') if isinstance(rec,dict) else ''} "
        f"intervene={rec.get('had_intervene_action','') if isinstance(rec,dict) else ''} "
        f"a=[{aa[0]:+.3f},{aa[1]:+.3f},{aa[2]:+.3f},{aa[3]:+.3f},{aa[4]:+.3f},{aa[5]:+.3f},{aa[6]:+.1f}] "
        f"delta_mm=[{pos_mm[0]:+.1f},{pos_mm[1]:+.1f},{pos_mm[2]:+.1f}] "
        f"rot_deg=[{rot_deg[0]:+.2f},{rot_deg[1]:+.2f},{rot_deg[2]:+.2f}] "
        f"gripper={gripper_label(aa[6])} reward={_safe_float(t.get('rewards',0.0)):.2f} done={bool(t.get('dones',False))} "
        f"gp={_safe_float(t.get('grasp_penalty',0.0)):.3f} "
        f"recorded_hw={ed.get('mapped_hw', ed.get('published_hw',''))} "
        f"fb_before={ed.get('feedback_before','')} fb_after={ed.get('feedback_after','')} "
        f"mem={ed.get('mem_before','')}->{ed.get('mem_after','')} reason={ed.get('map_reason','')} "
        f"critic_q_mean={q.get('critic_q_mean','') if isinstance(q,dict) else ''} "
        f"grasp_q_selected={q.get('grasp_q_selected','') if isinstance(q,dict) else ''}"
    )


def make_env():
    # 延迟 import，确保 dry-run 不需要 ROS/JAX 环境。
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
    from examples.galaxea_task.usb_pick_insertion_single.config import env_config
    config = env_config() if callable(env_config) else env_config
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    return RecordEpisodeStatistics(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./rlpd_checkpoints_single", help="checkpoint root 或 episode 目录")
    ap.add_argument("--path", default=None)
    ap.add_argument("--which", choices=["latest", "index", "episode_index"], default="latest")
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--episode_index", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--execute", action="store_true", help="真正驱动机器人；默认只打印 dry-run")
    ap.add_argument("--sleep", type=float, default=0.0, help="额外 sleep；通常 env.step 已经有节奏")
    ap.add_argument("--stop_on_done", action="store_true")
    args = ap.parse_args()

    path = resolve_path(args)
    payload = load_payload(path)
    meta = payload.get("metadata", {})
    transitions = payload.get("transitions", [])
    records = payload.get("step_records", [])
    q_values = payload.get("critic_q_values", [])
    pos_scale = float(meta.get("pos_scale", 0.02))
    rot_scale = float(meta.get("rot_scale", 0.04))

    end = len(transitions) if args.end is None else min(int(args.end), len(transitions))
    start = max(0, int(args.start))
    print("=" * 100)
    print("SAVED EPISODE REPLAY")
    print("=" * 100)
    print(f"path       : {path}")
    print(f"episode   : {meta.get('episode_index','N/A')}")
    print(f"range     : [{start}, {end}) / {len(transitions)}")
    print(f"execute   : {args.execute}")
    print(f"pos_scale : {pos_scale}")
    print(f"rot_scale : {rot_scale}")

    env = None
    if args.execute:
        print("⚠️ 即将真实回放动作。请确认机器人处于安全状态。")
        env = make_env()
        env.reset()

    for i in range(start, end):
        t = transitions[i]
        rec = records[i] if i < len(records) else {}
        q = q_values[i] if i < len(q_values) else {}
        print_step(i, t, rec, q, pos_scale, rot_scale)
        if args.execute:
            a = action_array(t)
            obs, reward, done, truncated, info = env.step(a)
            if isinstance(info, dict):
                gd = info.get("gripper_publish_debug", {})
                if gd:
                    print(f"  live_gripper_publish_debug={gd}")
            if args.stop_on_done and (done or truncated):
                print("stop_on_done: reached terminal")
                break
        if args.sleep > 0:
            time.sleep(float(args.sleep))

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
