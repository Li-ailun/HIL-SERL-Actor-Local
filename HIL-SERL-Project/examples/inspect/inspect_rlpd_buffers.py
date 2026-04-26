# 运行方式：

# cd /home/eren/HIL-SERL/HIL-SERL-Project/examples
# python inspect/inspect_rlpd_buffers.py --root ./rlpd_checkpoints_single --which latest

# 如果你想看所有保存过的 pkl 合并后的统计：

# python inspect_rlpd_buffers.py --root ./rlpd_checkpoints_single --which all

# 如果还想保存成 CSV：

# python inspect_rlpd_buffers.py \
#   --root ./rlpd_checkpoints_single \
#   --which all \
#   --csv ./rlpd_checkpoints_single/buffer_summary.csv

# 你重点看这几行：

# reward_sum
# reward > 0 数量
# done=True 数量
# mask=0 数量
# gripper action 分布

# 如果你看到：

# reward_sum > 0
# reward > 0 数量 > 0
# done=True 数量 > 0
# mask=0 数量 > 0

# 那就可以说明：return=1 的数据确实进 buffer 了，问题主要就是 success 打印/字段没有对齐。


import os
import re
import glob
import csv
import argparse
import pickle as pkl
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# 基础工具
# ============================================================

def natural_key(path: str):
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def safe_np(x):
    try:
        return np.asarray(x)
    except Exception:
        return None


def scalar_float(x, default=0.0):
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def scalar_bool(x, default=False):
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return bool(arr.reshape(-1)[0])
    except Exception:
        return default


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def normalize_loaded_object(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容几种常见保存格式：
    1) list[transition]
    2) tuple/list 包一层
    3) dict 里有 transitions/data/buffer
    4) 单个 transition dict
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict) and "observations" in obj[0]:
            return obj

        out = []
        for item in obj:
            out.extend(normalize_loaded_object(item))
        return out

    if isinstance(obj, tuple):
        out = []
        for item in obj:
            out.extend(normalize_loaded_object(item))
        return out

    if isinstance(obj, dict):
        if "observations" in obj and "actions" in obj:
            return [obj]

        for key in ["transitions", "data", "buffer", "dataset"]:
            if key in obj:
                return normalize_loaded_object(obj[key])

        vals = list(obj.values())
        if vals and isinstance(vals[0], dict) and "observations" in vals[0]:
            return vals

    return []


def get_transition_files(root: str, subdir: str) -> List[str]:
    d = os.path.join(root, subdir)
    if not os.path.isdir(d):
        return []
    files = glob.glob(os.path.join(d, "*.pkl"))
    return sorted(files, key=natural_key)


def select_files(files: List[str], which: str) -> List[str]:
    if not files:
        return []
    if which == "latest":
        return [files[-1]]
    if which == "all":
        return files
    raise ValueError(f"unknown which={which}")


# ============================================================
# action / gripper 工具
# ============================================================

def flatten_action(action):
    arr = safe_np(action)
    if arr is None:
        return None

    arr = arr.astype(np.float32)

    if arr.ndim == 0:
        return arr.reshape(1)

    while arr.ndim > 1:
        arr = arr[-1]

    return arr.reshape(-1)


def gripper_label_from_action(action):
    arr = flatten_action(action)
    if arr is None or arr.size == 0:
        return "missing"

    g = float(arr[-1])

    # 三值标签
    if g <= -0.5:
        return "close(-1)"
    if g >= 0.5 and g <= 1.5:
        return "open(+1)"
    if -0.5 < g < 0.5:
        return "hold(0)"

    # 硬件量程兜底
    if 0.0 <= g <= 30.0:
        return "hardware_close"
    if 70.0 <= g <= 100.0:
        return "hardware_open"

    return f"other({g:.3f})"


def expected_grasp_penalty_from_action(action, penalty_value=-0.02) -> float:
    """
    根据最终保存的 action[6] 三值事件标签，重算期望 grasp_penalty。

    当前设计：
      action[6] = -1 close event -> penalty_value
      action[6] =  0 hold        -> 0
      action[6] = +1 open event  -> penalty_value
    """
    a = flatten_action(action)
    if a is None or a.shape[0] != 7:
        return 0.0

    g = float(a[6])

    if g <= -0.5:
        return float(penalty_value)

    if g >= 0.5:
        return float(penalty_value)

    return 0.0


# ============================================================
# grasp_penalty 工具
# ============================================================

def extract_grasp_penalty(trans: Dict[str, Any]):
    """
    读取当前训练实际会用到的 grasp_penalty。

    优先级：
      1. trans["grasp_penalty"]
      2. trans["infos"]["grasp_penalty"]

    返回：
      (penalty_float, source_string)
      如果没有找到，返回 (None, None)
    """
    if not isinstance(trans, dict):
        return None, None

    if "grasp_penalty" in trans:
        return scalar_float(trans["grasp_penalty"]), "top_level"

    infos = trans.get("infos", trans.get("info", None))
    if isinstance(infos, dict) and "grasp_penalty" in infos:
        return scalar_float(infos["grasp_penalty"]), "infos"

    return None, None


def extract_all_grasp_penalty_fields(trans: Dict[str, Any]):
    """
    同时读取所有可能存在的 penalty 字段，方便排查：
      - top_level grasp_penalty
      - infos grasp_penalty
      - infos env_grasp_penalty_raw
      - infos top_level_grasp_penalty_raw
      - infos grasp_penalty_source
    """
    out = {}

    if not isinstance(trans, dict):
        return out

    if "grasp_penalty" in trans:
        out["top_level.grasp_penalty"] = scalar_float(trans["grasp_penalty"])

    infos = trans.get("infos", trans.get("info", None))
    if isinstance(infos, dict):
        for key in [
            "grasp_penalty",
            "env_grasp_penalty_raw",
            "top_level_grasp_penalty_raw",
            "grasp_penalty_source",
        ]:
            if key in infos:
                out[f"infos.{key}"] = infos[key]

    return out


def summarize_grasp_penalty(
    transitions: List[Dict[str, Any]],
    name: str,
    expected_penalty_value=-0.02,
    mismatch_tol=1e-6,
    max_print=30,
):
    """
    检查 stored grasp_penalty 是否与最终 action[6] 对齐。
    """
    print("\n" + "=" * 90)
    print(f"[{name}] grasp_penalty 检查")
    print("=" * 90)

    n = len(transitions)

    stored_values = []
    stored_sources = []
    expected_values = []

    missing_count = 0
    stored_nonzero_indices = []
    expected_nonzero_indices = []
    mismatch_indices = []

    top_level_count = 0
    infos_count = 0
    both_count = 0
    top_info_mismatch_indices = []

    raw_env_values = []

    gripper_all = Counter()
    gripper_stored_nonzero = Counter()
    gripper_expected_nonzero = Counter()
    gripper_mismatch = Counter()

    for i, trans in enumerate(transitions):
        stored, source = extract_grasp_penalty(trans)
        expected = expected_grasp_penalty_from_action(
            trans.get("actions"),
            penalty_value=expected_penalty_value,
        )

        expected_values.append(float(expected))

        fields = extract_all_grasp_penalty_fields(trans)
        has_top = "top_level.grasp_penalty" in fields
        has_info = "infos.grasp_penalty" in fields

        if has_top:
            top_level_count += 1
        if has_info:
            infos_count += 1
        if has_top and has_info:
            both_count += 1
            top_v = scalar_float(fields["top_level.grasp_penalty"])
            info_v = scalar_float(fields["infos.grasp_penalty"])
            if abs(top_v - info_v) > mismatch_tol:
                top_info_mismatch_indices.append(i)

        if "infos.env_grasp_penalty_raw" in fields:
            raw_env_values.append(scalar_float(fields["infos.env_grasp_penalty_raw"]))

        g_desc = gripper_label_from_action(trans.get("actions"))
        gripper_all[g_desc] += 1

        if abs(expected) > 1e-8:
            expected_nonzero_indices.append(i)
            gripper_expected_nonzero[g_desc] += 1

        if stored is None:
            missing_count += 1
            stored_values.append(np.nan)
            stored_sources.append("missing")
            continue

        stored = float(stored)
        stored_values.append(stored)
        stored_sources.append(source)

        if abs(stored) > 1e-8:
            stored_nonzero_indices.append(i)
            gripper_stored_nonzero[g_desc] += 1

        if abs(stored - expected) > mismatch_tol:
            mismatch_indices.append(i)
            gripper_mismatch[g_desc] += 1

    expected_np = np.asarray(expected_values, dtype=np.float32)
    stored_np = np.asarray(stored_values, dtype=np.float32)
    valid_mask = ~np.isnan(stored_np)
    valid_stored = stored_np[valid_mask]

    expected_vals, expected_counts = np.unique(np.round(expected_np, 8), return_counts=True)
    expected_dist = {
        float(v): int(c)
        for v, c in zip(expected_vals, expected_counts)
    }

    print(f"transition 总数                 : {n}")
    print(f"expected_penalty_value          : {expected_penalty_value}")
    print(f"stored grasp_penalty 缺失数量    : {missing_count}")
    print(f"top_level grasp_penalty 数量     : {top_level_count}")
    print(f"infos grasp_penalty 数量         : {infos_count}")
    print(f"top_level + infos 同时存在数量   : {both_count}")
    print(f"top_level 与 infos 不一致数量    : {len(top_info_mismatch_indices)}")

    if valid_stored.size > 0:
        stored_vals, stored_counts = np.unique(np.round(valid_stored, 8), return_counts=True)
        stored_dist = {
            float(v): int(c)
            for v, c in zip(stored_vals, stored_counts)
        }
        source_dist = dict(Counter(stored_sources))

        print(f"\nstored 来源分布                  : {source_dist}")
        print(f"stored grasp_penalty 取值分布     : {stored_dist}")
        print(f"stored grasp_penalty 非零数量     : {len(stored_nonzero_indices)}")
        print(f"stored grasp_penalty min          : {float(np.min(valid_stored)):.6f}")
        print(f"stored grasp_penalty max          : {float(np.max(valid_stored)):.6f}")
        print(f"stored grasp_penalty mean         : {float(np.mean(valid_stored)):.6f}")
        print(f"stored grasp_penalty sum          : {float(np.sum(valid_stored)):.6f}")
    else:
        print("\n❌ 没有找到任何 stored grasp_penalty 字段。")

    print(f"\nexpected grasp_penalty 取值分布   : {expected_dist}")
    print(f"expected grasp_penalty 非零数量   : {len(expected_nonzero_indices)}")
    print(f"expected grasp_penalty sum        : {float(np.sum(expected_np)):.6f}")

    if raw_env_values:
        raw_np = np.asarray(raw_env_values, dtype=np.float32)
        vals, cnts = np.unique(np.round(raw_np, 8), return_counts=True)
        raw_dist = {
            float(v): int(c)
            for v, c in zip(vals, cnts)
        }
        print(f"\nraw env_grasp_penalty_raw 数量    : {len(raw_env_values)}")
        print(f"raw env_grasp_penalty_raw 分布    : {raw_dist}")
        print(f"raw env_grasp_penalty_raw sum     : {float(np.sum(raw_np)):.6f}")

    print(f"\ngripper 全部分布                  : {dict(gripper_all)}")
    print(f"非零 stored penalty 对应 gripper  : {dict(gripper_stored_nonzero)}")
    print(f"非零 expected penalty 对应 gripper: {dict(gripper_expected_nonzero)}")
    print(f"stored vs expected mismatch 数量  : {len(mismatch_indices)}")
    print(f"mismatch 对应 gripper 分布         : {dict(gripper_mismatch)}")

    print("\n判断：")
    if valid_stored.size == 0:
        if len(expected_nonzero_indices) > 0:
            print("⚠️ 当前 buffer 没有 stored grasp_penalty，但按 action[6] 应该存在非零 penalty。")
        else:
            print("ℹ️ 当前 buffer 没有 stored grasp_penalty，且按 action[6] 也没有非零 penalty。")
    elif len(mismatch_indices) == 0:
        print("✅ stored grasp_penalty 与 expected_grasp_penalty 完全对齐。")
        print("✅ penalty 已经跟最终保存的 action[6] 三值事件一致。")
    else:
        print("❌ stored grasp_penalty 与 expected_grasp_penalty 不一致。")
        print("   这通常说明旧数据里保存了 env.step 提前计算的 grasp_penalty。")
        print("   正确目标：hold(0)->0，close/open 事件->-0.02。")

    if gripper_stored_nonzero.get("hold(0)", 0) > 0:
        print("❌ 检测到 stored penalty 落在 hold(0) 上。")
        print("   如果你的设计是事件惩罚，这说明该 buffer 仍是旧逻辑或未同步。")

    if (
        valid_stored.size > 0
        and len(mismatch_indices) == 0
        and len(stored_nonzero_indices) == len(expected_nonzero_indices)
    ):
        print("✅ 非零 stored penalty 数量与 gripper close/open 事件数量一致。")

    if top_info_mismatch_indices:
        print("\n⚠️ top_level grasp_penalty 和 infos.grasp_penalty 不一致，前若干 index:")
        print(top_info_mismatch_indices[:max_print])

    if stored_nonzero_indices:
        print("\n前若干非零 stored grasp_penalty transition:")
        for idx in stored_nonzero_indices[:max_print]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_values[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={gripper_label_from_action(trans.get('actions'))}, "
                f"reward={scalar_float(trans.get('rewards'))}, "
                f"done={scalar_bool(trans.get('dones'))}, "
                f"mask={scalar_float(trans.get('masks'))}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )

    if expected_nonzero_indices:
        print("\n前若干 expected 非零 grasp_penalty transition:")
        for idx in expected_nonzero_indices[:max_print]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_values[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={gripper_label_from_action(trans.get('actions'))}, "
                f"reward={scalar_float(trans.get('rewards'))}, "
                f"done={scalar_bool(trans.get('dones'))}, "
                f"mask={scalar_float(trans.get('masks'))}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )

    if mismatch_indices:
        print("\n前若干 stored vs expected 不一致 transition:")
        for idx in mismatch_indices[:max_print]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_values[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={gripper_label_from_action(trans.get('actions'))}, "
                f"reward={scalar_float(trans.get('rewards'))}, "
                f"done={scalar_bool(trans.get('dones'))}, "
                f"mask={scalar_float(trans.get('masks'))}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )

    return {
        "stored_gp_missing_count": int(missing_count),
        "stored_gp_nonzero_count": int(len(stored_nonzero_indices)),
        "expected_gp_nonzero_count": int(len(expected_nonzero_indices)),
        "gp_mismatch_count": int(len(mismatch_indices)),
        "stored_gp_on_hold_count": int(gripper_stored_nonzero.get("hold(0)", 0)),
        "expected_gp_sum": float(np.sum(expected_np)),
        "stored_gp_sum": float(np.sum(valid_stored)) if valid_stored.size > 0 else 0.0,
    }


# ============================================================
# observation 递归结构
# ============================================================

def recursive_array_summary(x, prefix="", max_items=80):
    """
    递归扫描 observation / next_observation，打印数组路径、shape、dtype。
    """
    results = []

    def visit(obj, path):
        if len(results) >= max_items:
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}/{k}" if path else str(k))
            return

        arr = safe_np(obj)
        if arr is None:
            results.append((path, type(obj).__name__, None, None, None, None))
            return

        shape = tuple(arr.shape)
        dtype = str(arr.dtype)

        mn = None
        mx = None
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            try:
                mn = float(np.nanmin(arr))
                mx = float(np.nanmax(arr))
            except Exception:
                pass

        results.append((path, dtype, shape, mn, mx, arr.size))

    visit(x, prefix)
    return results


# ============================================================
# transition 汇总
# ============================================================

def summarize_transitions(
    transitions: List[Dict[str, Any]],
    name: str,
    max_success_print=10,
    expected_grasp_penalty=-0.02,
):
    n = len(transitions)

    rewards = []
    dones = []
    masks = []
    action_dims = Counter()
    gripper_counts = Counter()
    positive_indices = []
    done_indices = []

    reward_values = Counter()
    action_min = None
    action_max = None
    action_abs_max = None
    arm_out_of_range_count = 0

    for i, t in enumerate(transitions):
        r = scalar_float(t.get("rewards", 0.0))
        d = scalar_bool(t.get("dones", False))
        m = scalar_float(t.get("masks", 1.0))

        rewards.append(r)
        dones.append(d)
        masks.append(m)

        reward_values[round(r, 6)] += 1

        if r > 0:
            positive_indices.append(i)
        if d:
            done_indices.append(i)

        a = flatten_action(t.get("actions", None))
        if a is not None:
            action_dims[tuple(a.shape)] += 1
            cur_min = float(np.min(a)) if a.size else 0.0
            cur_max = float(np.max(a)) if a.size else 0.0
            cur_abs = float(np.max(np.abs(a))) if a.size else 0.0

            action_min = cur_min if action_min is None else min(action_min, cur_min)
            action_max = cur_max if action_max is None else max(action_max, cur_max)
            action_abs_max = cur_abs if action_abs_max is None else max(action_abs_max, cur_abs)

            if a.shape[0] >= 6 and np.max(np.abs(a[:6])) > 1.0001:
                arm_out_of_range_count += 1

        gripper_counts[gripper_label_from_action(t.get("actions", None))] += 1

    rewards_np = np.asarray(rewards, dtype=np.float64) if rewards else np.zeros((0,))
    dones_np = np.asarray(dones, dtype=bool) if dones else np.zeros((0,), dtype=bool)
    masks_np = np.asarray(masks, dtype=np.float64) if masks else np.zeros((0,))

    print("\n" + "=" * 90)
    print(f"[{name}] transition 总数: {n}")
    print("=" * 90)

    if n == 0:
        print("空。")
        return {
            "name": name,
            "n": 0,
            "reward_sum": 0.0,
            "reward_mean": 0.0,
            "positive_count": 0,
            "done_count": 0,
            "mask0_count": 0,
            "gripper_counts": {},
            "action_shapes": {},
            "stored_gp_missing_count": 0,
            "stored_gp_nonzero_count": 0,
            "expected_gp_nonzero_count": 0,
            "gp_mismatch_count": 0,
            "stored_gp_on_hold_count": 0,
            "stored_gp_sum": 0.0,
            "expected_gp_sum": 0.0,
        }

    print(f"reward_sum             : {float(np.sum(rewards_np)):.6f}")
    print(f"reward_mean            : {float(np.mean(rewards_np)):.6f}")
    print(f"reward > 0 数量         : {len(positive_indices)}")
    print(f"done=True 数量          : {int(np.sum(dones_np))}")
    print(f"mask=0 数量             : {int(np.sum(np.isclose(masks_np, 0.0)))}")
    print(f"reward 取值分布         : {dict(reward_values)}")
    print(f"action shape 分布       : {dict(action_dims)}")
    print(f"action min/max/absmax   : {action_min}, {action_max}, {action_abs_max}")
    print(f"前 6 维超出 [-1,1] 数量 : {arm_out_of_range_count}")
    print(f"gripper action 分布     : {dict(gripper_counts)}")

    if positive_indices:
        print(f"\n前 {max_success_print} 个 reward>0 的 transition index:")
        for idx in positive_indices[:max_success_print]:
            t = transitions[idx]
            action = flatten_action(t.get("actions"))
            stored_gp, gp_source = extract_grasp_penalty(t)
            expected_gp = expected_grasp_penalty_from_action(
                t.get("actions"),
                penalty_value=expected_grasp_penalty,
            )
            print(
                f"  idx={idx}, reward={scalar_float(t.get('rewards'))}, "
                f"done={scalar_bool(t.get('dones'))}, mask={scalar_float(t.get('masks'))}, "
                f"gripper={gripper_label_from_action(t.get('actions'))}, "
                f"stored_gp={stored_gp}, expected_gp={expected_gp}, gp_source={gp_source}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}"
            )
    else:
        print("\n没有发现 reward>0 的 transition。")

    if done_indices:
        print(f"\n前 {max_success_print} 个 done=True 的 transition index:")
        for idx in done_indices[:max_success_print]:
            t = transitions[idx]
            stored_gp, gp_source = extract_grasp_penalty(t)
            expected_gp = expected_grasp_penalty_from_action(
                t.get("actions"),
                penalty_value=expected_grasp_penalty,
            )
            print(
                f"  idx={idx}, reward={scalar_float(t.get('rewards'))}, "
                f"done={scalar_bool(t.get('dones'))}, mask={scalar_float(t.get('masks'))}, "
                f"gripper={gripper_label_from_action(t.get('actions'))}, "
                f"stored_gp={stored_gp}, expected_gp={expected_gp}, gp_source={gp_source}"
            )
    else:
        print("\n没有发现 done=True 的 transition。")

    print("\n第一个 transition 的顶层 keys:")
    print(list(transitions[0].keys()))

    if "observations" in transitions[0]:
        print("\n第一个 observations 结构:")
        obs_summary = recursive_array_summary(transitions[0]["observations"], "observations")
        for path, dtype, shape, mn, mx, size in obs_summary:
            print(f"  {path:60s} dtype={dtype:10s} shape={shape} min={mn} max={mx}")

    if "next_observations" in transitions[0]:
        print("\n第一个 next_observations 结构:")
        next_summary = recursive_array_summary(transitions[0]["next_observations"], "next_observations")
        for path, dtype, shape, mn, mx, size in next_summary:
            print(f"  {path:60s} dtype={dtype:10s} shape={shape} min={mn} max={mx}")

    gp_summary = summarize_grasp_penalty(
        transitions,
        name=name,
        expected_penalty_value=expected_grasp_penalty,
        max_print=max_success_print,
    )

    summary = {
        "name": name,
        "n": n,
        "reward_sum": float(np.sum(rewards_np)),
        "reward_mean": float(np.mean(rewards_np)),
        "positive_count": len(positive_indices),
        "done_count": int(np.sum(dones_np)),
        "mask0_count": int(np.sum(np.isclose(masks_np, 0.0))),
        "gripper_counts": dict(gripper_counts),
        "action_shapes": {str(k): v for k, v in action_dims.items()},
        "arm_out_of_range_count": arm_out_of_range_count,
    }
    summary.update(gp_summary)
    return summary


# ============================================================
# 文件读取 / CSV
# ============================================================

def load_transitions_from_files(files: List[str]):
    all_transitions = []
    per_file = []

    for path in files:
        try:
            obj = load_pickle(path)
            transitions = normalize_loaded_object(obj)
        except Exception as e:
            print(f"❌ 读取失败: {path}, error={repr(e)}")
            transitions = []

        all_transitions.extend(transitions)
        per_file.append((path, len(transitions)))

    return all_transitions, per_file


def print_file_table(title: str, per_file: List[Tuple[str, int]]):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    if not per_file:
        print("没有找到文件。")
        return
    for path, n in per_file:
        print(f"{os.path.basename(path):40s} transitions={n:8d}")


def write_csv_summary(csv_path: str, summaries: List[Dict[str, Any]]):
    if not csv_path:
        return

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "name",
        "n",
        "reward_sum",
        "reward_mean",
        "positive_count",
        "done_count",
        "mask0_count",
        "gripper_counts",
        "action_shapes",
        "arm_out_of_range_count",
        "stored_gp_missing_count",
        "stored_gp_nonzero_count",
        "expected_gp_nonzero_count",
        "gp_mismatch_count",
        "stored_gp_on_hold_count",
        "stored_gp_sum",
        "expected_gp_sum",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\n✅ CSV 汇总已保存: {csv_path}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="检查 RLPD buffer / demo_buffer 里的 transition、reward、done、三值夹爪、grasp_penalty 对齐情况。"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="./rlpd_checkpoints_single",
        help="rlpd checkpoint 根目录，例如 ./rlpd_checkpoints_single",
    )

    parser.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["latest", "all"],
        help="latest 只看最新 pkl；all 合并检查所有 pkl",
    )

    parser.add_argument(
        "--buffer_subdir",
        type=str,
        default="buffer",
        help="普通在线 buffer 子目录名",
    )

    parser.add_argument(
        "--demo_subdir",
        type=str,
        default="demo_buffer",
        help="干预/demo buffer 子目录名",
    )

    parser.add_argument(
        "--max_success_print",
        type=int,
        default=20,
        help="最多打印多少个 reward>0 / done=True / grasp_penalty 异常 transition",
    )

    parser.add_argument(
        "--expected_grasp_penalty",
        type=float,
        default=-0.02,
        help="根据最终 action[6] 重算 expected grasp_penalty 时使用的 penalty 值。",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="可选：保存汇总 CSV 路径，例如 ./buffer_summary.csv",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)

    print("=" * 90)
    print("RLPD Buffer Inspector")
    print("=" * 90)
    print(f"root                   : {root}")
    print(f"which                  : {args.which}")
    print(f"expected_grasp_penalty : {args.expected_grasp_penalty}")

    buffer_files_all = get_transition_files(root, args.buffer_subdir)
    demo_files_all = get_transition_files(root, args.demo_subdir)

    buffer_files = select_files(buffer_files_all, args.which)
    demo_files = select_files(demo_files_all, args.which)

    print_file_table(
        f"{args.buffer_subdir} 文件列表（selected={args.which}）",
        [(p, -1) for p in buffer_files],
    )
    print_file_table(
        f"{args.demo_subdir} 文件列表（selected={args.which}）",
        [(p, -1) for p in demo_files],
    )

    buffer_transitions, buffer_per_file = load_transitions_from_files(buffer_files)
    demo_transitions, demo_per_file = load_transitions_from_files(demo_files)

    print_file_table(f"{args.buffer_subdir} 每文件 transition 数", buffer_per_file)
    print_file_table(f"{args.demo_subdir} 每文件 transition 数", demo_per_file)

    summaries = []

    summaries.append(
        summarize_transitions(
            buffer_transitions,
            name=args.buffer_subdir,
            max_success_print=args.max_success_print,
            expected_grasp_penalty=args.expected_grasp_penalty,
        )
    )

    summaries.append(
        summarize_transitions(
            demo_transitions,
            name=args.demo_subdir,
            max_success_print=args.max_success_print,
            expected_grasp_penalty=args.expected_grasp_penalty,
        )
    )

    print("\n" + "=" * 90)
    print("对比结论辅助")
    print("=" * 90)

    b = summaries[0]
    d = summaries[1]

    print(
        f"{args.buffer_subdir:15s}: "
        f"n={b['n']}, reward_sum={b['reward_sum']}, positive={b['positive_count']}, done={b['done_count']}, "
        f"stored_gp_nonzero={b.get('stored_gp_nonzero_count')}, "
        f"expected_gp_nonzero={b.get('expected_gp_nonzero_count')}, "
        f"gp_mismatch={b.get('gp_mismatch_count')}, "
        f"stored_gp_on_hold={b.get('stored_gp_on_hold_count')}"
    )

    print(
        f"{args.demo_subdir:15s}: "
        f"n={d['n']}, reward_sum={d['reward_sum']}, positive={d['positive_count']}, done={d['done_count']}, "
        f"stored_gp_nonzero={d.get('stored_gp_nonzero_count')}, "
        f"expected_gp_nonzero={d.get('expected_gp_nonzero_count')}, "
        f"gp_mismatch={d.get('gp_mismatch_count')}, "
        f"stored_gp_on_hold={d.get('stored_gp_on_hold_count')}"
    )

    print("\n判断规则：")
    print("1) 如果 actor 日志 return=1，但这里 buffer 里 reward_sum/positive_count 也 > 0，说明 reward 已经正确存入。")
    print("2) 如果 reward>0 的 transition 同时 done=True 且 mask=0，说明成功终止 transition 也基本正常。")
    print("3) 如果 success 日志一直是 0，但 reward/done/mask 正常，那就是 success 字段/打印逻辑问题。")
    print("4) demo_buffer 通常对应人工介入 transition，不一定等于全部成功 demo。它的数量和 buffer 不必完全一致。")
    print("5) gripper action 分布应该主要是 hold(0)，少量 close(-1)/open(+1)。如果出现大量 hardware_open/hardware_close，说明三值写入可能没统一。")
    print("6) stored_gp_nonzero 应该等于 expected_gp_nonzero；gp_mismatch 应该为 0。")
    print("7) stored_gp_on_hold 必须为 0；如果大于 0，说明 hold(0) 被错误处罚。")
    print("8) 新版 train_rlpd.py 生成的数据，应该看到 stored grasp_penalty 与 expected 完全对齐。")

    if args.csv:
        write_csv_summary(args.csv, summaries)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import os
# import re
# import glob
# import csv
# import argparse
# import pickle as pkl
# from collections import Counter, defaultdict
# from typing import Any, Dict, List, Tuple

# import numpy as np


# def natural_key(path: str):
#     name = os.path.basename(path)
#     nums = re.findall(r"\d+", name)
#     return int(nums[-1]) if nums else -1


# def safe_np(x):
#     try:
#         return np.asarray(x)
#     except Exception:
#         return None


# def scalar_float(x, default=0.0):
#     arr = safe_np(x)
#     if arr is None or arr.size == 0:
#         return default
#     try:
#         return float(arr.reshape(-1)[0])
#     except Exception:
#         return default


# def scalar_bool(x, default=False):
#     arr = safe_np(x)
#     if arr is None or arr.size == 0:
#         return default
#     try:
#         return bool(arr.reshape(-1)[0])
#     except Exception:
#         return default


# def load_pickle(path: str):
#     with open(path, "rb") as f:
#         return pkl.load(f)


# def normalize_loaded_object(obj: Any) -> List[Dict[str, Any]]:
#     """
#     兼容几种常见保存格式：
#     1) list[transition]
#     2) tuple/list 包一层
#     3) dict 里有 transitions/data/buffer
#     4) 单个 transition dict
#     """
#     if isinstance(obj, list):
#         if len(obj) == 0:
#             return []
#         if isinstance(obj[0], dict) and "observations" in obj[0]:
#             return obj
#         out = []
#         for item in obj:
#             out.extend(normalize_loaded_object(item))
#         return out

#     if isinstance(obj, tuple):
#         out = []
#         for item in obj:
#             out.extend(normalize_loaded_object(item))
#         return out

#     if isinstance(obj, dict):
#         if "observations" in obj and "actions" in obj:
#             return [obj]

#         for key in ["transitions", "data", "buffer", "dataset"]:
#             if key in obj:
#                 return normalize_loaded_object(obj[key])

#         # 有些保存格式可能是 index -> transition
#         vals = list(obj.values())
#         if vals and isinstance(vals[0], dict) and "observations" in vals[0]:
#             return vals

#     return []


# def get_transition_files(root: str, subdir: str) -> List[str]:
#     d = os.path.join(root, subdir)
#     if not os.path.isdir(d):
#         return []
#     files = glob.glob(os.path.join(d, "*.pkl"))
#     return sorted(files, key=natural_key)


# def select_files(files: List[str], which: str) -> List[str]:
#     if not files:
#         return []
#     if which == "latest":
#         return [files[-1]]
#     if which == "all":
#         return files
#     raise ValueError(f"unknown which={which}")


# def flatten_action(action):
#     arr = safe_np(action)
#     if arr is None:
#         return None
#     arr = arr.astype(np.float32)
#     if arr.ndim == 0:
#         return arr.reshape(1)
#     while arr.ndim > 1:
#         arr = arr[-1]
#     return arr.reshape(-1)


# def gripper_label_from_action(action):
#     arr = flatten_action(action)
#     if arr is None or arr.size == 0:
#         return "missing"

#     g = float(arr[-1])

#     # 三值标签
#     if g <= -0.5:
#         return "close(-1)"
#     if g >= 0.5 and g <= 1.5:
#         return "open(+1)"
#     if -0.5 < g < 0.5:
#         return "hold(0)"

#     # 硬件量程兜底
#     if 0.0 <= g <= 30.0:
#         return "hardware_close"
#     if 70.0 <= g <= 100.0:
#         return "hardware_open"

#     return f"other({g:.3f})"


# def recursive_array_summary(x, prefix="", max_items=80):
#     """
#     递归扫描 observation / next_observation，打印数组路径、shape、dtype。
#     """
#     results = []

#     def visit(obj, path):
#         if len(results) >= max_items:
#             return

#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 visit(v, f"{path}/{k}" if path else str(k))
#             return

#         arr = safe_np(obj)
#         if arr is None:
#             results.append((path, type(obj).__name__, None, None, None, None))
#             return

#         shape = tuple(arr.shape)
#         dtype = str(arr.dtype)

#         mn = None
#         mx = None
#         if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
#             try:
#                 mn = float(np.nanmin(arr))
#                 mx = float(np.nanmax(arr))
#             except Exception:
#                 pass

#         results.append((path, dtype, shape, mn, mx, arr.size))

#     visit(x, prefix)
#     return results


# def summarize_transitions(transitions: List[Dict[str, Any]], name: str, max_success_print=10):
#     n = len(transitions)

#     rewards = []
#     dones = []
#     masks = []
#     action_dims = Counter()
#     gripper_counts = Counter()
#     positive_indices = []
#     done_indices = []

#     reward_values = Counter()
#     action_min = None
#     action_max = None
#     action_abs_max = None

#     for i, t in enumerate(transitions):
#         r = scalar_float(t.get("rewards", 0.0))
#         d = scalar_bool(t.get("dones", False))
#         m = scalar_float(t.get("masks", 1.0))

#         rewards.append(r)
#         dones.append(d)
#         masks.append(m)

#         reward_values[round(r, 6)] += 1

#         if r > 0:
#             positive_indices.append(i)
#         if d:
#             done_indices.append(i)

#         a = flatten_action(t.get("actions", None))
#         if a is not None:
#             action_dims[tuple(a.shape)] += 1
#             cur_min = float(np.min(a)) if a.size else 0.0
#             cur_max = float(np.max(a)) if a.size else 0.0
#             cur_abs = float(np.max(np.abs(a))) if a.size else 0.0

#             action_min = cur_min if action_min is None else min(action_min, cur_min)
#             action_max = cur_max if action_max is None else max(action_max, cur_max)
#             action_abs_max = cur_abs if action_abs_max is None else max(action_abs_max, cur_abs)

#         gripper_counts[gripper_label_from_action(t.get("actions", None))] += 1

#     rewards_np = np.asarray(rewards, dtype=np.float64) if rewards else np.zeros((0,))
#     dones_np = np.asarray(dones, dtype=bool) if dones else np.zeros((0,), dtype=bool)
#     masks_np = np.asarray(masks, dtype=np.float64) if masks else np.zeros((0,))

#     print("\n" + "=" * 90)
#     print(f"[{name}] transition 总数: {n}")
#     print("=" * 90)

#     if n == 0:
#         print("空。")
#         return {
#             "name": name,
#             "n": 0,
#             "reward_sum": 0.0,
#             "positive_count": 0,
#             "done_count": 0,
#         }

#     print(f"reward_sum             : {float(np.sum(rewards_np)):.6f}")
#     print(f"reward_mean            : {float(np.mean(rewards_np)):.6f}")
#     print(f"reward > 0 数量         : {len(positive_indices)}")
#     print(f"done=True 数量          : {int(np.sum(dones_np))}")
#     print(f"mask=0 数量             : {int(np.sum(np.isclose(masks_np, 0.0)))}")
#     print(f"reward 取值分布         : {dict(reward_values)}")
#     print(f"action shape 分布       : {dict(action_dims)}")
#     print(f"action min/max/absmax   : {action_min}, {action_max}, {action_abs_max}")
#     print(f"gripper action 分布     : {dict(gripper_counts)}")

#     if positive_indices:
#         print(f"\n前 {max_success_print} 个 reward>0 的 transition index:")
#         for idx in positive_indices[:max_success_print]:
#             t = transitions[idx]
#             print(
#                 f"  idx={idx}, reward={scalar_float(t.get('rewards'))}, "
#                 f"done={scalar_bool(t.get('dones'))}, mask={scalar_float(t.get('masks'))}, "
#                 f"gripper={gripper_label_from_action(t.get('actions'))}, "
#                 f"action={np.round(flatten_action(t.get('actions')), 4).tolist() if flatten_action(t.get('actions')) is not None else None}"
#             )
#     else:
#         print("\n没有发现 reward>0 的 transition。")

#     if done_indices:
#         print(f"\n前 {max_success_print} 个 done=True 的 transition index:")
#         for idx in done_indices[:max_success_print]:
#             t = transitions[idx]
#             print(
#                 f"  idx={idx}, reward={scalar_float(t.get('rewards'))}, "
#                 f"done={scalar_bool(t.get('dones'))}, mask={scalar_float(t.get('masks'))}, "
#                 f"gripper={gripper_label_from_action(t.get('actions'))}"
#             )
#     else:
#         print("\n没有发现 done=True 的 transition。")

#     print("\n第一个 transition 的顶层 keys:")
#     print(list(transitions[0].keys()))

#     if "observations" in transitions[0]:
#         print("\n第一个 observations 结构:")
#         obs_summary = recursive_array_summary(transitions[0]["observations"], "observations")
#         for path, dtype, shape, mn, mx, size in obs_summary:
#             print(f"  {path:60s} dtype={dtype:10s} shape={shape} min={mn} max={mx}")

#     if "next_observations" in transitions[0]:
#         print("\n第一个 next_observations 结构:")
#         next_summary = recursive_array_summary(transitions[0]["next_observations"], "next_observations")
#         for path, dtype, shape, mn, mx, size in next_summary:
#             print(f"  {path:60s} dtype={dtype:10s} shape={shape} min={mn} max={mx}")

#     return {
#         "name": name,
#         "n": n,
#         "reward_sum": float(np.sum(rewards_np)),
#         "reward_mean": float(np.mean(rewards_np)),
#         "positive_count": len(positive_indices),
#         "done_count": int(np.sum(dones_np)),
#         "mask0_count": int(np.sum(np.isclose(masks_np, 0.0))),
#         "gripper_counts": dict(gripper_counts),
#         "action_shapes": {str(k): v for k, v in action_dims.items()},
#     }


# def load_transitions_from_files(files: List[str]):
#     all_transitions = []
#     per_file = []

#     for path in files:
#         try:
#             obj = load_pickle(path)
#             transitions = normalize_loaded_object(obj)
#         except Exception as e:
#             print(f"❌ 读取失败: {path}, error={repr(e)}")
#             transitions = []

#         all_transitions.extend(transitions)
#         per_file.append((path, len(transitions)))

#     return all_transitions, per_file


# def print_file_table(title: str, per_file: List[Tuple[str, int]]):
#     print("\n" + "-" * 90)
#     print(title)
#     print("-" * 90)
#     if not per_file:
#         print("没有找到文件。")
#         return
#     for path, n in per_file:
#         print(f"{os.path.basename(path):30s} transitions={n:8d}")


# def write_csv_summary(csv_path: str, summaries: List[Dict[str, Any]]):
#     if not csv_path:
#         return

#     os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

#     fieldnames = [
#         "name",
#         "n",
#         "reward_sum",
#         "reward_mean",
#         "positive_count",
#         "done_count",
#         "mask0_count",
#         "gripper_counts",
#         "action_shapes",
#     ]

#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for s in summaries:
#             row = {k: s.get(k, "") for k in fieldnames}
#             writer.writerow(row)

#     print(f"\n✅ CSV 汇总已保存: {csv_path}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="检查 RLPD buffer / demo_buffer 里的 transition、reward、done、三值夹爪分布。"
#     )
#     parser.add_argument(
#         "--root",
#         type=str,
#         default="./rlpd_checkpoints_single",
#         help="rlpd checkpoint 根目录，例如 ./rlpd_checkpoints_single",
#     )
#     parser.add_argument(
#         "--which",
#         type=str,
#         default="latest",
#         choices=["latest", "all"],
#         help="latest 只看最新 pkl；all 合并检查所有 pkl",
#     )
#     parser.add_argument(
#         "--buffer_subdir",
#         type=str,
#         default="buffer",
#         help="普通在线 buffer 子目录名",
#     )
#     parser.add_argument(
#         "--demo_subdir",
#         type=str,
#         default="demo_buffer",
#         help="干预/demo buffer 子目录名",
#     )
#     parser.add_argument(
#         "--max_success_print",
#         type=int,
#         default=20,
#         help="最多打印多少个 reward>0 / done=True transition",
#     )
#     parser.add_argument(
#         "--csv",
#         type=str,
#         default="",
#         help="可选：保存汇总 CSV 路径，例如 ./buffer_summary.csv",
#     )
#     args = parser.parse_args()

#     root = os.path.abspath(args.root)

#     print("=" * 90)
#     print("RLPD Buffer Inspector")
#     print("=" * 90)
#     print(f"root  : {root}")
#     print(f"which : {args.which}")

#     buffer_files_all = get_transition_files(root, args.buffer_subdir)
#     demo_files_all = get_transition_files(root, args.demo_subdir)

#     buffer_files = select_files(buffer_files_all, args.which)
#     demo_files = select_files(demo_files_all, args.which)

#     print_file_table(f"{args.buffer_subdir} 文件列表（selected={args.which}）", [(p, -1) for p in buffer_files])
#     print_file_table(f"{args.demo_subdir} 文件列表（selected={args.which}）", [(p, -1) for p in demo_files])

#     buffer_transitions, buffer_per_file = load_transitions_from_files(buffer_files)
#     demo_transitions, demo_per_file = load_transitions_from_files(demo_files)

#     print_file_table(f"{args.buffer_subdir} 每文件 transition 数", buffer_per_file)
#     print_file_table(f"{args.demo_subdir} 每文件 transition 数", demo_per_file)

#     summaries = []
#     summaries.append(
#         summarize_transitions(
#             buffer_transitions,
#             name=args.buffer_subdir,
#             max_success_print=args.max_success_print,
#         )
#     )
#     summaries.append(
#         summarize_transitions(
#             demo_transitions,
#             name=args.demo_subdir,
#             max_success_print=args.max_success_print,
#         )
#     )

#     print("\n" + "=" * 90)
#     print("对比结论辅助")
#     print("=" * 90)

#     b = summaries[0]
#     d = summaries[1]

#     print(f"{args.buffer_subdir:15s}: n={b['n']}, reward_sum={b['reward_sum']}, positive={b['positive_count']}, done={b['done_count']}")
#     print(f"{args.demo_subdir:15s}: n={d['n']}, reward_sum={d['reward_sum']}, positive={d['positive_count']}, done={d['done_count']}")

#     print("\n判断规则：")
#     print("1) 如果 actor 日志 return=1，但这里 buffer 里 reward_sum/positive_count 也 > 0，说明 reward 已经正确存入。")
#     print("2) 如果 reward>0 的 transition 同时 done=True 且 mask=0，说明成功终止 transition 也基本正常。")
#     print("3) 如果 success 日志一直是 0，但 reward/done/mask 正常，那就是 success 字段/打印逻辑问题。")
#     print("4) demo_buffer 通常对应人工介入 transition，不一定等于全部成功 demo。它的数量和 buffer 不必完全一致。")
#     print("5) gripper action 分布应该主要是 hold(0)，少量 close(-1)/open(+1)。如果出现大量 hardware_open/hardware_close，说明三值写入可能没统一。")

#     if args.csv:
#         write_csv_summary(args.csv, summaries)


# if __name__ == "__main__":
#     main()