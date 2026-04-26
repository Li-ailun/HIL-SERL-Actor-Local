# 运行方式：

# 第一种，指定 demo 目录：

# cd /home/eren/HIL-SERL/HIL-SERL-Project/examples/inspect

# python inspect_demo_pkl.py   --dir /home/eren/HIL-SERL/HIL-SERL-Project/examples/demo_data_single

# 第二种，直接指定某个 pkl 文件：

# python inspect_demo_pkl.py    --path /home/eren/HIL-SERL/HIL-SERL-Project/examples/demo_data_single/你的_demo文件.pkl

# 保存图像预览：

# python inspect_demo_pkl.py --save_preview

# 只检查两路相机：

# python inspect_demo_pkl.py --image_keys head_rgb right_wrist_rgb


import os
import re
import glob
import argparse
import pickle as pkl
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 基础工具
# ============================================================

def natural_key(path: str):
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def find_latest_pkl(directory: str) -> str:
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"在 {directory} 下没有找到任何 .pkl 文件")
    return max(pkl_files, key=os.path.getmtime)


def list_pkl_files(directory: str) -> List[str]:
    files = glob.glob(os.path.join(directory, "*.pkl"))
    return sorted(files, key=lambda p: (os.path.getmtime(p), natural_key(p)))


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def safe_np(x):
    try:
        return np.asarray(x)
    except Exception:
        return None


def safe_float(x, default=0.0):
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def safe_bool(x, default=False):
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return bool(arr.reshape(-1)[0])
    except Exception:
        return default


def summarize_array(name: str, arr: Any, max_items: int = 10):
    arr = np.asarray(arr)
    flat = arr.reshape(-1)

    preview = flat[:max_items]
    mn = None
    mx = None
    mean = None
    std = None

    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        try:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
        except Exception:
            pass

    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={mn}, max={mx}, mean={mean}, std={std}, preview={preview}"
    )


def normalize_loaded_object(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容几种保存格式：
    1. list[transition]
    2. dict 包 transitions / data / buffer / dataset
    3. 单个 transition dict
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict) and "observations" in obj[0] and "actions" in obj[0]:
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


# ============================================================
# observation / image 工具
# ============================================================

def get_obs_dict(trans: Dict[str, Any], key: str = "observations"):
    obs = trans.get(key, {})
    if not isinstance(obs, dict):
        return {}
    return obs


def get_image_from_obs(obs: Dict[str, Any], image_key: str):
    """
    支持两种结构：
    obs["head_rgb"]
    obs["images"]["head_rgb"]
    """
    if image_key in obs:
        return obs[image_key]

    if "images" in obs and isinstance(obs["images"], dict) and image_key in obs["images"]:
        return obs["images"][image_key]

    return None


def unwrap_image(img: Any):
    """
    把常见 image shape 规整成 HWC：
    (1, H, W, C) -> (H, W, C)
    (H, W, C) -> (H, W, C)
    """
    arr = np.asarray(img)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]

    return arr


def recursive_array_summary(obj: Any, prefix: str = "", max_items: int = 120):
    results = []

    def visit(x, path):
        if len(results) >= max_items:
            return

        if isinstance(x, dict):
            for k, v in x.items():
                next_path = f"{path}/{k}" if path else str(k)
                visit(v, next_path)
            return

        arr = safe_np(x)
        if arr is None:
            results.append((path, type(x).__name__, None, None, None, None, None))
            return

        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        mn = mx = mean = None

        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            try:
                mn = float(np.nanmin(arr))
                mx = float(np.nanmax(arr))
                mean = float(np.nanmean(arr))
            except Exception:
                pass

        results.append((path, dtype, shape, mn, mx, mean, arr.size))

    visit(obj, prefix)
    return results


def print_recursive_summary(title: str, obj: Any, max_items: int = 120):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    rows = recursive_array_summary(obj, prefix="", max_items=max_items)
    for path, dtype, shape, mn, mx, mean, size in rows:
        print(
            f"{path:65s} dtype={str(dtype):10s} "
            f"shape={shape} min={mn} max={mx} mean={mean}"
        )


# ============================================================
# action / gripper 工具
# ============================================================

def flatten_action(action: Any):
    arr = safe_np(action)
    if arr is None:
        return None

    arr = arr.astype(np.float32)

    if arr.ndim == 0:
        return arr.reshape(1)

    while arr.ndim > 1:
        arr = arr[-1]

    return arr.reshape(-1)


def classify_gripper_from_action(action: Any):
    arr = flatten_action(action)
    if arr is None or arr.size == 0:
        return "missing"

    g = float(arr[-1])

    # 三值动作标签
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

    return f"other({g:.4f})"


def action_range_stats(actions: List[np.ndarray]):
    if not actions:
        return None

    arr = np.stack(actions, axis=0)
    return {
        "shape": arr.shape,
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "absmax": np.max(np.abs(arr), axis=0),
        "global_min": float(np.min(arr)),
        "global_max": float(np.max(arr)),
        "global_absmax": float(np.max(np.abs(arr))),
    }


def print_action_stats(actions: List[np.ndarray]):
    stats = action_range_stats(actions)
    if stats is None:
        print("没有有效 action。")
        return

    print("\n" + "=" * 100)
    print("actions 详细统计")
    print("=" * 100)

    print(f"action array shape: {stats['shape']}")
    print(f"global min/max/absmax: {stats['global_min']}, {stats['global_max']}, {stats['global_absmax']}")

    dim = stats["shape"][-1]
    for i in range(dim):
        name = f"a[{i}]"
        if dim == 7:
            names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
            name = names[i]

        print(
            f"{i:02d} {name:10s} "
            f"min={stats['min'][i]: .6f}, "
            f"max={stats['max'][i]: .6f}, "
            f"mean={stats['mean'][i]: .6f}, "
            f"std={stats['std'][i]: .6f}, "
            f"absmax={stats['absmax'][i]: .6f}"
        )

    if dim == 7:
        arm_absmax = float(np.max(np.abs(np.stack(actions)[:, :6])))
        grip_values = np.stack(actions)[:, 6]
        unique_grip = Counter(np.round(grip_values, 6).tolist())

        print("\n单臂动作判断：")
        print("✅ action 维度为 7，通常为 [dx, dy, dz, droll, dpitch, dyaw, gripper]")
        print(f"前 6 维 arm absmax = {arm_absmax:.6f}")
        print(f"gripper 唯一值分布: {dict(unique_grip)}")

        if arm_absmax > 1.0001:
            print("⚠️ 前 6 维动作超过 [-1, 1]，说明 demo action 可能没有 clip/归一化。")
        else:
            print("✅ 前 6 维动作都在 [-1, 1] 范围内。")

        allowed = {-1.0, 0.0, 1.0}
        rounded_set = set(np.round(grip_values, 6).tolist())
        if rounded_set.issubset(allowed):
            print("✅ gripper 是标准三值标签：-1 / 0 / +1")
        else:
            print("⚠️ gripper 不是纯三值标签，可能混入硬件量程或连续值。")

    elif dim == 14:
        print("\n⚠️ action 维度为 14，看起来像双臂动作。")
    else:
        print(f"\nℹ️ action 维度为 {dim}，不是常见单臂 7 或双臂 14。")


# ============================================================
# transition 统计
# ============================================================

def get_reward(trans: Dict[str, Any]):
    return safe_float(trans.get("rewards", trans.get("reward", 0.0)))


def get_done(trans: Dict[str, Any]):
    return safe_bool(trans.get("dones", trans.get("done", False)))


def get_mask(trans: Dict[str, Any]):
    return safe_float(trans.get("masks", trans.get("mask", 1.0)))


def extract_grasp_penalty(trans: Dict[str, Any]):
    """
    读取当前训练实际会用到的 grasp_penalty。

    优先级：
    1. trans["grasp_penalty"]，新 demo / RLPD buffer 推荐顶层保存
    2. trans["infos"]["grasp_penalty"]，旧 demo 通常在 infos 里

    返回:
      (penalty_float, source_string)
      如果没有找到，返回 (None, None)
    """
    if not isinstance(trans, dict):
        return None, None

    if "grasp_penalty" in trans:
        return safe_float(trans["grasp_penalty"]), "top_level"

    infos = trans.get("infos", trans.get("info", None))
    if isinstance(infos, dict) and "grasp_penalty" in infos:
        return safe_float(infos["grasp_penalty"]), "infos"

    return None, None


def extract_all_grasp_penalty_fields(trans: Dict[str, Any]):
    """
    同时读取所有可能存在的 penalty 字段，便于排查：
    - top_level grasp_penalty
    - infos grasp_penalty
    - infos env_grasp_penalty_raw
    - infos top_level_grasp_penalty_raw
    """
    out = {}

    if not isinstance(trans, dict):
        return out

    if "grasp_penalty" in trans:
        out["top_level.grasp_penalty"] = safe_float(trans["grasp_penalty"])

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


def expected_grasp_penalty_from_action(action: Any, penalty_value: float = -0.02) -> float:
    """
    根据最终保存的 action[6] 三值事件标签重算期望 grasp_penalty。

    当前设计：
      action[6] = -1 close event -> penalty_value
      action[6] =  0 hold        -> 0
      action[6] = +1 open event  -> penalty_value

    这个函数用于检查 demos / buffer 里保存的 grasp_penalty 是否和最终 action[6] 对齐。
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


def print_grasp_penalty_stats(
    transitions: List[Dict[str, Any]],
    expected_penalty_value: float = -0.02,
    mismatch_tol: float = 1e-6,
):
    """
    统计 grasp_penalty，并和根据最终 action[6] 重算的 expected_grasp_penalty 对比。

    重点检查：
    - 当前保存的 penalty 分布
    - 根据 action[6] 应该得到的 penalty 分布
    - stored 与 expected 是否一致
    - 非零 penalty 是否落在 close/open 上，而不是 hold 上
    """
    print("\n" + "=" * 100)
    print("grasp_penalty 检查")
    print("=" * 100)

    stored_penalties = []
    stored_sources = []
    expected_penalties = []

    nonzero_stored_indices = []
    nonzero_expected_indices = []
    mismatch_indices = []
    missing_count = 0

    top_level_count = 0
    infos_count = 0
    both_count = 0
    top_info_mismatch = []

    raw_env_penalties = []
    raw_env_indices = []

    for i, trans in enumerate(transitions):
        stored, source = extract_grasp_penalty(trans)
        expected = expected_grasp_penalty_from_action(
            trans.get("actions"),
            penalty_value=expected_penalty_value,
        )

        expected_penalties.append(float(expected))

        fields = extract_all_grasp_penalty_fields(trans)
        has_top = "top_level.grasp_penalty" in fields
        has_info = "infos.grasp_penalty" in fields

        if has_top:
            top_level_count += 1
        if has_info:
            infos_count += 1
        if has_top and has_info:
            both_count += 1
            top_v = safe_float(fields["top_level.grasp_penalty"])
            info_v = safe_float(fields["infos.grasp_penalty"])
            if abs(top_v - info_v) > mismatch_tol:
                top_info_mismatch.append(i)

        if "infos.env_grasp_penalty_raw" in fields:
            raw_env_penalties.append(safe_float(fields["infos.env_grasp_penalty_raw"]))
            raw_env_indices.append(i)

        if stored is None:
            missing_count += 1
            stored_penalties.append(np.nan)
            stored_sources.append("missing")
            continue

        stored = float(stored)
        stored_penalties.append(stored)
        stored_sources.append(source)

        if abs(stored) > 1e-8:
            nonzero_stored_indices.append(i)

        if abs(float(expected)) > 1e-8:
            nonzero_expected_indices.append(i)

        if abs(stored - expected) > mismatch_tol:
            mismatch_indices.append(i)

    if len(transitions) == 0:
        print("文件为空，无法检查 grasp_penalty。")
        return

    expected_np = np.asarray(expected_penalties, dtype=np.float32)
    stored_np = np.asarray(stored_penalties, dtype=np.float32)

    valid_stored_mask = ~np.isnan(stored_np)
    valid_stored = stored_np[valid_stored_mask]

    expected_vals, expected_counts = np.unique(np.round(expected_np, 8), return_counts=True)
    expected_dist = {float(v): int(c) for v, c in zip(expected_vals, expected_counts)}

    print(f"expected_penalty_value         : {expected_penalty_value}")
    print(f"transition 总数                : {len(transitions)}")
    print(f"grasp_penalty 缺失数量          : {missing_count}")
    print(f"top_level grasp_penalty 数量    : {top_level_count}")
    print(f"infos grasp_penalty 数量        : {infos_count}")
    print(f"top_level + infos 同时存在数量  : {both_count}")
    print(f"top_level 与 infos 不一致数量   : {len(top_info_mismatch)}")

    if len(valid_stored) == 0:
        print("\n❌ 没有找到任何可用 grasp_penalty 字段。")
        print("   当前 pkl 里既没有 trans['grasp_penalty']，也没有 trans['infos']['grasp_penalty']。")
        print("   但下面仍会显示 expected_grasp_penalty，用于判断按 action[6] 应该有几个 penalty。")
    else:
        unique_vals, counts = np.unique(np.round(valid_stored, 8), return_counts=True)
        stored_dist = {float(v): int(c) for v, c in zip(unique_vals, counts)}
        source_dist = dict(Counter(stored_sources))

        print(f"\nstored grasp_penalty 来源分布    : {source_dist}")
        print(f"stored grasp_penalty 取值分布    : {stored_dist}")
        print(f"stored grasp_penalty 非零数量    : {len(nonzero_stored_indices)}")
        print(f"stored grasp_penalty min         : {float(np.min(valid_stored)):.6f}")
        print(f"stored grasp_penalty max         : {float(np.max(valid_stored)):.6f}")
        print(f"stored grasp_penalty mean        : {float(np.mean(valid_stored)):.6f}")
        print(f"stored grasp_penalty sum         : {float(np.sum(valid_stored)):.6f}")

    print(f"\nexpected grasp_penalty 取值分布  : {expected_dist}")
    print(f"expected grasp_penalty 非零数量  : {len(nonzero_expected_indices)}")
    print(f"expected grasp_penalty sum       : {float(np.sum(expected_np)):.6f}")

    if raw_env_penalties:
        raw_np = np.asarray(raw_env_penalties, dtype=np.float32)
        vals, cnts = np.unique(np.round(raw_np, 8), return_counts=True)
        raw_dist = {float(v): int(c) for v, c in zip(vals, cnts)}
        print(f"\nraw env_grasp_penalty_raw 数量   : {len(raw_env_penalties)}")
        print(f"raw env_grasp_penalty_raw 分布   : {raw_dist}")
        print(f"raw env_grasp_penalty_raw sum    : {float(np.sum(raw_np)):.6f}")

    # 和 gripper action 做对应统计
    gripper_all = Counter()
    gripper_with_stored_penalty = Counter()
    gripper_with_expected_penalty = Counter()
    gripper_with_mismatch = Counter()

    for i, trans in enumerate(transitions):
        g_desc = classify_gripper_from_action(trans.get("actions"))
        gripper_all[g_desc] += 1

        stored, _ = extract_grasp_penalty(trans)
        expected = expected_penalties[i]

        if stored is not None and abs(float(stored)) > 1e-8:
            gripper_with_stored_penalty[g_desc] += 1

        if abs(float(expected)) > 1e-8:
            gripper_with_expected_penalty[g_desc] += 1

        if i in mismatch_indices:
            gripper_with_mismatch[g_desc] += 1

    print(f"\ngripper 全部分布                 : {dict(gripper_all)}")
    print(f"非零 stored penalty 对应 gripper : {dict(gripper_with_stored_penalty)}")
    print(f"非零 expected penalty 对应 gripper: {dict(gripper_with_expected_penalty)}")
    print(f"stored vs expected mismatch 数量 : {len(mismatch_indices)}")
    print(f"mismatch 对应 gripper 分布        : {dict(gripper_with_mismatch)}")

    # 给出直接判断
    print("\n判断：")
    if len(valid_stored) == 0:
        if len(nonzero_expected_indices) > 0:
            print("⚠️ 当前文件没有 stored grasp_penalty，但按 action[6] 应该存在非零 penalty。")
        else:
            print("ℹ️ 当前文件没有 stored grasp_penalty，且按 action[6] 也没有非零 penalty。")
    elif len(mismatch_indices) == 0:
        print("✅ stored grasp_penalty 与 expected_grasp_penalty 完全对齐。")
        print("✅ penalty 已经跟最终保存的 action[6] 三值事件一致。")
    else:
        print("❌ stored grasp_penalty 与 expected_grasp_penalty 不一致。")
        print("   这通常说明 env.step 里提前算出的 info['grasp_penalty'] 没有在 action[6] 重写后同步重算。")
        print("   正确目标：hold(0)->0，close/open 事件->-0.02。")

    if gripper_with_stored_penalty.get("hold(0)", 0) > 0:
        print("❌ 检测到 stored penalty 落在 hold(0) 上。")
        print("   如果你的设计是事件惩罚，这说明当前 pkl 里的 grasp_penalty 仍是旧逻辑或未同步。")

    if len(nonzero_expected_indices) > 0 and len(nonzero_stored_indices) == len(nonzero_expected_indices) and len(mismatch_indices) == 0:
        print("✅ 非零 penalty 数量与 gripper close/open 事件数量一致。")

    if len(top_info_mismatch) > 0:
        print("\n⚠️ top_level grasp_penalty 和 infos.grasp_penalty 存在不一致。前若干 index:")
        print(top_info_mismatch[:30])

    if len(nonzero_stored_indices) > 0:
        print("\n前若干非零 stored grasp_penalty transition:")
        for idx in nonzero_stored_indices[:30]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_penalties[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}, "
                f"reward={get_reward(trans)}, done={get_done(trans)}, mask={get_mask(trans)}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )

    if len(nonzero_expected_indices) > 0:
        print("\n前若干 expected 非零 grasp_penalty transition:")
        for idx in nonzero_expected_indices[:30]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_penalties[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}, "
                f"reward={get_reward(trans)}, done={get_done(trans)}, mask={get_mask(trans)}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )

    if len(mismatch_indices) > 0:
        print("\n前若干 stored vs expected 不一致 transition:")
        for idx in mismatch_indices[:40]:
            trans = transitions[idx]
            stored, source = extract_grasp_penalty(trans)
            expected = expected_penalties[idx]
            action = flatten_action(trans.get("actions"))
            fields = extract_all_grasp_penalty_fields(trans)

            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}, "
                f"reward={get_reward(trans)}, done={get_done(trans)}, mask={get_mask(trans)}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, "
                f"fields={fields}"
            )


def summarize_transitions(
    transitions: List[Dict[str, Any]],
    image_keys: List[str],
    sample_count: int = 5,
    expected_grasp_penalty: float = -0.02,
):
    n = len(transitions)

    print("\n" + "=" * 100)
    print("Demo 文件总览")
    print("=" * 100)

    print(f"transition 总数: {n}")
    if n == 0:
        print("文件为空。")
        return

    first = transitions[0]
    print("\n顶层 keys:")
    print(list(first.keys()))

    obs = get_obs_dict(first, "observations")
    next_obs = get_obs_dict(first, "next_observations")

    print("\nobservations 顶层 keys:")
    print(list(obs.keys()))

    if next_obs:
        print("\nnext_observations 顶层 keys:")
        print(list(next_obs.keys()))
    else:
        print("\nnext_observations 不存在或不是 dict。")

    # ----------------------------------------
    # reward / done / mask
    # ----------------------------------------
    rewards = []
    dones = []
    masks = []
    done_indices = []
    reward_pos_indices = []

    for i, trans in enumerate(transitions):
        r = get_reward(trans)
        d = get_done(trans)
        m = get_mask(trans)

        rewards.append(r)
        dones.append(d)
        masks.append(m)

        if r > 0:
            reward_pos_indices.append(i)
        if d:
            done_indices.append(i)

    rewards_np = np.asarray(rewards, dtype=np.float64)
    dones_np = np.asarray(dones, dtype=bool)
    masks_np = np.asarray(masks, dtype=np.float64)

    print("\n" + "=" * 100)
    print("reward / done / mask 统计")
    print("=" * 100)
    print(f"reward_sum         : {float(np.sum(rewards_np)):.6f}")
    print(f"reward_mean        : {float(np.mean(rewards_np)):.6f}")
    print(f"reward > 0 数量     : {len(reward_pos_indices)}")
    print(f"done=True 数量      : {int(np.sum(dones_np))}")
    print(f"mask=0 数量         : {int(np.sum(np.isclose(masks_np, 0.0)))}")
    print(f"reward 取值分布     : {dict(Counter(np.round(rewards_np, 6).tolist()))}")

    if reward_pos_indices:
        print("\n前若干 reward>0 transition:")
        for idx in reward_pos_indices[:20]:
            trans = transitions[idx]
            action = flatten_action(trans.get("actions"))
            print(
                f"  idx={idx}, reward={get_reward(trans)}, "
                f"done={get_done(trans)}, mask={get_mask(trans)}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}"
            )
    else:
        print("\n没有 reward>0 的 transition。")

    if done_indices:
        print("\n前若干 done=True transition:")
        for idx in done_indices[:20]:
            trans = transitions[idx]
            print(
                f"  idx={idx}, reward={get_reward(trans)}, "
                f"done={get_done(trans)}, mask={get_mask(trans)}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}"
            )
    else:
        print("\n没有 done=True 的 transition。")

    # ----------------------------------------
    # episode 长度粗略统计
    # ----------------------------------------
    print("\n" + "=" * 100)
    print("episode 粗略统计")
    print("=" * 100)

    if done_indices:
        starts = [0] + [i + 1 for i in done_indices[:-1]]
        lengths = [d - s + 1 for s, d in zip(starts, done_indices)]
        tail_len = n - (done_indices[-1] + 1)

        print(f"已完成 episode 数: {len(done_indices)}")
        print(f"已完成 episode 长度: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")
        print(f"最后一个未完成片段长度: {tail_len}")
        print(f"done indices 前 20 个: {done_indices[:20]}")
    else:
        print("没有 done=True，无法按 done 切 episode。")

    # ----------------------------------------
    # action
    # ----------------------------------------
    actions = []
    dims = Counter()
    gripper_counts = Counter()
    static_count = 0
    out_of_range_count = 0

    for trans in transitions:
        a = flatten_action(trans.get("actions"))
        if a is None:
            continue

        actions.append(a)
        dims[a.shape[0]] += 1
        gripper_counts[classify_gripper_from_action(a)] += 1

        if np.allclose(a, 0.0, atol=1e-8):
            static_count += 1

        if a.shape[0] >= 6 and np.max(np.abs(a[:6])) > 1.0001:
            out_of_range_count += 1

    print("\n" + "=" * 100)
    print("action 基础统计")
    print("=" * 100)
    print(f"action 维度分布       : {dict(dims)}")
    print(f"静止动作数量          : {static_count}")
    print(f"前 6 维超出 [-1,1] 数量: {out_of_range_count}")
    print(f"gripper action 分布   : {dict(gripper_counts)}")

    print_action_stats(actions)

    print("\n前几条 actions 抽样:")
    for i in range(min(sample_count, n)):
        a = flatten_action(transitions[i].get("actions"))
        print(f"  [{i}] shape={a.shape if a is not None else None}, action={a}")

    # ----------------------------------------
    # state
    # ----------------------------------------
    print("\n" + "=" * 100)
    print("state / proprio 检查")
    print("=" * 100)

    if "state" in obs:
        state = obs["state"]
        summarize_array("first observations/state", state)

        state_arr = np.asarray(state)
        last_dim = state_arr.shape[-1] if state_arr.ndim > 0 else 1

        if last_dim == 8:
            print("✅ state 最后一维 = 8，很像单臂 proprio：7维 ee pose + 1维 gripper")
        elif last_dim == 16:
            print("⚠️ state 最后一维 = 16，很像双臂 proprio")
        else:
            print(f"ℹ️ state 最后一维 = {last_dim}，请结合 config 判断。")

        state_samples = []
        for trans in transitions:
            o = get_obs_dict(trans, "observations")
            if "state" in o:
                state_samples.append(np.asarray(o["state"]).reshape(-1))

        if state_samples:
            st = np.stack(state_samples, axis=0)
            print("\n全文件 state 统计:")
            for i in range(st.shape[-1]):
                print(
                    f"  state[{i:02d}] min={np.min(st[:, i]): .6f}, "
                    f"max={np.max(st[:, i]): .6f}, "
                    f"mean={np.mean(st[:, i]): .6f}, "
                    f"std={np.std(st[:, i]): .6f}"
                )
    else:
        print("observations 里没有 state。")

    # ----------------------------------------
    # images
    # ----------------------------------------
    print("\n" + "=" * 100)
    print("图像 key / shape 检查")
    print("=" * 100)

    for k in image_keys:
        img = get_image_from_obs(obs, k)
        if img is None:
            print(f"{k}: ❌ 不存在")
        else:
            summarize_array(k, img, max_items=3)

    image_shape_counter = defaultdict(Counter)
    image_missing_counter = Counter()

    for trans in transitions:
        o = get_obs_dict(trans, "observations")
        for k in image_keys:
            img = get_image_from_obs(o, k)
            if img is None:
                image_missing_counter[k] += 1
            else:
                arr = np.asarray(img)
                image_shape_counter[k][tuple(arr.shape)] += 1

    print("\n全文件图像 shape 分布:")
    for k in image_keys:
        print(f"  {k}:")
        if image_shape_counter[k]:
            for shape, cnt in image_shape_counter[k].items():
                print(f"    shape={shape}, count={cnt}")
        if image_missing_counter[k] > 0:
            print(f"    missing={image_missing_counter[k]}")

    # ----------------------------------------
    # observations 递归结构
    # ----------------------------------------
    print_recursive_summary("第一个 transition 的 observations 递归结构", obs)
    if next_obs:
        print_recursive_summary("第一个 transition 的 next_observations 递归结构", next_obs)

    # ----------------------------------------
    # infos
    # ----------------------------------------
    print("\n" + "=" * 100)
    print("infos 检查")
    print("=" * 100)

    infos = first.get("infos", first.get("info", {}))
    print("type(infos):", type(infos))
    if isinstance(infos, dict):
        print("infos keys:", list(infos.keys()))
        for k, v in infos.items():
            if isinstance(v, (int, float, bool, str)):
                print(f"  {k}: {v}")
            else:
                arr = safe_np(v)
                if arr is not None:
                    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    print(f"  {k}: type={type(v)}")
    else:
        print(infos)

    # ----------------------------------------
    # grasp_penalty
    # ----------------------------------------
    print_grasp_penalty_stats(
        transitions,
        expected_penalty_value=expected_grasp_penalty,
    )


# ============================================================
# 可选保存图片预览
# ============================================================

def save_image_previews(
    transitions: List[Dict[str, Any]],
    image_keys: List[str],
    out_dir: str,
    indices: Optional[List[int]] = None,
    max_count: int = 10,
    expected_grasp_penalty: float = -0.02,
):
    try:
        import cv2
    except Exception as e:
        print(f"⚠️ 无法导入 cv2，跳过图片保存: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)

    n = len(transitions)
    if n == 0:
        return

    if indices is None:
        candidate = [0, n // 4, n // 2, (3 * n) // 4, n - 1]

        done_indices = [i for i, t in enumerate(transitions) if get_done(t)]
        reward_indices = [i for i, t in enumerate(transitions) if get_reward(t) > 0]

        candidate.extend(done_indices[:max_count])
        candidate.extend(reward_indices[:max_count])

        # 去重并裁剪
        indices = []
        for x in candidate:
            if 0 <= x < n and x not in indices:
                indices.append(x)
        indices = indices[:max_count]

    print("\n" + "=" * 100)
    print("保存图像预览")
    print("=" * 100)
    print(f"输出目录: {out_dir}")
    print(f"indices: {indices}")

    for idx in indices:
        trans = transitions[idx]
        obs = get_obs_dict(trans, "observations")

        imgs = []

        for k in image_keys:
            img = get_image_from_obs(obs, k)
            if img is None:
                continue

            arr = unwrap_image(img)

            if arr.ndim != 3:
                continue

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            # 假设数据是 RGB，cv2 保存需要 BGR
            if arr.shape[-1] == 3:
                save_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                save_arr = arr

            imgs.append(save_arr)

        if not imgs:
            continue

        # 统一高度
        h = min(im.shape[0] for im in imgs)
        resized = []
        for im in imgs:
            scale = h / im.shape[0]
            w = int(im.shape[1] * scale)
            resized.append(cv2.resize(im, (w, h)))

        canvas = np.concatenate(resized, axis=1)

        stored_penalty, _ = extract_grasp_penalty(trans)
        expected_penalty = expected_grasp_penalty_from_action(
            trans.get("actions"),
            penalty_value=expected_grasp_penalty,
        )

        stored_text = "None" if stored_penalty is None else f"{stored_penalty:.4f}"
        expected_text = f"{expected_penalty:.4f}"

        text = (
            f"idx={idx} reward={get_reward(trans)} done={get_done(trans)} "
            f"mask={get_mask(trans)} gripper={classify_gripper_from_action(trans.get('actions'))} "
            f"stored_gp={stored_text} expected_gp={expected_text}"
        )

        # 加一条文字区域
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_h = 40
        board = np.zeros((canvas.shape[0] + text_h, canvas.shape[1], 3), dtype=np.uint8)
        board[:canvas.shape[0]] = canvas
        board[canvas.shape[0]:] = 255

        cv2.putText(
            board,
            text,
            (10, canvas.shape[0] + 26),
            font,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        filename = os.path.join(out_dir, f"preview_idx_{idx:06d}.jpg")
        cv2.imwrite(filename, board)
        print(f"保存: {filename}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="检查 / 分析录制的 SERL demos pkl 数据。"
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="指定某个 demo pkl 文件。不填则自动找 --dir 下最新 pkl。",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=BASE_DIR,
        help="demo pkl 所在目录。默认是脚本所在目录。",
    )

    parser.add_argument(
        "--image_keys",
        type=str,
        nargs="+",
        default=["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        help="需要检查的图像 key。",
    )

    parser.add_argument(
        "--sample_count",
        type=int,
        default=5,
        help="打印前几条 action 抽样。",
    )

    parser.add_argument(
        "--expected_grasp_penalty",
        type=float,
        default=-0.02,
        help="根据最终 action[6] 重算 expected grasp_penalty 时使用的 penalty 值。",
    )

    parser.add_argument(
        "--save_preview",
        action="store_true",
        help="是否保存若干帧图像预览。",
    )

    parser.add_argument(
        "--preview_dir",
        type=str,
        default=None,
        help="图像预览保存目录。默认放在 pkl 同目录 demo_preview/。",
    )

    parser.add_argument(
        "--preview_count",
        type=int,
        default=12,
        help="最多保存多少张预览图。",
    )

    args = parser.parse_args()

    directory = os.path.abspath(args.dir)
    path = os.path.abspath(args.path) if args.path else find_latest_pkl(directory)

    print("=" * 100)
    print("SERL Demo PKL Inspector")
    print("=" * 100)
    print(f"demo dir : {directory}")
    print(f"pkl path : {path}")
    print(f"expected_grasp_penalty: {args.expected_grasp_penalty}")

    obj = load_pickle(path)
    transitions = normalize_loaded_object(obj)

    summarize_transitions(
        transitions,
        image_keys=args.image_keys,
        sample_count=args.sample_count,
        expected_grasp_penalty=args.expected_grasp_penalty,
    )

    if args.save_preview:
        preview_dir = args.preview_dir
        if preview_dir is None:
            preview_dir = os.path.join(os.path.dirname(path), "demo_preview")

        save_image_previews(
            transitions,
            image_keys=args.image_keys,
            out_dir=os.path.abspath(preview_dir),
            indices=None,
            max_count=args.preview_count,
            expected_grasp_penalty=args.expected_grasp_penalty,
        )

    print("\n" + "=" * 100)
    print("结论提示")
    print("=" * 100)
    print("1. 单臂任务通常应看到 action shape=(7)，state 最后一维=8。")
    print("2. 如果 demos 用于当前 RLPD，demo 里至少要包含当前 image_keys 需要的相机。")
    print("3. 如果 action 前 6 维超过 [-1,1]，说明动作归一化/clip 有问题。")
    print("4. 如果 gripper 不是 -1/0/+1 三值，说明夹爪标签和当前训练定义不一致。")
    print("5. 如果 reward=1 的 transition 同时 done=True 且 mask=0，说明成功终止数据正常。")
    print("6. 如果图像 shape 都是 (1,128,128,3)，说明和当前 SERLObsWrapper / 视觉输入基本匹配。")
    print("7. expected_grasp_penalty 是根据最终保存的 action[6] 重算的正确参考值。")
    print("8. 如果 stored penalty 落在 hold(0) 上，说明 pkl 里的 grasp_penalty 没有和最终 action[6] 同步。")
    print("9. 新版 record_demos.py 应该看到 stored 与 expected 完全一致。")


if __name__ == "__main__":
    main()
