


"""
inspect_demo_pkl_all.py

增强版 SERL / HIL-SERL demo pkl 检查脚本。

主要用途：
1. 检查单个 demo pkl 文件。
2. 一次性检查目录下所有 demo pkl 文件，并给出总计统计。
3. 验证当前单臂 RLPD 数据格式：
   - action shape == (7,)
   - state 最后一维 == 8
   - image keys / shapes 正常
   - reward=1 / done=True / mask=0 对齐
   - gripper action[6] 为 -1 / 0 / +1 三值
   - grasp_penalty 与最终保存的 action[6] 完全一致
4. 可选保存图像预览。
5. 可选导出 per-file summary CSV。

典型用法：

# 检查目录下所有 pkl，并输出每个文件 + 总计
python inspect_demo_pkl.py \
  --dir /home/eren/HIL-SERL/HIL-SERL-Project/examples/demo_data_single \
  --all \
  --image_keys head_rgb right_wrist_rgb

# 只检查最新 pkl，输出详细信息
python inspect_demo_pkl.py \
  --dir /home/eren/HIL-SERL/HIL-SERL-Project/examples/demo_data_single \
  --image_keys head_rgb right_wrist_rgb

# 指定某个 pkl
python inspect_demo_pkl.py \
  --path /path/to/demo.pkl \
  --image_keys head_rgb right_wrist_rgb

# 保存预览图
python inspect_demo_pkl.py \
  --dir /path/to/demo_data_single \
  --all \
  --save_preview \
  --preview_count 12 \
  --image_keys head_rgb right_wrist_rgb
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle as pkl
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 基础工具
# =============================================================================


def natural_key(path: str) -> Tuple[Any, ...]:
    """自然排序 key：transitions_2.pkl 会排在 transitions_10.pkl 前。"""
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    out: List[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return tuple(out)


def list_pkl_files(directory: str, pattern: str = "*.pkl", recursive: bool = False) -> List[str]:
    directory = os.path.abspath(directory)
    pat = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
    files = glob.glob(pat, recursive=recursive)
    return sorted(files, key=natural_key)


def find_latest_pkl(directory: str, pattern: str = "*.pkl") -> str:
    files = list_pkl_files(directory, pattern=pattern, recursive=False)
    if not files:
        raise FileNotFoundError(f"在 {directory} 下没有找到匹配 {pattern!r} 的 pkl 文件")
    return max(files, key=os.path.getmtime)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pkl.load(f)


def safe_np(x: Any) -> Optional[np.ndarray]:
    try:
        return np.asarray(x)
    except Exception:
        return None


def safe_float(x: Any, default: float = 0.0) -> float:
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def safe_bool(x: Any, default: bool = False) -> bool:
    arr = safe_np(x)
    if arr is None or arr.size == 0:
        return default
    try:
        return bool(arr.reshape(-1)[0])
    except Exception:
        return default


def is_number_array(x: Any) -> bool:
    arr = safe_np(x)
    return arr is not None and arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def summarize_array(name: str, arr: Any, max_items: int = 10) -> None:
    arr = np.asarray(arr)
    flat = arr.reshape(-1)
    preview = flat[:max_items]

    mn = mx = mean = std = None
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


# =============================================================================
# pickle 格式兼容
# =============================================================================


def _looks_like_single_transition(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "observations" not in obj or "actions" not in obj:
        return False

    # list-of-transition 中的单条 action 通常 flatten 后是 7/14 维。
    # dict-of-arrays 的 actions 通常是 (N, 7)。
    a = safe_np(obj.get("actions"))
    if a is None:
        return True

    if a.ndim == 0:
        return True
    if a.ndim == 1:
        return True
    if a.ndim >= 2 and a.shape[-1] in (7, 14) and a.shape[0] != 1:
        return False

    # 保守：如果 rewards/dones 是数组且长度与 actions[0] 一致，则是 batched dict。
    for k in ("rewards", "dones", "masks"):
        if k in obj:
            v = safe_np(obj[k])
            if v is not None and v.ndim >= 1 and a.ndim >= 2 and len(v) == a.shape[0]:
                return False

    return True


def _infer_batch_n(obj: Dict[str, Any]) -> Optional[int]:
    if "actions" in obj:
        a = safe_np(obj["actions"])
        if a is not None and a.ndim >= 2:
            return int(a.shape[0])

    for k in ("rewards", "dones", "masks", "grasp_penalty"):
        if k in obj:
            v = safe_np(obj[k])
            if v is not None and v.ndim >= 1:
                return int(v.shape[0])

    obs = obj.get("observations", {})
    if isinstance(obs, dict):
        for v in obs.values():
            arr = safe_np(v)
            if arr is not None and arr.ndim >= 2:
                return int(arr.shape[0])

    return None


def _index_value(x: Any, i: int, n: int) -> Any:
    """从 batched dict 中取第 i 个样本；不能确定时原样返回。"""
    if isinstance(x, dict):
        return {k: _index_value(v, i, n) for k, v in x.items()}

    if isinstance(x, (list, tuple)) and len(x) == n:
        return x[i]

    arr = safe_np(x)
    if arr is not None and arr.ndim >= 1 and arr.shape[0] == n:
        return arr[i]

    return x


def normalize_loaded_object(obj: Any) -> List[Dict[str, Any]]:
    """
    兼容常见保存格式：
    1. list[transition]
    2. tuple/list 嵌套
    3. dict 包 transitions / data / buffer / dataset
    4. 单个 transition dict
    5. dict-of-arrays batched dataset
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict) and "observations" in obj[0] and "actions" in obj[0]:
            return obj
        out: List[Dict[str, Any]] = []
        for item in obj:
            out.extend(normalize_loaded_object(item))
        return out

    if isinstance(obj, tuple):
        out: List[Dict[str, Any]] = []
        for item in obj:
            out.extend(normalize_loaded_object(item))
        return out

    if isinstance(obj, dict):
        # wrapper dict
        for key in ("transitions", "data", "buffer", "dataset"):
            if key in obj:
                return normalize_loaded_object(obj[key])

        # transition or dict-of-arrays
        if "observations" in obj and "actions" in obj:
            if _looks_like_single_transition(obj):
                return [obj]
            n = _infer_batch_n(obj)
            if n is None:
                return [obj]
            return [{k: _index_value(v, i, n) for k, v in obj.items()} for i in range(n)]

        # dict of transitions keyed by id
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            out: List[Dict[str, Any]] = []
            for v in vals:
                out.extend(normalize_loaded_object(v))
            return out

    return []


# =============================================================================
# observation / image 工具
# =============================================================================


def get_obs_dict(trans: Dict[str, Any], key: str = "observations") -> Dict[str, Any]:
    obs = trans.get(key, {})
    return obs if isinstance(obs, dict) else {}


def get_image_from_obs(obs: Dict[str, Any], image_key: str) -> Any:
    if image_key in obs:
        return obs[image_key]
    if "images" in obs and isinstance(obs["images"], dict) and image_key in obs["images"]:
        return obs["images"][image_key]
    return None


def unwrap_image(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]
    return arr


def recursive_array_summary(obj: Any, prefix: str = "", max_items: int = 120) -> List[Tuple[str, str, Any, Any, Any, Any, Any]]:
    results: List[Tuple[str, str, Any, Any, Any, Any, Any]] = []

    def visit(x: Any, path: str) -> None:
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


def print_recursive_summary(title: str, obj: Any, max_items: int = 120) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for path, dtype, shape, mn, mx, mean, size in recursive_array_summary(obj, max_items=max_items):
        print(f"{path:65s} dtype={str(dtype):10s} shape={shape} min={mn} max={mx} mean={mean}")


# =============================================================================
# action / gripper / reward 工具
# =============================================================================


def flatten_action(action: Any) -> Optional[np.ndarray]:
    arr = safe_np(action)
    if arr is None:
        return None
    try:
        arr = arr.astype(np.float32)
    except Exception:
        return None

    if arr.ndim == 0:
        return arr.reshape(1)

    while arr.ndim > 1:
        arr = arr[-1]

    return arr.reshape(-1)


def classify_gripper_from_action(action: Any) -> str:
    arr = flatten_action(action)
    if arr is None or arr.size == 0:
        return "missing"
    g = float(arr[-1])

    if g <= -0.5:
        return "close(-1)"
    if 0.5 <= g <= 1.5:
        return "open(+1)"
    if -0.5 < g < 0.5:
        return "hold(0)"

    # 硬件量程兜底。注意：如果走到这里，通常说明 action[6] 不是三值标签。
    if 0.0 <= g <= 30.0:
        return "hardware_close"
    if 70.0 <= g <= 100.0:
        return "hardware_open"
    return f"other({g:.4f})"


def get_reward(trans: Dict[str, Any]) -> float:
    return safe_float(trans.get("rewards", trans.get("reward", 0.0)))


def get_done(trans: Dict[str, Any]) -> bool:
    return safe_bool(trans.get("dones", trans.get("done", False)))


def get_mask(trans: Dict[str, Any]) -> float:
    return safe_float(trans.get("masks", trans.get("mask", 1.0)))


def expected_grasp_penalty_from_action(action: Any, penalty_value: float = -0.02) -> float:
    a = flatten_action(action)
    if a is None or a.shape[0] != 7:
        return 0.0
    g = float(a[6])
    if g <= -0.5 or g >= 0.5:
        return float(penalty_value)
    return 0.0


def extract_grasp_penalty(trans: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(trans, dict):
        return None, None
    if "grasp_penalty" in trans:
        return safe_float(trans["grasp_penalty"]), "top_level"
    infos = trans.get("infos", trans.get("info", None))
    if isinstance(infos, dict) and "grasp_penalty" in infos:
        return safe_float(infos["grasp_penalty"]), "infos"
    return None, None


def extract_all_grasp_penalty_fields(trans: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(trans, dict):
        return out
    if "grasp_penalty" in trans:
        out["top_level.grasp_penalty"] = safe_float(trans["grasp_penalty"])
    infos = trans.get("infos", trans.get("info", None))
    if isinstance(infos, dict):
        for key in ("grasp_penalty", "env_grasp_penalty_raw", "top_level_grasp_penalty_raw", "grasp_penalty_source"):
            if key in infos:
                out[f"infos.{key}"] = infos[key]
    return out


# =============================================================================
# 统计数据结构
# =============================================================================


@dataclass
class FileStats:
    path: str
    transitions: int = 0
    reward_sum: float = 0.0
    reward_pos: int = 0
    done_count: int = 0
    mask0_count: int = 0
    tail_len: int = 0
    episode_lengths: List[int] = field(default_factory=list)

    action_dim_counter: Counter = field(default_factory=Counter)
    gripper_counter: Counter = field(default_factory=Counter)
    static_actions: int = 0
    arm_oob_count: int = 0
    action_global_min: Optional[float] = None
    action_global_max: Optional[float] = None
    action_global_absmax: Optional[float] = None
    action_mean: Optional[np.ndarray] = None
    action_std: Optional[np.ndarray] = None
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None
    action_absmax: Optional[np.ndarray] = None

    obs_keys_counter: Counter = field(default_factory=Counter)
    next_obs_keys_counter: Counter = field(default_factory=Counter)
    image_shape_counter: Counter = field(default_factory=Counter)
    image_missing_counter: Counter = field(default_factory=Counter)
    state_dim_counter: Counter = field(default_factory=Counter)

    gp_missing: int = 0
    gp_top_level_count: int = 0
    gp_infos_count: int = 0
    gp_both_count: int = 0
    gp_top_info_mismatch: int = 0
    gp_nonzero_stored: int = 0
    gp_nonzero_expected: int = 0
    gp_mismatch: int = 0
    gp_sum_stored: float = 0.0
    gp_sum_expected: float = 0.0
    gp_stored_dist: Counter = field(default_factory=Counter)
    gp_expected_dist: Counter = field(default_factory=Counter)
    gp_nonzero_gripper_counter: Counter = field(default_factory=Counter)
    gp_mismatch_gripper_counter: Counter = field(default_factory=Counter)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def episode_mean_len(self) -> Optional[float]:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else None

    @property
    def episode_min_len(self) -> Optional[int]:
        return int(min(self.episode_lengths)) if self.episode_lengths else None

    @property
    def episode_max_len(self) -> Optional[int]:
        return int(max(self.episode_lengths)) if self.episode_lengths else None


@dataclass
class AggregateStats:
    files: int = 0
    transitions: int = 0
    reward_sum: float = 0.0
    reward_pos: int = 0
    done_count: int = 0
    mask0_count: int = 0
    episode_lengths: List[int] = field(default_factory=list)

    action_dim_counter: Counter = field(default_factory=Counter)
    gripper_counter: Counter = field(default_factory=Counter)
    static_actions: int = 0
    arm_oob_count: int = 0
    obs_keys_counter: Counter = field(default_factory=Counter)
    image_shape_counter: Counter = field(default_factory=Counter)
    image_missing_counter: Counter = field(default_factory=Counter)
    state_dim_counter: Counter = field(default_factory=Counter)

    gp_missing: int = 0
    gp_top_level_count: int = 0
    gp_infos_count: int = 0
    gp_both_count: int = 0
    gp_top_info_mismatch: int = 0
    gp_nonzero_stored: int = 0
    gp_nonzero_expected: int = 0
    gp_mismatch: int = 0
    gp_sum_stored: float = 0.0
    gp_sum_expected: float = 0.0
    gp_stored_dist: Counter = field(default_factory=Counter)
    gp_expected_dist: Counter = field(default_factory=Counter)
    gp_nonzero_gripper_counter: Counter = field(default_factory=Counter)
    gp_mismatch_gripper_counter: Counter = field(default_factory=Counter)

    def add(self, s: FileStats) -> None:
        self.files += 1
        self.transitions += s.transitions
        self.reward_sum += s.reward_sum
        self.reward_pos += s.reward_pos
        self.done_count += s.done_count
        self.mask0_count += s.mask0_count
        self.episode_lengths.extend(s.episode_lengths)
        self.action_dim_counter.update(s.action_dim_counter)
        self.gripper_counter.update(s.gripper_counter)
        self.static_actions += s.static_actions
        self.arm_oob_count += s.arm_oob_count
        self.obs_keys_counter.update(s.obs_keys_counter)
        self.image_shape_counter.update(s.image_shape_counter)
        self.image_missing_counter.update(s.image_missing_counter)
        self.state_dim_counter.update(s.state_dim_counter)
        self.gp_missing += s.gp_missing
        self.gp_top_level_count += s.gp_top_level_count
        self.gp_infos_count += s.gp_infos_count
        self.gp_both_count += s.gp_both_count
        self.gp_top_info_mismatch += s.gp_top_info_mismatch
        self.gp_nonzero_stored += s.gp_nonzero_stored
        self.gp_nonzero_expected += s.gp_nonzero_expected
        self.gp_mismatch += s.gp_mismatch
        self.gp_sum_stored += s.gp_sum_stored
        self.gp_sum_expected += s.gp_sum_expected
        self.gp_stored_dist.update(s.gp_stored_dist)
        self.gp_expected_dist.update(s.gp_expected_dist)
        self.gp_nonzero_gripper_counter.update(s.gp_nonzero_gripper_counter)
        self.gp_mismatch_gripper_counter.update(s.gp_mismatch_gripper_counter)


# =============================================================================
# 核心分析
# =============================================================================


def _counter_key_for_obs_keys(obs: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(list(obs.keys())))


def analyze_transitions(
    transitions: List[Dict[str, Any]],
    path: str,
    image_keys: Sequence[str],
    expected_grasp_penalty: float = -0.02,
    mismatch_tol: float = 1e-6,
) -> FileStats:
    s = FileStats(path=path, transitions=len(transitions))

    actions: List[np.ndarray] = []
    cur_len = 0

    for trans in transitions:
        r = get_reward(trans)
        d = get_done(trans)
        m = get_mask(trans)
        s.reward_sum += r
        if r > 0:
            s.reward_pos += 1
        if abs(m) < 1e-8:
            s.mask0_count += 1

        # episode length
        cur_len += 1
        if d:
            s.done_count += 1
            s.episode_lengths.append(cur_len)
            cur_len = 0

        # action
        a = flatten_action(trans.get("actions"))
        if a is not None:
            actions.append(a)
            s.action_dim_counter[int(a.shape[0])] += 1
            g_desc = classify_gripper_from_action(a)
            s.gripper_counter[g_desc] += 1

            if np.allclose(a, 0.0, atol=1e-8):
                s.static_actions += 1
            if a.shape[0] >= 6 and float(np.max(np.abs(a[:6]))) > 1.0001:
                s.arm_oob_count += 1
        else:
            g_desc = "missing"
            s.gripper_counter[g_desc] += 1

        # observations
        obs = get_obs_dict(trans, "observations")
        next_obs = get_obs_dict(trans, "next_observations")
        s.obs_keys_counter[_counter_key_for_obs_keys(obs)] += 1
        if next_obs:
            s.next_obs_keys_counter[_counter_key_for_obs_keys(next_obs)] += 1

        if "state" in obs:
            st = safe_np(obs["state"])
            if st is not None and st.ndim >= 1:
                s.state_dim_counter[int(st.shape[-1])] += 1

        for k in image_keys:
            img = get_image_from_obs(obs, k)
            if img is None:
                s.image_missing_counter[k] += 1
            else:
                s.image_shape_counter[(k, tuple(np.asarray(img).shape))] += 1

        # grasp penalty
        fields = extract_all_grasp_penalty_fields(trans)
        has_top = "top_level.grasp_penalty" in fields
        has_info = "infos.grasp_penalty" in fields
        if has_top:
            s.gp_top_level_count += 1
        if has_info:
            s.gp_infos_count += 1
        if has_top and has_info:
            s.gp_both_count += 1
            top_v = safe_float(fields["top_level.grasp_penalty"])
            info_v = safe_float(fields["infos.grasp_penalty"])
            if abs(top_v - info_v) > mismatch_tol:
                s.gp_top_info_mismatch += 1

        stored, source = extract_grasp_penalty(trans)
        expected = expected_grasp_penalty_from_action(trans.get("actions"), penalty_value=expected_grasp_penalty)
        s.gp_expected_dist[round(float(expected), 8)] += 1
        s.gp_sum_expected += float(expected)
        if abs(float(expected)) > 1e-8:
            s.gp_nonzero_expected += 1

        if stored is None:
            s.gp_missing += 1
        else:
            stored_f = float(stored)
            s.gp_stored_dist[round(stored_f, 8)] += 1
            s.gp_sum_stored += stored_f
            if abs(stored_f) > 1e-8:
                s.gp_nonzero_stored += 1
                s.gp_nonzero_gripper_counter[g_desc] += 1
            if abs(stored_f - float(expected)) > mismatch_tol:
                s.gp_mismatch += 1
                s.gp_mismatch_gripper_counter[g_desc] += 1

    s.tail_len = cur_len

    # action detailed stats，仅当所有 action 维度一致时统计。
    if actions:
        dims = {tuple(a.shape) for a in actions}
        if len(dims) == 1:
            arr = np.stack(actions, axis=0)
            s.action_global_min = float(np.min(arr))
            s.action_global_max = float(np.max(arr))
            s.action_global_absmax = float(np.max(np.abs(arr)))
            s.action_min = np.min(arr, axis=0)
            s.action_max = np.max(arr, axis=0)
            s.action_mean = np.mean(arr, axis=0)
            s.action_std = np.std(arr, axis=0)
            s.action_absmax = np.max(np.abs(arr), axis=0)

    return s


# =============================================================================
# 打印函数
# =============================================================================


def print_file_compact(s: FileStats) -> None:
    print("\n" + "-" * 100)
    print(s.name)
    print("-" * 100)
    print(f"transitions     : {s.transitions}")
    print(f"episodes(done)  : {s.done_count}")
    print(f"reward_sum      : {s.reward_sum:.6f}")
    print(f"reward>0        : {s.reward_pos}")
    print(f"mask=0          : {s.mask0_count}")
    if s.episode_lengths:
        print(f"episode length  : min={s.episode_min_len}, max={s.episode_max_len}, mean={s.episode_mean_len:.2f}, tail={s.tail_len}")
    else:
        print(f"episode length  : None, tail={s.tail_len}")
    print(f"action dims     : {dict(s.action_dim_counter)}")
    print(f"gripper dist    : {dict(s.gripper_counter)}")
    print(f"static actions  : {s.static_actions}")
    print(f"arm out of range: {s.arm_oob_count}")
    print(f"gp missing      : {s.gp_missing}")
    print(f"gp mismatch     : {s.gp_mismatch}")
    print(f"gp nonzero      : stored={s.gp_nonzero_stored}, expected={s.gp_nonzero_expected}")
    print(f"obs keys        : {dict(s.obs_keys_counter)}")
    print(f"state dim       : {dict(s.state_dim_counter)}")
    print(f"image shapes    : {dict(s.image_shape_counter)}")
    missing = {k: v for k, v in s.image_missing_counter.items() if v > 0}
    if missing:
        print(f"image missing   : {missing}")

    flags = []
    if not (s.done_count == s.reward_pos == s.mask0_count):
        flags.append("reward/done/mask 不一致")
    if s.arm_oob_count:
        flags.append("arm action 越界")
    if s.gp_missing:
        flags.append("grasp_penalty 缺失")
    if s.gp_mismatch:
        flags.append("grasp_penalty 与 action[6] 不一致")
    bad_gripper = {k: v for k, v in s.gripper_counter.items() if k not in {"hold(0)", "close(-1)", "open(+1)"}}
    if bad_gripper:
        flags.append(f"gripper 非三值: {bad_gripper}")

    if flags:
        print("CHECK           : ⚠️ " + "; ".join(flags))
    else:
        print("CHECK           : ✅ OK")


def print_aggregate(agg: AggregateStats) -> None:
    print("\n" + "=" * 100)
    print("TOTAL / ALL PKL SUMMARY")
    print("=" * 100)
    print(f"files           : {agg.files}")
    print(f"transitions     : {agg.transitions}")
    print(f"episodes(done)  : {agg.done_count}")
    print(f"reward_sum      : {agg.reward_sum:.6f}")
    print(f"reward>0        : {agg.reward_pos}")
    print(f"mask=0          : {agg.mask0_count}")
    if agg.episode_lengths:
        print(
            f"episode length  : min={min(agg.episode_lengths)}, "
            f"max={max(agg.episode_lengths)}, mean={np.mean(agg.episode_lengths):.2f}"
        )
    print(f"action dims     : {dict(agg.action_dim_counter)}")
    print(f"gripper dist    : {dict(agg.gripper_counter)}")
    print(f"static actions  : {agg.static_actions}")
    print(f"arm out of range: {agg.arm_oob_count}")
    print(f"obs keys        : {dict(agg.obs_keys_counter)}")
    print(f"state dim       : {dict(agg.state_dim_counter)}")
    print(f"image shapes    : {dict(agg.image_shape_counter)}")
    missing = {k: v for k, v in agg.image_missing_counter.items() if v > 0}
    if missing:
        print(f"image missing   : {missing}")
    print("\ngrasp_penalty:")
    print(f"  top_level count       : {agg.gp_top_level_count}")
    print(f"  infos count           : {agg.gp_infos_count}")
    print(f"  both count            : {agg.gp_both_count}")
    print(f"  top/info mismatch     : {agg.gp_top_info_mismatch}")
    print(f"  missing               : {agg.gp_missing}")
    print(f"  stored dist           : {dict(agg.gp_stored_dist)}")
    print(f"  expected dist         : {dict(agg.gp_expected_dist)}")
    print(f"  nonzero stored        : {agg.gp_nonzero_stored}")
    print(f"  nonzero expected      : {agg.gp_nonzero_expected}")
    print(f"  stored sum            : {agg.gp_sum_stored:.6f}")
    print(f"  expected sum          : {agg.gp_sum_expected:.6f}")
    print(f"  stored/expected mismatch: {agg.gp_mismatch}")
    print(f"  nonzero gp gripper    : {dict(agg.gp_nonzero_gripper_counter)}")
    if agg.gp_mismatch_gripper_counter:
        print(f"  mismatch gripper      : {dict(agg.gp_mismatch_gripper_counter)}")

    print("\n判断：")
    ok = True
    if not (agg.done_count == agg.reward_pos == agg.mask0_count):
        ok = False
        print("❌ reward>0、done=True、mask=0 数量不一致。")
    else:
        print("✅ reward>0、done=True、mask=0 数量一致。")

    if agg.arm_oob_count == 0:
        print("✅ 前 6 维动作没有越界。")
    else:
        ok = False
        print(f"❌ 前 6 维动作越界数量: {agg.arm_oob_count}")

    bad_gripper = {k: v for k, v in agg.gripper_counter.items() if k not in {"hold(0)", "close(-1)", "open(+1)"}}
    if not bad_gripper:
        print("✅ gripper 是标准三值标签：-1 / 0 / +1。")
    else:
        ok = False
        print(f"❌ gripper 存在非三值标签: {bad_gripper}")

    if agg.gp_missing == 0 and agg.gp_mismatch == 0:
        print("✅ grasp_penalty 无缺失，且与 action[6] 重算结果完全一致。")
    else:
        ok = False
        print(f"❌ grasp_penalty 问题：missing={agg.gp_missing}, mismatch={agg.gp_mismatch}")

    expected_images = {"head_rgb", "right_wrist_rgb"}
    image_keys_present = {k for (k, shape), cnt in agg.image_shape_counter.items() if cnt > 0}
    if expected_images.issubset(image_keys_present):
        print("✅ 当前单臂常用图像 key head_rgb / right_wrist_rgb 存在。")
    else:
        print(f"⚠️ 当前图像 key 不完整，present={image_keys_present}")

    if ok:
        print("\nCHECK: ✅ OK: 这些 demo pkl 可以一起用于当前单臂 RLPD 训练。")
    else:
        print("\nCHECK: ⚠️ 请先处理上面的异常，再用于训练。")


def print_action_detail(s: FileStats) -> None:
    if s.action_min is None or s.action_max is None or s.action_mean is None or s.action_std is None or s.action_absmax is None:
        print("没有可打印的 action 详细统计。")
        return

    print("\n" + "=" * 100)
    print(f"actions 详细统计: {s.name}")
    print("=" * 100)
    dim = len(s.action_min)
    print(f"global min/max/absmax: {s.action_global_min}, {s.action_global_max}, {s.action_global_absmax}")
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"] if dim == 7 else [f"a[{i}]" for i in range(dim)]
    for i in range(dim):
        print(
            f"{i:02d} {names[i]:10s} "
            f"min={s.action_min[i]: .6f}, max={s.action_max[i]: .6f}, "
            f"mean={s.action_mean[i]: .6f}, std={s.action_std[i]: .6f}, "
            f"absmax={s.action_absmax[i]: .6f}"
        )


def print_single_file_detail(
    path: str,
    transitions: List[Dict[str, Any]],
    stats: FileStats,
    image_keys: Sequence[str],
    sample_count: int = 5,
    expected_grasp_penalty: float = -0.02,
) -> None:
    print("\n" + "=" * 100)
    print("Demo 文件详细检查")
    print("=" * 100)
    print(f"pkl path        : {path}")
    print(f"transition 总数 : {len(transitions)}")

    if not transitions:
        return

    first = transitions[0]
    obs = get_obs_dict(first, "observations")
    next_obs = get_obs_dict(first, "next_observations")

    print("\n顶层 keys:")
    print(list(first.keys()))
    print("\nobservations 顶层 keys:")
    print(list(obs.keys()))
    print("\nnext_observations 顶层 keys:")
    print(list(next_obs.keys()) if next_obs else "不存在或不是 dict")

    # reward/done 样本
    reward_pos_indices = [i for i, t in enumerate(transitions) if get_reward(t) > 0]
    done_indices = [i for i, t in enumerate(transitions) if get_done(t)]

    print("\n" + "=" * 100)
    print("reward / done / mask 样本")
    print("=" * 100)
    print(f"reward_sum         : {stats.reward_sum:.6f}")
    print(f"reward > 0 数量     : {stats.reward_pos}")
    print(f"done=True 数量      : {stats.done_count}")
    print(f"mask=0 数量         : {stats.mask0_count}")
    print(f"done indices 前 30 个: {done_indices[:30]}")

    if reward_pos_indices:
        print("\n前若干 reward>0 transition:")
        for idx in reward_pos_indices[:20]:
            trans = transitions[idx]
            action = flatten_action(trans.get("actions"))
            stored, _ = extract_grasp_penalty(trans)
            expected = expected_grasp_penalty_from_action(trans.get("actions"), expected_grasp_penalty)
            print(
                f"  idx={idx}, reward={get_reward(trans)}, done={get_done(trans)}, mask={get_mask(trans)}, "
                f"gripper={classify_gripper_from_action(trans.get('actions'))}, "
                f"stored_gp={stored}, expected_gp={expected}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}"
            )

    print_action_detail(stats)

    print("\n前几条 actions 抽样:")
    for i in range(min(sample_count, len(transitions))):
        a = flatten_action(transitions[i].get("actions"))
        print(f"  [{i}] shape={a.shape if a is not None else None}, action={a}")

    # state
    print("\n" + "=" * 100)
    print("state / proprio 检查")
    print("=" * 100)
    if "state" in obs:
        summarize_array("first observations/state", obs["state"])
        st_arr = safe_np(obs["state"])
        if st_arr is not None and st_arr.ndim > 0:
            last_dim = st_arr.shape[-1]
            if last_dim == 8:
                print("✅ state 最后一维 = 8，很像单臂 proprio：7维 ee pose + 1维 gripper")
            elif last_dim == 16:
                print("⚠️ state 最后一维 = 16，很像双臂 proprio")
            else:
                print(f"ℹ️ state 最后一维 = {last_dim}，请结合 config 判断。")

        state_samples = []
        for t in transitions:
            o = get_obs_dict(t, "observations")
            if "state" in o:
                state_samples.append(np.asarray(o["state"]).reshape(-1))
        if state_samples:
            st = np.stack(state_samples, axis=0)
            print("\n全文件 state 统计:")
            for i in range(st.shape[-1]):
                print(
                    f"  state[{i:02d}] min={np.min(st[:, i]): .6f}, max={np.max(st[:, i]): .6f}, "
                    f"mean={np.mean(st[:, i]): .6f}, std={np.std(st[:, i]): .6f}"
                )
    else:
        print("observations 里没有 state。")

    # images
    print("\n" + "=" * 100)
    print("图像 key / shape 检查")
    print("=" * 100)
    for k in image_keys:
        img = get_image_from_obs(obs, k)
        if img is None:
            print(f"{k}: ❌ 不存在")
        else:
            summarize_array(k, img, max_items=3)
    print("\n全文件图像 shape 分布:")
    for (k, shape), count in sorted(stats.image_shape_counter.items()):
        print(f"  {k}: shape={shape}, count={count}")
    missing = {k: v for k, v in stats.image_missing_counter.items() if v > 0}
    if missing:
        print(f"  missing: {missing}")

    print_recursive_summary("第一个 transition 的 observations 递归结构", obs)
    if next_obs:
        print_recursive_summary("第一个 transition 的 next_observations 递归结构", next_obs)

    # infos
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

    # grasp penalty detailed samples
    print("\n" + "=" * 100)
    print("grasp_penalty 检查")
    print("=" * 100)
    print(f"expected_penalty_value          : {expected_grasp_penalty}")
    print(f"transition 总数                 : {stats.transitions}")
    print(f"grasp_penalty 缺失数量           : {stats.gp_missing}")
    print(f"top_level grasp_penalty 数量     : {stats.gp_top_level_count}")
    print(f"infos grasp_penalty 数量         : {stats.gp_infos_count}")
    print(f"top_level + infos 同时存在数量   : {stats.gp_both_count}")
    print(f"top_level 与 infos 不一致数量    : {stats.gp_top_info_mismatch}")
    print(f"stored grasp_penalty 取值分布     : {dict(stats.gp_stored_dist)}")
    print(f"expected grasp_penalty 取值分布   : {dict(stats.gp_expected_dist)}")
    print(f"stored grasp_penalty 非零数量     : {stats.gp_nonzero_stored}")
    print(f"expected grasp_penalty 非零数量   : {stats.gp_nonzero_expected}")
    print(f"stored vs expected mismatch 数量  : {stats.gp_mismatch}")
    print(f"非零 stored penalty 对应 gripper  : {dict(stats.gp_nonzero_gripper_counter)}")
    print(f"mismatch 对应 gripper 分布         : {dict(stats.gp_mismatch_gripper_counter)}")

    nonzero_indices = []
    mismatch_indices = []
    for i, t in enumerate(transitions):
        stored, _ = extract_grasp_penalty(t)
        expected = expected_grasp_penalty_from_action(t.get("actions"), expected_grasp_penalty)
        if stored is not None and abs(float(stored)) > 1e-8:
            nonzero_indices.append(i)
        if stored is None or abs(float(stored) - expected) > 1e-6:
            mismatch_indices.append(i)

    if nonzero_indices:
        print("\n前若干非零 stored grasp_penalty transition:")
        for idx in nonzero_indices[:30]:
            t = transitions[idx]
            action = flatten_action(t.get("actions"))
            stored, source = extract_grasp_penalty(t)
            expected = expected_grasp_penalty_from_action(t.get("actions"), expected_grasp_penalty)
            fields = extract_all_grasp_penalty_fields(t)
            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={classify_gripper_from_action(t.get('actions'))}, "
                f"reward={get_reward(t)}, done={get_done(t)}, mask={get_mask(t)}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}, fields={fields}"
            )

    if mismatch_indices:
        print("\n前若干 grasp_penalty mismatch transition:")
        for idx in mismatch_indices[:30]:
            t = transitions[idx]
            action = flatten_action(t.get("actions"))
            stored, source = extract_grasp_penalty(t)
            expected = expected_grasp_penalty_from_action(t.get("actions"), expected_grasp_penalty)
            print(
                f"  idx={idx}, stored={stored}, expected={expected}, source={source}, "
                f"gripper={classify_gripper_from_action(t.get('actions'))}, "
                f"action={np.round(action, 4).tolist() if action is not None else None}"
            )


# =============================================================================
# 预览图
# =============================================================================


def save_image_previews(
    transitions: List[Dict[str, Any]],
    image_keys: Sequence[str],
    out_dir: str,
    indices: Optional[List[int]] = None,
    max_count: int = 10,
    expected_grasp_penalty: float = -0.02,
    prefix: str = "preview",
) -> None:
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
        nonzero_grip = [i for i, t in enumerate(transitions) if classify_gripper_from_action(t.get("actions")) != "hold(0)"]
        candidate.extend(done_indices[:max_count])
        candidate.extend(reward_indices[:max_count])
        candidate.extend(nonzero_grip[:max_count])
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
            if arr.shape[-1] == 3:
                save_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                save_arr = arr
            imgs.append(save_arr)

        if not imgs:
            continue

        h = min(im.shape[0] for im in imgs)
        resized = []
        for im in imgs:
            scale = h / im.shape[0]
            w = int(im.shape[1] * scale)
            resized.append(cv2.resize(im, (w, h)))
        canvas = np.concatenate(resized, axis=1)

        stored_penalty, _ = extract_grasp_penalty(trans)
        expected_penalty = expected_grasp_penalty_from_action(trans.get("actions"), penalty_value=expected_grasp_penalty)
        stored_text = "None" if stored_penalty is None else f"{stored_penalty:.4f}"
        text = (
            f"idx={idx} reward={get_reward(trans)} done={get_done(trans)} mask={get_mask(trans)} "
            f"gripper={classify_gripper_from_action(trans.get('actions'))} "
            f"stored_gp={stored_text} expected_gp={expected_penalty:.4f}"
        )

        text_h = 40
        board = np.zeros((canvas.shape[0] + text_h, canvas.shape[1], 3), dtype=np.uint8)
        board[: canvas.shape[0]] = canvas
        board[canvas.shape[0] :] = 255
        cv2.putText(board, text, (10, canvas.shape[0] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        filename = os.path.join(out_dir, f"{prefix}_idx_{idx:06d}.jpg")
        cv2.imwrite(filename, board)
        print(f"保存: {filename}")


# =============================================================================
# CSV 导出
# =============================================================================


def write_summary_csv(path: str, stats_list: Sequence[FileStats]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "file",
        "transitions",
        "episodes_done",
        "reward_sum",
        "reward_pos",
        "mask0",
        "episode_min_len",
        "episode_max_len",
        "episode_mean_len",
        "tail_len",
        "action_dims",
        "gripper_dist",
        "static_actions",
        "arm_oob_count",
        "gp_missing",
        "gp_mismatch",
        "gp_nonzero_stored",
        "gp_nonzero_expected",
        "gp_sum_stored",
        "gp_sum_expected",
        "obs_keys",
        "state_dim",
        "image_shapes",
        "image_missing",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats_list:
            writer.writerow(
                {
                    "file": s.name,
                    "transitions": s.transitions,
                    "episodes_done": s.done_count,
                    "reward_sum": f"{s.reward_sum:.6f}",
                    "reward_pos": s.reward_pos,
                    "mask0": s.mask0_count,
                    "episode_min_len": s.episode_min_len,
                    "episode_max_len": s.episode_max_len,
                    "episode_mean_len": f"{s.episode_mean_len:.6f}" if s.episode_mean_len is not None else "",
                    "tail_len": s.tail_len,
                    "action_dims": dict(s.action_dim_counter),
                    "gripper_dist": dict(s.gripper_counter),
                    "static_actions": s.static_actions,
                    "arm_oob_count": s.arm_oob_count,
                    "gp_missing": s.gp_missing,
                    "gp_mismatch": s.gp_mismatch,
                    "gp_nonzero_stored": s.gp_nonzero_stored,
                    "gp_nonzero_expected": s.gp_nonzero_expected,
                    "gp_sum_stored": f"{s.gp_sum_stored:.6f}",
                    "gp_sum_expected": f"{s.gp_sum_expected:.6f}",
                    "obs_keys": dict(s.obs_keys_counter),
                    "state_dim": dict(s.state_dim_counter),
                    "image_shapes": dict(s.image_shape_counter),
                    "image_missing": dict(s.image_missing_counter),
                }
            )
    print(f"\n已保存 summary CSV: {path}")


# =============================================================================
# 主流程
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 / 分析录制的 SERL demos pkl 数据，支持单文件和目录聚合。")
    parser.add_argument("--path", type=str, default=None, help="指定某个 demo pkl 文件。")
    parser.add_argument("--dir", type=str, default=BASE_DIR, help="demo pkl 所在目录。")
    parser.add_argument("--all", action="store_true", help="分析 --dir 下所有匹配的 pkl 文件。")
    parser.add_argument("--pattern", type=str, default="*.pkl", help="--all 时使用的文件匹配模式。默认 *.pkl。")
    parser.add_argument("--recursive", action="store_true", help="--all 时递归搜索子目录。")
    parser.add_argument(
        "--image_keys",
        type=str,
        nargs="+",
        default=["head_rgb", "right_wrist_rgb"],
        help="需要检查的图像 key。当前单臂通常为 head_rgb right_wrist_rgb。",
    )
    parser.add_argument("--sample_count", type=int, default=5, help="详细模式打印前几条 action 抽样。")
    parser.add_argument("--expected_grasp_penalty", type=float, default=-0.02, help="根据 action[6] 重算 expected grasp_penalty 使用的 penalty 值。")
    parser.add_argument("--compact", action="store_true", help="单文件也只打印 compact summary，不打印详细结构。")
    parser.add_argument("--detail_each", action="store_true", help="--all 时对每个文件都打印详细信息。默认只打印 compact summary。")
    parser.add_argument("--save_preview", action="store_true", help="是否保存若干帧图像预览。")
    parser.add_argument("--preview_dir", type=str, default=None, help="图像预览保存目录。默认放在 pkl 同目录 demo_preview/。")
    parser.add_argument("--preview_count", type=int, default=12, help="最多保存多少张预览图。")
    parser.add_argument("--summary_csv", type=str, default=None, help="可选：导出每个 pkl 的 summary CSV。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = os.path.abspath(args.dir)

    if args.path:
        paths = [os.path.abspath(args.path)]
    elif args.all:
        paths = list_pkl_files(directory, pattern=args.pattern, recursive=args.recursive)
        if not paths:
            raise FileNotFoundError(f"在 {directory} 下没有找到匹配 {args.pattern!r} 的 pkl 文件")
    else:
        paths = [find_latest_pkl(directory, pattern=args.pattern)]

    print("=" * 100)
    print("SERL Demo PKL Inspector / ALL-IN-ONE")
    print("=" * 100)
    print(f"demo dir                : {directory}")
    print(f"mode                    : {'all' if len(paths) > 1 or args.all else 'single'}")
    print(f"file count              : {len(paths)}")
    print(f"image_keys              : {args.image_keys}")
    print(f"expected_grasp_penalty  : {args.expected_grasp_penalty}")
    print("files:")
    for p in paths:
        print(f"  - {p}")

    all_stats: List[FileStats] = []
    agg = AggregateStats()

    # 如果要保存预览，all 模式下把不同文件放到不同子目录，避免重名。
    for path in paths:
        obj = load_pickle(path)
        transitions = normalize_loaded_object(obj)
        stats = analyze_transitions(
            transitions,
            path=path,
            image_keys=args.image_keys,
            expected_grasp_penalty=args.expected_grasp_penalty,
        )
        all_stats.append(stats)
        agg.add(stats)

        if len(paths) == 1 and not args.compact:
            print_single_file_detail(
                path,
                transitions,
                stats,
                image_keys=args.image_keys,
                sample_count=args.sample_count,
                expected_grasp_penalty=args.expected_grasp_penalty,
            )
        else:
            print_file_compact(stats)
            if args.detail_each:
                print_single_file_detail(
                    path,
                    transitions,
                    stats,
                    image_keys=args.image_keys,
                    sample_count=args.sample_count,
                    expected_grasp_penalty=args.expected_grasp_penalty,
                )

        if args.save_preview:
            if args.preview_dir:
                base_preview = os.path.abspath(args.preview_dir)
            else:
                base_preview = os.path.join(os.path.dirname(path), "demo_preview")
            subdir = os.path.join(base_preview, os.path.splitext(os.path.basename(path))[0]) if len(paths) > 1 else base_preview
            save_image_previews(
                transitions,
                image_keys=args.image_keys,
                out_dir=subdir,
                indices=None,
                max_count=args.preview_count,
                expected_grasp_penalty=args.expected_grasp_penalty,
                prefix=os.path.splitext(os.path.basename(path))[0],
            )

    print_aggregate(agg)

    if args.summary_csv:
        write_summary_csv(args.summary_csv, all_stats)

    print("\n" + "=" * 100)
    print("结论提示")
    print("=" * 100)
    print("1. 当前单臂任务通常应看到 action shape=(7)，state 最后一维=8。")
    print("2. 当前 RLPD 视觉输入通常需要 head_rgb 和 right_wrist_rgb。left_wrist_rgb 缺失对单臂右臂任务通常不是问题。")
    print("3. reward=1、done=True、mask=0 数量一致，说明成功终止数据正常。")
    print("4. action 前 6 维应在 [-1, 1]，否则说明动作归一化/clip 有问题。")
    print("5. action[6] 应为 -1/0/+1 三值。不要混入硬件夹爪值 10/80。")
    print("6. expected_grasp_penalty 根据最终 action[6] 重算；stored 与 expected 一致才说明 penalty 同步正确。")
    print("7. --all 模式下 TOTAL 的 episodes(done) 就是目录中所有 pkl 的 demo 总数。")


if __name__ == "__main__":
    main()
