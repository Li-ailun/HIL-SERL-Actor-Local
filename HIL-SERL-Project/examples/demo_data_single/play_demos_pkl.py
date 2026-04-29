# 使用方法
# 1. 自动读取当前目录最新 pkl
# python play_demos_pkl.py 
# 2. 手动指定 pkl
# python play_demo_merged_full.py --pkl /your/path/demo_xxx.pkl
# 3. 指定播放速度
# python play_demo_merged_full.py --fps 10
# 4. 指定图像 key 顺序
# python play_demo_merged_full.py --image_keys left_wrist_rgb head_rgb right_wrist_rgb

import os
import glob
import math
import pickle as pkl
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 默认配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = BASE_DIR

# 你常用的优先图像顺序；如果实际数据里没有，就自动补其它图像 key
PREFERRED_IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]

# 播放器默认参数
DEFAULT_FPS = 15.0
DEFAULT_MAX_CANVAS_W = 1600
DEFAULT_MAX_IMAGE_AREA_H = 720
DEFAULT_BG_COLOR = (18, 18, 18)
DEFAULT_TEXT_COLOR = (235, 235, 235)

# 最新夹爪标签定义
GRIPPER_CLOSE_LABEL = -1.0
GRIPPER_HOLD_LABEL = 0.0
GRIPPER_OPEN_LABEL = +1.0

GRIPPER_CLOSE_HW = 10.0
GRIPPER_OPEN_HW = 80.0

# 用反馈量程推断“稳定状态”时的阈值
GRIPPER_FEEDBACK_CLOSE_MAX = 30.0
GRIPPER_FEEDBACK_OPEN_MIN = 70.0


# ============================================================
# 基础工具
# ============================================================
def safe_float(x, default=None):
    try:
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[0])
    except Exception:
        return default


def safe_bool(x, default=False):
    try:
        return bool(x)
    except Exception:
        return default


def safe_str(x):
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


def flatten_scalar_info(obj, prefix="") -> List[str]:
    """
    递归展开 infos，尽量只保留标量/短向量，避免太乱。
    """
    lines = []

    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda z: str(z)):
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(flatten_scalar_info(obj[k], new_prefix))
        return lines

    if isinstance(obj, (list, tuple)):
        if len(obj) <= 8 and all(
            isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))
            for v in obj
        ):
            lines.append(f"{prefix} = {list(obj)}")
        else:
            lines.append(f"{prefix} = <list len={len(obj)}>")
        return lines

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            lines.append(f"{prefix} = {obj.item()}")
        elif obj.size <= 8:
            lines.append(f"{prefix} = {np.asarray(obj).reshape(-1).tolist()}")
        else:
            lines.append(f"{prefix} = <ndarray shape={obj.shape}>")
        return lines

    if isinstance(obj, (int, float, bool, np.integer, np.floating, np.bool_)):
        lines.append(f"{prefix} = {obj}")
        return lines

    # 忽略过大的复杂对象
    text = safe_str(obj)
    if len(text) > 80:
        text = text[:77] + "..."
    lines.append(f"{prefix} = {text}")
    return lines


def format_vec(arr, precision=4, signed=True):
    if arr is None:
        return "<None>"
    arr = np.asarray(arr).reshape(-1)
    if signed:
        fmt = f"{{:+.{precision}f}}"
    else:
        fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(float(x)) for x in arr) + "]"


def wrap_lines(lines: List[str], max_len=90) -> List[str]:
    """
    简单换行，避免一行过长。
    """
    out = []
    for line in lines:
        if len(line) <= max_len:
            out.append(line)
        else:
            s = line
            while len(s) > max_len:
                out.append(s[:max_len])
                s = "  " + s[max_len:]
            if s:
                out.append(s)
    return out


# ============================================================
# PKL 读取 / demo 切分
# ============================================================
def find_latest_demo_pkl(data_dir: str) -> str:
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到任何 .pkl 文件")
    return pkl_files[-1]


def load_demo_data(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"文件为空或格式不对: {path}")
    return data


def split_into_demos(data: List[dict]) -> List[List[dict]]:
    demos = []
    current = []

    for trans in data:
        current.append(trans)
        if bool(trans.get("dones", False)):
            demos.append(current)
            current = []

    if current:
        demos.append(current)

    return demos


# ============================================================
# 图像相关
# ============================================================
def get_image_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    if "images" in obs and isinstance(obs["images"], dict):
        return obs["images"]
    return {k: v for k, v in obs.items() if isinstance(v, (np.ndarray, list, tuple))}


def resolve_image_keys(obs: Dict[str, Any], preferred_keys=None) -> List[str]:
    img_dict = get_image_dict(obs)
    available = list(img_dict.keys())

    if preferred_keys is None:
        preferred_keys = []

    keys = []
    for k in preferred_keys:
        if k in available and k not in keys:
            keys.append(k)
    for k in available:
        if k not in keys:
            keys.append(k)

    return keys


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    while img.ndim > 3:
        img = img[-1]

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 假设原始是 RGB，转 BGR 以便 OpenCV 显示
    img = img[..., ::-1]
    return img


def resize_with_padding(img: np.ndarray, target_w: int, target_h: int, pad_color=(0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def build_image_grid(
    obs: Dict[str, Any],
    image_keys: List[str],
    max_canvas_w: int = DEFAULT_MAX_CANVAS_W,
    max_image_area_h: int = DEFAULT_MAX_IMAGE_AREA_H,
    gap: int = 8,
    bg_color=(10, 10, 10),
) -> np.ndarray:
    """
    根据图像数量自适应拼接：
    - 1~3 张：单行
    - 4~6 张：两行
    - 更多：自动多行
    """
    img_dict = get_image_dict(obs)
    valid_keys = [k for k in image_keys if k in img_dict]

    if len(valid_keys) == 0:
        return np.full((300, 800, 3), bg_color, dtype=np.uint8)

    n = len(valid_keys)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))

    tile_w = int((max_canvas_w - gap * (cols + 1)) / cols)
    tile_h = int((max_image_area_h - gap * (rows + 1)) / rows)
    tile_h = max(tile_h, 180)
    tile_w = max(tile_w, 240)

    grid_h = rows * tile_h + (rows + 1) * gap
    grid_w = cols * tile_w + (cols + 1) * gap
    canvas = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for idx, key in enumerate(valid_keys):
        r = idx // cols
        c = idx % cols
        x0 = gap + c * (tile_w + gap)
        y0 = gap + r * (tile_h + gap)

        img = normalize_image(img_dict[key])
        tile = resize_with_padding(img, tile_w, tile_h, pad_color=(0, 0, 0))
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        # 左上角画 key
        label_bg_h = 28
        cv2.rectangle(canvas, (x0, y0), (x0 + min(tile_w, 320), y0 + label_bg_h), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            key,
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ============================================================
# 状态 / 动作提取
# ============================================================
def get_state_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    state = obs.get("state", {})
    if isinstance(state, dict):
        return state
    return {}


def extract_gripper_feedback(obs: Dict[str, Any]) -> Optional[float]:
    if not isinstance(obs, dict):
        return None

    state = obs.get("state", None)
    if state is None:
        return None

    if isinstance(state, dict):
        preferred_keys = [
            "right_gripper",
            "left_gripper",
            "gripper",
            "state/right_gripper",
            "state/left_gripper",
        ]
        for key in preferred_keys:
            if key in state:
                arr = np.asarray(state[key]).reshape(-1)
                if arr.size > 0:
                    return float(arr[-1])

        for key, val in state.items():
            if "gripper" in str(key).lower():
                arr = np.asarray(val).reshape(-1)
                if arr.size > 0:
                    return float(arr[-1])
        return None

    arr = np.asarray(state)
    while arr.ndim > 1:
        arr = arr[-1]
    arr = arr.reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[-1])


def extract_ee_pose_abs(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict):
        return None

    state = obs.get("state", None)
    if state is None:
        return None

    if isinstance(state, dict):
        preferred_keys = [
            "right_ee_pose",
            "left_ee_pose",
            "ee_pose",
            "state/right_ee_pose",
            "state/left_ee_pose",
        ]
        for key in preferred_keys:
            if key in state:
                arr = np.asarray(state[key]).reshape(-1)
                if arr.size >= 7:
                    return arr[:7].astype(np.float32)

        for key, val in state.items():
            if "ee_pose" in str(key).lower():
                arr = np.asarray(val).reshape(-1)
                if arr.size >= 7:
                    return arr[:7].astype(np.float32)
        return None

    arr = np.asarray(state)
    while arr.ndim > 1:
        arr = arr[-1]
    arr = arr.reshape(-1)
    if arr.size >= 7:
        return arr[:7].astype(np.float32)
    return None


def infer_stable_gripper_state_from_feedback(
    gripper_feedback: Optional[float],
    prev_state: Optional[int],
    close_max=GRIPPER_FEEDBACK_CLOSE_MAX,
    open_min=GRIPPER_FEEDBACK_OPEN_MIN,
) -> Optional[int]:
    """
    输出稳定状态：
      -1 -> 当前稳定闭合
      +1 -> 当前稳定张开
    中间区 -> 延续上一稳定状态
    """
    if gripper_feedback is None:
        return prev_state

    x = float(gripper_feedback)
    if x <= close_max:
        return -1
    if x >= open_min:
        return +1
    return prev_state


def decode_gripper_label(grip_value: Optional[float], deadband=0.5) -> Tuple[Optional[int], str]:
    if grip_value is None:
        return None, "<not found>"

    g = float(grip_value)
    if g <= -deadband:
        return -1, "close(-1)"
    if g >= deadband:
        return +1, "open(+1)"
    return 0, "hold(0)"


def resolve_hw_from_gripper_label(label: Optional[int], prev_hw_cmd: float) -> float:
    if label is None:
        return prev_hw_cmd
    if label <= -1:
        return GRIPPER_CLOSE_HW
    if label >= +1:
        return GRIPPER_OPEN_HW
    return prev_hw_cmd


def analyze_demo_transitions(demo: List[dict]) -> List[dict]:
    """
    预分析每一帧的夹爪显示信息，保证跳帧时也正确。
    """
    results = []

    prev_feedback_state = None
    prev_hw_cmd = GRIPPER_OPEN_HW

    for idx, trans in enumerate(demo):
        obs = trans.get("observations", {})
        next_obs = trans.get("next_observations", {})
        action = np.asarray(trans.get("actions", []), dtype=np.float32).reshape(-1)

        obs_fb = extract_gripper_feedback(obs)
        next_fb = extract_gripper_feedback(next_obs)

        obs_state = infer_stable_gripper_state_from_feedback(obs_fb, prev_feedback_state)
        next_state = infer_stable_gripper_state_from_feedback(next_fb, obs_state)

        feedback_event = 0
        if obs_state is not None and next_state is not None:
            if obs_state == +1 and next_state == -1:
                feedback_event = -1
            elif obs_state == -1 and next_state == +1:
                feedback_event = +1
            else:
                feedback_event = 0

        action_grip = None
        if action.shape[0] >= 7:
            action_grip = float(action[6])

        action_label, action_label_text = decode_gripper_label(action_grip)
        feedback_label, feedback_label_text = decode_gripper_label(feedback_event)

        resolved_hw = resolve_hw_from_gripper_label(action_label, prev_hw_cmd)
        prev_hw_cmd = resolved_hw
        prev_feedback_state = next_state

        ee_obs = extract_ee_pose_abs(obs)
        ee_next = extract_ee_pose_abs(next_obs)

        xyz_delta = None
        quat_delta_raw = None
        if ee_obs is not None and ee_next is not None:
            xyz_delta = ee_next[:3] - ee_obs[:3]
            quat_delta_raw = ee_next[3:7] - ee_obs[3:7]

        results.append(
            {
                "obs_gripper_feedback": obs_fb,
                "next_gripper_feedback": next_fb,
                "feedback_state_prev": obs_state,
                "feedback_state_next": next_state,
                "feedback_event": feedback_event,
                "feedback_event_text": feedback_label_text,
                "action_gripper_raw": action_grip,
                "action_label": action_label,
                "action_label_text": action_label_text,
                "resolved_hw_cmd": resolved_hw,
                "ee_obs": ee_obs,
                "ee_next": ee_next,
                "xyz_delta": xyz_delta,
                "quat_delta_raw": quat_delta_raw,
            }
        )

    return results


# ============================================================
# 文本面板绘制
# ============================================================
def draw_multiline_text(
    img: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    dy=24,
    color=(255, 255, 255),
    scale=0.55,
    thickness=1,
):
    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        yy += dy


def build_info_columns(
    trans: dict,
    analyzed: dict,
    demo_idx: int,
    demo_total: int,
    frame_idx: int,
    frame_total: int,
    global_idx: int,
    global_total: int,
    image_keys: List[str],
    fps: float,
    playing: bool,
) -> Tuple[List[str], List[str], List[str]]:
    reward = safe_float(trans.get("rewards", 0.0), 0.0)
    done = safe_bool(trans.get("dones", False), False)
    info = trans.get("infos", {})

    action = np.asarray(trans.get("actions", []), dtype=np.float32).reshape(-1)
    act_dim = action.shape[0]
    act_norm = float(np.linalg.norm(action)) if action.size > 0 else 0.0

    col1 = []
    col1.append(f"Demo {demo_idx}/{demo_total} | Frame {frame_idx}/{frame_total} | Global {global_idx}/{global_total}")
    col1.append(f"Play={'ON' if playing else 'PAUSE'} | FPS={fps:.2f}")
    col1.append(f"reward={reward:.4f} | done={done}")
    if isinstance(info, dict) and "succeed" in info:
        col1.append(f"succeed={info['succeed']}")
    col1.append("image_keys = " + ", ".join(image_keys))
    col1.append(f"action_dim={act_dim} | action_norm={act_norm:.6f}")

    if act_dim == 7:
        pos = action[:3]
        rot = action[3:6]
        grip = action[6]
        col1.append(f"action.pos_delta = {format_vec(pos, 4, True)}")
        col1.append(f"action.rot_delta = {format_vec(rot, 4, True)}")
        col1.append(f"action.gripper_raw = {grip:+.4f}")
        col1.append(f"action.gripper_label = {analyzed['action_label_text']}")
        col1.append(f"action->hw_cmd = {analyzed['resolved_hw_cmd']:.1f}")
    elif act_dim > 0:
        col1.append("action = " + format_vec(action, 4, True))

    ee_obs = analyzed["ee_obs"]
    ee_next = analyzed["ee_next"]
    xyz_delta = analyzed["xyz_delta"]
    quat_delta_raw = analyzed["quat_delta_raw"]

    col2 = []
    col2.append("EE pose / delta")
    if ee_obs is None:
        col2.append("ee_pose(obs) = <not found>")
    else:
        col2.append(f"ee_pose(obs).xyz  = {format_vec(ee_obs[:3], 4, True)}")
        col2.append(f"ee_pose(obs).quat = {format_vec(ee_obs[3:7], 4, True)}")

    if ee_next is None:
        col2.append("ee_pose(next) = <not found>")
    else:
        col2.append(f"ee_pose(next).xyz  = {format_vec(ee_next[:3], 4, True)}")
        col2.append(f"ee_pose(next).quat = {format_vec(ee_next[3:7], 4, True)}")

    if xyz_delta is not None:
        col2.append(f"measured.xyz_delta = {format_vec(xyz_delta, 4, True)}")
    if quat_delta_raw is not None:
        col2.append(f"measured.quat_raw_delta = {format_vec(quat_delta_raw, 4, True)}")

    col3 = []
    col3.append("Gripper / infos")
    obs_fb = analyzed["obs_gripper_feedback"]
    next_fb = analyzed["next_gripper_feedback"]

    col3.append(
        "gripper_feedback(obs)  = "
        + (f"{obs_fb:.4f}" if obs_fb is not None else "<not found>")
    )
    col3.append(
        "gripper_feedback(next) = "
        + (f"{next_fb:.4f}" if next_fb is not None else "<not found>")
    )
    col3.append(f"feedback_state_prev = {analyzed['feedback_state_prev']}")
    col3.append(f"feedback_state_next = {analyzed['feedback_state_next']}")
    col3.append(f"feedback_event      = {analyzed['feedback_event_text']}")
    col3.append(f"stored_label        = {analyzed['action_label_text']}")
    col3.append(
        f"hw_mapping          = close->{GRIPPER_CLOSE_HW:.0f}, hold->prev, open->{GRIPPER_OPEN_HW:.0f}"
    )

    if isinstance(info, dict):
        info_lines = flatten_scalar_info(info)
        info_lines = wrap_lines(info_lines, max_len=70)
        if len(info_lines) > 0:
            col3.append("--- infos ---")
            col3.extend(info_lines[:10])
            if len(info_lines) > 10:
                col3.append(f"... ({len(info_lines) - 10} more)")

    return col1, col2, col3


def build_info_panel(
    trans: dict,
    analyzed: dict,
    demo_idx: int,
    demo_total: int,
    frame_idx: int,
    frame_total: int,
    global_idx: int,
    global_total: int,
    image_keys: List[str],
    canvas_w: int,
    fps: float,
    playing: bool,
) -> np.ndarray:
    col1, col2, col3 = build_info_columns(
        trans=trans,
        analyzed=analyzed,
        demo_idx=demo_idx,
        demo_total=demo_total,
        frame_idx=frame_idx,
        frame_total=frame_total,
        global_idx=global_idx,
        global_total=global_total,
        image_keys=image_keys,
        fps=fps,
        playing=playing,
    )

    controls = [
        "SPACE: play/pause",
        "A / , : prev frame",
        "D / . : next frame",
        "N : next demo",
        "B : prev demo",
        "R : restart current demo",
        "[ / - : slower",
        "] / = : faster",
        "J : jump to demo/frame",
        "H : print help",
        "ESC / Q : quit",
    ]

    line_h = 24
    top_margin = 30
    bottom_margin = 24
    footer_extra = 46

    max_lines = max(len(col1), len(col2), len(col3))
    panel_h = top_margin + max_lines * line_h + footer_extra + bottom_margin

    panel = np.full((panel_h, canvas_w, 3), DEFAULT_BG_COLOR, dtype=np.uint8)

    col_w = canvas_w // 3
    x1 = 18
    x2 = col_w + 18
    x3 = col_w * 2 + 18

    draw_multiline_text(panel, col1, x1, 28, dy=line_h, color=(0, 255, 0), scale=0.58, thickness=1)
    draw_multiline_text(panel, col2, x2, 28, dy=line_h, color=(0, 255, 255), scale=0.56, thickness=1)
    draw_multiline_text(panel, col3, x3, 28, dy=line_h, color=(255, 220, 0), scale=0.54, thickness=1)

    # 分割线
    cv2.line(panel, (col_w, 10), (col_w, panel_h - 10), (80, 80, 80), 1)
    cv2.line(panel, (col_w * 2, 10), (col_w * 2, panel_h - 10), (80, 80, 80), 1)

    controls_y = panel_h - 40
    control_line = " | ".join(controls)
    draw_multiline_text(panel, wrap_lines([control_line], 130), 18, controls_y, dy=22, color=(220, 220, 220), scale=0.50, thickness=1)

    return panel


# ============================================================
# 整体画布
# ============================================================
def build_canvas(
    trans: dict,
    analyzed: dict,
    demo_idx: int,
    demo_total: int,
    frame_idx: int,
    frame_total: int,
    global_idx: int,
    global_total: int,
    image_keys: List[str],
    fps: float,
    playing: bool,
    max_canvas_w: int,
    max_image_area_h: int,
) -> np.ndarray:
    obs = trans.get("observations", {})
    image_grid = build_image_grid(
        obs=obs,
        image_keys=image_keys,
        max_canvas_w=max_canvas_w,
        max_image_area_h=max_image_area_h,
        gap=8,
        bg_color=(8, 8, 8),
    )

    info_panel = build_info_panel(
        trans=trans,
        analyzed=analyzed,
        demo_idx=demo_idx,
        demo_total=demo_total,
        frame_idx=frame_idx,
        frame_total=frame_total,
        global_idx=global_idx,
        global_total=global_total,
        image_keys=image_keys,
        canvas_w=image_grid.shape[1],
        fps=fps,
        playing=playing,
    )

    canvas = np.vstack([image_grid, info_panel])
    return canvas


# ============================================================
# 播放器
# ============================================================
def print_help():
    print("\n========== Demo Player Help ==========")
    print("SPACE : 播放 / 暂停")
    print("A / , : 上一帧")
    print("D / . : 下一帧")
    print("N     : 下一条 demo")
    print("B     : 上一条 demo")
    print("R     : 当前 demo 从头播放")
    print("[ / - : 降低播放速度")
    print("] / = : 提高播放速度")
    print("J     : 跳转到指定 demo / frame")
    print("H     : 打印帮助")
    print("ESC/Q : 退出")
    print("======================================\n")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default=None, help="指定 demo pkl 路径；为空则自动读取当前目录最新 pkl")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="自动搜索 pkl 的目录")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="初始播放 FPS")
    parser.add_argument("--max_canvas_w", type=int, default=DEFAULT_MAX_CANVAS_W, help="图像区域最大宽度")
    parser.add_argument("--max_image_area_h", type=int, default=DEFAULT_MAX_IMAGE_AREA_H, help="图像区域最大高度")
    parser.add_argument(
        "--image_keys",
        nargs="*",
        default=None,
        help="手动指定图像 key 顺序；为空则自动按默认优先级 + 实际 key",
    )
    args = parser.parse_args()

    pkl_path = args.pkl if args.pkl is not None else find_latest_demo_pkl(args.data_dir)
    print(f"正在加载 demo 文件: {pkl_path}")

    data = load_demo_data(pkl_path)
    demos = split_into_demos(data)

    print(f"总 transition 数: {len(data)}")
    print(f"总 demo 数: {len(demos)}")

    if len(data) == 0:
        raise ValueError("数据为空")

    first_obs = data[0].get("observations", {})
    auto_image_keys = resolve_image_keys(first_obs, args.image_keys or PREFERRED_IMAGE_KEYS)

    print(f"图像 keys: {auto_image_keys}")
    print_help()

    # 预分析每个 demo
    analyzed_demos = [analyze_demo_transitions(demo) for demo in demos]

    win_name = "Merged Demo Player"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    current_demo_idx = 0
    current_frame_idx = 0
    playing = True
    fps = float(args.fps)

    # 用于 global frame index 显示
    demo_start_global = []
    running = 0
    for demo in demos:
        demo_start_global.append(running)
        running += len(demo)

    while True:
        current_demo_idx = clamp(current_demo_idx, 0, len(demos) - 1)
        demo = demos[current_demo_idx]
        analyzed_demo = analyzed_demos[current_demo_idx]
        current_frame_idx = clamp(current_frame_idx, 0, len(demo) - 1)

        trans = demo[current_frame_idx]
        analyzed = analyzed_demo[current_frame_idx]

        frame_total = len(demo)
        global_idx = demo_start_global[current_demo_idx] + current_frame_idx + 1

        # 每个 demo 可以重新根据当前 obs 自动补图像 keys
        obs = trans.get("observations", {})
        image_keys = resolve_image_keys(obs, args.image_keys or PREFERRED_IMAGE_KEYS)

        canvas = build_canvas(
            trans=trans,
            analyzed=analyzed,
            demo_idx=current_demo_idx + 1,
            demo_total=len(demos),
            frame_idx=current_frame_idx + 1,
            frame_total=frame_total,
            global_idx=global_idx,
            global_total=len(data),
            image_keys=image_keys,
            fps=fps,
            playing=playing,
            max_canvas_w=args.max_canvas_w,
            max_image_area_h=args.max_image_area_h,
        )

        cv2.imshow(win_name, canvas)

        delay = max(1, int(1000 / max(fps, 0.1))) if playing else 0
        key = cv2.waitKey(delay) & 0xFF

        # 自动播放且没有按键
        if key == 255:
            if playing:
                if current_frame_idx < len(demo) - 1:
                    current_frame_idx += 1
                else:
                    # 到结尾自动暂停，等你决定下一步
                    playing = False
            continue

        if key in [27, ord('q'), ord('Q')]:
            break

        elif key == ord(' '):
            playing = not playing

        elif key in [ord('a'), ord('A'), ord(',')]:
            playing = False
            current_frame_idx = max(0, current_frame_idx - 1)

        elif key in [ord('d'), ord('D'), ord('.')]:
            playing = False
            current_frame_idx = min(len(demo) - 1, current_frame_idx + 1)

        elif key in [ord('n'), ord('N')]:
            playing = False
            current_demo_idx = min(len(demos) - 1, current_demo_idx + 1)
            current_frame_idx = 0

        elif key in [ord('b'), ord('B')]:
            playing = False
            current_demo_idx = max(0, current_demo_idx - 1)
            current_frame_idx = 0

        elif key in [ord('r'), ord('R')]:
            playing = False
            current_frame_idx = 0

        elif key in [ord('['), ord('-')]:
            fps = max(0.5, fps / 1.25)
            print(f"当前 FPS: {fps:.2f}")

        elif key in [ord(']'), ord('=')]:
            fps = min(120.0, fps * 1.25)
            print(f"当前 FPS: {fps:.2f}")

        elif key in [ord('h'), ord('H')]:
            print_help()

        elif key in [ord('j'), ord('J')]:
            playing = False
            try:
                text = input("输入跳转位置，格式 demo_idx 或 demo_idx:frame_idx （从1开始）: ").strip()
                if ":" in text:
                    d_str, f_str = text.split(":")
                    d = int(d_str) - 1
                    f = int(f_str) - 1
                    d = clamp(d, 0, len(demos) - 1)
                    f = clamp(f, 0, len(demos[d]) - 1)
                    current_demo_idx = d
                    current_frame_idx = f
                else:
                    d = int(text) - 1
                    d = clamp(d, 0, len(demos) - 1)
                    current_demo_idx = d
                    current_frame_idx = 0
            except Exception as e:
                print(f"跳转失败: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()