#可视化图像+相位末端增量+夹爪量程反馈数值+绝对末端位姿+夹爪命令

import os
import glob
import pickle as pkl
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

# 如果你想手动指定文件，就填绝对路径；否则设为 None 自动读取最新 demo
PKL_PATH = None

FPS = 15
IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]
SHOW_SIZE = (256, 256)

# 新增：底部信息栏高度
INFO_PANEL_HEIGHT = 260


def find_latest_demo_pkl():
    pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下没有找到任何 .pkl demo 文件")
    return pkl_files[-1]


def load_demo_data(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"文件为空或格式不对: {path}")
    return data


def split_into_demos(data):
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


def extract_images(obs):
    if "images" in obs:
        img_dict = obs["images"]
    else:
        img_dict = obs

    frames = []
    for key in IMAGE_KEYS:
        if key not in img_dict:
            blank = np.zeros((SHOW_SIZE[1], SHOW_SIZE[0], 3), dtype=np.uint8)
            frames.append(blank)
            continue

        img = np.asarray(img_dict[key])

        while img.ndim > 3:
            img = img[-1]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = img[..., ::-1]  # RGB -> BGR
        img = cv2.resize(img, SHOW_SIZE)
        frames.append(img)

    return frames


def extract_gripper_feedback(obs):
    if "state" not in obs:
        return None

    state = obs["state"]

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
                val = np.asarray(state[key]).reshape(-1)
                if val.size > 0:
                    return float(val[-1])

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


def extract_ee_pose_abs(obs):
    if "state" not in obs:
        return None

    state = obs["state"]

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


def format_action_text(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    act_dim = action.shape[0]
    act_norm = float(np.linalg.norm(action))

    lines = [
        f"action_dim={act_dim} | action_norm={act_norm:.6f}",
    ]

    if act_dim == 7:
        pos = action[:3]
        rot = action[3:6]
        grip = action[6]
        lines.append(
            "pos = [{:+.4f}, {:+.4f}, {:+.4f}]".format(pos[0], pos[1], pos[2])
        )
        lines.append(
            "rot = [{:+.4f}, {:+.4f}, {:+.4f}]".format(rot[0], rot[1], rot[2])
        )
        lines.append(f"gripper_cmd = {grip:+.4f}")

    elif act_dim == 14:
        left = action[:7]
        right = action[7:]
        lines.append(
            "left  = [{}]".format(", ".join([f"{x:+.3f}" for x in left]))
        )
        lines.append(
            "right = [{}]".format(", ".join([f"{x:+.3f}" for x in right]))
        )

    else:
        lines.append(
            "action = [{}]".format(", ".join([f"{x:+.4f}" for x in action]))
        )

    return lines


def format_ee_pose_text(obs):
    ee_pose = extract_ee_pose_abs(obs)
    if ee_pose is None:
        return ["ee_pose_abs = <not found>"]

    xyz = ee_pose[:3]
    quat = ee_pose[3:7]

    lines = [
        "ee_pose_abs (xyz + quat)",
        "xyz  = [{:+.4f}, {:+.4f}, {:+.4f}]".format(xyz[0], xyz[1], xyz[2]),
        "quat = [{:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}]".format(
            quat[0], quat[1], quat[2], quat[3]
        ),
    ]
    return lines


def format_feedback_text(obs, next_obs):
    grip_fb = extract_gripper_feedback(obs)
    next_grip_fb = extract_gripper_feedback(next_obs)

    lines = []
    if grip_fb is None:
        lines.append("gripper_feedback(obs)      = <not found>")
    else:
        lines.append(f"gripper_feedback(obs)      = {grip_fb:+.4f}")

    if next_grip_fb is not None:
        lines.append(f"gripper_feedback(next_obs) = {next_grip_fb:+.4f}")

    return lines


def draw_multiline_text(img, lines, x, y, dy=22, color=(255, 255, 255), scale=0.52, thickness=1):
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


def make_canvas(trans, demo_idx, demo_total, frame_idx, frame_total, global_idx, global_total):
    obs = trans["observations"]
    next_obs = trans.get("next_observations", {})
    frames = extract_images(obs)
    image_strip = np.concatenate(frames, axis=1)

    reward = float(trans.get("rewards", 0.0))
    done = bool(trans.get("dones", False))
    info = trans.get("infos", {})

    header_lines = [
        f"Demo {demo_idx}/{demo_total} | Frame {frame_idx}/{frame_total} | Global {global_idx}/{global_total}",
        f"reward={reward:.4f} | done={done}",
    ]
    if isinstance(info, dict) and "succeed" in info:
        header_lines.append(f"succeed={info['succeed']}")

    action_lines = format_action_text(trans["actions"])
    ee_pose_lines = format_ee_pose_text(obs)
    feedback_lines = format_feedback_text(obs, next_obs)

    # 新增：底部黑色信息区
    h, w, _ = image_strip.shape
    info_panel = np.zeros((INFO_PANEL_HEIGHT, w, 3), dtype=np.uint8)

    draw_multiline_text(info_panel, header_lines, 20, 28, dy=22, color=(0, 255, 0), scale=0.62, thickness=2)
    draw_multiline_text(info_panel, action_lines, 20, 95, dy=22, color=(255, 255, 0), scale=0.54, thickness=1)
    draw_multiline_text(info_panel, ee_pose_lines, 20, 185, dy=22, color=(0, 255, 255), scale=0.52, thickness=1)
    draw_multiline_text(info_panel, feedback_lines, 430, 95, dy=22, color=(0, 200, 255), scale=0.52, thickness=1)

    footer_lines = [
        "ESC: quit | any key after demo end: next demo",
    ]
    draw_multiline_text(info_panel, footer_lines, 20, INFO_PANEL_HEIGHT - 18, dy=20, color=(255, 255, 255), scale=0.48, thickness=1)

    canvas = np.vstack([image_strip, info_panel])
    return canvas


def main():
    path = PKL_PATH if PKL_PATH is not None else find_latest_demo_pkl()
    print(f"正在加载 demo 文件: {path}")

    data = load_demo_data(path)
    demos = split_into_demos(data)

    print("总 transition 数:", len(data))
    print("总 demo 数:", len(demos))
    print("第一条 transition 的 observations keys:")
    print(data[0]["observations"].keys())
    print(type(data[0]["observations"]))

    cv2.namedWindow("demo playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo playback", 1500, 850)

    global_idx = 0

    for demo_idx, demo in enumerate(demos, start=1):
        frame_total = len(demo)
        print(f"\n▶️ 开始播放 Demo {demo_idx}/{len(demos)}，共 {frame_total} 帧")

        for frame_idx, trans in enumerate(demo, start=1):
            canvas = make_canvas(
                trans=trans,
                demo_idx=demo_idx,
                demo_total=len(demos),
                frame_idx=frame_idx,
                frame_total=frame_total,
                global_idx=global_idx + 1,
                global_total=len(data),
            )

            cv2.imshow("demo playback", canvas)
            key = cv2.waitKey(int(1000 / FPS)) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return

            global_idx += 1

        print(f"✅ Demo {demo_idx} 播放结束，共 {frame_total} 帧")
        print("按任意键播放下一条 demo，按 ESC 退出...")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#可视化图像+相位末端增量+夹爪命令对应+-0.9
# import os
# import glob
# import pickle as pkl
# import cv2
# import numpy as np


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = BASE_DIR

# # 如果你想手动指定文件，就填绝对路径；否则设为 None 自动读取最新 demo
# PKL_PATH = None

# FPS = 15
# IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]
# SHOW_SIZE = (256, 256)


# def find_latest_demo_pkl():
#     pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
#     if not pkl_files:
#         raise FileNotFoundError(f"在 {DATA_DIR} 下没有找到任何 .pkl demo 文件")
#     return pkl_files[-1]


# def load_demo_data(path):
#     with open(path, "rb") as f:
#         data = pkl.load(f)
#     if not isinstance(data, list) or len(data) == 0:
#         raise ValueError(f"文件为空或格式不对: {path}")
#     return data


# def split_into_demos(data):
#     demos = []
#     current = []

#     for trans in data:
#         current.append(trans)
#         if bool(trans.get("dones", False)):
#             demos.append(current)
#             current = []

#     if current:
#         # 万一最后一条没正确写 dones=True，也保底收进去
#         demos.append(current)

#     return demos


# def extract_images(obs):
#     """
#     兼容两种格式：
#     1) obs[key]
#     2) obs["images"][key]
#     """
#     if "images" in obs:
#         img_dict = obs["images"]
#     else:
#         img_dict = obs

#     frames = []
#     for key in IMAGE_KEYS:
#         if key not in img_dict:
#             blank = np.zeros((SHOW_SIZE[1], SHOW_SIZE[0], 3), dtype=np.uint8)
#             frames.append(blank)
#             continue

#         img = np.asarray(img_dict[key])

#         # 如果有 chunk/time 维，取最后一帧
#         while img.ndim > 3:
#             img = img[-1]

#         if img.dtype != np.uint8:
#             img = np.clip(img, 0, 255).astype(np.uint8)

#         # RGB -> BGR
#         img = img[..., ::-1]
#         img = cv2.resize(img, SHOW_SIZE)
#         frames.append(img)

#     return frames


# def format_action_text(action):
#     action = np.asarray(action, dtype=np.float32).reshape(-1)
#     act_dim = action.shape[0]
#     act_norm = float(np.linalg.norm(action))

#     lines = [
#         f"action_dim={act_dim} | action_norm={act_norm:.6f}",
#     ]

#     if act_dim == 7:
#         pos = action[:3]
#         rot = action[3:6]
#         grip = action[6]
#         lines.append(
#             "pos = [{:+.4f}, {:+.4f}, {:+.4f}]".format(pos[0], pos[1], pos[2])
#         )
#         lines.append(
#             "rot = [{:+.4f}, {:+.4f}, {:+.4f}]".format(rot[0], rot[1], rot[2])
#         )
#         lines.append(f"gripper = {grip:+.4f}")

#     elif act_dim == 14:
#         left = action[:7]
#         right = action[7:]
#         lines.append(
#             "left  = [{}]".format(", ".join([f"{x:+.3f}" for x in left]))
#         )
#         lines.append(
#             "right = [{}]".format(", ".join([f"{x:+.3f}" for x in right]))
#         )

#     else:
#         lines.append(
#             "action = [{}]".format(", ".join([f"{x:+.4f}" for x in action]))
#         )

#     return lines


# def draw_multiline_text(img, lines, x, y, dy=28, color=(255, 255, 255), scale=0.65, thickness=2):
#     yy = y
#     for line in lines:
#         cv2.putText(
#             img,
#             line,
#             (x, yy),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             scale,
#             color,
#             thickness,
#             cv2.LINE_AA,
#         )
#         yy += dy


# def make_canvas(trans, demo_idx, demo_total, frame_idx, frame_total, global_idx, global_total):
#     obs = trans["observations"]
#     frames = extract_images(obs)
#     canvas = np.concatenate(frames, axis=1)

#     reward = float(trans.get("rewards", 0.0))
#     done = bool(trans.get("dones", False))
#     info = trans.get("infos", {})

#     header_lines = [
#         f"Demo {demo_idx}/{demo_total} | Frame {frame_idx}/{frame_total} | Global {global_idx}/{global_total}",
#         f"reward={reward:.4f} | done={done}",
#     ]

#     # 如果最后一帧里有 succeed 信息，也显示出来
#     if isinstance(info, dict) and "succeed" in info:
#         header_lines.append(f"succeed={info['succeed']}")

#     action_lines = format_action_text(trans["actions"])

#     footer_lines = [
#         "ESC: quit | any key after demo end: next demo",
#     ]

#     draw_multiline_text(canvas, header_lines, 20, 30, color=(0, 255, 0), scale=0.7, thickness=2)
#     draw_multiline_text(canvas, action_lines, 20, 110, color=(255, 255, 0), scale=0.6, thickness=2)
#     draw_multiline_text(canvas, footer_lines, 20, canvas.shape[0] - 20, dy=24, color=(255, 255, 255), scale=0.5, thickness=1)

#     return canvas


# def main():
#     path = PKL_PATH if PKL_PATH is not None else find_latest_demo_pkl()
#     print(f"正在加载 demo 文件: {path}")

#     data = load_demo_data(path)
#     demos = split_into_demos(data)

#     print("总 transition 数:", len(data))
#     print("总 demo 数:", len(demos))
#     print("第一条 transition 的 observations keys:")
#     print(data[0]["observations"].keys())
#     print(type(data[0]["observations"]))

#     cv2.namedWindow("demo playback", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("demo playback", 1400, 500)

#     global_idx = 0

#     for demo_idx, demo in enumerate(demos, start=1):
#         frame_total = len(demo)
#         print(f"\n▶️ 开始播放 Demo {demo_idx}/{len(demos)}，共 {frame_total} 帧")

#         for frame_idx, trans in enumerate(demo, start=1):
#             canvas = make_canvas(
#                 trans=trans,
#                 demo_idx=demo_idx,
#                 demo_total=len(demos),
#                 frame_idx=frame_idx,
#                 frame_total=frame_total,
#                 global_idx=global_idx + 1,
#                 global_total=len(data),
#             )

#             cv2.imshow("demo playback", canvas)
#             key = cv2.waitKey(int(1000 / FPS)) & 0xFF
#             if key == 27:  # ESC
#                 cv2.destroyAllWindows()
#                 return

#             global_idx += 1

#         print(f"✅ Demo {demo_idx} 播放结束，共 {frame_total} 帧")
#         print("按任意键播放下一条 demo，按 ESC 退出...")
#         key = cv2.waitKey(0) & 0xFF
#         if key == 27:
#             break

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()