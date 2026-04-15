import os
import pickle as pkl
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "galaxea_usb_insertion_2_demos_2026-04-14_15-40-33.pkl")
FPS = 15

IMAGE_KEYS = ["left_wrist_rgb", "head_rgb", "right_wrist_rgb"]

with open(PKL_PATH, "rb") as f:
    data = pkl.load(f)

print("总 transition 数:", len(data))
print("第一条 transition 的 observations keys:")
print(data[0]["observations"].keys())
print(type(data[0]["observations"]))

def extract_images(obs):
    if "images" in obs:
        img_dict = obs["images"]
    else:
        img_dict = obs

    frames = []
    for key in IMAGE_KEYS:
        if key not in img_dict:
            continue

        img = np.asarray(img_dict[key])

        while img.ndim > 3:
            img = img[-1]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = img[..., ::-1]  # RGB -> BGR
        img = cv2.resize(img, (256, 256))
        frames.append(img)

    return frames

demo_idx = 1
frame_in_demo = 0

for i, trans in enumerate(data):
    obs = trans["observations"]
    frames = extract_images(obs)

    if not frames:
        print(f"第 {i} 帧没有找到可显示图像，跳过")
        continue

    canvas = np.concatenate(frames, axis=1)

    # 左上角打字
    text = f"Demo {demo_idx} | Frame {frame_in_demo}"
    cv2.putText(
        canvas,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("demo playback", canvas)
    key = cv2.waitKey(int(1000 / FPS)) & 0xFF

    if key == 27:  # ESC
        break

    frame_in_demo += 1

    # 一条 demo 结束
    if bool(trans.get("dones", False)):
        print(f"✅ Demo {demo_idx} 播放结束，共 {frame_in_demo} 帧")

        # 停住等你看
        print("按任意键播放下一条 demo，按 ESC 退出...")
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

        demo_idx += 1
        frame_in_demo = 0

cv2.destroyAllWindows()