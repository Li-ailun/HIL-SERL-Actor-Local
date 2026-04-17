#需要Ros2Bridge
#需要vr介入
#即：
# 走真实环境导入链。
# 所以一路导到：
# wrapper.py
# dual_galaxea_env.py
# rs_capture.py

import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import threading
from absl import app, flags
from pynput import keyboard

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 导入任务配置入口
from examples.galaxea_task.usb_pick_insertion_single.config import env_config
#from examples.galaxea_task.usb_pick_insertion.config import env_config

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")

success_presses = 0
success_lock = threading.Lock()

# 新增：防止长按连发
space_is_down = False

def on_press(key):
    global success_presses, space_is_down
    try:
        if key == keyboard.Key.space:
            with success_lock:
                # 只有“第一次按下”才记一次
                if not space_is_down:
                    success_presses += 1
                    space_is_down = True
                    pending = success_presses
                    print(f"\n⌨️ 检测到空格，待保存成功帧 +1（当前待消费: {pending}）")
    except Exception as e:
        print(f"\n⚠️ 键盘监听异常(on_press): {e}")

def on_release(key):
    global space_is_down
    try:
        if key == keyboard.Key.space:
            with success_lock:
                space_is_down = False
    except Exception as e:
        print(f"\n⚠️ 键盘监听异常(on_release): {e}")

def main(_):
    global success_presses

    print(f"🚀 开始采集视觉分类器数据：{FLAGS.exp_name}")
    print("💡 操作指南：")
    print("   1. 初次启动时会自动复位一次。")
    print("   2. 进入采集后，不再自动复位。")
    print("   3. 戴上 VR 操控机械臂随意移动，默认持续收集失败样本。")
    print("   4. 当 U 盘完美插入后，按一次【空格】，记录 1 帧成功样本。")
    print("   5. 长按空格不会连发，必须松开后再次按下才会再记 1 帧。\n")

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()

    env = env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    obs, _ = env.reset()
    print("✅ 环境已重置，开始高频采集...")

    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="✅ 已收集成功帧数")

    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, rew, done, truncated, info = env.step(actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - float(done),
                dones=bool(done),
            )
        )
        obs = next_obs

        consume_success = False
        with success_lock:
            if success_presses > 0:
                success_presses -= 1
                consume_success = True

        if consume_success:
            successes.append(transition)
            pbar.update(1)
            print(f"\n✅ 已记录 1 帧成功画面，当前成功总数: {len(successes)}")
        else:
            failures.append(transition)

        sys.stdout.write(f"\r❌ 后台已收集负样本(Failures)帧数: {len(failures)}  ")
        sys.stdout.flush()

        # 只启动时 reset 一次，之后忽略 done/truncated
        if done or truncated:
            done = False
            truncated = False

    print("\n\n✅ 达到目标！停止录制。")
    listener.stop()

    save_dir = os.path.join(os.path.dirname(__file__), "classifier_data_single")
    os.makedirs(save_dir, exist_ok=True)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    success_file = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    )
    with open(success_file, "wb") as f:
        pkl.dump(successes, f)
        print(f"💾 成功！保存了 {success_needed} 帧【成功画面】至 {success_file}")

    failure_file = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    )
    with open(failure_file, "wb") as f:
        pkl.dump(failures, f)
        print(f"💾 成功！保存了 {len(failures)} 帧【失败画面】至 {failure_file}")

if __name__ == "__main__":
    app.run(main)





