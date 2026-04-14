

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

from examples.galaxea_task.usb_pick_insertion.wrapper import make_env

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

    env = make_env(reward_classifier_model=None, use_manual_reward=False)

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

    save_dir = os.path.join(os.path.dirname(__file__), "classifier_data")
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





# import os
# import sys
# from tqdm import tqdm
# import numpy as np
# import copy
# import pickle as pkl
# import datetime
# from absl import app, flags
# from pynput import keyboard

# # ==============================================================
# # 🔥 核心路径配置 (彻底解决 ModuleNotFoundError)
# # ==============================================================
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# # 精准导入你的任务定制环境构建器
# from examples.galaxea_task.usb_pick_insertion.wrapper import make_env

# FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
# # 注意：这里的 200 不是 200 个回合，而是 200 帧成功的画面 (Frames/Transitions)
# flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")

# # 全局变量，记录空格键是否被按下
# success_key = False

# def on_press(key):
#     """键盘监听回调函数：一旦检测到按下空格键，就将标志位置为 True"""
#     global success_key
#     try:
#         if str(key) == 'Key.space':
#             success_key = True
#     except AttributeError:
#         pass

# def main(_):
#     global success_key
#     print(f"🚀 开始采集视觉分类器数据：{FLAGS.exp_name}")
#     print("💡 操作指南：")
#     print("   1. 戴上 VR 操控机械臂随意移动 (收集失败样本)。")
#     print("   2. 将 U 盘完美插入插座后，【按一下电脑键盘的空格键】 (记录一帧成功样本)。")
#     print("   3. 稍微改变一点点插入角度/光线，再按空格，直到收集够 200 帧成功画面！\n")

#     # 启动后台键盘监听线程
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()
    
#     # 实例化环境
#     # 注意：这里我们不需要人工打分器弹窗打断我们，所以传入 use_manual_reward=False
#     # (如果你的 make_env 依然弹窗，遇到弹窗敲 0 继续即可)
#     env = make_env(reward_classifier_model=None, use_manual_reward=False)

#     obs, _ = env.reset()
#     print("✅ 环境已重置，开始高频采集...")
    
#     successes = []
#     failures = []
#     success_needed = FLAGS.successes_needed
#     pbar = tqdm(total=success_needed, desc="✅ 已收集成功帧数")
    
#     # ==============================================================
#     # 🔄 极速死循环：不看回合，只看帧 (Frame-Level Collection)
#     # ==============================================================
#     while len(successes) < success_needed:
#         # AI 默认不输出动作
#         actions = np.zeros(env.action_space.shape) 
#         next_obs, rew, done, truncated, info = env.step(actions)
        
#         # 截获 VR 手柄的真实动作 (虽然分类器只看画面，但保存动作好习惯)
#         if "intervene_action" in info:
#             actions = info["intervene_action"]

#         # 打包当前这一帧 (Transition)
#         transition = copy.deepcopy(
#             dict(
#                 observations=obs,
#                 actions=actions,
#                 next_observations=next_obs,
#                 rewards=rew,
#                 masks=1.0 - done,
#                 dones=done,
#             )
#         )
#         obs = next_obs
        
#         # 🌟 核心分拣逻辑：听从键盘的审判
#         if success_key:
#             # 如果你按下了空格键，这一帧就是“成功”的黄金样本
#             successes.append(transition)
#             pbar.update(1)
#             # 记录完一帧后，立刻重置标志位，等待你的下一次敲击
#             success_key = False
#         else:
#             # 只要没按空格，这一刻所有的画面都被归类为“失败” (负样本)
#             failures.append(transition)

#         # 终端刷新，让你知道采集了多少负样本了
#         sys.stdout.write(f"\r❌ 后台已收集负样本(Failures)帧数: {len(failures)}  ")
#         sys.stdout.flush()

#         # 如果环境超时了，自动复位，你可以继续下一把
#         if done or truncated:
#             obs, _ = env.reset()

#     print("\n\n✅ 达到目标！停止录制。")
    
#     # ==============================================================
#     # 💾 数据分离保存 (分别保存正样本和负样本)
#     # ==============================================================
#     save_dir = os.path.join(os.path.dirname(__file__), "classifier_data")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
        
#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
#     # 1. 保存正样本 (Successes)
#     success_file = os.path.join(save_dir, f"{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl")
#     with open(success_file, "wb") as f:
#         pkl.dump(successes, f)
#         print(f"💾 成功！保存了 {success_needed} 帧【成功画面】至 {success_file}")

#     # 2. 保存负样本 (Failures)
#     failure_file = os.path.join(save_dir, f"{FLAGS.exp_name}_failure_images_{uuid}.pkl")
#     with open(failure_file, "wb") as f:
#         pkl.dump(failures, f)
#         print(f"💾 成功！保存了 {len(failures)} 帧【失败画面】至 {failure_file}")
        
# if __name__ == "__main__":
#     app.run(main)



# import copy
# import os
# from tqdm import tqdm
# import numpy as np
# import pickle as pkl
# import datetime
# from absl import app, flags
# from pynput import keyboard

# from experiments.mappings import CONFIG_MAPPING

# FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


# success_key = False
# def on_press(key):
#     global success_key
#     try:
#         if str(key) == 'Key.space':
#             success_key = True
#     except AttributeError:
#         pass

# def main(_):
#     global success_key
#     listener = keyboard.Listener(
#         on_press=on_press)
#     listener.start()
#     assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
#     config = CONFIG_MAPPING[FLAGS.exp_name]()
#     env = config.get_environment(fake_env=False, save_video=False, classifier=False)

#     obs, _ = env.reset()
#     successes = []
#     failures = []
#     success_needed = FLAGS.successes_needed
#     pbar = tqdm(total=success_needed)
    
#     while len(successes) < success_needed:
#         actions = np.zeros(env.action_space.sample().shape) 
#         next_obs, rew, done, truncated, info = env.step(actions)
#         if "intervene_action" in info:
#             actions = info["intervene_action"]

#         transition = copy.deepcopy(
#             dict(
#                 observations=obs,
#                 actions=actions,
#                 next_observations=next_obs,
#                 rewards=rew,
#                 masks=1.0 - done,
#                 dones=done,
#             )
#         )
#         obs = next_obs
#         if success_key:
#             successes.append(transition)
#             pbar.update(1)
#             success_key = False
#         else:
#             failures.append(transition)

#         if done or truncated:
#             obs, _ = env.reset()

#     if not os.path.exists("./classifier_data"):
#         os.makedirs("./classifier_data")
#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
#     with open(file_name, "wb") as f:
#         pkl.dump(successes, f)
#         print(f"saved {success_needed} successful transitions to {file_name}")

#     file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
#     with open(file_name, "wb") as f:
#         pkl.dump(failures, f)
#         print(f"saved {len(failures)} failure transitions to {file_name}")
        
# if __name__ == "__main__":
#     app.run(main)
