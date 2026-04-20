#录制专家轨迹，用于后续先bc再强化学习（人类在环监督或者无监督两种方式）

#该脚本的本质：（1）记录intervene_action（把供bc模仿学习的完美演示数据和共强化学习的人类介入数据都定义为intervene_action，这两个数据都保存仅专家池）
#           （2）所以即使这是录制脚本，也是依靠intervene_action信号记录数据的

#该脚本构成（高度模块化）
#（1）数据输入接口：无该内容，输入接口都在dual_galaxea_env.py里封装好了，只要dual_galaxea_env.py能正常收到需要的数据，则完成；
#（2）怎么录制：无该内容，录制控制都在wrappers.py里封装好了，此处录制的完整流程演示数据和wrappers.py的vr接管数据都被存入专家经验池（Demo Buffer））里
#（3）存放路径：存进 demo_data 文件夹
#（4）何时停止录制：
#       1,基于wrappers.py定义的mode0无人类介入，
                 #mode2人类介入的逻辑： 一旦切入 mode2，它就在这一帧的数据里挂上一个牌子（info["intervene_action"] = ...）。
                 #mode0时不记录，mode2后才记录
#       2,预设固定步长时间，超时自动停止录制
#       3,所以mode2提前结束任务，等待补偿结束，自动记录完美演示数据 

#总逻辑：
# dual_galaxea_env.py (底层)：负责和星海图的相机、机械臂打交道，把物理世界的声光电变成 NumPy 数组。
# wrappers.py (中层)：智能安检，负责监听你的 VR 手柄，把人类的动作替换进去，并贴上 intervene_action 的 VIP 标签。
# record_demos.py / train_rlpd.py (顶层)：数据分拣员。
#      只管看着 info 字典，看到有 VIP 标签的数据，就直接往专家经验池（Demo Buffer）里扔。
#      如果没有标签，就扔进在线探索池（Replay Buffer）。

# 默认 20 条成功 demo：
# python record_demos.py

# 限制每条最多 300 步：
# python record_demos.py --max_episode_steps=300

# 先少录几条测试：
# python record_demos.py --successes_needed=2 --max_episode_steps=200


#过滤静止帧再保存demos（ros2 topic echo /motion_control/pose_ee_arm_right可以确认vr暂停后数值不会变化，所以可以视为静止帧 ）
#（可以回放录制的demos，看看动作是不是流畅的）

import os
import sys

# ==============================================================
# 🔥 像 actor 一样，先强制本地 CPU，避免 classifier=True 时本地环境炸
# ==============================================================
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

# ==============================================================
# 🔥 核心路径配置（确保模块可被正确导入）
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config

# ==============================================================
# ⚙️ 命令行参数配置
# ==============================================================
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "galaxea_usb_insertion_single",
    "Name of experiment corresponding to folder.",
)
flags.DEFINE_integer(
    "successes_needed",
    20,
    "Number of successful demos to collect.",
)
flags.DEFINE_integer(
    "max_episode_steps",
    650,
    "Maximum number of steps per demo episode before forcing truncation.",
)
flags.DEFINE_boolean(
    "classifier",
    True,
    "Whether to use reward classifier as the success/end signal.",
)
flags.DEFINE_boolean(
    "save_video",
    False,
    "Whether to save video during recording.",
)
flags.DEFINE_boolean(
    "manual_confirm_on_success",
    False,
    "Whether to manually confirm success even when classifier says succeed=True.",
)


def ask_success_from_terminal():
    while True:
        try:
            manual_rew = int(input("Success? (1/0): ").strip())
            if manual_rew in [0, 1]:
                return bool(manual_rew)
            print("❌ 请输入 1 或 0。")
        except ValueError:
            print("❌ 输入无效，请输入 1 或 0。")


def extract_gripper_feedback(obs):
    """
    从 obs["state"] 中提取夹爪反馈量程。
    兼容：
    1) state 是 dict，含 right_gripper / gripper
    2) state 是 ndarray，单臂常见最后一维为 gripper
    """
    if "state" not in obs:
        return None

    state = obs["state"]

    if isinstance(state, dict):
        for key in ["right_gripper", "left_gripper", "gripper", "state/right_gripper", "state/left_gripper"]:
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


def infer_gripper_label_from_feedback(
    gripper_feedback,
    prev_label,
    raw_action_gripper=None,
    close_max=30.0,
    open_min=70.0,
):
    """
    按你的规则：
      0~30   -> 闭合标签 -1.0
      70~100 -> 张开标签 +1.0
      中间区 -> 保持上一标签
    如果上一标签也没有，则回退到 raw_action_gripper 或按 50 分界。
    """
    if gripper_feedback is None:
        if prev_label is not None:
            return float(prev_label)
        if raw_action_gripper is not None:
            return -1.0 if float(raw_action_gripper) < 0 else 1.0
        return 1.0

    x = float(gripper_feedback)

    if x <= close_max:
        return -1.0
    if x >= open_min:
        return 1.0

    if prev_label is not None:
        return float(prev_label)

    if raw_action_gripper is not None:
        return -1.0 if float(raw_action_gripper) < 0 else 1.0

    return -1.0 if x < 50.0 else 1.0


def rewrite_gripper_action_with_feedback(action, next_obs, prev_label):
    """
    用 next_obs 里的夹爪反馈，反推这一帧应保存的夹爪训练标签。
    只改 action[6]，其他维保持不变。
    """
    action = np.asarray(action, dtype=np.float32).copy()

    if action.shape[0] != 7:
        return action, prev_label

    feedback = extract_gripper_feedback(next_obs)
    new_label = infer_gripper_label_from_feedback(
        gripper_feedback=feedback,
        prev_label=prev_label,
        raw_action_gripper=action[6],
    )
    action[6] = new_label
    return action, new_label


def main(_):
    print(f"🚀 开始录制专家数据：{FLAGS.exp_name}")
    print(f"🧠 reward classifier: {'开启' if FLAGS.classifier else '关闭'}")
    print("📌 当前脚本策略：")
    print("   - classifier=True 时：成功由 reward ckpt 在成功瞬间触发。")
    print("   - max_episode_steps 只做失败/超时兜底。")
    print("   - 夹爪标签不再保存瞬时按钮命令，而是由反馈量程反推：")
    print("       0~30   -> 闭合(-1)")
    print("       70~100 -> 张开(+1)")
    print("       中间区 -> 保持上一标签\n")

    env = env_config.get_environment(
        fake_env=False,
        save_video=FLAGS.save_video,
        classifier=FLAGS.classifier,
    )

    obs, info = env.reset()
    print("✅ 环境重置完成，请戴上 VR 头显准备接管！")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    max_episode_steps = FLAGS.max_episode_steps

    pbar = tqdm(total=success_needed, desc="成功收集的 Demo 数量")

    trajectory = []
    returns = 0.0
    episode_step = 0

    # 记录当前 episode 的夹爪稳定标签
    prev_gripper_label = None

    while success_count < success_needed:
        raw_actions = np.zeros(env.action_space.shape, dtype=np.float32)

        next_obs, rew, done, truncated, info = env.step(raw_actions)
        returns += float(rew)
        episode_step += 1

        if "intervene_action" in info:
            raw_actions = np.asarray(info["intervene_action"], dtype=np.float32)

        # 先判断“静止帧”，避免因为后面把 gripper 改成 ±1 而导致每帧都非静止
        is_static = np.allclose(raw_actions, 0.0, atol=1e-8)

        # 用夹爪反馈量程反推 gripper 训练标签
        actions, prev_gripper_label = rewrite_gripper_action_with_feedback(
            raw_actions,
            next_obs,
            prev_gripper_label,
        )

        forced_timeout = False
        if episode_step >= max_episode_steps and not (done or truncated):
            forced_timeout = True
            truncated = True
            print(f"\n⏰ 达到最大录制时长：{max_episode_steps} 步，强制截断当前回合。")

        episode_end = bool(done or truncated)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=float(rew),
                masks=1.0 - float(episode_end),
                dones=episode_end,
                infos=info,
            )
        )

        if not is_static:
            trajectory.append(transition)

        pbar.set_description(
            f"成功 Demo 数: {success_count}/{success_needed} | "
            f"Return: {returns:.2f} | "
            f"Step: {episode_step}/{max_episode_steps}"
        )

        obs = next_obs

        if episode_end:
            print("\n🔄 回合结束。")
            print(f"   reward={rew}, done={done}, truncated={truncated}, forced_timeout={forced_timeout}")
            print(f"   info.succeed={info.get('succeed', None)}")

            if FLAGS.classifier:
                succeed = bool(info.get("succeed", False))
                if succeed and FLAGS.manual_confirm_on_success:
                    print("📝 classifier 判定成功，请人工确认本回合是否真的成功。")
                    succeed = ask_success_from_terminal()

                if len(trajectory) > 0:
                    trajectory[-1]["infos"] = copy.deepcopy(info)
                    trajectory[-1]["infos"]["succeed"] = succeed
            else:
                print("📝 当前 classifier=False，使用人工判定 success / fail。")
                succeed = ask_success_from_terminal()
                if len(trajectory) > 0:
                    trajectory[-1]["infos"] = copy.deepcopy(info)
                    trajectory[-1]["infos"]["succeed"] = succeed
                    trajectory[-1]["rewards"] = float(succeed)

            if succeed and len(trajectory) > 0:
                for trans in trajectory:
                    transitions.append(copy.deepcopy(trans))
                success_count += 1
                pbar.update(1)

                print(f"🎉 成功录制 1 条 Demo！当前累计成功条数: {success_count}")
                print(f"🎉 剔除静止帧后，纯净序列长度: {len(trajectory)}")
                print(f"📦 本条轨迹长度: {len(trajectory)}")
            else:
                print("❌ 当前回合失败，或没有有效操作帧，已丢弃该轨迹。")

            trajectory = []
            returns = 0.0
            episode_step = 0
            prev_gripper_label = None

            if success_count >= success_needed:
                break

            print("🔄 正在复位机器人...")
            obs, info = env.reset()
            print("✅ 复位完成，可进行下一次演示。\n" + "-" * 40)

    save_dir = os.path.join(os.path.dirname(__file__), "demo_data_single")
    os.makedirs(save_dir, exist_ok=True)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(
        save_dir,
        f"{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl",
    )

    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)

    print(f"\n💾 恭喜！成功保存 {success_needed} 条 Demo 数据至 {file_name}")
    print(f"📊 总 transition 数量: {len(transitions)}")


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


# # ==============================================================
# # 🔥 核心路径配置（确保模块可被正确导入）
# # ==============================================================
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# # 导入你任务定制的环境构建函数
# #from examples.galaxea_task.usb_pick_insertion.config import env_config
# from examples.galaxea_task.usb_pick_insertion_single.config import env_config

# # ==============================================================
# # ⚙️ 命令行参数配置
# # ==============================================================
# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     "exp_name",
#     "galaxea_usb_insertion_single",
#     "Name of experiment corresponding to folder.",
# )
# flags.DEFINE_integer(
#     "successes_needed",
#     20,
#     "Number of successful demos to collect.",
# )
# flags.DEFINE_integer(
#     "max_episode_steps",
#     650,
#     "Maximum number of steps per demo episode before forcing truncation.",
# )


# def ask_success_from_terminal():
#     """人工输入本回合成功/失败。1=成功，0=失败。"""
#     while True:
#         try:
#             manual_rew = int(input("Success? (1/0): ").strip())
#             if manual_rew in [0, 1]:
#                 return bool(manual_rew)
#             print("❌ 请输入 1 或 0。")
#         except ValueError:
#             print("❌ 输入无效，请输入 1 或 0。")


# def main(_):
#     print(f"🚀 开始录制专家数据：{FLAGS.exp_name}")

#     # ==========================================================
#     # 🌍 实例化真实环境
#     # ==========================================================
#     # 录制专家演示阶段：
#     # - 真实环境
#     # - 使用人工成功/失败打分
#     env = env_config.get_environment(
#         fake_env=False,
#         save_video=False,
#         classifier=True,# False,#True,#  #应该主动打开奖励分类器，最大步长设置为最低底线，不能当最优先的demos判定基准
#     )

#     obs, info = env.reset()
#     print("✅ 环境重置完成，请戴上 VR 头显准备接管！")

#     transitions = []
#     success_count = 0
#     success_needed = FLAGS.successes_needed
#     max_episode_steps = FLAGS.max_episode_steps

#     pbar = tqdm(total=success_needed, desc="成功收集的 Demo 数量")

#     # 当前回合的完整轨迹（官方逻辑：整条成功回合全保存）
#     trajectory = []
#     returns = 0.0
#     episode_step = 0

#     while success_count < success_needed:
#         # 官方逻辑：默认零动作，如果有 intervene_action 再覆盖
#         actions = np.zeros(env.action_space.shape, dtype=np.float32)

#         next_obs, rew, done, truncated, info = env.step(actions)
#         returns += rew
#         episode_step += 1

#         if "intervene_action" in info:
#             actions = np.asarray(info["intervene_action"], dtype=np.float32)

#         # ==========================================================
#         # ⏰ 超时截断保护
#         # ==========================================================
#         forced_timeout = False
#         if episode_step >= max_episode_steps and not (done or truncated):
#             forced_timeout = True
#             truncated = True
#             print(f"\n⏰ 达到最大录制时长：{max_episode_steps} 步，强制截断当前回合。")

#         episode_end = bool(done or truncated)

#         transition = copy.deepcopy(
#             dict(
#                 observations=obs,
#                 actions=actions,
#                 next_observations=next_obs,
#                 rewards=rew,
#                 masks=1.0 - float(episode_end),
#                 dones=episode_end,
#                 infos=info,
#             )
#         )

#         is_static = np.allclose(actions, 0.0, atol=1e-8)

#         #只保留真正有动作的帧，去掉静止帧，回放demos确定数据是纯净的
#         if not is_static:
#             trajectory.append(transition)

#         pbar.set_description(
#             f"成功 Demo 数: {success_count}/{success_needed} | "
#             f"Return: {returns:.2f} | "
#             f"Step: {episode_step}/{max_episode_steps}"
#         )

#         obs = next_obs

#         if episode_end:
#             print("\n🔄 回合结束。")

#             # ======================================================
#             # ✅ 人工 success/fail 判定补丁
#             # ======================================================
#             # 1) 正常 done 时，底层 use_manual_reward=True 通常会给 info["succeed"]
#             # 2) 但如果是我们这里强制 timeout 截断，或者底层没给 succeed，
#             #    就在顶层补一次人工输入
#             if "succeed" not in info or forced_timeout or truncated:
#                 print("📝 当前回合需要人工判定 success / fail。")
#                 info["succeed"] = ask_success_from_terminal()

#                 # 同步回写最后一帧 transition 的 infos，保证保存的数据一致
#                 if len(trajectory) > 0:
#                     trajectory[-1]["infos"] = copy.deepcopy(info)

#                     #不使用classifier=True，仅想跑通demos录制脚本时使用
#                     #trajectory[-1]["rewards"] = float(info["succeed"])

#             # 官方逻辑：只要这个回合 succeed，就把整条轨迹全存下来
#             if info.get("succeed", False) and len(trajectory)> 0:
#                 for trans in trajectory:
#                     transitions.append(copy.deepcopy(trans))
#                 success_count += 1
#                 pbar.update(1)
#                 print(f"🎉 成功录制 1 条 Demo！当前累计成功条数: {success_count}")
#                 print(f"🎉 剔除静止帧后，纯净序列长度: {len(trajectory)}")

#                 print(f"📦 本条轨迹长度: {len(trajectory)}")
#             else:
#                 print("❌ 当前回合失败，或没有有效操作帧，已丢弃该轨迹。")

#             # 清空当前回合缓存
#             trajectory = []
#             returns = 0.0
#             episode_step = 0

#             # 如果已经够了，就不要再多 reset 一次
#             if success_count >= success_needed:
#                 break

#             print("🔄 正在复位机器人...")
#             obs, info = env.reset()
#             print("✅ 复位完成，可进行下一次演示。\n" + "-" * 40)

#     # ==========================================================
#     # 💾 保存数据
#     # ==========================================================
#     save_dir = os.path.join(os.path.dirname(__file__), "demo_data_single")
#     os.makedirs(save_dir, exist_ok=True)

#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     file_name = os.path.join(
#         save_dir,
#         f"{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl",
#     )

#     with open(file_name, "wb") as f:
#         pkl.dump(transitions, f)

#     print(f"\n💾 恭喜！成功保存 {success_needed} 条 Demo 数据至 {file_name}")
#     print(f"📊 总 transition 数量: {len(transitions)}")


# if __name__ == "__main__":
#     app.run(main)



