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


#保留全部帧，bc过滤静止帧再训练
import os
import sys
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

# 导入你任务定制的环境构建函数
#from examples.galaxea_task.usb_pick_insertion.config import env_config
from examples.galaxea_task.usb_pick_insertion_single.config import env_config

# ==============================================================
# ⚙️ 命令行参数配置
# ==============================================================
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "galaxea_usb_insertion",
    "Name of experiment corresponding to folder.",
)
flags.DEFINE_integer(
    "successes_needed",
    20,
    "Number of successful demos to collect.",
)
flags.DEFINE_integer(
    "max_episode_steps",
    300,
    "Maximum number of steps per demo episode before forcing truncation.",
)


def ask_success_from_terminal():
    """人工输入本回合成功/失败。1=成功，0=失败。"""
    while True:
        try:
            manual_rew = int(input("Success? (1/0): ").strip())
            if manual_rew in [0, 1]:
                return bool(manual_rew)
            print("❌ 请输入 1 或 0。")
        except ValueError:
            print("❌ 输入无效，请输入 1 或 0。")


def main(_):
    print(f"🚀 开始录制专家数据：{FLAGS.exp_name}")

    # ==========================================================
    # 🌍 实例化真实环境
    # ==========================================================
    # 录制专家演示阶段：
    # - 真实环境
    # - 使用人工成功/失败打分
    env = env_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    obs, info = env.reset()
    print("✅ 环境重置完成，请戴上 VR 头显准备接管！")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    max_episode_steps = FLAGS.max_episode_steps

    pbar = tqdm(total=success_needed, desc="成功收集的 Demo 数量")

    # 当前回合的完整轨迹（官方逻辑：整条成功回合全保存）
    trajectory = []
    returns = 0.0
    episode_step = 0

    while success_count < success_needed:
        # 官方逻辑：默认零动作，如果有 intervene_action 再覆盖
        actions = np.zeros(env.action_space.shape, dtype=np.float32)

        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        episode_step += 1

        if "intervene_action" in info:
            actions = np.asarray(info["intervene_action"], dtype=np.float32)

        # ==========================================================
        # ⏰ 超时截断保护
        # ==========================================================
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
                rewards=rew,
                masks=1.0 - float(episode_end),
                dones=episode_end,
                infos=info,
            )
        )
        trajectory.append(transition)

        pbar.set_description(
            f"成功 Demo 数: {success_count}/{success_needed} | "
            f"Return: {returns:.2f} | "
            f"Step: {episode_step}/{max_episode_steps}"
        )

        obs = next_obs

        if episode_end:
            print("\n🔄 回合结束。")

            # ======================================================
            # ✅ 人工 success/fail 判定补丁
            # ======================================================
            # 1) 正常 done 时，底层 use_manual_reward=True 通常会给 info["succeed"]
            # 2) 但如果是我们这里强制 timeout 截断，或者底层没给 succeed，
            #    就在顶层补一次人工输入
            if "succeed" not in info or forced_timeout or truncated:
                print("📝 当前回合需要人工判定 success / fail。")
                info["succeed"] = ask_success_from_terminal()

                # 同步回写最后一帧 transition 的 infos，保证保存的数据一致
                if len(trajectory) > 0:
                    trajectory[-1]["infos"] = copy.deepcopy(info)
                    trajectory[-1]["rewards"] = float(info["succeed"])

            # 官方逻辑：只要这个回合 succeed，就把整条轨迹全存下来
            if info.get("succeed", False):
                for trans in trajectory:
                    transitions.append(copy.deepcopy(trans))
                success_count += 1
                pbar.update(1)
                print(f"🎉 成功录制 1 条 Demo！当前累计成功条数: {success_count}")
                print(f"📦 本条轨迹长度: {len(trajectory)}")
            else:
                print("❌ 当前回合失败，已丢弃该轨迹。")

            # 清空当前回合缓存
            trajectory = []
            returns = 0.0
            episode_step = 0

            # 如果已经够了，就不要再多 reset 一次
            if success_count >= success_needed:
                break

            print("🔄 正在复位机器人...")
            obs, info = env.reset()
            print("✅ 复位完成，可进行下一次演示。\n" + "-" * 40)

    # ==========================================================
    # 💾 保存数据
    # ==========================================================
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



