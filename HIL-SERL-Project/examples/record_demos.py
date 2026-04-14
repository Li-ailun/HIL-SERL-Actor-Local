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


import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags

# ==============================================================
# 🔥 核心路径配置 (确保模块可被正确导入)
# ==============================================================
# 1. 动态定位项目根目录。
# __file__ 获取当前脚本的绝对路径，os.path.dirname 取其所在目录 (examples/)，
# os.path.join(..., '..') 退回上一级到达 HIL-SERL-Project。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. 将项目根目录注入系统环境变量 sys.path。
# 作用：使得 Python 解释器在遇到 `from examples...` 时，知道去根目录下寻找，
# 从而彻底解决 "ModuleNotFoundError"。
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 3. 导入任务定制的环境构建工厂函数。
# 这里关联了你在 usb_pick_insertion/wrapper.py 中定义的 make_env 逻辑，
# 确保底层的硬件接口、Wrappers 和我们的 VRInterventionWrapper 被正确组装。
from examples.galaxea_task.usb_pick_insertion.wrapper import make_env

# ==============================================================
# ⚙️ 命令行参数配置 (利用 absl 库进行灵活传参)
# ==============================================================
FLAGS = flags.FLAGS
# 定义实验名称，如果不通过命令行传入，默认值为 "galaxea_usb_insertion"。
# 该名称将作为最终保存的 .pkl 文件的前缀。
flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
# 定义目标采集数量，即需要多少个 "成功的演示 (succeed=True)" 才会终止脚本。
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    """主执行函数，由 app.run(main) 调用"""
    print(f"🚀 开始录制专家数据：{FLAGS.exp_name}")
    
    # ==============================================================
    # 🌍 环境实例化
    # ==============================================================
    # 核心设定：当前处于“录制纯专家数据”阶段，视觉打分模型还未训练，
    # 必须强制开启使用 HumanClassifierWrapper (use_manual_reward=True)，
    # 将打分权交给操作员终端。
    env = make_env(reward_classifier_model=None, use_manual_reward=True)
    
    # 初始化环境，让机械臂回到安全/起始位置，获取第一帧图像和状态。
    obs, info = env.reset()
    print("✅ 环境重置完成，请戴上 VR 头显准备接管！")
    
    # transitions: 用于存放所有成功回合的数据，最终将被序列化为 .pkl
    transitions = []
    # 成功回合计数器
    success_count = 0
    success_needed = FLAGS.successes_needed
    # 初始化进度条，直观显示收集进度
    pbar = tqdm(total=success_needed, desc="成功收集的 Demo 数量")
    
    # ==============================================================
    # 🔄 外层循环：回合级控制 (Episode-Level)
    # ==============================================================
    # 只要成功收集的 Demo 数量不够，就一直开启新的回合
    while success_count < success_needed:
        # 回合内步数计数器，用于超时判定
        episode_step = 0
        # 超时阈值：以底层 20Hz 控制频率计算，300 步约等于 15 秒的最大操作时间。
        # 作用：防止人类操作员在未完成任务时发呆，导致程序死锁。
        MAX_EPISODE_STEPS = 300  
        
        # --- 🌟 数据清洗 (Auto-Trimming) 状态机标志位 ---
        # 用于记录人类专家是否已经按下了 VR 手柄的接管键
        has_started_intervention = False
        # 用于记录人类专家是否在接管后又松开了按键
        has_finished_intervention = False
        # 临时缓冲池：只存放“掐头去尾”后的高浓度专家操作帧
        pure_trajectory = [] 
        
        # 回合结束标志
        done = False
        # 累积奖励 (在本脚本的人工打分模式下，通常只有最后一步为 1，其余为 0)
        returns = 0
        
        # ==============================================================
        # 🔄 内层循环：帧级步进 (Step-Level)
        # ==============================================================
        # 只要当前回合没结束，就疯狂进行高频采集
        while not done:
            # 1. 默认赋予 AI 全 0 动作（因为当前是纯专家演示阶段，AI 还没学会动）
            actions = np.zeros(env.action_space.shape) 
            
            # 2. 推动物理世界步进。
            # 内部逻辑：如果 VRInterventionWrapper 检测到模式切入，它会拦截上面的 0 动作，
            # 替换为真实的专家动作，并通过 info 返回给此脚本。
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            
            # ==========================================================
            # ✂️ 核心裁剪状态机：判断当前处于哪个操作阶段
            # ==========================================================
            # 检测 info 字典中是否被打上了专家干预的标签
            is_intervening = "intervene_action" in info
            
            if is_intervening:
                # 阶段 2：专家正在发力干活。
                # 提取真实的 VR 动作覆盖默认的 0 动作
                actions = info["intervene_action"]
                # 触发状态：专家已起步
                has_started_intervention = True
                
            elif has_started_intervention and not is_intervening:
                # 阶段 3：专家之前干活了，但现在松手了（退出接管）。
                # 触发状态：专家工作已结束。
                has_finished_intervention = True

            # ==========================================================
            # 📦 数据打包：将 RL 所需的所有元素合并为一个字典
            # ==========================================================
            transition = copy.deepcopy(
                dict(
                    observations=obs,          # 当前帧画面/状态 (S_t)
                    actions=actions,           # 实际执行的动作 (A_t)
                    next_observations=next_obs,# 下一帧画面/状态 (S_t+1)
                    rewards=rew,               # 即时奖励 (R_t)
                    masks=1.0 - done,          # 存活掩码，给后续 RL 算法计算 TD 误差用
                    dones=done,                # 结束标志
                    infos=info,                # 额外信息 (包含是否干预等)
                )
            )
            
            # ==========================================================
            # 🌟 剪辑与保存：丢弃垃圾数据
            # ==========================================================
            # 逻辑精髓：
            # (1) 没开始干活前 (has_started=False)：丢弃发呆数据。
            # (2) 松手干完活后 (has_finished=True)：丢弃等待超时的垃圾时间。
            # (3) 只有中间干活的这几秒 (started=True 且 finished=False)，才存入精华池。
            if has_started_intervention and not has_finished_intervention:
                pure_trajectory.append(transition)
            
            # --- 超时保护机制 ---
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS and not done:
                print(f"\n⏰ 达到最大录制时间 ({MAX_EPISODE_STEPS}步)，强制结束当前回合！")
                done = True # 强行跳出内层 while
                
                # 补救措施：由于是顶层脚本强行判停，底层的 HumanClassifierWrapper 可能无法
                # 捕获正常的 done 信号，我们需要手动弹出一个终端打分器，防止此次操作白费。
                while True:
                    try:
                        manual_rew = int(input("Success? (1/0): "))
                        assert manual_rew in [0, 1]
                        info["succeed"] = bool(manual_rew)
                        break
                    except ValueError:
                        print("Invalid input. Please enter 1 or 0.")
            
            # 动态刷新终端状态（\r 使其在同一行覆盖打印，保持终端整洁）
            sys.stdout.write(f"\r⏳ 纯净专家帧数: {len(pure_trajectory)} | 步数限制: {episode_step}/{MAX_EPISODE_STEPS}  ")
            sys.stdout.flush()

            # 将世界的状态推进到下一帧
            obs = next_obs
            
        # ==============================================================
        # 🧾 回合结算 (脱离内层循环后触发)
        # ==============================================================
        print(f"\n🔄 回合结束。")
        
        # 严格过滤：不仅操作员要打分认定成功 (succeed=True)，
        # 且这个回合必须真的包含了专家的接管数据 (pure_trajectory > 0)，才能算数。
        if info.get("succeed", False) and len(pure_trajectory) > 0:
            print(f"🎉 记录成功！剔除发呆时间后，纯净序列长度: {len(pure_trajectory)}")
            # 将这条精华轨迹中的每一帧，追加到总的数据集列表中
            for transition in pure_trajectory:
                transitions.append(copy.deepcopy(transition))
            success_count += 1
            pbar.update(1) # 更新进度条
        else:
            print("❌ 失败、放弃或没有有效专家操作，已丢弃该轨迹！")
            
        # 清空环境，复位机械臂，准备迎接下一个外层循环
        print("🔄 正在复位机器人...")
        obs, info = env.reset()
        print("✅ 复位完成，可进行下一次演示。\n" + "-"*40)
            
    # ==============================================================
    # 💾 数据持久化 (序列化为 Pickle 文件)
    # ==============================================================
    # 确定保存目录：当前脚本下的 demo_data/ 文件夹
    save_dir = os.path.join(os.path.dirname(__file__), "demo_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) # 如果不存在则创建
        
    # 生成唯一的时间戳标识符
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 拼装最终文件名，格式如：galaxea_usb_insertion_20_demos_2026-04-12_17-30-00.pkl
    file_name = os.path.join(save_dir, f"{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl")
    
    # 写入文件
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"\n💾 恭喜！成功保存 {success_needed} 条 Demo 数据至 {file_name}")

if __name__ == "__main__":
    # 使用 absl 框架启动 main 函数，方便解析命令行传入的 FLAGS
    app.run(main)



# import os
# from tqdm import tqdm
# import numpy as np
# import copy
# import pickle as pkl
# import datetime
# from absl import app, flags
# import time

# from experiments.mappings import CONFIG_MAPPING

# FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

# def main(_):
#     assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
#     config = CONFIG_MAPPING[FLAGS.exp_name]()
#     env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
#     obs, info = env.reset()
#     print("Reset done")
#     transitions = []
#     success_count = 0
#     success_needed = FLAGS.successes_needed
#     pbar = tqdm(total=success_needed)
#     trajectory = []
#     returns = 0
    
#     while success_count < success_needed:
#         actions = np.zeros(env.action_space.sample().shape) 
#         next_obs, rew, done, truncated, info = env.step(actions)
#         returns += rew
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
#                 infos=info,
#             )
#         )
#         trajectory.append(transition)
        
#         pbar.set_description(f"Return: {returns}")

#         obs = next_obs
#         if done:
#             if info["succeed"]:
#                 for transition in trajectory:
#                     transitions.append(copy.deepcopy(transition))
#                 success_count += 1
#                 pbar.update(1)
#             trajectory = []
#             returns = 0
#             obs, info = env.reset()
            
#     if not os.path.exists("./demo_data"):
#         os.makedirs("./demo_data")
#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
#     with open(file_name, "wb") as f:
#         pkl.dump(transitions, f)
#         print(f"saved {success_needed} demos to {file_name}")

# if __name__ == "__main__":
#     app.run(main)
