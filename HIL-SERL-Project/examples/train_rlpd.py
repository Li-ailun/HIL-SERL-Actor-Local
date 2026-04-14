#!/usr/bin/env python3
#人类在环监督+RL


import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

# ==============================================================
# 🧠 SERL 强化学习算法核心组件
# ==============================================================
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.launcher import make_trainer_config

# ==============================================================
# 📡 分布式训练核心架构 (AgentLace RPC 通信框架)
# ==============================================================
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# ==============================================================
# 🔥 核心路径配置 (保持纯粹的解耦与安全引用)
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 💡 直调星海图专属定制的环境和配置
from examples.galaxea_task.usb_pick_insertion.wrapper import make_env
from examples.galaxea_task.usb_pick_insertion.config import env_config 

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")

# 🌟 分布式架构核心开关
flags.DEFINE_boolean("learner", False, "是否启动为 Learner (后台大本营：负责在 GPU 上疯狂更新神经网络参数).")
flags.DEFINE_boolean("actor", False, "是否启动为 Actor (前线打工人：负责控制机械臂、收集数据并发送给 Learner).")
flags.DEFINE_string("ip", "localhost", "Learner 的 IP 地址 (如果 Actor 和 Learner 在同一台电脑上，保持 localhost).")

# 路径配置
flags.DEFINE_multi_string("demo_path", None, "存放初始专家数据的绝对路径 (供 Learner 加载).")
flags.DEFINE_string("checkpoint_path", "./rlpd_checkpoints", "保存网络权重和回放缓冲池的路径.")

# 评估(推理)模式专属配置
flags.DEFINE_integer("eval_checkpoint_step", 0, "如果要评估特定步数的模型，填入步数 (大于0时进入纯推理模式).")
flags.DEFINE_integer("eval_n_trajs", 0, "评估模式下测试的回合数.")
flags.DEFINE_boolean("save_video", False, "是否保存评估录像.")
flags.DEFINE_boolean("debug", False, "Debug mode (关闭 Wandb 日志上传).")

# JAX 多设备 (GPU/TPU) 配置
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

def print_green(x):
    print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))

# ============================================================================
# 🎭 Actor 逻辑：前线执行者 (负责动作采样、环境交互、记录数据)
# ============================================================================
def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    当 "--actor=True" 时运行。
    它是连接真实物理世界（星海图机器人）和数字世界的桥梁。
    """
    # ---------------------------------------------------------
    # 🔍 分支 1：纯评估模式 (Evaluation Mode)
    # 只有当传入了 --eval_checkpoint_step > 0 时才会触发
    # ---------------------------------------------------------
    if FLAGS.eval_checkpoint_step:
        print_green(f"🚀 启动纯评估推理模式，测试步数: {FLAGS.eval_checkpoint_step}")
        success_counter = 0
        time_list = []

        # 从硬盘恢复指定的模型权重
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        # 开始连续测试 FLAGS.eval_n_trajs 个回合
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                # 采样动作：注意 argmax=False，允许适当的随机性，也可以改为 True 实现完全贪婪策略
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                # 步进真实环境
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                # 评估结算
                if done:
                    if reward: # 视觉分类器给出了 1 分 (成功)
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print_green(f"✅ 第 {episode + 1} 回合成功！耗时: {dt:.2f} 秒")
                    else:
                        print_yellow(f"❌ 第 {episode + 1} 回合失败。")
                        
                    success_counter += int(reward)
                    print(f"📊 累计成绩: 成功 {success_counter} / 总计 {episode + 1}")

        print_green(f"🏆 最终评估成功率: {success_counter / FLAGS.eval_n_trajs:.2%}")
        if time_list:
            print_green(f"⏱️ 成功任务平均耗时: {np.mean(time_list):.2f} 秒")
        return  # 评估结束，直接退出函数，不进行后续的训练收集逻辑
    
    # ---------------------------------------------------------
    # 🚀 分支 2：训练收集模式 (Data Collection Mode)
    # ---------------------------------------------------------
    # 尝试从本地寻找最新的 buffer 文件，推断当前跑到了第几步，实现断点续传
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")) and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    # Actor 维护两个发送队列：
    # "actor_env"：存放 AI 自己摸索的所有在线经验。
    # "actor_env_intvn"：存放人类 VR 接管期间的高权重专家纠错经验。
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    # 初始化 RPC 客户端，连接到 Learner 的 IP 地址
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        {}, # 空字典占位，替代原版的 make_trainer_config()
        #make_trainer_config(), # <--- 恢复官方的 config 生成器
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # 定义回调：当客户端收到 Learner 发来的新模型参数时，立刻更新 Actor 自己的大脑
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    # 注册监听，保持大脑时刻最新
    client.recv_network_callback(update_params)

    transitions = []       # 内存缓存区 (在线数据)
    demo_transitions = []  # 内存缓存区 (干预数据)

    obs, _ = env.reset()
    done = False
    timer = Timer()
    running_return = 0.0
    
    # 干预状态追踪
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, env_config.max_steps), dynamic_ncols=True, desc="🎭 Actor Loop")
    for step in pbar:
        timer.tick("total")

        # 1. 动作采样
        with timer.context("sample_actions"):
            # 在极其早期的探索阶段，完全使用随机动作填补基础数据
            if step < env_config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False, # 训练时使用概率分布采样以保持探索性
                )
                actions = np.asarray(jax.device_get(actions))

        # 2. 环境交互
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            
            # 清理无关信息，防止序列化时数据包过大
            if "left" in info: info.pop("left")
            if "right" in info: info.pop("right")

            # 🌟 人类 VR 物理离合器：用专家动作覆盖 AI 动作
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            
            # 打包当前帧数据
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # 夹爪微调训练的专属防抽搐奖励记录
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
                
            # 3. 发送数据！将这一帧推给 Learner 端
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition)) # 留底备份
            
            if already_intervened:
                # 只有被人类干预过的数据，才有资格进入贵宾通道 (intvn_data_store)
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            
            # 4. 回合结束清理与同步
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  
                client.request("send-stats", stats) # 汇报给大本营
                pbar.set_description(f"Last Return: {running_return}")
                
                # 重置变量
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                
                # 强制触发一次网络事件，并重置物理环境
                client.update()
                obs, _ = env.reset()

        # 5. 定期落盘保护 (Backup)
        # 防止训练了几天几夜突然断电，把内存在线数据存入硬盘
        if step > 0 and env_config.buffer_period > 0 and step % env_config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        # 定期汇报 Actor 的耗时情况
        if step % env_config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

# ============================================================================
# 🧠 Learner 逻辑：坐镇大本营，疯狂炼丹 (在 GPU/TPU 上更新权重)
# ============================================================================
def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    当 "--learner=True" 时运行。
    它是整个系统的智力核心，负责融合所有数据，通过反向传播计算梯度。
    """
    # 获取最新的断点步数
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            start_step = int(os.path.basename(latest)[11:]) + 1
    step = start_step

    # 日志回调函数
    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {} 

    # 1. 启动 RPC 服务器
    server = TrainerServer({}, request_callback=stats_callback)
    #server = TrainerServer(make_trainer_config(), request_callback=stats_callback)# <--- 恢复官方的 config 生成器
    # 将服务端的两个大池子和 Actor 传来的名称进行绑定
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # 2. 等待初始数据填充 (Warmup)
    # 在开始算梯度之前，池子里必须有足够的数据，否则会过拟合。这里卡住等待 Actor 塞数据。
    pbar = tqdm.tqdm(
        total=env_config.training_starts,
        initial=len(replay_buffer),
        desc="⏳ 等待 Actor 填充在线探索数据...",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < env_config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n) 
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n) 
    pbar.close()

    # 将初始模型权重发布出去，喂给干等着的 Actor
    server.publish_network(agent.state.params)
    print_green("✅ 初始网络权重已发布给 Actor！")

    # 3. 🌟 构建 RLPD 数据管道 (50% 随机，50% 专家)
    # 这是 RLPD 算法能够在单卡上高效学习复杂灵巧操作的核心原因。
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": env_config.batch_size // 2, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": env_config.batch_size // 2, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )

    timer = Timer()
    
    # 配置要更新的子网络 (SAC由 Actor网络 和 Critic网络 组成)
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        # 处理带有自定义夹爪打分的混合架构
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    print_green("🔥 Learner 训练引擎全开！")
    # ==========================================================
    # ⚙️ 核心反向传播循环
    # ==========================================================
    for step in tqdm.tqdm(range(start_step, env_config.max_steps), dynamic_ncols=True, desc="🧠 Learner"):
        
        # 优化策略：Critic(价值评估) 更新 n-1 次，Actor(动作选择) 才更新 1 次。
        # 减轻 GPU 与 CPU 之间的 Batch 拷贝耗时。
        for critic_step in range(env_config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                # 拼接：在线数据 + 专家数据
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch, networks_to_update=train_critic_networks_to_update,
                )

        # 全网络联合更新
        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch, networks_to_update=train_networks_to_update,
            )
            
        # 定期将变聪明的网络权重发布出去
        if step > 0 and step % (env_config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent) # 确保计算全部完成
            server.publish_network(agent.state.params)

        # 记录日志
        if step % env_config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        # 保存断点
        if step > 0 and env_config.checkpoint_period and step % env_config.checkpoint_period == 0:
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )

# ============================================================================
# 🚦 主控函数 (程序入口点)
# ============================================================================
def main(_):
    # 直接对接你定义的任务配置
    global config # 避免作用域问题
    config = env_config

    assert config.batch_size % num_devices == 0, "Batch_size 必须能被计算设备整除"
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # 实例化环境结构
    # 若为 Learner，则只起一个空壳环境用于获取维度，不调用真实相机 (fake_env=True)
    env = make_env(
        reward_classifier_model=None, 
        use_manual_reward=False, 
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    # 动态匹配 SAC 变体结构
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"未知设置模式: {config.setup_mode}")

    # 将网络分发到 JAX 并行计算单元
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    # 断点续传拦截
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path) and glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*")):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt:
            input(f"⚠️ 检测到已存在的 Checkpoint ({latest_ckpt})。按 Enter 键恢复训练...")
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state)
            agent = agent.replace(state=ckpt)
            print_green(f"✅ 成功加载历史权重。")

    # ---------------------------------------------------------
    # 启动大本营 (Learner)
    # ---------------------------------------------------------
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        
        # 构建超大内存池
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space, env.action_space, capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys, include_grasp_penalty=include_grasp_penalty,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space, env.action_space, capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys, include_grasp_penalty=include_grasp_penalty,
        )

        wandb_logger = make_wandb_logger(project="hil-serl", description=FLAGS.exp_name, debug=FLAGS.debug)

        # 将最初用 record_demos.py 辛苦录制的 .pkl 数据一次性塞进 demo_buffer
        assert FLAGS.demo_path is not None, "❌ Learner 必须通过 --demo_path 传入初始的 demo_data 路径！"
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
                    
        print_green(f"📦 初始专家数据池加载完成，大小: {len(demo_buffer)}")

        # 从硬盘恢复之前跑到一半的 Buffer 数据 (如果程序崩溃重启)
        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(f"♻️ 成功恢复在线历史数据. 当前大小: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"♻️ 成功恢复人工干预历史数据. 当前大小: {len(demo_buffer)}")

        print_green("🚀 启动 Learner 服务中心...")
        learner(sampling_rng, agent, replay_buffer, demo_buffer, wandb_logger)

    # ---------------------------------------------------------
    # 启动前线采集器 (Actor)
    # ---------------------------------------------------------
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        
        # Actor 端的队列 (只负责搬运，不需要几 GB 的内存)
        data_store = QueuedDataStore(50000)  
        intvn_data_store = QueuedDataStore(50000)

        print_green("🚀 启动 Actor 客户端...")
        actor(agent, data_store, intvn_data_store, env, sampling_rng)

    else:
        raise NotImplementedError("❌ 你必须在命令行指定 --learner=True 或者 --actor=True")

if __name__ == "__main__":
    app.run(main)


# import glob
# import time
# import jax
# import jax.numpy as jnp
# import numpy as np
# import tqdm
# from absl import app, flags
# from flax.training import checkpoints
# import os
# import copy
# import pickle as pkl
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
# from natsort import natsorted

# from serl_launcher.agents.continuous.sac import SACAgent
# from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
# from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
# from serl_launcher.utils.timer_utils import Timer
# from serl_launcher.utils.train_utils import concat_batches

# from agentlace.trainer import TrainerServer, TrainerClient
# from agentlace.data.data_store import QueuedDataStore

# from serl_launcher.utils.launcher import (
#     make_sac_pixel_agent,
#     make_sac_pixel_agent_hybrid_single_arm,
#     make_sac_pixel_agent_hybrid_dual_arm,
#     make_trainer_config,
#     make_wandb_logger,
# )
# from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# from experiments.mappings import CONFIG_MAPPING

# FLAGS = flags.FLAGS

# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("seed", 42, "Random seed.")
# flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
# flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
# flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
# flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
# flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
# flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
# flags.DEFINE_boolean("save_video", False, "Save video.")

# flags.DEFINE_boolean(
#     "debug", False, "Debug mode."
# )  # debug mode will disable wandb logging


# devices = jax.local_devices()
# num_devices = len(devices)
# sharding = jax.sharding.PositionalSharding(devices)


# def print_green(x):
#     return print("\033[92m {}\033[00m".format(x))


# ##############################################################################


# def actor(agent, data_store, intvn_data_store, env, sampling_rng):
#     """
#     This is the actor loop, which runs when "--actor" is set to True.
#     """
#     if FLAGS.eval_checkpoint_step:
#         success_counter = 0
#         time_list = []

#         ckpt = checkpoints.restore_checkpoint(
#             os.path.abspath(FLAGS.checkpoint_path),
#             agent.state,
#             step=FLAGS.eval_checkpoint_step,
#         )
#         agent = agent.replace(state=ckpt)

#         for episode in range(FLAGS.eval_n_trajs):
#             obs, _ = env.reset()
#             done = False
#             start_time = time.time()
#             while not done:
#                 sampling_rng, key = jax.random.split(sampling_rng)
#                 actions = agent.sample_actions(
#                     observations=jax.device_put(obs),
#                     argmax=False,
#                     seed=key
#                 )
#                 actions = np.asarray(jax.device_get(actions))

#                 next_obs, reward, done, truncated, info = env.step(actions)
#                 obs = next_obs

#                 if done:
#                     if reward:
#                         dt = time.time() - start_time
#                         time_list.append(dt)
#                         print(dt)

#                     success_counter += reward
#                     print(reward)
#                     print(f"{success_counter}/{episode + 1}")

#         print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
#         print(f"average time: {np.mean(time_list)}")
#         return  # after done eval, return and exit
    
#     start_step = (
#         int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
#         if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
#         else 0
#     )

#     datastore_dict = {
#         "actor_env": data_store,
#         "actor_env_intvn": intvn_data_store,
#     }

#     client = TrainerClient(
#         "actor_env",
#         FLAGS.ip,
#         make_trainer_config(),
#         data_stores=datastore_dict,
#         wait_for_server=True,
#         timeout_ms=3000,
#     )

#     # Function to update the agent with new params
#     def update_params(params):
#         nonlocal agent
#         agent = agent.replace(state=agent.state.replace(params=params))

#     client.recv_network_callback(update_params)

#     transitions = []
#     demo_transitions = []

#     obs, _ = env.reset()
#     done = False

#     # training loop
#     timer = Timer()
#     running_return = 0.0
#     already_intervened = False
#     intervention_count = 0
#     intervention_steps = 0

#     pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
#     for step in pbar:
#         timer.tick("total")

#         with timer.context("sample_actions"):
#             if step < config.random_steps:
#                 actions = env.action_space.sample()
#             else:
#                 sampling_rng, key = jax.random.split(sampling_rng)
#                 actions = agent.sample_actions(
#                     observations=jax.device_put(obs),
#                     seed=key,
#                     argmax=False,
#                 )
#                 actions = np.asarray(jax.device_get(actions))

#         # Step environment
#         with timer.context("step_env"):

#             next_obs, reward, done, truncated, info = env.step(actions)
#             if "left" in info:
#                 info.pop("left")
#             if "right" in info:
#                 info.pop("right")

#             # override the action with the intervention action
#             if "intervene_action" in info:
#                 actions = info.pop("intervene_action")
#                 intervention_steps += 1
#                 if not already_intervened:
#                     intervention_count += 1
#                 already_intervened = True
#             else:
#                 already_intervened = False

#             running_return += reward
#             transition = dict(
#                 observations=obs,
#                 actions=actions,
#                 next_observations=next_obs,
#                 rewards=reward,
#                 masks=1.0 - done,
#                 dones=done,
#             )
#             if 'grasp_penalty' in info:
#                 transition['grasp_penalty']= info['grasp_penalty']
#             data_store.insert(transition)
#             transitions.append(copy.deepcopy(transition))
#             if already_intervened:
#                 intvn_data_store.insert(transition)
#                 demo_transitions.append(copy.deepcopy(transition))

#             obs = next_obs
#             if done or truncated:
#                 info["episode"]["intervention_count"] = intervention_count
#                 info["episode"]["intervention_steps"] = intervention_steps
#                 stats = {"environment": info}  # send stats to the learner to log
#                 client.request("send-stats", stats)
#                 pbar.set_description(f"last return: {running_return}")
#                 running_return = 0.0
#                 intervention_count = 0
#                 intervention_steps = 0
#                 already_intervened = False
#                 client.update()
#                 obs, _ = env.reset()

#         if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
#             # dump to pickle file
#             buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
#             demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
#             if not os.path.exists(buffer_path):
#                 os.makedirs(buffer_path)
#             if not os.path.exists(demo_buffer_path):
#                 os.makedirs(demo_buffer_path)
#             with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
#                 pkl.dump(transitions, f)
#                 transitions = []
#             with open(
#                 os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
#             ) as f:
#                 pkl.dump(demo_transitions, f)
#                 demo_transitions = []

#         timer.tock("total")

#         if step % config.log_period == 0:
#             stats = {"timer": timer.get_average_times()}
#             client.request("send-stats", stats)


# ##############################################################################


# def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
#     """
#     The learner loop, which runs when "--learner" is set to True.
#     """
#     start_step = (
#         int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
#         + 1
#         if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
#         else 0
#     )
#     step = start_step

#     def stats_callback(type: str, payload: dict) -> dict:
#         """Callback for when server receives stats request."""
#         assert type == "send-stats", f"Invalid request type: {type}"
#         if wandb_logger is not None:
#             wandb_logger.log(payload, step=step)
#         return {}  # not expecting a response

#     # Create server
#     server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
#     server.register_data_store("actor_env", replay_buffer)
#     server.register_data_store("actor_env_intvn", demo_buffer)
#     server.start(threaded=True)

#     # Loop to wait until replay_buffer is filled
#     pbar = tqdm.tqdm(
#         total=config.training_starts,
#         initial=len(replay_buffer),
#         desc="Filling up replay buffer",
#         position=0,
#         leave=True,
#     )
#     while len(replay_buffer) < config.training_starts:
#         pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
#         time.sleep(1)
#     pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
#     pbar.close()

#     # send the initial network to the actor
#     server.publish_network(agent.state.params)
#     print_green("sent initial network to actor")

#     # 50/50 sampling from RLPD, half from demo and half from online experience
#     replay_iterator = replay_buffer.get_iterator(
#         sample_args={
#             "batch_size": config.batch_size // 2,
#             "pack_obs_and_next_obs": True,
#         },
#         device=sharding.replicate(),
#     )
#     demo_iterator = demo_buffer.get_iterator(
#         sample_args={
#             "batch_size": config.batch_size // 2,
#             "pack_obs_and_next_obs": True,
#         },
#         device=sharding.replicate(),
#     )

#     # wait till the replay buffer is filled with enough data
#     timer = Timer()
    
#     if isinstance(agent, SACAgent):
#         train_critic_networks_to_update = frozenset({"critic"})
#         train_networks_to_update = frozenset({"critic", "actor", "temperature"})
#     else:
#         train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
#         train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

#     for step in tqdm.tqdm(
#         range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
#     ):
#         # run n-1 critic updates and 1 critic + actor update.
#         # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
#         for critic_step in range(config.cta_ratio - 1):
#             with timer.context("sample_replay_buffer"):
#                 batch = next(replay_iterator)
#                 demo_batch = next(demo_iterator)
#                 batch = concat_batches(batch, demo_batch, axis=0)

#             with timer.context("train_critics"):
#                 agent, critics_info = agent.update(
#                     batch,
#                     networks_to_update=train_critic_networks_to_update,
#                 )

#         with timer.context("train"):
#             batch = next(replay_iterator)
#             demo_batch = next(demo_iterator)
#             batch = concat_batches(batch, demo_batch, axis=0)
#             agent, update_info = agent.update(
#                 batch,
#                 networks_to_update=train_networks_to_update,
#             )
#         # publish the updated network
#         if step > 0 and step % (config.steps_per_update) == 0:
#             agent = jax.block_until_ready(agent)
#             server.publish_network(agent.state.params)

#         if step % config.log_period == 0 and wandb_logger:
#             wandb_logger.log(update_info, step=step)
#             wandb_logger.log({"timer": timer.get_average_times()}, step=step)

#         if (
#             step > 0
#             and config.checkpoint_period
#             and step % config.checkpoint_period == 0
#         ):
#             checkpoints.save_checkpoint(
#                 os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
#             )


# ##############################################################################


# def main(_):
#     global config
#     config = CONFIG_MAPPING[FLAGS.exp_name]()

#     assert config.batch_size % num_devices == 0
#     # seed
#     rng = jax.random.PRNGKey(FLAGS.seed)
#     rng, sampling_rng = jax.random.split(rng)

#     assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
#     env = config.get_environment(
#         fake_env=FLAGS.learner,
#         save_video=FLAGS.save_video,
#         classifier=True,
#     )
#     env = RecordEpisodeStatistics(env)

#     rng, sampling_rng = jax.random.split(rng)
    
#     if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
#         agent: SACAgent = make_sac_pixel_agent(
#             seed=FLAGS.seed,
#             sample_obs=env.observation_space.sample(),
#             sample_action=env.action_space.sample(),
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = False
#     elif config.setup_mode == 'single-arm-learned-gripper':
#         agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
#             seed=FLAGS.seed,
#             sample_obs=env.observation_space.sample(),
#             sample_action=env.action_space.sample(),
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = True
#     elif config.setup_mode == 'dual-arm-learned-gripper':
#         agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
#             seed=FLAGS.seed,
#             sample_obs=env.observation_space.sample(),
#             sample_action=env.action_space.sample(),
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = True
#     else:
#         raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

#     # replicate agent across devices
#     # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
#     agent = jax.device_put(
#         jax.tree_map(jnp.array, agent), sharding.replicate()
#     )

#     if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
#         input("Checkpoint path already exists. Press Enter to resume training.")
#         ckpt = checkpoints.restore_checkpoint(
#             os.path.abspath(FLAGS.checkpoint_path),
#             agent.state,
#         )
#         agent = agent.replace(state=ckpt)
#         ckpt_number = os.path.basename(
#             checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
#         )[11:]
#         print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

#     def create_replay_buffer_and_wandb_logger():
#         replay_buffer = MemoryEfficientReplayBufferDataStore(
#             env.observation_space,
#             env.action_space,
#             capacity=config.replay_buffer_capacity,
#             image_keys=config.image_keys,
#             include_grasp_penalty=include_grasp_penalty,
#         )
#         # set up wandb and logging
#         wandb_logger = make_wandb_logger(
#             project="hil-serl",
#             description=FLAGS.exp_name,
#             debug=FLAGS.debug,
#         )
#         return replay_buffer, wandb_logger

#     if FLAGS.learner:
#         sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
#         replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
#         demo_buffer = MemoryEfficientReplayBufferDataStore(
#             env.observation_space,
#             env.action_space,
#             capacity=config.replay_buffer_capacity,
#             image_keys=config.image_keys,
#             include_grasp_penalty=include_grasp_penalty,
#         )

#         assert FLAGS.demo_path is not None
#         for path in FLAGS.demo_path:
#             with open(path, "rb") as f:
#                 transitions = pkl.load(f)
#                 for transition in transitions:
#                     if 'infos' in transition and 'grasp_penalty' in transition['infos']:
#                         transition['grasp_penalty'] = transition['infos']['grasp_penalty']
#                     demo_buffer.insert(transition)
#         print_green(f"demo buffer size: {len(demo_buffer)}")
#         print_green(f"online buffer size: {len(replay_buffer)}")

#         if FLAGS.checkpoint_path is not None and os.path.exists(
#             os.path.join(FLAGS.checkpoint_path, "buffer")
#         ):
#             for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
#                 with open(file, "rb") as f:
#                     transitions = pkl.load(f)
#                     for transition in transitions:
#                         replay_buffer.insert(transition)
#             print_green(
#                 f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
#             )

#         if FLAGS.checkpoint_path is not None and os.path.exists(
#             os.path.join(FLAGS.checkpoint_path, "demo_buffer")
#         ):
#             for file in glob.glob(
#                 os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
#             ):
#                 with open(file, "rb") as f:
#                     transitions = pkl.load(f)
#                     for transition in transitions:
#                         demo_buffer.insert(transition)
#             print_green(
#                 f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
#             )

#         # learner loop
#         print_green("starting learner loop")
#         learner(
#             sampling_rng,
#             agent,
#             replay_buffer,
#             demo_buffer=demo_buffer,
#             wandb_logger=wandb_logger,
#         )

#     elif FLAGS.actor:
#         sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
#         data_store = QueuedDataStore(50000)  # the queue size on the actor
#         intvn_data_store = QueuedDataStore(50000)

#         # actor loop
#         print_green("starting actor loop")
#         actor(
#             agent,
#             data_store,
#             intvn_data_store,
#             env,
#             sampling_rng,
#         )

#     else:
#         raise NotImplementedError("Must be either a learner or an actor")


# if __name__ == "__main__":
#     app.run(main)
