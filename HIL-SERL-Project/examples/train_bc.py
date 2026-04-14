#!/usr/bin/env python3
#对录制的数据进行第一步处理：bc克隆基本动作，保证强化学习初步试探的动作不会太夸张

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# ==============================================================
# 🔥 核心路径配置 (彻底解决 ModuleNotFoundError)
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 💡 精准导入你任务目录下的专属构建器和配置
from examples.galaxea_task.usb_pick_insertion.wrapper import make_env
# 假设你在该任务目录下也有一个 config 文件定义了超参数 (如 image_keys, batch_size 等)
from examples.galaxea_task.usb_pick_insertion.config import env_config 

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", "./bc_checkpoints", "Path to save checkpoints.")
# eval_n_trajs 决定模式：
# 如果为 0，则执行训练模式；如果大于 0，则进入推理(验证)模式，测试训好的模型。
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")

flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

# JAX 多设备 (GPU/TPU) 配置
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

def print_green(x):
    print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))

##############################################################################

def eval(env, bc_agent: BCAgent, sampling_rng):
    """
    模型验证推理循环 (Actor Loop)
    工作原理：加载训好的 BC 权重，根据当前环境状态，让神经网络推断出动作，并操控机械臂。
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        
        while not done:
            rng, key = jax.random.split(sampling_rng)
            # 核心：调用神经网络生成当前状态的最优动作
            actions = bc_agent.sample_actions(observations=obs, seed=key)
            # 将 JAX 数组转回 CPU 的 NumPy 数组发给环境
            actions = np.asarray(jax.device_get(actions))
            
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            
            if done:
                if info.get("succeed", False):
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print_green(f"✅ 第 {episode + 1} 回合成功！耗时: {dt:.2f}s")
                    success_counter += 1
                else:
                    print_yellow(f"❌ 第 {episode + 1} 回合失败。")
                print(f"📊 当前成功率: {success_counter}/{episode + 1}")

    print_green(f"🏆 最终评估成功率: {success_counter / FLAGS.eval_n_trajs:.2%}")
    if time_list:
        print_green(f"⏱️ 成功任务平均耗时: {np.mean(time_list):.2f}s")

##############################################################################

def train(bc_agent: BCAgent, bc_replay_buffer, config, wandb_logger=None):
    """
    行为克隆训练循环 (Learner Loop)
    工作原理：从刚才录制的 .pkl 数据中不断抽取 batch，计算 Loss 并更新神经网络权重。
    """
    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    
    # 开始 BC 预训练
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="🧠 BC Pretraining",
    ):
        # 取出一个批次的数据
        batch = next(bc_replay_iterator)
        
        # 核心：反向传播更新网络参数
        bc_agent, bc_update_info = bc_agent.update(batch)
        
        # 记录日志到 Wandb (监控 Loss 下降情况)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
            
        # 训练快结束时，高频保存最后 5 个 Checkpoint
        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            if not os.path.exists(FLAGS.bc_checkpoint_path):
                os.makedirs(FLAGS.bc_checkpoint_path)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
            )
            
    print_green("✅ BC 预训练完成并已保存 Checkpoint！")

##############################################################################

def main(_):
    # 使用你本地 config.py 里定义的超参
    config = env_config
    assert config.batch_size % num_devices == 0, "Batch size 必须能被 GPU 数量整除！"

    eval_mode = FLAGS.eval_n_trajs > 0
    
    # 实例化环境
    # 如果是训练模式，我们不需要真的启动机械臂(fake_env=True)，只需要拿到 action 和 obs 的 shape 即可
    env = make_env(
        reward_classifier_model=None, 
        use_manual_reward=False, # 训练 BC 时不需要打分
        # 注意：此处假定 make_env 内部对于 fake_env 的支持已适配，
        # 如果你的底层还不支持假环境，此处可能需硬编码传入 fake_env 标识。
    )
    env = RecordEpisodeStatistics(env)

    # 实例化 BC 智能体 (也就是你的演员网络 Actor Network)
    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # 将网络参数推送到 GPU/TPU 设备上
    bc_agent = jax.device_put(
        jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    )

    # ==========================================================
    # 🔄 训练模式 (Training)
    # ==========================================================
    if not eval_mode:
        # 检查是否已存在同名断点，防止覆盖
        if os.path.isdir(os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")):
            print_yellow("⚠️ 警告：目标 checkpoint 似乎已存在，可能会被覆盖！")

        # 初始化存放专家数据的内存池
        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # 设置 WandB 日志记录，方便你在网页端看 Loss 曲线
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        # 🔍 核心：寻找刚才录制的 demo_data
        demo_dir = os.path.join(os.path.dirname(__file__), "demo_data")
        demo_path = glob.glob(os.path.join(demo_dir, "*.pkl"))
        
        # 修复原版隐蔽 bug：必须显式判断列表长度
        assert len(demo_path) > 0, f"❌ 找不到专家数据！请确保 {demo_dir} 目录下有 .pkl 文件。"

        # 将所有的 .pkl 文件加载进内存池
        for path in demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    # 💡 官方极客设计：自动剔除发呆帧。
                    # 如果这帧动作的 L2 范数为 0，说明手没动，直接丢弃，不污染神经网络。
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
                        
        print_green(f"📦 BC 专家数据池加载完毕，有效帧数: {len(bc_replay_buffer)}")

        # 启动反向传播！
        print_green("🚀 开始执行 Learner Loop (网络训练)...")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    # ==========================================================
    # 🎮 推理验证模式 (Evaluation)
    # 当 --eval_n_trajs 大于 0 时进入此分支
    # ==========================================================
    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        # 从保存的路径中恢复训练好的权重
        print(f"⏳ 正在加载权重: {FLAGS.bc_checkpoint_path}")
        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("🚀 权重加载成功，进入 Actor Loop (实车验证)...")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )

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
# import pickle as pkl
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

# from serl_launcher.agents.continuous.bc import BCAgent

# from serl_launcher.utils.launcher import (
#     make_bc_agent,
#     make_trainer_config,
#     make_wandb_logger,
# )
# from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# from experiments.mappings import CONFIG_MAPPING
# from experiments.config import DefaultTrainingConfig
# FLAGS = flags.FLAGS

# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("seed", 42, "Random seed.")
# flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
# flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
# flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
# flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")


# flags.DEFINE_boolean(
#     "debug", False, "Debug mode."
# )  # debug mode will disable wandb logging


# devices = jax.local_devices()
# num_devices = len(devices)
# sharding = jax.sharding.PositionalSharding(devices)


# def print_green(x):
#     return print("\033[92m {}\033[00m".format(x))


# def print_yellow(x):
#     return print("\033[93m {}\033[00m".format(x))


# ##############################################################################

# def eval(
#     env,
#     bc_agent: BCAgent,
#     sampling_rng,
# ):
#     """
#     This is the actor loop, which runs when "--actor" is set to True.
#     """
#     success_counter = 0
#     time_list = []
#     for episode in range(FLAGS.eval_n_trajs):
#         obs, _ = env.reset()
#         done = False
#         start_time = time.time()
#         while not done:
#             rng, key = jax.random.split(sampling_rng)

#             actions = bc_agent.sample_actions(observations=obs, seed=key)
#             actions = np.asarray(jax.device_get(actions))
#             next_obs, reward, done, truncated, info = env.step(actions)
#             obs = next_obs
#             if done:
#                 if reward:
#                     dt = time.time() - start_time
#                     time_list.append(dt)
#                     print(dt)
#                 success_counter += reward
#                 print(reward)
#                 print(f"{success_counter}/{episode + 1}")

#     print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
#     print(f"average time: {np.mean(time_list)}")


# ##############################################################################


# def train(
#     bc_agent: BCAgent,
#     bc_replay_buffer,
#     config: DefaultTrainingConfig,
#     wandb_logger=None,
# ):

#     bc_replay_iterator = bc_replay_buffer.get_iterator(
#         sample_args={
#             "batch_size": config.batch_size,
#             "pack_obs_and_next_obs": False,
#         },
#         device=sharding.replicate(),
#     )
    
#     # Pretrain BC policy to get started
#     for step in tqdm.tqdm(
#         range(FLAGS.train_steps),
#         dynamic_ncols=True,
#         desc="bc_pretraining",
#     ):
#         batch = next(bc_replay_iterator)
#         bc_agent, bc_update_info = bc_agent.update(batch)
#         if step % config.log_period == 0 and wandb_logger:
#             wandb_logger.log({"bc": bc_update_info}, step=step)
#         if step > FLAGS.train_steps - 100 and step % 10 == 0:
#             checkpoints.save_checkpoint(
#                 os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
#             )
#     print_green("bc pretraining done and saved checkpoint")


# ##############################################################################


# def main(_):
#     config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

#     assert config.batch_size % num_devices == 0
#     assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
#     eval_mode = FLAGS.eval_n_trajs > 0
#     env = config.get_environment(
#         fake_env=not eval_mode,
#         save_video=FLAGS.save_video,
#         classifier=True,
#     )
#     env = RecordEpisodeStatistics(env)

#     bc_agent: BCAgent = make_bc_agent(
#         seed=FLAGS.seed,
#         sample_obs=env.observation_space.sample(),
#         sample_action=env.action_space.sample(),
#         image_keys=config.image_keys,
#         encoder_type=config.encoder_type,
#     )

#     # replicate agent across devices
#     # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
#     bc_agent: BCAgent = jax.device_put(
#         jax.tree_map(jnp.array, bc_agent), sharding.replicate()
#     )

#     if not eval_mode:
#         assert not os.path.isdir(
#             os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
#         )

#         bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
#             env.observation_space,
#             env.action_space,
#             capacity=config.replay_buffer_capacity,
#             image_keys=config.image_keys,
#         )

#         # set up wandb and logging
#         wandb_logger = make_wandb_logger(
#             project="hil-serl",
#             description=FLAGS.exp_name,
#             debug=FLAGS.debug,
#         )

#         demo_path = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        
#         assert demo_path is not []

#         for path in demo_path:
#             with open(path, "rb") as f:
#                 transitions = pkl.load(f)
#                 for transition in transitions:
#                     if np.linalg.norm(transition['actions']) > 0.0:
#                         bc_replay_buffer.insert(transition)
#         print(f"bc replay buffer size: {len(bc_replay_buffer)}")

#         # learner loop
#         print_green("starting learner loop")
#         train(
#             bc_agent=bc_agent,
#             bc_replay_buffer=bc_replay_buffer,
#             wandb_logger=wandb_logger,
#             config=config,
#         )

#     else:
#         rng = jax.random.PRNGKey(FLAGS.seed)
#         sampling_rng = jax.device_put(rng, sharding.replicate())

#         bc_ckpt = checkpoints.restore_checkpoint(
#             FLAGS.bc_checkpoint_path,
#             bc_agent.state,
#         )
#         bc_agent = bc_agent.replace(state=bc_ckpt)

#         print_green("starting actor loop")
#         eval(
#             env=env,
#             bc_agent=bc_agent,
#             sampling_rng=sampling_rng,
#         )


# if __name__ == "__main__":
#     app.run(main)
