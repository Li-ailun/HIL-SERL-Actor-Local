#!/usr/bin/env python3
#对录制的数据进行第一步处理：bc克隆基本动作，保证强化学习初步试探的动作不会太夸张

#训练：
#python train_bc.py --exp_name=galaxea_usb_insertion
#评估：
#python train_bc.py --exp_name=galaxea_usb_insertion --eval_n_trajs=10

#!/usr/bin/env python3
# 对录制的数据进行第一步处理：BC 克隆基本动作，保证强化学习初步试探的动作不会太夸张
# 训练阶段：纯离线，不需要真机
# 评估阶段：需要真实环境

#python train_bc.py --exp_name=galaxea_usb_insertion --eval_n_trajs=0（0表示离线训练模式）
#python train_bc.py --exp_name=galaxea_usb_insertion --eval_n_trajs=10（10表示在线评估模式，评估10个episode）

#完整离线训练参考指令：
# python train_bc.py \
#   --exp_name=galaxea_usb_insertion \
#   --eval_n_trajs=0 \
#   --train_steps=20000 \
#   --bc_checkpoint_path=./bc_checkpoints


#完整在线评估参考指令：
# python train_bc.py \
#   --exp_name=galaxea_usb_insertion \
#   --eval_n_trajs=5 \
#   --bc_checkpoint_path=./bc_checkpoints \
#   --save_video=False

#目前可cpu：python train_bc.py --exp_name=galaxea_usb_insertion --eval_n_trajs=0 --debug=True


import os
import sys
import glob
import time
import pickle as pkl

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium import spaces
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

# ==============================================================
# 🔥 核心路径配置
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERL_LAUNCHER_ROOT = os.path.join(PROJECT_ROOT, "serl_launcher")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SERL_LAUNCHER_ROOT not in sys.path:
    sys.path.insert(0, SERL_LAUNCHER_ROOT)

# ==============================================================
# SERL / 任务导入
# ==============================================================
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import make_bc_agent, make_wandb_logger
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore


from examples.galaxea_task.usb_pick_insertion.config import env_config

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", "./bc_checkpoints", "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

# JAX 多设备配置
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))


# ==============================================================
# 训练阶段：从 demo_data 推断 space / sample，彻底绕开真实环境
# ==============================================================

def infer_space_from_value(x):
    """递归从样本值推断 gymnasium space。"""
    if isinstance(x, dict):
        return spaces.Dict({k: infer_space_from_value(v) for k, v in x.items()})

    arr = np.asarray(x)

    if arr.dtype == np.uint8:
        return spaces.Box(low=0, high=255, shape=arr.shape, dtype=np.uint8)
    elif np.issubdtype(arr.dtype, np.bool_):
        return spaces.Box(low=0, high=1, shape=arr.shape, dtype=np.bool_)
    elif np.issubdtype(arr.dtype, np.integer):
        return spaces.Box(
            low=np.iinfo(arr.dtype).min,
            high=np.iinfo(arr.dtype).max,
            shape=arr.shape,
            dtype=arr.dtype,
        )
    else:
        # 注意：这里只是为了描述 shape/dtype，不用于 sample()
        return spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)


def get_demo_paths():
    demo_dir = os.path.join(os.path.dirname(__file__), "demo_data")
    demo_paths = glob.glob(os.path.join(demo_dir, "*.pkl"))
    assert len(demo_paths) > 0, f"❌ 找不到专家数据！请确保 {demo_dir} 目录下有 .pkl 文件。"
    return demo_paths


def get_first_valid_transition(demo_paths):
    """找第一条有效 transition，用来推断 obs/action 结构。"""
    for path in demo_paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)

        for transition in transitions:
            if np.linalg.norm(np.asarray(transition["actions"])) > 0.0:
                return transition

    raise ValueError("❌ demo_data 中没有找到有效的非零动作 transition，无法初始化 BC agent。")


def build_spaces_and_samples_from_demos(demo_paths):
    """
    从 demo_data 中同时推断：
    1. observation_space
    2. action_space
    3. sample_obs
    4. sample_action
    """
    sample_transition = get_first_valid_transition(demo_paths)

    observation_space = infer_space_from_value(sample_transition["observations"])
    action_space = infer_space_from_value(sample_transition["actions"])

    sample_obs = sample_transition["observations"]
    sample_action = np.asarray(sample_transition["actions"])

    return observation_space, action_space, sample_obs, sample_action


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
            actions = bc_agent.sample_actions(observations=obs, seed=key)
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
    工作原理：从录制好的 .pkl 数据中不断抽取 batch，计算 Loss 并更新神经网络权重。
    """
    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="🧠 BC Pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)

        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            if not os.path.exists(FLAGS.bc_checkpoint_path):
                os.makedirs(FLAGS.bc_checkpoint_path)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path),
                bc_agent.state,
                step=step,
                keep=5,
            )

    print_green("✅ BC 预训练完成并已保存 Checkpoint！")


##############################################################################

def main(_):
    config = env_config
    assert config.batch_size % num_devices == 0, "Batch size 必须能被 GPU 数量整除！"

    eval_mode = FLAGS.eval_n_trajs > 0

    # ==========================================================
    # 🔄 训练模式：不创建真实环境
    # ==========================================================
    if not eval_mode:
        demo_paths = get_demo_paths()
        observation_space, action_space, sample_obs, sample_action = build_spaces_and_samples_from_demos(demo_paths)

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )

        bc_agent = jax.device_put(
            jax.tree_map(jnp.array, bc_agent), sharding.replicate()
        )

        if os.path.isdir(os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")):
            print_yellow("⚠️ 警告：目标 checkpoint 似乎已存在，可能会被覆盖！")

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space,
            action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        for path in demo_paths:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    # 自动剔除发呆帧
                    if np.linalg.norm(np.asarray(transition["actions"])) > 0.0:
                        bc_replay_buffer.insert(transition)

        print_green(f" BC 专家数据池加载完毕，有效帧数: {len(bc_replay_buffer)}")
        print_green(" 开始执行 Learner Loop (网络训练)...")

        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    # ==========================================================
    # 🎮 评估模式：才创建真实环境
    # ==========================================================
    else:
        env = env_config.get_environment(
            fake_env=False,
            #save_video=False,
            save_video=FLAGS.save_video,  #默认不保存视频，指令输入控制此次执行脚本保存视频
            classifier=False,
            use_vr=False,
        )
        env = RecordEpisodeStatistics(env)

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )

        bc_agent = jax.device_put(
            jax.tree_map(jnp.array, bc_agent), sharding.replicate()
        )

        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

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

