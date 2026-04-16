#!/usr/bin/env python3
#人类在环监督+RL

# 1. Learner 模式（训练网络参数-生成权重）
# --learner=True

# 作用是：
# 启动后台训练服务
# 维护两个 buffer
# 一个是在线探索经验 replay_buffer，一个是专家/人工干预经验 demo_buffer
# 按 50% 在线数据 + 50% demo 数据 混合采样
# 在 GPU/JAX 上持续更新 SAC 网络
# 定期发布最新参数给 Actor
# 定期保存 checkpoint。

# JAX GPU
# 训练依赖链
# SAC / RLPD 更新稳定性
# checkpoint 保存恢复
# 大 batch / 长时间训练

# 2. Actor 模式（接受数据-传给learn）
# --actor=True

# 作用是：
# 连接真实环境
# 执行策略动作
# 接收 VR 干预
# 把在线经验推送给 Learner
# 把人工干预阶段的数据单独推给 demo_buffer
# 定期把本地缓存保存成 buffer/*.pkl 和 demo_buffer/*.pkl。

# 连真实环境
# 接收 ROS 状态
# 相机采图
# 训练 Actor 时支持 VR 干预
# 跑策略前向推理
# 把 transition 发给 Learner

# 3. Actor 评估模式（状态输入-加载权重-输出动作）
# --actor=True --eval_checkpoint_step=xxx --eval_n_trajs=N

# 作用是：
# 不参与训练
# 只加载某一步 checkpoint
# 在真实环境中做纯推理评估
# 统计成功率和耗时。



#####################################################################
#执行脚本：
# Learner 训练端
# 这个端负责：
# 吃 demo 数据
# 等 Actor 送在线数据
# 做 RLPD / SAC 更新
# 存 checkpoint

# python train_rlpd.py \
#   --exp_name=galaxea_usb_insertion \
#   --learner=True \
#   --ip=localhost \
#   --demo_path=./demo_data \
#   --checkpoint_path=./rlpd_checkpoints \
#   --debug=True

# python train_rlpd.py \
#   --exp_name=galaxea_usb_insertion \
#   --actor=True \
#   --ip=localhost \
#   --checkpoint_path=./rlpd_checkpoints \
#   --debug=True


#含义
  # 1,--learner=True
# 启动 Learner 端。负责训练和参数更新。

# 2,--actor=True
  # 启动 Actor 端。负责真机交互和数据采集。

# 3,--ip=...
  # Actor 连接 Learner 的地址；单机就写 localhost。

# 4,--demo_path=  ./demo_data \
  # Learner 启动时加载初始 demo 数据，灌进 demo_buffer。

# 5,--checkpoint_path=  ./rlpd_checkpoints \
  # 保存和读取：

  # checkpoint
  # buffer/*.pkl
  # demo_buffer/*.pkl

# 7,  --eval_checkpoint_step= n   （和）
  # 大于 0 就进入  actor评估模式，不做训练采集。
  # 只评估第n步生成的权重
  # = 0 或者不写默认  learn或者actor模式

# 8,  --eval_n_trajs=...
  # 评估时跑多少条 episode。

# 9,  --save_video=True/False
  # 评估时是否保存视频。你的 config.py 已经支持这个参数一路传进去。

# 10, --debug=True（False或者不输入debug则默认wandb记录）
  # 关闭 wandb 上传，适合先本地调通


########################################################################

#################################################
#功能配置
#reward功能：
#train_rlpd.py actor 训练：通常应该开 classifier，因为 RLPD 训练需要奖励信号


# (1)Learner 不该碰真机、
# (2)Actor 训练要保留 VR 干预、
# (3)Actor 评估不要 VR，但保留 ROS/相机/分类器。

####################################################

#本地（一直开着）：ssh -p 2122 -L 5588:localhost:5588 lixiang@service.qich.top
#用处：转接learner（服务器）端口5588,
#另开本地终端输入指令：
# python train_rlpd.py \
#   --exp_name=galaxea_usb_insertion \
#   --actor=True \
#   --ip=localhost \
#   --demo_path=./demo_data \
#   --checkpoint_path=./rlpd_checkpoints \
#   --debug=True



import os
import sys
import glob
import time
import copy
import pickle as pkl


def _should_force_cpu_for_actor():
    """在 absl flags 解析前，先从原始 argv 判断是不是 actor 模式。"""
    argv = [arg.lower() for arg in sys.argv[1:]]

    for arg in argv:
        if arg == "--actor":
            return True
        if arg.startswith("--actor="):
            value = arg.split("=", 1)[1]
            if value in ("true", "1", "yes", "y", "t"):
                return True
    return False


# 关键：必须放在 import jax 之前
if _should_force_cpu_for_actor():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium import spaces
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

# ==============================================================
# SERL 强化学习算法核心组件
# ==============================================================
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# ==============================================================
# AgentLace RPC 通信框架
# ==============================================================
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

# ==============================================================
# 路径配置
# ==============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================================================
# 任务配置
# ==============================================================
from examples.galaxea_task.usb_pick_insertion.config import env_config

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")

flags.DEFINE_boolean("learner", False, "Whether this process is the learner.")
flags.DEFINE_boolean("actor", False, "Whether this process is the actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")

flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo data for learner bootstrap.")
flags.DEFINE_string("checkpoint_path", "./rlpd_checkpoints", "Path to save checkpoints / buffers.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save evaluation videos.")
flags.DEFINE_boolean("debug", False, "Debug mode, disable wandb upload.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))


# ==============================================================
# Learner 侧辅助：从 demo 推断空间 / 样本
# ==============================================================
def infer_space_from_value(x):
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
        return spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)


def resolve_demo_paths(paths):
    resolved = []
    for p in paths:
        if os.path.isdir(p):
            resolved.extend(glob.glob(os.path.join(p, "*.pkl")))
        else:
            resolved.extend(glob.glob(p))
    resolved = [p for p in resolved if p.endswith(".pkl")]
    assert len(resolved) > 0, "❌ 没有找到任何 demo .pkl 文件。"
    return resolved


def get_first_valid_transition(paths):
    for path in paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)

        for transition in transitions:
            if "actions" in transition and "observations" in transition:
                return transition

    raise ValueError("❌ 无法从 demo_path 中找到有效 transition。")


def build_spaces_and_samples_from_demos(paths):
    sample_transition = get_first_valid_transition(paths)
    observation_space = infer_space_from_value(sample_transition["observations"])
    action_space = infer_space_from_value(sample_transition["actions"])
    sample_obs = sample_transition["observations"]
    sample_action = np.asarray(sample_transition["actions"])
    return observation_space, action_space, sample_obs, sample_action


# ==============================================================
# Actor 逻辑（官方结构）
# ==============================================================
def actor(agent, data_store, intvn_data_store, env, sampling_rng, config):
    """
    官方 Actor 结构：
    - 评估模式：纯推理
    - 训练模式：和环境交互、接收人类干预、发送数据到 Learner
    """
    # ---------------------------------------------------------
    # 纯评估模式
    # ---------------------------------------------------------
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()

            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print_green(f"✅ 第 {episode + 1} 回合成功！耗时: {dt:.2f}s")
                    else:
                        print_yellow(f"❌ 第 {episode + 1} 回合失败。")

                    success_counter += reward
                    print(f"📊 当前成绩: {success_counter}/{episode + 1}")

        print_green(f"🏆 success rate: {success_counter / FLAGS.eval_n_trajs:.2%}")
        if time_list:
            print_green(f"⏱️ average time: {np.mean(time_list):.2f}s")
        return

    # ---------------------------------------------------------
    # 训练采集模式
    # ---------------------------------------------------------
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path
        and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=False,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="actor")
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # 官方人类干预逻辑
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))

            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs

            if done or truncated:
                if "episode" not in info:
                    info["episode"] = {}
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps

                stats = {"environment": info}
                client.request("send-stats", stats)

                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False

                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
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

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


# ==============================================================
# Learner 逻辑（官方结构）
# ==============================================================
def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger, config):
    """
    官方 Learner 结构：
    - 等待 online buffer 填充
    - 50/50 online + demo 采样
    - cta_ratio 控制 critic / actor 更新节奏
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(req_type: str, payload: dict) -> dict:
        assert req_type == "send-stats", f"Invalid request type: {req_type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    timer = Timer()

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, _ = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )


# ==============================================================
# 主函数
# ==============================================================
def main(_):
    config = env_config
    assert config.batch_size % num_devices == 0, "Batch size 必须能被设备数整除"

    if FLAGS.learner == FLAGS.actor:
        raise ValueError("❌ 必须且只能指定一个：--learner=True 或 --actor=True")

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    print_green(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # ---------------------------------------------------------
    # Learner / Actor 都先从 demo 推断空间和样本
    # Actor 的真实环境延后到 agent 初始化之后再创建
    # ---------------------------------------------------------
    if FLAGS.learner:
        assert FLAGS.demo_path is not None, "❌ Learner 必须通过 --demo_path 传入初始 demo 数据路径"
        demo_paths = resolve_demo_paths(FLAGS.demo_path)
        observation_space, action_space, sample_obs, sample_action = build_spaces_and_samples_from_demos(demo_paths)
        env = None
    else:
        assert FLAGS.demo_path is not None, "❌ Actor 现在也需要通过 --demo_path 提供一份 demo，用于初始化网络结构"
        demo_paths = resolve_demo_paths(FLAGS.demo_path)
        observation_space, action_space, sample_obs, sample_action = build_spaces_and_samples_from_demos(demo_paths)
        env = None

    rng, sampling_rng = jax.random.split(rng)
    
    # ---------------------------------------------------------
    # 按官方结构动态选择 SAC 变体
    # ---------------------------------------------------------
    if config.setup_mode in ["single-arm-fixed-gripper", "dual-arm-fixed-gripper"]:
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False

    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True

    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True

    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    if FLAGS.learner:
        agent = jax.device_put(
            jax.tree_util.tree_map(jnp.array, agent),
            sharding.replicate(),
        )
    else:
        # Actor 只做单机推理，不做多卡复制
        agent = jax.tree_util.tree_map(jnp.array, agent)
        
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path) and glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*")):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt:
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    # ---------------------------------------------------------
    # Learner
    # ---------------------------------------------------------
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space,
            action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space,
            action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        demo_paths = resolve_demo_paths(FLAGS.demo_path)
        for path in demo_paths:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                    demo_buffer.insert(transition)

        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,

            wandb_logger=wandb_logger,
            config=config,
        )

    # ---------------------------------------------------------
    # Actor
    # ---------------------------------------------------------
    else:
        # Actor 只用单设备 RNG
        sampling_rng = jax.device_put(sampling_rng)

        # 训练 Actor：需要 VR
        # 评估 Actor：不要 VR
        use_vr = False if FLAGS.eval_checkpoint_step > 0 else True

        # 关键：先完成 agent 初始化，再启动真实环境
        env = env_config.get_environment(
            fake_env=False,
            save_video=FLAGS.save_video if FLAGS.eval_checkpoint_step > 0 else False,
            classifier=False,   # 先只解决 classifier=False 这条
            use_vr=use_vr,
        )
        env = RecordEpisodeStatistics(env)

        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            config,
        )

if __name__ == "__main__":
    app.run(main)

