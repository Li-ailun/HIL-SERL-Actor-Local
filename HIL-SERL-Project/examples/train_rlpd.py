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
#   --exp_name=galaxea_usb_insertion_single \
#   --actor=True \
#   --ip=localhost \
#   --demo_path=./demo_data \
#   --checkpoint_path=./rlpd_checkpoints_single \
#   --debug=True



#输出分析：
#输出1：episode结束时actor请求learner回复，learner若正在编译过期恢复则提示超时，只要learner的0/1000000楷书变化则影响不大
#   （对应输出：Failed to send message to localhost:5588: 资源暂时不可用, potential timeout）
#    增大actor的timeout_ms=3000改成20000，尽量保证每次actor的数据都可以发送给learner


# 最新合并版不需要再做“大逻辑回退”；
# 只要保持“actor 走修正版、learner 走修正版、合并两个修正版，整体结构继续贴近官方”，
# 当前还需要保留的偏离官方逻辑主要有 5 个：
# actor 预先强制 CPU、
# actor 先 demo 后 env、actor 不 replicate、
# learner checkpoint 判空、
# learner 提前 publish 初始网络。
# TrainerClient 统一成 wait_for_server=True +
# timeout_ms=10000/20000，
# 其余不用再向官方对齐。

# actor 里的 x/1000000，数的是actor 在真实环境里跑了多少个 step；
# learner 里的 y/1000000，数的是learner 做了多少次参数更新。

# steps_per_update=50 是 learner 的 50 个训练步（每到y更新50步后，把最新权重给actor，每到y更新2000步，保留权重到自己的learner）；
# 每到这个点，learner 会把最新权重发给 actor（但是本地没有权重，发给了actor的内存里的参数）。
# actor 本地虽然也有一份网络，但那份权重是 learner 传过来的，不是 actor 自己训练出来的；
# 因此 actor 的动作改善，取决于 learner 是否已经学到更好的参数并成功同步过来。


import os
import sys
import glob
import time
import copy
import pickle as pkl

# ==============================================================
# 【合并前差异说明】
# --------------------------------------------------------------
# 版本 A（你用于让 learner 跑起来的版本）主要修复了：
# 1) Learner 启动时 latest_checkpoint(None) 的判空问题。
# 2) Learner 的 RPC server 明确起线程、打印状态、便于调试。
# 3) Learner 先 publish 初始网络，再等待 online replay buffer，
#    避免 Actor/Learner 互相等待的“死锁”。
#
# 版本 B（你用于让 actor 跑起来的版本）主要修复了：
# 1) Actor 模式下在 import jax 前强制 CPU，避免本地 GPU/JAX 初始化崩溃。
# 2) Actor 不做多卡 replicate，只做单机推理。
# 3) Actor 先用 demo 推断网络结构，再创建真实环境，
#    避免真实机器人/相机/VR 环境过早启动导致问题。
# 4) Actor 的 TrainerClient timeout 加大，适应 SSH 隧道与 learner 首次编译。
#
# 【本合并版原则】
# --------------------------------------------------------------
# 1) Learner 分支：保留版本 A 的核心修复。
# 2) Actor 分支：保留版本 B 的核心修复。
# 3) 共同部分：保留你已经验证能工作的结构，尽量不再做额外“理论优化”。
# 4) 保持整体结构仍然贴近官方 train_rlpd.py，只对你当前场景必要处做修改。
# ==============================================================


def _should_force_cpu_for_actor() -> bool:
    """
    【来源：Actor 可运行版】
    在 absl flags 真正解析前，直接检查原始 argv，判断当前是否是 actor 模式。

    目的：
    - 你的本地笔记本在 actor 模式下，JAX+GPU 初始化不稳定；
    - 因此 actor 强制走 CPU 推理，learner 继续在服务器走 GPU 训练；
    - 注意：这一段必须写在 import jax 之前，否则环境变量不会生效。
    """
    argv = [arg.lower() for arg in sys.argv[1:]]
    for arg in argv:
        if arg == "--actor":
            return True
        if arg.startswith("--actor="):
            value = arg.split("=", 1)[1]
            if value in ("true", "1", "yes", "y", "t"):
                return True
    return False


# ==============================================================
# 【合并后保留：Actor 预先强制 CPU】
# --------------------------------------------------------------
# 只在 actor 模式时生效。
# learner 端仍会正常使用服务器 GPU。
# ==============================================================
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
from examples.galaxea_task.usb_pick_insertion_single.config import env_config

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

# ==============================================================
# 【说明】
# --------------------------------------------------------------
# 这里暂时继续保留 PositionalSharding，是为了先保持你当前能跑通的行为。
# 后续如果要进一步和新版 JAX 靠齐，可以整体迁移到 NamedSharding。
# ==============================================================
devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))

def _block_until_ready_tree(tree):
    """
    递归等待 pytree 里的设备数组 ready。
    避免在 publish/save 时隐式触发大规模异步同步。
    """
    def _block(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_block, tree)


def _to_host_pytree(tree):
    """
    把 pytree 中的 jax/sharded 数组安全转成 host numpy。
    这是修复 PositionalSharding 转换 warning / 卡死的关键。
    """
    def _convert(x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return np.asarray(jax.device_get(x))
        return x

    return jax.tree_util.tree_map(_convert, tree)


def _to_loggable_pytree(tree):
    """
    给 wandb / 普通日志用：
    标量转 python scalar，非标量转 numpy。
    """
    def _convert(x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            x = np.asarray(jax.device_get(x))
            if x.shape == ():
                return x.item()
            return x
        return x

    return jax.tree_util.tree_map(_convert, tree)


def _publish_network_to_actor(server, params):
    """
    先把分片 params 拉回 host，再发给 actor。
    避免 server.publish_network 里触发 PositionalSharding 转换问题。
    """
    params = _block_until_ready_tree(params)
    params = _to_host_pytree(params)
    server.publish_network(params)


def _save_checkpoint_host(checkpoint_path, state, step, keep=100):
    """
    保存前先把分片 state 拉回 host numpy，再写 checkpoint。
    这是修复 2000 step 左右卡住/断开的关键。
    """
    state = _block_until_ready_tree(state)
    state = _to_host_pytree(state)
    checkpoints.save_checkpoint(
        os.path.abspath(checkpoint_path),
        state,
        step=step,
        keep=keep,
    )


def extract_gripper_feedback_from_obs(obs):
    """
    从 obs["state"] 提取夹爪反馈量程。
    兼容：
    1) state 是 dict，含 right_gripper / gripper
    2) state 是 ndarray，单臂常见最后一维为 gripper
    """
    if obs is None or "state" not in obs:
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
      0~30   -> 闭合(-1)
      70~100 -> 张开(+1)
      中间区 -> 保持上一标签
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


def rewrite_single_arm_gripper_action_with_feedback(action, next_obs, prev_label):
    """
    用 next_obs 里的夹爪反馈量程，重写单臂 action[6] 为稳定 gripper 标签。
    """
    action = np.asarray(action, dtype=np.float32).copy()

    if action.shape[0] != 7:
        return action, prev_label

    feedback = extract_gripper_feedback_from_obs(next_obs)
    new_label = infer_gripper_label_from_feedback(
        gripper_feedback=feedback,
        prev_label=prev_label,
        raw_action_gripper=action[6],
    )
    action[6] = new_label
    return action, new_label

# ==============================================================
# Learner / Actor 共用辅助：从 demo 推断空间与网络样本
# --------------------------------------------------------------
# 【合并后保留】
# 你当前的 learner 与 actor 都已验证：从 demo 推断 obs/action 结构更稳。
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
# Actor 逻辑
# --------------------------------------------------------------
# 【合并后说明】
# 1) 保留官方结构：eval / train 两种模式。
# 2) 保留你当前能工作的训练采集逻辑。
# 3) 保留更稳的 timeout_ms=10000。
# 4) wait_for_server 这里采用 True：
#    - learner 已经改为“先起服务、先发初始网络”；
#    - 因此不会再像之前那样死锁；
#    - 同时 actor 在 learner 还没完全 ready 时也不会立即崩掉。
# ==============================================================
def actor(agent, data_store, intvn_data_store, env, sampling_rng, config):
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

                if reward or done or truncated:
                    print(f"[reward-debug] reward={reward}, done={done}, truncated={truncated}")

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
        wait_for_server=True,
        timeout_ms=10000,
    )

    def update_params(params):
        nonlocal agent
        params = jax.tree_util.tree_map(jnp.array, params)
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    # 当前 episode 的夹爪稳定标签
    prev_gripper_label = None

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="actor")
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                raw_actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                raw_actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                raw_actions = np.asarray(jax.device_get(raw_actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(raw_actions)

            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # 如果有人类接管，就用 intervene_action 作为原始动作命令
            if "intervene_action" in info:
                raw_actions = np.asarray(info.pop("intervene_action"), dtype=np.float32)
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            # -------------------------------------------------
            # 用夹爪反馈量程反推稳定 gripper 训练标签
            # 只改 action[6]，其余维保持不变
            # -------------------------------------------------
            actions, prev_gripper_label = rewrite_single_arm_gripper_action_with_feedback(
                raw_actions,
                next_obs,
                prev_gripper_label,
            )

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
                prev_gripper_label = None

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
# Learner 逻辑
# --------------------------------------------------------------
# 【合并后说明】
# 1) 保留版本 A 的 latest_checkpoint 判空修复。
# 2) 保留版本 A 的显式 server 线程包装与调试打印。
# 3) 保留“先 publish 初始网络，再等待 replay buffer”的修复。
# ==============================================================
def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger, config):
    latest_ckpt = None
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))

    if latest_ckpt is None:
        start_step = 0
    else:
        start_step = int(os.path.basename(latest_ckpt)[11:]) + 1

    step = start_step

    def stats_callback(req_type: str, payload: dict) -> dict:
        assert req_type == "send-stats", f"Invalid request type: {req_type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    import threading
    import traceback

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)

    def _run_server():
        try:
            print_green(f"starting req_rep_server on port {make_trainer_config().port_number}")
            server.req_rep_server.run()
        except Exception as e:
            print_yellow(f"REQ/REP server crashed: {e!r}")
            traceback.print_exc()
            raise

    server.thread = threading.Thread(target=_run_server, daemon=True)
    server.thread.start()

    time.sleep(1)
    print_green(f"server thread alive: {server.thread.is_alive()}")

    # ---------------------------------------------------------
    # 【关键修复】
    # 先发初始网络，避免 actor 一启动就因为拿不到参数而等待或崩溃。
    # ---------------------------------------------------------
    _publish_network_to_actor(server, agent.state.params)
    print_green("sent initial network to actor")

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

    # 补发一次当前网络：
    # 早发一次用于避免 actor 启动时拿不到参数，
    # 这里再发一次用于和官方“buffer 填满后发网络”的时机对齐。
    _publish_network_to_actor(server, agent.state.params)
    print_green("resent initial network to actor after replay buffer warmup")

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
           _publish_network_to_actor(server, agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(_to_loggable_pytree(update_info), step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
           os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
           _save_checkpoint_host(
                FLAGS.checkpoint_path,
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
    # 【合并后关键差异】
    # Learner 与 Actor 都先从 demo 推断网络结构。
    # Actor 的真实环境延后到 agent 初始化之后再创建。
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
    # 按 setup_mode 创建对应 SAC agent
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

    # ---------------------------------------------------------
    # 【合并后关键差异】
    # Learner：继续多设备 replicate
    # Actor：只保留单机参数树，不做 replicate
    # ---------------------------------------------------------
    if FLAGS.learner:
        agent = jax.device_put(
            jax.tree_util.tree_map(jnp.array, agent),
            sharding.replicate(),
        )
    else:
        agent = jax.tree_util.tree_map(jnp.array, agent)

    # ---------------------------------------------------------
    # checkpoint 恢复逻辑
    # ---------------------------------------------------------
    if (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*"))
    ):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )  

        if FLAGS.learner:
            ckpt = jax.device_put(
                jax.tree_util.tree_map(jnp.array, ckpt),
                sharding.replicate(),
    )
        else:
            ckpt = jax.tree_util.tree_map(jnp.array, ckpt)

        agent = agent.replace(state=ckpt)

        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt:
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    # ---------------------------------------------------------
    # Learner 分支
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
    # Actor 分支
    # ---------------------------------------------------------
    else:
        # Actor 只用单设备 RNG
        sampling_rng = jax.device_put(sampling_rng)

        # 训练 Actor：需要 VR；评估 Actor：不要 VR
        use_vr = False if FLAGS.eval_checkpoint_step > 0 else True

        # -----------------------------------------------------
        # 【合并后关键差异】
        # 真实环境延后到 agent 初始化之后再创建。
        # 这对你当前本地 actor 更稳。
        # classifier=True：保留你当前已验证能工作的奖励分类器路径。
        # -----------------------------------------------------
        env = env_config.get_environment(
            fake_env=False,
            save_video=FLAGS.save_video if FLAGS.eval_checkpoint_step > 0 else False,
            classifier=True,
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


