#rlpd的继续说明，
#gpu后，现在8~14 it/s，和之前 CPU 的 3~4 it/s 相比，符合这个变化。
#learner更新50步网络后，会发给actor最新的网络，所以actor在接受新网络后会卡顿（0.05级别的延迟）一下再继续输出动作
# 当前 step 先按当前已经生效的网络跑完
# 新网络先收进来，到下一次安全点 loop_start 再切换，下一步开始才用新网络出动作
# 对一个具体 step 来说，动作要么是：
#  旧网络输出的完整动作
#  要么是新网络输出的完整动作，不会在“收到包的那一瞬间”还持续记录一串冻结动作。


#  /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_actor_gpu_ros.sh
import os
import sys
import glob
import time
import copy
import pickle as pkl
import threading

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


def _raw_flag_value(name: str, default=None):
    """在 absl flags 解析前，直接从 sys.argv 读取原始 flag 值。"""
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg == f"--{name}":
            return True
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _raw_flag_bool(name: str, default: bool = False) -> bool:
    value = _raw_flag_value(name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")


def _is_actor_mode_from_argv() -> bool:
    return _raw_flag_bool("actor", False)


# ==============================================================
# 【GPU 版 Actor 预配置】
# --------------------------------------------------------------
# 目标：
# 1) 默认允许 actor 使用本地 GPU（例如笔记本 4060）
# 2) 保留开关：若本地 GPU 仍然报错，可通过 --force_actor_cpu=True 回退到 CPU
# 3) 这些环境变量必须写在 import jax 之前，否则不会生效
# ==============================================================
if _is_actor_mode_from_argv():
    force_actor_cpu = _raw_flag_bool("force_actor_cpu", False)
    actor_cuda_visible_devices = _raw_flag_value("actor_cuda_visible_devices", "0")
    actor_disable_preallocate = _raw_flag_bool("actor_disable_preallocate", True)
    actor_mem_fraction = _raw_flag_value("actor_mem_fraction", None)
    actor_jax_platforms = _raw_flag_value("actor_jax_platforms", None)

    if force_actor_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        # 不强制写死成 cpu，让本地 actor 优先尝试 GPU。
        # 默认只暴露 1 张卡，避免笔记本/工作站误占多卡。
        if actor_cuda_visible_devices not in (None, "", "auto"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_cuda_visible_devices)

        # 对笔记本 GPU，默认关闭预分配，减少一启动就吃满显存的概率。
        if actor_disable_preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        if actor_mem_fraction not in (None, "", "0", 0):
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(actor_mem_fraction)

        # 只有在你显式传 --actor_jax_platforms=... 时才覆盖。
        # 默认不设置，让 JAX 自己选择可用后端（通常会优先 GPU）。
        if actor_jax_platforms not in (None, ""):
            os.environ["JAX_PLATFORMS"] = str(actor_jax_platforms)

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
flags.DEFINE_integer("print_period", 50, "How often to print actor/learner debug lines.")
flags.DEFINE_integer("request_timeout_ms", 15000, "TrainerClient REQ/REP timeout in milliseconds.")
flags.DEFINE_integer("client_update_period", 10, "How many actor env steps between explicit client.update() calls.")
flags.DEFINE_float("client_update_interval_sec", 0.5, "Background interval in seconds for non-blocking client.update() on actor.")
flags.DEFINE_boolean("client_update_background", True, "Whether to run client.update() in a background thread on actor.")
flags.DEFINE_integer("warmup_publish_period_s", 5, "How often learner re-publishes current network during replay warmup.")
flags.DEFINE_integer("trainer_port", 0, "Override TrainerConfig.port_number when > 0.")
flags.DEFINE_integer("trainer_broadcast_port", 0, "Override TrainerConfig.broadcast_port when > 0.")
flags.DEFINE_boolean("print_trainer_config", True, "Print resolved trainer config and ports.")
flags.DEFINE_boolean("minimal_logs", True, "Only keep the most important logs.")
flags.DEFINE_boolean("force_actor_cpu", False, "Force actor to run on CPU even if a local GPU is available.")
flags.DEFINE_string("actor_cuda_visible_devices", "0", "CUDA_VISIBLE_DEVICES used by actor before importing JAX. Use auto or empty to leave unchanged.")
flags.DEFINE_boolean("actor_disable_preallocate", True, "Disable JAX GPU memory preallocation on actor to reduce laptop GPU OOM risk.")
flags.DEFINE_float("actor_mem_fraction", 0.0, "Optional XLA_PYTHON_CLIENT_MEM_FRACTION for actor when > 0.")
flags.DEFINE_string("actor_jax_platforms", "", "Optional JAX_PLATFORMS override for actor, e.g. 'cuda' or 'cpu'. Empty means let JAX auto-select.")
flags.DEFINE_boolean("actor_expect_gpu", True, "Warn if actor does not start on GPU.")

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


def _suggest_ssh_forward_command(reqrep_port: int, broadcast_port: int) -> str:
    return (
        f"ssh -p 2122 -L {reqrep_port}:localhost:{reqrep_port} "
        f"-L {broadcast_port}:localhost:{broadcast_port} lixiang@service.qich.top"
    )


def print_blue(x):
    print("\033[94m {}\033[00m".format(x))


def print_red(x):
    print("\033[91m {}\033[00m".format(x))


def _log_enabled(kind: str) -> bool:
    if not getattr(FLAGS, "minimal_logs", False):
        return True
    keep = {
        "main",
        "checkpoint",
        "actor_network",
        "actor_episode",
        "actor_warning",
        "actor_error",
        "learner_publish",
        "learner_step",
        "learner_checkpoint",
        "learner_env",
    }
    return kind in keep


def _log_info(kind: str, msg: str, color: str = "blue"):
    if not _log_enabled(kind):
        return
    fn = {
        "blue": print_blue,
        "green": print_green,
        "yellow": print_yellow,
        "red": print_red,
    }.get(color, print)
    fn(msg)


def _as_python_scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError('empty array cannot be converted to scalar')
    return arr.reshape(-1)[0].item()


def _safe_float(x, default=0.0):
    try:
        return float(_as_python_scalar(x))
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        return int(_as_python_scalar(x))
    except Exception:
        return default


def _trainer_config_dict(cfg):
    out = {}
    for name in dir(cfg):
        if name.startswith('_'):
            continue
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (int, float, str, bool, type(None), list, tuple, dict)):
            out[name] = value
    return out


def _build_trainer_config():
    cfg = make_trainer_config()
    if FLAGS.trainer_port > 0 and hasattr(cfg, 'port_number'):
        cfg.port_number = FLAGS.trainer_port
    if FLAGS.trainer_broadcast_port > 0 and hasattr(cfg, 'broadcast_port'):
        cfg.broadcast_port = FLAGS.trainer_broadcast_port
    return cfg


def _log_trainer_config(cfg, role):
    if not FLAGS.print_trainer_config or FLAGS.minimal_logs:
        return
    cfg_dict = _trainer_config_dict(cfg)
    print_blue(f"[{role}-trainer-config] {cfg_dict}")


def _tree_debug_signature(tree, max_leaves=8, elems_per_leaf=8):
    """
    生成一个轻量级参数指纹，方便对齐 learner publish 和 actor recv。
    不追求密码学哈希，只追求调试时稳定、便宜、可读。
    """
    leaves, _ = jax.tree_util.tree_flatten(tree)

    sampled = []
    total_params = 0
    leaf_shapes = []

    for idx, leaf in enumerate(leaves):
        arr = np.asarray(leaf)
        total_params += int(arr.size)
        if idx < max_leaves:
            leaf_shapes.append(tuple(arr.shape))
            if arr.size > 0:
                flat = arr.reshape(-1)
                sampled.extend(flat[:elems_per_leaf].astype(np.float64).tolist())

    sample_arr = np.asarray(sampled, dtype=np.float64) if sampled else np.zeros((1,), dtype=np.float64)
    checksum = float(sample_arr.sum())
    abs_mean = float(np.mean(np.abs(sample_arr)))
    sample_std = float(np.std(sample_arr))

    return {
        "leaf_count": len(leaves),
        "total_params": total_params,
        "checksum": checksum,
        "abs_mean": abs_mean,
        "sample_std": sample_std,
        "sample_head": [round(float(x), 6) for x in sample_arr[:6]],
        "leaf_shapes": leaf_shapes[:4],
    }


def _format_signature(sig):
    return (
        f"leafs={sig['leaf_count']}, total_params={sig['total_params']}, "
        f"checksum={sig['checksum']:.6f}, abs_mean={sig['abs_mean']:.6f}, "
        f"sample_std={sig['sample_std']:.6f}, head={sig['sample_head']}, shapes={sig['leaf_shapes']}"
    )


def _extract_episode_debug_info(info):
    episode = info.get("episode", {}) if isinstance(info, dict) else {}
    return {
        "return": _safe_float(episode.get("r", episode.get("return", 0.0))),
        "length": _safe_int(episode.get("l", episode.get("length", 0)), 0),
        "duration": _safe_float(episode.get("t", episode.get("time", 0.0))),
        "success": _safe_float(info.get("success", info.get("is_success", 0.0))),
        "intervention_count": _safe_int(episode.get("intervention_count", 0), 0),
        "intervention_steps": _safe_int(episode.get("intervention_steps", 0), 0),
    }


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


def _publish_network_to_actor(server, params, *, reason="periodic_update", step=None):
    """
    先把分片 params 拉回 host，再发给 actor，并打印参数指纹。
    这样 learner/actor 两端能肉眼确认是不是同一版网络。
    """
    t0 = time.time()
    params = _block_until_ready_tree(params)
    params = _to_host_pytree(params)
    sig = _tree_debug_signature(params)
    server.publish_network(params)
    dt = time.time() - t0
    _log_info('learner_publish', f"[learner-network-publish] reason={reason}, step={step}, cost={dt:.3f}s, {_format_signature(sig)}", 'blue')
    return sig


def _save_checkpoint_host(checkpoint_path, state, step, keep=100):
    """
    保存前先把分片 state 拉回 host numpy，再写 checkpoint。
    这是修复 2000 step 左右卡住/断开的关键。
    """
    t0 = time.time()
    state = _block_until_ready_tree(state)
    state = _to_host_pytree(state)
    sig = _tree_debug_signature(state.params if hasattr(state, "params") else state)
    checkpoints.save_checkpoint(
        os.path.abspath(checkpoint_path),
        state,
        step=step,
        keep=keep,
    )
    dt = time.time() - t0
    _log_info('learner_checkpoint', f"[learner-checkpoint-save] step={step}, cost={dt:.3f}s, path={os.path.abspath(checkpoint_path)}, {_format_signature(sig)}", 'blue')


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

def map_single_arm_exec_action_to_hardware(
    action,
    prev_hw_cmd,
    close_cmd=20.0,
    open_cmd=80.0,
    deadband=0.5,
):
    """
    把训练/策略空间里的单臂 gripper 动作，映射成真机执行值。

    语义：
    - policy / demo / intervene_action 里的 action[6]：
        +1 代表张开
        -1 代表闭合
        中间死区代表“保持上一状态”
    - 真机执行时：
        open_cmd  -> 发送给硬件的张开值（如 80）
        close_cmd -> 发送给硬件的闭合值（如 20）

    兼容：
    - 如果传进来的 action[6] 本身已经是 0~100 的硬件值，就直接透传。
    """
    action = np.asarray(action, dtype=np.float32).copy()

    if action.shape[0] != 7:
        return action, prev_hw_cmd

    grip = float(action[6])

    # 已经是硬件量程值：直接透传
    if 0.0 <= grip <= 100.0 and abs(grip) > 5.0:
        hw_cmd = grip
    else:
        # 标签空间 / 策略空间：±1 + deadband
        if grip >= deadband:
            hw_cmd = open_cmd
        elif grip <= -deadband:
            hw_cmd = close_cmd
        else:
            hw_cmd = prev_hw_cmd

    exec_action = action.copy()
    exec_action[6] = np.float32(hw_cmd)
    return exec_action, float(hw_cmd)


def build_single_arm_open_exec_action_like(action):
    """
    用于 reset 后初始化“上一夹爪执行值”。
    这里只关心 gripper 默认保持张开。
    """
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[0] == 7:
        action[6] = 80.0
    return action

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
def actor(agent, data_store, intvn_data_store, env, sampling_rng, config, trainer_cfg):
    network_debug = {
        "recv_count": 0,
        "applied_count": 0,
        "duplicate_recv_count": 0,
        "last_recv_time": None,
        "last_apply_time": None,
        "last_sig": None,
        "last_applied_sig": None,
        "pending_params": None,
        "pending_sig": None,
        "pending_recv_time": None,
        "warned_missing_broadcast": False,
        "last_update_log_time": None,
    }
    agent_lock = threading.Lock()
    client_rpc_lock = threading.Lock()
    client_stop_event = threading.Event()
    client_force_event = threading.Event()
    client_thread = None

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
        print_blue(
            f"[actor-eval] loaded checkpoint step={FLAGS.eval_checkpoint_step}, "
            f"{_format_signature(_tree_debug_signature(ckpt.params if hasattr(ckpt, 'params') else ckpt))}"
        )

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            prev_exec_gripper_cmd = 80.0

            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                policy_actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key,
                )
                policy_actions = np.asarray(jax.device_get(policy_actions), dtype=np.float32)

                dbg_exec_actions, prev_exec_gripper_cmd = map_single_arm_exec_action_to_hardware(
                    policy_actions,
                    prev_exec_gripper_cmd,
                )

                next_obs, reward, done, truncated, info = env.step(policy_actions)
                obs = next_obs

                if reward or done or truncated:
                    print(
                        f"[actor-eval-step] ep={episode + 1}, reward={reward}, done={done}, truncated={truncated}, "
                        f"policy_gripper={policy_actions[6] if policy_actions.shape[0] == 7 else 'N/A'}, "
                        f"mapped_hw={dbg_exec_actions[6] if dbg_exec_actions.shape[0] == 7 else 'N/A'}"
                    )

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

    _log_trainer_config(trainer_cfg, 'actor')
    if not FLAGS.minimal_logs:
        print_blue(
            f"[actor-client-init] learner_ip={FLAGS.ip}, wait_for_server=True, timeout_ms={FLAGS.request_timeout_ms}, "
            f"start_step={start_step}, random_steps={config.random_steps}, "
            f"background_update={FLAGS.client_update_background}, update_interval_sec={FLAGS.client_update_interval_sec}"
        )

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        trainer_cfg,
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=FLAGS.request_timeout_ms,
    )

    def update_params(params):
        now = time.time()
        since_prev = None if network_debug["last_recv_time"] is None else now - network_debug["last_recv_time"]
        sig = _tree_debug_signature(params)
        with agent_lock:
            same_as_pending = network_debug["pending_sig"] == sig
            same_as_applied = network_debug["last_applied_sig"] == sig
            if same_as_pending or same_as_applied:
                network_debug["duplicate_recv_count"] += 1
                network_debug["last_recv_time"] = now
                network_debug["last_sig"] = sig
                dup_n = network_debug["duplicate_recv_count"]
                if (not FLAGS.minimal_logs) and (dup_n <= 3 or dup_n % 10 == 0):
                    print_blue(
                        f"[actor-network-recv-skip] duplicate_count={dup_n}, "
                        f"same_as={'pending' if same_as_pending else 'applied'}, "
                        f"since_prev={None if since_prev is None else round(since_prev, 3)}, {_format_signature(sig)}"
                    )
                return
            params = jax.tree_util.tree_map(jnp.array, params)
            network_debug["pending_params"] = params
            network_debug["pending_sig"] = sig
            network_debug["pending_recv_time"] = now
        network_debug["recv_count"] += 1
        network_debug["last_recv_time"] = now
        network_debug["last_sig"] = sig
        _log_info('actor_network', f"[actor-network-recv] recv_count={network_debug['recv_count']}, since_prev={None if since_prev is None else round(since_prev, 3)}, {_format_signature(sig)}", 'blue')

    def _apply_pending_params(reason, *, force_print=False):
        nonlocal agent
        with agent_lock:
            pending_params = network_debug["pending_params"]
            pending_sig = network_debug["pending_sig"]
            pending_recv_time = network_debug["pending_recv_time"]
            network_debug["pending_params"] = None
            network_debug["pending_sig"] = None
            network_debug["pending_recv_time"] = None
        if pending_params is None:
            return False
        agent = agent.replace(state=agent.state.replace(params=pending_params))
        now = time.time()
        network_debug["applied_count"] += 1
        network_debug["last_apply_time"] = now
        network_debug["last_applied_sig"] = pending_sig
        if (not FLAGS.minimal_logs and force_print) or network_debug["applied_count"] <= 5:
            lag = None if pending_recv_time is None else round(now - pending_recv_time, 3)
            _log_info('actor_network', f"[actor-network-apply] applied_count={network_debug['applied_count']}, reason={reason}, apply_lag={lag}, {_format_signature(pending_sig)}", 'blue')
        return True

    client.recv_network_callback(update_params)
    if not FLAGS.minimal_logs:
        print_blue("[actor-client-init] recv_network_callback registered")
    if (not FLAGS.minimal_logs) and FLAGS.ip == 'localhost' and getattr(trainer_cfg, 'broadcast_port', None):
        print_yellow(
            "[actor-broadcast-hint] 你当前使用 localhost。若 actor 通过 SSH 连接远端 learner，"
            f"请确认不仅转发了 req/rep 端口 {trainer_cfg.port_number}，还转发了 broadcast 端口 {trainer_cfg.broadcast_port}。"
        )
        print_yellow(
            f"[actor-broadcast-hint] 示例: {_suggest_ssh_forward_command(trainer_cfg.port_number, trainer_cfg.broadcast_port)}"
        )

    transitions = []
    demo_transitions = []

    def _client_update(reason, *, force_print=False):
        t0 = time.time()
        ok = False
        err = None
        try:
            with client_rpc_lock:
                ok = bool(client.update())
        except Exception as e:
            err = repr(e)
        dt = time.time() - t0
        now = time.time()
        should_log = (not FLAGS.minimal_logs) and (force_print or err or dt > 1.0)
        if not should_log and (not FLAGS.minimal_logs):
            last_log_time = network_debug["last_update_log_time"]
            if last_log_time is None or (now - last_log_time) > 10.0:
                should_log = True
        if should_log:
            network_debug["last_update_log_time"] = now
            print_blue(
                f"[actor-client-update] reason={reason}, ok={ok}, dt={dt:.3f}s, recv_count={network_debug['recv_count']}, "
                f"applied_count={network_debug['applied_count']}, duplicate_recv_count={network_debug['duplicate_recv_count']}, "
                f"last_sig={None if network_debug['last_sig'] is None else _format_signature(network_debug['last_sig'])}, err={err}"
            )
        return ok

    def _client_update_worker():
        # 后台线程负责把队列里的 transition 刷到 learner；
        # 主线程永远继续按当前 agent 发动作，不再被 client.update() 卡住。
        _client_update('background_start', force_print=True)
        while not client_stop_event.is_set():
            triggered = client_force_event.wait(timeout=max(0.05, float(FLAGS.client_update_interval_sec)))
            client_force_event.clear()
            if client_stop_event.is_set():
                break
            reason = 'background_force' if triggered else 'background_periodic'
            _client_update(reason, force_print=False)

    if FLAGS.client_update_background:
        client_thread = threading.Thread(target=_client_update_worker, daemon=True)
        client_thread.start()
    else:
        _log_info('actor_warning', '[actor-client-update] background disabled; actor 主线程将退化为阻塞式 update。', 'yellow')

    obs, _ = env.reset()

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_index = 0

    prev_gripper_label = None
    prev_exec_gripper_cmd = 80.0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="actor")
    try:
        for step in pbar:
            timer.tick("total")

            # 若广播线程已经收到了新网络，只在主循环边界原子替换，不阻塞动作输出。
            _apply_pending_params('loop_start', force_print=(step == start_step or step % FLAGS.print_period == 0))

            if (
                network_debug['recv_count'] == 0
                and not network_debug['warned_missing_broadcast']
                and step >= start_step + max(FLAGS.print_period * 2, 100)
            ):
                network_debug['warned_missing_broadcast'] = True
                _log_info('actor_warning', "[actor-network-warning] actor 已经跑了较长时间，但 recv_count 仍为 0。这通常说明 learner 的参数广播没有到达 actor。", 'yellow')
                _log_info('actor_warning', f"[actor-network-warning] 当前 trainer_config: port={trainer_cfg.port_number}, broadcast_port={getattr(trainer_cfg, 'broadcast_port', None)}, ip={FLAGS.ip}", 'yellow')
                if FLAGS.ip == 'localhost' and getattr(trainer_cfg, 'broadcast_port', None):
                    _log_info('actor_warning', f"[actor-network-warning] 若你通过 SSH 隧道连接远端 learner，请确认执行了: {_suggest_ssh_forward_command(trainer_cfg.port_number, trainer_cfg.broadcast_port)}", 'yellow')

            # 兼容保留：如果显式关闭后台更新，则退回旧的步进式 update。
            if (not FLAGS.client_update_background) and (
                step == start_step or (FLAGS.client_update_period > 0 and step % FLAGS.client_update_period == 0)
            ):
                _client_update('periodic_step', force_print=(step == start_step or step % FLAGS.print_period == 0))

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    policy_actions = np.asarray(env.action_space.sample(), dtype=np.float32)
                    action_source = "random"
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    policy_actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    policy_actions = np.asarray(jax.device_get(policy_actions), dtype=np.float32)
                    action_source = "policy"

            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(policy_actions)

                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                had_intervene_action = "intervene_action" in info
                stored_actions = policy_actions.copy()

                if had_intervene_action:
                    stored_actions = np.asarray(info.pop("intervene_action"), dtype=np.float32)
                    _, prev_exec_gripper_cmd = map_single_arm_exec_action_to_hardware(
                        stored_actions,
                        prev_exec_gripper_cmd,
                    )

                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                actions, prev_gripper_label = rewrite_single_arm_gripper_action_with_feedback(
                    stored_actions,
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

                if (not FLAGS.minimal_logs) and step % FLAGS.print_period == 0:
                    dbg_exec_actions, _ = map_single_arm_exec_action_to_hardware(
                        policy_actions,
                        prev_exec_gripper_cmd,
                    )
                    since_last_recv = None
                    if network_debug["last_recv_time"] is not None:
                        since_last_recv = round(time.time() - network_debug["last_recv_time"], 3)
                    since_last_apply = None
                    if network_debug["last_apply_time"] is not None:
                        since_last_apply = round(time.time() - network_debug["last_apply_time"], 3)
                    print_blue(
                        f"[actor-step-debug] step={step}, action_source={action_source}, reward={reward}, "
                        f"done={done}, truncated={truncated}, recv_count={network_debug['recv_count']}, "
                        f"applied_count={network_debug['applied_count']}, since_last_recv={since_last_recv}, "
                        f"since_last_apply={since_last_apply}, replay_queue={len(data_store)}, intvn_queue={len(intvn_data_store)}, "
                        f"stored_label={actions[6] if actions.shape[0] == 7 else 'N/A'}, "
                        f"policy_raw={policy_actions[6] if policy_actions.shape[0] == 7 else 'N/A'}, "
                        f"mapped_hw={dbg_exec_actions[6] if dbg_exec_actions.shape[0] == 7 else 'N/A'}, "
                        f"had_intervene_action={had_intervene_action}"
                    )

                if done or truncated:
                    if "episode" not in info:
                        info["episode"] = {}
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps

                    ep_debug = _extract_episode_debug_info(info)
                    _log_info('actor_episode', f"[actor-episode-end] episode={episode_index}, step={step}, return={running_return:.4f}, env_return={ep_debug['return']:.4f}, length={ep_debug['length']}, duration={ep_debug['duration']:.3f}, success={ep_debug['success']}, intervention_count={intervention_count}, intervention_steps={intervention_steps}, recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}", 'yellow')

                    stats = {"environment": info}
                    try:
                        with client_rpc_lock:
                            client.request("send-stats", stats)
                    except Exception as e:
                        _log_info('actor_error', f"[actor-send-stats-error] reason=episode_end, err={e!r}", 'red')

                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    prev_gripper_label = None
                    prev_exec_gripper_cmd = 80.0

                    client_force_event.set()
                    episode_index += 1
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

                if not FLAGS.minimal_logs:
                    print_blue(
                        f"[actor-buffer-save] step={step}, buffer_path={buffer_path}, demo_buffer_path={demo_buffer_path}"
                    )

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                try:
                    with client_rpc_lock:
                        client.request("send-stats", stats)
                except Exception as e:
                    _log_info('actor_error', f"[actor-send-stats-error] reason=timer, step={step}, err={e!r}", 'red')
    finally:
        client_stop_event.set()
        client_force_event.set()
        if client_thread is not None:
            client_thread.join(timeout=1.0)


# ==============================================================
# Learner 逻辑
# --------------------------------------------------------------
# 【合并后说明】
# 1) 保留版本 A 的 latest_checkpoint 判空修复。
# 2) 保留版本 A 的显式 server 线程包装与调试打印。
# 3) 保留“先 publish 初始网络，再等待 replay buffer”的修复。
# ==============================================================
def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger, config, trainer_cfg):
    latest_ckpt = None
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))

    if latest_ckpt is None:
        start_step = 0
    else:
        start_step = int(os.path.basename(latest_ckpt)[11:]) + 1

    step = start_step
    publish_count = 0

    def stats_callback(req_type: str, payload: dict) -> dict:
        assert req_type == "send-stats", f"Invalid request type: {req_type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)

        if isinstance(payload, dict) and "environment" in payload:
            ep_debug = _extract_episode_debug_info(payload["environment"])
            _log_info('learner_env', f"[learner-env-stats] step={step}, return={ep_debug['return']:.4f}, length={ep_debug['length']}, duration={ep_debug['duration']:.3f}, success={ep_debug['success']}, intervention_count={ep_debug['intervention_count']}, intervention_steps={ep_debug['intervention_steps']}", 'yellow')
        elif isinstance(payload, dict) and "timer" in payload:
            if not FLAGS.minimal_logs:
                print_blue(f"[learner-actor-timer] step={step}, timer={payload['timer']}")
        return {}

    import threading
    import traceback

    _log_trainer_config(trainer_cfg, 'learner')
    server = TrainerServer(trainer_cfg, request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)

    def _run_server():
        try:
            if not FLAGS.minimal_logs:
                print_green(
                    f"starting req_rep_server on port {trainer_cfg.port_number} and broadcast_port {getattr(trainer_cfg, 'broadcast_port', 'N/A')}"
                )
            server.req_rep_server.run()
        except Exception as e:
            print_red(f"REQ/REP server crashed: {e!r}")
            traceback.print_exc()
            raise

    server.thread = threading.Thread(target=_run_server, daemon=True)
    server.thread.start()

    time.sleep(1)
    if not FLAGS.minimal_logs:
        print_green(
            f"server thread alive: {server.thread.is_alive()}, replay_buffer={len(replay_buffer)}, demo_buffer={len(demo_buffer)}"
        )

    publish_count += 1
    _publish_network_to_actor(server, agent.state.params, reason="initial_before_warmup", step=start_step)
    if not FLAGS.minimal_logs:
        print_green("sent initial network to actor")

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    warmup_last_publish_t = time.time()
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        if not FLAGS.minimal_logs:
            print_blue(
                f"[learner-buffer-warmup] online={len(replay_buffer)}/{config.training_starts}, demo={len(demo_buffer)}"
            )
        if FLAGS.warmup_publish_period_s > 0 and time.time() - warmup_last_publish_t >= FLAGS.warmup_publish_period_s:
            publish_count += 1
            _publish_network_to_actor(server, agent.state.params, reason='warmup_republish', step=start_step)
            warmup_last_publish_t = time.time()
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    publish_count += 1
    _publish_network_to_actor(server, agent.state.params, reason="initial_after_warmup", step=start_step)
    if not FLAGS.minimal_logs:
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
                agent, critic_info = agent.update(
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
            publish_count += 1
            _publish_network_to_actor(server, agent.state.params, reason="train_periodic_update", step=step)

        if step % FLAGS.print_period == 0:
            printable_update_info = _to_loggable_pytree(update_info)
            critic_loss = printable_update_info.get("critic_loss", printable_update_info.get("critic/critic_loss", "N/A"))
            actor_loss = printable_update_info.get("actor_loss", printable_update_info.get("actor/actor_loss", "N/A"))
            temperature = printable_update_info.get("temperature", printable_update_info.get("alpha", "N/A"))
            _log_info('learner_step', f"[learner-step] step={step}, publish_count={publish_count}, replay_buffer={len(replay_buffer)}, demo_buffer={len(demo_buffer)}, critic_loss={critic_loss}, actor_loss={actor_loss}, temperature={temperature}", 'blue')

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
    if FLAGS.actor:
        visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset>")
        jax_platforms_env = os.environ.get("JAX_PLATFORMS", "<unset>")
        _log_info(
            'main',
            f"[actor-runtime] default_backend={jax.default_backend()}, local_devices={jax.local_devices()}, "
            f"CUDA_VISIBLE_DEVICES={visible_cuda}, JAX_PLATFORMS={jax_platforms_env}, "
            f"XLA_PYTHON_CLIENT_PREALLOCATE={prealloc}, force_actor_cpu={FLAGS.force_actor_cpu}",
            'blue',
        )
        if FLAGS.actor_expect_gpu and (not FLAGS.force_actor_cpu) and jax.default_backend() != 'gpu':
            _log_info(
                'actor_warning',
                "[actor-gpu-warning] actor 当前没有跑在 GPU 上。若你本地 4060 可用，请先确认安装的是 GPU 版 jax/jaxlib、CUDA/cuDNN 版本匹配，并检查 nvidia-smi。",
                'yellow',
            )
    trainer_cfg = _build_trainer_config()
    _log_info('main', f"[main-config] learner={FLAGS.learner}, actor={FLAGS.actor}, ip={FLAGS.ip}, checkpoint_path={FLAGS.checkpoint_path}, print_period={FLAGS.print_period}, trainer_port={getattr(trainer_cfg, 'port_number', None)}, trainer_broadcast_port={getattr(trainer_cfg, 'broadcast_port', None)}, minimal_logs={FLAGS.minimal_logs}", 'blue')

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

    if not FLAGS.minimal_logs:
        print_blue(
            f"[main-demo-infer] observation_space={observation_space}, action_space={action_space}, "
            f"sample_action_shape={np.asarray(sample_action).shape}"
        )

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

    if not FLAGS.minimal_logs:
        print_blue(f"[main-agent-init] setup_mode={config.setup_mode}, include_grasp_penalty={include_grasp_penalty}")

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

    if not FLAGS.minimal_logs:
        print_blue(
            f"[main-agent-device] is_learner={FLAGS.learner}, init_signature={_format_signature(_tree_debug_signature(agent.state.params))}"
        )

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
        _log_info('checkpoint', f"[main-checkpoint-restore] signature={_format_signature(_tree_debug_signature(ckpt.params if hasattr(ckpt, 'params') else ckpt))}", 'blue')

        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt:
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            _log_info('checkpoint', f"Loaded previous checkpoint at step {ckpt_number}.", 'green')

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

        if not FLAGS.minimal_logs:
            print_green(f"demo buffer size: {len(demo_buffer)}")
            print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            if not FLAGS.minimal_logs:
                print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            if not FLAGS.minimal_logs:
                print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        _log_info('main', '[main] starting learner loop', 'green')
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            config=config,
            trainer_cfg=trainer_cfg,
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

        _log_info('main', '[main] starting actor loop', 'green')
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            config,
            trainer_cfg=trainer_cfg,
        )


if __name__ == "__main__":
    app.run(main)









