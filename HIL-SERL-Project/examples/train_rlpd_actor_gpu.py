#rlpd的继续说明，
#gpu后，现在8~14 it/s，和之前 CPU 的 3~4 it/s 相比，符合这个变化。
#learner更新50步网络后，会发给actor最新的网络，所以actor在接受新网络后会卡顿（0.05级别的延迟）一下再继续输出动作
# 当前 step 先按当前已经生效的网络跑完
# 新网络先收进来，到下一次安全点 loop_start 再切换，下一步开始才用新网络出动作
# 对一个具体 step 来说，动作要么是：
#  旧网络输出的完整动作
#  要么是新网络输出的完整动作，不会在“收到包的那一瞬间”还持续记录一串冻结动作。


#conda activate hilserl_actor_gpu_py310
#cd HIL-SERL/HIL-SERL-Project/examples
#  /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_actor_gpu_ros.sh

# ssh -p 2122  -L 5588:localhost:5588  -L 5589:localhost:5589  lixiang@service.qich.top

"""
train_rlpd.py

本版核心目标：
1) 网络更新改成接近官方 HIL-SERL 结构：
   - actor 注册 client.recv_network_callback(update_params)
   - 收到网络后只缓存为 pending_params；
               episode 内不替换 actor 参数；
               episode 结束 / reset 后 / 下一条 episode 第一帧前再 apply pending 网络。
   - episode 内不后台 update、不周期性 update、不 loop_start apply
   - episode 结束后 client.update()
   - reset 后等待网络更新，再开始下一 episode 第一帧动作

2) VR 介入期间尽量不被网络同步打断，保证 env.step 连续记录 transition。

3) learner 端增加终端训练指标打印：critic / grasp_critic / actor / temperature / timer。

4) 所有可调配置集中在文件前方。
"""

# =============================================================================
# 0. 文件前方总配置区
# =============================================================================

# ---- 官方式 actor 网络更新配置 ----
OFFICIAL_EPISODE_NETWORK_UPDATE = True
# episode 结束后 update 一次，然后 reset；reset 后再等待网络，之后才输出下一步动作。
WAIT_NETWORK_BEFORE_FIRST_ACTION = True
WAIT_NETWORK_AFTER_EVERY_RESET = True
# True: 等到 recv_count 增加，也就是确实收到一版 learner 发布的网络。
# 如果你想“保证拿到新网络再动”，保持 True。
NETWORK_WAIT_REQUIRE_NEW = True
# None 表示无限等待，最符合“必须保证更新到网络再输出动作”。
# 如果不想因为 learner 没 publish 而永久等待，可以改成 30.0（等待30s）。
NETWORK_WAIT_TIMEOUT_SEC = None  # 如需避免 learner 暂停导致 actor 永久等待，可改成 30.0
NETWORK_WAIT_RETRY_SLEEP_SEC = 0.10
# episode 结束后是否先 update，再 reset；保留 True。
UPDATE_AT_EPISODE_END_BEFORE_RESET = True
# reset 后等待网络前，是否先做一次 client.update。
UPDATE_AFTER_RESET_BEFORE_WAIT = True

# ---- learner 网络发布配置 ----
# learner server 启动后先发一次初始网络，让 actor 初始等待能收到。
PUBLISH_INITIAL_NETWORK_BEFORE_WARMUP = True
# replay warmup 结束后再发一次。
PUBLISH_NETWORK_AFTER_WARMUP = True
# warmup 阶段定期重发当前网络，避免 actor 初始化等待时错过网络。
DEFAULT_WARMUP_PUBLISH_PERIOD_S = 5

# ---- learner 终端训练指标打印配置 ----
PRINT_LEARNER_TRAIN_DEBUG = True
PRINT_LEARNER_TRAIN_DEBUG_EVERY_LOG_PERIOD = True

# ---- Actor -> Learner observation 裁剪配置 ----
# None: 自动使用 config.image_keys。
# "all": 不裁剪图像。
# ["head_rgb", "right_wrist_rgb"]: 显式指定。
ACTOR_TO_LEARNER_IMAGE_KEYS = None
ACTOR_TO_LEARNER_EXTRA_OBS_KEYS = ["state"]
ACTOR_TO_LEARNER_STRICT_KEYS = True

# ---- 单臂 action 存储约定 ----
ARM_ACTION_LOW = -1.0
ARM_ACTION_HIGH = 1.0
DEFAULT_GRASP_PENALTY_VALUE = -0.02



# ---- GPU actor 预配置默认值；注意这些值会在 import jax 前通过 sys.argv 生效 ----
DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES = "0"
DEFAULT_ACTOR_DISABLE_PREALLOCATE = True


# =============================================================================
# 1. import 前 actor GPU 环境变量处理
# =============================================================================

import os
import sys
import glob
import time
import copy
import pickle as pkl
import threading


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


if _is_actor_mode_from_argv():
    force_actor_cpu = _raw_flag_bool("force_actor_cpu", False)
    actor_cuda_visible_devices = _raw_flag_value(
        "actor_cuda_visible_devices",
        DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES,
    )
    actor_disable_preallocate = _raw_flag_bool(
        "actor_disable_preallocate",
        DEFAULT_ACTOR_DISABLE_PREALLOCATE,
    )
    actor_mem_fraction = _raw_flag_value("actor_mem_fraction", None)
    actor_jax_platforms = _raw_flag_value("actor_jax_platforms", None)

    if force_actor_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        if actor_cuda_visible_devices not in (None, "", "auto"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_cuda_visible_devices)
        if actor_disable_preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if actor_mem_fraction not in (None, "", "0", 0):
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(actor_mem_fraction)
        if actor_jax_platforms not in (None, ""):
            os.environ["JAX_PLATFORMS"] = str(actor_jax_platforms)


# =============================================================================
# 2. imports
# =============================================================================

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium import spaces
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

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

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config


# =============================================================================
# 3. flags
# =============================================================================

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Experiment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this process is the learner.")
flags.DEFINE_boolean("actor", False, "Whether this process is the actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo data for learner bootstrap.")
flags.DEFINE_string("checkpoint_path", "./rlpd_checkpoints", "Path to save checkpoints / buffers.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save evaluation videos.")
flags.DEFINE_boolean("debug", False, "Debug mode, disables wandb upload.")

flags.DEFINE_integer("print_period", 50, "How often to print actor debug lines.")
flags.DEFINE_integer("request_timeout_ms", 15000, "TrainerClient timeout in ms.")

# 下面三个 flag 保留兼容旧命令；本版 official episode update 模式中 actor 不再按 step/background update。
flags.DEFINE_integer("client_update_period", 0, "Deprecated in official episode update mode.")
flags.DEFINE_float("client_update_interval_sec", 0.5, "Deprecated in official episode update mode.")
flags.DEFINE_boolean("client_update_background", False, "Deprecated in official episode update mode.")

flags.DEFINE_integer("warmup_publish_period_s", DEFAULT_WARMUP_PUBLISH_PERIOD_S, "Warmup network republish period.")
flags.DEFINE_integer("trainer_port", 0, "Override TrainerConfig.port_number when > 0.")
flags.DEFINE_integer("trainer_broadcast_port", 0, "Override TrainerConfig.broadcast_port when > 0.")
flags.DEFINE_boolean("print_trainer_config", True, "Print trainer config and ports.")
flags.DEFINE_boolean("minimal_logs", True, "Only keep important logs.")

flags.DEFINE_boolean("force_actor_cpu", False, "Force actor to run on CPU.")
flags.DEFINE_string("actor_cuda_visible_devices", DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES, "CUDA_VISIBLE_DEVICES for actor.")
flags.DEFINE_boolean("actor_disable_preallocate", DEFAULT_ACTOR_DISABLE_PREALLOCATE, "Disable JAX GPU preallocation on actor.")
flags.DEFINE_float("actor_mem_fraction", 0.0, "Optional XLA_PYTHON_CLIENT_MEM_FRACTION.")
flags.DEFINE_string("actor_jax_platforms", "", "Optional JAX_PLATFORMS override.")
flags.DEFINE_boolean("actor_expect_gpu", True, "Warn if actor does not start on GPU.")

flags.DEFINE_string(
    "actor_to_learner_image_keys",
    "",
    "Comma-separated image keys. Empty uses top config; 'all' disables pruning.",
)
flags.DEFINE_boolean(
    "actor_to_learner_strict_keys",
    True,
    "If True, error when selected actor_to_learner_image_keys are missing.",
)
flags.DEFINE_float(
    "grasp_penalty_value",
    DEFAULT_GRASP_PENALTY_VALUE,
    "Penalty written into buffers when final action[6] is close/open event.",
)


# =============================================================================
# 4. JAX sharding
# =============================================================================

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


# =============================================================================
# 5. logging helpers
# =============================================================================

def print_green(x):
    print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    print("\033[93m {}\033[00m".format(x))


def print_blue(x):
    print("\033[94m {}\033[00m".format(x))


def print_red(x):
    print("\033[91m {}\033[00m".format(x))


def _suggest_ssh_forward_command(reqrep_port: int, broadcast_port: int) -> str:
    return (
        f"ssh -p 2122 -L {reqrep_port}:localhost:{reqrep_port} "
        f"-L {broadcast_port}:localhost:{broadcast_port} lixiang@service.qich.top"
    )


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
    fn = {"blue": print_blue, "green": print_green, "yellow": print_yellow, "red": print_red}.get(color, print)
    fn(msg)


def _as_python_scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("empty array cannot be converted to scalar")
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
        if name.startswith("_"):
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
    if FLAGS.trainer_port > 0 and hasattr(cfg, "port_number"):
        cfg.port_number = FLAGS.trainer_port
    if FLAGS.trainer_broadcast_port > 0 and hasattr(cfg, "broadcast_port"):
        cfg.broadcast_port = FLAGS.trainer_broadcast_port
    return cfg


def _log_trainer_config(cfg, role):
    if not FLAGS.print_trainer_config or FLAGS.minimal_logs:
        return
    _log_info("main", f"[{role}-trainer-config] {_trainer_config_dict(cfg)}", "blue")


# =============================================================================
# 6. pytree / network helpers
# =============================================================================

def _tree_debug_signature(tree, max_leaves=8, elems_per_leaf=8):
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
                sampled.extend(arr.reshape(-1)[:elems_per_leaf].astype(np.float64).tolist())
    sample_arr = np.asarray(sampled, dtype=np.float64) if sampled else np.zeros((1,), dtype=np.float64)
    return {
        "leaf_count": len(leaves),
        "total_params": total_params,
        "checksum": float(sample_arr.sum()),
        "abs_mean": float(np.mean(np.abs(sample_arr))),
        "sample_std": float(np.std(sample_arr)),
        "sample_head": [round(float(x), 6) for x in sample_arr[:6]],
        "leaf_shapes": leaf_shapes[:4],
    }


def _format_signature(sig):
    if sig is None:
        return "None"
    return (
        f"leafs={sig['leaf_count']}, total_params={sig['total_params']}, "
        f"checksum={sig['checksum']:.6f}, abs_mean={sig['abs_mean']:.6f}, "
        f"sample_std={sig['sample_std']:.6f}, head={sig['sample_head']}, shapes={sig['leaf_shapes']}"
    )


def _block_until_ready_tree(tree):
    def _block(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x
    return jax.tree_util.tree_map(_block, tree)


def _to_host_pytree(tree):
    def _convert(x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return np.asarray(jax.device_get(x))
        return x
    return jax.tree_util.tree_map(_convert, tree)


def _to_loggable_pytree(tree):
    def _convert(x):
        if isinstance(x, (jax.Array, jnp.ndarray)):
            x = np.asarray(jax.device_get(x))
            if x.shape == ():
                return x.item()
            return x
        return x
    return jax.tree_util.tree_map(_convert, tree)


def _publish_network_to_actor(server, params, *, reason="periodic_update", step=None):
    t0 = time.time()
    params = _block_until_ready_tree(params)
    params = _to_host_pytree(params)
    sig = _tree_debug_signature(params)
    server.publish_network(params)
    dt = time.time() - t0
    _log_info(
        "learner_publish",
        f"[learner-network-publish] reason={reason}, step={step}, cost={dt:.3f}s, {_format_signature(sig)}",
        "blue",
    )
    return sig


def _save_checkpoint_host(checkpoint_path, state, step, keep=100):
    t0 = time.time()
    state = _block_until_ready_tree(state)
    state = _to_host_pytree(state)
    sig = _tree_debug_signature(state.params if hasattr(state, "params") else state)
    checkpoints.save_checkpoint(os.path.abspath(checkpoint_path), state, step=step, keep=keep)
    dt = time.time() - t0
    _log_info(
        "learner_checkpoint",
        f"[learner-checkpoint-save] step={step}, cost={dt:.3f}s, path={os.path.abspath(checkpoint_path)}, {_format_signature(sig)}",
        "blue",
    )


# =============================================================================
# 7. gripper / action storage helpers
# =============================================================================

def extract_gripper_feedback_from_obs(obs):
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


def infer_stable_gripper_state_from_feedback(gripper_feedback, prev_state, close_max=30.0, open_min=70.0):
    if gripper_feedback is None:
        return prev_state
    x = float(gripper_feedback)
    if x <= close_max:
        return -1
    if x >= open_min:
        return +1
    return prev_state


def rewrite_single_arm_gripper_action_to_three_value(action, obs, next_obs, prev_stable_state):
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.shape[0] != 7:
        return action, prev_stable_state

    prev_feedback = extract_gripper_feedback_from_obs(obs)
    next_feedback = extract_gripper_feedback_from_obs(next_obs)
    prev_state = infer_stable_gripper_state_from_feedback(prev_feedback, prev_stable_state)
    next_state = infer_stable_gripper_state_from_feedback(next_feedback, prev_state)

    gripper_event = 0.0
    if prev_state is not None and next_state is not None:
        if prev_state == +1 and next_state == -1:
            gripper_event = -1.0
        elif prev_state == -1 and next_state == +1:
            gripper_event = +1.0
        else:
            gripper_event = 0.0
    action[6] = np.float32(gripper_event)
    return action.astype(np.float32), next_state


def map_single_arm_exec_action_to_hardware(action, prev_hw_cmd, close_cmd=10.0, open_cmd=80.0, deadband=0.5):
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.shape[0] != 7:
        return action, prev_hw_cmd
    grip = float(action[6])
    if 0.0 <= grip <= 100.0 and abs(grip) > 5.0:
        hw_cmd = grip
    else:
        if grip >= deadband:
            hw_cmd = open_cmd
        elif grip <= -deadband:
            hw_cmd = close_cmd
        else:
            hw_cmd = prev_hw_cmd
    exec_action = action.copy()
    exec_action[6] = np.float32(hw_cmd)
    return exec_action, float(hw_cmd)


def describe_gripper_three_value(x):
    x = float(x)
    if x <= -0.5:
        return "close(-1)"
    if x >= 0.5:
        return "open(+1)"
    return "hold(0)"


def sanitize_single_arm_action_for_storage(action, *, quantize_gripper=True, source="unknown", return_changed=False):
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if a.shape[0] != 7:
        if return_changed:
            return a.astype(np.float32), False, False
        return a.astype(np.float32)
    before = a.copy()
    a[:6] = np.clip(a[:6], ARM_ACTION_LOW, ARM_ACTION_HIGH)
    if quantize_gripper:
        g = float(a[6])
        if g <= -0.5:
            a[6] = -1.0
        elif g >= 0.5:
            a[6] = 1.0
        else:
            a[6] = 0.0
    else:
        a[6] = np.clip(a[6], ARM_ACTION_LOW, ARM_ACTION_HIGH)
    was_out_of_range = bool(np.any(np.abs(before[:6]) > 1.0001))
    changed = bool(not np.allclose(before, a, atol=1e-6, rtol=1e-6))
    if return_changed:
        return a.astype(np.float32), changed, was_out_of_range
    return a.astype(np.float32)


def sanitize_transition_action_for_storage(transition, *, source="transition", return_changed=False):
    trans = copy.deepcopy(transition)
    if "actions" not in trans:
        if return_changed:
            return trans, False, False
        return trans
    clean_action, changed, was_out = sanitize_single_arm_action_for_storage(
        trans["actions"], quantize_gripper=True, source=source, return_changed=True
    )
    trans["actions"] = clean_action
    if return_changed:
        return trans, changed, was_out
    return trans


def sanitize_transition_list_for_storage(transitions, *, source="transitions", print_summary=True):
    clean = []
    changed_count = 0
    out_count = 0
    for idx, transition in enumerate(transitions):
        trans, changed, was_out = sanitize_transition_action_for_storage(
            transition, source=f"{source}[{idx}]", return_changed=True
        )
        clean.append(trans)
        changed_count += int(changed)
        out_count += int(was_out)
    if print_summary:
        _log_info(
            "main",
            f"[action-sanitize] source={source}, n={len(transitions)}, changed={changed_count}, arm_out_of_range={out_count}",
            "yellow" if out_count > 0 else "green",
        )
    return clean


def recompute_grasp_penalty_from_stored_action(action, penalty_value=None):
    if penalty_value is None:
        penalty_value = float(getattr(FLAGS, "grasp_penalty_value", DEFAULT_GRASP_PENALTY_VALUE))
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        return 0.0
    g = float(a[6])
    if g <= -0.5 or g >= 0.5:
        return float(penalty_value)
    return 0.0


def sync_grasp_penalty_with_stored_action(transition, *, penalty_value=None, source="unknown", preserve_raw_in_infos=True):
    if not isinstance(transition, dict) or "actions" not in transition:
        return transition
    expected = recompute_grasp_penalty_from_stored_action(transition["actions"], penalty_value=penalty_value)
    old_top = transition.get("grasp_penalty", None)
    transition["grasp_penalty"] = float(expected)
    infos = transition.get("infos", transition.get("info", None))
    if isinstance(infos, dict):
        if preserve_raw_in_infos:
            if "grasp_penalty" in infos and "env_grasp_penalty_raw" not in infos:
                infos["env_grasp_penalty_raw"] = _safe_float(infos["grasp_penalty"], 0.0)
            if old_top is not None and "top_level_grasp_penalty_raw" not in infos:
                infos["top_level_grasp_penalty_raw"] = _safe_float(old_top, 0.0)
        infos["grasp_penalty"] = float(expected)
        infos["grasp_penalty_source"] = f"recomputed_from_final_action:{source}"
        if "infos" in transition:
            transition["infos"] = infos
        elif "info" in transition:
            transition["info"] = infos
    return transition


def sync_transition_list_grasp_penalty(transitions, *, source="transitions", penalty_value=None, print_summary=True):
    if penalty_value is None:
        penalty_value = float(getattr(FLAGS, "grasp_penalty_value", DEFAULT_GRASP_PENALTY_VALUE))
    synced = []
    mismatch_before = 0
    nonzero_after = 0
    event_count = 0
    hold_penalty_after = 0
    for idx, transition in enumerate(transitions):
        trans = copy.deepcopy(transition)
        old = None
        if isinstance(trans, dict):
            if "grasp_penalty" in trans:
                old = _safe_float(trans.get("grasp_penalty"), 0.0)
            else:
                infos = trans.get("infos", trans.get("info", None))
                if isinstance(infos, dict) and "grasp_penalty" in infos:
                    old = _safe_float(infos.get("grasp_penalty"), 0.0)
        expected = recompute_grasp_penalty_from_stored_action(trans.get("actions", np.zeros(0)), penalty_value)
        if old is not None and abs(float(old) - float(expected)) > 1e-6:
            mismatch_before += 1
        trans = sync_grasp_penalty_with_stored_action(
            trans, penalty_value=penalty_value, source=f"{source}[{idx}]"
        )
        a = np.asarray(trans.get("actions", []), dtype=np.float32).reshape(-1)
        if a.shape[0] == 7:
            g = float(a[6])
            if abs(g) > 0.5:
                event_count += 1
            if abs(float(trans.get("grasp_penalty", 0.0))) > 1e-8:
                nonzero_after += 1
                if abs(g) <= 0.5:
                    hold_penalty_after += 1
        synced.append(trans)
    if print_summary:
        color = "green" if hold_penalty_after == 0 and nonzero_after == event_count else "yellow"
        _log_info(
            "main",
            f"[grasp-penalty-sync] source={source}, n={len(transitions)}, penalty_value={penalty_value}, "
            f"mismatch_before={mismatch_before}, gripper_event_count={event_count}, "
            f"nonzero_after={nonzero_after}, hold_penalty_after={hold_penalty_after}",
            color,
        )
    return synced


# =============================================================================
# 8. observation pruning / demo space helpers
# =============================================================================

DEFAULT_KNOWN_IMAGE_KEYS = {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"}


def _parse_comma_keys(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    value = str(value).strip()
    if value == "":
        return None
    if value.lower() in ("none", "config", "default"):
        return None
    if value.lower() in ("all", "*"):
        return "all"
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_actor_to_learner_image_keys(config):
    cli_value = _parse_comma_keys(getattr(FLAGS, "actor_to_learner_image_keys", ""))
    if cli_value == "all":
        return "all"
    if cli_value is not None:
        return cli_value
    top_value = _parse_comma_keys(ACTOR_TO_LEARNER_IMAGE_KEYS)
    if top_value == "all":
        return "all"
    if top_value is not None:
        return top_value
    return list(getattr(config, "image_keys", []))


def validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys):
    if actor_to_learner_image_keys == "all":
        return
    policy_keys = list(getattr(config, "image_keys", []))
    selected = list(actor_to_learner_image_keys or [])
    missing_policy_keys = [k for k in policy_keys if k not in selected]
    if missing_policy_keys:
        raise ValueError(
            "actor_to_learner_image_keys 必须包含所有 config.image_keys。"
            f" missing={missing_policy_keys}, selected={selected}, policy_keys={policy_keys}"
        )
    env_keys = list(getattr(config, "ENV_IMAGE_KEYS", []))
    if env_keys:
        missing_env_keys = [k for k in selected if k not in env_keys]
        if missing_env_keys:
            raise ValueError(
                "actor_to_learner_image_keys 中有 key 不在 ENV_IMAGE_KEYS 里。"
                f" missing={missing_env_keys}, selected={selected}, ENV_IMAGE_KEYS={env_keys}"
            )


def get_known_image_keys(config, actor_to_learner_image_keys=None):
    keys = set(DEFAULT_KNOWN_IMAGE_KEYS)
    for attr in ["ENV_IMAGE_KEYS", "DISPLAY_IMAGE_KEYS", "image_keys", "classifier_keys"]:
        value = getattr(config, attr, None)
        if isinstance(value, (list, tuple)):
            keys.update([str(x) for x in value])
    if isinstance(actor_to_learner_image_keys, (list, tuple)):
        keys.update([str(x) for x in actor_to_learner_image_keys])
    return keys


def _get_obs_value_by_key(obs, key):
    if key in obs:
        return obs[key]
    images = obs.get("images", None)
    if isinstance(images, dict) and key in images:
        return images[key]
    raise KeyError(f"obs 中找不到图像 key={key}, obs.keys={list(obs.keys())}")


def prune_observation_for_actor_to_learner(obs, actor_to_learner_image_keys, config, *, strict=True):
    if obs is None or not isinstance(obs, dict):
        return obs
    if actor_to_learner_image_keys == "all":
        return obs
    image_keys = list(actor_to_learner_image_keys or [])
    known_image_keys = get_known_image_keys(config, image_keys)
    extra_keys = set(ACTOR_TO_LEARNER_EXTRA_OBS_KEYS or [])
    pruned = {}
    for key in image_keys:
        try:
            pruned[key] = _get_obs_value_by_key(obs, key)
        except KeyError:
            if strict:
                raise
            print_yellow(f"⚠️ actor_to_learner_image_key={key} 不在 obs 中，已跳过。obs.keys={list(obs.keys())}")
    for key, value in obs.items():
        if key == "images" or key in pruned:
            continue
        is_known_image = key in known_image_keys
        looks_like_image = (
            key.endswith("_rgb")
            or key.endswith("_depth")
            or key.endswith("_image")
            or (hasattr(value, "shape") and len(np.asarray(value).shape) >= 3 and str(key).lower() != "state")
        )
        if key in extra_keys or (not is_known_image and not looks_like_image):
            pruned[key] = value
    return pruned


def prune_transition_for_actor_to_learner(transition, actor_to_learner_image_keys, config, *, strict=True):
    trans = copy.deepcopy(transition)
    if "observations" in trans:
        trans["observations"] = prune_observation_for_actor_to_learner(
            trans["observations"], actor_to_learner_image_keys, config, strict=strict
        )
    if "next_observations" in trans:
        trans["next_observations"] = prune_observation_for_actor_to_learner(
            trans["next_observations"], actor_to_learner_image_keys, config, strict=strict
        )
    return trans


def prune_transition_list_for_actor_to_learner(transitions, actor_to_learner_image_keys, config, *, source="transitions", strict=True, print_summary=True):
    if actor_to_learner_image_keys == "all":
        if print_summary:
            _log_info("main", f"[obs-prune] source={source}, mode=all, n={len(transitions)}, 不裁剪图像", "yellow")
        return transitions
    clean = [
        prune_transition_for_actor_to_learner(t, actor_to_learner_image_keys, config, strict=strict)
        for t in transitions
    ]
    if print_summary:
        keys = []
        if len(clean) > 0 and isinstance(clean[0].get("observations", None), dict):
            keys = list(clean[0]["observations"].keys())
        _log_info(
            "main",
            f"[obs-prune] source={source}, actor_to_learner_image_keys={actor_to_learner_image_keys}, n={len(clean)}, stored_obs_keys={keys}",
            "green",
        )
    return clean


def print_observation_keys_summary(transition_or_obs, *, name="obs"):
    obs = transition_or_obs.get("observations", transition_or_obs) if isinstance(transition_or_obs, dict) else transition_or_obs
    if isinstance(obs, dict):
        _log_info("main", f"[obs-summary] {name}: keys={list(obs.keys())}", "green")
        for k, v in obs.items():
            try:
                arr = np.asarray(v)
                _log_info("main", f"[obs-summary]   {k}: shape={arr.shape}, dtype={arr.dtype}", "green")
            except Exception:
                _log_info("main", f"[obs-summary]   {k}: type={type(v)}", "green")


def infer_space_from_value(x):
    if isinstance(x, dict):
        return spaces.Dict({k: infer_space_from_value(v) for k, v in x.items()})
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return spaces.Box(low=0, high=255, shape=arr.shape, dtype=np.uint8)
    if np.issubdtype(arr.dtype, np.bool_):
        return spaces.Box(low=0, high=1, shape=arr.shape, dtype=np.bool_)
    if np.issubdtype(arr.dtype, np.integer):
        return spaces.Box(low=np.iinfo(arr.dtype).min, high=np.iinfo(arr.dtype).max, shape=arr.shape, dtype=arr.dtype)
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


def build_spaces_and_samples_from_demos(paths, config, actor_to_learner_image_keys):
    sample_transition = sanitize_transition_action_for_storage(get_first_valid_transition(paths), source="sample_demo_infer")
    sample_transition = prune_transition_for_actor_to_learner(
        sample_transition, actor_to_learner_image_keys, config, strict=FLAGS.actor_to_learner_strict_keys
    )
    observation_space = infer_space_from_value(sample_transition["observations"])
    sample_action = np.asarray(sample_transition["actions"], dtype=np.float32).reshape(-1)
    if sample_action.shape[0] == 7:
        action_space = spaces.Box(low=-1.0, high=1.0, shape=sample_action.shape, dtype=np.float32)
    else:
        action_space = infer_space_from_value(sample_action)
    sample_obs = sample_transition["observations"]
    print_observation_keys_summary(sample_obs, name="sample_obs_for_agent_and_buffer")
    return observation_space, action_space, sample_obs, sample_action


def _extract_episode_debug_info(info):
    episode = info.get("episode", {}) if isinstance(info, dict) else {}
    ep_return = _safe_float(episode.get("r", episode.get("return", 0.0)))
    raw_success = info.get("success", info.get("is_success", info.get("succeed", 0.0)))
    success = max(_safe_float(raw_success, 0.0), float(ep_return > 0.0))
    return {
        "return": ep_return,
        "length": _safe_int(episode.get("l", episode.get("length", 0)), 0),
        "duration": _safe_float(episode.get("t", episode.get("time", 0.0))),
        "success": success,
        "intervention_count": _safe_int(episode.get("intervention_count", 0), 0),
        "intervention_steps": _safe_int(episode.get("intervention_steps", 0), 0),
    }


# =============================================================================
# 9. learner terminal metric helpers
# =============================================================================

def _fmt_metric(info, key, default=None):
    if info is None or key not in info:
        return default
    try:
        x = np.asarray(jax.device_get(info[key]))
        if x.size == 0:
            return default
        return float(x.reshape(-1)[0])
    except Exception:
        return default


def _format_metric_value(x, digits=6):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _print_learner_training_debug(step, update_info, critics_info, timer):
    if not PRINT_LEARNER_TRAIN_DEBUG:
        return

    critic_loss = _fmt_metric(update_info, "critic/critic_loss") or _fmt_metric(critics_info, "critic/critic_loss")
    predicted_qs = _fmt_metric(update_info, "critic/predicted_qs") or _fmt_metric(critics_info, "critic/predicted_qs")
    target_qs = _fmt_metric(update_info, "critic/target_qs") or _fmt_metric(critics_info, "critic/target_qs")
    rewards = _fmt_metric(update_info, "critic/rewards") or _fmt_metric(critics_info, "critic/rewards")

    grasp_loss = _fmt_metric(update_info, "grasp_critic/grasp_critic_loss") or _fmt_metric(critics_info, "grasp_critic/grasp_critic_loss")
    grasp_pred_q = _fmt_metric(update_info, "grasp_critic/predicted_grasp_qs") or _fmt_metric(critics_info, "grasp_critic/predicted_grasp_qs")
    grasp_target_q = _fmt_metric(update_info, "grasp_critic/target_grasp_qs") or _fmt_metric(critics_info, "grasp_critic/target_grasp_qs")
    grasp_rewards = _fmt_metric(update_info, "grasp_critic/grasp_rewards") or _fmt_metric(critics_info, "grasp_critic/grasp_rewards")

    actor_loss = _fmt_metric(update_info, "actor/actor_loss")
    entropy = _fmt_metric(update_info, "actor/entropy")
    temperature = _fmt_metric(update_info, "actor/temperature")
    temp_loss = _fmt_metric(update_info, "temperature/temperature_loss")

    times = timer.get_average_times()
    _log_info(
        "learner_step",
        "[learner-train-debug] "
        f"step={step} | "
        f"critic_loss={_format_metric_value(critic_loss)} pred_q={_format_metric_value(predicted_qs)} "
        f"target_q={_format_metric_value(target_qs)} reward_mean={_format_metric_value(rewards)} | "
        f"grasp_loss={_format_metric_value(grasp_loss)} grasp_pred_q={_format_metric_value(grasp_pred_q)} "
        f"grasp_target_q={_format_metric_value(grasp_target_q)} grasp_reward_mean={_format_metric_value(grasp_rewards)} | "
        f"actor_loss={_format_metric_value(actor_loss)} entropy={_format_metric_value(entropy)} "
        f"temperature={_format_metric_value(temperature)} temp_loss={_format_metric_value(temp_loss)} | "
        f"timer_train={_format_metric_value(times.get('train'), 4)} "
        f"timer_train_critics={_format_metric_value(times.get('train_critics'), 4)} "
        f"timer_sample_replay={_format_metric_value(times.get('sample_replay_buffer'), 4)}",
        "blue",
    )



# =============================================================================
# 9.5 actor buffer save/load helpers
# =============================================================================

BUFFER_FILE_PREFIX = "transitions_"
BUFFER_FILE_SUFFIX = ".pkl"


def _extract_numeric_step_from_transition_file(path):
    """
    只接受旧版周期 buffer 文件名：
        transitions_1000.pkl
        transitions_2000.pkl
        transitions_3000.pkl

    明确忽略：
        transitions_197_final.pkl
        transitions_final.pkl
        任何非纯数字 step 文件
    """
    name = os.path.basename(path)
    if not (name.startswith(BUFFER_FILE_PREFIX) and name.endswith(BUFFER_FILE_SUFFIX)):
        return None

    stem = name[len(BUFFER_FILE_PREFIX):-len(BUFFER_FILE_SUFFIX)]
    if not stem.isdigit():
        return None

    return int(stem)


def _list_numeric_transition_files(buffer_dir):
    """
    返回 [(step, path), ...]，只包含纯数字 step 的周期保存文件。
    """
    if not buffer_dir or not os.path.exists(buffer_dir):
        return []

    out = []
    for path in glob.glob(os.path.join(buffer_dir, f"{BUFFER_FILE_PREFIX}*{BUFFER_FILE_SUFFIX}")):
        step = _extract_numeric_step_from_transition_file(path)
        if step is not None:
            out.append((step, path))

    out.sort(key=lambda x: x[0])
    return out


def _infer_actor_start_step_from_numeric_buffers(checkpoint_path):
    """
    actor 恢复 step 只看纯数字周期 buffer：
        buffer/transitions_1000.pkl
        buffer/transitions_2000.pkl

    不再读取 *_final.pkl，也不会因为 transitions_197_final.pkl 报错。
    """
    if not checkpoint_path:
        return 0

    buffer_dir = os.path.join(checkpoint_path, "buffer")
    numeric_files = _list_numeric_transition_files(buffer_dir)
    if not numeric_files:
        return 0

    return numeric_files[-1][0] + 1


def _load_numeric_transition_files_from_dir(
    buffer_dir,
    *,
    actor_to_learner_image_keys,
    config,
    source_prefix,
    strict,
    penalty_value,
):
    """
    learner 恢复历史 online/demo buffer 时，只读取纯数字周期文件。
    这会读取：
        transitions_1000.pkl
        transitions_2000.pkl
        transitions_3000.pkl

    会忽略：
        transitions_197_final.pkl
        transitions_foo.pkl
    """
    loaded = []
    numeric_files = _list_numeric_transition_files(buffer_dir)

    if numeric_files:
        _log_info(
            "main",
            f"[buffer-load] source={source_prefix}, numeric_files={[os.path.basename(p) for _, p in numeric_files]}",
            "green",
        )
    else:
        _log_info(
            "main",
            f"[buffer-load] source={source_prefix}, no numeric periodic buffer files found in {buffer_dir}",
            "yellow",
        )

    for step, file in numeric_files:
        with open(file, "rb") as f:
            transitions = pkl.load(f)

        transitions = prune_transition_list_for_actor_to_learner(
            transitions,
            actor_to_learner_image_keys,
            config,
            source=f"{source_prefix}:{file}",
            strict=strict,
            print_summary=False,
        )
        transitions = sanitize_transition_list_for_storage(
            transitions,
            source=f"{source_prefix}:{file}",
            print_summary=False,
        )
        transitions = sync_transition_list_grasp_penalty(
            transitions,
            source=f"{source_prefix}:{file}",
            penalty_value=penalty_value,
            print_summary=False,
        )
        loaded.extend(transitions)

    return loaded


def _save_periodic_actor_buffers(
    *,
    checkpoint_path,
    step,
    transitions,
    demo_transitions,
    penalty_value,
):
    """
    旧版逻辑：只在 step 命中 config.buffer_period 时保存：
        buffer/transitions_{step}.pkl
        demo_buffer/transitions_{step}.pkl

    保存后由调用方清空内存 list。
    不保存 *_final.pkl。
    """
    buffer_path = os.path.join(checkpoint_path, "buffer")
    demo_buffer_path = os.path.join(checkpoint_path, "demo_buffer")
    os.makedirs(buffer_path, exist_ok=True)
    os.makedirs(demo_buffer_path, exist_ok=True)

    transitions_to_save = sync_transition_list_grasp_penalty(
        sanitize_transition_list_for_storage(
            transitions,
            source=f"actor_buffer_save_step_{step}",
            print_summary=True,
        ),
        source=f"actor_buffer_save_step_{step}",
        penalty_value=penalty_value,
        print_summary=True,
    )

    demo_to_save = sync_transition_list_grasp_penalty(
        sanitize_transition_list_for_storage(
            demo_transitions,
            source=f"actor_demo_buffer_save_step_{step}",
            print_summary=True,
        ),
        source=f"actor_demo_buffer_save_step_{step}",
        penalty_value=penalty_value,
        print_summary=True,
    )

    buffer_file = os.path.join(buffer_path, f"{BUFFER_FILE_PREFIX}{step}{BUFFER_FILE_SUFFIX}")
    demo_file = os.path.join(demo_buffer_path, f"{BUFFER_FILE_PREFIX}{step}{BUFFER_FILE_SUFFIX}")

    with open(buffer_file, "wb") as f:
        pkl.dump(transitions_to_save, f)

    with open(demo_file, "wb") as f:
        pkl.dump(demo_to_save, f)

    _log_info(
        "checkpoint",
        f"[actor-buffer-save] step={step}, buffer_file={buffer_file}, demo_file={demo_file}, "
        f"buffer_saved={len(transitions_to_save)}, demo_saved={len(demo_to_save)}",
        "green",
    )

    return len(transitions_to_save), len(demo_to_save)


# =============================================================================
# 9.9 actor online intervention abs-pose -> relative-action helpers
# =============================================================================

def _get_actor_env_arm_side(env):
    try:
        return str(getattr(env.unwrapped, "arm_side", "right")).lower()
    except Exception:
        return "right"


def _extract_single_arm_feedback_pose_from_obs(obs, *, arm_side="right"):
    """
    从 actor obs / next_obs 中读取 feedback EE pose。

    支持：
      1) obs["state"] 是 dict:
           right_ee_pose / left_ee_pose / tcp_pose 等
      2) obs["state"] 是 array:
           shape=(1,8) 或 (8,)：前 7 维为 xyz+quat，最后 1 维为 gripper

    返回:
      np.ndarray shape=(7,) 或 shape=(6,)；优先 7 维 xyz+quat
    """
    if obs is None or not isinstance(obs, dict):
        return None
    if "state" not in obs:
        return None

    state = obs["state"]

    if isinstance(state, dict):
        preferred = []
        if arm_side == "left":
            preferred.extend([
                "left_ee_pose",
                "left/tcp_pose",
                "left_tcp_pose",
                "state/left_ee_pose",
                "state/left/tcp_pose",
            ])
        else:
            preferred.extend([
                "right_ee_pose",
                "right/tcp_pose",
                "right_tcp_pose",
                "state/right_ee_pose",
                "state/right/tcp_pose",
            ])

        preferred.extend([
            "ee_pose",
            "tcp_pose",
            "pose_ee",
            "pose_ee_arm_right",
            "pose_ee_arm_left",
        ])

        for key in preferred:
            if key in state:
                try:
                    arr = np.asarray(state[key], dtype=np.float32).reshape(-1)
                    if arr.size >= 7:
                        return arr[:7].copy()
                    if arr.size >= 6:
                        return arr[:6].copy()
                except Exception:
                    pass

        for key, value in state.items():
            k = str(key).lower()
            if ("pose" in k or "tcp" in k or "ee" in k) and "gripper" not in k:
                try:
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    if arr.size >= 7:
                        return arr[:7].copy()
                    if arr.size >= 6:
                        return arr[:6].copy()
                except Exception:
                    pass

        return None

    try:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    # 常见单臂 state:
    # [x, y, z, qx, qy, qz, qw, gripper]
    if arr.size >= 7:
        return arr[:7].copy()
    if arr.size >= 6:
        return arr[:6].copy()
    return None


def _quat_xyzw_normalize(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return None
    return q / n


def _quat_xyzw_conj(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_xyzw_mul(q1, q2):
    """
    xyzw quaternion multiplication: q = q1 * q2
    """
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def _quat_xyzw_to_rotvec(q):
    """
    xyzw quaternion -> rotvec.
    使用最短旋转，返回 shape=(3,)。
    """
    q = _quat_xyzw_normalize(q)
    if q is None:
        return None

    # q 和 -q 表示同一个姿态；让 w >= 0，得到最短旋转。
    if q[3] < 0:
        q = -q

    v = q[:3]
    w = float(q[3])
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        # 小角度近似：rotvec ~= 2 * v
        return (2.0 * v).astype(np.float32)

    angle = 2.0 * np.arctan2(s, w)
    axis = v / s
    return (axis * angle).astype(np.float32)


def _euler_xyz_to_quat_xyzw(euler):
    """
    兜底支持 6D pose: xyz + euler xyz.
    """
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64).reshape(3)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _pose_to_pos_quat_xyzw_for_actor_abs2rel(pose):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)

    if pose.size >= 7:
        pos = pose[:3].astype(np.float32)
        quat = _quat_xyzw_normalize(pose[3:7])
        if quat is None:
            return None, None
        return pos, quat.astype(np.float32)

    if pose.size >= 6:
        pos = pose[:3].astype(np.float32)
        quat = _euler_xyz_to_quat_xyzw(pose[3:6])
        quat = _quat_xyzw_normalize(quat)
        if quat is None:
            return None, None
        return pos, quat.astype(np.float32)

    return None, None


def _feedback_abs2rel_action_from_transition(
    transition,
    *,
    env,
    config,
    fallback_action=None,
):
    """
    用 transition 的 observations / next_observations 中的 feedback EE pose
    转换出 normalized action。

    只用于 actor 在线 VR intervention transition 的 episode-end 转换。

    action[:3] = (next_pos - prev_pos) / POS_SCALE
    action[3:6] = relative_rotvec(prev_quat -> next_quat) / ROT_SCALE
    action[6] = fallback_action[6]，也就是已由 gripper feedback 重写后的三值事件标签
    """
    action_dim = None
    if fallback_action is not None:
        fallback_action = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
        action_dim = int(fallback_action.shape[0])

    if action_dim != 7:
        return None, "fallback_action_not_7d"

    arm_side = _get_actor_env_arm_side(env)

    prev_pose = _extract_single_arm_feedback_pose_from_obs(
        transition.get("observations", None),
        arm_side=arm_side,
    )
    next_pose = _extract_single_arm_feedback_pose_from_obs(
        transition.get("next_observations", None),
        arm_side=arm_side,
    )

    if prev_pose is None:
        return None, "missing_prev_feedback_pose"
    if next_pose is None:
        return None, "missing_next_feedback_pose"

    prev_pos, prev_quat = _pose_to_pos_quat_xyzw_for_actor_abs2rel(prev_pose)
    next_pos, next_quat = _pose_to_pos_quat_xyzw_for_actor_abs2rel(next_pose)

    if prev_pos is None or prev_quat is None:
        return None, "invalid_prev_pose"
    if next_pos is None or next_quat is None:
        return None, "invalid_next_pose"

    pos_scale = float(getattr(config, "POS_SCALE", getattr(env.unwrapped, "pos_scale", 0.02)))
    rot_scale = float(getattr(config, "ROT_SCALE", getattr(env.unwrapped, "rot_scale", 0.04)))

    if pos_scale <= 1e-12:
        return None, "invalid_pos_scale"
    if rot_scale <= 1e-12:
        return None, "invalid_rot_scale"

    try:
        pos_delta = next_pos.astype(np.float32) - prev_pos.astype(np.float32)

        # 对齐 env / wrapper 的执行语义：
        # next_rot = delta_rot * prev_rot
        # delta_rot = next_rot * inv(prev_rot)
        prev_inv = _quat_xyzw_conj(prev_quat)
        delta_quat = _quat_xyzw_mul(next_quat, prev_inv)
        rot_delta = _quat_xyzw_to_rotvec(delta_quat)

        if rot_delta is None:
            return None, "invalid_rot_delta"

        action = np.zeros((7,), dtype=np.float32)
        action[:3] = pos_delta / pos_scale
        action[3:6] = rot_delta / rot_scale

        # 夹爪维度沿用 step 时已经通过 feedback 重写出的三值事件。
        action[6] = float(fallback_action[6])

        action[:6] = np.clip(action[:6], ARM_ACTION_LOW, ARM_ACTION_HIGH)

        return action.astype(np.float32), ""

    except Exception as e:
        return None, f"exception:{repr(e)}"



# =============================================================================
# 10. actor
# =============================================================================

def actor(agent, data_store, intvn_data_store, env, sampling_rng, config, trainer_cfg):
    actor_to_learner_image_keys = resolve_actor_to_learner_image_keys(config)
    validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys)
    _log_info(
        "main",
        f"[actor-obs-prune-config] actor_to_learner_image_keys={actor_to_learner_image_keys}. "
        f"episode 内 callback 只缓存 pending 网络；只在 episode/reset 边界 apply 网络。",
        "green",
    )

    if FLAGS.eval_checkpoint_step:
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()

            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                policy_obs = prune_observation_for_actor_to_learner(
                    obs,
                    actor_to_learner_image_keys,
                    config,
                    strict=FLAGS.actor_to_learner_strict_keys,
                )
                actions = agent.sample_actions(
                    observations=jax.device_put(policy_obs),
                    argmax=False,
                    seed=key,
                )
                actions = np.asarray(jax.device_get(actions), dtype=np.float32)

                obs, reward, done, truncated, info = env.step(actions)

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print_green(f"✅ 第 {episode + 1} 回合成功！耗时: {dt:.2f}s")
                    else:
                        print_yellow(f"❌ 第 {episode + 1} 回合失败。")

                    success_counter += reward
                    print(f"📊 当前成绩: {success_counter}/{episode + 1}")

        print_green(f"🏆 success rate: {success_counter / max(1, FLAGS.eval_n_trajs):.2%}")

        if time_list:
            print_green(f"⏱️ average time: {np.mean(time_list):.2f}s")

        return

    start_step = _infer_actor_start_step_from_numeric_buffers(FLAGS.checkpoint_path)
    _log_info(
        "checkpoint",
        f"[actor-start-step] start_step={start_step}; only numeric periodic buffers are used, e.g. transitions_1000.pkl.",
        "green",
    )

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        trainer_cfg,
        data_stores={"actor_env": data_store, "actor_env_intvn": intvn_data_store},
        wait_for_server=True,
        timeout_ms=FLAGS.request_timeout_ms,
    )

    network_debug = {
        "recv_count": 0,
        "applied_count": 0,
        "duplicate_recv_count": 0,
        "pending_duplicate_recv_count": 0,

        "last_recv_time": None,
        "last_apply_time": None,
        "last_sig": None,
        "last_applied_sig": None,

        "pending_params": None,
        "pending_sig": None,
        "pending_recv_count": 0,
        "pending_recv_time": None,

        "last_update_log_time": None,
    }

    agent_lock = threading.Lock()
    client_rpc_lock = threading.Lock()

    def update_params(params):
        """
        网络 callback 只缓存 learner 发来的最新网络到 pending。
        绝不在 episode 中途替换 actor 参数。
        """
        now = time.time()
        since_prev = None if network_debug["last_recv_time"] is None else now - network_debug["last_recv_time"]
        sig = _tree_debug_signature(params)

        with agent_lock:
            network_debug["recv_count"] += 1
            network_debug["last_recv_time"] = now
            network_debug["last_sig"] = sig

            if network_debug["last_applied_sig"] == sig:
                network_debug["duplicate_recv_count"] += 1

            if network_debug["pending_sig"] == sig:
                network_debug["pending_duplicate_recv_count"] += 1

            network_debug["pending_params"] = params
            network_debug["pending_sig"] = sig
            network_debug["pending_recv_count"] = network_debug["recv_count"]
            network_debug["pending_recv_time"] = now

            recv_count = network_debug["recv_count"]
            applied_count = network_debug["applied_count"]
            pending_dup = network_debug["pending_duplicate_recv_count"]
            applied_dup = network_debug["duplicate_recv_count"]

        if not FLAGS.minimal_logs:
            _log_info(
                "actor_network",
                f"[actor-network-recv-pending] recv_count={recv_count}, applied_count={applied_count}, "
                f"duplicate_vs_applied={applied_dup}, duplicate_vs_pending={pending_dup}, "
                f"since_prev={None if since_prev is None else round(since_prev, 3)}, {_format_signature(sig)}",
                "blue",
            )

    def _apply_pending_network(reason, *, force=False):
        """
        只允许在 episode/reset 边界调用。
        """
        nonlocal agent

        with agent_lock:
            pending_params = network_debug.get("pending_params", None)
            pending_sig = network_debug.get("pending_sig", None)
            pending_recv_count = network_debug.get("pending_recv_count", 0)

            if pending_params is None:
                _log_info(
                    "actor_network",
                    f"[actor-network-apply-skip] reason={reason}, no pending network. "
                    f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}",
                    "yellow" if force else "blue",
                )
                return False

            if (not force) and network_debug["last_applied_sig"] == pending_sig:
                network_debug["pending_params"] = None
                network_debug["pending_sig"] = None
                network_debug["pending_recv_count"] = 0
                network_debug["pending_recv_time"] = None

                _log_info(
                    "actor_network",
                    f"[actor-network-apply-skip] reason={reason}, pending equals current applied. "
                    f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}, "
                    f"pending_recv_count={pending_recv_count}, {_format_signature(pending_sig)}",
                    "blue",
                )
                return False

            params_jnp = jax.tree_util.tree_map(jnp.array, pending_params)
            agent = agent.replace(state=agent.state.replace(params=params_jnp))

            network_debug["applied_count"] += 1
            network_debug["last_apply_time"] = time.time()
            network_debug["last_applied_sig"] = pending_sig

            network_debug["pending_params"] = None
            network_debug["pending_sig"] = None
            network_debug["pending_recv_count"] = 0
            network_debug["pending_recv_time"] = None

            applied_count = network_debug["applied_count"]
            recv_count = network_debug["recv_count"]

        _log_info(
            "actor_network",
            f"[actor-network-apply-boundary] reason={reason}, recv_count={recv_count}, "
            f"applied_count={applied_count}, pending_recv_count={pending_recv_count}, "
            f"{_format_signature(pending_sig)}",
            "green",
        )
        return True

    client.recv_network_callback(update_params)
    _log_info(
        "actor_network",
        "[actor-client-init] recv_network_callback registered; pending-only apply enabled.",
        "blue",
    )

    if FLAGS.ip == "localhost" and getattr(trainer_cfg, "broadcast_port", None):
        _log_info(
            "actor_warning",
            f"[actor-broadcast-hint] 如果通过 SSH 连接远端 learner，请转发 req/rep={trainer_cfg.port_number}, broadcast={trainer_cfg.broadcast_port}",
            "yellow",
        )
        _log_info(
            "actor_warning",
            f"[actor-broadcast-hint] {_suggest_ssh_forward_command(trainer_cfg.port_number, trainer_cfg.broadcast_port)}",
            "yellow",
        )

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

        if force_print or err is not None or dt > 1.0 or not FLAGS.minimal_logs:
            color = "yellow" if err is not None else "blue"
            _log_info(
                "actor_network",
                f"[actor-client-update] reason={reason}, ok={ok}, dt={dt:.3f}s, "
                f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}, "
                f"pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
                f"duplicate_recv_count={network_debug['duplicate_recv_count']}, err={err}",
                color,
            )

        return ok

    def _wait_for_network(reason, *, require_new=NETWORK_WAIT_REQUIRE_NEW, timeout_sec=NETWORK_WAIT_TIMEOUT_SEC):
        """
        边界等待网络广播，然后在同一个边界 apply pending 网络。
        这部分保持原逻辑不变。
        """
        before = int(network_debug["recv_count"])
        t0 = time.time()

        _log_info(
            "actor_network",
            f"[actor-network-wait] reason={reason}, before_recv={before}, "
            f"require_new_broadcast={require_new}, timeout_sec={timeout_sec}",
            "blue",
        )

        while True:
            _client_update(reason, force_print=False)

            after = int(network_debug["recv_count"])
            got_new = after > before

            if not require_new or got_new:
                applied = _apply_pending_network(f"{reason}:after_wait")
                _log_info(
                    "actor_network",
                    f"[actor-network-wait-done] reason={reason}, recv_count={after}, "
                    f"got_new_broadcast={got_new}, applied_now={applied}, "
                    f"applied_count={network_debug['applied_count']}",
                    "green",
                )
                return got_new

            if timeout_sec is not None and (time.time() - t0) >= float(timeout_sec):
                applied = _apply_pending_network(f"{reason}:timeout_apply_existing")
                _log_info(
                    "actor_warning",
                    f"[actor-network-wait-timeout] reason={reason}, 没等到新网络，继续使用当前/已缓存网络。 "
                    f"recv_count={after}, applied_now={applied}, applied_count={network_debug['applied_count']}",
                    "yellow",
                )
                return False

            time.sleep(NETWORK_WAIT_RETRY_SLEEP_SEC)

    transitions = []
    demo_transitions = []

    # 当前 episode 内暂存的 VR intervention transition。
    # 注意：这些 transition 不会逐步 insert，而是在 episode end 时统一用 feedback abs pose 转 action 后再 insert。
    episode_pending_interventions = []

    def _insert_transition_to_online_and_local_buffers(transition, *, also_demo=False):
        """
        插入 learner queue 和本地周期保存 list。
        """
        data_store.insert(transition)
        transitions.append(copy.deepcopy(transition))

        if also_demo:
            intvn_data_store.insert(transition)
            demo_transitions.append(copy.deepcopy(transition))

    def _flush_episode_pending_interventions(reason):
        """
        episode 结束时，把暂存的人类 intervention transitions：
          obs / next_obs feedback pose -> abs2rel action
        然后发送给 learner replay buffer 和 intervention demo buffer。

        必须在 _client_update("episode_end_before_reset") 之前调用，
        这样本轮 episode 的 intervention 数据可以随这次 update 发送给 learner。
        """
        nonlocal episode_pending_interventions

        n_total = len(episode_pending_interventions)
        if n_total == 0:
            return {
                "pending": 0,
                "converted": 0,
                "fallback": 0,
            }

        converted_count = 0
        fallback_count = 0
        fallback_reasons = {}

        for idx, raw_transition in enumerate(episode_pending_interventions):
            transition = copy.deepcopy(raw_transition)

            fallback_action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)

            feedback_action, fail_reason = _feedback_abs2rel_action_from_transition(
                transition,
                env=env,
                config=config,
                fallback_action=fallback_action,
            )

            if feedback_action is not None:
                actions = sanitize_single_arm_action_for_storage(
                    feedback_action,
                    quantize_gripper=True,
                    source=f"actor_episode_feedback_abs2rel:{reason}[{idx}]",
                )
                converted_count += 1
            else:
                actions = sanitize_single_arm_action_for_storage(
                    fallback_action,
                    quantize_gripper=True,
                    source=f"actor_episode_feedback_abs2rel_fallback:{reason}[{idx}]",
                )
                fallback_count += 1
                fallback_reasons[fail_reason] = fallback_reasons.get(fail_reason, 0) + 1

            transition["actions"] = actions

            transition = sync_grasp_penalty_with_stored_action(
                transition,
                penalty_value=FLAGS.grasp_penalty_value,
                source=f"actor_episode_feedback_abs2rel:{reason}[{idx}]",
                preserve_raw_in_infos=False,
            )

            _insert_transition_to_online_and_local_buffers(
                transition,
                also_demo=True,
            )

        _log_info(
            "actor_episode",
            f"[actor-intervention-episode-flush] reason={reason}, "
            f"pending={n_total}, converted_feedback_abs2rel={converted_count}, "
            f"fallback={fallback_count}, fallback_reasons={fallback_reasons}",
            "green" if fallback_count == 0 else "yellow",
        )

        episode_pending_interventions = []

        return {
            "pending": n_total,
            "converted": converted_count,
            "fallback": fallback_count,
        }

    obs, _ = env.reset()

    if WAIT_NETWORK_BEFORE_FIRST_ACTION:
        _wait_for_network("initial_after_reset_before_first_action")
    else:
        _apply_pending_network("initial_after_reset_no_wait")

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_index = 0
    stable_gripper_state = None
    prev_exec_gripper_cmd = 80.0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="actor")

    try:
        for step in pbar:
            timer.tick("total")

            # episode 内不 client.update、不 apply pending、不后台 update。
            with timer.context("sample_actions"):
                if step < config.random_steps:
                    policy_actions = np.asarray(env.action_space.sample(), dtype=np.float32)
                    action_source = "random"
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)

                    policy_obs = prune_observation_for_actor_to_learner(
                        obs,
                        actor_to_learner_image_keys,
                        config,
                        strict=FLAGS.actor_to_learner_strict_keys,
                    )

                    with agent_lock:
                        current_agent = agent

                    policy_actions = current_agent.sample_actions(
                        observations=jax.device_put(policy_obs),
                        seed=key,
                        argmax=False,
                    )
                    policy_actions = np.asarray(jax.device_get(policy_actions), dtype=np.float32)
                    action_source = "policy"

            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(policy_actions)

                info.pop("left", None)
                info.pop("right", None)

                had_intervene_action = "intervene_action" in info

                if had_intervene_action:
                    # 这里只取 intervene_action 的夹爪事件/临时动作作 fallback。
                    # 前 6 维最终不会直接用它，而是在 episode end 用 feedback obs->next_obs 统一重算。
                    raw_intervene_action = np.asarray(
                        info.pop("intervene_action"),
                        dtype=np.float32,
                    ).reshape(-1)

                    stored_actions = raw_intervene_action.copy()

                    _, prev_exec_gripper_cmd = map_single_arm_exec_action_to_hardware(
                        stored_actions,
                        prev_exec_gripper_cmd,
                    )

                    intervention_steps += 1

                    if not already_intervened:
                        intervention_count += 1

                    already_intervened = True

                else:
                    stored_actions = policy_actions.copy()
                    already_intervened = False

                stored_actions = sanitize_single_arm_action_for_storage(
                    stored_actions,
                    quantize_gripper=False,
                    source="actor_online_before_gripper_rewrite",
                )

                # gripper 维度仍然沿用当前在线逻辑：
                # 根据 obs -> next_obs 的真实 gripper feedback 重写成 -1/0/+1。
                # 对 intervention transition，episode end 只重算 action[:6]，保留这里得到的 action[6]。
                actions, stable_gripper_state = rewrite_single_arm_gripper_action_to_three_value(
                    stored_actions,
                    obs,
                    next_obs,
                    stable_gripper_state,
                )

                actions = sanitize_single_arm_action_for_storage(
                    actions,
                    quantize_gripper=True,
                    source="actor_online_after_gripper_rewrite",
                )

                running_return += reward

                obs_to_store = prune_observation_for_actor_to_learner(
                    obs,
                    actor_to_learner_image_keys,
                    config,
                    strict=FLAGS.actor_to_learner_strict_keys,
                )
                next_obs_to_store = prune_observation_for_actor_to_learner(
                    next_obs,
                    actor_to_learner_image_keys,
                    config,
                    strict=FLAGS.actor_to_learner_strict_keys,
                )

                transition = dict(
                    observations=obs_to_store,
                    actions=actions,
                    next_observations=next_obs_to_store,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )

                if had_intervene_action:
                    # 关键改变：
                    # human intervention transition 不立即 insert。
                    # 先暂存，episode end 时用 feedback absolute pose 统一转换 action[:6] 后再 insert。
                    episode_pending_interventions.append(copy.deepcopy(transition))

                    transition_for_debug = sync_grasp_penalty_with_stored_action(
                        copy.deepcopy(transition),
                        penalty_value=FLAGS.grasp_penalty_value,
                        source="actor_online_intervention_pending_debug",
                        preserve_raw_in_infos=False,
                    )

                else:
                    # actor 自主段保持原逻辑：policy_action 直接入 online replay buffer。
                    transition = sync_grasp_penalty_with_stored_action(
                        transition,
                        penalty_value=FLAGS.grasp_penalty_value,
                        source="actor_online_policy_after_gripper_rewrite",
                        preserve_raw_in_infos=False,
                    )

                    _insert_transition_to_online_and_local_buffers(
                        transition,
                        also_demo=False,
                    )

                    transition_for_debug = transition

                obs = next_obs

                if (not FLAGS.minimal_logs) and step % FLAGS.print_period == 0:
                    dbg_exec_actions, _ = map_single_arm_exec_action_to_hardware(
                        policy_actions,
                        prev_exec_gripper_cmd,
                    )
                    since_last_recv = (
                        None
                        if network_debug["last_recv_time"] is None
                        else round(time.time() - network_debug["last_recv_time"], 3)
                    )

                    print_blue(
                        f"[actor-step-debug] step={step}, action_source={action_source}, reward={reward}, "
                        f"done={done}, truncated={truncated}, recv_count={network_debug['recv_count']}, "
                        f"applied_count={network_debug['applied_count']}, pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
                        f"since_last_recv={since_last_recv}, "
                        f"replay_queue={len(data_store)}, intvn_queue={len(intvn_data_store)}, "
                        f"stored_gripper={describe_gripper_three_value(actions[6]) if actions.shape[0] == 7 else 'N/A'}, "
                        f"policy_raw={describe_gripper_three_value(policy_actions[6]) if policy_actions.shape[0] == 7 else 'N/A'}, "
                        f"mapped_hw={dbg_exec_actions[6] if dbg_exec_actions.shape[0] == 7 else 'N/A'}, "
                        f"grasp_penalty={transition_for_debug.get('grasp_penalty', 'N/A')}, "
                        f"had_intervene_action={had_intervene_action}, "
                        f"pending_interventions={len(episode_pending_interventions)}"
                    )

                if done or truncated:
                    # 关键：先把本 episode 的 intervention raw transitions
                    # 用 feedback abs pose 统一转换并 insert 到 data_store / intvn_data_store。
                    # 随后的 _client_update("episode_end_before_reset") 会把这些数据发给 learner。
                    flush_stats = _flush_episode_pending_interventions(
                        reason=f"episode_{episode_index}_end_step_{step}"
                    )

                    if "episode" not in info:
                        info["episode"] = {}

                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps

                    info["episode"]["intervention_pending_flushed"] = int(flush_stats["pending"])
                    info["episode"]["intervention_feedback_abs2rel_converted"] = int(flush_stats["converted"])
                    info["episode"]["intervention_feedback_abs2rel_fallback"] = int(flush_stats["fallback"])

                    ep_debug = _extract_episode_debug_info(info)

                    _log_info(
                        "actor_episode",
                        f"[actor-episode-end] episode={episode_index}, step={step}, return={running_return:.4f}, "
                        f"env_return={ep_debug['return']:.4f}, length={ep_debug['length']}, duration={ep_debug['duration']:.3f}, "
                        f"success={ep_debug['success']}, intervention_count={intervention_count}, "
                        f"intervention_steps={intervention_steps}, "
                        f"feedback_abs2rel_converted={flush_stats['converted']}, fallback={flush_stats['fallback']}, "
                        f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}",
                        "green" if ep_debug["success"] > 0 else "yellow",
                    )

                    try:
                        client.request("send-stats", {"environment": info})
                    except Exception as e:
                        _log_info(
                            "actor_warning",
                            f"[actor-send-stats-warning] {e!r}",
                            "yellow",
                        )

                    pbar.set_description(f"last return: {running_return}")

                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    stable_gripper_state = None
                    prev_exec_gripper_cmd = 80.0
                    episode_index += 1

                    if UPDATE_AT_EPISODE_END_BEFORE_RESET:
                        # 保持原网络通信逻辑：
                        # 这里 client.update 会把刚 flush 的 intervention data 发给 learner。
                        _client_update("episode_end_before_reset", force_print=True)
                        _apply_pending_network("episode_end_before_reset")

                    obs, _ = env.reset()

                    if WAIT_NETWORK_AFTER_EVERY_RESET:
                        if UPDATE_AFTER_RESET_BEFORE_WAIT:
                            _client_update("after_reset_pre_wait", force_print=True)
                            _apply_pending_network("after_reset_pre_wait")

                        _wait_for_network("after_reset_before_next_episode_first_action")
                    else:
                        _apply_pending_network("after_reset_no_wait")

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                _save_periodic_actor_buffers(
                    checkpoint_path=FLAGS.checkpoint_path,
                    step=step,
                    transitions=transitions,
                    demo_transitions=demo_transitions,
                    penalty_value=FLAGS.grasp_penalty_value,
                )
                transitions = []
                demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                try:
                    client.request("send-stats", {"timer": timer.get_average_times()})
                except Exception as e:
                    if not FLAGS.minimal_logs:
                        print_yellow(f"[actor-send-timer-warning] {e!r}")

    finally:
        remaining_online = len(transitions)
        remaining_demo = len(demo_transitions)
        remaining_pending_interventions = len(episode_pending_interventions)

        print_yellow(
            f"[actor-exit] actor loop exited. Unsaved partial buffers are discarded by design: "
            f"online={remaining_online}, demo={remaining_demo}, "
            f"pending_interventions_not_flushed={remaining_pending_interventions}, "
            f"pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
            f"recv_count={network_debug.get('recv_count', 0)}, applied_count={network_debug.get('applied_count', 0)}. "
            f"Only periodic numeric files transitions_1000.pkl, transitions_2000.pkl, ... are persisted."
        )

# =============================================================================
# 11. learner
# =============================================================================

def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None
    start_step = int(os.path.basename(latest_ckpt)[11:]) + 1 if latest_ckpt is not None else 0
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    trainer_cfg = _build_trainer_config()
    _log_trainer_config(trainer_cfg, "learner")
    server = TrainerServer(trainer_cfg, request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)
    print_green("learner TrainerServer started.")

    if PUBLISH_INITIAL_NETWORK_BEFORE_WARMUP:
        _publish_network_to_actor(server, agent.state.params, reason="initial_before_replay_warmup", step=start_step)
        print_green("sent initial network to actor before replay warmup")

    pbar = tqdm.tqdm(total=config.training_starts, initial=len(replay_buffer), desc="Filling up replay buffer", position=0, leave=True)
    last_warmup_publish_t = time.time()
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        now = time.time()
        if FLAGS.warmup_publish_period_s > 0 and now - last_warmup_publish_t >= FLAGS.warmup_publish_period_s:
            _publish_network_to_actor(server, agent.state.params, reason="warmup_republish", step=step)
            last_warmup_publish_t = now
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    if PUBLISH_NETWORK_AFTER_WARMUP:
        _publish_network_to_actor(server, agent.state.params, reason="after_replay_warmup", step=start_step)
        print_green("resent initial network to actor after replay warmup")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
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
        last_critics_info = None
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            with timer.context("train_critics"):
                agent, critics_info = agent.update(batch, networks_to_update=train_critic_networks_to_update)
                last_critics_info = critics_info

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(batch, networks_to_update=train_networks_to_update)

        if step > 0 and step % config.steps_per_update == 0:
            _publish_network_to_actor(server, agent.state.params, reason="periodic_update", step=step)

        if step % config.log_period == 0:
            update_info_loggable = _to_loggable_pytree(update_info)
            critics_info_loggable = _to_loggable_pytree(last_critics_info) if last_critics_info is not None else {}
            if wandb_logger is not None:
                wandb_logger.log(update_info_loggable, step=step)
                if critics_info_loggable:
                    wandb_logger.log(critics_info_loggable, step=step)
                wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            _print_learner_training_debug(step, update_info, last_critics_info, timer)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            _save_checkpoint_host(FLAGS.checkpoint_path, agent.state, step=step, keep=100)


# =============================================================================
# 12. main
# =============================================================================

def _make_agent_and_buffers(config, env, rng, sample_obs=None, sample_action=None):
    if sample_obs is None:
        sample_obs = env.observation_space.sample()
    if sample_action is None:
        sample_action = env.action_space.sample()

    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_dual_arm(
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
    return agent, include_grasp_penalty


def _resolve_env_config_object():
    """
    兼容两种 config.py 写法：

    1) env_config = GalaxeaUSBTrainConfig()
       -> env_config 已经是对象，不能再 env_config()

    2) env_config = GalaxeaUSBTrainConfig
       或 def env_config(): ...
       -> env_config 可调用，需要 env_config()
    """
    return env_config() if callable(env_config) else env_config


def main(_):
    global config
    config = _resolve_env_config_object()

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # learner 端通常 fake_env=True；你的 Galaxea config 在 fake_env=True 时可能返回 None。
    # 因此只有 env 真正存在时才套 RecordEpisodeStatistics。
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    if env is not None:
        env = RecordEpisodeStatistics(env)
    elif FLAGS.actor:
        raise RuntimeError("actor=True 时 env 不能为 None；请检查 config.get_environment(fake_env=False)。")
    else:
        print_yellow("[learner-env] config.get_environment(fake_env=True) returned None; learner 将使用 demo 推断 observation/action spaces。")

    actor_to_learner_image_keys = resolve_actor_to_learner_image_keys(config)
    validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys)

    sample_obs = None
    sample_action = None
    demo_files = None
    demo_observation_space = None
    demo_action_space = None

    if FLAGS.demo_path is not None:
        demo_files = resolve_demo_paths(FLAGS.demo_path)
        demo_observation_space, demo_action_space, sample_obs, sample_action = build_spaces_and_samples_from_demos(
            demo_files,
            config,
            actor_to_learner_image_keys,
        )

    if FLAGS.learner:
        assert FLAGS.demo_path is not None, "learner 必须提供 --demo_path，因为 fake_env=None 时要靠 demo 推断网络和 buffer spaces。"
        assert sample_obs is not None and sample_action is not None
        assert demo_observation_space is not None and demo_action_space is not None

    agent, include_grasp_penalty = _make_agent_and_buffers(
        config,
        env,
        rng,
        sample_obs=sample_obs,
        sample_action=sample_action,
    )

    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None:
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if os.path.exists(FLAGS.checkpoint_path) else None
        if latest is not None:
            input("Checkpoint path already has checkpoint. Press Enter to resume training.")
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state)
            agent = agent.replace(state=ckpt)
            print_green(f"Loaded previous checkpoint: {latest}")

    def _get_spaces_for_buffers():
        """
        actor 端：用真实 env spaces。
        learner 端 fake_env=None：用 demo 推断出的 spaces。
        """
        if env is not None:
            return env.observation_space, env.action_space

        assert demo_observation_space is not None, "env=None 时缺少 demo_observation_space"
        assert demo_action_space is not None, "env=None 时缺少 demo_action_space"
        return demo_observation_space, demo_action_space

    def create_replay_buffer_and_wandb_logger():
        observation_space, action_space = _get_spaces_for_buffers()
        replay_buffer = MemoryEfficientReplayBufferDataStore(
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
        return replay_buffer, wandb_logger

    trainer_cfg = _build_trainer_config()

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        observation_space, action_space = _get_spaces_for_buffers()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space,
            action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        demo_files = resolve_demo_paths(FLAGS.demo_path)
        for path in demo_files:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
            transitions = prune_transition_list_for_actor_to_learner(
                transitions,
                actor_to_learner_image_keys,
                config,
                source=f"demo_load:{path}",
                strict=FLAGS.actor_to_learner_strict_keys,
                print_summary=True,
            )
            transitions = sanitize_transition_list_for_storage(
                transitions,
                source=f"demo_load:{path}",
                print_summary=True,
            )
            transitions = sync_transition_list_grasp_penalty(
                transitions,
                source=f"demo_load:{path}",
                penalty_value=FLAGS.grasp_penalty_value,
                print_summary=True,
            )
            for transition in transitions:
                demo_buffer.insert(transition)

        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            loaded_transitions = _load_numeric_transition_files_from_dir(
                os.path.join(FLAGS.checkpoint_path, "buffer"),
                actor_to_learner_image_keys=actor_to_learner_image_keys,
                config=config,
                source_prefix="buffer_load",
                strict=FLAGS.actor_to_learner_strict_keys,
                penalty_value=FLAGS.grasp_penalty_value,
            )
            for transition in loaded_transitions:
                replay_buffer.insert(transition)
            print_green(f"Loaded previous numeric buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
            loaded_demo_transitions = _load_numeric_transition_files_from_dir(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer"),
                actor_to_learner_image_keys=actor_to_learner_image_keys,
                config=config,
                source_prefix="demo_buffer_load",
                strict=FLAGS.actor_to_learner_strict_keys,
                penalty_value=FLAGS.grasp_penalty_value,
            )
            for transition in loaded_demo_transitions:
                demo_buffer.insert(transition)
            print_green(f"Loaded previous numeric demo buffer data. Demo buffer size: {len(demo_buffer)}")

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        if FLAGS.actor_expect_gpu:
            backend = jax.default_backend()
            if backend != "gpu" and backend != "cuda":
                print_yellow(f"⚠️ actor 当前 JAX backend={backend}，不是 GPU/CUDA。")
            else:
                print_green(f"✅ actor JAX backend={backend}")

        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)
        print_green("starting actor loop")
        actor(agent, data_store, intvn_data_store, env, sampling_rng, config, trainer_cfg)
    else:
        raise NotImplementedError("Must set either --learner=True or --actor=True")


if __name__ == "__main__":
    app.run(main)



# """
# train_rlpd.py

# 本版核心目标：
# 1) 网络更新改成接近官方 HIL-SERL 结构：
#    - actor 注册 client.recv_network_callback(update_params)
#    - 收到网络后只缓存为 pending_params；
#                episode 内不替换 actor 参数；
#                episode 结束 / reset 后 / 下一条 episode 第一帧前再 apply pending 网络。
#    - episode 内不后台 update、不周期性 update、不 loop_start apply
#    - episode 结束后 client.update()
#    - reset 后等待网络更新，再开始下一 episode 第一帧动作

# 2) VR 介入期间尽量不被网络同步打断，保证 env.step 连续记录 transition。

# 3) learner 端增加终端训练指标打印：critic / grasp_critic / actor / temperature / timer。

# 4) 所有可调配置集中在文件前方。
# """

# # =============================================================================
# # 0. 文件前方总配置区
# # =============================================================================

# # ---- 官方式 actor 网络更新配置 ----
# OFFICIAL_EPISODE_NETWORK_UPDATE = True
# # episode 结束后 update 一次，然后 reset；reset 后再等待网络，之后才输出下一步动作。
# WAIT_NETWORK_BEFORE_FIRST_ACTION = True
# WAIT_NETWORK_AFTER_EVERY_RESET = True
# # True: 等到 recv_count 增加，也就是确实收到一版 learner 发布的网络。
# # 如果你想“保证拿到新网络再动”，保持 True。
# NETWORK_WAIT_REQUIRE_NEW = True
# # None 表示无限等待，最符合“必须保证更新到网络再输出动作”。
# # 如果不想因为 learner 没 publish 而永久等待，可以改成 30.0（等待30s）。
# NETWORK_WAIT_TIMEOUT_SEC = None  # 如需避免 learner 暂停导致 actor 永久等待，可改成 30.0
# NETWORK_WAIT_RETRY_SLEEP_SEC = 0.10
# # episode 结束后是否先 update，再 reset；保留 True。
# UPDATE_AT_EPISODE_END_BEFORE_RESET = True
# # reset 后等待网络前，是否先做一次 client.update。
# UPDATE_AFTER_RESET_BEFORE_WAIT = True

# # ---- learner 网络发布配置 ----
# # learner server 启动后先发一次初始网络，让 actor 初始等待能收到。
# PUBLISH_INITIAL_NETWORK_BEFORE_WARMUP = True
# # replay warmup 结束后再发一次。
# PUBLISH_NETWORK_AFTER_WARMUP = True
# # warmup 阶段定期重发当前网络，避免 actor 初始化等待时错过网络。
# DEFAULT_WARMUP_PUBLISH_PERIOD_S = 5

# # ---- learner 终端训练指标打印配置 ----
# PRINT_LEARNER_TRAIN_DEBUG = True
# PRINT_LEARNER_TRAIN_DEBUG_EVERY_LOG_PERIOD = True

# # ---- Actor -> Learner observation 裁剪配置 ----
# # None: 自动使用 config.image_keys。
# # "all": 不裁剪图像。
# # ["head_rgb", "right_wrist_rgb"]: 显式指定。
# ACTOR_TO_LEARNER_IMAGE_KEYS = None
# ACTOR_TO_LEARNER_EXTRA_OBS_KEYS = ["state"]
# ACTOR_TO_LEARNER_STRICT_KEYS = True

# # ---- 单臂 action 存储约定 ----
# ARM_ACTION_LOW = -1.0
# ARM_ACTION_HIGH = 1.0
# DEFAULT_GRASP_PENALTY_VALUE = -0.02

# # ---- actor 动作输出频率打印配置 ----
# # 只增加诊断打印，不改变 actor/env/learner/buffer 逻辑。
# PRINT_ACTOR_ACTION_FREQ = True

# # 每多少 actor step 打印一次。None 表示复用 FLAGS.print_period。
# ACTOR_ACTION_FREQ_PRINT_PERIOD = None

# # minimal_logs=True 时也打印 actor 频率。
# PRINT_ACTOR_ACTION_FREQ_IN_MINIMAL_LOGS = True

# # 频率统计滑动窗口长度。
# ACTOR_ACTION_FREQ_WINDOW = 50


# # ---- GPU actor 预配置默认值；注意这些值会在 import jax 前通过 sys.argv 生效 ----
# DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES = "0"
# DEFAULT_ACTOR_DISABLE_PREALLOCATE = True


# # =============================================================================
# # 1. import 前 actor GPU 环境变量处理
# # =============================================================================

# import os
# import sys
# import glob
# import time
# import copy
# import pickle as pkl
# import threading


# def _raw_flag_value(name: str, default=None):
#     """在 absl flags 解析前，直接从 sys.argv 读取原始 flag 值。"""
#     prefix = f"--{name}="
#     for arg in sys.argv[1:]:
#         if arg == f"--{name}":
#             return True
#         if arg.startswith(prefix):
#             return arg[len(prefix):]
#     return default


# def _raw_flag_bool(name: str, default: bool = False) -> bool:
#     value = _raw_flag_value(name, default)
#     if isinstance(value, bool):
#         return value
#     if value is None:
#         return default
#     return str(value).strip().lower() in ("true", "1", "yes", "y", "t")


# def _is_actor_mode_from_argv() -> bool:
#     return _raw_flag_bool("actor", False)


# if _is_actor_mode_from_argv():
#     force_actor_cpu = _raw_flag_bool("force_actor_cpu", False)
#     actor_cuda_visible_devices = _raw_flag_value(
#         "actor_cuda_visible_devices",
#         DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES,
#     )
#     actor_disable_preallocate = _raw_flag_bool(
#         "actor_disable_preallocate",
#         DEFAULT_ACTOR_DISABLE_PREALLOCATE,
#     )
#     actor_mem_fraction = _raw_flag_value("actor_mem_fraction", None)
#     actor_jax_platforms = _raw_flag_value("actor_jax_platforms", None)

#     if force_actor_cpu:
#         os.environ["JAX_PLATFORMS"] = "cpu"
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#         os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#     else:
#         if actor_cuda_visible_devices not in (None, "", "auto"):
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(actor_cuda_visible_devices)
#         if actor_disable_preallocate:
#             os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#         if actor_mem_fraction not in (None, "", "0", 0):
#             os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(actor_mem_fraction)
#         if actor_jax_platforms not in (None, ""):
#             os.environ["JAX_PLATFORMS"] = str(actor_jax_platforms)


# # =============================================================================
# # 2. imports
# # =============================================================================

# import jax
# import jax.numpy as jnp
# import numpy as np
# import tqdm
# from absl import app, flags
# from flax.training import checkpoints
# from gymnasium import spaces
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
# from natsort import natsorted

# from serl_launcher.agents.continuous.sac import SACAgent
# from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
# from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
# from serl_launcher.utils.timer_utils import Timer
# from serl_launcher.utils.train_utils import concat_batches
# from serl_launcher.utils.launcher import (
#     make_sac_pixel_agent,
#     make_sac_pixel_agent_hybrid_single_arm,
#     make_sac_pixel_agent_hybrid_dual_arm,
#     make_trainer_config,
#     make_wandb_logger,
# )
# from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# from agentlace.trainer import TrainerServer, TrainerClient
# from agentlace.data.data_store import QueuedDataStore

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# from examples.galaxea_task.usb_pick_insertion_single.config import env_config


# # =============================================================================
# # 3. flags
# # =============================================================================

# FLAGS = flags.FLAGS

# flags.DEFINE_string("exp_name", "galaxea_usb_insertion", "Experiment name.")
# flags.DEFINE_integer("seed", 42, "Random seed.")
# flags.DEFINE_boolean("learner", False, "Whether this process is the learner.")
# flags.DEFINE_boolean("actor", False, "Whether this process is the actor.")
# flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo data for learner bootstrap.")
# flags.DEFINE_string("checkpoint_path", "./rlpd_checkpoints", "Path to save checkpoints / buffers.")

# flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate checkpoint.")
# flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
# flags.DEFINE_boolean("save_video", False, "Save evaluation videos.")
# flags.DEFINE_boolean("debug", False, "Debug mode, disables wandb upload.")

# flags.DEFINE_integer("print_period", 50, "How often to print actor debug lines.")
# flags.DEFINE_integer("request_timeout_ms", 15000, "TrainerClient timeout in ms.")

# # 下面三个 flag 保留兼容旧命令；本版 official episode update 模式中 actor 不再按 step/background update。
# flags.DEFINE_integer("client_update_period", 0, "Deprecated in official episode update mode.")
# flags.DEFINE_float("client_update_interval_sec", 0.5, "Deprecated in official episode update mode.")
# flags.DEFINE_boolean("client_update_background", False, "Deprecated in official episode update mode.")

# flags.DEFINE_integer("warmup_publish_period_s", DEFAULT_WARMUP_PUBLISH_PERIOD_S, "Warmup network republish period.")
# flags.DEFINE_integer("trainer_port", 0, "Override TrainerConfig.port_number when > 0.")
# flags.DEFINE_integer("trainer_broadcast_port", 0, "Override TrainerConfig.broadcast_port when > 0.")
# flags.DEFINE_boolean("print_trainer_config", True, "Print trainer config and ports.")
# flags.DEFINE_boolean("minimal_logs", True, "Only keep important logs.")

# flags.DEFINE_boolean("force_actor_cpu", False, "Force actor to run on CPU.")
# flags.DEFINE_string("actor_cuda_visible_devices", DEFAULT_ACTOR_CUDA_VISIBLE_DEVICES, "CUDA_VISIBLE_DEVICES for actor.")
# flags.DEFINE_boolean("actor_disable_preallocate", DEFAULT_ACTOR_DISABLE_PREALLOCATE, "Disable JAX GPU preallocation on actor.")
# flags.DEFINE_float("actor_mem_fraction", 0.0, "Optional XLA_PYTHON_CLIENT_MEM_FRACTION.")
# flags.DEFINE_string("actor_jax_platforms", "", "Optional JAX_PLATFORMS override.")
# flags.DEFINE_boolean("actor_expect_gpu", True, "Warn if actor does not start on GPU.")

# flags.DEFINE_string(
#     "actor_to_learner_image_keys",
#     "",
#     "Comma-separated image keys. Empty uses top config; 'all' disables pruning.",
# )
# flags.DEFINE_boolean(
#     "actor_to_learner_strict_keys",
#     True,
#     "If True, error when selected actor_to_learner_image_keys are missing.",
# )
# flags.DEFINE_float(
#     "grasp_penalty_value",
#     DEFAULT_GRASP_PENALTY_VALUE,
#     "Penalty written into buffers when final action[6] is close/open event.",
# )


# # =============================================================================
# # 4. JAX sharding
# # =============================================================================

# devices = jax.local_devices()
# num_devices = len(devices)
# sharding = jax.sharding.PositionalSharding(devices)


# # =============================================================================
# # 5. logging helpers
# # =============================================================================

# def print_green(x):
#     print("\033[92m {}\033[00m".format(x))


# def print_yellow(x):
#     print("\033[93m {}\033[00m".format(x))


# def print_blue(x):
#     print("\033[94m {}\033[00m".format(x))


# def print_red(x):
#     print("\033[91m {}\033[00m".format(x))


# def _suggest_ssh_forward_command(reqrep_port: int, broadcast_port: int) -> str:
#     return (
#         f"ssh -p 2122 -L {reqrep_port}:localhost:{reqrep_port} "
#         f"-L {broadcast_port}:localhost:{broadcast_port} lixiang@service.qich.top"
#     )


# def _log_enabled(kind: str) -> bool:
#     if not getattr(FLAGS, "minimal_logs", False):
#         return True
#     keep = {
#         "main",
#         "checkpoint",
#         "actor_network",
#         "actor_episode",
#         "actor_warning",
#         "actor_error",
#         "learner_publish",
#         "learner_step",
#         "learner_checkpoint",
#         "learner_env",
#     }

#     # 频率诊断默认在 minimal_logs=True 时也保留。
#     if PRINT_ACTOR_ACTION_FREQ_IN_MINIMAL_LOGS:
#         keep.add("actor_action_freq")

#     return kind in keep


# def _log_info(kind: str, msg: str, color: str = "blue"):
#     if not _log_enabled(kind):
#         return
#     fn = {"blue": print_blue, "green": print_green, "yellow": print_yellow, "red": print_red}.get(color, print)
#     fn(msg)


# def _as_python_scalar(x):
#     arr = np.asarray(x)
#     if arr.size == 0:
#         raise ValueError("empty array cannot be converted to scalar")
#     return arr.reshape(-1)[0].item()


# def _safe_float(x, default=0.0):
#     try:
#         return float(_as_python_scalar(x))
#     except Exception:
#         return default


# def _safe_int(x, default=0):
#     try:
#         return int(_as_python_scalar(x))
#     except Exception:
#         return default


# def _trainer_config_dict(cfg):
#     out = {}
#     for name in dir(cfg):
#         if name.startswith("_"):
#             continue
#         try:
#             value = getattr(cfg, name)
#         except Exception:
#             continue
#         if callable(value):
#             continue
#         if isinstance(value, (int, float, str, bool, type(None), list, tuple, dict)):
#             out[name] = value
#     return out


# def _build_trainer_config():
#     cfg = make_trainer_config()
#     if FLAGS.trainer_port > 0 and hasattr(cfg, "port_number"):
#         cfg.port_number = FLAGS.trainer_port
#     if FLAGS.trainer_broadcast_port > 0 and hasattr(cfg, "broadcast_port"):
#         cfg.broadcast_port = FLAGS.trainer_broadcast_port
#     return cfg


# def _log_trainer_config(cfg, role):
#     if not FLAGS.print_trainer_config or FLAGS.minimal_logs:
#         return
#     _log_info("main", f"[{role}-trainer-config] {_trainer_config_dict(cfg)}", "blue")


# # =============================================================================
# # 6. pytree / network helpers
# # =============================================================================

# def _tree_debug_signature(tree, max_leaves=8, elems_per_leaf=8):
#     leaves, _ = jax.tree_util.tree_flatten(tree)
#     sampled = []
#     total_params = 0
#     leaf_shapes = []
#     for idx, leaf in enumerate(leaves):
#         arr = np.asarray(leaf)
#         total_params += int(arr.size)
#         if idx < max_leaves:
#             leaf_shapes.append(tuple(arr.shape))
#             if arr.size > 0:
#                 sampled.extend(arr.reshape(-1)[:elems_per_leaf].astype(np.float64).tolist())
#     sample_arr = np.asarray(sampled, dtype=np.float64) if sampled else np.zeros((1,), dtype=np.float64)
#     return {
#         "leaf_count": len(leaves),
#         "total_params": total_params,
#         "checksum": float(sample_arr.sum()),
#         "abs_mean": float(np.mean(np.abs(sample_arr))),
#         "sample_std": float(np.std(sample_arr)),
#         "sample_head": [round(float(x), 6) for x in sample_arr[:6]],
#         "leaf_shapes": leaf_shapes[:4],
#     }


# def _format_signature(sig):
#     if sig is None:
#         return "None"
#     return (
#         f"leafs={sig['leaf_count']}, total_params={sig['total_params']}, "
#         f"checksum={sig['checksum']:.6f}, abs_mean={sig['abs_mean']:.6f}, "
#         f"sample_std={sig['sample_std']:.6f}, head={sig['sample_head']}, shapes={sig['leaf_shapes']}"
#     )


# def _block_until_ready_tree(tree):
#     def _block(x):
#         if hasattr(x, "block_until_ready"):
#             x.block_until_ready()
#         return x
#     return jax.tree_util.tree_map(_block, tree)


# def _to_host_pytree(tree):
#     def _convert(x):
#         if isinstance(x, (jax.Array, jnp.ndarray)):
#             return np.asarray(jax.device_get(x))
#         return x
#     return jax.tree_util.tree_map(_convert, tree)


# def _to_loggable_pytree(tree):
#     def _convert(x):
#         if isinstance(x, (jax.Array, jnp.ndarray)):
#             x = np.asarray(jax.device_get(x))
#             if x.shape == ():
#                 return x.item()
#             return x
#         return x
#     return jax.tree_util.tree_map(_convert, tree)


# def _publish_network_to_actor(server, params, *, reason="periodic_update", step=None):
#     t0 = time.time()
#     params = _block_until_ready_tree(params)
#     params = _to_host_pytree(params)
#     sig = _tree_debug_signature(params)
#     server.publish_network(params)
#     dt = time.time() - t0
#     _log_info(
#         "learner_publish",
#         f"[learner-network-publish] reason={reason}, step={step}, cost={dt:.3f}s, {_format_signature(sig)}",
#         "blue",
#     )
#     return sig


# def _save_checkpoint_host(checkpoint_path, state, step, keep=100):
#     t0 = time.time()
#     state = _block_until_ready_tree(state)
#     state = _to_host_pytree(state)
#     sig = _tree_debug_signature(state.params if hasattr(state, "params") else state)
#     checkpoints.save_checkpoint(os.path.abspath(checkpoint_path), state, step=step, keep=keep)
#     dt = time.time() - t0
#     _log_info(
#         "learner_checkpoint",
#         f"[learner-checkpoint-save] step={step}, cost={dt:.3f}s, path={os.path.abspath(checkpoint_path)}, {_format_signature(sig)}",
#         "blue",
#     )


# # =============================================================================
# # 7. gripper / action storage helpers
# # =============================================================================

# def extract_gripper_feedback_from_obs(obs):
#     if obs is None or "state" not in obs:
#         return None
#     state = obs["state"]
#     if isinstance(state, dict):
#         for key in ["right_gripper", "left_gripper", "gripper", "state/right_gripper", "state/left_gripper"]:
#             if key in state:
#                 arr = np.asarray(state[key]).reshape(-1)
#                 if arr.size > 0:
#                     return float(arr[-1])
#         for key, val in state.items():
#             if "gripper" in str(key).lower():
#                 arr = np.asarray(val).reshape(-1)
#                 if arr.size > 0:
#                     return float(arr[-1])
#         return None
#     arr = np.asarray(state)
#     while arr.ndim > 1:
#         arr = arr[-1]
#     arr = arr.reshape(-1)
#     if arr.size == 0:
#         return None
#     return float(arr[-1])


# def infer_stable_gripper_state_from_feedback(gripper_feedback, prev_state, close_max=30.0, open_min=70.0):
#     if gripper_feedback is None:
#         return prev_state
#     x = float(gripper_feedback)
#     if x <= close_max:
#         return -1
#     if x >= open_min:
#         return +1
#     return prev_state


# def rewrite_single_arm_gripper_action_to_three_value(action, obs, next_obs, prev_stable_state):
#     action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
#     if action.shape[0] != 7:
#         return action, prev_stable_state

#     prev_feedback = extract_gripper_feedback_from_obs(obs)
#     next_feedback = extract_gripper_feedback_from_obs(next_obs)
#     prev_state = infer_stable_gripper_state_from_feedback(prev_feedback, prev_stable_state)
#     next_state = infer_stable_gripper_state_from_feedback(next_feedback, prev_state)

#     gripper_event = 0.0
#     if prev_state is not None and next_state is not None:
#         if prev_state == +1 and next_state == -1:
#             gripper_event = -1.0
#         elif prev_state == -1 and next_state == +1:
#             gripper_event = +1.0
#         else:
#             gripper_event = 0.0
#     action[6] = np.float32(gripper_event)
#     return action.astype(np.float32), next_state


# def map_single_arm_exec_action_to_hardware(action, prev_hw_cmd, close_cmd=10.0, open_cmd=80.0, deadband=0.5):
#     action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
#     if action.shape[0] != 7:
#         return action, prev_hw_cmd
#     grip = float(action[6])
#     if 0.0 <= grip <= 100.0 and abs(grip) > 5.0:
#         hw_cmd = grip
#     else:
#         if grip >= deadband:
#             hw_cmd = open_cmd
#         elif grip <= -deadband:
#             hw_cmd = close_cmd
#         else:
#             hw_cmd = prev_hw_cmd
#     exec_action = action.copy()
#     exec_action[6] = np.float32(hw_cmd)
#     return exec_action, float(hw_cmd)


# def describe_gripper_three_value(x):
#     x = float(x)
#     if x <= -0.5:
#         return "close(-1)"
#     if x >= 0.5:
#         return "open(+1)"
#     return "hold(0)"


# def sanitize_single_arm_action_for_storage(action, *, quantize_gripper=True, source="unknown", return_changed=False):
#     a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
#     if a.shape[0] != 7:
#         if return_changed:
#             return a.astype(np.float32), False, False
#         return a.astype(np.float32)
#     before = a.copy()
#     a[:6] = np.clip(a[:6], ARM_ACTION_LOW, ARM_ACTION_HIGH)
#     if quantize_gripper:
#         g = float(a[6])
#         if g <= -0.5:
#             a[6] = -1.0
#         elif g >= 0.5:
#             a[6] = 1.0
#         else:
#             a[6] = 0.0
#     else:
#         a[6] = np.clip(a[6], ARM_ACTION_LOW, ARM_ACTION_HIGH)
#     was_out_of_range = bool(np.any(np.abs(before[:6]) > 1.0001))
#     changed = bool(not np.allclose(before, a, atol=1e-6, rtol=1e-6))
#     if return_changed:
#         return a.astype(np.float32), changed, was_out_of_range
#     return a.astype(np.float32)


# def sanitize_transition_action_for_storage(transition, *, source="transition", return_changed=False):
#     trans = copy.deepcopy(transition)
#     if "actions" not in trans:
#         if return_changed:
#             return trans, False, False
#         return trans
#     clean_action, changed, was_out = sanitize_single_arm_action_for_storage(
#         trans["actions"], quantize_gripper=True, source=source, return_changed=True
#     )
#     trans["actions"] = clean_action
#     if return_changed:
#         return trans, changed, was_out
#     return trans


# def sanitize_transition_list_for_storage(transitions, *, source="transitions", print_summary=True):
#     clean = []
#     changed_count = 0
#     out_count = 0
#     for idx, transition in enumerate(transitions):
#         trans, changed, was_out = sanitize_transition_action_for_storage(
#             transition, source=f"{source}[{idx}]", return_changed=True
#         )
#         clean.append(trans)
#         changed_count += int(changed)
#         out_count += int(was_out)
#     if print_summary:
#         _log_info(
#             "main",
#             f"[action-sanitize] source={source}, n={len(transitions)}, changed={changed_count}, arm_out_of_range={out_count}",
#             "yellow" if out_count > 0 else "green",
#         )
#     return clean


# def recompute_grasp_penalty_from_stored_action(action, penalty_value=None):
#     if penalty_value is None:
#         penalty_value = float(getattr(FLAGS, "grasp_penalty_value", DEFAULT_GRASP_PENALTY_VALUE))
#     a = np.asarray(action, dtype=np.float32).reshape(-1)
#     if a.shape[0] != 7:
#         return 0.0
#     g = float(a[6])
#     if g <= -0.5 or g >= 0.5:
#         return float(penalty_value)
#     return 0.0


# def sync_grasp_penalty_with_stored_action(transition, *, penalty_value=None, source="unknown", preserve_raw_in_infos=True):
#     if not isinstance(transition, dict) or "actions" not in transition:
#         return transition
#     expected = recompute_grasp_penalty_from_stored_action(transition["actions"], penalty_value=penalty_value)
#     old_top = transition.get("grasp_penalty", None)
#     transition["grasp_penalty"] = float(expected)
#     infos = transition.get("infos", transition.get("info", None))
#     if isinstance(infos, dict):
#         if preserve_raw_in_infos:
#             if "grasp_penalty" in infos and "env_grasp_penalty_raw" not in infos:
#                 infos["env_grasp_penalty_raw"] = _safe_float(infos["grasp_penalty"], 0.0)
#             if old_top is not None and "top_level_grasp_penalty_raw" not in infos:
#                 infos["top_level_grasp_penalty_raw"] = _safe_float(old_top, 0.0)
#         infos["grasp_penalty"] = float(expected)
#         infos["grasp_penalty_source"] = f"recomputed_from_final_action:{source}"
#         if "infos" in transition:
#             transition["infos"] = infos
#         elif "info" in transition:
#             transition["info"] = infos
#     return transition


# def sync_transition_list_grasp_penalty(transitions, *, source="transitions", penalty_value=None, print_summary=True):
#     if penalty_value is None:
#         penalty_value = float(getattr(FLAGS, "grasp_penalty_value", DEFAULT_GRASP_PENALTY_VALUE))
#     synced = []
#     mismatch_before = 0
#     nonzero_after = 0
#     event_count = 0
#     hold_penalty_after = 0
#     for idx, transition in enumerate(transitions):
#         trans = copy.deepcopy(transition)
#         old = None
#         if isinstance(trans, dict):
#             if "grasp_penalty" in trans:
#                 old = _safe_float(trans.get("grasp_penalty"), 0.0)
#             else:
#                 infos = trans.get("infos", trans.get("info", None))
#                 if isinstance(infos, dict) and "grasp_penalty" in infos:
#                     old = _safe_float(infos.get("grasp_penalty"), 0.0)
#         expected = recompute_grasp_penalty_from_stored_action(trans.get("actions", np.zeros(0)), penalty_value)
#         if old is not None and abs(float(old) - float(expected)) > 1e-6:
#             mismatch_before += 1
#         trans = sync_grasp_penalty_with_stored_action(
#             trans, penalty_value=penalty_value, source=f"{source}[{idx}]"
#         )
#         a = np.asarray(trans.get("actions", []), dtype=np.float32).reshape(-1)
#         if a.shape[0] == 7:
#             g = float(a[6])
#             if abs(g) > 0.5:
#                 event_count += 1
#             if abs(float(trans.get("grasp_penalty", 0.0))) > 1e-8:
#                 nonzero_after += 1
#                 if abs(g) <= 0.5:
#                     hold_penalty_after += 1
#         synced.append(trans)
#     if print_summary:
#         color = "green" if hold_penalty_after == 0 and nonzero_after == event_count else "yellow"
#         _log_info(
#             "main",
#             f"[grasp-penalty-sync] source={source}, n={len(transitions)}, penalty_value={penalty_value}, "
#             f"mismatch_before={mismatch_before}, gripper_event_count={event_count}, "
#             f"nonzero_after={nonzero_after}, hold_penalty_after={hold_penalty_after}",
#             color,
#         )
#     return synced


# # =============================================================================
# # 8. observation pruning / demo space helpers
# # =============================================================================

# DEFAULT_KNOWN_IMAGE_KEYS = {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"}


# def _parse_comma_keys(value):
#     if value is None:
#         return None
#     if isinstance(value, (list, tuple)):
#         return [str(x).strip() for x in value if str(x).strip()]
#     value = str(value).strip()
#     if value == "":
#         return None
#     if value.lower() in ("none", "config", "default"):
#         return None
#     if value.lower() in ("all", "*"):
#         return "all"
#     return [x.strip() for x in value.split(",") if x.strip()]


# def resolve_actor_to_learner_image_keys(config):
#     cli_value = _parse_comma_keys(getattr(FLAGS, "actor_to_learner_image_keys", ""))
#     if cli_value == "all":
#         return "all"
#     if cli_value is not None:
#         return cli_value
#     top_value = _parse_comma_keys(ACTOR_TO_LEARNER_IMAGE_KEYS)
#     if top_value == "all":
#         return "all"
#     if top_value is not None:
#         return top_value
#     return list(getattr(config, "image_keys", []))


# def validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys):
#     if actor_to_learner_image_keys == "all":
#         return
#     policy_keys = list(getattr(config, "image_keys", []))
#     selected = list(actor_to_learner_image_keys or [])
#     missing_policy_keys = [k for k in policy_keys if k not in selected]
#     if missing_policy_keys:
#         raise ValueError(
#             "actor_to_learner_image_keys 必须包含所有 config.image_keys。"
#             f" missing={missing_policy_keys}, selected={selected}, policy_keys={policy_keys}"
#         )
#     env_keys = list(getattr(config, "ENV_IMAGE_KEYS", []))
#     if env_keys:
#         missing_env_keys = [k for k in selected if k not in env_keys]
#         if missing_env_keys:
#             raise ValueError(
#                 "actor_to_learner_image_keys 中有 key 不在 ENV_IMAGE_KEYS 里。"
#                 f" missing={missing_env_keys}, selected={selected}, ENV_IMAGE_KEYS={env_keys}"
#             )


# def get_known_image_keys(config, actor_to_learner_image_keys=None):
#     keys = set(DEFAULT_KNOWN_IMAGE_KEYS)
#     for attr in ["ENV_IMAGE_KEYS", "DISPLAY_IMAGE_KEYS", "image_keys", "classifier_keys"]:
#         value = getattr(config, attr, None)
#         if isinstance(value, (list, tuple)):
#             keys.update([str(x) for x in value])
#     if isinstance(actor_to_learner_image_keys, (list, tuple)):
#         keys.update([str(x) for x in actor_to_learner_image_keys])
#     return keys


# def _get_obs_value_by_key(obs, key):
#     if key in obs:
#         return obs[key]
#     images = obs.get("images", None)
#     if isinstance(images, dict) and key in images:
#         return images[key]
#     raise KeyError(f"obs 中找不到图像 key={key}, obs.keys={list(obs.keys())}")


# def prune_observation_for_actor_to_learner(obs, actor_to_learner_image_keys, config, *, strict=True):
#     if obs is None or not isinstance(obs, dict):
#         return obs
#     if actor_to_learner_image_keys == "all":
#         return obs
#     image_keys = list(actor_to_learner_image_keys or [])
#     known_image_keys = get_known_image_keys(config, image_keys)
#     extra_keys = set(ACTOR_TO_LEARNER_EXTRA_OBS_KEYS or [])
#     pruned = {}
#     for key in image_keys:
#         try:
#             pruned[key] = _get_obs_value_by_key(obs, key)
#         except KeyError:
#             if strict:
#                 raise
#             print_yellow(f"⚠️ actor_to_learner_image_key={key} 不在 obs 中，已跳过。obs.keys={list(obs.keys())}")
#     for key, value in obs.items():
#         if key == "images" or key in pruned:
#             continue
#         is_known_image = key in known_image_keys
#         looks_like_image = (
#             key.endswith("_rgb")
#             or key.endswith("_depth")
#             or key.endswith("_image")
#             or (hasattr(value, "shape") and len(np.asarray(value).shape) >= 3 and str(key).lower() != "state")
#         )
#         if key in extra_keys or (not is_known_image and not looks_like_image):
#             pruned[key] = value
#     return pruned


# def prune_transition_for_actor_to_learner(transition, actor_to_learner_image_keys, config, *, strict=True):
#     trans = copy.deepcopy(transition)
#     if "observations" in trans:
#         trans["observations"] = prune_observation_for_actor_to_learner(
#             trans["observations"], actor_to_learner_image_keys, config, strict=strict
#         )
#     if "next_observations" in trans:
#         trans["next_observations"] = prune_observation_for_actor_to_learner(
#             trans["next_observations"], actor_to_learner_image_keys, config, strict=strict
#         )
#     return trans


# def prune_transition_list_for_actor_to_learner(transitions, actor_to_learner_image_keys, config, *, source="transitions", strict=True, print_summary=True):
#     if actor_to_learner_image_keys == "all":
#         if print_summary:
#             _log_info("main", f"[obs-prune] source={source}, mode=all, n={len(transitions)}, 不裁剪图像", "yellow")
#         return transitions
#     clean = [
#         prune_transition_for_actor_to_learner(t, actor_to_learner_image_keys, config, strict=strict)
#         for t in transitions
#     ]
#     if print_summary:
#         keys = []
#         if len(clean) > 0 and isinstance(clean[0].get("observations", None), dict):
#             keys = list(clean[0]["observations"].keys())
#         _log_info(
#             "main",
#             f"[obs-prune] source={source}, actor_to_learner_image_keys={actor_to_learner_image_keys}, n={len(clean)}, stored_obs_keys={keys}",
#             "green",
#         )
#     return clean


# def print_observation_keys_summary(transition_or_obs, *, name="obs"):
#     obs = transition_or_obs.get("observations", transition_or_obs) if isinstance(transition_or_obs, dict) else transition_or_obs
#     if isinstance(obs, dict):
#         _log_info("main", f"[obs-summary] {name}: keys={list(obs.keys())}", "green")
#         for k, v in obs.items():
#             try:
#                 arr = np.asarray(v)
#                 _log_info("main", f"[obs-summary]   {k}: shape={arr.shape}, dtype={arr.dtype}", "green")
#             except Exception:
#                 _log_info("main", f"[obs-summary]   {k}: type={type(v)}", "green")


# def infer_space_from_value(x):
#     if isinstance(x, dict):
#         return spaces.Dict({k: infer_space_from_value(v) for k, v in x.items()})
#     arr = np.asarray(x)
#     if arr.dtype == np.uint8:
#         return spaces.Box(low=0, high=255, shape=arr.shape, dtype=np.uint8)
#     if np.issubdtype(arr.dtype, np.bool_):
#         return spaces.Box(low=0, high=1, shape=arr.shape, dtype=np.bool_)
#     if np.issubdtype(arr.dtype, np.integer):
#         return spaces.Box(low=np.iinfo(arr.dtype).min, high=np.iinfo(arr.dtype).max, shape=arr.shape, dtype=arr.dtype)
#     return spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)


# def resolve_demo_paths(paths):
#     resolved = []
#     for p in paths:
#         if os.path.isdir(p):
#             resolved.extend(glob.glob(os.path.join(p, "*.pkl")))
#         else:
#             resolved.extend(glob.glob(p))
#     resolved = [p for p in resolved if p.endswith(".pkl")]
#     assert len(resolved) > 0, "❌ 没有找到任何 demo .pkl 文件。"
#     return resolved


# def get_first_valid_transition(paths):
#     for path in paths:
#         with open(path, "rb") as f:
#             transitions = pkl.load(f)
#         for transition in transitions:
#             if "actions" in transition and "observations" in transition:
#                 return transition
#     raise ValueError("❌ 无法从 demo_path 中找到有效 transition。")


# def build_spaces_and_samples_from_demos(paths, config, actor_to_learner_image_keys):
#     sample_transition = sanitize_transition_action_for_storage(get_first_valid_transition(paths), source="sample_demo_infer")
#     sample_transition = prune_transition_for_actor_to_learner(
#         sample_transition, actor_to_learner_image_keys, config, strict=FLAGS.actor_to_learner_strict_keys
#     )
#     observation_space = infer_space_from_value(sample_transition["observations"])
#     sample_action = np.asarray(sample_transition["actions"], dtype=np.float32).reshape(-1)
#     if sample_action.shape[0] == 7:
#         action_space = spaces.Box(low=-1.0, high=1.0, shape=sample_action.shape, dtype=np.float32)
#     else:
#         action_space = infer_space_from_value(sample_action)
#     sample_obs = sample_transition["observations"]
#     print_observation_keys_summary(sample_obs, name="sample_obs_for_agent_and_buffer")
#     return observation_space, action_space, sample_obs, sample_action


# def _extract_episode_debug_info(info):
#     episode = info.get("episode", {}) if isinstance(info, dict) else {}
#     ep_return = _safe_float(episode.get("r", episode.get("return", 0.0)))
#     raw_success = info.get("success", info.get("is_success", info.get("succeed", 0.0)))
#     success = max(_safe_float(raw_success, 0.0), float(ep_return > 0.0))
#     return {
#         "return": ep_return,
#         "length": _safe_int(episode.get("l", episode.get("length", 0)), 0),
#         "duration": _safe_float(episode.get("t", episode.get("time", 0.0))),
#         "success": success,
#         "intervention_count": _safe_int(episode.get("intervention_count", 0), 0),
#         "intervention_steps": _safe_int(episode.get("intervention_steps", 0), 0),
#     }


# # =============================================================================
# # 9. learner terminal metric helpers
# # =============================================================================

# def _fmt_metric(info, key, default=None):
#     if info is None or key not in info:
#         return default
#     try:
#         x = np.asarray(jax.device_get(info[key]))
#         if x.size == 0:
#             return default
#         return float(x.reshape(-1)[0])
#     except Exception:
#         return default


# def _format_metric_value(x, digits=6):
#     if x is None:
#         return "N/A"
#     try:
#         return f"{float(x):.{digits}f}"
#     except Exception:
#         return "N/A"


# def _print_learner_training_debug(step, update_info, critics_info, timer):
#     if not PRINT_LEARNER_TRAIN_DEBUG:
#         return

#     critic_loss = _fmt_metric(update_info, "critic/critic_loss") or _fmt_metric(critics_info, "critic/critic_loss")
#     predicted_qs = _fmt_metric(update_info, "critic/predicted_qs") or _fmt_metric(critics_info, "critic/predicted_qs")
#     target_qs = _fmt_metric(update_info, "critic/target_qs") or _fmt_metric(critics_info, "critic/target_qs")
#     rewards = _fmt_metric(update_info, "critic/rewards") or _fmt_metric(critics_info, "critic/rewards")

#     grasp_loss = _fmt_metric(update_info, "grasp_critic/grasp_critic_loss") or _fmt_metric(critics_info, "grasp_critic/grasp_critic_loss")
#     grasp_pred_q = _fmt_metric(update_info, "grasp_critic/predicted_grasp_qs") or _fmt_metric(critics_info, "grasp_critic/predicted_grasp_qs")
#     grasp_target_q = _fmt_metric(update_info, "grasp_critic/target_grasp_qs") or _fmt_metric(critics_info, "grasp_critic/target_grasp_qs")
#     grasp_rewards = _fmt_metric(update_info, "grasp_critic/grasp_rewards") or _fmt_metric(critics_info, "grasp_critic/grasp_rewards")

#     actor_loss = _fmt_metric(update_info, "actor/actor_loss")
#     entropy = _fmt_metric(update_info, "actor/entropy")
#     temperature = _fmt_metric(update_info, "actor/temperature")
#     temp_loss = _fmt_metric(update_info, "temperature/temperature_loss")

#     times = timer.get_average_times()
#     _log_info(
#         "learner_step",
#         "[learner-train-debug] "
#         f"step={step} | "
#         f"critic_loss={_format_metric_value(critic_loss)} pred_q={_format_metric_value(predicted_qs)} "
#         f"target_q={_format_metric_value(target_qs)} reward_mean={_format_metric_value(rewards)} | "
#         f"grasp_loss={_format_metric_value(grasp_loss)} grasp_pred_q={_format_metric_value(grasp_pred_q)} "
#         f"grasp_target_q={_format_metric_value(grasp_target_q)} grasp_reward_mean={_format_metric_value(grasp_rewards)} | "
#         f"actor_loss={_format_metric_value(actor_loss)} entropy={_format_metric_value(entropy)} "
#         f"temperature={_format_metric_value(temperature)} temp_loss={_format_metric_value(temp_loss)} | "
#         f"timer_train={_format_metric_value(times.get('train'), 4)} "
#         f"timer_train_critics={_format_metric_value(times.get('train_critics'), 4)} "
#         f"timer_sample_replay={_format_metric_value(times.get('sample_replay_buffer'), 4)}",
#         "blue",
#     )



# # =============================================================================
# # 9.5 actor buffer save/load helpers
# # =============================================================================

# BUFFER_FILE_PREFIX = "transitions_"
# BUFFER_FILE_SUFFIX = ".pkl"


# def _extract_numeric_step_from_transition_file(path):
#     """
#     只接受旧版周期 buffer 文件名：
#         transitions_1000.pkl
#         transitions_2000.pkl
#         transitions_3000.pkl

#     明确忽略：
#         transitions_197_final.pkl
#         transitions_final.pkl
#         任何非纯数字 step 文件
#     """
#     name = os.path.basename(path)
#     if not (name.startswith(BUFFER_FILE_PREFIX) and name.endswith(BUFFER_FILE_SUFFIX)):
#         return None

#     stem = name[len(BUFFER_FILE_PREFIX):-len(BUFFER_FILE_SUFFIX)]
#     if not stem.isdigit():
#         return None

#     return int(stem)


# def _list_numeric_transition_files(buffer_dir):
#     """
#     返回 [(step, path), ...]，只包含纯数字 step 的周期保存文件。
#     """
#     if not buffer_dir or not os.path.exists(buffer_dir):
#         return []

#     out = []
#     for path in glob.glob(os.path.join(buffer_dir, f"{BUFFER_FILE_PREFIX}*{BUFFER_FILE_SUFFIX}")):
#         step = _extract_numeric_step_from_transition_file(path)
#         if step is not None:
#             out.append((step, path))

#     out.sort(key=lambda x: x[0])
#     return out


# def _infer_actor_start_step_from_numeric_buffers(checkpoint_path):
#     """
#     actor 恢复 step 只看纯数字周期 buffer：
#         buffer/transitions_1000.pkl
#         buffer/transitions_2000.pkl

#     不再读取 *_final.pkl，也不会因为 transitions_197_final.pkl 报错。
#     """
#     if not checkpoint_path:
#         return 0

#     buffer_dir = os.path.join(checkpoint_path, "buffer")
#     numeric_files = _list_numeric_transition_files(buffer_dir)
#     if not numeric_files:
#         return 0

#     return numeric_files[-1][0] + 1


# def _load_numeric_transition_files_from_dir(
#     buffer_dir,
#     *,
#     actor_to_learner_image_keys,
#     config,
#     source_prefix,
#     strict,
#     penalty_value,
# ):
#     """
#     learner 恢复历史 online/demo buffer 时，只读取纯数字周期文件。
#     这会读取：
#         transitions_1000.pkl
#         transitions_2000.pkl
#         transitions_3000.pkl

#     会忽略：
#         transitions_197_final.pkl
#         transitions_foo.pkl
#     """
#     loaded = []
#     numeric_files = _list_numeric_transition_files(buffer_dir)

#     if numeric_files:
#         _log_info(
#             "main",
#             f"[buffer-load] source={source_prefix}, numeric_files={[os.path.basename(p) for _, p in numeric_files]}",
#             "green",
#         )
#     else:
#         _log_info(
#             "main",
#             f"[buffer-load] source={source_prefix}, no numeric periodic buffer files found in {buffer_dir}",
#             "yellow",
#         )

#     for step, file in numeric_files:
#         with open(file, "rb") as f:
#             transitions = pkl.load(f)

#         transitions = prune_transition_list_for_actor_to_learner(
#             transitions,
#             actor_to_learner_image_keys,
#             config,
#             source=f"{source_prefix}:{file}",
#             strict=strict,
#             print_summary=False,
#         )
#         transitions = sanitize_transition_list_for_storage(
#             transitions,
#             source=f"{source_prefix}:{file}",
#             print_summary=False,
#         )
#         transitions = sync_transition_list_grasp_penalty(
#             transitions,
#             source=f"{source_prefix}:{file}",
#             penalty_value=penalty_value,
#             print_summary=False,
#         )
#         loaded.extend(transitions)

#     return loaded


# def _save_periodic_actor_buffers(
#     *,
#     checkpoint_path,
#     step,
#     transitions,
#     demo_transitions,
#     penalty_value,
# ):
#     """
#     旧版逻辑：只在 step 命中 config.buffer_period 时保存：
#         buffer/transitions_{step}.pkl
#         demo_buffer/transitions_{step}.pkl

#     保存后由调用方清空内存 list。
#     不保存 *_final.pkl。
#     """
#     buffer_path = os.path.join(checkpoint_path, "buffer")
#     demo_buffer_path = os.path.join(checkpoint_path, "demo_buffer")
#     os.makedirs(buffer_path, exist_ok=True)
#     os.makedirs(demo_buffer_path, exist_ok=True)

#     transitions_to_save = sync_transition_list_grasp_penalty(
#         sanitize_transition_list_for_storage(
#             transitions,
#             source=f"actor_buffer_save_step_{step}",
#             print_summary=True,
#         ),
#         source=f"actor_buffer_save_step_{step}",
#         penalty_value=penalty_value,
#         print_summary=True,
#     )

#     demo_to_save = sync_transition_list_grasp_penalty(
#         sanitize_transition_list_for_storage(
#             demo_transitions,
#             source=f"actor_demo_buffer_save_step_{step}",
#             print_summary=True,
#         ),
#         source=f"actor_demo_buffer_save_step_{step}",
#         penalty_value=penalty_value,
#         print_summary=True,
#     )

#     buffer_file = os.path.join(buffer_path, f"{BUFFER_FILE_PREFIX}{step}{BUFFER_FILE_SUFFIX}")
#     demo_file = os.path.join(demo_buffer_path, f"{BUFFER_FILE_PREFIX}{step}{BUFFER_FILE_SUFFIX}")

#     with open(buffer_file, "wb") as f:
#         pkl.dump(transitions_to_save, f)

#     with open(demo_file, "wb") as f:
#         pkl.dump(demo_to_save, f)

#     _log_info(
#         "checkpoint",
#         f"[actor-buffer-save] step={step}, buffer_file={buffer_file}, demo_file={demo_file}, "
#         f"buffer_saved={len(transitions_to_save)}, demo_saved={len(demo_to_save)}",
#         "green",
#     )

#     return len(transitions_to_save), len(demo_to_save)


# # =============================================================================
# # 9.9 actor online intervention abs-pose -> relative-action helpers
# # =============================================================================

# def _get_actor_env_arm_side(env):
#     try:
#         return str(getattr(env.unwrapped, "arm_side", "right")).lower()
#     except Exception:
#         return "right"


# def _extract_single_arm_feedback_pose_from_obs(obs, *, arm_side="right"):
#     """
#     从 actor obs / next_obs 中读取 feedback EE pose。

#     支持：
#       1) obs["state"] 是 dict:
#            right_ee_pose / left_ee_pose / tcp_pose 等
#       2) obs["state"] 是 array:
#            shape=(1,8) 或 (8,)：前 7 维为 xyz+quat，最后 1 维为 gripper

#     返回:
#       np.ndarray shape=(7,) 或 shape=(6,)；优先 7 维 xyz+quat
#     """
#     if obs is None or not isinstance(obs, dict):
#         return None
#     if "state" not in obs:
#         return None

#     state = obs["state"]

#     if isinstance(state, dict):
#         preferred = []
#         if arm_side == "left":
#             preferred.extend([
#                 "left_ee_pose",
#                 "left/tcp_pose",
#                 "left_tcp_pose",
#                 "state/left_ee_pose",
#                 "state/left/tcp_pose",
#             ])
#         else:
#             preferred.extend([
#                 "right_ee_pose",
#                 "right/tcp_pose",
#                 "right_tcp_pose",
#                 "state/right_ee_pose",
#                 "state/right/tcp_pose",
#             ])

#         preferred.extend([
#             "ee_pose",
#             "tcp_pose",
#             "pose_ee",
#             "pose_ee_arm_right",
#             "pose_ee_arm_left",
#         ])

#         for key in preferred:
#             if key in state:
#                 try:
#                     arr = np.asarray(state[key], dtype=np.float32).reshape(-1)
#                     if arr.size >= 7:
#                         return arr[:7].copy()
#                     if arr.size >= 6:
#                         return arr[:6].copy()
#                 except Exception:
#                     pass

#         for key, value in state.items():
#             k = str(key).lower()
#             if ("pose" in k or "tcp" in k or "ee" in k) and "gripper" not in k:
#                 try:
#                     arr = np.asarray(value, dtype=np.float32).reshape(-1)
#                     if arr.size >= 7:
#                         return arr[:7].copy()
#                     if arr.size >= 6:
#                         return arr[:6].copy()
#                 except Exception:
#                     pass

#         return None

#     try:
#         arr = np.asarray(state, dtype=np.float32).reshape(-1)
#     except Exception:
#         return None

#     # 常见单臂 state:
#     # [x, y, z, qx, qy, qz, qw, gripper]
#     if arr.size >= 7:
#         return arr[:7].copy()
#     if arr.size >= 6:
#         return arr[:6].copy()
#     return None


# def _quat_xyzw_normalize(q):
#     q = np.asarray(q, dtype=np.float64).reshape(4)
#     n = float(np.linalg.norm(q))
#     if n < 1e-12:
#         return None
#     return q / n


# def _quat_xyzw_conj(q):
#     q = np.asarray(q, dtype=np.float64).reshape(4)
#     return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


# def _quat_xyzw_mul(q1, q2):
#     """
#     xyzw quaternion multiplication: q = q1 * q2
#     """
#     x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
#     x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

#     return np.array([
#         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
#         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#     ], dtype=np.float64)


# def _quat_xyzw_to_rotvec(q):
#     """
#     xyzw quaternion -> rotvec.
#     使用最短旋转，返回 shape=(3,)。
#     """
#     q = _quat_xyzw_normalize(q)
#     if q is None:
#         return None

#     # q 和 -q 表示同一个姿态；让 w >= 0，得到最短旋转。
#     if q[3] < 0:
#         q = -q

#     v = q[:3]
#     w = float(q[3])
#     s = float(np.linalg.norm(v))

#     if s < 1e-12:
#         # 小角度近似：rotvec ~= 2 * v
#         return (2.0 * v).astype(np.float32)

#     angle = 2.0 * np.arctan2(s, w)
#     axis = v / s
#     return (axis * angle).astype(np.float32)


# def _euler_xyz_to_quat_xyzw(euler):
#     """
#     兜底支持 6D pose: xyz + euler xyz.
#     """
#     roll, pitch, yaw = np.asarray(euler, dtype=np.float64).reshape(3)

#     cr = np.cos(roll * 0.5)
#     sr = np.sin(roll * 0.5)
#     cp = np.cos(pitch * 0.5)
#     sp = np.sin(pitch * 0.5)
#     cy = np.cos(yaw * 0.5)
#     sy = np.sin(yaw * 0.5)

#     qx = sr * cp * cy - cr * sp * sy
#     qy = cr * sp * cy + sr * cp * sy
#     qz = cr * cp * sy - sr * sp * cy
#     qw = cr * cp * cy + sr * sp * sy

#     return np.array([qx, qy, qz, qw], dtype=np.float64)


# def _pose_to_pos_quat_xyzw_for_actor_abs2rel(pose):
#     pose = np.asarray(pose, dtype=np.float32).reshape(-1)

#     if pose.size >= 7:
#         pos = pose[:3].astype(np.float32)
#         quat = _quat_xyzw_normalize(pose[3:7])
#         if quat is None:
#             return None, None
#         return pos, quat.astype(np.float32)

#     if pose.size >= 6:
#         pos = pose[:3].astype(np.float32)
#         quat = _euler_xyz_to_quat_xyzw(pose[3:6])
#         quat = _quat_xyzw_normalize(quat)
#         if quat is None:
#             return None, None
#         return pos, quat.astype(np.float32)

#     return None, None


# def _feedback_abs2rel_action_from_transition(
#     transition,
#     *,
#     env,
#     config,
#     fallback_action=None,
# ):
#     """
#     用 transition 的 observations / next_observations 中的 feedback EE pose
#     转换出 normalized action。

#     只用于 actor 在线 VR intervention transition 的 episode-end 转换。

#     action[:3] = (next_pos - prev_pos) / POS_SCALE
#     action[3:6] = relative_rotvec(prev_quat -> next_quat) / ROT_SCALE
#     action[6] = fallback_action[6]，也就是已由 gripper feedback 重写后的三值事件标签
#     """
#     action_dim = None
#     if fallback_action is not None:
#         fallback_action = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
#         action_dim = int(fallback_action.shape[0])

#     if action_dim != 7:
#         return None, "fallback_action_not_7d"

#     arm_side = _get_actor_env_arm_side(env)

#     prev_pose = _extract_single_arm_feedback_pose_from_obs(
#         transition.get("observations", None),
#         arm_side=arm_side,
#     )
#     next_pose = _extract_single_arm_feedback_pose_from_obs(
#         transition.get("next_observations", None),
#         arm_side=arm_side,
#     )

#     if prev_pose is None:
#         return None, "missing_prev_feedback_pose"
#     if next_pose is None:
#         return None, "missing_next_feedback_pose"

#     prev_pos, prev_quat = _pose_to_pos_quat_xyzw_for_actor_abs2rel(prev_pose)
#     next_pos, next_quat = _pose_to_pos_quat_xyzw_for_actor_abs2rel(next_pose)

#     if prev_pos is None or prev_quat is None:
#         return None, "invalid_prev_pose"
#     if next_pos is None or next_quat is None:
#         return None, "invalid_next_pose"

#     pos_scale = float(getattr(config, "POS_SCALE", getattr(env.unwrapped, "pos_scale", 0.02)))
#     rot_scale = float(getattr(config, "ROT_SCALE", getattr(env.unwrapped, "rot_scale", 0.04)))

#     if pos_scale <= 1e-12:
#         return None, "invalid_pos_scale"
#     if rot_scale <= 1e-12:
#         return None, "invalid_rot_scale"

#     try:
#         pos_delta = next_pos.astype(np.float32) - prev_pos.astype(np.float32)

#         # 对齐 env / wrapper 的执行语义：
#         # next_rot = delta_rot * prev_rot
#         # delta_rot = next_rot * inv(prev_rot)
#         prev_inv = _quat_xyzw_conj(prev_quat)
#         delta_quat = _quat_xyzw_mul(next_quat, prev_inv)
#         rot_delta = _quat_xyzw_to_rotvec(delta_quat)

#         if rot_delta is None:
#             return None, "invalid_rot_delta"

#         action = np.zeros((7,), dtype=np.float32)
#         action[:3] = pos_delta / pos_scale
#         action[3:6] = rot_delta / rot_scale

#         # 夹爪维度沿用 step 时已经通过 feedback 重写出的三值事件。
#         action[6] = float(fallback_action[6])

#         action[:6] = np.clip(action[:6], ARM_ACTION_LOW, ARM_ACTION_HIGH)

#         return action.astype(np.float32), ""

#     except Exception as e:
#         return None, f"exception:{repr(e)}"



# # =============================================================================
# # 10. actor
# # =============================================================================

# def actor(agent, data_store, intvn_data_store, env, sampling_rng, config, trainer_cfg):
#     actor_to_learner_image_keys = resolve_actor_to_learner_image_keys(config)
#     validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys)
#     _log_info(
#         "main",
#         f"[actor-obs-prune-config] actor_to_learner_image_keys={actor_to_learner_image_keys}. "
#         f"episode 内 callback 只缓存 pending 网络；只在 episode/reset 边界 apply 网络。",
#         "green",
#     )

#     if FLAGS.eval_checkpoint_step:
#         ckpt = checkpoints.restore_checkpoint(
#             os.path.abspath(FLAGS.checkpoint_path),
#             agent.state,
#             step=FLAGS.eval_checkpoint_step,
#         )
#         agent = agent.replace(state=ckpt)
#         success_counter = 0
#         time_list = []

#         for episode in range(FLAGS.eval_n_trajs):
#             obs, _ = env.reset()
#             done = False
#             start_time = time.time()

#             while not done:
#                 sampling_rng, key = jax.random.split(sampling_rng)
#                 policy_obs = prune_observation_for_actor_to_learner(
#                     obs,
#                     actor_to_learner_image_keys,
#                     config,
#                     strict=FLAGS.actor_to_learner_strict_keys,
#                 )
#                 actions = agent.sample_actions(
#                     observations=jax.device_put(policy_obs),
#                     argmax=False,
#                     seed=key,
#                 )
#                 actions = np.asarray(jax.device_get(actions), dtype=np.float32)

#                 obs, reward, done, truncated, info = env.step(actions)

#                 if done:
#                     if reward:
#                         dt = time.time() - start_time
#                         time_list.append(dt)
#                         print_green(f"✅ 第 {episode + 1} 回合成功！耗时: {dt:.2f}s")
#                     else:
#                         print_yellow(f"❌ 第 {episode + 1} 回合失败。")

#                     success_counter += reward
#                     print(f"📊 当前成绩: {success_counter}/{episode + 1}")

#         print_green(f"🏆 success rate: {success_counter / max(1, FLAGS.eval_n_trajs):.2%}")

#         if time_list:
#             print_green(f"⏱️ average time: {np.mean(time_list):.2f}s")

#         return

#     start_step = _infer_actor_start_step_from_numeric_buffers(FLAGS.checkpoint_path)
#     _log_info(
#         "checkpoint",
#         f"[actor-start-step] start_step={start_step}; only numeric periodic buffers are used, e.g. transitions_1000.pkl.",
#         "green",
#     )

#     client = TrainerClient(
#         "actor_env",
#         FLAGS.ip,
#         trainer_cfg,
#         data_stores={"actor_env": data_store, "actor_env_intvn": intvn_data_store},
#         wait_for_server=True,
#         timeout_ms=FLAGS.request_timeout_ms,
#     )

#     network_debug = {
#         "recv_count": 0,
#         "applied_count": 0,
#         "duplicate_recv_count": 0,
#         "pending_duplicate_recv_count": 0,

#         "last_recv_time": None,
#         "last_apply_time": None,
#         "last_sig": None,
#         "last_applied_sig": None,

#         "pending_params": None,
#         "pending_sig": None,
#         "pending_recv_count": 0,
#         "pending_recv_time": None,

#         "last_update_log_time": None,
#     }

#     agent_lock = threading.Lock()
#     client_rpc_lock = threading.Lock()

#     def update_params(params):
#         """
#         网络 callback 只缓存 learner 发来的最新网络到 pending。
#         绝不在 episode 中途替换 actor 参数。
#         """
#         now = time.time()
#         since_prev = None if network_debug["last_recv_time"] is None else now - network_debug["last_recv_time"]
#         sig = _tree_debug_signature(params)

#         with agent_lock:
#             network_debug["recv_count"] += 1
#             network_debug["last_recv_time"] = now
#             network_debug["last_sig"] = sig

#             if network_debug["last_applied_sig"] == sig:
#                 network_debug["duplicate_recv_count"] += 1

#             if network_debug["pending_sig"] == sig:
#                 network_debug["pending_duplicate_recv_count"] += 1

#             network_debug["pending_params"] = params
#             network_debug["pending_sig"] = sig
#             network_debug["pending_recv_count"] = network_debug["recv_count"]
#             network_debug["pending_recv_time"] = now

#             recv_count = network_debug["recv_count"]
#             applied_count = network_debug["applied_count"]
#             pending_dup = network_debug["pending_duplicate_recv_count"]
#             applied_dup = network_debug["duplicate_recv_count"]

#         if not FLAGS.minimal_logs:
#             _log_info(
#                 "actor_network",
#                 f"[actor-network-recv-pending] recv_count={recv_count}, applied_count={applied_count}, "
#                 f"duplicate_vs_applied={applied_dup}, duplicate_vs_pending={pending_dup}, "
#                 f"since_prev={None if since_prev is None else round(since_prev, 3)}, {_format_signature(sig)}",
#                 "blue",
#             )

#     def _apply_pending_network(reason, *, force=False):
#         """
#         只允许在 episode/reset 边界调用。
#         """
#         nonlocal agent

#         with agent_lock:
#             pending_params = network_debug.get("pending_params", None)
#             pending_sig = network_debug.get("pending_sig", None)
#             pending_recv_count = network_debug.get("pending_recv_count", 0)

#             if pending_params is None:
#                 _log_info(
#                     "actor_network",
#                     f"[actor-network-apply-skip] reason={reason}, no pending network. "
#                     f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}",
#                     "yellow" if force else "blue",
#                 )
#                 return False

#             if (not force) and network_debug["last_applied_sig"] == pending_sig:
#                 network_debug["pending_params"] = None
#                 network_debug["pending_sig"] = None
#                 network_debug["pending_recv_count"] = 0
#                 network_debug["pending_recv_time"] = None

#                 _log_info(
#                     "actor_network",
#                     f"[actor-network-apply-skip] reason={reason}, pending equals current applied. "
#                     f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}, "
#                     f"pending_recv_count={pending_recv_count}, {_format_signature(pending_sig)}",
#                     "blue",
#                 )
#                 return False

#             params_jnp = jax.tree_util.tree_map(jnp.array, pending_params)
#             agent = agent.replace(state=agent.state.replace(params=params_jnp))

#             network_debug["applied_count"] += 1
#             network_debug["last_apply_time"] = time.time()
#             network_debug["last_applied_sig"] = pending_sig

#             network_debug["pending_params"] = None
#             network_debug["pending_sig"] = None
#             network_debug["pending_recv_count"] = 0
#             network_debug["pending_recv_time"] = None

#             applied_count = network_debug["applied_count"]
#             recv_count = network_debug["recv_count"]

#         _log_info(
#             "actor_network",
#             f"[actor-network-apply-boundary] reason={reason}, recv_count={recv_count}, "
#             f"applied_count={applied_count}, pending_recv_count={pending_recv_count}, "
#             f"{_format_signature(pending_sig)}",
#             "green",
#         )
#         return True

#     client.recv_network_callback(update_params)
#     _log_info(
#         "actor_network",
#         "[actor-client-init] recv_network_callback registered; pending-only apply enabled.",
#         "blue",
#     )

#     if FLAGS.ip == "localhost" and getattr(trainer_cfg, "broadcast_port", None):
#         _log_info(
#             "actor_warning",
#             f"[actor-broadcast-hint] 如果通过 SSH 连接远端 learner，请转发 req/rep={trainer_cfg.port_number}, broadcast={trainer_cfg.broadcast_port}",
#             "yellow",
#         )
#         _log_info(
#             "actor_warning",
#             f"[actor-broadcast-hint] {_suggest_ssh_forward_command(trainer_cfg.port_number, trainer_cfg.broadcast_port)}",
#             "yellow",
#         )

#     def _client_update(reason, *, force_print=False):
#         t0 = time.time()
#         ok = False
#         err = None

#         try:
#             with client_rpc_lock:
#                 ok = bool(client.update())
#         except Exception as e:
#             err = repr(e)

#         dt = time.time() - t0

#         if force_print or err is not None or dt > 1.0 or not FLAGS.minimal_logs:
#             color = "yellow" if err is not None else "blue"
#             _log_info(
#                 "actor_network",
#                 f"[actor-client-update] reason={reason}, ok={ok}, dt={dt:.3f}s, "
#                 f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}, "
#                 f"pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
#                 f"duplicate_recv_count={network_debug['duplicate_recv_count']}, err={err}",
#                 color,
#             )

#         return ok

#     def _wait_for_network(reason, *, require_new=NETWORK_WAIT_REQUIRE_NEW, timeout_sec=NETWORK_WAIT_TIMEOUT_SEC):
#         """
#         边界等待网络广播，然后在同一个边界 apply pending 网络。
#         这部分保持原逻辑不变。
#         """
#         before = int(network_debug["recv_count"])
#         t0 = time.time()

#         _log_info(
#             "actor_network",
#             f"[actor-network-wait] reason={reason}, before_recv={before}, "
#             f"require_new_broadcast={require_new}, timeout_sec={timeout_sec}",
#             "blue",
#         )

#         while True:
#             _client_update(reason, force_print=False)

#             after = int(network_debug["recv_count"])
#             got_new = after > before

#             if not require_new or got_new:
#                 applied = _apply_pending_network(f"{reason}:after_wait")
#                 _log_info(
#                     "actor_network",
#                     f"[actor-network-wait-done] reason={reason}, recv_count={after}, "
#                     f"got_new_broadcast={got_new}, applied_now={applied}, "
#                     f"applied_count={network_debug['applied_count']}",
#                     "green",
#                 )
#                 return got_new

#             if timeout_sec is not None and (time.time() - t0) >= float(timeout_sec):
#                 applied = _apply_pending_network(f"{reason}:timeout_apply_existing")
#                 _log_info(
#                     "actor_warning",
#                     f"[actor-network-wait-timeout] reason={reason}, 没等到新网络，继续使用当前/已缓存网络。 "
#                     f"recv_count={after}, applied_now={applied}, applied_count={network_debug['applied_count']}",
#                     "yellow",
#                 )
#                 return False

#             time.sleep(NETWORK_WAIT_RETRY_SLEEP_SEC)

#     transitions = []
#     demo_transitions = []

#     # 当前 episode 内暂存的 VR intervention transition。
#     # 注意：这些 transition 不会逐步 insert，而是在 episode end 时统一用 feedback abs pose 转 action 后再 insert。
#     episode_pending_interventions = []

#     def _insert_transition_to_online_and_local_buffers(transition, *, also_demo=False):
#         """
#         插入 learner queue 和本地周期保存 list。
#         """
#         data_store.insert(transition)
#         transitions.append(copy.deepcopy(transition))

#         if also_demo:
#             intvn_data_store.insert(transition)
#             demo_transitions.append(copy.deepcopy(transition))

#     def _flush_episode_pending_interventions(reason):
#         """
#         episode 结束时，把暂存的人类 intervention transitions：
#           obs / next_obs feedback pose -> abs2rel action
#         然后发送给 learner replay buffer 和 intervention demo buffer。

#         必须在 _client_update("episode_end_before_reset") 之前调用，
#         这样本轮 episode 的 intervention 数据可以随这次 update 发送给 learner。
#         """
#         nonlocal episode_pending_interventions

#         n_total = len(episode_pending_interventions)
#         if n_total == 0:
#             return {
#                 "pending": 0,
#                 "converted": 0,
#                 "fallback": 0,
#             }

#         converted_count = 0
#         fallback_count = 0
#         fallback_reasons = {}

#         for idx, raw_transition in enumerate(episode_pending_interventions):
#             transition = copy.deepcopy(raw_transition)

#             fallback_action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)

#             feedback_action, fail_reason = _feedback_abs2rel_action_from_transition(
#                 transition,
#                 env=env,
#                 config=config,
#                 fallback_action=fallback_action,
#             )

#             if feedback_action is not None:
#                 actions = sanitize_single_arm_action_for_storage(
#                     feedback_action,
#                     quantize_gripper=True,
#                     source=f"actor_episode_feedback_abs2rel:{reason}[{idx}]",
#                 )
#                 converted_count += 1
#             else:
#                 actions = sanitize_single_arm_action_for_storage(
#                     fallback_action,
#                     quantize_gripper=True,
#                     source=f"actor_episode_feedback_abs2rel_fallback:{reason}[{idx}]",
#                 )
#                 fallback_count += 1
#                 fallback_reasons[fail_reason] = fallback_reasons.get(fail_reason, 0) + 1

#             transition["actions"] = actions

#             transition = sync_grasp_penalty_with_stored_action(
#                 transition,
#                 penalty_value=FLAGS.grasp_penalty_value,
#                 source=f"actor_episode_feedback_abs2rel:{reason}[{idx}]",
#                 preserve_raw_in_infos=False,
#             )

#             _insert_transition_to_online_and_local_buffers(
#                 transition,
#                 also_demo=True,
#             )

#         _log_info(
#             "actor_episode",
#             f"[actor-intervention-episode-flush] reason={reason}, "
#             f"pending={n_total}, converted_feedback_abs2rel={converted_count}, "
#             f"fallback={fallback_count}, fallback_reasons={fallback_reasons}",
#             "green" if fallback_count == 0 else "yellow",
#         )

#         episode_pending_interventions = []

#         return {
#             "pending": n_total,
#             "converted": converted_count,
#             "fallback": fallback_count,
#         }

#     obs, _ = env.reset()

#     if WAIT_NETWORK_BEFORE_FIRST_ACTION:
#         _wait_for_network("initial_after_reset_before_first_action")
#     else:
#         _apply_pending_network("initial_after_reset_no_wait")

#     timer = Timer()
#     running_return = 0.0
#     already_intervened = False
#     intervention_count = 0
#     intervention_steps = 0
#     episode_index = 0
#     stable_gripper_state = None
#     prev_exec_gripper_cmd = 80.0

#     # -------------------------------------------------------------------------
#     # actor action output frequency diagnostics
#     # -------------------------------------------------------------------------
#     actor_freq_last_loop_t = None
#     actor_freq_last_action_t = None

#     actor_freq_loop_dt_hist = []          # for-loop start to next for-loop start
#     actor_freq_action_dt_hist = []        # action ready time to next action ready time
#     actor_freq_sample_dt_hist = []        # policy/random action generation cost
#     actor_freq_env_dt_hist = []           # env.step cost
#     actor_freq_total_dt_hist = []         # full actor loop cost

#     actor_freq_window = int(ACTOR_ACTION_FREQ_WINDOW)

#     pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="actor")

#     try:
#         for step in pbar:
#             actor_loop_start_t = time.time()

#             if actor_freq_last_loop_t is None:
#                 actor_loop_dt = 0.0
#             else:
#                 actor_loop_dt = actor_loop_start_t - actor_freq_last_loop_t

#             actor_freq_last_loop_t = actor_loop_start_t

#             if actor_loop_dt > 0:
#                 actor_freq_loop_dt_hist.append(float(actor_loop_dt))
#                 actor_freq_loop_dt_hist = actor_freq_loop_dt_hist[-actor_freq_window:]

#             timer.tick("total")

#             # episode 内不 client.update、不 apply pending、不后台 update。
#             actor_sample_start_t = time.time()

#             with timer.context("sample_actions"):
#                 if step < config.random_steps:
#                     policy_actions = np.asarray(env.action_space.sample(), dtype=np.float32)
#                     action_source = "random"
#                 else:
#                     sampling_rng, key = jax.random.split(sampling_rng)

#                     policy_obs = prune_observation_for_actor_to_learner(
#                         obs,
#                         actor_to_learner_image_keys,
#                         config,
#                         strict=FLAGS.actor_to_learner_strict_keys,
#                     )

#                     with agent_lock:
#                         current_agent = agent

#                     policy_actions = current_agent.sample_actions(
#                         observations=jax.device_put(policy_obs),
#                         seed=key,
#                         argmax=False,
#                     )
#                     policy_actions = np.asarray(jax.device_get(policy_actions), dtype=np.float32)
#                     action_source = "policy"

#             actor_action_output_t = time.time()
#             actor_sample_dt = actor_action_output_t - actor_sample_start_t
#             actor_freq_sample_dt_hist.append(float(actor_sample_dt))
#             actor_freq_sample_dt_hist = actor_freq_sample_dt_hist[-actor_freq_window:]

#             if actor_freq_last_action_t is None:
#                 actor_action_dt = 0.0
#             else:
#                 actor_action_dt = actor_action_output_t - actor_freq_last_action_t

#             actor_freq_last_action_t = actor_action_output_t

#             if actor_action_dt > 0:
#                 actor_freq_action_dt_hist.append(float(actor_action_dt))
#                 actor_freq_action_dt_hist = actor_freq_action_dt_hist[-actor_freq_window:]

#             actor_env_step_start_t = time.time()

#             with timer.context("step_env"):
#                 next_obs, reward, done, truncated, info = env.step(policy_actions)

#                 actor_env_step_end_t = time.time()
#                 actor_env_step_dt = actor_env_step_end_t - actor_env_step_start_t
#                 actor_freq_env_dt_hist.append(float(actor_env_step_dt))
#                 actor_freq_env_dt_hist = actor_freq_env_dt_hist[-actor_freq_window:]

#                 info.pop("left", None)
#                 info.pop("right", None)

#                 had_intervene_action = "intervene_action" in info

#                 if had_intervene_action:
#                     # 这里只取 intervene_action 的夹爪事件/临时动作作 fallback。
#                     # 前 6 维最终不会直接用它，而是在 episode end 用 feedback obs->next_obs 统一重算。
#                     raw_intervene_action = np.asarray(
#                         info.pop("intervene_action"),
#                         dtype=np.float32,
#                     ).reshape(-1)

#                     stored_actions = raw_intervene_action.copy()

#                     _, prev_exec_gripper_cmd = map_single_arm_exec_action_to_hardware(
#                         stored_actions,
#                         prev_exec_gripper_cmd,
#                     )

#                     intervention_steps += 1

#                     if not already_intervened:
#                         intervention_count += 1

#                     already_intervened = True

#                 else:
#                     stored_actions = policy_actions.copy()
#                     already_intervened = False

#                 stored_actions = sanitize_single_arm_action_for_storage(
#                     stored_actions,
#                     quantize_gripper=False,
#                     source="actor_online_before_gripper_rewrite",
#                 )

#                 # gripper 维度仍然沿用当前在线逻辑：
#                 # 根据 obs -> next_obs 的真实 gripper feedback 重写成 -1/0/+1。
#                 # 对 intervention transition，episode end 只重算 action[:6]，保留这里得到的 action[6]。
#                 actions, stable_gripper_state = rewrite_single_arm_gripper_action_to_three_value(
#                     stored_actions,
#                     obs,
#                     next_obs,
#                     stable_gripper_state,
#                 )

#                 actions = sanitize_single_arm_action_for_storage(
#                     actions,
#                     quantize_gripper=True,
#                     source="actor_online_after_gripper_rewrite",
#                 )

#                 running_return += reward

#                 obs_to_store = prune_observation_for_actor_to_learner(
#                     obs,
#                     actor_to_learner_image_keys,
#                     config,
#                     strict=FLAGS.actor_to_learner_strict_keys,
#                 )
#                 next_obs_to_store = prune_observation_for_actor_to_learner(
#                     next_obs,
#                     actor_to_learner_image_keys,
#                     config,
#                     strict=FLAGS.actor_to_learner_strict_keys,
#                 )

#                 transition = dict(
#                     observations=obs_to_store,
#                     actions=actions,
#                     next_observations=next_obs_to_store,
#                     rewards=reward,
#                     masks=1.0 - done,
#                     dones=done,
#                 )

#                 if had_intervene_action:
#                     # 关键改变：
#                     # human intervention transition 不立即 insert。
#                     # 先暂存，episode end 时用 feedback absolute pose 统一转换 action[:6] 后再 insert。
#                     episode_pending_interventions.append(copy.deepcopy(transition))

#                     transition_for_debug = sync_grasp_penalty_with_stored_action(
#                         copy.deepcopy(transition),
#                         penalty_value=FLAGS.grasp_penalty_value,
#                         source="actor_online_intervention_pending_debug",
#                         preserve_raw_in_infos=False,
#                     )

#                 else:
#                     # actor 自主段保持原逻辑：policy_action 直接入 online replay buffer。
#                     transition = sync_grasp_penalty_with_stored_action(
#                         transition,
#                         penalty_value=FLAGS.grasp_penalty_value,
#                         source="actor_online_policy_after_gripper_rewrite",
#                         preserve_raw_in_infos=False,
#                     )

#                     _insert_transition_to_online_and_local_buffers(
#                         transition,
#                         also_demo=False,
#                     )

#                     transition_for_debug = transition

#                 obs = next_obs

#                 # --------------------------------------------------------------
#                 # actor action output frequency diagnostics
#                 # --------------------------------------------------------------
#                 actor_total_dt = time.time() - actor_loop_start_t
#                 actor_freq_total_dt_hist.append(float(actor_total_dt))
#                 actor_freq_total_dt_hist = actor_freq_total_dt_hist[-actor_freq_window:]

#                 def _actor_freq_mean_dt(xs):
#                     if len(xs) == 0:
#                         return 0.0
#                     return float(np.mean(xs))

#                 def _actor_freq_hz_from_dt(dt):
#                     if dt <= 1e-9:
#                         return 0.0
#                     return float(1.0 / dt)

#                 actor_freq_period = (
#                     int(FLAGS.print_period)
#                     if ACTOR_ACTION_FREQ_PRINT_PERIOD is None
#                     else int(ACTOR_ACTION_FREQ_PRINT_PERIOD)
#                 )
#                 actor_freq_period = max(1, actor_freq_period)

#                 if PRINT_ACTOR_ACTION_FREQ and step % actor_freq_period == 0:
#                     mean_loop_dt = _actor_freq_mean_dt(actor_freq_loop_dt_hist)
#                     mean_action_dt = _actor_freq_mean_dt(actor_freq_action_dt_hist)
#                     mean_sample_dt = _actor_freq_mean_dt(actor_freq_sample_dt_hist)
#                     mean_env_dt = _actor_freq_mean_dt(actor_freq_env_dt_hist)
#                     mean_total_dt = _actor_freq_mean_dt(actor_freq_total_dt_hist)

#                     step_timing = info.get("step_timing", {}) if isinstance(info, dict) else {}
#                     env_effective_hz = _safe_float(step_timing.get("effective_hz", 0.0), 0.0)
#                     env_total_step_dt = _safe_float(step_timing.get("total_step_dt_sec", 0.0), 0.0)
#                     env_action_settle = _safe_float(step_timing.get("action_settle_sec", 0.0), 0.0)
#                     env_publish_dt = _safe_float(step_timing.get("publish_dt_sec", 0.0), 0.0)
#                     env_obs_dt = _safe_float(step_timing.get("obs_dt_sec", 0.0), 0.0)

#                     pa = np.asarray(policy_actions, dtype=np.float32).reshape(-1)
#                     if pa.shape[0] >= 7:
#                         policy_absmax_arm = float(np.max(np.abs(pa[:6])))
#                         policy_grip = float(pa[6])
#                         policy_grip_desc = describe_gripper_three_value(policy_grip)
#                     else:
#                         policy_absmax_arm = 0.0
#                         policy_grip_desc = "N/A"

#                     stored_absmax_arm = float(np.max(np.abs(actions[:6]))) if actions.shape[0] >= 7 else 0.0
#                     stored_grip_desc = describe_gripper_three_value(actions[6]) if actions.shape[0] >= 7 else "N/A"

#                     _log_info(
#                         "actor_action_freq",
#                         "[actor-action-frequency] "
#                         f"step={step}, episode={episode_index}, action_source={action_source}, "
#                         f"had_intervene_action={had_intervene_action}, "
#                         f"pending_interventions={len(episode_pending_interventions)}, "
#                         f"loop_hz={_actor_freq_hz_from_dt(mean_loop_dt):.2f}, loop_dt={mean_loop_dt:.4f}s, "
#                         f"action_output_hz={_actor_freq_hz_from_dt(mean_action_dt):.2f}, "
#                         f"action_output_dt={mean_action_dt:.4f}s, "
#                         f"sample_dt={mean_sample_dt:.4f}s, "
#                         f"env_step_hz={_actor_freq_hz_from_dt(mean_env_dt):.2f}, env_step_dt={mean_env_dt:.4f}s, "
#                         f"total_hz={_actor_freq_hz_from_dt(mean_total_dt):.2f}, total_dt={mean_total_dt:.4f}s, "
#                         f"env_effective_hz={env_effective_hz:.2f}, "
#                         f"env_total_step_dt={env_total_step_dt:.4f}s, "
#                         f"settle={env_action_settle:.4f}s, publish={env_publish_dt:.4f}s, obs={env_obs_dt:.4f}s, "
#                         f"policy_absmax_arm={policy_absmax_arm:.3f}, policy_gripper={policy_grip_desc}, "
#                         f"stored_absmax_arm={stored_absmax_arm:.3f}, stored_gripper={stored_grip_desc}, "
#                         f"reward={reward}, done={done}, truncated={truncated}",
#                         "yellow" if policy_absmax_arm >= 0.98 or stored_absmax_arm >= 0.98 else "blue",
#                     )

#                 if (not FLAGS.minimal_logs) and step % FLAGS.print_period == 0:
#                     dbg_exec_actions, _ = map_single_arm_exec_action_to_hardware(
#                         policy_actions,
#                         prev_exec_gripper_cmd,
#                     )
#                     since_last_recv = (
#                         None
#                         if network_debug["last_recv_time"] is None
#                         else round(time.time() - network_debug["last_recv_time"], 3)
#                     )

#                     print_blue(
#                         f"[actor-step-debug] step={step}, action_source={action_source}, reward={reward}, "
#                         f"done={done}, truncated={truncated}, recv_count={network_debug['recv_count']}, "
#                         f"applied_count={network_debug['applied_count']}, pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
#                         f"since_last_recv={since_last_recv}, "
#                         f"replay_queue={len(data_store)}, intvn_queue={len(intvn_data_store)}, "
#                         f"stored_gripper={describe_gripper_three_value(actions[6]) if actions.shape[0] == 7 else 'N/A'}, "
#                         f"policy_raw={describe_gripper_three_value(policy_actions[6]) if policy_actions.shape[0] == 7 else 'N/A'}, "
#                         f"mapped_hw={dbg_exec_actions[6] if dbg_exec_actions.shape[0] == 7 else 'N/A'}, "
#                         f"grasp_penalty={transition_for_debug.get('grasp_penalty', 'N/A')}, "
#                         f"had_intervene_action={had_intervene_action}, "
#                         f"pending_interventions={len(episode_pending_interventions)}"
#                     )

#                 if done or truncated:
#                     # 关键：先把本 episode 的 intervention raw transitions
#                     # 用 feedback abs pose 统一转换并 insert 到 data_store / intvn_data_store。
#                     # 随后的 _client_update("episode_end_before_reset") 会把这些数据发给 learner。
#                     flush_stats = _flush_episode_pending_interventions(
#                         reason=f"episode_{episode_index}_end_step_{step}"
#                     )

#                     if "episode" not in info:
#                         info["episode"] = {}

#                     info["episode"]["intervention_count"] = intervention_count
#                     info["episode"]["intervention_steps"] = intervention_steps

#                     info["episode"]["intervention_pending_flushed"] = int(flush_stats["pending"])
#                     info["episode"]["intervention_feedback_abs2rel_converted"] = int(flush_stats["converted"])
#                     info["episode"]["intervention_feedback_abs2rel_fallback"] = int(flush_stats["fallback"])

#                     ep_debug = _extract_episode_debug_info(info)

#                     _log_info(
#                         "actor_episode",
#                         f"[actor-episode-end] episode={episode_index}, step={step}, return={running_return:.4f}, "
#                         f"env_return={ep_debug['return']:.4f}, length={ep_debug['length']}, duration={ep_debug['duration']:.3f}, "
#                         f"success={ep_debug['success']}, intervention_count={intervention_count}, "
#                         f"intervention_steps={intervention_steps}, "
#                         f"feedback_abs2rel_converted={flush_stats['converted']}, fallback={flush_stats['fallback']}, "
#                         f"recv_count={network_debug['recv_count']}, applied_count={network_debug['applied_count']}",
#                         "green" if ep_debug["success"] > 0 else "yellow",
#                     )

#                     try:
#                         client.request("send-stats", {"environment": info})
#                     except Exception as e:
#                         _log_info(
#                             "actor_warning",
#                             f"[actor-send-stats-warning] {e!r}",
#                             "yellow",
#                         )

#                     pbar.set_description(f"last return: {running_return}")

#                     running_return = 0.0
#                     intervention_count = 0
#                     intervention_steps = 0
#                     already_intervened = False
#                     stable_gripper_state = None
#                     prev_exec_gripper_cmd = 80.0
#                     episode_index += 1

#                     if UPDATE_AT_EPISODE_END_BEFORE_RESET:
#                         # 保持原网络通信逻辑：
#                         # 这里 client.update 会把刚 flush 的 intervention data 发给 learner。
#                         _client_update("episode_end_before_reset", force_print=True)
#                         _apply_pending_network("episode_end_before_reset")

#                     obs, _ = env.reset()

#                     if WAIT_NETWORK_AFTER_EVERY_RESET:
#                         if UPDATE_AFTER_RESET_BEFORE_WAIT:
#                             _client_update("after_reset_pre_wait", force_print=True)
#                             _apply_pending_network("after_reset_pre_wait")

#                         _wait_for_network("after_reset_before_next_episode_first_action")
#                     else:
#                         _apply_pending_network("after_reset_no_wait")

#             if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
#                 _save_periodic_actor_buffers(
#                     checkpoint_path=FLAGS.checkpoint_path,
#                     step=step,
#                     transitions=transitions,
#                     demo_transitions=demo_transitions,
#                     penalty_value=FLAGS.grasp_penalty_value,
#                 )
#                 transitions = []
#                 demo_transitions = []

#             timer.tock("total")

#             if step % config.log_period == 0:
#                 try:
#                     client.request("send-stats", {"timer": timer.get_average_times()})
#                 except Exception as e:
#                     if not FLAGS.minimal_logs:
#                         print_yellow(f"[actor-send-timer-warning] {e!r}")

#     finally:
#         remaining_online = len(transitions)
#         remaining_demo = len(demo_transitions)
#         remaining_pending_interventions = len(episode_pending_interventions)

#         print_yellow(
#             f"[actor-exit] actor loop exited. Unsaved partial buffers are discarded by design: "
#             f"online={remaining_online}, demo={remaining_demo}, "
#             f"pending_interventions_not_flushed={remaining_pending_interventions}, "
#             f"pending_recv_count={network_debug.get('pending_recv_count', 0)}, "
#             f"recv_count={network_debug.get('recv_count', 0)}, applied_count={network_debug.get('applied_count', 0)}. "
#             f"Only periodic numeric files transitions_1000.pkl, transitions_2000.pkl, ... are persisted."
#         )

# # =============================================================================
# # 11. learner
# # =============================================================================

# def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
#     latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None
#     start_step = int(os.path.basename(latest_ckpt)[11:]) + 1 if latest_ckpt is not None else 0
#     step = start_step

#     def stats_callback(type: str, payload: dict) -> dict:
#         assert type == "send-stats", f"Invalid request type: {type}"
#         if wandb_logger is not None:
#             wandb_logger.log(payload, step=step)
#         return {}

#     trainer_cfg = _build_trainer_config()
#     _log_trainer_config(trainer_cfg, "learner")
#     server = TrainerServer(trainer_cfg, request_callback=stats_callback)
#     server.register_data_store("actor_env", replay_buffer)
#     server.register_data_store("actor_env_intvn", demo_buffer)
#     server.start(threaded=True)
#     print_green("learner TrainerServer started.")

#     if PUBLISH_INITIAL_NETWORK_BEFORE_WARMUP:
#         _publish_network_to_actor(server, agent.state.params, reason="initial_before_replay_warmup", step=start_step)
#         print_green("sent initial network to actor before replay warmup")

#     pbar = tqdm.tqdm(total=config.training_starts, initial=len(replay_buffer), desc="Filling up replay buffer", position=0, leave=True)
#     last_warmup_publish_t = time.time()
#     while len(replay_buffer) < config.training_starts:
#         pbar.update(len(replay_buffer) - pbar.n)
#         now = time.time()
#         if FLAGS.warmup_publish_period_s > 0 and now - last_warmup_publish_t >= FLAGS.warmup_publish_period_s:
#             _publish_network_to_actor(server, agent.state.params, reason="warmup_republish", step=step)
#             last_warmup_publish_t = now
#         time.sleep(1)
#     pbar.update(len(replay_buffer) - pbar.n)
#     pbar.close()

#     if PUBLISH_NETWORK_AFTER_WARMUP:
#         _publish_network_to_actor(server, agent.state.params, reason="after_replay_warmup", step=start_step)
#         print_green("resent initial network to actor after replay warmup")

#     replay_iterator = replay_buffer.get_iterator(
#         sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
#         device=sharding.replicate(),
#     )
#     demo_iterator = demo_buffer.get_iterator(
#         sample_args={"batch_size": config.batch_size // 2, "pack_obs_and_next_obs": True},
#         device=sharding.replicate(),
#     )

#     timer = Timer()
#     if isinstance(agent, SACAgent):
#         train_critic_networks_to_update = frozenset({"critic"})
#         train_networks_to_update = frozenset({"critic", "actor", "temperature"})
#     else:
#         train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
#         train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

#     for step in tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
#         last_critics_info = None
#         for _ in range(config.cta_ratio - 1):
#             with timer.context("sample_replay_buffer"):
#                 batch = next(replay_iterator)
#                 demo_batch = next(demo_iterator)
#                 batch = concat_batches(batch, demo_batch, axis=0)
#             with timer.context("train_critics"):
#                 agent, critics_info = agent.update(batch, networks_to_update=train_critic_networks_to_update)
#                 last_critics_info = critics_info

#         with timer.context("train"):
#             batch = next(replay_iterator)
#             demo_batch = next(demo_iterator)
#             batch = concat_batches(batch, demo_batch, axis=0)
#             agent, update_info = agent.update(batch, networks_to_update=train_networks_to_update)

#         if step > 0 and step % config.steps_per_update == 0:
#             _publish_network_to_actor(server, agent.state.params, reason="periodic_update", step=step)

#         if step % config.log_period == 0:
#             update_info_loggable = _to_loggable_pytree(update_info)
#             critics_info_loggable = _to_loggable_pytree(last_critics_info) if last_critics_info is not None else {}
#             if wandb_logger is not None:
#                 wandb_logger.log(update_info_loggable, step=step)
#                 if critics_info_loggable:
#                     wandb_logger.log(critics_info_loggable, step=step)
#                 wandb_logger.log({"timer": timer.get_average_times()}, step=step)
#             _print_learner_training_debug(step, update_info, last_critics_info, timer)

#         if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
#             _save_checkpoint_host(FLAGS.checkpoint_path, agent.state, step=step, keep=100)


# # =============================================================================
# # 12. main
# # =============================================================================

# def _make_agent_and_buffers(config, env, rng, sample_obs=None, sample_action=None):
#     if sample_obs is None:
#         sample_obs = env.observation_space.sample()
#     if sample_action is None:
#         sample_action = env.action_space.sample()

#     if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
#         agent = make_sac_pixel_agent(
#             seed=FLAGS.seed,
#             sample_obs=sample_obs,
#             sample_action=sample_action,
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = False
#     elif config.setup_mode == "single-arm-learned-gripper":
#         agent = make_sac_pixel_agent_hybrid_single_arm(
#             seed=FLAGS.seed,
#             sample_obs=sample_obs,
#             sample_action=sample_action,
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = True
#     elif config.setup_mode == "dual-arm-learned-gripper":
#         agent = make_sac_pixel_agent_hybrid_dual_arm(
#             seed=FLAGS.seed,
#             sample_obs=sample_obs,
#             sample_action=sample_action,
#             image_keys=config.image_keys,
#             encoder_type=config.encoder_type,
#             discount=config.discount,
#         )
#         include_grasp_penalty = True
#     else:
#         raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")
#     return agent, include_grasp_penalty


# def _resolve_env_config_object():
#     """
#     兼容两种 config.py 写法：

#     1) env_config = GalaxeaUSBTrainConfig()
#        -> env_config 已经是对象，不能再 env_config()

#     2) env_config = GalaxeaUSBTrainConfig
#        或 def env_config(): ...
#        -> env_config 可调用，需要 env_config()
#     """
#     return env_config() if callable(env_config) else env_config


# def main(_):
#     global config
#     config = _resolve_env_config_object()

#     assert config.batch_size % num_devices == 0
#     rng = jax.random.PRNGKey(FLAGS.seed)
#     rng, sampling_rng = jax.random.split(rng)

#     # learner 端通常 fake_env=True；你的 Galaxea config 在 fake_env=True 时可能返回 None。
#     # 因此只有 env 真正存在时才套 RecordEpisodeStatistics。
#     env = config.get_environment(
#         fake_env=FLAGS.learner,
#         save_video=FLAGS.save_video,
#         classifier=True,
#     )
#     if env is not None:
#         env = RecordEpisodeStatistics(env)
#     elif FLAGS.actor:
#         raise RuntimeError("actor=True 时 env 不能为 None；请检查 config.get_environment(fake_env=False)。")
#     else:
#         print_yellow("[learner-env] config.get_environment(fake_env=True) returned None; learner 将使用 demo 推断 observation/action spaces。")

#     actor_to_learner_image_keys = resolve_actor_to_learner_image_keys(config)
#     validate_actor_to_learner_image_keys(config, actor_to_learner_image_keys)

#     sample_obs = None
#     sample_action = None
#     demo_files = None
#     demo_observation_space = None
#     demo_action_space = None

#     if FLAGS.demo_path is not None:
#         demo_files = resolve_demo_paths(FLAGS.demo_path)
#         demo_observation_space, demo_action_space, sample_obs, sample_action = build_spaces_and_samples_from_demos(
#             demo_files,
#             config,
#             actor_to_learner_image_keys,
#         )

#     if FLAGS.learner:
#         assert FLAGS.demo_path is not None, "learner 必须提供 --demo_path，因为 fake_env=None 时要靠 demo 推断网络和 buffer spaces。"
#         assert sample_obs is not None and sample_action is not None
#         assert demo_observation_space is not None and demo_action_space is not None

#     agent, include_grasp_penalty = _make_agent_and_buffers(
#         config,
#         env,
#         rng,
#         sample_obs=sample_obs,
#         sample_action=sample_action,
#     )

#     agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

#     if FLAGS.checkpoint_path is not None:
#         latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)) if os.path.exists(FLAGS.checkpoint_path) else None
#         if latest is not None:
#             input("Checkpoint path already has checkpoint. Press Enter to resume training.")
#             ckpt = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state)
#             agent = agent.replace(state=ckpt)
#             print_green(f"Loaded previous checkpoint: {latest}")

#     def _get_spaces_for_buffers():
#         """
#         actor 端：用真实 env spaces。
#         learner 端 fake_env=None：用 demo 推断出的 spaces。
#         """
#         if env is not None:
#             return env.observation_space, env.action_space

#         assert demo_observation_space is not None, "env=None 时缺少 demo_observation_space"
#         assert demo_action_space is not None, "env=None 时缺少 demo_action_space"
#         return demo_observation_space, demo_action_space

#     def create_replay_buffer_and_wandb_logger():
#         observation_space, action_space = _get_spaces_for_buffers()
#         replay_buffer = MemoryEfficientReplayBufferDataStore(
#             observation_space,
#             action_space,
#             capacity=config.replay_buffer_capacity,
#             image_keys=config.image_keys,
#             include_grasp_penalty=include_grasp_penalty,
#         )
#         wandb_logger = make_wandb_logger(
#             project="hil-serl",
#             description=FLAGS.exp_name,
#             debug=FLAGS.debug,
#         )
#         return replay_buffer, wandb_logger

#     trainer_cfg = _build_trainer_config()

#     if FLAGS.learner:
#         sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
#         replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

#         observation_space, action_space = _get_spaces_for_buffers()
#         demo_buffer = MemoryEfficientReplayBufferDataStore(
#             observation_space,
#             action_space,
#             capacity=config.replay_buffer_capacity,
#             image_keys=config.image_keys,
#             include_grasp_penalty=include_grasp_penalty,
#         )

#         demo_files = resolve_demo_paths(FLAGS.demo_path)
#         for path in demo_files:
#             with open(path, "rb") as f:
#                 transitions = pkl.load(f)
#             transitions = prune_transition_list_for_actor_to_learner(
#                 transitions,
#                 actor_to_learner_image_keys,
#                 config,
#                 source=f"demo_load:{path}",
#                 strict=FLAGS.actor_to_learner_strict_keys,
#                 print_summary=True,
#             )
#             transitions = sanitize_transition_list_for_storage(
#                 transitions,
#                 source=f"demo_load:{path}",
#                 print_summary=True,
#             )
#             transitions = sync_transition_list_grasp_penalty(
#                 transitions,
#                 source=f"demo_load:{path}",
#                 penalty_value=FLAGS.grasp_penalty_value,
#                 print_summary=True,
#             )
#             for transition in transitions:
#                 demo_buffer.insert(transition)

#         print_green(f"demo buffer size: {len(demo_buffer)}")
#         print_green(f"online buffer size: {len(replay_buffer)}")

#         if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
#             loaded_transitions = _load_numeric_transition_files_from_dir(
#                 os.path.join(FLAGS.checkpoint_path, "buffer"),
#                 actor_to_learner_image_keys=actor_to_learner_image_keys,
#                 config=config,
#                 source_prefix="buffer_load",
#                 strict=FLAGS.actor_to_learner_strict_keys,
#                 penalty_value=FLAGS.grasp_penalty_value,
#             )
#             for transition in loaded_transitions:
#                 replay_buffer.insert(transition)
#             print_green(f"Loaded previous numeric buffer data. Replay buffer size: {len(replay_buffer)}")

#         if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer")):
#             loaded_demo_transitions = _load_numeric_transition_files_from_dir(
#                 os.path.join(FLAGS.checkpoint_path, "demo_buffer"),
#                 actor_to_learner_image_keys=actor_to_learner_image_keys,
#                 config=config,
#                 source_prefix="demo_buffer_load",
#                 strict=FLAGS.actor_to_learner_strict_keys,
#                 penalty_value=FLAGS.grasp_penalty_value,
#             )
#             for transition in loaded_demo_transitions:
#                 demo_buffer.insert(transition)
#             print_green(f"Loaded previous numeric demo buffer data. Demo buffer size: {len(demo_buffer)}")

#         print_green("starting learner loop")
#         learner(
#             sampling_rng,
#             agent,
#             replay_buffer,
#             demo_buffer=demo_buffer,
#             wandb_logger=wandb_logger,
#         )

#     elif FLAGS.actor:
#         if FLAGS.actor_expect_gpu:
#             backend = jax.default_backend()
#             if backend != "gpu" and backend != "cuda":
#                 print_yellow(f"⚠️ actor 当前 JAX backend={backend}，不是 GPU/CUDA。")
#             else:
#                 print_green(f"✅ actor JAX backend={backend}")

#         sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
#         data_store = QueuedDataStore(50000)
#         intvn_data_store = QueuedDataStore(50000)
#         print_green("starting actor loop")
#         actor(agent, data_store, intvn_data_store, env, sampling_rng, config, trainer_cfg)
#     else:
#         raise NotImplementedError("Must set either --learner=True or --actor=True")


# if __name__ == "__main__":
#     app.run(main)



