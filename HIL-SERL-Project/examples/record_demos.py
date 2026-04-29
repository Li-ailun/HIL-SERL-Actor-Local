"""
record_abs_pose_demos_and_convert.py

录制 raw absolute demos，并离线转换成当前 RLPD/SERL 训练使用的相对增量 demo pkl。

记录内容：
  - observations / next_observations：图像、state 等 env 输出
  - gripper feedback：从 obs["state"] 最后一维或 gripper key 提取
  - feedback pose：/motion_control/pose_ee_arm_right
  - target pose  ：/motion_target/target_pose_arm_right

转换方式：
  1 = feedback 版本：用 /motion_control/pose_ee_arm_right 生成 action[:6]
  2 = target   版本：用 /motion_target/target_pose_arm_right 生成 action[:6]
  3 = both     版本：两个版本都导出，并比较二者增量误差

最终导出的训练 pkl 格式：
  list[transition]
  transition = {
    "observations", "actions", "next_observations", "rewards", "masks", "dones", "infos", "grasp_penalty"
  }
  actions.shape == (7,)
  action[:6] = normalized relative delta pose, clipped to [-1, 1]
  action[6]  = gripper event label: close=-1, hold=0, open=+1

典型用法：
  # 录制 raw，并在末尾输入 1/2/3 选择转换版本（反馈，指令，都导出并计算两者误差）
  python record_abs_pose_demos_and_convert.py 
    --successes_needed=20 
    --convert_source=prompt

  #如果后续需要改变动作缩放测试效果，直接修改具体任务的config，或者如下手动指定动作缩放：
  python record_abs_pose_demos_and_convert.py \
  --raw_input_path ./raw_abs_demo_data_single/galaxea_usb_insertion_single_20_raw_abs_xxx.pkl \
  --convert_source=target \
  --pos_scale=0.02 \
  --rot_scale=0.04

  # 不重新录制，只用已有 raw 重新转换 target 版本
  python record_abs_pose_demos_and_convert.py \
    --raw_input_path ./raw_abs_demo_data_single/xxx_raw_abs.pkl \
    --convert_source=target


"""

"""
record_abs_pose_demos_and_convert_wait_vr.py

在原 demo 录制逻辑基础上，只增加 raw absolute pose 记录和离线转换功能。

关键保持不变 / 特别注意：
1. reset 后不会立刻 env.step(zero)。
   reset 后脚本只等待 VR mode 切换；等待期间不调用 env.step，不发送 zero action。
2. 检测到 VR 接管后，才进入原来的 env.step(raw_actions=zeros) 录制循环。
3. 成功/失败、classifier、manual_confirm、静止帧过滤、max_episode_steps 逻辑保持原录制脚本风格。
4. 新增记录：
   - observations / next_observations
   - reward / done / truncated / info
   - gripper_feedback_before / after
   - raw_intervene_action
   - /motion_control/pose_ee_arm_right before / after
   - /motion_target/target_pose_arm_right before / after
5. 录完后离线转换成当前 RLPD/SERL 可读的标准 demo pkl：
   transition = {
       observations,
       actions: [dx, dy, dz, droll, dpitch, dyaw, gripper_event],
       next_observations,
       rewards,
       masks,
       dones,
       infos,
       grasp_penalty,
   }

转换选择：
  1 / feedback：用 /motion_control/pose_ee_arm_right 反馈位姿生成 action[:6]
  2 / target  ：用 /motion_target/target_pose_arm_right 指令位姿生成 action[:6]
  3 / both    ：两个版本都导出，并计算两者增量误差

典型录制：
  python record_abs_pose_demos_and_convert_wait_vr.py \
    --successes_needed=20 \
    --classifier=True \
    --convert_source=prompt


只重新转换已有 raw，不重新录：
  python record_demos.py \
    --raw_input_path ./raw_abs_demo_data_single/galaxea_usb_insertion_single_1_raw_abs_2026-04-29_14-40-08.pkl \
    --convert_source=feedback \
    --pos_scale=0.01 \
    --rot_scale=0.03 \
    --converted_save_dir ./demo_data_single
"""

# =============================================================================
# 0. 像 actor 一样，先强制本地 CPU，避免 classifier=True 时本地环境炸
# =============================================================================
import os
import sys

os.environ.pop("JAX_PLATFORMS", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# =============================================================================
# 1. imports
# =============================================================================
import copy
import datetime
import json
import math
import pickle as pkl
import threading
import time
from collections import Counter

import numpy as np
from absl import app, flags
from tqdm import tqdm

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Int8, Int16, Int32, UInt8, UInt16, UInt32
except Exception as e:
    rclpy = None
    Node = object
    PoseStamped = None
    Int8 = Int16 = Int32 = UInt8 = UInt16 = UInt32 = None
    _ROS_IMPORT_ERROR = e
else:
    _ROS_IMPORT_ERROR = None

# =============================================================================
# 2. 路径配置
# =============================================================================
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(THIS_DIR) == "inspect":
    EXAMPLES_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
else:
    EXAMPLES_DIR = THIS_DIR

PROJECT_ROOT = os.path.abspath(os.path.join(EXAMPLES_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from examples.galaxea_task.usb_pick_insertion_single.config import env_config

# =============================================================================
# 3. flags
# =============================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "galaxea_usb_insertion_single", "Name of experiment.")
flags.DEFINE_integer("successes_needed", 30, "Number of successful demos to collect.")
flags.DEFINE_integer("max_episode_steps", 650, "Maximum env steps per demo episode.")
flags.DEFINE_boolean("classifier", True, "Whether to use reward classifier as success/end signal.")
flags.DEFINE_boolean("save_video", False, "Whether to save video during recording.")
flags.DEFINE_boolean("manual_confirm_on_success", False, "Manually confirm success even when classifier says succeed=True.")

# raw recording / conversion mode
flags.DEFINE_string("raw_input_path", "", "If set, skip robot recording and only convert this raw absolute demo pkl.")
flags.DEFINE_string("convert_source", "prompt", "prompt / feedback / target / both, or 1 / 2 / 3.")

# ROS pose topics
flags.DEFINE_string("feedback_pose_topic", "/motion_control/pose_ee_arm_right", "Actual EE feedback PoseStamped topic.")
flags.DEFINE_string("target_pose_topic", "/motion_target/target_pose_arm_right", "Target command PoseStamped topic.")
flags.DEFINE_float("wait_initial_pose_sec", 2.0, "Wait seconds for initial pose messages after subscriber starts.")

# Optional control mode topic. 如果不设置，会尝试从 env 属性读；读不到则人工回车确认。
flags.DEFINE_string(
    "control_mode_topic",
    "",
    "Optional std_msgs integer topic for control mode. Empty means try env attrs, then manual Enter fallback.",
)
flags.DEFINE_integer("vr_control_mode_value", 0, "Control mode value that means VR control/takeover.")
flags.DEFINE_float("wait_vr_poll_sec", 0.05, "Polling sleep while waiting VR takeover without env.step.")
flags.DEFINE_boolean(
    "manual_start_if_no_control_mode",
    True,
    "If true, when control mode cannot be read automatically, ask user to press Enter after switching VR mode.",
)

# output dirs
flags.DEFINE_string("raw_save_dir", "", "Raw absolute demo output directory. Empty = examples/raw_abs_demo_data_single.")
flags.DEFINE_string("converted_save_dir", "", "Converted train demo output directory. Empty = examples/demo_data_single.")
flags.DEFINE_string("report_save_dir", "", "Diagnostic report output directory. Empty = converted_save_dir.")

# normalization
flags.DEFINE_float("pos_scale", 0.0, "Override POS_SCALE. 0 means use config.POS_SCALE.")
flags.DEFINE_float("rot_scale", 0.0, "Override ROT_SCALE. 0 means use config.ROT_SCALE.")
flags.DEFINE_boolean("clip_action", True, "Clip normalized action[:6] to [-1, 1].")

# gripper / penalty
flags.DEFINE_float("grasp_penalty_value", -0.02, "Penalty for close/open gripper event.")
flags.DEFINE_float("feedback_close_max", 30.0, "Gripper feedback <= close_max means stable closed.")
flags.DEFINE_float("feedback_open_min", 70.0, "Gripper feedback >= open_min means stable open.")

# filtering / diagnostics
flags.DEFINE_boolean("record_all_raw_steps", False, "Keep all raw env steps after VR starts.")
flags.DEFINE_boolean("drop_static_converted", False, "Drop zero-action non-terminal converted transitions.")
flags.DEFINE_float("static_action_eps", 1e-8, "Epsilon for converted static action filtering.")
flags.DEFINE_boolean(
    "keep_pose_motion_without_intervention",
    False,
    "If true, keep raw steps with pose motion even when info['intervene_action'] is missing/zero. Default false to preserve old demo filtering.",
)
flags.DEFINE_float("pose_static_pos_eps", 1e-5, "Raw step static threshold in meters.")
flags.DEFINE_float("pose_static_rot_eps", 1e-4, "Raw step static threshold in radians.")

# final converted demo observation pruning
flags.DEFINE_string(
    "demo_image_keys",
    "",
    "Comma-separated image keys saved into converted training demo. Empty means use env_config.image_keys, e.g. head_rgb,right_wrist_rgb.",
)
flags.DEFINE_string(
    "demo_extra_obs_keys",
    "state",
    "Comma-separated non-image obs keys saved into converted training demo. Default: state.",
)
flags.DEFINE_boolean(
    "strict_demo_obs_keys",
    True,
    "If true, raise error when final demo image_keys/state are missing during conversion.",
)

# =============================================================================
# 4. print / misc utils
# =============================================================================
def print_green(x):
    print("\033[92m{}\033[00m".format(x))


def print_yellow(x):
    print("\033[93m{}\033[00m".format(x))


def print_blue(x):
    print("\033[94m{}\033[00m".format(x))


def print_red(x):
    print("\033[91m{}\033[00m".format(x))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def ask_success_from_terminal():
    while True:
        try:
            manual_rew = int(input("Success? (1/0): ").strip())
            if manual_rew in [0, 1]:
                return bool(manual_rew)
            print("❌ 请输入 1 或 0。")
        except ValueError:
            print("❌ 输入无效，请输入 1 或 0。")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def get_config_attr(name, default=None):
    if hasattr(env_config, name):
        return getattr(env_config, name)
    try:
        cfg = env_config()
        if hasattr(cfg, name):
            return getattr(cfg, name)
    except Exception:
        pass
    return default


def resolve_scales():
    pos_scale = float(FLAGS.pos_scale) if float(FLAGS.pos_scale) > 0 else float(get_config_attr("POS_SCALE", 0.02))
    rot_scale = float(FLAGS.rot_scale) if float(FLAGS.rot_scale) > 0 else float(get_config_attr("ROT_SCALE", 0.04))
    if pos_scale <= 0 or rot_scale <= 0:
        raise ValueError(f"POS_SCALE/ROT_SCALE 必须 >0, got pos_scale={pos_scale}, rot_scale={rot_scale}")
    return pos_scale, rot_scale


# =============================================================================
# 4.1 final demo observation pruning
# =============================================================================
def _split_csv_keys(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def get_policy_image_keys_for_demo():
    """
    最终 converted demo 保存哪些图像。

    旧 demo 保存逻辑：按 RLPD policy 的 image_keys 保存，而不是按 ENV_IMAGE_KEYS 保存。
    ENV_IMAGE_KEYS 只是环境/相机里可用图像，classifier_keys 只是奖励分类器输入。
    """
    override = _split_csv_keys(getattr(FLAGS, "demo_image_keys", ""))
    if override:
        return override

    keys = get_config_attr("image_keys", None)
    if keys is None:
        keys = ["head_rgb", "right_wrist_rgb"]
    return list(keys)


def get_extra_obs_keys_for_demo():
    keys = _split_csv_keys(getattr(FLAGS, "demo_extra_obs_keys", "state"))
    return keys if keys else ["state"]


def _get_obs_value_by_key(obs, key):
    if obs is None or not isinstance(obs, dict):
        raise KeyError(f"obs is not dict, cannot get key={key}, type={type(obs)}")

    if key in obs:
        return obs[key]

    images = obs.get("images", None)
    if isinstance(images, dict) and key in images:
        return images[key]

    raise KeyError(f"observation 中找不到 key={key}; available={list(obs.keys())}")


def prune_obs_for_converted_demo(obs, image_keys=None, extra_obs_keys=None, strict=None):
    """
    只用于最终 converted training pkl。

    raw absolute pkl 继续保存完整 ENV_IMAGE_KEYS，方便 classifier/诊断。
    converted demo pkl 必须按旧 demo 逻辑，只保存 policy image_keys + state。
    """
    if image_keys is None:
        image_keys = get_policy_image_keys_for_demo()
    if extra_obs_keys is None:
        extra_obs_keys = get_extra_obs_keys_for_demo()
    if strict is None:
        strict = bool(FLAGS.strict_demo_obs_keys)

    keep = {}
    for key in list(image_keys) + list(extra_obs_keys):
        try:
            keep[key] = copy.deepcopy(_get_obs_value_by_key(obs, key))
        except KeyError as e:
            if strict:
                raise
            print_yellow(f"⚠️ converted demo pruning skipped missing obs key={key}: {e}")
    return keep


def summarize_obs_keys_for_transitions(transitions, name):
    if not transitions:
        print_yellow(f"⚠️ {name}: no transitions to summarize obs keys")
        return
    obs = transitions[0].get("observations", {})
    next_obs = transitions[0].get("next_observations", {})
    print_blue("\n===== converted demo obs pruning check =====")
    print(f"{name} observations keys     : {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print(f"{name} next_observations keys: {list(next_obs.keys()) if isinstance(next_obs, dict) else type(next_obs)}")
    print(f"expected policy image_keys   : {get_policy_image_keys_for_demo()}")
    print(f"expected extra obs keys      : {get_extra_obs_keys_for_demo()}")

# =============================================================================
# 5. gripper helpers
# =============================================================================
def extract_gripper_feedback(obs):
    """从 obs['state'] 提取夹爪反馈。单臂常见 state 最后一维是 gripper。"""
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


def stable_gripper_from_feedback(feedback, prev_state=None):
    """返回稳定夹爪状态：-1 closed, +1 open；中间区保持 prev_state。"""
    if feedback is None:
        return prev_state
    x = float(feedback)
    if x <= float(FLAGS.feedback_close_max):
        return -1
    if x >= float(FLAGS.feedback_open_min):
        return +1
    return prev_state


def gripper_event_from_feedback(prev_feedback, next_feedback, prev_stable_state):
    """生成事件标签：open->closed: -1, closed->open: +1, otherwise 0。"""
    prev_state = stable_gripper_from_feedback(prev_feedback, prev_stable_state)
    if prev_state is None:
        prev_state = +1  # reset 后默认开爪
    next_state = stable_gripper_from_feedback(next_feedback, prev_state)

    if prev_state == +1 and next_state == -1:
        return -1.0, next_state, prev_state
    if prev_state == -1 and next_state == +1:
        return +1.0, next_state, prev_state
    return 0.0, next_state, prev_state


def describe_gripper_event(x):
    x = float(x)
    if x <= -0.5:
        return "close(-1)"
    if x >= 0.5:
        return "open(+1)"
    return "hold(0)"


def recompute_grasp_penalty_from_action(action):
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        return 0.0
    return float(FLAGS.grasp_penalty_value) if abs(float(a[6])) > 0.5 else 0.0

# =============================================================================
# 6. quaternion / pose helpers
# =============================================================================
def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / n
    if q[3] < 0:
        q = -q
    return q


def quat_inv(q):
    q = normalize_quat(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = normalize_quat(q1)
    x2, y2, z2, w2 = normalize_quat(q2)
    return normalize_quat(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64))


def quat_to_rotvec(q):
    q = normalize_quat(q)
    xyz = q[:3]
    w = float(q[3])
    if w < 0:
        q = -q
        xyz = q[:3]
        w = float(q[3])
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    axis = xyz / sin_half
    return axis * angle


def pose_to_array(pose_rec):
    if pose_rec is None:
        return None
    if isinstance(pose_rec, dict):
        if "pose" in pose_rec:
            return pose_to_array(pose_rec["pose"])
        keys = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if all(k in pose_rec for k in keys):
            arr = np.asarray([pose_rec[k] for k in keys], dtype=np.float64)
            arr[3:7] = normalize_quat(arr[3:7])
            return arr
        return None
    arr = np.asarray(pose_rec, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        return None
    arr = arr.copy()
    arr[3:7] = normalize_quat(arr[3:7])
    return arr


def pose_delta_to_components(pose_before, pose_after):
    p0 = pose_to_array(pose_before)
    p1 = pose_to_array(pose_after)
    if p0 is None or p1 is None:
        return None, None

    pos0, q0 = p0[:3], normalize_quat(p0[3:7])
    pos1, q1 = p1[:3], normalize_quat(p1[3:7])
    if np.dot(q0, q1) < 0:
        q1 = -q1

    delta_pos = pos1 - pos0
    q_delta = quat_mul(q1, quat_inv(q0))
    rotvec = quat_to_rotvec(q_delta)
    return delta_pos.astype(np.float64), rotvec.astype(np.float64)


def normalized_action6_from_pose_delta(pose_before, pose_after, pos_scale, rot_scale):
    delta_pos, rotvec = pose_delta_to_components(pose_before, pose_after)
    if delta_pos is None:
        return None, None

    raw = np.zeros(6, dtype=np.float64)
    raw[:3] = delta_pos / float(pos_scale)
    raw[3:6] = rotvec / float(rot_scale)
    clipped = np.clip(raw, -1.0, 1.0) if FLAGS.clip_action else raw.copy()

    extra = {
        "delta_pos_m": delta_pos,
        "delta_rotvec_rad": rotvec,
        "raw_normalized_action6": raw,
        "clipped_action6": clipped,
        "was_clipped": bool(np.any(np.abs(raw) > 1.0 + 1e-8)),
    }
    return clipped.astype(np.float32), extra


def pose_motion_norm(pose_before, pose_after):
    delta_pos, rotvec = pose_delta_to_components(pose_before, pose_after)
    if delta_pos is None:
        return None, None
    return float(np.linalg.norm(delta_pos)), float(np.linalg.norm(rotvec))

# =============================================================================
# 7. ROS subscriber node: pose + optional control mode
# =============================================================================
def stamp_to_float(stamp):
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return None


def pose_stamped_to_record(msg):
    if msg is None:
        return None
    return {
        "pose": np.asarray([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64),
        "frame_id": str(msg.header.frame_id),
        "stamp": {
            "sec": int(msg.header.stamp.sec),
            "nanosec": int(msg.header.stamp.nanosec),
            "stamp_float": stamp_to_float(msg.header.stamp),
        },
        "recv_time": time.time(),
    }


def int_msg_to_value(msg):
    try:
        return int(msg.data)
    except Exception:
        return None


STD_INT_MSG_TYPES = {
    "std_msgs/msg/Int8": Int8,
    "std_msgs/msg/Int16": Int16,
    "std_msgs/msg/Int32": Int32,
    "std_msgs/msg/UInt8": UInt8,
    "std_msgs/msg/UInt16": UInt16,
    "std_msgs/msg/UInt32": UInt32,
}


class RecorderRosNode(Node):
    def __init__(self, feedback_topic, target_topic, control_mode_topic):
        super().__init__("abs_pose_demo_recorder_node")
        self._lock = threading.Lock()

        self.feedback_pose = None
        self.target_pose = None
        self.feedback_count = 0
        self.target_count = 0

        self.control_mode_topic = control_mode_topic
        self.control_mode = None
        self.control_mode_count = 0
        self.control_mode_type = None

        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, feedback_topic, self._feedback_cb, pose_qos)
        self.create_subscription(PoseStamped, target_topic, self._target_cb, pose_qos)

        if control_mode_topic:
            self._try_create_control_mode_subscription(control_mode_topic)

    def _try_create_control_mode_subscription(self, topic):
        # 尝试根据 ROS graph 里的 topic type 创建订阅。
        msg_cls = None
        topic_type = None
        try:
            for name, types in self.get_topic_names_and_types():
                if name == topic and len(types) > 0:
                    topic_type = types[0]
                    msg_cls = STD_INT_MSG_TYPES.get(topic_type)
                    break
        except Exception:
            pass

        # 如果 graph 还没发现 topic，默认试 Int32。
        if msg_cls is None and topic_type is None:
            topic_type = "std_msgs/msg/Int32(default)"
            msg_cls = Int32

        if msg_cls is None:
            print_yellow(f"⚠️ control_mode_topic={topic} 类型 {topic_type} 暂不支持；将不用 ROS topic 自动检测 VR mode。")
            return

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(msg_cls, topic, self._control_mode_cb, qos)
        self.control_mode_type = topic_type
        print_green(f"✅ control mode subscriber: {topic}, type={topic_type}")

    def _feedback_cb(self, msg):
        rec = pose_stamped_to_record(msg)
        with self._lock:
            self.feedback_pose = rec
            self.feedback_count += 1

    def _target_cb(self, msg):
        rec = pose_stamped_to_record(msg)
        with self._lock:
            self.target_pose = rec
            self.target_count += 1

    def _control_mode_cb(self, msg):
        value = int_msg_to_value(msg)
        with self._lock:
            self.control_mode = value
            self.control_mode_count += 1

    def snapshot(self):
        with self._lock:
            return {
                "feedback": copy.deepcopy(self.feedback_pose),
                "target": copy.deepcopy(self.target_pose),
                "feedback_count": int(self.feedback_count),
                "target_count": int(self.target_count),
                "control_mode": self.control_mode,
                "control_mode_count": int(self.control_mode_count),
                "control_mode_type": self.control_mode_type,
                "snapshot_time": time.time(),
            }


class RosRecorderSubscriber:
    def __init__(self, feedback_topic, target_topic, control_mode_topic):
        if rclpy is None:
            raise RuntimeError(f"无法导入 ROS2 rclpy / PoseStamped: {_ROS_IMPORT_ERROR!r}")
        self.feedback_topic = feedback_topic
        self.target_topic = target_topic
        self.control_mode_topic = control_mode_topic
        self.node = None
        self.thread = None
        self._started = False

    def start(self):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = RecorderRosNode(self.feedback_topic, self.target_topic, self.control_mode_topic)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()
        self._started = True
        print_green("✅ ROS2 subscribers started:")
        print_green(f"   feedback pose: {self.feedback_topic}")
        print_green(f"   target pose  : {self.target_topic}")
        if self.control_mode_topic:
            print_green(f"   control mode : {self.control_mode_topic}")

    def stop(self):
        if not self._started:
            return
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception as e:
            print_yellow(f"⚠️ destroy ROS node failed: {e!r}")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._started = False

    def snapshot(self):
        if self.node is None:
            return None
        return self.node.snapshot()

    def wait_initial_pose_messages(self, timeout_sec):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            snap = self.snapshot()
            if snap and snap.get("feedback") is not None and snap.get("target") is not None:
                print_green(f"✅ 已收到初始 pose: feedback_count={snap['feedback_count']}, target_count={snap['target_count']}")
                return True
            time.sleep(0.05)
        snap = self.snapshot() or {}
        print_yellow(
            "⚠️ 等待初始 pose 超时: "
            f"feedback_received={snap.get('feedback') is not None}, "
            f"target_received={snap.get('target') is not None}, "
            f"feedback_count={snap.get('feedback_count', 0)}, "
            f"target_count={snap.get('target_count', 0)}"
        )
        return False

# =============================================================================
# 8. wait VR takeover without env.step
# =============================================================================
def try_read_control_mode_from_env(env):
    """尽量从 env / wrapper 属性读 control_mode；不调用 env.step。
    兼容旧 env 的 script_control_enabled：True=Mode2 脚本控制，False=Mode0 VR。
    """
    # 先兼容你之前 env 里的 script_control_enabled：False 表示已经回到 VR Mode0。
    for name in ["script_control_enabled"]:
        try:
            if hasattr(env, "get_wrapper_attr"):
                value = env.get_wrapper_attr(name)
                if value is not None:
                    return (2 if bool(value) else 0), f"env.get_wrapper_attr({name})"
        except Exception:
            pass
    try:
        base = getattr(env, "unwrapped", None)
        if base is not None and hasattr(base, "script_control_enabled"):
            value = getattr(base, "script_control_enabled")
            if value is not None:
                return (2 if bool(value) else 0), "env.unwrapped.script_control_enabled"
    except Exception:
        pass

    # get_wrapper_attr 是 gymnasium 推荐接口。
    for name in ["control_mode", "current_control_mode", "mode"]:
        try:
            if hasattr(env, "get_wrapper_attr"):
                value = env.get_wrapper_attr(name)
                if value is not None:
                    return int(value), f"env.get_wrapper_attr({name})"
        except Exception:
            pass

    objs = [env]
    for attr in ["unwrapped", "env"]:
        try:
            obj = getattr(env, attr, None)
            if obj is not None and obj not in objs:
                objs.append(obj)
        except Exception:
            pass

    for obj in objs:
        for name in ["control_mode", "current_control_mode", "mode"]:
            try:
                if hasattr(obj, name):
                    value = getattr(obj, name)
                    if value is not None and not callable(value):
                        return int(value), f"{type(obj).__name__}.{name}"
            except Exception:
                pass
        for name in ["get_control_mode", "get_current_control_mode"]:
            try:
                fn = getattr(obj, name, None)
                if callable(fn):
                    value = fn()
                    if value is not None:
                        return int(value), f"{type(obj).__name__}.{name}()"
            except Exception:
                pass
    return None, None


def wait_for_vr_takeover_no_env_step(env, ros_sub, reason):
    """
    reset 后等待 VR 接管。
    这里绝对不调用 env.step，也不发送 zero action。
    """
    print_yellow("\n⏳ reset 后等待 VR 接管：此阶段不调用 env.step，不发送 zero action，不记录 reset 等待帧。")
    print_yellow(f"   reason={reason}, expected_vr_control_mode={FLAGS.vr_control_mode_value}")

    warned_manual = False
    last_print_t = 0.0

    while True:
        # 1) 优先读 ROS control_mode topic
        snap = ros_sub.snapshot() if ros_sub is not None else None
        if snap is not None and snap.get("control_mode") is not None:
            mode = int(snap["control_mode"])
            if mode == int(FLAGS.vr_control_mode_value):
                print_green(f"🎮 检测到 VR control_mode={mode}，开始记录/执行 env.step。")
                return

        # 2) 再尝试从 env 属性读，不调用 step
        mode, source = try_read_control_mode_from_env(env)
        if mode is not None:
            if mode == int(FLAGS.vr_control_mode_value):
                print_green(f"🎮 检测到 VR control_mode={mode} from {source}，开始记录/执行 env.step。")
                return

        # 3) 没有可用自动检测时，人工确认
        if (snap is None or snap.get("control_mode") is None) and mode is None and FLAGS.manual_start_if_no_control_mode:
            if not warned_manual:
                print_yellow("⚠️ 当前没有可读的 control_mode。")
                print_yellow("   请先切到 VR 控制模式；确认已经切好后按 Enter 开始。")
                input("按 Enter 开始本条 demo 的 env.step 录制...")
                return
            warned_manual = True

        now = time.time()
        if now - last_print_t > 2.0:
            cm = None if snap is None else snap.get("control_mode")
            cc = 0 if snap is None else snap.get("control_mode_count", 0)
            print(f"   waiting VR... ros_control_mode={cm}, count={cc}, env_mode={mode}")
            last_print_t = now
        time.sleep(float(FLAGS.wait_vr_poll_sec))

# =============================================================================
# 9. raw recording helpers
# =============================================================================
def make_raw_step(obs, next_obs, reward, done, truncated, info, raw_actions, before_pose_snapshot, after_pose_snapshot, episode_step, wall_dt):
    episode_end = bool(done or truncated)
    raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
    return {
        "observations": copy.deepcopy(obs),
        "next_observations": copy.deepcopy(next_obs),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "episode_end": bool(episode_end),
        "mask": 1.0 - float(episode_end),
        "infos": copy.deepcopy(info),
        "raw_intervene_action": raw_actions.astype(np.float32),
        "has_intervene_action": bool("intervene_action" in info),
        "episode_step": int(episode_step),
        "record_time": time.time(),
        "env_step_wall_dt": float(wall_dt),
        "gripper_feedback_before": extract_gripper_feedback(obs),
        "gripper_feedback_after": extract_gripper_feedback(next_obs),
        "poses": {
            "feedback": {
                "before": copy.deepcopy((before_pose_snapshot or {}).get("feedback")),
                "after": copy.deepcopy((after_pose_snapshot or {}).get("feedback")),
            },
            "target": {
                "before": copy.deepcopy((before_pose_snapshot or {}).get("target")),
                "after": copy.deepcopy((after_pose_snapshot or {}).get("target")),
            },
        },
        "pose_counts": {
            "before_feedback_count": int((before_pose_snapshot or {}).get("feedback_count", 0)),
            "before_target_count": int((before_pose_snapshot or {}).get("target_count", 0)),
            "after_feedback_count": int((after_pose_snapshot or {}).get("feedback_count", 0)),
            "after_target_count": int((after_pose_snapshot or {}).get("target_count", 0)),
        },
    }


def should_keep_raw_step_like_old_demo(raw_step, raw_actions, episode_end):
    """
    保留原 demo 脚本的核心过滤逻辑：
    - raw action 非零才记录；
    - reset 等待阶段不会进到这里；
    - 额外保留终止帧 / 夹爪反馈变化帧，保证 reward/done 和 gripper event 不丢。
    """
    if FLAGS.record_all_raw_steps:
        return True

    raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
    is_static = np.allclose(raw_actions, 0.0, atol=1e-8)
    if not is_static:
        return True

    # 为了避免成功帧丢失，终止帧保留。
    if episode_end or abs(float(raw_step.get("reward", 0.0))) > 1e-12:
        return True

    # 夹爪反馈变化帧保留，避免 close/open 事件丢失。
    gb = raw_step.get("gripper_feedback_before", None)
    ga = raw_step.get("gripper_feedback_after", None)
    if gb is not None and ga is not None and abs(float(ga) - float(gb)) > 1e-6:
        return True

    if FLAGS.keep_pose_motion_without_intervention:
        for source in ("feedback", "target"):
            p = raw_step.get("poses", {}).get(source, {})
            pnorm, rnorm = pose_motion_norm(p.get("before"), p.get("after"))
            if pnorm is not None and pnorm > float(FLAGS.pose_static_pos_eps):
                return True
            if rnorm is not None and rnorm > float(FLAGS.pose_static_rot_eps):
                return True

    return False


def record_raw_absolute_demos():
    print_blue(f"🚀 开始录制 raw absolute demos：{FLAGS.exp_name}")
    print_blue("📌 保留旧逻辑：reset 后先等待 VR mode；等待阶段不 env.step，不发 zero，不记录。")
    print_blue("📌 新增：记录 feedback pose / target pose，结束后离线转换为标准相对增量 demo。\n")

    ros_sub = RosRecorderSubscriber(FLAGS.feedback_pose_topic, FLAGS.target_pose_topic, FLAGS.control_mode_topic)
    ros_sub.start()
    ros_sub.wait_initial_pose_messages(float(FLAGS.wait_initial_pose_sec))

    env = env_config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=FLAGS.classifier)

    obs, info = env.reset()
    print_green("✅ 环境 reset 完成。")
    wait_for_vr_takeover_no_env_step(env, ros_sub, reason="initial_reset")

    raw_episodes = []
    current_episode = []
    returns = 0.0
    episode_step = 0
    success_count = 0

    pbar = tqdm(total=int(FLAGS.successes_needed), desc="成功收集的 Raw Demo 数量")

    try:
        while success_count < int(FLAGS.successes_needed):
            raw_actions = np.zeros(env.action_space.shape, dtype=np.float32)

            before_pose_snapshot = ros_sub.snapshot()
            step_t0 = time.time()
            next_obs, rew, done, truncated, info = env.step(raw_actions)
            wall_dt = time.time() - step_t0
            after_pose_snapshot = ros_sub.snapshot()

            if "intervene_action" in info:
                raw_actions = np.asarray(info["intervene_action"], dtype=np.float32)

            returns += float(rew)
            episode_step += 1

            forced_timeout = False
            if episode_step >= int(FLAGS.max_episode_steps) and not (done or truncated):
                forced_timeout = True
                truncated = True
                print_yellow(f"\n⏰ 达到最大录制时长：{FLAGS.max_episode_steps} 步，强制截断当前回合。")

            episode_end = bool(done or truncated)
            raw_step = make_raw_step(
                obs=obs,
                next_obs=next_obs,
                reward=rew,
                done=done,
                truncated=truncated,
                info=info,
                raw_actions=raw_actions,
                before_pose_snapshot=before_pose_snapshot,
                after_pose_snapshot=after_pose_snapshot,
                episode_step=episode_step,
                wall_dt=wall_dt,
            )
            raw_step["forced_timeout"] = bool(forced_timeout)

            if should_keep_raw_step_like_old_demo(raw_step, raw_actions, episode_end):
                current_episode.append(raw_step)

            pbar.set_description(
                f"Raw Demo {success_count}/{FLAGS.successes_needed} | "
                f"Return {returns:.2f} | kept {len(current_episode)} | step {episode_step}/{FLAGS.max_episode_steps}"
            )

            obs = next_obs

            if episode_end:
                print("\n🔄 回合结束。")
                print(f"   reward={rew}, done={done}, truncated={truncated}, forced_timeout={forced_timeout}")
                print(f"   info.succeed={info.get('succeed', None)}")
                print(f"   kept raw steps={len(current_episode)}, env episode_step={episode_step}")

                if FLAGS.classifier:
                    succeed = bool(info.get("succeed", False))
                    if succeed and FLAGS.manual_confirm_on_success:
                        print("📝 classifier 判定成功，请人工确认本回合是否真的成功。")
                        succeed = ask_success_from_terminal()
                else:
                    print("📝 当前 classifier=False，使用人工判定 success / fail。")
                    succeed = ask_success_from_terminal()

                if len(current_episode) > 0:
                    current_episode[-1]["infos"] = copy.deepcopy(info)
                    current_episode[-1]["infos"]["succeed"] = bool(succeed)

                if succeed and len(current_episode) > 0:
                    # 确保最终导出的 demo reward/done/mask 结构和之前一致。
                    current_episode[-1]["reward"] = 1.0
                    current_episode[-1]["done"] = True
                    current_episode[-1]["episode_end"] = True
                    current_episode[-1]["mask"] = 0.0

                    raw_episodes.append(copy.deepcopy(current_episode))
                    success_count += 1
                    pbar.update(1)
                    print_green(f"🎉 成功录制 1 条 Raw Demo！累计={success_count}, 长度={len(current_episode)}")
                else:
                    print_yellow("❌ 当前回合失败，或没有有效操作帧，已丢弃该 raw 轨迹。")

                current_episode = []
                returns = 0.0
                episode_step = 0

                if success_count >= int(FLAGS.successes_needed):
                    break

                print("🔄 正在复位机器人...")
                obs, info = env.reset()
                print_green("✅ reset 完成。")
                wait_for_vr_takeover_no_env_step(env, ros_sub, reason=f"after_episode_{success_count}")
                print("-" * 60)

    finally:
        pbar.close()
        ros_sub.stop()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_save_dir = FLAGS.raw_save_dir or os.path.join(EXAMPLES_DIR, "raw_abs_demo_data_single")
    ensure_dir(raw_save_dir)
    raw_file = os.path.join(raw_save_dir, f"{FLAGS.exp_name}_{len(raw_episodes)}_raw_abs_{uuid}.pkl")

    payload = {
        "format_version": "raw_abs_pose_demo_v2_wait_vr_no_env_step",
        "metadata": {
            "exp_name": FLAGS.exp_name,
            "successes_needed": int(FLAGS.successes_needed),
            "created_time": uuid,
            "feedback_pose_topic": FLAGS.feedback_pose_topic,
            "target_pose_topic": FLAGS.target_pose_topic,
            "control_mode_topic": FLAGS.control_mode_topic,
            "vr_control_mode_value": int(FLAGS.vr_control_mode_value),
            "classifier": bool(FLAGS.classifier),
            "max_episode_steps": int(FLAGS.max_episode_steps),
            "record_all_raw_steps": bool(FLAGS.record_all_raw_steps),
            "keep_pose_motion_without_intervention": bool(FLAGS.keep_pose_motion_without_intervention),
        },
        "episodes": raw_episodes,
    }

    with open(raw_file, "wb") as f:
        pkl.dump(payload, f)

    print_green(f"\n💾 Raw absolute demos saved: {raw_file}")
    print_green(f"📊 Raw episode count: {len(raw_episodes)}")
    print_green(f"📊 Raw transition count: {sum(len(ep) for ep in raw_episodes)}")
    return raw_file, payload

# =============================================================================
# 10. conversion
# =============================================================================
def load_raw_payload(path):
    with open(path, "rb") as f:
        payload = pkl.load(f)
    if isinstance(payload, dict) and "episodes" in payload:
        return payload
    if isinstance(payload, list):
        return {"format_version": "raw_abs_pose_demo_unknown", "metadata": {"source_path": path}, "episodes": payload}
    raise ValueError(f"无法识别 raw absolute demo 格式: {type(payload)}")


def raw_pose_for_source(raw_step, source, when):
    return raw_step.get("poses", {}).get(source, {}).get(when, None)


def is_converted_static(action, reward, done):
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != 7:
        return False
    if done or abs(float(reward)) > 1e-12:
        return False
    return bool(np.all(np.abs(a) <= float(FLAGS.static_action_eps)))


def convert_raw_episodes_to_transitions(raw_episodes, source, pos_scale, rot_scale):
    if source not in ("feedback", "target"):
        raise ValueError(f"source must be feedback/target, got {source}")

    transitions = []
    stats = Counter()
    action_list = []
    episode_lengths = []

    for ep_idx, episode in enumerate(raw_episodes):
        prev_stable_state = None
        ep_count = 0
        for raw_idx, raw_step in enumerate(episode):
            stats["raw_steps"] += 1

            pose_before = raw_pose_for_source(raw_step, source, "before")
            pose_after = raw_pose_for_source(raw_step, source, "after")
            action6, extra = normalized_action6_from_pose_delta(pose_before, pose_after, pos_scale, rot_scale)
            if action6 is None:
                stats["skipped_missing_pose"] += 1
                continue

            prev_feedback = raw_step.get("gripper_feedback_before", extract_gripper_feedback(raw_step.get("observations", {})))
            next_feedback = raw_step.get("gripper_feedback_after", extract_gripper_feedback(raw_step.get("next_observations", {})))
            gripper_event, prev_stable_state, stable_before = gripper_event_from_feedback(
                prev_feedback,
                next_feedback,
                prev_stable_state,
            )

            action = np.zeros(7, dtype=np.float32)
            action[:6] = action6
            action[6] = np.float32(gripper_event)

            reward = float(raw_step.get("reward", 0.0))
            done = bool(raw_step.get("episode_end", raw_step.get("done", False)))
            mask = 1.0 - float(done)

            if FLAGS.drop_static_converted and is_converted_static(action, reward, done):
                stats["dropped_static"] += 1
                continue

            gp = recompute_grasp_penalty_from_action(action)
            info = copy.deepcopy(raw_step.get("infos", {}))
            if not isinstance(info, dict):
                info = {"raw_info_repr": repr(info)}

            info["abs_pose_conversion"] = {
                "pose_source": source,
                "episode_index": ep_idx,
                "raw_step_index": raw_idx,
                "pos_scale": float(pos_scale),
                "rot_scale": float(rot_scale),
                "delta_pos_m": np.asarray(extra["delta_pos_m"], dtype=np.float64),
                "delta_rotvec_rad": np.asarray(extra["delta_rotvec_rad"], dtype=np.float64),
                "raw_normalized_action6": np.asarray(extra["raw_normalized_action6"], dtype=np.float64),
                "was_clipped": bool(extra["was_clipped"]),
                "pose_before": copy.deepcopy(pose_before),
                "pose_after": copy.deepcopy(pose_after),
                "gripper_feedback_before": prev_feedback,
                "gripper_feedback_after": next_feedback,
                "stable_gripper_before": stable_before,
                "stable_gripper_after": prev_stable_state,
                "gripper_event": float(gripper_event),
            }
            info["grasp_penalty"] = float(gp)
            info["grasp_penalty_source"] = f"abs_pose_conversion:{source}"

            transition = dict(
                # 按旧 demo 逻辑保存图像：只保存 env_config.image_keys + state。
                # 不按 ENV_IMAGE_KEYS 保存，避免 left_wrist_rgb 混入 RLPD policy demo。
                observations=prune_obs_for_converted_demo(raw_step["observations"]),
                actions=action.astype(np.float32),
                next_observations=prune_obs_for_converted_demo(raw_step["next_observations"]),
                rewards=float(reward),
                masks=float(mask),
                dones=bool(done),
                infos=info,
                grasp_penalty=float(gp),
            )
            transitions.append(transition)
            action_list.append(action.copy())
            ep_count += 1

            stats["converted_steps"] += 1
            stats["clip_count"] += int(extra["was_clipped"])
            stats["reward_pos"] += int(reward > 0)
            stats["done"] += int(done)
            stats["mask0"] += int(abs(mask) < 1e-12)
            if abs(gripper_event) > 0.5:
                stats["gripper_event"] += 1
                if gripper_event < 0:
                    stats["close"] += 1
                else:
                    stats["open"] += 1
            else:
                stats["hold"] += 1

        if ep_count:
            episode_lengths.append(ep_count)

    arr = np.stack(action_list, axis=0) if action_list else np.zeros((0, 7), dtype=np.float32)
    summary = {
        "source": source,
        "episodes": int(len(raw_episodes)),
        "raw_steps": int(stats["raw_steps"]),
        "converted_steps": int(stats["converted_steps"]),
        "skipped_missing_pose": int(stats["skipped_missing_pose"]),
        "dropped_static": int(stats["dropped_static"]),
        "clip_count": int(stats["clip_count"]),
        "reward_pos": int(stats["reward_pos"]),
        "done": int(stats["done"]),
        "mask0": int(stats["mask0"]),
        "gripper_dist": {
            "hold(0)": int(stats["hold"]),
            "close(-1)": int(stats["close"]),
            "open(+1)": int(stats["open"]),
        },
        "episode_lengths": episode_lengths,
        "action_min": arr.min(axis=0) if arr.size else None,
        "action_max": arr.max(axis=0) if arr.size else None,
        "action_mean": arr.mean(axis=0) if arr.size else None,
        "action_std": arr.std(axis=0) if arr.size else None,
    }
    return transitions, summary


def stats_of_values(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "max": None, "p95": None, "min": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
    }


def compute_feedback_target_delta_diagnostics(raw_episodes, pos_scale, rot_scale):
    pos_err_norms = []
    rot_err_norms = []
    norm_action_err_norms = []
    target_pos_norms = []
    feedback_pos_norms = []
    target_rot_norms = []
    feedback_rot_norms = []
    missing_count = 0
    total = 0
    examples = []

    for ep_idx, episode in enumerate(raw_episodes):
        for raw_idx, raw_step in enumerate(episode):
            total += 1
            fb_pos, fb_rot = pose_delta_to_components(
                raw_pose_for_source(raw_step, "feedback", "before"),
                raw_pose_for_source(raw_step, "feedback", "after"),
            )
            tg_pos, tg_rot = pose_delta_to_components(
                raw_pose_for_source(raw_step, "target", "before"),
                raw_pose_for_source(raw_step, "target", "after"),
            )
            if fb_pos is None or tg_pos is None:
                missing_count += 1
                continue
            pos_err = tg_pos - fb_pos
            rot_err = tg_rot - fb_rot
            norm_action_err = np.concatenate([pos_err / float(pos_scale), rot_err / float(rot_scale)])

            pos_err_norms.append(float(np.linalg.norm(pos_err)))
            rot_err_norms.append(float(np.linalg.norm(rot_err)))
            norm_action_err_norms.append(float(np.linalg.norm(norm_action_err)))
            target_pos_norms.append(float(np.linalg.norm(tg_pos)))
            feedback_pos_norms.append(float(np.linalg.norm(fb_pos)))
            target_rot_norms.append(float(np.linalg.norm(tg_rot)))
            feedback_rot_norms.append(float(np.linalg.norm(fb_rot)))

            if len(examples) < 20:
                examples.append({
                    "episode": ep_idx,
                    "raw_step": raw_idx,
                    "target_delta_pos_m": tg_pos,
                    "feedback_delta_pos_m": fb_pos,
                    "pos_error_m": pos_err,
                    "target_rotvec_rad": tg_rot,
                    "feedback_rotvec_rad": fb_rot,
                    "rot_error_rad": rot_err,
                    "normalized_action_error": norm_action_err,
                    "pos_error_norm_m": float(np.linalg.norm(pos_err)),
                    "rot_error_norm_rad": float(np.linalg.norm(rot_err)),
                })

    return {
        "total_raw_steps": int(total),
        "missing_pose_pairs": int(missing_count),
        "valid_pairs": int(total - missing_count),
        "pos_error_norm_m": stats_of_values(pos_err_norms),
        "rot_error_norm_rad": stats_of_values(rot_err_norms),
        "normalized_action_error_norm": stats_of_values(norm_action_err_norms),
        "target_delta_pos_norm_m": stats_of_values(target_pos_norms),
        "feedback_delta_pos_norm_m": stats_of_values(feedback_pos_norms),
        "target_delta_rot_norm_rad": stats_of_values(target_rot_norms),
        "feedback_delta_rot_norm_rad": stats_of_values(feedback_rot_norms),
        "examples": examples,
    }


def obs_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(obs_equal(a[k], b[k]) for k in a.keys())
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:
        return a == b


def compare_converted_non_action(feedback_transitions, target_transitions):
    n = min(len(feedback_transitions), len(target_transitions))
    out = Counter()
    out["feedback_count"] = len(feedback_transitions)
    out["target_count"] = len(target_transitions)
    out["min_count"] = n
    for i in range(n):
        f = feedback_transitions[i]
        t = target_transitions[i]
        out["obs_mismatch"] += int(not obs_equal(f.get("observations"), t.get("observations")))
        out["next_obs_mismatch"] += int(not obs_equal(f.get("next_observations"), t.get("next_observations")))
        out["reward_mismatch"] += int(abs(float(f.get("rewards", 0.0)) - float(t.get("rewards", 0.0))) > 1e-12)
        out["done_mismatch"] += int(bool(f.get("dones", False)) != bool(t.get("dones", False)))
        out["mask_mismatch"] += int(abs(float(f.get("masks", 1.0)) - float(t.get("masks", 1.0))) > 1e-12)
        fa = np.asarray(f.get("actions", []), dtype=np.float32).reshape(-1)
        ta = np.asarray(t.get("actions", []), dtype=np.float32).reshape(-1)
        out["gripper_action_mismatch"] += int(fa.size != 7 or ta.size != 7 or abs(float(fa[6]) - float(ta[6])) > 1e-6)
        out["grasp_penalty_mismatch"] += int(abs(float(f.get("grasp_penalty", 0.0)) - float(t.get("grasp_penalty", 0.0))) > 1e-12)
    return dict(out)


def print_conversion_summary(s):
    print_blue(f"\n===== Converted stats: {s['source']} =====")
    print(f"episodes              : {s['episodes']}")
    print(f"raw_steps             : {s['raw_steps']}")
    print(f"converted_steps       : {s['converted_steps']}")
    print(f"skipped_missing_pose  : {s['skipped_missing_pose']}")
    print(f"dropped_static        : {s['dropped_static']}")
    print(f"clip_count            : {s['clip_count']}")
    print(f"reward>0/done/mask0   : {s['reward_pos']} / {s['done']} / {s['mask0']}")
    print(f"gripper_dist          : {s['gripper_dist']}")
    if s["episode_lengths"]:
        arr = np.asarray(s["episode_lengths"])
        print(f"episode length        : min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
    if s["action_min"] is not None:
        print(f"action min            : {np.round(s['action_min'], 4).tolist()}")
        print(f"action max            : {np.round(s['action_max'], 4).tolist()}")
        print(f"action mean           : {np.round(s['action_mean'], 4).tolist()}")
        print(f"action std            : {np.round(s['action_std'], 4).tolist()}")


def resolve_sources_interactive():
    value = str(FLAGS.convert_source).strip().lower()
    aliases = {
        "1": "feedback",
        "feedback": "feedback",
        "motion_control": "feedback",
        "control": "feedback",
        "ee": "feedback",
        "2": "target",
        "target": "target",
        "motion_target": "target",
        "cmd": "target",
        "command": "target",
        "3": "both",
        "both": "both",
        "all": "both",
    }
    if value == "prompt":
        print("\n请选择用哪一种绝对位姿转换成训练 action：")
        print("  1 = feedback 版本：/motion_control/pose_ee_arm_right，机器人实际末端反馈")
        print("  2 = target   版本：/motion_target/target_pose_arm_right，控制目标/实际指令")
        print("  3 = both     版本：两个都导出，并计算二者增量误差")
        while True:
            ans = input("请输入 1 / 2 / 3: ").strip()
            if ans in aliases:
                value = aliases[ans]
                break
            print("❌ 输入无效，请输入 1、2 或 3。")
    else:
        value = aliases.get(value, value)

    if value == "feedback":
        return ["feedback"]
    if value == "target":
        return ["target"]
    if value == "both":
        return ["feedback", "target"]
    raise ValueError(f"Unknown convert_source={FLAGS.convert_source!r}; use prompt/feedback/target/both")


def save_converted_outputs(raw_payload, sources):
    pos_scale, rot_scale = resolve_scales()
    print_green("\n✅ 开始归一化 / 离线转换")
    print_green(f"   POS_SCALE = {pos_scale}")
    print_green(f"   ROT_SCALE = {rot_scale}")
    print_green(f"   convert sources = {sources}")

    raw_episodes = raw_payload["episodes"]
    converted_save_dir = FLAGS.converted_save_dir or os.path.join(EXAMPLES_DIR, "demo_data_single")
    report_save_dir = FLAGS.report_save_dir or converted_save_dir
    ensure_dir(converted_save_dir)
    ensure_dir(report_save_dir)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    converted_by_source = {}
    stats_by_source = {}
    output_files = {}

    for source in sources:
        transitions, stats = convert_raw_episodes_to_transitions(raw_episodes, source, pos_scale, rot_scale)
        converted_by_source[source] = transitions
        stats_by_source[source] = stats
        print_conversion_summary(stats)
        summarize_obs_keys_for_transitions(transitions, name=f"converted_{source}")

        file_name = os.path.join(
            converted_save_dir,
            f"{FLAGS.exp_name}_{stats['episodes']}_demos_abs2rel_{source}_{uuid}.pkl",
        )
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
        output_files[source] = file_name
        print_green(f"💾 Converted {source} demo saved: {file_name}")

    delta_diag = compute_feedback_target_delta_diagnostics(raw_episodes, pos_scale, rot_scale)
    compare_diag = None
    if "feedback" in converted_by_source and "target" in converted_by_source:
        compare_diag = compare_converted_non_action(converted_by_source["feedback"], converted_by_source["target"])
        print_blue("\n===== feedback vs target converted non-action compare =====")
        print(json.dumps(json_safe(compare_diag), indent=2, ensure_ascii=False))

    print_blue("\n===== feedback-target delta diagnostics =====")
    short_diag = {
        "total_raw_steps": delta_diag["total_raw_steps"],
        "missing_pose_pairs": delta_diag["missing_pose_pairs"],
        "valid_pairs": delta_diag["valid_pairs"],
        "pos_error_norm_m": delta_diag["pos_error_norm_m"],
        "rot_error_norm_rad": delta_diag["rot_error_norm_rad"],
        "normalized_action_error_norm": delta_diag["normalized_action_error_norm"],
        "target_delta_pos_norm_m": delta_diag["target_delta_pos_norm_m"],
        "feedback_delta_pos_norm_m": delta_diag["feedback_delta_pos_norm_m"],
        "target_delta_rot_norm_rad": delta_diag["target_delta_rot_norm_rad"],
        "feedback_delta_rot_norm_rad": delta_diag["feedback_delta_rot_norm_rad"],
    }
    print(json.dumps(json_safe(short_diag), indent=2, ensure_ascii=False))

    report = {
        "format_version": "abs_pose_conversion_report_v2_wait_vr_no_env_step",
        "pos_scale": float(pos_scale),
        "rot_scale": float(rot_scale),
        "sources": list(sources),
        "converted_files": output_files,
        "stats_by_source": stats_by_source,
        "feedback_target_delta_diagnostics": delta_diag,
        "converted_non_action_compare": compare_diag,
        "converted_demo_image_keys": get_policy_image_keys_for_demo(),
        "converted_demo_extra_obs_keys": get_extra_obs_keys_for_demo(),
        "raw_metadata": raw_payload.get("metadata", {}),
    }
    report_json = os.path.join(report_save_dir, f"{FLAGS.exp_name}_abs2rel_report_{uuid}.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False)
    print_green(f"📊 Conversion report saved: {report_json}")

    return output_files, report_json

# =============================================================================
# 11. main
# =============================================================================
def main(_):
    print_blue("=" * 100)
    print_blue("Absolute Pose Demo Recorder + Offline Relative Action Converter")
    print_blue("=" * 100)

    sources = resolve_sources_interactive()

    if FLAGS.raw_input_path:
        raw_file = FLAGS.raw_input_path
        print_green(f"📂 使用已有 raw absolute demo: {raw_file}")
        raw_payload = load_raw_payload(raw_file)
    else:
        raw_file, raw_payload = record_raw_absolute_demos()

    output_files, report_json = save_converted_outputs(raw_payload, sources)

    print_green("\n✅ 全部完成")
    print_green(f"Raw file     : {raw_file}")
    for source, path in output_files.items():
        print_green(f"Demo {source:8s}: {path}")
    print_green(f"Report       : {report_json}")

    print("\n下一步建议：")
    print("1. 用 inspect_demo_pkl_all.py 检查导出的 demo pkl：")
    print("   python inspect/inspect_demo_pkl_all.py --path <Demo pkl> --image_keys head_rgb right_wrist_rgb")
    print("2. 如果 feedback 和 target 的 delta 误差较大，优先用 target 版本训练；")
    print("   feedback 版本更适合诊断机器人实际执行是否跟上。")
    print("3. 后续改 POS_SCALE / ROT_SCALE 时，不用重新录 raw，只需要：")
    print("   python record_abs_pose_demos_and_convert_wait_vr.py --raw_input_path <raw pkl> --convert_source target")


if __name__ == "__main__":
    app.run(main)
