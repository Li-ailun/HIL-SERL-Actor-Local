# chmod +x /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_demos_ros.sh
#后
# /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_demos_ros.sh
#!/usr/bin/env bash
#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# HIL-SERL absolute-pose demo recorder launcher
#
# chmod +x /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_abs_pose_demos_ros.sh
# /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_abs_pose_demos_ros.sh
#
# 如果你的 target topic 真实名字是 /motion_target/target_pose_arm_rightht：
# TARGET_POSE_TOPIC=/motion_target/target_pose_arm_rightht \
# /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_abs_pose_demos_ros.sh
# ==============================================================================

cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

source /home/eren/miniconda3/etc/profile.d/conda.sh
conda activate hilserl_actor_gpu_py310

export PYTHONNOUSERSITE=1

# ==============================================================================
# 清理 CUDA 11.8 lib64，避免和 conda / ROS / JAX 动态库冲突
# ==============================================================================
OLD_LD="${LD_LIBRARY_PATH-}"
CLEAN_LD=""
IFS=':' read -ra PARTS <<< "${OLD_LD}"
for p in "${PARTS[@]}"; do
  [[ -z "${p}" ]] && continue
  [[ "${p}" == "/usr/local/cuda-11.8/lib64" ]] && continue

  if [[ -z "${CLEAN_LD}" ]]; then
    CLEAN_LD="${p}"
  else
    case ":${CLEAN_LD}:" in
      *":${p}:"*) ;;
      *) CLEAN_LD="${CLEAN_LD}:${p}" ;;
    esac
  fi
done
export LD_LIBRARY_PATH="${CLEAN_LD}"

safe_source() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set +u
    source "$f"
    set -u
  else
    echo "⚠️ setup file not found, skip: $f"
  fi
}

set -u

# ==============================================================================
# ROS2 / camera / VR workspaces
# ==============================================================================
safe_source /opt/ros/humble/setup.bash
safe_source /home/eren/camera/realsense_ws/install/setup.bash
safe_source /home/eren/camera/zed2_ws/install/setup.bash
safe_source /home/eren/VR_APP/ros2_ws/install/setup.bash
safe_source /home/eren/HIL-SERL/HIL-SERL-Project/serl_robot_infra/Galaxea_env/VR/ros2_ws/install/setup.bash

# ==============================================================================
# 再清一次 LD_LIBRARY_PATH，防止 source 后又带回 CUDA lib64
# ==============================================================================
OLD_LD="${LD_LIBRARY_PATH-}"
CLEAN_LD=""
IFS=':' read -ra PARTS <<< "${OLD_LD}"
for p in "${PARTS[@]}"; do
  [[ -z "${p}" ]] && continue
  [[ "${p}" == "/usr/local/cuda-11.8/lib64" ]] && continue

  if [[ -z "${CLEAN_LD}" ]]; then
    CLEAN_LD="${p}"
  else
    case ":${CLEAN_LD}:" in
      *":${p}:"*) ;;
      *) CLEAN_LD="${CLEAN_LD}:${p}" ;;
    esac
  fi
done
export LD_LIBRARY_PATH="${CLEAN_LD}"

# ==============================================================================
# 录 demo 建议本地 JAX / classifier 走 CPU，避免抢 GPU 或 JAX 初始化炸
# 新脚本内部也会设置 CPU，这里再显式设置一次更安全
# ==============================================================================
unset JAX_PLATFORMS
export CUDA_VISIBLE_DEVICES="0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ==============================================================================
# 可配置参数
# ==============================================================================
SCRIPT_NAME="${SCRIPT_NAME:-record_demos.py}"

EXP_NAME="${EXP_NAME:-galaxea_usb_insertion_single}"
SUCCESSES_NEEDED="${SUCCESSES_NEEDED:-30}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-400}"

CLASSIFIER="${CLASSIFIER:-True}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
MANUAL_CONFIRM_ON_SUCCESS="${MANUAL_CONFIRM_ON_SUCCESS:-False}"

# prompt: 录完后输入 1/2/3
# feedback: 直接导出 /motion_control/pose_ee_arm_right 版本
# target:   直接导出 /motion_target/target_pose_arm_right 版本
# both:     两个都导出并计算误差
CONVERT_SOURCE="${CONVERT_SOURCE:-prompt}"

FEEDBACK_POSE_TOPIC="${FEEDBACK_POSE_TOPIC:-/motion_control/pose_ee_arm_right}"
TARGET_POSE_TOPIC="${TARGET_POSE_TOPIC:-/motion_target/target_pose_arm_right}"

WAIT_INITIAL_POSE_SEC="${WAIT_INITIAL_POSE_SEC:-2.0}"

# 默认 0 表示从 config 读取 POS_SCALE / ROT_SCALE
POS_SCALE="${POS_SCALE:-0.0}"
ROT_SCALE="${ROT_SCALE:-0.0}"

# 是否保留所有 raw step。
# False 更接近旧 demo 脚本：不保存无效静止帧。
RECORD_ALL_RAW_STEPS="${RECORD_ALL_RAW_STEPS:-False}"

# 是否在转换后的 pkl 里丢弃 zero-action transition。
# 默认 False，避免误删成功帧或夹爪状态相关帧。
DROP_STATIC_CONVERTED="${DROP_STATIC_CONVERTED:-False}"

# ==============================================================================
# 基础检查
# ==============================================================================
echo "================================================================================"
echo "Python / ROS environment check"
echo "================================================================================"

python - <<'PY'
import sys
print("python =", sys.executable)
import rclpy
print("rclpy ok =", rclpy.__file__)
PY

echo ""
echo "================================================================================"
echo "Topic check"
echo "================================================================================"
echo "feedback topic: ${FEEDBACK_POSE_TOPIC}"
echo "target topic  : ${TARGET_POSE_TOPIC}"
echo ""

if ros2 topic info "${FEEDBACK_POSE_TOPIC}" >/tmp/feedback_pose_topic_info.txt 2>&1; then
  cat /tmp/feedback_pose_topic_info.txt
else
  echo "⚠️ cannot get topic info: ${FEEDBACK_POSE_TOPIC}"
fi

echo ""

if ros2 topic info "${TARGET_POSE_TOPIC}" >/tmp/target_pose_topic_info.txt 2>&1; then
  cat /tmp/target_pose_topic_info.txt
else
  echo "⚠️ cannot get topic info: ${TARGET_POSE_TOPIC}"
fi

echo ""
echo "Trying one-shot echo, timeout 3s..."

if timeout 3s ros2 topic echo "${FEEDBACK_POSE_TOPIC}" --once >/tmp/feedback_pose_once.txt 2>&1; then
  echo "✅ feedback pose echo OK"
else
  echo "⚠️ feedback pose echo timeout or failed: ${FEEDBACK_POSE_TOPIC}"
fi

if timeout 3s ros2 topic echo "${TARGET_POSE_TOPIC}" --once >/tmp/target_pose_once.txt 2>&1; then
  echo "✅ target pose echo OK"
else
  echo "⚠️ target pose echo timeout or failed: ${TARGET_POSE_TOPIC}"
fi

echo ""
echo "================================================================================"
echo "Run recorder"
echo "================================================================================"
echo "script                  : ${SCRIPT_NAME}"
echo "exp_name                : ${EXP_NAME}"
echo "successes_needed        : ${SUCCESSES_NEEDED}"
echo "max_episode_steps       : ${MAX_EPISODE_STEPS}"
echo "classifier              : ${CLASSIFIER}"
echo "convert_source          : ${CONVERT_SOURCE}"
echo "feedback_pose_topic     : ${FEEDBACK_POSE_TOPIC}"
echo "target_pose_topic       : ${TARGET_POSE_TOPIC}"
echo "pos_scale               : ${POS_SCALE}  # 0 means read config"
echo "rot_scale               : ${ROT_SCALE}  # 0 means read config"
echo "record_all_raw_steps    : ${RECORD_ALL_RAW_STEPS}"
echo "drop_static_converted   : ${DROP_STATIC_CONVERTED}"
echo "================================================================================"
echo ""

exec python "${SCRIPT_NAME}" \
  --exp_name="${EXP_NAME}" \
  --successes_needed="${SUCCESSES_NEEDED}" \
  --max_episode_steps="${MAX_EPISODE_STEPS}" \
  --classifier="${CLASSIFIER}" \
  --save_video="${SAVE_VIDEO}" \
  --manual_confirm_on_success="${MANUAL_CONFIRM_ON_SUCCESS}" \
  --convert_source="${CONVERT_SOURCE}" \
  --feedback_pose_topic="${FEEDBACK_POSE_TOPIC}" \
  --target_pose_topic="${TARGET_POSE_TOPIC}" \
  --wait_initial_pose_sec="${WAIT_INITIAL_POSE_SEC}" \
  --pos_scale="${POS_SCALE}" \
  --rot_scale="${ROT_SCALE}" \
  --record_all_raw_steps="${RECORD_ALL_RAW_STEPS}" \
  --drop_static_converted="${DROP_STATIC_CONVERTED}"


# set -eo pipefail

# cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

# source /home/eren/miniconda3/etc/profile.d/conda.sh
# conda activate hilserl_actor_gpu_py310

# export PYTHONNOUSERSITE=1

# OLD_LD="${LD_LIBRARY_PATH-}"
# CLEAN_LD=""
# IFS=':' read -ra PARTS <<< "${OLD_LD}"
# for p in "${PARTS[@]}"; do
#   [[ -z "${p}" ]] && continue
#   [[ "${p}" == "/usr/local/cuda-11.8/lib64" ]] && continue
#   if [[ -z "${CLEAN_LD}" ]]; then
#     CLEAN_LD="${p}"
#   else
#     case ":${CLEAN_LD}:" in
#       *":${p}:"*) ;;
#       *) CLEAN_LD="${CLEAN_LD}:${p}" ;;
#     esac
#   fi
# done
# export LD_LIBRARY_PATH="${CLEAN_LD}"

# safe_source() {
#   local f="$1"
#   if [[ -f "$f" ]]; then
#     set +u
#     source "$f"
#     set -u
#   fi
# }

# set -u
# safe_source /opt/ros/humble/setup.bash
# safe_source /home/eren/camera/realsense_ws/install/setup.bash
# safe_source /home/eren/camera/zed2_ws/install/setup.bash
# safe_source /home/eren/VR_APP/ros2_ws/install/setup.bash
# safe_source /home/eren/HIL-SERL/HIL-SERL-Project/serl_robot_infra/Galaxea_env/VR/ros2_ws/install/setup.bash

# OLD_LD="${LD_LIBRARY_PATH-}"
# CLEAN_LD=""
# IFS=':' read -ra PARTS <<< "${OLD_LD}"
# for p in "${PARTS[@]}"; do
#   [[ -z "${p}" ]] && continue
#   [[ "${p}" == "/usr/local/cuda-11.8/lib64" ]] && continue
#   if [[ -z "${CLEAN_LD}" ]]; then
#     CLEAN_LD="${p}"
#   else
#     case ":${CLEAN_LD}:" in
#       *":${p}:"*) ;;
#       *) CLEAN_LD="${CLEAN_LD}:${p}" ;;
#     esac
#   fi
# done
# export LD_LIBRARY_PATH="${CLEAN_LD}"

# export CUDA_VISIBLE_DEVICES=0
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# unset JAX_PLATFORMS

# python - <<'PY'
# import sys
# print("python =", sys.executable)
# import rclpy
# print("rclpy ok =", rclpy.__file__)
# PY

# exec python record_demos.py