#!/usr/bin/env bash
set -eo pipefail
#chmod +x /home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_success_fail_ros.sh
#后
#/home/eren/HIL-SERL/HIL-SERL-Project/examples/sh/run_record_success_fail_ros.sh


cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

# 1) conda 环境
source /home/eren/miniconda3/etc/profile.d/conda.sh
conda activate hilserl_actor_gpu_py310

# 2) 保持 Python 干净
export PYTHONNOUSERSITE=1

# 3) 清掉旧 CUDA toolkit 路径，避免和 JAX pip CUDA12 轮子打架
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

# 4) 安全 source：避免 ROS setup.bash 在 nounset 下炸掉
safe_source() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set +u
    source "$f"
    set -u
  fi
}

# 5) 加载 ROS2 基础环境和 overlay
set -u
safe_source /opt/ros/humble/setup.bash
safe_source /home/eren/camera/realsense_ws/install/setup.bash
safe_source /home/eren/camera/zed2_ws/install/setup.bash
safe_source /home/eren/VR_APP/ros2_ws/install/setup.bash
safe_source /home/eren/HIL-SERL/HIL-SERL-Project/serl_robot_infra/Galaxea_env/VR/ros2_ws/install/setup.bash

# 6) 再次去掉 ROS source 可能重新带回来的 CUDA 11.8 路径
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

# 7) JAX / 运行设置
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS

# 8) 缺 pynput 就自动安装
python - <<'PY'
import importlib.util
missing = []
for m in ["pynput", "rclpy"]:
    if importlib.util.find_spec(m) is None:
        missing.append(m)
print("缺失模块:", missing)
raise SystemExit(0 if "pynput" not in missing else 1)
PY

if [[ $? -ne 0 ]]; then
  echo "正在安装 pynput ..."
  python -m pip install pynput
fi

# 9) 预检查
python - <<'PY'
import sys
print("python =", sys.executable)

import rclpy
print("rclpy ok =", rclpy.__file__)

import pynput
print("pynput ok =", pynput.__file__)
PY

# 10) 启动 success/fail 录制
exec python record_success_fail.py "$@"