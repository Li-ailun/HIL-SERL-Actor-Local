#!/usr/bin/env bash
set -eo pipefail

cd /home/eren/HIL-SERL/HIL-SERL-Project/examples

source /home/eren/miniconda3/etc/profile.d/conda.sh
conda activate hilserl_actor_gpu_py310

export PYTHONNOUSERSITE=1

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
  fi
}

set -u
safe_source /opt/ros/humble/setup.bash
safe_source /home/eren/camera/realsense_ws/install/setup.bash
safe_source /home/eren/camera/zed2_ws/install/setup.bash
safe_source /home/eren/VR_APP/ros2_ws/install/setup.bash
safe_source /home/eren/HIL-SERL/HIL-SERL-Project/serl_robot_infra/Galaxea_env/VR/ros2_ws/install/setup.bash

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

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS

python - <<'PY'
import sys
print("python =", sys.executable)
import rclpy
print("rclpy ok =", rclpy.__file__)
PY

exec python record_demos.py