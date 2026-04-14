#函数解释
# 姿态表示法”的转换工具，专门用来在“四元数 (Quaternion)”和“欧拉角 (Euler Angles)”之间来回翻译。
# quat_2_euler(quat) (四元数转欧拉角)
# euler_2_quat(xyz) (欧拉角转四元数) —— ⚠️ 危险警告！（星海图机器人和frank机械臂的定义有差别！！！）

#代码作用
# rotations.py (基础姿态计算)
# 作用：只负责最基本的“四元数”和“欧拉角”的相互转换，以及简单的旋转增量叠加。
# 谁在用它？：底层的 dual_galaxea_env.py。因为底层环境只需要知道“把当前手腕的角度加上手柄拨动的角度”。


from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    """四元数 [x, y, z, w] -> 欧拉角 xyz"""
    quat = np.asarray(quat, dtype=np.float64)
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    """
    欧拉角 (x, y, z) -> 四元数 (x, y, z, w)
    完美适配 ROS 2 的 PoseStamped 格式
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    return R.from_euler("xyz", xyz).as_quat()


def apply_delta_rotation(current_quat, rot_delta_vec):
    """
    给当前姿态应用一个旋转增量

    current_quat: 当前四元数 [x, y, z, w]
    rot_delta_vec: 旋转向量增量 [rx, ry, rz]
    返回: 新的四元数 [x, y, z, w]
    """
    current_quat = np.asarray(current_quat, dtype=np.float64)
    rot_delta_vec = np.asarray(rot_delta_vec, dtype=np.float64)

    delta_rot = R.from_rotvec(rot_delta_vec)
    curr_rot = R.from_quat(current_quat)
    return (delta_rot * curr_rot).as_quat()


def clip_rotation(quat, rpy_low, rpy_high):
    """
    把四元数旋转限制在指定的 RPY 范围内

    quat: [x, y, z, w]
    rpy_low: [roll_low, pitch_low, yaw_low]
    rpy_high: [roll_high, pitch_high, yaw_high]
    """
    quat = np.asarray(quat, dtype=np.float64).copy()
    rpy_low = np.asarray(rpy_low, dtype=np.float64)
    rpy_high = np.asarray(rpy_high, dtype=np.float64)

    # 防御性处理：四元数归一化，避免数值漂移
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        # 极端情况下给一个单位四元数
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = quat / norm

    # 这里统一使用 R，不再用未定义的 Rotation
    euler = R.from_quat(quat).as_euler("xyz")

    # 保留你原来的第一个角特殊处理逻辑
    sign = np.sign(euler[0]) if abs(euler[0]) > 1e-8 else 1.0
    euler[0] = sign * np.clip(np.abs(euler[0]), rpy_low[0], rpy_high[0])

    # 限制其余两个角
    euler[1:] = np.clip(euler[1:], rpy_low[1:], rpy_high[1:])

    return R.from_euler("xyz", euler).as_quat()