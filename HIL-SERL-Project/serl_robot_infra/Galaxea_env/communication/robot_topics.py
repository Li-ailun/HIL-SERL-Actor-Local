from typing import Dict
from dataclasses import dataclass, field
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped

@dataclass(frozen=True)
class Topic:
    channel: str
    msg_type: type  # 修正类型标注以便更好地兼容

@dataclass
class RobotTopicsConfig:
    # 状态反馈话题 (Subscribers)
    state: Dict[str, Topic] = field(
        default_factory=lambda: {
            "left_arm": Topic("/hdas/feedback_arm_left", JointState),
            "right_arm": Topic("/hdas/feedback_arm_right", JointState),
            # "torso": Topic("/hdas/feedback_torso", JointState),
            # "chassis": Topic("/hdas/feedback_chassis", JointState),
            "left_ee_pose": Topic("/motion_control/pose_ee_arm_left", PoseStamped),
            "right_ee_pose": Topic("/motion_control/pose_ee_arm_right", PoseStamped),
            "left_gripper": Topic("/hdas/feedback_gripper_left", JointState),
            "right_gripper": Topic("/hdas/feedback_gripper_right", JointState),
        }
    )

    # 图像话题 (Subscribers - 基于你的实测列表)
    images: Dict[str, Topic] = field(
        default_factory=lambda: {
            # # 头部 ZED 相机彩色压缩图像
            # "head_rgb": Topic("/zed/zed_node/rgb/color/rect/image/compressed", CompressedImage),
            # # 左腕 RealSense 压缩图像 (校对自你的 topic list)
            # "left_wrist_rgb": Topic("/camera/camera_wrist_left/color/image_rect_raw/compressed", CompressedImage),
            # # 右腕 RealSense 压缩图像 (校对自你的 topic list)
            # "right_wrist_rgb": Topic("/camera/camera_wrist_right/color/image_rect_raw/compressed", CompressedImage),
        }
    )

    # 动作指令话题 (Publishers)
    action: Dict[str, Topic] = field(
        default_factory=lambda: {
            "left_arm": Topic("/motion_target/target_joint_state_arm_left", JointState),
            "right_arm": Topic("/motion_target/target_joint_state_arm_right", JointState),
            "torso": Topic("/motion_target/target_joint_state_torso", JointState),
            "left_ee_pose": Topic("/motion_target/target_pose_arm_left", PoseStamped),
            "right_ee_pose": Topic("/motion_target/target_pose_arm_right", PoseStamped),
            "left_gripper": Topic("/motion_target/target_position_gripper_left", JointState),
            "right_gripper": Topic("/motion_target/target_position_gripper_right", JointState),
        }
    )

    # QoS 配置：针对实时具身智能任务优化
    qos: Dict[str, QoSProfile] = field(
        default_factory=lambda: {
            "sub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT, # 传感器数据丢包不补发，保证实时性
                history=HistoryPolicy.KEEP_LAST,
                depth=1, # 只要最新的一帧
                durability=DurabilityPolicy.VOLATILE
            ),
            "pub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            ),
        }
    )

    camera_deque_length: int = 3  # 15Hz 视觉反馈，保留近 3 帧
    state_deque_length: int = 80  # 高频关节反馈，保留 0.2s 内的数据


