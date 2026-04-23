# 直接跑全套：

# python test_action_scale_right_arm.py --switch_to_script

# 只测旋转：

# python test_action_scale_right_arm.py --switch_to_script --tests rx,ry,rz

# 打印每一步详情：

# python test_action_scale_right_arm.py --switch_to_script --tests rx,ry,rz --print_each_step

         ###########################################
        #该脚本使用的reset，不同测试，reset不同，不然会撞击场景
          # default="0.2,-0.3,-0.25,0.0,0.0,0.0,1.0",
                 #######################
   ##############或者直接在空旷区域进行#############################

                   #####################
         ##############################################


#!/usr/bin/env python3
import argparse
import time
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from teleoperation_msg_ros2.srv import SwitchControlModeVR


def parse_csv_floats(text: str, expected_len: int):
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"期望 {expected_len} 个数，实际收到 {len(vals)} 个：{text}")
    return np.array(vals, dtype=np.float64)


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def clip_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    return np.clip(action, -1.0, 1.0)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_to_euler_xyz(q_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(normalize_quat_xyzw(q_xyzw)).as_euler("xyz", degrees=False)


def apply_delta_rotation_xyzw(curr_quat_xyzw: np.ndarray, delta_rpy_xyz: np.ndarray) -> np.ndarray:
    curr = R.from_quat(normalize_quat_xyzw(curr_quat_xyzw))
    delta = R.from_euler("xyz", np.asarray(delta_rpy_xyz, dtype=np.float64), degrees=False)
    nxt = curr * delta
    return normalize_quat_xyzw(nxt.as_quat())


def relative_rpy_xyz(start_q_xyzw: np.ndarray, end_q_xyzw: np.ndarray) -> np.ndarray:
    r0 = R.from_quat(normalize_quat_xyzw(start_q_xyzw))
    r1 = R.from_quat(normalize_quat_xyzw(end_q_xyzw))
    rel = r0.inv() * r1
    return wrap_to_pi(rel.as_euler("xyz", degrees=False))


def apply_action_increment(
    pose7_xyzw: np.ndarray,
    action7: np.ndarray,
    pos_scale: float,
    rot_scale: float,
) -> np.ndarray:
    pose7_xyzw = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    action7 = clip_action(np.asarray(action7, dtype=np.float64).reshape(7))

    next_pose = pose7_xyzw.copy()
    next_pose[:3] = next_pose[:3] + action7[:3] * pos_scale
    next_pose[3:7] = apply_delta_rotation_xyzw(next_pose[3:7], action7[3:6] * rot_scale)
    return next_pose


def fmt_pose(pose7):
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    pos = np.round(pose7[:3], 6).tolist()
    quat = np.round(pose7[3:7], 6).tolist()
    euler = np.round(quat_to_euler_xyz(pose7[3:7]), 6).tolist()
    return f"pos={pos}, quat_xyzw={quat}, euler_xyz={euler}"


class ActionScaleTester(Node):
    def __init__(self, rate_hz: float):
        super().__init__("action_scale_tester_right_arm_full")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(
            PoseStamped,
            "/motion_target/target_pose_arm_right",
            qos,
        )
        self.sub = self.create_subscription(
            PoseStamped,
            "/motion_control/pose_ee_arm_right",
            self._pose_cb,
            qos,
        )
        self.cli = self.create_client(
            SwitchControlModeVR,
            "/switch_control_mode_vr",
        )

        self.latest_feedback_pose7: Optional[np.ndarray] = None
        self.latest_feedback_stamp = None
        self.rate_hz = float(rate_hz)

    def _pose_cb(self, msg: PoseStamped):
        self.latest_feedback_pose7 = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        self.latest_feedback_stamp = msg.header.stamp

    def wait_for_feedback(self, timeout_sec: float = 5.0):
        t0 = time.time()
        while rclpy.ok() and self.latest_feedback_pose7 is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - t0 > timeout_sec:
                raise TimeoutError("等待 /motion_control/pose_ee_arm_right 超时")
        return self.latest_feedback_pose7.copy()

    def switch_control_mode(self, use_vr_mode: bool, timeout_sec: float = 5.0):
        self.get_logger().info(f"请求切换 /switch_control_mode_vr: use_vr_mode={use_vr_mode}")
        if not self.cli.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/switch_control_mode_vr 服务未上线")

        req = SwitchControlModeVR.Request()
        req.use_vr_mode = bool(use_vr_mode)
        future = self.cli.call_async(req)

        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if future.done():
                resp = future.result()
                self.get_logger().info(
                    f"服务返回: success={getattr(resp, 'success', None)}, "
                    f"message={getattr(resp, 'message', '')}"
                )
                return resp
            if time.time() - t0 > timeout_sec:
                raise TimeoutError("调用 /switch_control_mode_vr 超时")

    def publish_pose_once(self, pose7_xyzw: np.ndarray):
        pose7_xyzw = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pose7_xyzw[0])
        msg.pose.position.y = float(pose7_xyzw[1])
        msg.pose.position.z = float(pose7_xyzw[2])
        msg.pose.orientation.x = float(pose7_xyzw[3])
        msg.pose.orientation.y = float(pose7_xyzw[4])
        msg.pose.orientation.z = float(pose7_xyzw[5])
        msg.pose.orientation.w = float(pose7_xyzw[6])
        self.pub.publish(msg)

    def publish_pose_for_duration(self, pose7_xyzw: np.ndarray, duration_sec: float):
        period = 1.0 / self.rate_hz
        t0 = time.time()
        count = 0
        while time.time() - t0 < duration_sec and rclpy.ok():
            loop_t = time.time()
            self.publish_pose_once(pose7_xyzw)
            rclpy.spin_once(self, timeout_sec=0.0)
            count += 1
            dt = time.time() - loop_t
            time.sleep(max(0.0, period - dt))
        return count


def build_case_list(items: List[str]) -> List[Tuple[str, np.ndarray]]:
    cases = []
    for item in items:
        key = item.lower().strip()
        if key == "x":
            cases.append(("+x", np.array([+1, 0, 0, 0, 0, 0, 0], dtype=np.float64)))
            cases.append(("-x", np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.float64)))
        elif key == "y":
            cases.append(("+y", np.array([0, +1, 0, 0, 0, 0, 0], dtype=np.float64)))
            cases.append(("-y", np.array([0, -1, 0, 0, 0, 0, 0], dtype=np.float64)))
        elif key == "z":
            cases.append(("+z", np.array([0, 0, +1, 0, 0, 0, 0], dtype=np.float64)))
            cases.append(("-z", np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.float64)))
        elif key == "rx":
            cases.append(("+rx", np.array([0, 0, 0, +1, 0, 0, 0], dtype=np.float64)))
            cases.append(("-rx", np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.float64)))
        elif key == "ry":
            cases.append(("+ry", np.array([0, 0, 0, 0, +1, 0, 0], dtype=np.float64)))
            cases.append(("-ry", np.array([0, 0, 0, 0, -1, 0, 0], dtype=np.float64)))
        elif key == "rz":
            cases.append(("+rz", np.array([0, 0, 0, 0, 0, +1, 0], dtype=np.float64)))
            cases.append(("-rz", np.array([0, 0, 0, 0, 0, -1, 0], dtype=np.float64)))
        else:
            raise ValueError(f"不支持的测试项: {item}，只支持 x/y/z/rx/ry/rz")
    return cases


def main():
    parser = argparse.ArgumentParser(description="右臂平移+旋转动作缩放测试脚本")
    parser.add_argument(
        "--reset_pose",
        type=str,
        default="0.2,-0.3,-0.25,0.0,0.0,0.0,1.0",
        help="reset pose: x,y,z,qx,qy,qz,qw",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="x,y,z,rx,ry,rz",
        help="要测试的项，逗号分隔，例如 x,y,z,rx,ry,rz 或 x,rz",
    )
    parser.add_argument("--steps", type=int, default=10, help="每组重复多少步")
    parser.add_argument("--publish_rate", type=float, default=30.0, help="发布频率 Hz")
    parser.add_argument("--reset_hold_sec", type=float, default=2.0, help="每组测试前 reset 位姿持续发布时间")
    parser.add_argument("--settle_sec", type=float, default=2.2, help="reset 后额外稳定等待时间")
    parser.add_argument("--step_hold_sec", type=float, default=0.35, help="每一步目标持续发布时间")
    parser.add_argument("--pos_scale", type=float, default=0.01, help="位置缩放")
    parser.add_argument("--rot_scale", type=float, default=0.05, help="旋转缩放（rad/step）")
    parser.add_argument("--switch_to_script", action="store_true", help="启动时切到 use_vr_mode=False")
    parser.add_argument("--switch_back_to_vr", action="store_true", help="结束时切回 use_vr_mode=True")
    parser.add_argument("--print_each_step", action="store_true", help="打印每一步详情")
    args = parser.parse_args()

    reset_pose = parse_csv_floats(args.reset_pose, 7)
    tests = [x.strip() for x in args.tests.split(",") if x.strip()]
    cases = build_case_list(tests)

    print("========== 平移 + 旋转动作缩放验证 ==========")
    print(f"reset_pose      : {fmt_pose(reset_pose)}")
    print(f"tests           : {tests}")
    print(f"steps           : {args.steps}")
    print(f"publish_rate    : {args.publish_rate} Hz")
    print(f"reset_hold_sec  : {args.reset_hold_sec}")
    print(f"settle_sec      : {args.settle_sec}")
    print(f"step_hold_sec   : {args.step_hold_sec}")
    print(f"pos_scale       : {args.pos_scale}")
    print(f"rot_scale       : {args.rot_scale}")
    print("测试顺序        : " + ", ".join([name for name, _ in cases]))
    print("============================================\n")

    rclpy.init()
    node = ActionScaleTester(rate_hz=args.publish_rate)

    summaries = []

    try:
        print("等待反馈位姿...")
        fb0 = node.wait_for_feedback(timeout_sec=5.0)
        print(f"[初始反馈] {fmt_pose(fb0)}\n")

        if args.switch_to_script:
            node.switch_control_mode(use_vr_mode=False)
            time.sleep(0.3)

        for case_name, action in cases:
            clipped_action = clip_action(action)
            expected_delta_xyz_per_step = clipped_action[:3] * args.pos_scale
            expected_total_xyz = expected_delta_xyz_per_step * args.steps

            expected_delta_rpy_per_step = clipped_action[3:6] * args.rot_scale
            expected_total_rpy = expected_delta_rpy_per_step * args.steps
            expected_total_rpy = wrap_to_pi(expected_total_rpy)

            print(f"\n================ {case_name} 测试开始 ================")
            print(f"raw_action            : {action.tolist()}")
            print(f"clipped_action        : {clipped_action.tolist()}")
            print(f"expected Δxyz/step    : {np.round(expected_delta_xyz_per_step, 6).tolist()}")
            print(f"expected total Δxyz   : {np.round(expected_total_xyz, 6).tolist()}")
            print(f"expected Δrpy/step    : {np.round(expected_delta_rpy_per_step, 6).tolist()}")
            print(f"expected total Δrpy   : {np.round(expected_total_rpy, 6).tolist()}")

            print(f"先持续发布 reset 位姿 {args.reset_hold_sec:.2f}s ...")
            cnt = node.publish_pose_for_duration(reset_pose, args.reset_hold_sec)
            rclpy.spin_once(node, timeout_sec=0.1)
            fb_reset = node.wait_for_feedback(timeout_sec=2.0)
            print(f"[reset后反馈] publish_count={cnt}, {fmt_pose(fb_reset)}")

            print(f"等待 {args.settle_sec:.2f}s 让反馈稳定...")
            time.sleep(args.settle_sec)
            rclpy.spin_once(node, timeout_sec=0.1)
            fb_start = node.wait_for_feedback(timeout_sec=2.0)
            print(f"[正式起点反馈] {fmt_pose(fb_start)}")

            target_pose = reset_pose.copy()
            for i in range(1, args.steps + 1):
                target_pose = apply_action_increment(
                    pose7_xyzw=target_pose,
                    action7=action,
                    pos_scale=args.pos_scale,
                    rot_scale=args.rot_scale,
                )
                cnt = node.publish_pose_for_duration(target_pose, args.step_hold_sec)
                rclpy.spin_once(node, timeout_sec=0.1)
                fb_now = node.wait_for_feedback(timeout_sec=2.0)

                if args.print_each_step:
                    delta_from_start_xyz = fb_now[:3] - fb_start[:3]
                    delta_from_start_rpy = relative_rpy_xyz(fb_start[3:7], fb_now[3:7])
                    print(f"[step {i:02d}]")
                    print(f"  target_pose        : {fmt_pose(target_pose)}")
                    print(f"  feedback_pose      : {fmt_pose(fb_now)}")
                    print(f"  feedback Δxyz累计  : {np.round(delta_from_start_xyz, 6).tolist()}")
                    print(f"  feedback Δrpy累计  : {np.round(delta_from_start_rpy, 6).tolist()}")
                    print(f"  publish_count      : {cnt}")

            time.sleep(0.2)
            rclpy.spin_once(node, timeout_sec=0.1)
            fb_final = node.wait_for_feedback(timeout_sec=2.0)

            actual_total_xyz = fb_final[:3] - fb_start[:3]
            error_xyz = actual_total_xyz - expected_total_xyz

            actual_total_rpy = relative_rpy_xyz(fb_start[3:7], fb_final[3:7])
            error_rpy = wrap_to_pi(actual_total_rpy - expected_total_rpy)

            print(f"[终点反馈] {fmt_pose(fb_final)}")
            print(f"actual total Δxyz     : {np.round(actual_total_xyz, 6).tolist()}")
            print(f"expected total Δxyz   : {np.round(expected_total_xyz, 6).tolist()}")
            print(f"error Δxyz            : {np.round(error_xyz, 6).tolist()}")
            print(f"actual total Δrpy     : {np.round(actual_total_rpy, 6).tolist()}")
            print(f"expected total Δrpy   : {np.round(expected_total_rpy, 6).tolist()}")
            print(f"error Δrpy            : {np.round(error_rpy, 6).tolist()}")
            print(f"================ {case_name} 测试结束 ================\n")

            summaries.append(
                {
                    "case": case_name,
                    "actual_total_xyz": actual_total_xyz,
                    "expected_total_xyz": expected_total_xyz,
                    "error_xyz": error_xyz,
                    "actual_total_rpy": actual_total_rpy,
                    "expected_total_rpy": expected_total_rpy,
                    "error_rpy": error_rpy,
                }
            )

        print("\n================ 最终汇总 ================")
        for s in summaries:
            print(
                f"{s['case']:>4} | "
                f"xyz actual={np.round(s['actual_total_xyz'], 6).tolist()} | "
                f"xyz expected={np.round(s['expected_total_xyz'], 6).tolist()} | "
                f"xyz error={np.round(s['error_xyz'], 6).tolist()} | "
                f"rpy actual={np.round(s['actual_total_rpy'], 6).tolist()} | "
                f"rpy expected={np.round(s['expected_total_rpy'], 6).tolist()} | "
                f"rpy error={np.round(s['error_rpy'], 6).tolist()}"
            )
        print("=========================================\n")

        if args.switch_back_to_vr:
            node.switch_control_mode(use_vr_mode=True)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()