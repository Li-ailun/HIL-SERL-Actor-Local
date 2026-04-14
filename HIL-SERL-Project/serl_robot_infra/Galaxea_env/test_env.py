

import time
import numpy as np
import cv2
from geometry_msgs.msg import PoseStamped

# 🌟 确保从 envs 文件夹导入
from envs.dual_galaxea_env import GalaxeaDualArmEnv

# ==========================================
# 🌟 修复版 MockConfig：同时支持点访问和字典访问
# ==========================================
# ==========================================
# 🌟 最终修正版 MockConfig：补全所有底层需要的 Key
# ==========================================
class MockConfig:
    # 1. 属性访问 (满足 dual_galaxea_env.py 中的复位逻辑)
    RESET_L = [0.4, -0.2, 0.2, 0.0, 1.0, 0.0, 0.0]
    RESET_R = [0.4,  0.2, 0.2, 0.0, 1.0, 0.0, 0.0]

    # 2. 字典访问 (满足 Ros2Bridge 的底层初始化)
    def __getitem__(self, key):
        if key == "robot":
            return {
                "hardware": "R1_PRO",
                "enable_publish": [
                    "left_ee_pose", 
                    "right_ee_pose", 
                    "left_gripper", 
                    "right_gripper"
                ]
            }
        raise KeyError(f"Key {key} not found in MockConfig")

def main():
    print("="*50)
    print("🚀 启动星海图双臂 HIL-SERL 环境测试 (绝对坐标测试版)...")
    print("="*50)

    # 1. 初始化环境配置
    mock_config = MockConfig()
    mock_cfg = {}

    # 初始化环境
    # 注意：如果你还没装 pynput，这里可能会报警告，但不影响运行
    env = GalaxeaDualArmEnv(
        config=mock_config, 
        cfg=mock_cfg,
        display_images=True,  # 开启实时监控窗口
        save_video=False      # 测试阶段先不录像
    ) 
    
    # 2. 测试 Reset 获取初始状态
    print("\n[测试 1] 正在连接 ROS2 网络... 等待获取对齐数据")
    obs, info = env.reset()
    
    if obs is not None:
        print("✅ 成功获取初始观测值！")
        print(f"👉 当前左臂末端位姿 (XYZ): {obs['state']['left_ee_pose'][:3]}")
        
        print("\n📺 窗口已弹出！你应该能看到三个相机的画面拼在一起。")
        print("💡 如果画面不动，是因为程序在等待你的回车指令。")
    else:
        print("❌ 获取初始观测值失败，请检查底层通信。")
        return

    # 3. 发送绝对坐标指令
    print("\n[测试 2] 准备发送【绝对坐标】指令...")
    print("🎯 目标位置 (左臂): X=0.2, Y=0.25, Z=-0.3")
    print("⚠️ 警告：请确保路径安全，避免碰撞！")
    input("👉 准备好后，按回车键发送动作指令...")

    # 构建 PoseStamped 消息
    target_msg = PoseStamped()
    target_msg.header.stamp = env.bridge.node.get_clock().now().to_msg()
    target_msg.header.frame_id = 'base_link'
    target_msg.pose.position.x = 0.2
    target_msg.pose.position.y = 0.25
    target_msg.pose.position.z = -0.3
    target_msg.pose.orientation.w = 1.0

    # 获取左臂的话题并发布
    left_topic_key = env.bridge.topics_config.action["left_ee_pose"]
    env.bridge.publishers[left_topic_key].publish(target_msg)
    
    print("\n✅ 绝对坐标指令已发送！")
    
    # 保持 3 秒刷新，让你在监控窗口看到移动过程
    print("正在刷新画面，观察移动...")
    for _ in range(45): # 3秒 * 15Hz
        env._get_sync_obs() # 这会触发图像推送到显示窗口
        time.sleep(1/15.0)
    
    print("\n[测试完成] 正在关闭环境...")
    env.close()

if __name__ == "__main__":
    main()