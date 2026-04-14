import cv2
import time

# 直接导入同目录下的文件
from rs_capture import RSCapture
from video_capture import VideoCapture
from multi_video_capture import MultiVideoCapture

def main():
    print("🚀 正在初始化 USB 直连相机阵列，请稍等...")
    
    try:
        # 1. 启动左右手腕 RealSense (自带 15fps 限制)
        left_wrist_cap = RSCapture(name="left_wrist_rgb", serial_number="230322270950", dim=(640, 480), fps=15)
        right_wrist_cap = RSCapture(name="right_wrist_rgb", serial_number="230322271216", dim=(640, 480), fps=15)
        
        # 2. 启动头部 ZED (已确认为 /dev/video2)
        zed_cv2 = cv2.VideoCapture(2)
        zed_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        zed_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        zed_cv2.set(cv2.CAP_PROP_FPS, 15)
        
        if not zed_cv2.isOpened():
            raise RuntimeError("无法打开 ZED 相机 (video2)，请检查 USB 连接或权限！")
            
        head_cap = VideoCapture(zed_cv2, name="head_rgb")

        # 3. 将它们打包进软同步管理器
        multi_cap = MultiVideoCapture({
            "head_rgb": head_cap,
            "left_wrist_rgb": left_wrist_cap,
            "right_wrist_rgb": right_wrist_cap
        })
        
        print("✅ 相机阵列启动成功！(在任意弹出的画面上按 'q' 键退出)")

        # 4. 死循环拉取画面并显示
        while True:
            # 这里的 read() 永远返回最新的对齐画面，并且自带 5 秒超时保护
            frames = multi_cap.read()
            
            if frames is None:
                print("⚠️ 等待相机画面超时...")
                continue

            # 分别提取并显示画面
            if "head_rgb" in frames:
                head_img = frames["head_rgb"]
                # 如果 ZED 吐出的是超宽双目拼接图，切出左半边
                if head_img.shape[1] > head_img.shape[0] * 2:
                    head_img = head_img[:, :head_img.shape[1]//2, :]
                # 稍微缩小一点显示，防止占满整个屏幕
                head_img = cv2.resize(head_img, (640, 360))
                cv2.imshow("Head (ZED)", head_img)

            if "left_wrist_rgb" in frames:
                cv2.imshow("Left Wrist (RS)", frames["left_wrist_rgb"])

            if "right_wrist_rgb" in frames:
                cv2.imshow("Right Wrist (RS)", frames["right_wrist_rgb"])

            # OpenCV 的按键检测，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 接收到退出指令...")
                break

    except Exception as e:
        print(f"\n❌ 运行发生严重错误: {e}")
        
    finally:
        print("🧹 正在安全释放硬件资源...")
        if 'multi_cap' in locals():
            multi_cap.close()
        cv2.destroyAllWindows()
        print("👋 拜拜！")

if __name__ == "__main__":
    main()