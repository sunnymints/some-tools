import pyrealsense2 as rs
import numpy as np
import cv2
import time  # 新增时间模块

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()

# 启用流配置（深度和RGB）
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动流
pipeline.start(config)

# 创建对齐对象（将深度对齐到RGB）
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 等待一组连贯的帧
        frames = pipeline.wait_for_frames()
        
        # 对齐深度帧到RGB视角
        aligned_frames = align.process(frames)
        
        # 获取对齐后的深度和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 应用颜色映射到深度图（用于可视化）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )

        # 显示图像
        cv2.imshow('RGB', color_image)
        cv2.imshow('Depth', depth_colormap)

        # 检测按键（新增保存逻辑）
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            break
        elif key in (ord('e'), ord('E')):  # 按E或e保存
            # 生成时间戳文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # 保存RGB图像
            cv2.imwrite(f"rgb_{timestamp}.png", color_image)
            # 保存伪彩色深度图
            cv2.imwrite(f"depth_colormap_{timestamp}.png", depth_colormap)
            # 保存原始深度数据（可选）
            np.save(f"depth_raw_{timestamp}.npy", depth_image)
            print(f"[INFO] 已保存图像：rgb_{timestamp}.png 和 depth_colormap_{timestamp}.png")

finally:
    # 停止流并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()
