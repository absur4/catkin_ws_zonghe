import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


def depth_to_point_cloud(depth_m, color_bgr, intrinsics, max_depth=5.0):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth_m > 0) & np.isfinite(depth_m) & (depth_m < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    z = depth_m[valid]
    x = (u[valid] - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v[valid] - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.stack((x, y, z), axis=1).astype(np.float32)

    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    colors = (rgb[valid].astype(np.float32) / 255.0)
    return points, colors


def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] 未检测到 RealSense 设备，请检查 USB 连接和供电。")
        return

    dev = devices[0]
    try:
        dev_name = dev.get_info(rs.camera_info.name)
        dev_sn = dev.get_info(rs.camera_info.serial_number)
        print(f"[INFO] 已检测到设备: {dev_name}, SN: {dev_sn}")
    except Exception:
        print("[INFO] 已检测到 RealSense 设备")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] RealSense 启动失败: {e}")
        return
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 5)
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth_scale = {depth_scale}")

    for _ in range(30):
        pipeline.wait_for_frames()

    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

    temporal_filter = rs.temporal_filter()
    temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling_filter = rs.hole_filling_filter()
    hole_filling_filter.set_option(rs.option.holes_fill, 1)

    side_offset = np.array([0.6, 0.0, 0.0], dtype=np.float32)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("[ERROR] 单帧采集失败，未获取到有效的 depth/color 帧")
            return

        filtered_depth_frame = spatial_filter.process(depth_frame)
        filtered_depth_frame = temporal_filter.process(filtered_depth_frame)
        filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)

        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
        }

        color_image = np.asanyarray(color_frame.get_data())
        raw_depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        filtered_depth_m = (
            np.asanyarray(filtered_depth_frame.get_data()).astype(np.float32) * depth_scale
        )

        raw_points, raw_colors = depth_to_point_cloud(raw_depth_m, color_image, intrinsics)
        filtered_points, filtered_colors = depth_to_point_cloud(
            filtered_depth_m, color_image, intrinsics
        )

        print(f"[INFO] 单帧点数 - raw: {raw_points.shape[0]}, filtered: {filtered_points.shape[0]}")
        if raw_points.shape[0] == 0 and filtered_points.shape[0] == 0:
            valid_depth = raw_depth_m[np.isfinite(raw_depth_m) & (raw_depth_m > 0)]
            if valid_depth.size > 0:
                print(
                    "[WARN] 当前单帧点云为空，原始深度范围: "
                    f"{valid_depth.min():.3f}m ~ {valid_depth.max():.3f}m"
                )
            else:
                print("[WARN] 当前单帧没有有效深度值，请检查相机视野/反光/距离")
            return

        geometries = []

        if raw_points.shape[0] > 0:
            raw_pcd = o3d.geometry.PointCloud()
            raw_pcd.points = o3d.utility.Vector3dVector(raw_points)
            raw_pcd.colors = o3d.utility.Vector3dVector(raw_colors)
            geometries.append(raw_pcd)

        if filtered_points.shape[0] > 0:
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_points[:, :3] = filtered_points[:, :3] + side_offset
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
            geometries.append(filtered_pcd)

        if not geometries:
            print("[ERROR] 没有可视化点云")
            return

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Single Shot Raw vs Filtered Point Cloud",
            width=1600,
            height=900,
        )

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()