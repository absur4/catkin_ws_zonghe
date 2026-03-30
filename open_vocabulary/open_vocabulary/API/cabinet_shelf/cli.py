#!/usr/bin/env python3
"""CLI entry for CabinetShelfDetector."""

try:
    from .detector import CabinetShelfDetector
    from .runtime import (
        DEFAULT_CAMERA_INTRINSICS,
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        load_camera_intrinsics_from_file,
    )
except ImportError:
    from cabinet_shelf.detector import CabinetShelfDetector
    from cabinet_shelf.runtime import (
        DEFAULT_CAMERA_INTRINSICS,
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        load_camera_intrinsics_from_file,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="柜子分层检测：检测物品并判断其位于柜子第几层（支持 2D 和 3D 模式）"
    )
    parser.add_argument("--image", default=None, help="输入图像路径")
    parser.add_argument(
        "--objects", default=None,
        help="物品检测提示词，如 'battery. charger. tool. drill.'（不传则仅检测柜子+层板线）",
    )
    parser.add_argument("--output", required=True, help="输出目录路径")
    parser.add_argument("--cabinet-prompt", default="cabinet.", help="柜子检测提示词")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--num-shelves-hint", type=int, default=4, help="未检测到层板线时的等分层数")
    parser.add_argument("--min-coverage", type=float, default=0.4, help="层板线最小宽度覆盖率（0~1），越高越严格")
    parser.add_argument("--cluster-gap", type=int, default=100)
    parser.add_argument("--min-lines-hint", type=int, default=2)
    parser.add_argument("--no-hough-fallback", action="store_true")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=60)
    parser.add_argument("--hough-min-line-len-ratio", type=float, default=0.5)
    parser.add_argument("--hough-max-line-gap", type=int, default=20)
    parser.add_argument("--horizontal-tol", type=int, default=2)
    parser.add_argument("--min-len-ratio", type=float, default=0.15)
    parser.add_argument("--max-angle-deg", type=float, default=12.0)
    parser.add_argument("--ransac-tol", type=float, default=10.0)
    parser.add_argument("--ransac-iters", type=int, default=80)
    parser.add_argument("--ransac-min-inliers", type=int, default=20)
    parser.add_argument("--dbscan-eps", type=float, default=28.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=4)
    parser.add_argument("--camera-param-file", default=None, help="相机参数文本路径（可直接传 result.txt，自动解析 fx/fy/cx/cy）")
    parser.add_argument("--fx", type=float, default=None, help="相机内参 fx（可覆盖文件值）")
    parser.add_argument("--fy", type=float, default=None, help="相机内参 fy（可覆盖文件值）")
    parser.add_argument("--cx", type=float, default=None, help="相机内参 cx（可覆盖文件值）")
    parser.add_argument("--cy", type=float, default=None, help="相机内参 cy（可覆盖文件值）")
    parser.add_argument("--depth", default=None, help="对齐深度图路径（实验五模式）")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="深度缩放因子，depth_raw * depth_scale = 米")
    parser.add_argument("--experiment5", action="store_true", help="启用实验五：多层柜子物品类别识别与空位检测（RGB+Depth）")
    parser.add_argument("--exp5-object-prompt", default=EXPERIMENT5_DEFAULT_OBJECT_PROMPT, help="实验五开放词汇提示词")
    parser.add_argument("--exp5-nms-iou", type=float, default=0.45, help="实验五目标框去重 NMS IoU 阈值")
    parser.add_argument("--empty-min-width-m", type=float, default=0.08, help="空位最小宽度（米）")
    parser.add_argument("--empty-min-height-m", type=float, default=0.05, help="空位最小高度（米）")
    parser.add_argument("--empty-min-clearance-m", type=float, default=0.10, help="空位最小净空（米）")
    parser.add_argument("--empty-margin-px", type=int, default=6, help="空位检测时障碍膨胀边距（像素）")
    parser.add_argument("--empty-max-per-layer", type=int, default=2, help="每层最多输出空位数量")
    parser.add_argument("--empty-min-area-cells", type=int, default=120, help="空位最小面积（像素格）")
    parser.add_argument("--place-boundary-margin-px", type=int, default=8, help="放置点避开层边界的像素边距")
    parser.add_argument("--place-min-clearance-m", type=float, default=0.03, help="放置引导点最小安全间隙（米）")
    parser.add_argument("--exp5-plane-method", default="robust_fusion", choices=["robust_fusion", "robust", "hms"], help="实验五层平面估计方法")
    parser.add_argument("--exp5-plane-inlier-tol-m", type=float, default=0.012, help="实验五层平面内点厚度阈值（米）")
    parser.add_argument("--exp5-layer-roi-expand-y-px", type=int, default=12, help="实验五每层 ROI 在 y 方向外扩像素（仅层 ROI 外扩，柜子框不外扩）")
    parser.add_argument("--exp5-top-layers", type=int, default=0, help="仅处理柜子上方前 N 层；0 表示全部层")
    parser.add_argument("--exp5-lower-row-tol-px", type=int, default=18, help="同层仅保留底排框时的像素容差")
    parser.add_argument("--pcd", default=None, help="PCD 点云文件路径（启用 3D 模式）")
    parser.add_argument("--visualize-3d", action="store_true", help="弹出 Open3D 3D 可视化窗口")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="体素下采样大小（米）")
    parser.add_argument("--hms-iterations", type=int, default=300, help="HMS-RANSAC 迭代次数")
    parser.add_argument("--angular-threshold", type=float, default=10.0, help="水平面角度容差（度）")
    parser.add_argument("--inlier-distance", type=float, default=0.01, help="内点距离阈值（米）")
    parser.add_argument("--hms-min-inliers", type=int, default=3000, help="HMS-RANSAC 最少内点数")
    parser.add_argument("--duplicate-distance", type=float, default=0.03, help="重复平面合并距离（米）")
    parser.add_argument("--grid-resolution", type=float, default=0.01, help="占用栅格分辨率（米）")
    parser.add_argument("--max-2d-layer-dist-px", type=int, default=25, help="3D 平面到 2D 层中心的最大像素距离（用于过滤 3D 平面）")
    args = parser.parse_args()

    camera_intrinsics = None
    if args.camera_param_file:
        camera_intrinsics = load_camera_intrinsics_from_file(args.camera_param_file)
        if camera_intrinsics is not None:
            print(
                "[CabinetShelfDetector] 已加载相机内参: "
                f"fx={camera_intrinsics['fx']:.6f}, fy={camera_intrinsics['fy']:.6f}, "
                f"cx={camera_intrinsics['cx']:.6f}, cy={camera_intrinsics['cy']:.6f}"
            )
    if any(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        base = camera_intrinsics or DEFAULT_CAMERA_INTRINSICS
        camera_intrinsics = {
            "fx": args.fx if args.fx is not None else base["fx"],
            "fy": args.fy if args.fy is not None else base["fy"],
            "cx": args.cx if args.cx is not None else base["cx"],
            "cy": args.cy if args.cy is not None else base["cy"],
        }
        print(
            "[CabinetShelfDetector] 使用覆盖后的相机内参: "
            f"fx={camera_intrinsics['fx']:.6f}, fy={camera_intrinsics['fy']:.6f}, "
            f"cx={camera_intrinsics['cx']:.6f}, cy={camera_intrinsics['cy']:.6f}"
        )

    detector = CabinetShelfDetector(camera_intrinsics=camera_intrinsics)

    if args.experiment5:
        if not args.image:
            parser.error("实验五模式需要 --image")
        if not args.depth:
            parser.error("实验五模式需要 --depth（对齐深度图）")
        result = detector.detect_experiment5(
            image_path=args.image,
            depth_path=args.depth,
            output_dir=args.output,
            object_prompt=args.exp5_object_prompt,
            cabinet_prompt=args.cabinet_prompt,
            depth_scale=args.depth_scale,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_iou=args.exp5_nms_iou,
            num_shelves_hint=args.num_shelves_hint,
            min_coverage=args.min_coverage,
            cluster_min_gap=args.cluster_gap,
            min_lines_hint=args.min_lines_hint,
            hough_fallback=not args.no_hough_fallback,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            hough_threshold=args.hough_threshold,
            hough_min_line_len_ratio=args.hough_min_line_len_ratio,
            hough_max_line_gap=args.hough_max_line_gap,
            horizontal_tol=args.horizontal_tol,
            min_len_ratio=args.min_len_ratio,
            max_angle_deg=args.max_angle_deg,
            ransac_tol=args.ransac_tol,
            ransac_iters=args.ransac_iters,
            ransac_min_inliers=args.ransac_min_inliers,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            empty_min_width_m=args.empty_min_width_m,
            empty_min_height_m=args.empty_min_height_m,
            empty_min_clearance_m=args.empty_min_clearance_m,
            empty_margin_px=args.empty_margin_px,
            empty_max_per_layer=args.empty_max_per_layer,
            empty_min_area_cells=args.empty_min_area_cells,
            place_boundary_margin_px=args.place_boundary_margin_px,
            place_min_clearance_m=args.place_min_clearance_m,
            layer_roi_expand_y_px=args.exp5_layer_roi_expand_y_px,
            plane_method=args.exp5_plane_method,
            plane_inlier_tol_m=args.exp5_plane_inlier_tol_m,
            top_layers_to_process=args.exp5_top_layers,
            lower_row_tol_px=args.exp5_lower_row_tol_px,
        )
        print(f"\n{'='*60}")
        print("实验五结果摘要:")
        print(f"  层数: {result['summary']['num_layers']}")
        print(f"  目标物体数: {result['summary']['num_target_objects']}")
        print(f"  空位数: {result['summary']['num_empty_spaces']}")
        print(f"{'='*60}")
        return

    if args.pcd:
        result = detector.detect_3d(
            pcd_path=args.pcd,
            image_path=args.image,
            output_dir=args.output,
            object_prompt=args.objects,
            visualize_3d_flag=args.visualize_3d,
            voxel_size=args.voxel_size,
            min_iterations=args.hms_iterations,
            angular_threshold_deg=args.angular_threshold,
            inlier_distance=args.inlier_distance,
            min_inliers=args.hms_min_inliers,
            duplicate_distance=args.duplicate_distance,
            grid_resolution=args.grid_resolution,
            max_2d_layer_dist_px=args.max_2d_layer_dist_px,
            cabinet_prompt=args.cabinet_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            num_shelves_hint=args.num_shelves_hint,
        )
        print(f"\n{'='*60}")
        print("3D 检测结果摘要:")
        print(f"  平面数: {len(result['point_cloud_3d']['surfaces'])}")
        print(f"  放置点数: {len(result['point_cloud_3d']['placement_points'])}")
        print(f"{'='*60}")
        return

    if not args.image:
        parser.error("2D 模式需要 --image")
    result = detector.detect(
        image_path=args.image,
        object_prompt=args.objects,
        cabinet_prompt=args.cabinet_prompt,
        output_dir=args.output,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        num_shelves_hint=args.num_shelves_hint,
        min_coverage=args.min_coverage,
        cluster_min_gap=args.cluster_gap,
        min_lines_hint=args.min_lines_hint,
        hough_fallback=not args.no_hough_fallback,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
        hough_min_line_len_ratio=args.hough_min_line_len_ratio,
        hough_max_line_gap=args.hough_max_line_gap,
        horizontal_tol=args.horizontal_tol,
        min_len_ratio=args.min_len_ratio,
        max_angle_deg=args.max_angle_deg,
        ransac_tol=args.ransac_tol,
        ransac_iters=args.ransac_iters,
        ransac_min_inliers=args.ransac_min_inliers,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    print(f"\n{'='*60}")
    print("检测结果摘要:")
    print(f"  层数: {result['num_layers']}")
    print(f"  物品数: {len(result['objects'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
