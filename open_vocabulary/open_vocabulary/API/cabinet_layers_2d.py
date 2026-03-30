#!/usr/bin/env python3
"""2D cabinet ROI detection and shelf layering."""

import json

try:
    from .cabinet_shelf.detector import CabinetShelfDetector
    from .cabinet_shelf.runtime import DEFAULT_CAMERA_INTRINSICS, load_camera_intrinsics_from_file
except ImportError:
    from cabinet_shelf.detector import CabinetShelfDetector
    from cabinet_shelf.runtime import DEFAULT_CAMERA_INTRINSICS, load_camera_intrinsics_from_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="2D 柜子检测与分层")
    parser.add_argument("--image", required=True, help="RGB 图像路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--cabinet-prompt", default="cabinet.", help="柜子检测提示词")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--num-shelves-hint", type=int, default=4)
    parser.add_argument("--min-coverage", type=float, default=0.4)
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
    parser.add_argument("--camera-param-file", default=None, help="相机参数文件，可选")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    args = parser.parse_args()

    camera_intrinsics = None
    if args.camera_param_file:
        camera_intrinsics = load_camera_intrinsics_from_file(args.camera_param_file)
    if any(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        base = camera_intrinsics or DEFAULT_CAMERA_INTRINSICS
        camera_intrinsics = {
            "fx": args.fx if args.fx is not None else base["fx"],
            "fy": args.fy if args.fy is not None else base["fy"],
            "cx": args.cx if args.cx is not None else base["cx"],
            "cy": args.cy if args.cy is not None else base["cy"],
        }

    detector = CabinetShelfDetector(camera_intrinsics=camera_intrinsics)
    result = detector.detect(
        image_path=args.image,
        object_prompt=None,
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
    print(json.dumps({
        "cabinet_bbox": result["cabinet"]["bbox"],
        "num_layers": result["num_layers"],
        "shelf_lines_y": result["shelf_lines_y"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
