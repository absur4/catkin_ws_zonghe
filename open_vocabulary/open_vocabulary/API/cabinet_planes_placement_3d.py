#!/usr/bin/env python3
"""Cabinet-only RGB-D plane extraction and placement point generation."""

import json
import os
from typing import Dict, List

import cv2
import numpy as np

try:
    from .cabinet_shelf.detector import CabinetShelfDetector
    from .cabinet_shelf.runtime import DEFAULT_CAMERA_INTRINSICS, load_camera_intrinsics_from_file
    from .utils.pointcloud_geometry import (
        HorizontalSurface,
        compute_placement_points,
        reconstruct_surface_bounds,
        surfaces_to_shelf_y,
    )
except ImportError:
    from cabinet_shelf.detector import CabinetShelfDetector
    from cabinet_shelf.runtime import DEFAULT_CAMERA_INTRINSICS, load_camera_intrinsics_from_file
    from utils.pointcloud_geometry import (
        HorizontalSurface,
        compute_placement_points,
        reconstruct_surface_bounds,
        surfaces_to_shelf_y,
    )


def _build_camera_intrinsics(args) -> Dict[str, float]:
    intr = None
    if args.camera_param_file:
        intr = load_camera_intrinsics_from_file(args.camera_param_file)
    if any(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        base = intr or DEFAULT_CAMERA_INTRINSICS
        intr = {
            "fx": args.fx if args.fx is not None else base["fx"],
            "fy": args.fy if args.fy is not None else base["fy"],
            "cx": args.cx if args.cx is not None else base["cx"],
            "cy": args.cy if args.cy is not None else base["cy"],
        }
    return intr or dict(DEFAULT_CAMERA_INTRINSICS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="柜子 3D 平面与放置点提取")
    parser.add_argument("--image", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="深度图路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--camera-param-file", default=None, help="相机参数文件")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--cabinet-prompt", default="cabinet.")
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
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--plane-method", default="robust_fusion", choices=["robust_fusion", "robust", "hms"])
    parser.add_argument("--grid-resolution", type=float, default=0.01)
    parser.add_argument("--object-height-min", type=float, default=0.02)
    parser.add_argument("--object-height-max", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--max-placements-per-surface", type=int, default=3)
    parser.add_argument("--min-free-area-m2", type=float, default=0.004)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    intr = _build_camera_intrinsics(args)
    detector = CabinetShelfDetector(camera_intrinsics=intr)

    shelf_result = detector.detect(
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

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图像: {args.image}")
    depth_m = detector._load_depth_in_meters(args.depth, depth_scale=args.depth_scale)
    if depth_m.shape[:2] != image.shape[:2]:
        depth_m = cv2.resize(depth_m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    cabinet_bbox = tuple(shelf_result["cabinet"]["bbox"])
    plane_candidates, cabinet_points_clean = detector._detect_cabinet_planes_from_depth(
        depth_m=depth_m,
        cabinet_bbox=cabinet_bbox,
        method=args.plane_method,
        return_filtered_cloud=True,
    )

    surfaces: List[HorizontalSurface] = []
    for cand in plane_candidates:
        pts = np.asarray(cand.get("points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 100:
            continue
        normal, d = detector._fit_plane_tls(pts, up_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64))
        surface = HorizontalSurface(
            plane_equation=np.array([normal[0], normal[1], normal[2], d], dtype=np.float64),
            height_y=float(np.mean(pts[:, 1])),
            inlier_points=pts,
            num_inliers=int(pts.shape[0]),
        )
        reconstruct_surface_bounds(surface)
        surfaces.append(surface)

    placements = []
    for i, surface in enumerate(surfaces):
        surface_placements, _ = compute_placement_points(
            surface=surface,
            all_points=cabinet_points_clean if cabinet_points_clean.size > 0 else surface.inlier_points,
            surface_index=i,
            grid_resolution=args.grid_resolution,
            object_height_min=args.object_height_min,
            object_height_max=args.object_height_max,
            margin=args.margin,
            max_placements=args.max_placements_per_surface,
            min_free_area_m2=args.min_free_area_m2,
            debug_dir=None,
        )
        placements.extend(surface_placements)

    surface_v = surfaces_to_shelf_y(surfaces, detector.camera_intrinsics, image.shape[:2])
    plane_rows = []
    for i, surface in enumerate(surfaces):
        plane_rows.append({
            "index": i,
            "plane_equation": [float(v) for v in surface.plane_equation.tolist()],
            "height_y_m": float(surface.height_y),
            "num_inliers": int(surface.num_inliers),
            "bounds_3d": {k: float(v) for k, v in surface.bounds_3d.items()},
            "surface_area_m2": float(surface.surface_area_m2),
            "v_median_px": float(surface_v[i]) if i < len(surface_v) else -1.0,
        })

    placement_rows = []
    for pp in placements:
        placement_rows.append({
            "surface_index": int(pp.surface_index),
            "position_3d_m": [float(v) for v in pp.position_3d.tolist()],
            "free_area_m2": float(pp.free_area_m2),
            "rect_bounds_3d_m": {k: float(v) for k, v in pp.rect_bounds_3d.items()},
        })

    overlay = detector._draw_3d_overlay(image, surfaces, placements, detector.camera_intrinsics)
    overlay_path = os.path.join(args.output, "cabinet_3d_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)

    if surfaces:
        detector._save_colored_plane_cloud(surfaces, os.path.join(args.output, "cabinet_planes_colored.ply"))
        detector._save_plane_cloud_with_placements(
            surfaces,
            placements,
            os.path.join(args.output, "cabinet_planes_with_placements.ply"),
        )

    result = {
        "image_path": os.path.abspath(args.image),
        "depth_path": os.path.abspath(args.depth),
        "cabinet_bbox": list(cabinet_bbox),
        "shelf_layers": shelf_result["shelf_layers"],
        "planes": plane_rows,
        "placement_points": placement_rows,
        "outputs": {
            "shelf_lines_image": os.path.abspath(os.path.join(args.output, "shelf_lines.jpg")),
            "overlay_image": os.path.abspath(overlay_path),
            "planes_colored_ply": os.path.abspath(os.path.join(args.output, "cabinet_planes_colored.ply")) if surfaces else None,
            "planes_with_placements_ply": os.path.abspath(os.path.join(args.output, "cabinet_planes_with_placements.ply")) if surfaces else None,
        },
    }

    json_path = os.path.join(args.output, "cabinet_planes_placements.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "cabinet_bbox": result["cabinet_bbox"],
        "num_layers": len(result["shelf_layers"]),
        "num_planes": len(result["planes"]),
        "num_placements": len(result["placement_points"]),
        "json_path": os.path.abspath(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
