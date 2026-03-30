#!/usr/bin/env python3
"""3D cabinet shelf detection workflow."""

import json
import os
from typing import Dict, Optional

try:
    from ..runtime import (
        cv2,
        draw_3d_overlay,
        np,
        visualize_3d,
    )
    from ...utils.pointcloud_geometry import (
        detect_horizontal_surfaces,
        project_points_to_image,
        surfaces_to_shelf_y,
    )
except ImportError:
    from cabinet_shelf.runtime import (
        cv2,
        draw_3d_overlay,
        np,
        visualize_3d,
    )
    from utils.pointcloud_geometry import (
        detect_horizontal_surfaces,
        project_points_to_image,
        surfaces_to_shelf_y,
    )
def detect_3d(detector, pcd_path: str, image_path: Optional[str] = None,
              output_dir: Optional[str] = None, object_prompt: Optional[str] = None,
              visualize_3d_flag: bool = False, up_axis=None, voxel_size: float = 0.005,
              min_iterations: int = 300, angular_threshold_deg: float = 10.0,
              inlier_distance: float = 0.01, min_inliers: int = 3000,
              duplicate_distance: float = 0.03, grid_resolution: float = 0.01,
              object_height_min: float = 0.02, object_height_max: float = 0.50,
              margin: float = 0.02, max_placements_per_surface: int = 3,
              max_2d_layer_dist_px: int = 25, cabinet_prompt: str = "cabinet.",
              box_threshold: float = 0.3, text_threshold: float = 0.25,
              num_shelves_hint: int = 4) -> Dict:

    if up_axis is None:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    image = None
    image_2d_result = None
    if image_path is not None and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            print("\n[CabinetShelfDetector] detect_3d: 运行 2D 层板检测...")
            try:
                image_2d_result = detector.detect(
                    image_path=image_path,
                    object_prompt=object_prompt,
                    cabinet_prompt=cabinet_prompt,
                    output_dir=output_dir,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    num_shelves_hint=num_shelves_hint,
                )
            except Exception as e:
                print(f"  2D 检测失败: {e}")

    debug_dir = os.path.join(output_dir, "debug_3d") if output_dir else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print("\n[CabinetShelfDetector] detect_3d: 开始 3D 点云分析...")
    surfaces, placements, all_points = detect_horizontal_surfaces(
        pcd_path=pcd_path, up_axis=up_axis, voxel_size=voxel_size, min_iterations=min_iterations,
        angular_threshold_deg=angular_threshold_deg, inlier_distance=inlier_distance,
        min_inliers=min_inliers, duplicate_distance=duplicate_distance,
        grid_resolution=grid_resolution, object_height_min=object_height_min,
        object_height_max=object_height_max, margin=margin,
        max_placements_per_surface=max_placements_per_surface, debug_dir=debug_dir,
    )

    if image is not None and image_2d_result and image_2d_result.get("shelf_layers"):
        img_h, img_w = image.shape[:2]
        shelf_y_3d = surfaces_to_shelf_y(surfaces, detector.camera_intrinsics, (img_h, img_w))
        shelf_layers_2d = image_2d_result["shelf_layers"]
        layer_mids = [(layer["y_top"] + layer["y_bottom"]) / 2.0 for layer in shelf_layers_2d]
        if shelf_y_3d and layer_mids:
            selected = []
            for mid in layer_mids:
                best_i = None
                best_dist = float("inf")
                for i, y_3d in enumerate(shelf_y_3d):
                    if y_3d < 0:
                        continue
                    dist = abs(y_3d - mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_i = i
                if best_i is not None:
                    if best_dist > max_2d_layer_dist_px:
                        print(f"  2D-3D 距离过大: layer_mid={mid:.1f}px, dist={best_dist:.1f}px")
                    selected.append(best_i)
            selected_set = []
            for i in selected:
                if i not in selected_set:
                    selected_set.append(i)
            n_layers = len(layer_mids)
            if len(selected_set) < n_layers:
                remaining = [i for i in range(len(shelf_y_3d)) if i not in selected_set]
                remaining_sorted = sorted(remaining, key=lambda i: min(abs(shelf_y_3d[i] - mid) for mid in layer_mids))
                selected_set.extend(remaining_sorted[:max(0, n_layers - len(selected_set))])
            keep_idx = set(selected_set[:n_layers])
            index_map = {}
            filtered_surfaces = []
            for old_idx, s in enumerate(surfaces):
                if old_idx in keep_idx:
                    index_map[old_idx] = len(filtered_surfaces)
                    filtered_surfaces.append(s)
            filtered_placements = []
            for pp in placements:
                if pp.surface_index in index_map:
                    pp.surface_index = index_map[pp.surface_index]
                    filtered_placements.append(pp)
            print(f"  2D 层数约束保留最近平面: {len(surfaces)} -> {len(filtered_surfaces)}")
            surfaces = filtered_surfaces
            placements = filtered_placements

    surfaces_json = []
    for i, s in enumerate(surfaces):
        surfaces_json.append({
            "index": i,
            "plane_equation": s.plane_equation.tolist(),
            "height_y_m": round(s.height_y, 6),
            "num_inliers": s.num_inliers,
            "bounds_3d": {k: round(v, 6) for k, v in s.bounds_3d.items()},
            "surface_area_m2": round(s.surface_area_m2, 6),
            "matched_2d_layer": None,
        })
    placements_json = []
    for pp in placements:
        placements_json.append({
            "surface_index": pp.surface_index,
            "position_3d": [round(v, 6) for v in pp.position_3d.tolist()],
            "free_area_m2": round(pp.free_area_m2, 6),
            "rect_bounds_3d": {k: round(v, 6) for k, v in pp.rect_bounds_3d.items()},
            "projected_2d": None,
        })
    result = {
        "point_cloud_3d": {
            "pcd_path": os.path.abspath(pcd_path),
            "surfaces": surfaces_json,
            "placement_points": placements_json,
        },
        "image_2d": image_2d_result,
        "surface_layer_mapping": {},
    }

    if image is not None:
        img_h, img_w = image.shape[:2]
        shelf_y_3d = surfaces_to_shelf_y(surfaces, detector.camera_intrinsics, (img_h, img_w))
        for pp_json, pp in zip(placements_json, placements):
            uv = project_points_to_image(pp.position_3d, detector.camera_intrinsics, (img_h, img_w))
            u, v = uv[0]
            if u >= 0 and v >= 0:
                pp_json["projected_2d"] = {"u": int(round(u)), "v": int(round(v))}
        if image_2d_result and image_2d_result.get("shelf_layers"):
            shelf_layers_2d = image_2d_result["shelf_layers"]
            candidates = []
            for i, y_3d in enumerate(shelf_y_3d):
                if y_3d < 0:
                    continue
                for layer in shelf_layers_2d:
                    mid = (layer["y_top"] + layer["y_bottom"]) / 2.0
                    dist = abs(y_3d - mid)
                    candidates.append((dist, i, layer["layer"]))
            candidates.sort(key=lambda x: x[0])
            used_surfaces = set()
            used_layers = set()
            for _, i, layer_id in candidates:
                if i in used_surfaces or layer_id in used_layers:
                    continue
                used_surfaces.add(i)
                used_layers.add(layer_id)
                if i < len(surfaces_json):
                    surfaces_json[i]["matched_2d_layer"] = layer_id
                    result["surface_layer_mapping"][str(i)] = layer_id
        if output_dir is not None:
            vis_2d = draw_3d_overlay(image, surfaces, placements, detector.camera_intrinsics)
            vis_path = os.path.join(output_dir, "3d_overlay.jpg")
            cv2.imwrite(vis_path, vis_2d)
            print(f"  2D 叠加图已保存至: {vis_path}")

    if visualize_3d_flag:
        visualize_3d(all_points, surfaces, placements)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "3d_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n3D 检测 JSON 已保存至: {json_path}")
    print(f"\n[CabinetShelfDetector] detect_3d 完成: {len(surfaces)} 个平面, {len(placements)} 个放置点")
    return result
