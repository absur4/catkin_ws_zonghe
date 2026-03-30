#!/usr/bin/env python3
"""Experiment 5 workflow for cabinet shelf detection."""

import json
import os
from typing import Dict

try:
    from ..runtime import (
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        assign_global_planes_to_layers,
        assign_object_to_layer,
        cluster_horizontal_lines,
        compute_shelf_layers,
        cv2,
        detect_cabinet_planes_from_depth,
        detect_shelf_lines,
        draw_3d_overlay,
        draw_experiment5_image,
        estimate_object_contact_depth,
        fit_plane_tls,
        generate_equal_layers,
        keep_only_lower_row_boxes,
        load_depth_in_meters,
        map_label_to_experiment5_target,
        nms_xyxy_numpy,
        np,
        o3d,
        pixel_to_camera_xyz,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
    )
    from ...utils.pointcloud_geometry import (
        HorizontalSurface,
        compute_placement_points,
        project_points_to_image,
        reconstruct_surface_bounds,
        surfaces_to_shelf_y,
    )
except ImportError:
    from cabinet_shelf.runtime import (
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        assign_global_planes_to_layers,
        assign_object_to_layer,
        cluster_horizontal_lines,
        compute_shelf_layers,
        cv2,
        detect_cabinet_planes_from_depth,
        detect_shelf_lines,
        draw_3d_overlay,
        draw_experiment5_image,
        estimate_object_contact_depth,
        fit_plane_tls,
        generate_equal_layers,
        keep_only_lower_row_boxes,
        load_depth_in_meters,
        map_label_to_experiment5_target,
        nms_xyxy_numpy,
        np,
        o3d,
        pixel_to_camera_xyz,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
    )
    from utils.pointcloud_geometry import (
        HorizontalSurface,
        compute_placement_points,
        project_points_to_image,
        reconstruct_surface_bounds,
        surfaces_to_shelf_y,
    )
def detect_experiment5(detector, image_path: str, depth_path: str, output_dir: str,
                       object_prompt: str = EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
                       cabinet_prompt: str = "cabinet.", depth_scale: float = 0.001,
                       box_threshold: float = 0.3, text_threshold: float = 0.25,
                       nms_iou: float = 0.45, num_shelves_hint: int = 4,
                       min_coverage: float = 0.4, cluster_min_gap: int = 100,
                       min_lines_hint: int = 2, hough_fallback: bool = True,
                       canny_low: int = 50, canny_high: int = 150,
                       hough_threshold: int = 60, hough_min_line_len_ratio: float = 0.5,
                       hough_max_line_gap: int = 20, horizontal_tol: int = 2,
                       min_len_ratio: float = 0.15, max_angle_deg: float = 12.0,
                       ransac_tol: float = 10.0, ransac_iters: int = 80,
                       ransac_min_inliers: int = 20, dbscan_eps: float = 28.0,
                       dbscan_min_samples: int = 4, empty_min_width_m: float = 0.08,
                       empty_min_height_m: float = 0.05, empty_min_clearance_m: float = 0.10,
                       empty_margin_px: int = 6, empty_max_per_layer: int = 2,
                       empty_min_area_cells: int = 120, place_boundary_margin_px: int = 8,
                       place_min_clearance_m: float = 0.03, layer_roi_expand_y_px: int = 12,
                       plane_method: str = "robust_fusion", plane_inlier_tol_m: float = 0.012,
                       top_layers_to_process: int = 0, lower_row_tol_px: int = 18) -> Dict:

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图像: {image_path}")
    depth_m = load_depth_in_meters(depth_path, depth_scale=depth_scale)
    img_h, img_w = image.shape[:2]
    if depth_m.shape[:2] != (img_h, img_w):
        print(
            f"[CabinetShelfDetector] 深度尺寸 {depth_m.shape[:2]} 与 RGB {image.shape[:2]} 不一致，"
            "将使用最近邻缩放到 RGB 尺寸"
        )
        depth_m = cv2.resize(depth_m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    os.makedirs(output_dir, exist_ok=True)
    print("\n[CabinetShelfDetector][Exp5] Step 1: 检测柜子...")
    cab_pred = detector._predict_dino_boxes(
        image_path=image_path,
        prompt=cabinet_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if cab_pred["boxes"]:
        boxes_np = np.array(cab_pred["boxes"], dtype=np.float32)
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        best_idx = int(np.argmax(areas))
        cb = boxes_np[best_idx]
        cabinet_bbox = (
            int(max(0, min(img_w - 1, round(float(cb[0]))))),
            int(max(0, min(img_h - 1, round(float(cb[1]))))),
            int(max(1, min(img_w, round(float(cb[2]))))),
            int(max(1, min(img_h, round(float(cb[3]))))),
        )
    else:
        cabinet_bbox = (0, 0, img_w, img_h)
    print(f"  cabinet bbox: {cabinet_bbox}")

    print("\n[CabinetShelfDetector][Exp5] Step 2: 3D plane extraction (HMS flow)...")
    global_plane_candidates, cabinet_points_clean = detect_cabinet_planes_from_depth(
        depth_m=depth_m,
        cabinet_bbox=cabinet_bbox,
        camera_intrinsics=detector.camera_intrinsics,
        method=plane_method,
        return_filtered_cloud=True,
    )
    surfaces_3d = []
    surface_sources = []
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for cand in global_plane_candidates:
        pts = np.asarray(cand.get("points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 100:
            continue
        n_fit, d_fit = fit_plane_tls(pts, up_axis=up_axis)
        surf = HorizontalSurface(
            plane_equation=np.array([n_fit[0], n_fit[1], n_fit[2], d_fit], dtype=np.float64),
            height_y=float(np.mean(pts[:, 1])),
            inlier_points=pts,
            num_inliers=int(pts.shape[0]),
        )
        reconstruct_surface_bounds(surf)
        surfaces_3d.append(surf)
        surface_sources.append(str(cand.get("source", "unknown")))

    placements_3d = []
    min_free_area_m2 = max(0.001, float(empty_min_width_m * empty_min_height_m))
    for i, surf in enumerate(surfaces_3d):
        pps, _ = compute_placement_points(
            surface=surf,
            all_points=cabinet_points_clean if cabinet_points_clean.size > 0 else surf.inlier_points,
            surface_index=i,
            grid_resolution=0.01,
            object_height_min=0.02,
            object_height_max=0.50,
            margin=0.02,
            max_placements=max(1, int(empty_max_per_layer)),
            min_free_area_m2=min_free_area_m2,
            debug_dir=None,
        )
        placements_3d.extend(pps)

    filtered_cloud_path = None
    planes_ply_path = None
    planes_with_placements_ply_path = None
    if o3d is not None and cabinet_points_clean.size > 0:
        filtered_cloud_path = os.path.join(output_dir, "cabinet_filtered_cloud.ply")
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(cabinet_points_clean)
        o3d.io.write_point_cloud(filtered_cloud_path, pcd_clean)
    if surfaces_3d:
        planes_ply_path = os.path.join(output_dir, "hms_planes_colored.ply")
        save_colored_plane_cloud(surfaces_3d, planes_ply_path)
        planes_with_placements_ply_path = os.path.join(output_dir, "hms_planes_with_placements.ply")
        save_plane_cloud_with_placements(surfaces_3d, placements_3d, planes_with_placements_ply_path)

    print(f"  global planes: {len(surfaces_3d)}")
    print(f"  placement candidates (3D): {len(placements_3d)}")

    print("\n[CabinetShelfDetector][Exp5] Step 3: detect shelf layers (2D for filtering)...")
    debug_dir = os.path.join(output_dir, "debug_exp5")
    os.makedirs(debug_dir, exist_ok=True)
    raw_y = detect_shelf_lines(
        image, roi=cabinet_bbox, min_coverage=min_coverage, debug_dir=debug_dir,
        min_lines_hint=min_lines_hint, hough_fallback=hough_fallback, canny_low=canny_low,
        canny_high=canny_high, hough_threshold=hough_threshold,
        hough_min_line_len_ratio=hough_min_line_len_ratio, hough_max_line_gap=hough_max_line_gap,
        horizontal_tol=horizontal_tol, min_len_ratio=min_len_ratio, max_angle_deg=max_angle_deg,
        ransac_tol=ransac_tol, ransac_iters=ransac_iters, ransac_min_inliers=ransac_min_inliers,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
    )
    shelf_y = cluster_horizontal_lines(raw_y, min_gap=cluster_min_gap)
    if shelf_y:
        full_shelf_layers = compute_shelf_layers(cabinet_bbox, shelf_y)
    else:
        full_shelf_layers = generate_equal_layers(cabinet_bbox, num_shelves_hint)
        shelf_y = [layer["y_bottom"] for layer in full_shelf_layers[:-1]]
    shelf_layers = full_shelf_layers[:top_layers_to_process] if top_layers_to_process > 0 else full_shelf_layers
    shelf_y = [layer["y_bottom"] for layer in shelf_layers[:-1]]
    print(f"  shelf lines: {shelf_y}")
    print(f"  processing top layers: {len(shelf_layers)}")
    print(f"  layer ROI expand (y): {int(layer_roi_expand_y_px)} px")
    print(f"  layer plane method: {plane_method}")

    plane_candidates_for_match = []
    surface_v_medians = surfaces_to_shelf_y(surfaces_3d, detector.camera_intrinsics, (img_h, img_w))
    for i, surf in enumerate(surfaces_3d):
        plane_candidates_for_match.append({
            "index": int(i),
            "plane_y_m": float(surf.height_y),
            "v_median_px": float(surface_v_medians[i]) if i < len(surface_v_medians) else -1.0,
            "num_inliers": int(surf.num_inliers),
            "source": surface_sources[i] if i < len(surface_sources) else "unknown",
        })
    layer_plane_hints = assign_global_planes_to_layers(planes=plane_candidates_for_match, shelf_layers=shelf_layers)
    print(f"  layer plane hints: {len(layer_plane_hints)}/{len(shelf_layers)}")
    surface_to_layer = {}
    for layer_id, hint in layer_plane_hints.items():
        if "plane_index" in hint:
            surface_to_layer[int(hint["plane_index"])] = int(layer_id)

    print("\n[CabinetShelfDetector][Exp5] Step 4: detect objects (2D semantics)...")
    obj_pred = detector._predict_dino_boxes(
        image_path=image_path, prompt=object_prompt, box_threshold=box_threshold, text_threshold=text_threshold
    )
    boxes_np = np.array(obj_pred["boxes"], dtype=np.float32) if obj_pred["boxes"] else np.zeros((0, 4), dtype=np.float32)
    scores_np = np.array(obj_pred["confidences"], dtype=np.float32) if obj_pred["confidences"] else np.zeros((0,), dtype=np.float32)
    labels = obj_pred["labels"]
    keep_idx = nms_xyxy_numpy(boxes_np, scores_np, iou_threshold=nms_iou) if len(boxes_np) > 0 else []
    all_objects_in_active_layers = []
    target_objects = []
    for i in keep_idx:
        b = boxes_np[i]
        score = float(scores_np[i])
        raw_label = labels[i]
        bx1 = int(max(0, min(img_w - 1, round(float(b[0])))))
        by1 = int(max(0, min(img_h - 1, round(float(b[1])))))
        bx2 = int(max(bx1 + 1, min(img_w, round(float(b[2])))))
        by2 = int(max(by1 + 1, min(img_h, round(float(b[3])))))
        bbox = (bx1, by1, bx2, by2)
        layer = assign_object_to_layer(bbox, shelf_layers)
        if layer <= 0:
            continue
        z_contact = estimate_object_contact_depth(depth_m, bbox)
        anchor_3d = None
        anchor_pixel = None
        if z_contact is not None:
            anchor_u = int(round((bx1 + bx2) * 0.5))
            anchor_v = max(0, min(img_h - 1, by2 - 2))
            x_m, y_m, z_m = pixel_to_camera_xyz(anchor_u, anchor_v, z_contact, detector.camera_intrinsics)
            anchor_pixel = {"u": anchor_u, "v": anchor_v}
            anchor_3d = {"x": round(x_m, 4), "y": round(y_m, 4), "z": round(z_m, 4)}
        obj_base = {
            "class_name": raw_label.strip(),
            "bbox": [bx1, by1, bx2, by2],
            "confidence": round(score, 4),
            "layer": int(layer),
            "anchor_pixel": anchor_pixel,
            "anchor_3d_m": anchor_3d,
        }
        all_objects_in_active_layers.append(obj_base)
        target_name = map_label_to_experiment5_target(raw_label)
        if target_name is not None:
            target_obj = {"name": target_name, **obj_base}
            target_objects.append(target_obj)
            print(f"  target: {target_name}, layer={layer}, conf={score:.2f}, label='{raw_label.strip()}'")
    target_objects = keep_only_lower_row_boxes(target_objects, y_tol_px=lower_row_tol_px)

    print("\n[CabinetShelfDetector][Exp5] Step 5: build empty spaces from 3D placements...")
    empty_spaces_raw = []
    all_candidate_spaces = []
    for pp in placements_3d:
        sid = int(pp.surface_index)
        if sid not in surface_to_layer:
            continue
        layer_id = int(surface_to_layer[sid])
        rect = pp.rect_bounds_3d
        width_m = float(rect["x_max"] - rect["x_min"])
        depth_m_rect = float(rect["z_max"] - rect["z_min"])
        clearance_proxy = 0.5 * min(width_m, depth_m_rect)
        uv = project_points_to_image(np.asarray(pp.position_3d, dtype=np.float64).reshape(1, 3), detector.camera_intrinsics, (img_h, img_w))
        u, v = uv[0]
        pixel = None
        if u >= 0 and v >= 0:
            pixel = {"u": int(round(float(u))), "v": int(round(float(v)))}
        entry = {
            "layer": layer_id,
            "label": "Empty Space",
            "surface_index": sid,
            "free_area_m2": round(float(pp.free_area_m2), 4),
            "placement_point_pixel": pixel,
            "placement_point_3d_m": {
                "x": round(float(pp.position_3d[0]), 4),
                "y": round(float(pp.position_3d[1]), 4),
                "z": round(float(pp.position_3d[2]), 4),
            },
            "placement_clearance_proxy_m": round(float(clearance_proxy), 4),
            "rect_bounds_3d_m": {
                "x_min": round(float(rect["x_min"]), 4),
                "x_max": round(float(rect["x_max"]), 4),
                "z_min": round(float(rect["z_min"]), 4),
                "z_max": round(float(rect["z_max"]), 4),
                "y": round(float(rect["y"]), 4),
            },
        }
        all_candidate_spaces.append(entry)
        if clearance_proxy >= float(place_min_clearance_m):
            empty_spaces_raw.append(entry)
    empty_spaces = []
    by_layer = {}
    for es in empty_spaces_raw:
        by_layer.setdefault(int(es["layer"]), []).append(es)
    for layer_id in sorted(by_layer.keys()):
        rows = sorted(by_layer[layer_id], key=lambda x: float(x["free_area_m2"]), reverse=True)
        empty_spaces.extend(rows[:max(1, int(empty_max_per_layer))])
    if not empty_spaces and all_candidate_spaces:
        by_layer_fb = {}
        for es in all_candidate_spaces:
            by_layer_fb.setdefault(int(es["layer"]), []).append(es)
        for layer_id in sorted(by_layer_fb.keys()):
            rows = sorted(by_layer_fb[layer_id], key=lambda x: float(x["free_area_m2"]), reverse=True)
            empty_spaces.extend(rows[:1])
    print(f"  empty spaces (3D points): {len(empty_spaces)}")
    placements_for_overlay = [pp for pp in placements_3d if int(pp.surface_index) in surface_to_layer]

    result = {
        "task": "experiment5_milk_cereal_empty",
        "image_path": os.path.abspath(image_path),
        "depth_path": os.path.abspath(depth_path),
        "depth_scale_to_m": float(depth_scale),
        "camera_intrinsics": {k: float(detector.camera_intrinsics[k]) for k in ("fx", "fy", "cx", "cy")},
        "cabinet": {"bbox": list(cabinet_bbox)},
        "shelf_lines_y": [int(v) for v in shelf_y],
        "shelf_layers": shelf_layers,
        "target_objects": target_objects,
        "empty_spaces": empty_spaces,
        "hms_planes": [
            {
                "layer": int(layer_id),
                "plane_y_m": round(float(hint["plane_y_m"]), 4),
                "num_inliers": int(hint.get("num_inliers", 0)),
                "source": str(hint.get("source", "unknown")),
            }
            for layer_id, hint in sorted(layer_plane_hints.items(), key=lambda kv: kv[0])
        ],
        "global_plane_candidates": [
            {
                "index": int(p["index"]),
                "plane_y_m": round(float(p["plane_y_m"]), 4),
                "v_median_px": round(float(p["v_median_px"]), 2),
                "num_inliers": int(p["num_inliers"]),
                "source": str(p["source"]),
            }
            for p in plane_candidates_for_match
        ],
        "layer_plane_hints": [
            {
                "layer": int(layer_id),
                "plane_y_m": round(float(hint["plane_y_m"]), 4),
                "v_median_px": round(float(hint.get("v_median_px", -1.0)), 2),
                "num_inliers": int(hint.get("num_inliers", 0)),
                "source": str(hint.get("source", "unknown")),
                "plane_index": int(hint["plane_index"]) if "plane_index" in hint else None,
            }
            for layer_id, hint in sorted(layer_plane_hints.items(), key=lambda kv: kv[0])
        ],
        "surface_layer_mapping": {str(k): int(v) for k, v in sorted(surface_to_layer.items())},
        "obstacle_count_in_active_layers": len(all_objects_in_active_layers),
        "placements_3d": {
            "target_objects": [
                {"name": obj["name"], "layer": obj["layer"], "center_3d_m": obj["anchor_3d_m"]}
                for obj in target_objects
            ],
            "empty_spaces": [
                {"layer": es["layer"], "placement_point_3d_m": es["placement_point_3d_m"]}
                for es in empty_spaces
            ],
        },
        "outputs": {
            "filtered_cloud_ply": os.path.abspath(filtered_cloud_path) if filtered_cloud_path else None,
            "planes_colored_ply": os.path.abspath(planes_ply_path) if planes_ply_path else None,
            "planes_with_placements_ply": os.path.abspath(planes_with_placements_ply_path) if planes_with_placements_ply_path else None,
            "pointcloud_projection_2d_image": None,
        },
        "summary": {
            "num_layers": len(shelf_layers),
            "num_target_objects": len(target_objects),
            "num_empty_spaces": len(empty_spaces),
            "processed_top_layers": int(top_layers_to_process) if top_layers_to_process > 0 else len(shelf_layers),
            "layer_roi_expand_y_px": int(layer_roi_expand_y_px),
            "plane_method": str(plane_method),
            "num_global_planes": int(len(plane_candidates_for_match)),
            "num_layer_plane_hints": int(len(layer_plane_hints)),
        },
    }

    vis = draw_experiment5_image(image, cabinet_bbox, shelf_layers, target_objects, empty_spaces, plane_segments=None)
    vis_path = os.path.join(output_dir, "experiment5_annotated.jpg")
    pc2d_vis_path = os.path.join(output_dir, "pointcloud_projection_2d.jpg")
    json_path = os.path.join(output_dir, "experiment5_results.json")
    cv2.imwrite(vis_path, vis)
    if surfaces_3d:
        pc2d_vis = draw_3d_overlay(image, surfaces_3d, placements_for_overlay, detector.camera_intrinsics)
        cv2.imwrite(pc2d_vis_path, pc2d_vis)
        result["outputs"]["pointcloud_projection_2d_image"] = os.path.abspath(pc2d_vis_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[CabinetShelfDetector][Exp5] 可视化已保存: {vis_path}")
    if result["outputs"]["pointcloud_projection_2d_image"] is not None:
        print(f"[CabinetShelfDetector][Exp5] 点云2D投影已保存: {pc2d_vis_path}")
    print(f"[CabinetShelfDetector][Exp5] JSON 已保存: {json_path}")
    print("[CabinetShelfDetector][Exp5] 3D Coordinates (meters):")
    for obj in target_objects:
        if obj.get("anchor_3d_m") is not None:
            xyz = obj["anchor_3d_m"]
            print(f"  Object {obj['name']} L{obj['layer']}: ({xyz['x']:.4f}, {xyz['y']:.4f}, {xyz['z']:.4f})")
    for i, es in enumerate(empty_spaces):
        xyz = es["placement_point_3d_m"]
        print(f"  EmptySpace#{i+1} L{es['layer']}: ({xyz['x']:.4f}, {xyz['y']:.4f}, {xyz['z']:.4f})")
    return result


