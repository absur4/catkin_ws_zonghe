"""Visualization / drawing functions for cabinet shelf detection."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

try:
    import open3d as o3d
except ImportError:
    o3d = None


# 每层对应的可视化颜色 (BGR)
LAYER_COLORS = [
    (0, 0, 255),    # 红
    (0, 165, 255),  # 橙
    (0, 255, 255),  # 黄
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (255, 0, 255),  # 紫
    (128, 128, 0),  # 青
    (0, 128, 255),  # 深橙
]


def draw_shelf_lines_image(
    image: np.ndarray,
    cabinet_bbox: tuple,
    shelf_y: List[int],
    shelf_layers: List[Dict],
) -> np.ndarray:
    """绘制中间结果图：仅柜子框 + 层板线 + 层编号"""
    vis = image.copy()
    x1, y1, x2, y2 = cabinet_bbox

    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(
        vis, "Cabinet", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
    )

    for sy in shelf_y:
        cv2.line(vis, (x1, sy), (x2, sy), (0, 255, 0), 2)

    for layer_info in shelf_layers:
        layer_num = layer_info["layer"]
        ly_mid = (layer_info["y_top"] + layer_info["y_bottom"]) // 2
        label_text = f"L{layer_num}"
        cv2.putText(
            vis, label_text, (x1 + 5, ly_mid + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    return vis


def draw_annotated_image(
    image: np.ndarray,
    cabinet_bbox: tuple,
    shelf_y: List[int],
    shelf_layers: List[Dict],
    objects_info: List[Dict],
) -> np.ndarray:
    """绘制标注图像：柜子框、层板线、层编号、物品框+层标签"""
    vis = image.copy()
    x1, y1, x2, y2 = cabinet_bbox

    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(
        vis, "Cabinet", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
    )

    for sy in shelf_y:
        cv2.line(vis, (x1, sy), (x2, sy), (0, 255, 0), 2)

    for layer_info in shelf_layers:
        layer_num = layer_info["layer"]
        ly_mid = (layer_info["y_top"] + layer_info["y_bottom"]) // 2
        label_text = f"L{layer_num}"
        cv2.putText(
            vis, label_text, (x1 + 5, ly_mid + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    for obj in objects_info:
        layer = obj["shelf_layer"]
        color = LAYER_COLORS[(layer - 1) % len(LAYER_COLORS)] if layer > 0 else (128, 128, 128)
        bx1, by1, bx2, by2 = obj["bbox"]

        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)

        label = f"{obj['class_name']} L{layer} ({obj['confidence']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)
        cv2.putText(
            vis, label, (bx1 + 2, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    return vis


def draw_experiment5_image(
    image: np.ndarray,
    cabinet_bbox: Tuple[int, int, int, int],
    shelf_layers: List[Dict],
    target_objects: List[Dict],
    empty_spaces: List[Dict],
    plane_segments: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """Experiment-5 visualization: English labels, target objects + empty spaces."""
    vis = image.copy()
    x1, y1, x2, y2 = cabinet_bbox
    main_color = (255, 0, 0)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        vis, "Cabinet",
        (x1 + 4, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
    )

    for layer in shelf_layers:
        ly1 = int(layer["y_top"])
        ly2 = int(layer["y_bottom"])
        lid = int(layer["layer"])
        cv2.rectangle(vis, (x1, ly1), (x2, ly2), (0, 0, 255), 2)
        cv2.putText(
            vis, f"Layer {lid}",
            (x1 + 6, min(max(18, ly1 + 20), vis.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    if plane_segments:
        alpha = 0.35
        for seg in plane_segments:
            sx1, sy1, sx2, sy2 = seg["roi"]
            mask = seg["mask"].astype(bool)
            if mask.size == 0 or not np.any(mask):
                continue
            layer_id = int(seg["layer"])
            plane_color = LAYER_COLORS[(layer_id - 1) % len(LAYER_COLORS)]

            roi = vis[sy1:sy2, sx1:sx2]
            colored = np.zeros_like(roi)
            colored[:, :] = plane_color
            roi[mask] = (
                (1.0 - alpha) * roi[mask].astype(np.float32)
                + alpha * colored[mask].astype(np.float32)
            ).astype(np.uint8)
            vis[sy1:sy2, sx1:sx2] = roi

            mask_u8 = seg["mask"].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                cnt_global = cnt.copy()
                cnt_global[:, 0, 0] += sx1
                cnt_global[:, 0, 1] += sy1
                cv2.drawContours(vis, [cnt_global], -1, plane_color, 2)

            y_val = float(seg.get("plane_y_m", 0.0))
            cv2.putText(
                vis,
                f"HMS Plane L{layer_id} y={y_val:.3f}m",
                (sx1 + 6, min(sy2 - 6, sy1 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, plane_color, 2,
            )

    for obj in target_objects:
        bx1, by1, bx2, by2 = obj["bbox"]
        obj_color = (255, 255, 255) if obj["name"] == "Milk" else main_color
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), obj_color, 2)

        label = f"{obj['name']} {obj['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        y_text = max(0, by1 - th - 8)
        cv2.rectangle(vis, (bx1, y_text), (bx1 + tw + 4, by1), obj_color, -1)
        cv2.putText(
            vis, label,
            (bx1 + 2, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (0, 0, 0) if obj["name"] == "Milk" else (255, 255, 255), 1,
        )

        anchor = obj.get("anchor_pixel")
        if anchor is not None:
            cv2.circle(vis, (int(anchor["u"]), int(anchor["v"])), 5, obj_color, -1)

    for es in empty_spaces:
        layer_id = es["layer"]
        if "bbox" in es:
            ex1, ey1, ex2, ey2 = es["bbox"]
            cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), main_color, 2)
            label = f"Empty Space L{layer_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            y_text = max(0, ey1 - th - 8)
            cv2.rectangle(vis, (ex1, y_text), (ex1 + tw + 4, ey1), main_color, -1)
            cv2.putText(
                vis, label,
                (ex1 + 2, ey1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
            )

        place_pt = es.get("placement_point_pixel")
        if place_pt:
            u = int(place_pt["u"])
            v = int(place_pt["v"])
            cv2.circle(vis, (u, v), 6, main_color, -1)
            cv2.putText(
                vis, f"Empty L{layer_id}",
                (u + 8, max(12, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, main_color, 2,
            )

    return vis


def draw_3d_overlay(
    image: np.ndarray,
    surfaces,
    placements,
    intrinsics: Dict[str, float],
) -> np.ndarray:
    """在 RGB 图上叠加 3D 平面点投影 + 放置点标记"""
    try:
        from .pointcloud_geometry import project_points_to_image
    except ImportError:
        from pointcloud_geometry import project_points_to_image

    vis = image.copy()
    img_h, img_w = vis.shape[:2]

    for i, surface in enumerate(surfaces):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        pts = surface.inlier_points

        step = max(1, len(pts) // 5000)
        sampled = pts[::step]

        uv = project_points_to_image(sampled, intrinsics, (img_h, img_w))
        for u, v in uv:
            if u >= 0 and v >= 0:
                cv2.circle(vis, (int(u), int(v)), 1, color, -1)

        valid_uv = uv[uv[:, 0] >= 0]
        if len(valid_uv) > 0:
            mu = int(np.median(valid_uv[:, 0]))
            mv = int(np.median(valid_uv[:, 1]))
            label = f"S{i} h={surface.height_y:.3f}m"
            cv2.putText(vis, label, (mu, mv - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for pp in placements:
        uv = project_points_to_image(
            pp.position_3d, intrinsics, (img_h, img_w)
        )
        u, v = uv[0]
        if u >= 0 and v >= 0:
            u, v = int(u), int(v)
            size = 12
            cv2.line(vis, (u - size, v), (u + size, v), (0, 255, 0), 2)
            cv2.line(vis, (u, v - size), (u, v + size), (0, 255, 0), 2)
            cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)
            label = f"P{pp.surface_index} {pp.free_area_m2:.3f}m2"
            cv2.putText(vis, label, (u + 8, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return vis


def save_colored_plane_cloud(surfaces, save_path: str) -> bool:
    """保存仅平面点云（与 hms_plane_from_rgbd.py 风格一致）。"""
    if o3d is None:
        return False

    all_pts = []
    all_cols = []
    for i, s in enumerate(surfaces):
        pts = np.asarray(s.inlier_points, dtype=np.float64)
        if pts.size == 0:
            continue
        color_bgr = np.array(LAYER_COLORS[i % len(LAYER_COLORS)], dtype=np.float64) / 255.0
        color_rgb = color_bgr[::-1]
        cols = np.tile(color_rgb.reshape(1, 3), (pts.shape[0], 1))
        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        return False

    pts_cat = np.concatenate(all_pts, axis=0)
    col_cat = np.concatenate(all_cols, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cat)
    pcd.colors = o3d.utility.Vector3dVector(col_cat)
    return bool(o3d.io.write_point_cloud(save_path, pcd))


def save_plane_cloud_with_placements(
    surfaces,
    placements,
    save_path: str,
    marker_radius_m: float = 0.012,
    marker_points: int = 180,
) -> bool:
    """保存平面+放置点点云（放置点用绿色小球点簇）。"""
    if o3d is None:
        return False

    all_pts = []
    all_cols = []
    for i, s in enumerate(surfaces):
        pts = np.asarray(s.inlier_points, dtype=np.float64)
        if pts.size == 0:
            continue
        color_bgr = np.array(LAYER_COLORS[i % len(LAYER_COLORS)], dtype=np.float64) / 255.0
        color_rgb = color_bgr[::-1]
        cols = np.tile(color_rgb.reshape(1, 3), (pts.shape[0], 1))
        all_pts.append(pts)
        all_cols.append(cols)

    n = max(32, int(marker_points))
    idx = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (idx + 0.5) / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * (idx + 0.5)
    unit = np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], axis=1)

    for pp in placements:
        center = np.asarray(pp.position_3d, dtype=np.float64).reshape(1, 3)
        marker = center + float(marker_radius_m) * unit
        cols = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64).reshape(1, 3), (marker.shape[0], 1))
        all_pts.append(marker)
        all_cols.append(cols)

    if not all_pts:
        return False

    pts_cat = np.concatenate(all_pts, axis=0)
    col_cat = np.concatenate(all_cols, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cat)
    pcd.colors = o3d.utility.Vector3dVector(col_cat)
    return bool(o3d.io.write_point_cloud(save_path, pcd))


def visualize_3d(all_points, surfaces, placements):
    """Open3D 3D 可视化：灰色原始点云 + 彩色平面 + 绿色球放置点"""
    try:
        import open3d as o3d
    except ImportError:
        print("  Open3D 不可用，跳过 3D 可视化")
        return

    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(pcd)

    colors_rgb = [
        [1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [0.5, 0, 0.5],
    ]
    for i, surface in enumerate(surfaces):
        pcd_s = o3d.geometry.PointCloud()
        pcd_s.points = o3d.utility.Vector3dVector(surface.inlier_points)
        c = colors_rgb[i % len(colors_rgb)]
        pcd_s.paint_uniform_color(c)
        geometries.append(pcd_s)

    for pp in placements:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(pp.position_3d)
        sphere.paint_uniform_color([0, 1, 0])
        geometries.append(sphere)

    o3d.visualization.draw_geometries(
        geometries, window_name="3D Shelf Detection",
        width=1280, height=720,
    )
