"""Empty space detection and placement point calculation."""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from .plane_estimation import (
    estimate_layer_height_hms,
    estimate_layer_height_robust_fusion,
)


def find_largest_free_rectangle(free_mask: np.ndarray) -> Tuple[int, int, int, int, int]:
    """
    在二值空闲图上找最大矩形。

    Returns:
        (r_start, c_start, r_end, c_end, area_cells)
    """
    if free_mask.size == 0:
        return (0, 0, 0, 0, 0)

    rows, cols = free_mask.shape
    histogram = np.zeros(cols, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, 0, 0)

    for r in range(rows):
        for c in range(cols):
            if free_mask[r, c]:
                histogram[c] += 1
            else:
                histogram[c] = 0

        stack = []
        for c in range(cols + 1):
            h = histogram[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                sc, sh = stack.pop()
                area = sh * (c - sc)
                if area > best_area:
                    best_area = area
                    best_rect = (r - sh + 1, sc, r + 1, c)
                start = sc
            stack.append((start, h))

    r_start, c_start, r_end, c_end = best_rect
    return (r_start, c_start, r_end, c_end, best_area)


def find_top_k_free_rectangles(
    free_mask: np.ndarray,
    k: int = 3,
    min_area_cells: int = 10,
) -> List[Tuple[int, int, int, int, int]]:
    mask = free_mask.copy()
    rects = []
    for _ in range(k):
        rect = find_largest_free_rectangle(mask)
        r_start, c_start, r_end, c_end, area = rect
        if area < min_area_cells:
            break
        rects.append(rect)
        mask[r_start:r_end, c_start:c_end] = False
    return rects


def uv_to_xz_on_plane(
    u: np.ndarray,
    v: np.ndarray,
    plane_y_m: float,
    camera_intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素 (u,v) 投影到水平平面 y=plane_y_m，返回 (N,2) 的 xz 和有效掩码。
    """
    intr = camera_intrinsics
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    u = u.astype(np.float64)
    v = v.astype(np.float64)
    denom = v - cy
    valid = np.abs(denom) > 1e-6

    z = np.zeros_like(u, dtype=np.float64)
    z[valid] = plane_y_m * fy / denom[valid]
    valid &= (z > 0.05) & (z < 8.0)

    x = np.zeros_like(u, dtype=np.float64)
    x[valid] = (u[valid] - cx) * z[valid] / fx

    xz = np.stack([x, z], axis=1)
    return xz, valid


def select_placement_point_from_3d_clearance(
    depth_roi: np.ndarray,
    occ: np.ndarray,
    layer_plane_y: Optional[float],
    layer_plane_mask: Optional[np.ndarray],
    rect_local: Tuple[int, int, int, int],
    roi_origin: Tuple[int, int],
    camera_intrinsics: Dict[str, float],
) -> Optional[Tuple[int, int, float, float, float, float]]:
    """
    在空位矩形内选择 3D 引导放置点（最大化与障碍/边界的 3D 平面距离）。

    Returns:
        (u, v, clearance_m, x_m, y_m, z_m) 或 None
    """
    r_start, c_start, r_end, c_end = rect_local
    ox, oy = roi_origin
    if r_end <= r_start or c_end <= c_start:
        return None

    rect_occ = occ[r_start:r_end, c_start:c_end]
    rect_depth = depth_roi[r_start:r_end, c_start:c_end]
    rect_plane: Optional[np.ndarray] = None

    if layer_plane_mask is not None:
        rect_plane = layer_plane_mask[r_start:r_end, c_start:c_end].astype(bool)
        cand_mask = rect_plane & (~rect_occ)
    else:
        cand_mask = (~rect_occ) & (rect_depth > 1e-6)

    rr, cc = np.where(cand_mask)
    if rr.size == 0:
        return None

    max_candidates = 3000
    if rr.size > max_candidates:
        step = int(np.ceil(rr.size / max_candidates))
        rr = rr[::step]
        cc = cc[::step]

    r_cand = rr.copy()
    c_cand = cc.copy()
    u_cand = (ox + c_start + cc).astype(np.float64)
    v_cand = (oy + r_start + rr).astype(np.float64)

    intr = camera_intrinsics
    if layer_plane_y is not None:
        xz_cand, valid_cand = uv_to_xz_on_plane(u_cand, v_cand, layer_plane_y, camera_intrinsics)
        if not np.any(valid_cand):
            return None
        r_cand = r_cand[valid_cand]
        c_cand = c_cand[valid_cand]
        u_cand = u_cand[valid_cand]
        v_cand = v_cand[valid_cand]
        xz_cand = xz_cand[valid_cand]
        y_cand = np.full((len(xz_cand),), layer_plane_y, dtype=np.float64)
    else:
        z_cand = rect_depth[rr, cc].astype(np.float64)
        valid_cand = z_cand > 1e-6
        if not np.any(valid_cand):
            return None
        r_cand = r_cand[valid_cand]
        c_cand = c_cand[valid_cand]
        u_cand = u_cand[valid_cand]
        v_cand = v_cand[valid_cand]
        z_cand = z_cand[valid_cand]
        x_cand = (u_cand - intr["cx"]) * z_cand / intr["fx"]
        y_cand = (v_cand - intr["cy"]) * z_cand / intr["fy"]
        xz_cand = np.stack([x_cand, z_cand], axis=1)

    # 障碍点（3D xz）
    obs_mask = occ & (depth_roi > 1e-6)
    ro, co = np.where(obs_mask)
    if ro.size > 5000:
        step = int(np.ceil(ro.size / 5000))
        ro = ro[::step]
        co = co[::step]

    if ro.size > 0:
        z_obs = depth_roi[ro, co].astype(np.float64)
        u_obs = (ox + co).astype(np.float64)
        x_obs = (u_obs - intr["cx"]) * z_obs / intr["fx"]
        xz_obs = np.stack([x_obs, z_obs], axis=1)
    else:
        xz_obs = np.zeros((0, 2), dtype=np.float64)

    # 矩形边界作为不可越界约束（投影到同一平面）
    edge_step = 2
    top_u = np.arange(c_start, c_end, edge_step, dtype=np.int32)
    top_v = np.full_like(top_u, r_start)
    bot_u = np.arange(c_start, c_end, edge_step, dtype=np.int32)
    bot_v = np.full_like(bot_u, r_end - 1)
    lef_v = np.arange(r_start, r_end, edge_step, dtype=np.int32)
    lef_u = np.full_like(lef_v, c_start)
    rig_v = np.arange(r_start, r_end, edge_step, dtype=np.int32)
    rig_u = np.full_like(rig_v, c_end - 1)
    bu = np.concatenate([top_u, bot_u, lef_u, rig_u]).astype(np.float64) + ox
    bv = np.concatenate([top_v, bot_v, lef_v, rig_v]).astype(np.float64) + oy

    if layer_plane_y is not None:
        xz_border, valid_b = uv_to_xz_on_plane(bu, bv, layer_plane_y, camera_intrinsics)
        xz_border = xz_border[valid_b]
    else:
        xz_border = np.zeros((0, 2), dtype=np.float64)

    obstacles = xz_obs
    if xz_border.size > 0:
        obstacles = np.vstack([obstacles, xz_border]) if obstacles.size > 0 else xz_border
    if obstacles.size == 0:
        return None

    # 最近邻距离（3D 平面距离）
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(obstacles)
        dists, _ = tree.query(xz_cand, k=1, workers=-1)
    except Exception:
        dists = np.full((len(xz_cand),), np.inf, dtype=np.float64)
        chunk = 512
        for i in range(0, len(obstacles), chunk):
            blk = obstacles[i:i + chunk]
            diff = xz_cand[:, None, :] - blk[None, :, :]
            dd = np.sqrt(np.sum(diff * diff, axis=2))
            dists = np.minimum(dists, np.min(dd, axis=1))

    if dists.size == 0:
        return None

    order = np.argsort(dists)[::-1]
    best_i = int(order[0])
    max_check = min(len(order), 120)
    for oi in order[:max_check]:
        idx = int(oi)
        r = int(r_cand[idx])
        c = int(c_cand[idx])
        rect_h = rect_occ.shape[0]
        rect_w = rect_occ.shape[1]

        border_dist = min(
            r,
            (rect_h - 1) - r,
            c,
            (rect_w - 1) - c,
        )
        if border_dist < 2:
            continue

        rs = max(0, r - 3)
        re = min(rect_occ.shape[0], r + 4)
        cs = max(0, c - 3)
        ce = min(rect_occ.shape[1], c + 4)
        if re <= rs or ce <= cs:
            continue

        local_occ = rect_occ[rs:re, cs:ce]
        if float(np.mean(local_occ)) > 0.20:
            continue

        if rect_plane is not None:
            local_plane = rect_plane[rs:re, cs:ce]
            if float(np.mean(local_plane)) < 0.35:
                continue
        else:
            local_valid = rect_depth[rs:re, cs:ce] > 1e-6
            if float(np.mean(local_valid)) < 0.55:
                continue

        best_i = idx
        break

    clearance_m = float(dists[best_i])
    u_best = int(round(float(u_cand[best_i])))
    v_best = int(round(float(v_cand[best_i])))

    if layer_plane_y is not None:
        x_best = float(xz_cand[best_i, 0])
        z_best = float(xz_cand[best_i, 1])
        y_best = float(layer_plane_y)
    else:
        z_best = float(xz_cand[best_i, 1])
        x_best = float(xz_cand[best_i, 0])
        y_best = float(y_cand[best_i])

    return u_best, v_best, clearance_m, x_best, y_best, z_best


def find_empty_spaces_by_layer(
    depth_m: np.ndarray,
    cabinet_bbox: Tuple[int, int, int, int],
    shelf_layers: List[Dict],
    obstacle_boxes: List[Tuple[int, int, int, int]],
    camera_intrinsics: Dict[str, float],
    min_width_m: float = 0.08,
    min_height_m: float = 0.05,
    min_clearance_m: float = 0.10,
    occupied_margin_px: int = 6,
    max_spaces_per_layer: int = 2,
    min_area_cells: int = 120,
    layer_boundary_margin_px: int = 8,
    min_place_clearance_m: float = 0.03,
    layer_roi_expand_y_px: int = 0,
    plane_method: str = "robust_fusion",
    plane_inlier_tol_m: float = 0.012,
    layer_plane_hints: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按层检测空位，并返回 HMS 风格层平面分割掩码（用于可视化）。"""
    intr = camera_intrinsics
    fx = intr["fx"]
    fy = intr["fy"]
    img_h, img_w = depth_m.shape[:2]
    cx1, cy1, cx2, cy2 = cabinet_bbox
    cx1 = max(0, min(img_w - 1, cx1))
    cy1 = max(0, min(img_h - 1, cy1))
    cx2 = max(0, min(img_w, cx2))
    cy2 = max(0, min(img_h, cy2))

    empty_spaces = []
    plane_segments = []
    for layer in shelf_layers:
        base_ly1 = max(cy1, int(layer["y_top"]))
        base_ly2 = min(cy2, int(layer["y_bottom"]))
        expand_y = max(0, int(layer_roi_expand_y_px))
        ly1 = max(cy1, base_ly1 - expand_y)
        ly2 = min(cy2, base_ly2 + expand_y)
        if ly2 - ly1 < 8 or cx2 - cx1 < 8:
            continue

        roi_h = ly2 - ly1
        roi_w = cx2 - cx1
        occ = np.zeros((roi_h, roi_w), dtype=bool)

        depth_roi = depth_m[ly1:ly2, cx1:cx2]
        occ |= depth_roi <= 1e-6

        for bx1, by1, bx2, by2 in obstacle_boxes:
            ox1 = max(cx1, bx1)
            ox2 = min(cx2, bx2)
            oy1 = max(ly1, by1)
            oy2 = min(ly2, by2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            occ[oy1 - ly1:oy2 - ly1, ox1 - cx1:ox2 - cx1] = True

        if occupied_margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * occupied_margin_px + 1, 2 * occupied_margin_px + 1),
            )
            occ = cv2.dilate(occ.astype(np.uint8), kernel, iterations=1) > 0

        if layer_boundary_margin_px > 0 and roi_h > 2 * layer_boundary_margin_px:
            occ[:layer_boundary_margin_px, :] = True
            occ[-layer_boundary_margin_px:, :] = True

        free_mask = ~occ
        if not np.any(free_mask):
            continue

        rects = find_top_k_free_rectangles(
            free_mask,
            k=max_spaces_per_layer,
            min_area_cells=max(1, min_area_cells),
        )

        layer_clearance_candidates = depth_roi[depth_roi > 1e-6]
        if layer_clearance_candidates.size == 0:
            continue
        layer_z = float(np.median(layer_clearance_candidates))
        layer_clearance_m = (ly2 - ly1) * layer_z / max(fy, 1e-6)
        if layer_clearance_m < min_clearance_m:
            continue

        layer_id = int(layer["layer"])
        layer_plane_y: Optional[float] = None
        layer_plane_mask: Optional[np.ndarray] = None
        plane_source = "none"

        if layer_plane_hints is not None and layer_id in layer_plane_hints:
            hint = layer_plane_hints[layer_id]
            layer_plane_y = float(hint["plane_y_m"])
            plane_source = str(hint.get("source", "global_hint"))
            vv = np.arange(ly1, ly2, dtype=np.float32)[:, None]
            y_map = (vv - intr["cy"]) * depth_roi / max(intr["fy"], 1e-6)
            valid_plane = (depth_roi > 1e-6) & (~occ)
            layer_plane_mask = valid_plane & (np.abs(y_map - layer_plane_y) <= plane_inlier_tol_m)

        if (
            layer_plane_y is None
            or layer_plane_mask is None
            or int(np.sum(layer_plane_mask)) < 30
        ):
            plane_method_norm = str(plane_method).strip().lower()
            if plane_method_norm == "hms":
                layer_plane_y, layer_plane_mask = estimate_layer_height_hms(
                    depth_m=depth_m,
                    x1=cx1,
                    y1=ly1,
                    x2=cx2,
                    y2=ly2,
                    camera_intrinsics=camera_intrinsics,
                    occupied_mask=occ,
                    inlier_tol_m=plane_inlier_tol_m,
                )
                plane_source = "hms"
            elif plane_method_norm == "robust":
                layer_plane_y, layer_plane_mask, plane_source = estimate_layer_height_robust_fusion(
                    depth_m=depth_m,
                    x1=cx1,
                    y1=ly1,
                    x2=cx2,
                    y2=ly2,
                    camera_intrinsics=camera_intrinsics,
                    occupied_mask=occ,
                    inlier_tol_m=plane_inlier_tol_m,
                    fuse_with_hms=False,
                )
            else:
                layer_plane_y, layer_plane_mask, plane_source = estimate_layer_height_robust_fusion(
                    depth_m=depth_m,
                    x1=cx1,
                    y1=ly1,
                    x2=cx2,
                    y2=ly2,
                    camera_intrinsics=camera_intrinsics,
                    occupied_mask=occ,
                    inlier_tol_m=plane_inlier_tol_m,
                    fuse_with_hms=True,
                )

        if layer_plane_y is not None and layer_plane_mask is not None and np.any(layer_plane_mask):
            plane_segments.append({
                "layer": int(layer["layer"]),
                "roi": [int(cx1), int(ly1), int(cx2), int(ly2)],
                "mask": layer_plane_mask.astype(np.uint8),
                "plane_y_m": float(layer_plane_y),
                "num_inlier_pixels": int(np.sum(layer_plane_mask)),
                "source": plane_source,
            })

        for r_start, c_start, r_end, c_end, area_cells in rects:
            ex1 = cx1 + c_start
            ex2 = cx1 + c_end
            ey1 = ly1 + r_start
            ey2 = ly1 + r_end

            if ex2 - ex1 < 4 or ey2 - ey1 < 4:
                continue

            depth_patch = depth_m[ey1:ey2, ex1:ex2]
            valid = depth_patch[depth_patch > 1e-6]
            if valid.size < 20:
                continue

            z_m = float(np.median(valid))
            width_m = (ex2 - ex1) * z_m / max(fx, 1e-6)
            height_m = (ey2 - ey1) * z_m / max(fy, 1e-6)

            if width_m < min_width_m or height_m < min_height_m:
                continue

            guidance = select_placement_point_from_3d_clearance(
                depth_roi=depth_roi,
                occ=occ,
                layer_plane_y=layer_plane_y,
                layer_plane_mask=layer_plane_mask,
                rect_local=(r_start, c_start, r_end, c_end),
                roi_origin=(cx1, ly1),
                camera_intrinsics=camera_intrinsics,
            )
            if guidance is None:
                continue
            cu, cv, clearance_m, x_m, y_m, z_m = guidance
            if clearance_m < min_place_clearance_m:
                continue

            empty_spaces.append({
                "layer": int(layer["layer"]),
                "bbox": [int(ex1), int(ey1), int(ex2), int(ey2)],
                "label": "Empty Space",
                "area_pixels": int(area_cells),
                "size_estimate_m": {
                    "width_m": round(width_m, 4),
                    "height_m": round(height_m, 4),
                    "clearance_m": round(layer_clearance_m, 4),
                },
                "placement_point_pixel": {"u": int(cu), "v": int(cv)},
                "placement_point_3d_m": {"x": round(x_m, 4), "y": round(y_m, 4), "z": round(z_m, 4)},
                "placement_clearance_m": round(clearance_m, 4),
                "placement_method": "hms_3d_max_clearance",
            })

    empty_spaces.sort(key=lambda e: (e["layer"], -e["area_pixels"]))
    return empty_spaces, plane_segments
