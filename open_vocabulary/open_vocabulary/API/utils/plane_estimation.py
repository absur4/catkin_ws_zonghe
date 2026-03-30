"""Plane height estimation: HMS, Robust fusion, and related utilities."""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union

try:
    import open3d as o3d
except ImportError:
    o3d = None


def fit_plane_tls(
    points_xyz: np.ndarray,
    up_axis: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """最小二乘拟合平面，返回 (normal, d)，满足 normal·x + d = 0。"""
    if points_xyz.shape[0] < 3:
        n = up_axis.astype(np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        d = -float(np.mean(points_xyz @ n)) if points_xyz.shape[0] > 0 else 0.0
        return n, d

    center = np.mean(points_xyz, axis=0)
    x = points_xyz - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    if float(np.dot(n, up_axis)) < 0.0:
        n = -n
    d = -float(np.dot(n, center))
    return n, d


def estimate_layer_height_hms(
    depth_m: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    camera_intrinsics: Dict[str, float],
    occupied_mask: Optional[np.ndarray] = None,
    sample_step: int = 3,
    iterations: int = 300,
    inlier_tol_m: float = 0.012,
    min_inliers: int = 120,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    参考 HMS-RANSAC：仅假设"水平面"，在层 ROI 中估计平面高度 y（米）。
    为避免选到隔板下表面，这里采用"两阶段候选 + 底部优先评分"：
    1) 先在层 ROI 下半区域寻找候选平面；
    2) 若下半区域不稳定，再回退到全 ROI；
    3) 候选评分综合考虑垂直位置（更靠下优先）、支持度与横向覆盖。
    """
    intr = camera_intrinsics
    fy = intr["fy"]
    cy = intr["cy"]

    ys = []
    vs = []
    us = []
    h, w = depth_m.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None

    for v in range(y1, y2, sample_step):
        for u in range(x1, x2, sample_step):
            if occupied_mask is not None:
                if occupied_mask[v - y1, u - x1]:
                    continue
            z = float(depth_m[v, u])
            if z <= 1e-6:
                continue
            y_m = (v - cy) * z / max(fy, 1e-6)
            ys.append(y_m)
            vs.append(float(v))
            us.append(float(u))

    if len(ys) < 30:
        return None, None
    ys = np.array(ys, dtype=np.float32)
    vs = np.array(vs, dtype=np.float32)
    us = np.array(us, dtype=np.float32)

    def _build_inlier_mask(plane_y: float) -> np.ndarray:
        depth_roi = depth_m[y1:y2, x1:x2]
        vv = np.arange(y1, y2, dtype=np.float32)[:, None]
        y_map = (vv - cy) * depth_roi / max(fy, 1e-6)
        valid = depth_roi > 1e-6
        if occupied_mask is not None:
            valid &= ~occupied_mask
        return valid & (np.abs(y_map - plane_y) <= inlier_tol_m)

    roi_h = max(1.0, float(y2 - y1))
    roi_w = max(1.0, float(x2 - x1))

    def _pick_plane(
        ys_sub: np.ndarray,
        vs_sub: np.ndarray,
        us_sub: np.ndarray,
        prefer_lower: bool,
    ) -> Optional[float]:
        if ys_sub.size < 35:
            return None

        bin_size = max(0.004, inlier_tol_m * 0.5)
        y_min = float(np.min(ys_sub))
        y_max = float(np.max(ys_sub))
        if y_max - y_min < bin_size * 2:
            return float(np.median(ys_sub))

        bins = np.arange(y_min, y_max + bin_size, bin_size, dtype=np.float32)
        hist, edges = np.histogram(ys_sub, bins=bins)
        centers = (edges[:-1] + edges[1:]) * 0.5

        peak_indices = []
        for i in range(len(hist)):
            left = hist[i - 1] if i > 0 else -1
            right = hist[i + 1] if i < len(hist) - 1 else -1
            if hist[i] > 0 and hist[i] >= left and hist[i] >= right:
                peak_indices.append(i)
        if not peak_indices:
            peak_indices = [int(np.argmax(hist))]

        candidates = []
        min_base_support = max(20, min_inliers // 5)
        for pi in peak_indices:
            y0 = float(centers[pi])
            inlier = np.abs(ys_sub - y0) <= inlier_tol_m
            cnt = int(np.sum(inlier))
            if cnt < min_base_support:
                continue

            y_med = float(np.median(ys_sub[inlier]))
            v_vals = vs_sub[inlier]
            u_vals = us_sub[inlier]
            v_med = float(np.median(v_vals))

            u_q5 = float(np.percentile(u_vals, 5))
            u_q95 = float(np.percentile(u_vals, 95))
            u_cov = max(0.0, min(1.0, (u_q95 - u_q5) / roi_w))

            v_norm_vals = np.clip((v_vals - float(y1)) / roi_h, 0.0, 1.0)
            v_norm = float(np.clip((v_med - float(y1)) / roi_h, 0.0, 1.0))
            lower_frac = float(np.mean(v_norm_vals >= 0.55))
            candidates.append(
                {
                    "cnt": cnt,
                    "y_med": y_med,
                    "v_norm": v_norm,
                    "u_cov": u_cov,
                    "lower_frac": lower_frac,
                }
            )

        if not candidates:
            return None

        max_cnt = max(c["cnt"] for c in candidates)
        min_cnt_keep = max(24, int(0.10 * max_cnt))
        filtered = [c for c in candidates if c["cnt"] >= min_cnt_keep]
        if not filtered:
            filtered = candidates

        if prefer_lower:
            lower_filtered = [
                c for c in filtered
                if (c["v_norm"] >= 0.40) or (c["lower_frac"] >= 0.30)
            ]
            if lower_filtered:
                filtered = lower_filtered
        else:
            lower_pref = [
                c for c in filtered
                if (c["v_norm"] >= 0.45) and (c["cnt"] >= max(24, int(0.08 * max_cnt)))
            ]
            if lower_pref:
                filtered = lower_pref

        best = None
        best_score = -1.0
        for c in filtered:
            cnt_norm = c["cnt"] / max(1.0, float(max_cnt))
            if prefer_lower:
                score = (
                    0.62 * c["v_norm"]
                    + 0.20 * cnt_norm
                    + 0.10 * c["u_cov"]
                    + 0.08 * c["lower_frac"]
                )
            else:
                score = (
                    0.60 * c["v_norm"]
                    + 0.24 * cnt_norm
                    + 0.10 * c["u_cov"]
                    + 0.06 * c["lower_frac"]
                )
            if score > best_score:
                best_score = score
                best = c

        if best is None:
            return None
        return float(best["y_med"])

    if len(ys) < min_inliers:
        plane_y = float(np.median(ys))
        inlier_mask = _build_inlier_mask(plane_y)
        return plane_y, inlier_mask

    # 两阶段：优先层下半区，失败再回退全层
    v_norm_all = np.clip((vs - float(y1)) / roi_h, 0.0, 1.0)
    lower_half_mask = v_norm_all >= 0.52
    plane_y = None
    if int(np.sum(lower_half_mask)) >= max(80, min_inliers // 2):
        plane_y = _pick_plane(
            ys_sub=ys[lower_half_mask],
            vs_sub=vs[lower_half_mask],
            us_sub=us[lower_half_mask],
            prefer_lower=True,
        )
    if plane_y is None:
        plane_y = _pick_plane(
            ys_sub=ys,
            vs_sub=vs,
            us_sub=us,
            prefer_lower=False,
        )
    if plane_y is None:
        plane_y = float(np.median(ys))

    inlier_mask = _build_inlier_mask(plane_y)

    return plane_y, inlier_mask


def estimate_layer_height_robust_fusion(
    depth_m: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    camera_intrinsics: Dict[str, float],
    occupied_mask: Optional[np.ndarray] = None,
    sample_step: int = 2,
    inlier_tol_m: float = 0.012,
    min_inliers: int = 120,
    robust_horizontal_angle_deg: float = 15.0,
    robust_normal_variance_deg: float = 25.0,
    robust_coplanarity_deg: float = 70.0,
    robust_outlier_ratio: float = 0.65,
    robust_min_plane_edge_m: float = 0.08,
    robust_min_points: int = 160,
    robust_knn: int = 30,
    fuse_with_hms: bool = True,
) -> Tuple[Optional[float], Optional[np.ndarray], str]:
    """
    Robust planar patches + HMS 融合层板高度估计。
    返回: (plane_y_m, inlier_mask, source)
        source ∈ {"robust", "hms", "robust+hms", "none"}
    """
    intr = camera_intrinsics
    fy = intr["fy"]
    cy = intr["cy"]
    fx = intr["fx"]
    cx = intr["cx"]
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    h, w = depth_m.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, "none"

    depth_roi = depth_m[y1:y2, x1:x2]
    valid_roi = depth_roi > 1e-6
    if occupied_mask is not None:
        valid_roi &= ~occupied_mask
    if int(np.sum(valid_roi)) < 50:
        return None, None, "none"

    vv_abs = np.arange(y1, y2, dtype=np.float32)[:, None]
    y_map = (vv_abs - cy) * depth_roi / max(fy, 1e-6)

    hms_cache: Dict[str, Any] = {"done": False, "y": None, "mask": None}

    def _get_hms() -> Tuple[Optional[float], Optional[np.ndarray]]:
        if not hms_cache["done"]:
            hy, hm = estimate_layer_height_hms(
                depth_m=depth_m,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                camera_intrinsics=camera_intrinsics,
                occupied_mask=occupied_mask,
                sample_step=max(2, sample_step),
                inlier_tol_m=inlier_tol_m,
                min_inliers=min_inliers,
            )
            hms_cache["done"] = True
            hms_cache["y"] = hy
            hms_cache["mask"] = hm
        return hms_cache["y"], hms_cache["mask"]

    candidates: List[Dict[str, Any]] = []
    if o3d is not None:
        step = max(1, int(sample_step))
        uu = np.arange(x1, x2, step, dtype=np.float32)
        vv = np.arange(y1, y2, step, dtype=np.float32)
        if uu.size > 0 and vv.size > 0:
            u_grid, v_grid = np.meshgrid(uu, vv)
            z_sub = depth_m[y1:y2:step, x1:x2:step]
            valid_sub = z_sub > 1e-6
            if occupied_mask is not None:
                valid_sub &= ~occupied_mask[::step, ::step]

            if np.any(valid_sub):
                u_sub = u_grid[valid_sub].astype(np.float64)
                v_sub = v_grid[valid_sub].astype(np.float64)
                z_sub = z_sub[valid_sub].astype(np.float64)
                x_sub = (u_sub - cx) * z_sub / max(fx, 1e-6)
                y_sub = (v_sub - cy) * z_sub / max(fy, 1e-6)
                pts = np.stack([x_sub, y_sub, z_sub], axis=1)

                if pts.shape[0] >= max(80, robust_min_points):
                    try:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=0.03, max_nn=30
                            )
                        )
                        try:
                            pcd.orient_normals_consistent_tangent_plane(20)
                        except Exception:
                            pass

                        oboxes = pcd.detect_planar_patches(
                            normal_variance_threshold_deg=float(robust_normal_variance_deg),
                            coplanarity_deg=float(robust_coplanarity_deg),
                            outlier_ratio=float(robust_outlier_ratio),
                            min_plane_edge_length=float(robust_min_plane_edge_m),
                            min_num_points=int(robust_min_points),
                            search_param=o3d.geometry.KDTreeSearchParamKNN(
                                knn=max(8, int(robust_knn))
                            ),
                        )
                        cos_th = float(np.cos(np.deg2rad(max(0.1, robust_horizontal_angle_deg))))
                        for obb in oboxes:
                            r = np.asarray(obb.R, dtype=np.float64)
                            if r.shape != (3, 3):
                                continue
                            n_obb = r[:, 2]
                            n_obb = n_obb / (np.linalg.norm(n_obb) + 1e-12)
                            if abs(float(np.dot(n_obb, up_axis))) < cos_th:
                                continue

                            ext = np.asarray(obb.extent, dtype=np.float64).copy()
                            if ext.shape[0] != 3:
                                continue
                            ext[2] = max(ext[2], float(3.0 * inlier_tol_m))
                            obb_fat = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, ext)
                            patch_pts = np.asarray(pcd.crop(obb_fat).points, dtype=np.float64)
                            if patch_pts.shape[0] < max(40, robust_min_points // 3):
                                continue

                            n_fit, d_fit = fit_plane_tls(patch_pts, up_axis=up_axis)
                            dist = np.abs(patch_pts @ n_fit + d_fit)
                            keep = dist <= max(0.008, 1.25 * inlier_tol_m)
                            inlier_pts = patch_pts[keep] if np.any(keep) else patch_pts
                            if inlier_pts.shape[0] < max(40, robust_min_points // 4):
                                continue

                            y_est = float(np.median(inlier_pts @ up_axis))
                            candidates.append({
                                "y": y_est,
                                "source": "robust",
                                "hint": 1.0,
                            })
                    except Exception:
                        pass

    if fuse_with_hms:
        hy, _ = _get_hms()
        if hy is not None:
            candidates.append({
                "y": float(hy),
                "source": "hms",
                "hint": 0.75,
            })

    if not candidates:
        hy, hm = _get_hms()
        if hy is None or hm is None:
            return None, None, "none"
        return hy, hm, "hms"

    # 合并高度接近的候选，避免重复评分
    merge_tol = max(0.006, inlier_tol_m * 1.5)
    candidates.sort(key=lambda c: c["y"])
    merged: List[Dict[str, Any]] = []
    for cand in candidates:
        if not merged:
            merged.append(cand)
            continue
        if abs(float(cand["y"]) - float(merged[-1]["y"])) <= merge_tol:
            if float(cand.get("hint", 0.0)) > float(merged[-1].get("hint", 0.0)):
                merged[-1] = cand
        else:
            merged.append(cand)

    scored: List[Dict[str, Any]] = []
    roi_h = max(1.0, float(y2 - y1))
    for cand in merged:
        y0 = float(cand["y"])
        inlier_mask = valid_roi & (np.abs(y_map - y0) <= inlier_tol_m)
        cnt = int(np.sum(inlier_mask))
        if cnt < 20:
            continue
        rows = np.where(inlier_mask)[0]
        if rows.size == 0:
            continue
        v_med = float(np.median(rows + y1))
        v_norm = float(np.clip((v_med - float(y1)) / roi_h, 0.0, 1.0))
        scored.append({
            "y": y0,
            "source": str(cand.get("source", "robust")),
            "hint": float(cand.get("hint", 0.0)),
            "cnt": cnt,
            "v_norm": v_norm,
            "mask": inlier_mask,
        })

    if not scored:
        hy, hm = _get_hms()
        if hy is None or hm is None:
            return None, None, "none"
        return hy, hm, "hms"

    max_cnt = max(s["cnt"] for s in scored)
    best = max(
        scored,
        key=lambda s: (
            0.58 * (s["cnt"] / max(1.0, float(max_cnt)))
            + 0.32 * s["v_norm"]
            + 0.10 * s["hint"],
            s["cnt"],
            s["y"],
        ),
    )
    return float(best["y"]), best["mask"].astype(bool), str(best["source"])


def depth_bbox_to_points(
    depth_m: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    camera_intrinsics: Dict[str, float],
    sample_step: int = 1,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.5,
) -> np.ndarray:
    """将 depth+bbox 转为相机坐标系点云 (N,3)。"""
    intr = camera_intrinsics
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    step = max(1, int(sample_step))
    uu = np.arange(x1, x2, step, dtype=np.float32)
    vv = np.arange(y1, y2, step, dtype=np.float32)
    if uu.size == 0 or vv.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    u_grid, v_grid = np.meshgrid(uu, vv)
    z = depth_m[y1:y2:step, x1:x2:step]
    valid = (z > min_depth_m) & (z < max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    x = (u - cx) * z / max(fx, 1e-6)
    y = (v - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=1)


def merge_plane_candidates_by_height(
    candidates: List[Dict[str, Any]],
    merge_tol_m: float = 0.03,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    merged: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: float(c["plane_y_m"])):
        if not merged:
            merged.append(cand)
            continue

        last = merged[-1]
        if abs(float(cand["plane_y_m"]) - float(last["plane_y_m"])) <= float(merge_tol_m):
            pts_last = last["points"]
            pts_new = cand["points"]
            pts = np.vstack([pts_last, pts_new]) if pts_last.size and pts_new.size else (pts_last if pts_last.size else pts_new)
            src = last["source"] if last["source"] == cand["source"] else "robust+hms"
            merged[-1] = {
                "plane_y_m": float(np.mean(pts[:, 1])) if pts.size > 0 else float(last["plane_y_m"]),
                "source": src,
                "points": pts,
            }
        else:
            merged.append(cand)
    return merged


def detect_cabinet_planes_from_depth(
    depth_m: np.ndarray,
    cabinet_bbox: Tuple[int, int, int, int],
    camera_intrinsics: Dict[str, float],
    method: str = "robust_fusion",
    return_filtered_cloud: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
    """
    复用 hms_plane_from_rgbd.py 的思路：
    - robust planar patches（默认）
    - 可选 HMS 补平面融合
    """
    if o3d is None:
        if return_filtered_cloud:
            return [], np.zeros((0, 3), dtype=np.float64)
        return []

    try:
        from .pointcloud_geometry import filter_horizontal_points, hms_ransac
    except ImportError:
        from pointcloud_geometry import filter_horizontal_points, hms_ransac

    points_roi = depth_bbox_to_points(depth_m, cabinet_bbox, camera_intrinsics, sample_step=1)
    if points_roi.shape[0] < 500:
        if return_filtered_cloud:
            return [], np.zeros((0, 3), dtype=np.float64)
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_roi)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points_clean = np.asarray(pcd.points, dtype=np.float64)
    if points_clean.shape[0] < 300:
        if return_filtered_cloud:
            return [], points_clean
        return []

    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    candidates: List[Dict[str, Any]] = []
    method_norm = str(method).strip().lower()

    horizontal_pts, _ = filter_horizontal_points(
        points_clean,
        pcd,
        up_axis=up_axis,
        normal_angle_threshold_deg=30.0,
        normal_search_radius=0.03,
        normal_max_nn=30,
    )

    def _run_hms_candidates() -> List[Dict[str, Any]]:
        if horizontal_pts.shape[0] < 50:
            return []
        surfaces_hms = hms_ransac(
            points=horizontal_pts,
            all_points=points_clean,
            up_axis=up_axis,
            min_iterations=300,
            angular_threshold_deg=10.0,
            inlier_distance=0.01,
            min_inliers=1500,
            duplicate_distance=0.03,
            refine_with_o3d=True,
        )
        out = []
        for s in surfaces_hms:
            out.append({
                "plane_y_m": float(s.height_y),
                "source": "hms",
                "points": np.asarray(s.inlier_points, dtype=np.float64),
            })
        return out

    if method_norm in ("robust", "robust_fusion"):
        pcd_r = o3d.geometry.PointCloud()
        pcd_r.points = o3d.utility.Vector3dVector(points_clean)
        pcd_r.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        try:
            pcd_r.orient_normals_consistent_tangent_plane(20)
        except Exception:
            pass

        try:
            oboxes = pcd_r.detect_planar_patches(
                normal_variance_threshold_deg=25.0,
                coplanarity_deg=70.0,
                outlier_ratio=0.65,
                min_plane_edge_length=0.10,
                min_num_points=300,
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
            )
        except Exception:
            oboxes = []

        cos_th = float(np.cos(np.deg2rad(15.0)))
        for obb in oboxes:
            r = np.asarray(obb.R, dtype=np.float64)
            if r.shape != (3, 3):
                continue
            n = r[:, 2]
            n = n / (np.linalg.norm(n) + 1e-12)
            if abs(float(np.dot(n, up_axis))) < cos_th:
                continue

            ext = np.asarray(obb.extent, dtype=np.float64).copy()
            if ext.shape[0] != 3:
                continue
            ext[2] = max(ext[2], 0.03)
            obb_fat = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, ext)
            patch_pts = np.asarray(pcd_r.crop(obb_fat).points, dtype=np.float64)
            if patch_pts.shape[0] < 120:
                continue

            n_fit, d_fit = fit_plane_tls(patch_pts, up_axis=up_axis)
            dist = np.abs(patch_pts @ n_fit + d_fit)
            keep = dist <= 0.015
            inlier_pts = patch_pts[keep] if np.any(keep) else patch_pts
            if inlier_pts.shape[0] < 100:
                continue
            candidates.append({
                "plane_y_m": float(np.mean(inlier_pts[:, 1])),
                "source": "robust",
                "points": inlier_pts,
            })

        if method_norm == "robust_fusion":
            hms_only = _run_hms_candidates()
            for hs in hms_only:
                has_near = any(
                    abs(float(hs["plane_y_m"]) - float(rs["plane_y_m"])) <= 0.06
                    for rs in candidates
                )
                if not has_near:
                    candidates.append(hs)

        if len(candidates) == 0:
            candidates = _run_hms_candidates()

    elif method_norm == "hms":
        candidates = _run_hms_candidates()

    merged = merge_plane_candidates_by_height(candidates, merge_tol_m=0.03)
    merged.sort(key=lambda c: float(c["plane_y_m"]))

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]
    for cand in merged:
        pts = cand["points"]
        if pts.size == 0:
            cand["v_median_px"] = -1.0
            cand["num_inliers"] = 0
            continue
        z = pts[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            cand["v_median_px"] = -1.0
            cand["num_inliers"] = int(pts.shape[0])
            continue
        v = (pts[valid, 1] * fy / z[valid]) + cy
        cand["v_median_px"] = float(np.median(v))
        cand["num_inliers"] = int(pts.shape[0])
    if return_filtered_cloud:
        return merged, points_clean
    return merged


def assign_global_planes_to_layers(
    planes: List[Dict[str, Any]],
    shelf_layers: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """将全柜平面按投影 y 中位值匹配到每层（每层最多匹配一个平面）。"""
    hints: Dict[int, Dict[str, Any]] = {}
    if not planes or not shelf_layers:
        return hints

    valid_planes = [p for p in planes if float(p.get("v_median_px", -1.0)) >= 0.0]
    if not valid_planes:
        return hints

    used = set()
    for layer in shelf_layers:
        layer_id = int(layer["layer"])
        mid = 0.5 * (float(layer["y_top"]) + float(layer["y_bottom"]))
        best_i = None
        best_dist = float("inf")
        for i, p in enumerate(valid_planes):
            if i in used:
                continue
            dist = abs(float(p["v_median_px"]) - mid)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i is not None:
            used.add(best_i)
            p = valid_planes[best_i]
            hints[layer_id] = {
                "plane_y_m": float(p["plane_y_m"]),
                "source": str(p.get("source", "global")),
                "v_median_px": float(p.get("v_median_px", -1.0)),
                "num_inliers": int(p.get("num_inliers", 0)),
            }
            if "index" in p:
                hints[layer_id]["plane_index"] = int(p["index"])
    return hints
