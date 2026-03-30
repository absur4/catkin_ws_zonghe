#!/usr/bin/env python3
"""
基于 HMS-RANSAC 的 3D 点云平面检测与放置点推荐

核心功能：
1. 点云加载与预处理（体素下采样 + 离群点去除 + 法向量过滤）
2. HMS-RANSAC 水平平面检测（法向量预过滤 + 密度峰值 → 精准定位层板）
3. 平面边界重建（凸包 / alpha shape）
4. 占用栅格分析 + 最大空闲矩形 → 放置点推荐

依赖：open3d, numpy, scipy
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class HorizontalSurface:
    """一个检测到的水平平面"""
    plane_equation: np.ndarray       # [a, b, c, d], ax+by+cz+d=0
    height_y: float                  # 平面在 Y 轴上的高度（相机坐标系）
    inlier_points: np.ndarray        # (N, 3) 内点坐标
    num_inliers: int
    bounds_3d: Dict[str, float] = field(default_factory=dict)  # x_min/x_max/y_min/y_max/z_min/z_max
    surface_area_m2: float = 0.0     # 近似面积（平方米）


@dataclass
class PlacementPoint:
    """一个推荐的放置点"""
    position_3d: np.ndarray          # [x, y, z] 放置点 3D 坐标（米）
    surface_index: int               # 所属平面索引
    free_area_m2: float              # 可用空闲区域面积
    rect_bounds_3d: Dict[str, float] = field(default_factory=dict)  # 空闲矩形的 3D 边界


# ──────────────────────────────────────────────
# 1. 点云加载与预处理
# ──────────────────────────────────────────────

def load_and_preprocess_pointcloud(
    pcd_path: str,
    voxel_size: float = 0.005,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, "o3d.geometry.PointCloud"]:
    """
    加载 PCD 文件并做预处理（下采样 + 离群点去除）。

    Args:
        pcd_path: PCD 文件路径
        voxel_size: 体素下采样大小（米），默认 5mm
        nb_neighbors: 统计离群点去除的邻域大小
        std_ratio: 离群点标准差倍数阈值

    Returns:
        (points_np, pcd_o3d): NumPy 点阵列 (N,3) 和 Open3D 点云对象
    """
    if o3d is None:
        raise ImportError("需要 open3d 库。请安装: pip install open3d")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"点云文件不存在: {pcd_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"点云文件为空: {pcd_path}")

    print(f"  原始点云: {len(pcd.points)} 点")

    # 体素下采样
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  体素下采样后 (voxel={voxel_size}m): {len(pcd.points)} 点")

    # 统计离群点去除
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(f"  离群点去除后: {len(pcd.points)} 点")

    points = np.asarray(pcd.points, dtype=np.float64)
    return points, pcd


def filter_horizontal_points(
    points: np.ndarray,
    pcd: "o3d.geometry.PointCloud",
    up_axis: np.ndarray,
    normal_angle_threshold_deg: float = 30.0,
    normal_search_radius: float = 0.03,
    normal_max_nn: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用法向量过滤，只保留位于水平表面上的点。

    计算每个点的法向量，保留法向量与 up_axis 夹角小于阈值的点。
    这能有效去除背板、侧板、物品表面等非水平面的点。

    Args:
        points: (N, 3) 点坐标
        pcd: Open3D 点云对象（用于法向量估计）
        up_axis: 竖直方向单位向量
        normal_angle_threshold_deg: 法向量与 up_axis 的最大夹角（度）
        normal_search_radius: 法向量估计的搜索半径（米）
        normal_max_nn: 法向量估计的最大邻居数

    Returns:
        (filtered_points, filter_mask): 过滤后的点 (M, 3) 和布尔索引
    """
    if o3d is None:
        raise ImportError("需要 open3d 库")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_search_radius, max_nn=normal_max_nn
        )
    )
    normals = np.asarray(pcd.normals, dtype=np.float64)

    # cos(angle) between normal and up_axis
    cos_angles = np.abs(normals @ up_axis)
    cos_threshold = math.cos(math.radians(normal_angle_threshold_deg))
    mask = cos_angles >= cos_threshold

    filtered = points[mask]
    print(f"  法向量过滤 (<{normal_angle_threshold_deg}°): "
          f"{len(filtered)} 点 ({100*len(filtered)/max(1,len(points)):.1f}%)")

    return filtered, mask


# ──────────────────────────────────────────────
# 2. HMS-RANSAC 核心算法
# ──────────────────────────────────────────────

def hms_ransac(
    points: np.ndarray,
    all_points: np.ndarray = None,
    up_axis: np.ndarray = None,
    min_iterations: int = 300,
    angular_threshold_deg: float = 10.0,
    inlier_distance: float = 0.01,
    min_inliers: int = 500,
    duplicate_distance: float = 0.03,
    refine_with_o3d: bool = True,
    peak_sigma: float = 3.0,
    peak_min_prominence: float = 30.0,
    peak_min_distance_bins: int = 8,
) -> List[HorizontalSurface]:
    """
    HMS-RANSAC: 高效检测所有水平平面。

    输入应为经过法向量预过滤的点（只含水平面上的点），
    算法通过高度直方图密度峰值定位平面高度，再从完整点云中提取内点。

    Args:
        points: (N, 3) 法向量过滤后的点云坐标（水平面点）
        all_points: (M, 3) 完整点云坐标，用于提取最终内点。若为 None 则用 points。
        up_axis: 竖直方向单位向量，默认 [0, 1, 0]
        min_iterations: 随机采样次数（备用，密度峰值优先）
        angular_threshold_deg: 水平面角度容差（度）
        inlier_distance: 内点距离阈值（米），即平面"厚度"
        min_inliers: 最少内点数
        duplicate_distance: 重复平面合并距离（米）
        refine_with_o3d: 是否用 Open3D segment_plane 精细拟合
        peak_sigma: 直方图高斯平滑 sigma（bin 数）
        peak_min_prominence: 峰值最小突出度
        peak_min_distance_bins: 峰值之间最小距离（bin 数）

    Returns:
        按高度排序的 HorizontalSurface 列表
    """
    if up_axis is None:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-12)

    if all_points is None:
        all_points = points

    n_filtered = len(points)
    n_total = len(all_points)
    if n_filtered < 3:
        return []

    print(f"  HMS-RANSAC: {n_filtered} 水平面点 (全量 {n_total} 点)")

    # 1. 计算过滤点在 up_axis 方向的高度
    heights = points @ up_axis  # (N,)
    all_heights = all_points @ up_axis  # (M,)

    # 2. 密度峰值法：在高度直方图中找真实平面
    h_min, h_max = float(heights.min()), float(heights.max())
    h_range = h_max - h_min
    if h_range < 1e-6:
        return []

    # bin 宽度 ~= inlier_distance，确保一个平面的点集中在几个 bin 内
    n_bins = max(20, int(h_range / (inlier_distance * 0.5)))
    n_bins = min(n_bins, 500)
    hist, edges = np.histogram(heights, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # 高斯平滑
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
        smoothed = gaussian_filter1d(hist.astype(float), sigma=peak_sigma)
        # 找峰值
        mean_val = float(np.mean(smoothed[smoothed > 0])) if np.any(smoothed > 0) else 1.0
        peaks, properties = find_peaks(
            smoothed,
            height=mean_val * 1.2,
            distance=peak_min_distance_bins,
            prominence=peak_min_prominence,
        )
        peak_heights = [float(centers[p]) for p in peaks]
    except ImportError:
        # scipy 不可用，回退到随机采样
        print("  scipy 不可用，回退到随机采样模式")
        peak_heights = []
        rng = np.random.default_rng(42)
        seen = []
        for _ in range(min_iterations):
            idx = rng.integers(0, n_filtered)
            h = heights[idx]
            count = int(np.sum(np.abs(heights - h) <= inlier_distance))
            if count >= min_inliers:
                is_dup = any(abs(h - sh) < duplicate_distance for sh in seen)
                if not is_dup:
                    peak_heights.append(float(h))
                    seen.append(float(h))

    if not peak_heights:
        print(f"  HMS-RANSAC: 未找到密度峰值")
        return []

    print(f"  HMS-RANSAC: 找到 {len(peak_heights)} 个密度峰值: "
          f"{[f'{h:.4f}' for h in peak_heights]}")

    # 3. 对每个峰值高度，从完整点云中提取内点并精细拟合
    surfaces = []
    angular_threshold_rad = math.radians(angular_threshold_deg)

    for peak_h in peak_heights:
        # 从完整点云中提取该高度附近的点
        inlier_mask = np.abs(all_heights - peak_h) <= inlier_distance
        inlier_pts = all_points[inlier_mask]

        if len(inlier_pts) < min_inliers:
            print(f"    峰值 h={peak_h:.4f}: 内点 {len(inlier_pts)} < {min_inliers}, 跳过")
            continue

        if refine_with_o3d and o3d is not None and len(inlier_pts) >= 3:
            pcd_inlier = o3d.geometry.PointCloud()
            pcd_inlier.points = o3d.utility.Vector3dVector(inlier_pts)
            try:
                plane_model, refined_inliers = pcd_inlier.segment_plane(
                    distance_threshold=inlier_distance,
                    ransac_n=3,
                    num_iterations=200,
                )
                a, b, c, d = plane_model
                normal = np.array([a, b, c], dtype=np.float64)
                normal_len = np.linalg.norm(normal)
                if normal_len < 1e-12:
                    continue
                normal = normal / normal_len

                cos_angle = abs(np.dot(normal, up_axis))
                angle = math.acos(np.clip(cos_angle, 0.0, 1.0))
                if angle > angular_threshold_rad:
                    continue

                refined_pts = inlier_pts[refined_inliers]
                if len(refined_pts) < min_inliers:
                    continue

                plane_eq = np.array([a, b, c, d], dtype=np.float64)
                height_y = float(np.mean(refined_pts @ up_axis))

                surface = HorizontalSurface(
                    plane_equation=plane_eq,
                    height_y=height_y,
                    inlier_points=refined_pts,
                    num_inliers=len(refined_pts),
                )
                surfaces.append(surface)
            except Exception:
                plane_eq = np.array([up_axis[0], up_axis[1], up_axis[2], -peak_h],
                                    dtype=np.float64)
                surface = HorizontalSurface(
                    plane_equation=plane_eq,
                    height_y=float(peak_h),
                    inlier_points=inlier_pts,
                    num_inliers=len(inlier_pts),
                )
                surfaces.append(surface)
        else:
            plane_eq = np.array([up_axis[0], up_axis[1], up_axis[2], -peak_h],
                                dtype=np.float64)
            surface = HorizontalSurface(
                plane_equation=plane_eq,
                height_y=float(peak_h),
                inlier_points=inlier_pts,
                num_inliers=len(inlier_pts),
            )
            surfaces.append(surface)

    # 按高度排序
    surfaces.sort(key=lambda s: s.height_y)
    print(f"  HMS-RANSAC: 最终检测到 {len(surfaces)} 个水平平面")
    for i, s in enumerate(surfaces):
        print(f"    平面 {i}: height_y={s.height_y:.4f}m, "
              f"inliers={s.num_inliers}")

    return surfaces


# ──────────────────────────────────────────────
# 3. 平面边界重建
# ──────────────────────────────────────────────

def reconstruct_surface_bounds(surface: HorizontalSurface) -> HorizontalSurface:
    """
    重建平面的 3D 边界和面积。

    将内点投影到 XZ 平面，计算凸包得到面积和边界。

    Args:
        surface: 待重建的水平平面

    Returns:
        更新了 bounds_3d 和 surface_area_m2 的 surface
    """
    pts = surface.inlier_points
    if len(pts) == 0:
        return surface

    # 3D 边界
    surface.bounds_3d = {
        "x_min": float(np.min(pts[:, 0])),
        "x_max": float(np.max(pts[:, 0])),
        "y_min": float(np.min(pts[:, 1])),
        "y_max": float(np.max(pts[:, 1])),
        "z_min": float(np.min(pts[:, 2])),
        "z_max": float(np.max(pts[:, 2])),
    }

    # 投影到 XZ 平面计算面积
    xz = pts[:, [0, 2]]  # (N, 2)

    try:
        from scipy.spatial import ConvexHull
        if len(xz) >= 3:
            hull = ConvexHull(xz)
            surface.surface_area_m2 = float(hull.volume)  # 2D ConvexHull.volume = area
        else:
            surface.surface_area_m2 = 0.0
    except Exception:
        # 凸包失败时用边界框面积近似
        dx = surface.bounds_3d["x_max"] - surface.bounds_3d["x_min"]
        dz = surface.bounds_3d["z_max"] - surface.bounds_3d["z_min"]
        surface.surface_area_m2 = float(dx * dz)

    return surface


# ──────────────────────────────────────────────
# 4. 最大空闲矩形 DP 算法
# ──────────────────────────────────────────────

def find_largest_free_rectangle(
    free_mask: np.ndarray,
) -> Tuple[int, int, int, int, int]:
    """
    在二值占用栅格中找最大空闲矩形（直方图 DP 算法）。

    Args:
        free_mask: (H, W) bool 数组，True = 空闲, False = 被占用

    Returns:
        (row_start, col_start, row_end, col_end, area)
        矩形范围 [row_start:row_end, col_start:col_end]，以及面积（格子数）。
        若无空闲区域，返回 (0, 0, 0, 0, 0)。
    """
    if free_mask.size == 0:
        return (0, 0, 0, 0, 0)

    rows, cols = free_mask.shape
    # 构建直方图：每列连续空闲格子高度
    histogram = np.zeros(cols, dtype=np.int32)

    best_area = 0
    best_rect = (0, 0, 0, 0)

    for r in range(rows):
        for c in range(cols):
            if free_mask[r, c]:
                histogram[c] += 1
            else:
                histogram[c] = 0

        # 用单调栈求直方图最大矩形
        stack = []  # (col_index, height)
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


def _find_top_k_free_rectangles(
    free_mask: np.ndarray,
    k: int = 3,
    min_area_cells: int = 10,
) -> List[Tuple[int, int, int, int, int]]:
    """
    迭代找 top-K 个不重叠的最大空闲矩形。

    Args:
        free_mask: (H, W) bool 数组
        k: 最多返回几个矩形
        min_area_cells: 最小面积（格子数）

    Returns:
        矩形列表 [(row_start, col_start, row_end, col_end, area), ...]
    """
    mask = free_mask.copy()
    results = []

    for _ in range(k):
        rect = find_largest_free_rectangle(mask)
        r_start, c_start, r_end, c_end, area = rect
        if area < min_area_cells:
            break
        results.append(rect)
        # 标记已选区域为占用
        mask[r_start:r_end, c_start:c_end] = False

    return results


# ──────────────────────────────────────────────
# 5. 放置点计算
# ──────────────────────────────────────────────

def compute_placement_points(
    surface: HorizontalSurface,
    all_points: np.ndarray,
    surface_index: int = 0,
    grid_resolution: float = 0.01,
    object_height_min: float = 0.02,
    object_height_max: float = 0.50,
    margin: float = 0.02,
    dilate_radius: int = 2,
    max_placements: int = 3,
    min_free_area_m2: float = 0.001,
    debug_dir: Optional[str] = None,
) -> Tuple[List[PlacementPoint], Optional[np.ndarray]]:
    """
    在一个水平平面上计算可放置物品的空闲位置。

    算法：
    1. 在平面 XZ 范围内建立 2D 占用栅格
    2. 平面轮廓外 → 不可用；平面上方的点 → 被物体占用
    3. 对占用区域做膨胀（安全间距）
    4. 用最大矩形 DP 算法找最大空闲矩形
    5. 矩形中心 → 放置点 3D 坐标

    Args:
        surface: 目标水平平面
        all_points: (M, 3) 完整点云坐标
        surface_index: 该平面的索引编号
        grid_resolution: 栅格分辨率（米），默认 1cm
        object_height_min: 物体最小高度（低于此视为噪声）
        object_height_max: 物体最大高度检测范围
        margin: 放置点距边缘安全距离（米）
        dilate_radius: 占用区域膨胀半径（格子数）
        max_placements: 每层最多返回的放置点数
        min_free_area_m2: 最小空闲面积（平方米）
        debug_dir: 调试图保存目录

    Returns:
        (placements, occupancy_grid):
          placements: PlacementPoint 列表
          occupancy_grid: 调试用占用栅格图像 (H, W, 3) BGR，若无 debug_dir 则为 None
    """
    bounds = surface.bounds_3d
    if not bounds:
        return [], None

    height_y = surface.height_y

    # 栅格范围（加 margin 收缩）
    x_min = bounds["x_min"] + margin
    x_max = bounds["x_max"] - margin
    z_min = bounds["z_min"] + margin
    z_max = bounds["z_max"] - margin

    if x_max <= x_min or z_max <= z_min:
        return [], None

    # 栅格尺寸
    n_cols = max(1, int(round((x_max - x_min) / grid_resolution)))
    n_rows = max(1, int(round((z_max - z_min) / grid_resolution)))

    if n_rows * n_cols > 5_000_000:
        # 防止内存溢出：栅格过大时增大分辨率
        scale = math.sqrt(n_rows * n_cols / 5_000_000)
        grid_resolution *= scale
        n_cols = max(1, int(round((x_max - x_min) / grid_resolution)))
        n_rows = max(1, int(round((z_max - z_min) / grid_resolution)))

    # 初始化栅格：全部标记为不可用
    free_mask = np.zeros((n_rows, n_cols), dtype=bool)

    # 标记平面内点所在的格子为空闲
    inlier_xz = surface.inlier_points[:, [0, 2]]
    ci = np.floor((inlier_xz[:, 0] - x_min) / grid_resolution).astype(np.int32)
    ri = np.floor((inlier_xz[:, 1] - z_min) / grid_resolution).astype(np.int32)
    valid = (ci >= 0) & (ci < n_cols) & (ri >= 0) & (ri < n_rows)
    ci = ci[valid]
    ri = ri[valid]
    free_mask[ri, ci] = True

    # 标记平面上方有物体的格子为占用
    # "上方"在相机坐标系中: Y 轴向下，所以 "上方" 对应 Y 值更小
    # 但统一用高度差的绝对方向: 物体在平面上方 → Y < height_y (相机坐标系)
    # 实际上这里用更通用的方式：检查平面附近 XZ 范围内, Y 在
    # [height_y - object_height_max, height_y - object_height_min] 的点
    # （相机坐标系 Y 轴向下时，物体在上方意味着 Y 更小）
    y_upper = height_y - object_height_min
    y_lower = height_y - object_height_max

    # 筛选平面上方的点
    above_mask = (
        (all_points[:, 1] >= y_lower) &
        (all_points[:, 1] <= y_upper) &
        (all_points[:, 0] >= x_min) &
        (all_points[:, 0] <= x_max) &
        (all_points[:, 2] >= z_min) &
        (all_points[:, 2] <= z_max)
    )
    above_pts = all_points[above_mask]

    if len(above_pts) > 0:
        oc = np.floor((above_pts[:, 0] - x_min) / grid_resolution).astype(np.int32)
        orow = np.floor((above_pts[:, 2] - z_min) / grid_resolution).astype(np.int32)
        ov = (oc >= 0) & (oc < n_cols) & (orow >= 0) & (orow < n_rows)
        oc = oc[ov]
        orow = orow[ov]
        free_mask[orow, oc] = False

    # 膨胀占用区域（安全间距）
    if dilate_radius > 0:
        import cv2
        occupied = (~free_mask).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_radius + 1, 2 * dilate_radius + 1),
        )
        occupied = cv2.dilate(occupied, kernel, iterations=1)
        free_mask = occupied == 0

    # 找最大空闲矩形
    rects = _find_top_k_free_rectangles(
        free_mask, k=max_placements,
        min_area_cells=max(1, int(min_free_area_m2 / (grid_resolution ** 2))),
    )

    placements = []
    for r_start, c_start, r_end, c_end, area_cells in rects:
        # 使用距离变换选“最稳健放置点”（离障碍/边界最远），而不是矩形几何中心
        import cv2
        rect_free = free_mask[r_start:r_end, c_start:c_end]
        if rect_free.size == 0 or not np.any(rect_free):
            continue

        dt = cv2.distanceTransform(rect_free.astype(np.uint8), cv2.DIST_L2, 5)
        rr, _ = np.indices(rect_free.shape, dtype=np.float32)
        front_bias = 1.0 - rr / max(1.0, float(rect_free.shape[0] - 1))
        score = dt * (0.90 + 0.10 * front_bias)
        score[~rect_free] = -1.0

        best_idx = int(np.argmax(score))
        br, bc = np.unravel_index(best_idx, score.shape)
        row_idx = r_start + int(br)
        col_idx = c_start + int(bc)

        cx = x_min + (col_idx + 0.5) * grid_resolution
        cz = z_min + (row_idx + 0.5) * grid_resolution
        area_m2 = area_cells * (grid_resolution ** 2)

        pp = PlacementPoint(
            position_3d=np.array([cx, height_y, cz], dtype=np.float64),
            surface_index=surface_index,
            free_area_m2=float(area_m2),
            rect_bounds_3d={
                "x_min": float(x_min + c_start * grid_resolution),
                "x_max": float(x_min + c_end * grid_resolution),
                "z_min": float(z_min + r_start * grid_resolution),
                "z_max": float(z_min + r_end * grid_resolution),
                "y": float(height_y),
            },
        )
        placements.append(pp)

    # 调试可视化
    occupancy_vis = None
    if debug_dir is not None:
        import cv2
        os.makedirs(debug_dir, exist_ok=True)
        # 白=空闲, 红=占用
        vis = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
        vis[free_mask] = [255, 255, 255]
        vis[~free_mask] = [0, 0, 200]

        # 绿色矩形标记选中区域
        for r_start, c_start, r_end, c_end, _ in rects:
            cv2.rectangle(vis, (c_start, r_start), (c_end - 1, r_end - 1),
                          (0, 255, 0), max(1, min(n_rows, n_cols) // 100))

        # 放大以便查看
        scale = max(1, 400 // max(n_rows, n_cols, 1))
        if scale > 1:
            vis = cv2.resize(vis, (n_cols * scale, n_rows * scale),
                             interpolation=cv2.INTER_NEAREST)

        path = os.path.join(debug_dir, f"occupancy_surface_{surface_index}.jpg")
        cv2.imwrite(path, vis)
        occupancy_vis = vis

    return placements, occupancy_vis


# ──────────────────────────────────────────────
# 6. 辅助函数：3D → 2D 投影
# ──────────────────────────────────────────────

def project_points_to_image(
    points_3d: np.ndarray,
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    将 3D 点投影到 2D 图像平面。

    Args:
        points_3d: (N, 3) 3D 坐标（相机坐标系）
        intrinsics: 相机内参 {"fx", "fy", "cx", "cy"}
        image_shape: (height, width) 图像尺寸

    Returns:
        (N, 2) 像素坐标 [u, v]，无效点为 [-1, -1]
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)

    h, w = image_shape
    z = pts[:, 2]
    valid = z > 1e-3

    uv = np.full((len(pts), 2), -1, dtype=np.float64)
    if np.any(valid):
        u = (pts[valid, 0] * fx / z[valid]) + cx
        v = (pts[valid, 1] * fy / z[valid]) + cy
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uv_valid = np.stack([u, v], axis=1)
        uv_valid[~in_bounds] = -1
        uv[valid] = uv_valid

    return uv


def surfaces_to_shelf_y(
    surfaces: List[HorizontalSurface],
    intrinsics: Dict[str, float],
    image_shape: Tuple[int, int],
) -> List[int]:
    """
    将 3D 水平平面映射到 2D 图像中的层板线 y 坐标。

    对每个平面取内点投影的中位 v 坐标。

    Args:
        surfaces: 检测到的水平平面列表
        intrinsics: 相机内参
        image_shape: (height, width)

    Returns:
        每个平面对应的 y 坐标列表（像素），从上到下排序
    """
    y_list = []
    for surface in surfaces:
        pts = surface.inlier_points
        if len(pts) == 0:
            continue
        uv = project_points_to_image(pts, intrinsics, image_shape)
        valid = uv[:, 1] >= 0
        if np.any(valid):
            y_list.append(int(round(float(np.median(uv[valid, 1])))))
        else:
            y_list.append(-1)

    return y_list


# ──────────────────────────────────────────────
# 7. 完整管线：detect_horizontal_surfaces
# ──────────────────────────────────────────────

def detect_horizontal_surfaces(
    pcd_path: str,
    up_axis: np.ndarray = None,
    voxel_size: float = 0.005,
    min_iterations: int = 300,
    angular_threshold_deg: float = 10.0,
    inlier_distance: float = 0.01,
    min_inliers: int = 8000,
    duplicate_distance: float = 0.03,
    grid_resolution: float = 0.01,
    object_height_min: float = 0.02,
    object_height_max: float = 0.50,
    margin: float = 0.02,
    max_placements_per_surface: int = 3,
    debug_dir: Optional[str] = None,
) -> Tuple[List[HorizontalSurface], List[PlacementPoint], np.ndarray]:
    """
    完整管线：加载点云 → HMS-RANSAC → 边界重建 → 放置点计算。

    Args:
        pcd_path: PCD 文件路径
        up_axis: 竖直方向，默认 [0, 1, 0]
        voxel_size: 体素下采样大小
        min_iterations: HMS-RANSAC 迭代次数
        angular_threshold_deg: 水平面角度容差
        inlier_distance: 内点距离阈值
        min_inliers: 最少内点数
        duplicate_distance: 重复平面合并距离
        grid_resolution: 占用栅格分辨率
        object_height_min: 物体最小高度
        object_height_max: 物体最大高度检测范围
        margin: 安全距离
        max_placements_per_surface: 每层最多放置点数
        debug_dir: 调试输出目录

    Returns:
        (surfaces, placements, all_points):
          surfaces: 检测到的水平平面列表
          placements: 所有放置点列表
          all_points: 预处理后的点云 (N, 3)
    """
    if up_axis is None:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # 1. 加载与预处理
    print("[PointCloudGeometry] Step 1: 加载与预处理点云...")
    all_points, pcd_o3d = load_and_preprocess_pointcloud(
        pcd_path, voxel_size=voxel_size
    )

    # 2. HMS-RANSAC
    print("[PointCloudGeometry] Step 2: HMS-RANSAC 水平平面检测...")
    surfaces = hms_ransac(
        all_points,
        up_axis=up_axis,
        min_iterations=min_iterations,
        angular_threshold_deg=angular_threshold_deg,
        inlier_distance=inlier_distance,
        min_inliers=min_inliers,
        duplicate_distance=duplicate_distance,
    )

    # 3. 边界重建
    print("[PointCloudGeometry] Step 3: 平面边界重建...")
    for surface in surfaces:
        reconstruct_surface_bounds(surface)

    # 4. 放置点计算
    print("[PointCloudGeometry] Step 4: 放置点计算...")
    all_placements = []
    for i, surface in enumerate(surfaces):
        placements, _ = compute_placement_points(
            surface,
            all_points,
            surface_index=i,
            grid_resolution=grid_resolution,
            object_height_min=object_height_min,
            object_height_max=object_height_max,
            margin=margin,
            max_placements=max_placements_per_surface,
            debug_dir=debug_dir,
        )
        all_placements.extend(placements)
        print(f"  平面 {i}: {len(placements)} 个放置点")

    print(f"[PointCloudGeometry] 完成: {len(surfaces)} 个平面, "
          f"{len(all_placements)} 个放置点")

    return surfaces, all_placements, all_points
