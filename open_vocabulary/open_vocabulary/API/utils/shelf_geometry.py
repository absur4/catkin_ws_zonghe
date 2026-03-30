#!/usr/bin/env python3
"""
柜子层板几何检测工具

纯 OpenCV 几何计算函数，用于检测柜子中的水平层板线并将物品分配到对应层。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def detect_shelf_lines(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    blur_ksize: int = 5,
    min_coverage: float = 0.35,  # 兼容旧接口（此实现不使用）
    # LSD + 多尺度 + RANSAC + DBSCAN 参数
    scales: Tuple[float, ...] = (1.0, 0.75, 0.5),
    min_len_ratio: float = 0.15,
    max_angle_deg: float = 12.0,
    ransac_tol: float = 10.0,
    ransac_iters: int = 80,
    ransac_min_inliers: int = 20,
    dbscan_eps: float = 28.0,
    dbscan_min_samples: int = 4,
    debug_dir: Optional[str] = None,
    **kwargs,
) -> List[int]:
    """
    在柜子 ROI 区域内检测水平层板线。

    核心思路：
    1) 多尺度 LSD 线段检测
    2) 仅保留近水平且足够长的线段
    3) 对线段 y 坐标做 DBSCAN 聚类
    4) 每个簇用 RANSAC 稳健估计层板 y

    Args:
        image: BGR 原图 (H, W, 3)
        roi: 柜子边界框 (x1, y1, x2, y2)，绝对像素坐标
        blur_ksize: 高斯模糊核大小
        min_coverage: 兼容旧接口（此实现不使用）
        scales: 多尺度比例
        min_len_ratio: 线段长度最小比例（相对 ROI 宽度）
        max_angle_deg: 线段最大倾斜角（度）
        ransac_tol: RANSAC y 方向内点阈值（像素）
        ransac_iters: RANSAC 迭代次数
        ransac_min_inliers: RANSAC 最少内点数
        dbscan_eps: DBSCAN 聚类半径（像素）
        dbscan_min_samples: DBSCAN 最小样本数
        debug_dir: 如不为 None，保存中间调试图像到此目录

    Returns:
        水平层板线的 y 坐标列表（绝对坐标，从上到下排序）
    """
    import os
    import math

    x1, y1, x2, y2 = roi
    roi_img = image[y1:y2, x1:x2]
    if roi_img.size == 0:
        return []

    roi_h, roi_w = roi_img.shape[:2]

    # ── 预处理：CLAHE 对比度增强 ──
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # ── 多尺度 LSD 线段检测 ──
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    y_candidates = []
    y_weights = []
    line_vis = roi_img.copy()

    max_angle_rad = math.radians(max_angle_deg)
    min_len = roi_w * min_len_ratio

    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            img_s = blurred
            inv_scale = 1.0
        else:
            new_w = max(1, int(round(roi_w * scale)))
            new_h = max(1, int(round(roi_h * scale)))
            img_s = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
            inv_scale = 1.0 / scale

        detected = lsd.detect(img_s)[0]
        if detected is None:
            continue

        for line in detected:
            x0, y0, x1_l, y1_l = line[0]
            x0 *= inv_scale
            y0 *= inv_scale
            x1_l *= inv_scale
            y1_l *= inv_scale
            dx = x1_l - x0
            dy = y1_l - y0
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = math.atan2(abs(dy), abs(dx) + 1e-6)
            if angle > max_angle_rad:
                continue
            y_mid = (y0 + y1_l) * 0.5 + y1
            y_candidates.append(float(y_mid))
            y_weights.append(float(length))
            if debug_dir is not None:
                cv2.line(
                    line_vis,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1_l)), int(round(y1_l))),
                    (0, 255, 255),
                    1,
                )

    # ── DBSCAN 聚类 + RANSAC 估计 ──
    if not y_candidates:
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "debug_lsd_lines.jpg"), line_vis)
        return []

    y_arr = np.array(y_candidates, dtype=np.float32).reshape(-1, 1)
    clusters = []
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(y_arr).labels_
        for label in set(labels.tolist()):
            if label == -1:
                continue
            ys = y_arr[labels == label].reshape(-1).tolist()
            clusters.append(ys)
    except Exception:
        # fallback: 简易 1D 聚类（按 eps 分组）
        ys_sorted = sorted(y_candidates)
        cur = [ys_sorted[0]]
        for v in ys_sorted[1:]:
            if abs(v - cur[-1]) <= dbscan_eps:
                cur.append(v)
            else:
                if len(cur) >= dbscan_min_samples:
                    clusters.append(cur)
                cur = [v]
        if len(cur) >= dbscan_min_samples:
            clusters.append(cur)

    def _ransac_refine(ys: List[float]) -> float:
        if len(ys) <= 2:
            return float(np.median(ys))
        rng = np.random.default_rng(0)
        ys_np = np.array(ys, dtype=np.float32)
        best_inliers = None
        best_count = -1
        for _ in range(ransac_iters):
            y0 = float(rng.choice(ys_np))
            inliers = ys_np[np.abs(ys_np - y0) <= ransac_tol]
            if inliers.size > best_count:
                best_count = int(inliers.size)
                best_inliers = inliers
        if best_inliers is not None and best_count >= max(2, ransac_min_inliers):
            return float(np.median(best_inliers))
        return float(np.median(ys_np))

    y_all = []
    for ys in clusters if clusters else [y_candidates]:
        y_all.append(_ransac_refine(ys))

    # ── 保存调试图像 ──
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "debug_gray_clahe.jpg"), gray)
        cv2.imwrite(os.path.join(debug_dir, "debug_lsd_lines.jpg"), line_vis)

    return sorted([int(round(y)) for y in y_all])
def cluster_horizontal_lines(
    y_positions: List[float],
    min_gap: int = 30,  # 调整为30，可以区分距离较近的层板
) -> List[int]:
    """
    合并相近的 y 坐标（同一层板的上下边缘）。

    Args:
        y_positions: 水平线的 y 坐标列表（已排序）
        min_gap: 合并阈值，同一簇内 y 坐标差小于此值的线被合并

    Returns:
        合并后的层板 y 坐标列表（整数，从上到下排序）
    """
    if not y_positions:
        return []

    clusters = []
    current_cluster = [y_positions[0]]

    for y in y_positions[1:]:
        if y - current_cluster[-1] <= min_gap:
            current_cluster.append(y)
        else:
            clusters.append(current_cluster)
            current_cluster = [y]
    clusters.append(current_cluster)

    # 每个簇取均值
    return sorted([int(round(np.mean(c))) for c in clusters])


def compute_shelf_layers(
    cabinet_bbox: Tuple[int, int, int, int],
    shelf_y_positions: List[int],
) -> List[Dict]:
    """
    根据柜子边界框和层板 y 坐标，将柜子划分为编号层。

    第1层 = 最上面，第N层 = 最下面。
    每一层由上边界和下边界定义：
      - 第1层: cabinet_top ~ 第1条层板线
      - 中间层: 第i条层板线 ~ 第i+1条层板线
      - 最后一层: 最后一条层板线 ~ cabinet_bottom

    Args:
        cabinet_bbox: 柜子边界框 (x1, y1, x2, y2)
        shelf_y_positions: 层板 y 坐标列表（从上到下排序）

    Returns:
        层信息列表，每项为 {"layer": int, "y_top": int, "y_bottom": int}
    """
    _, y_top, _, y_bottom = cabinet_bbox

    # 过滤掉太靠近顶部/底部的线（柜子框架边缘，不是层板）
    cab_h = y_bottom - y_top
    margin = int(cab_h * 0.08)
    valid_lines = [y for y in shelf_y_positions
                   if (y_top + margin) < y < (y_bottom - margin)]

    boundaries = [y_top] + valid_lines + [y_bottom]
    layers = []
    for i in range(len(boundaries) - 1):
        layers.append({
            "layer": i + 1,
            "y_top": boundaries[i],
            "y_bottom": boundaries[i + 1],
        })

    return layers


def assign_object_to_layer(
    object_bbox: Tuple[int, int, int, int],
    shelf_layers: List[Dict],
) -> int:
    """
    根据物品 bbox 中心 y 坐标分配到对应层。

    如果物品中心不在任何层内（在柜子外），返回 0。
    如果物品跨越多层，分配到重叠面积最大的层。

    Args:
        object_bbox: 物品边界框 (x1, y1, x2, y2)
        shelf_layers: 层信息列表 (来自 compute_shelf_layers)

    Returns:
        层号（1-based），0 表示在柜子外
    """
    if not shelf_layers:
        return 0

    ox1, oy1, ox2, oy2 = object_bbox
    obj_cy = (oy1 + oy2) / 2.0

    # 先尝试用中心点定位
    for layer in shelf_layers:
        if layer["y_top"] <= obj_cy <= layer["y_bottom"]:
            return layer["layer"]

    # 中心不在任何层内 → 计算重叠面积，分配到最大重叠的层
    best_layer = 0
    best_overlap = 0.0
    for layer in shelf_layers:
        overlap_top = max(oy1, layer["y_top"])
        overlap_bottom = min(oy2, layer["y_bottom"])
        if overlap_bottom > overlap_top:
            overlap = overlap_bottom - overlap_top
            if overlap > best_overlap:
                best_overlap = overlap
                best_layer = layer["layer"]

    return best_layer


def generate_equal_layers(
    cabinet_bbox: Tuple[int, int, int, int],
    num_shelves: int,
) -> List[Dict]:
    """
    当未检测到层板线时，将柜子等分为指定层数。

    Args:
        cabinet_bbox: 柜子边界框 (x1, y1, x2, y2)
        num_shelves: 等分层数

    Returns:
        层信息列表
    """
    _, y_top, _, y_bottom = cabinet_bbox
    height = y_bottom - y_top
    step = height / num_shelves

    layers = []
    for i in range(num_shelves):
        layers.append({
            "layer": i + 1,
            "y_top": int(round(y_top + i * step)),
            "y_bottom": int(round(y_top + (i + 1) * step)),
        })

    return layers
