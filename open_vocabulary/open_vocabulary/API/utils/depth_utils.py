"""Depth map and camera intrinsics utilities."""

import os
import re
import cv2
import numpy as np
from typing import Optional, Dict, Tuple


def load_camera_intrinsics_from_file(param_file: str) -> Optional[Dict[str, float]]:
    """从文本中解析相机内参 fx/fy/cx/cy（返回首次匹配到的一组）"""
    if not os.path.exists(param_file):
        print(f"[CabinetShelfDetector] 相机参数文件不存在: {param_file}")
        return None

    with open(param_file, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = {
        "fx": r"fx\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "fy": r"fy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "cx": r"cx\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "cy": r"cy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    }

    intrinsics = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m is None:
            print(f"[CabinetShelfDetector] 参数文件缺少 {key}: {param_file}")
            return None
        intrinsics[key] = float(m.group(1))

    return intrinsics


def load_depth_in_meters(depth_path: str, depth_scale: float = 0.001) -> np.ndarray:
    """读取深度图并转换为米。默认 RealSense: depth_uint16 * 0.001。"""
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]

    if np.issubdtype(depth_raw.dtype, np.floating):
        depth_m = depth_raw.astype(np.float32)
        max_val = float(np.nanmax(depth_m)) if depth_m.size > 0 else 0.0
        if max_val > 20.0:
            depth_m *= depth_scale
    else:
        depth_m = depth_raw.astype(np.float32) * depth_scale

    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0
    return depth_m


def median_depth_window(
    depth_m: np.ndarray,
    u: int,
    v: int,
    window: int = 5,
) -> Optional[float]:
    h, w = depth_m.shape[:2]
    if h == 0 or w == 0:
        return None

    x1 = max(0, u - window)
    x2 = min(w, u + window + 1)
    y1 = max(0, v - window)
    y2 = min(h, v + window + 1)
    roi = depth_m[y1:y2, x1:x2]
    valid = roi[roi > 1e-6]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_camera_xyz(
    u: float,
    v: float,
    z_m: float,
    intrinsics: Dict[str, float],
) -> Tuple[float, float, float]:
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    return float(x_m), float(y_m), float(z_m)


def estimate_object_contact_depth(
    depth_m: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Optional[float]:
    """估计物体与层板接触区域深度：优先用 bbox 底部窄条中值。"""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bh = y2 - y1
    strip_h = max(3, int(round(0.20 * bh)))
    ys = max(y1, y2 - strip_h)
    strip = depth_m[ys:y2, x1:x2]
    valid = strip[strip > 1e-6]
    if valid.size > 10:
        return float(np.median(valid))

    cu = int(round((x1 + x2) * 0.5))
    cv = max(0, y2 - 2)
    return median_depth_window(depth_m, cu, cv, window=7)
