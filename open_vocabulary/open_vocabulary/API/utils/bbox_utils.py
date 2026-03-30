"""Bbox utility functions: IoU, NMS, row filtering."""

import numpy as np
from typing import List, Dict, Any, Tuple


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def nms_xyxy_numpy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """纯 NumPy NMS，返回保留索引（按分数从高到低）"""
    if len(boxes_xyxy) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = np.array(
            [bbox_iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) for j in rest],
            dtype=np.float32,
        )
        order = rest[ious < iou_threshold]

    return keep


def keep_only_lower_row_boxes(
    items: List[Dict[str, Any]],
    y_tol_px: int = 18,
) -> List[Dict[str, Any]]:
    """
    同层保留"底部那一排"框。
    若同层出现上下分布（即使横向几乎不重叠），仅保留更靠下的框。
    """
    if not items:
        return []

    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for it in items:
        layer = int(it.get("layer", 0))
        by_layer.setdefault(layer, []).append(it)

    kept = []
    for layer, layer_items in by_layer.items():
        bottoms = [int(it["bbox"][3]) for it in layer_items]
        max_bottom = max(bottoms)
        for it in layer_items:
            if int(it["bbox"][3]) >= max_bottom - y_tol_px:
                kept.append(it)

    kept.sort(key=lambda x: (int(x.get("layer", 0)), int(x["bbox"][1]), int(x["bbox"][0])))
    return kept
