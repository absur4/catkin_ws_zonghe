#!/usr/bin/env python3
"""
Cabinet shelf detector shared setup and imports.
"""

import os
import sys
from typing import Any, Dict

# Configure MPS fallback before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from torchvision.ops import box_convert

try:
    import open3d as o3d
except ImportError:
    o3d = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "Grounded-SAM-2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "grounding_dino"))

from grounding_dino.groundingdino.util.inference import load_image, predict

DEFAULT_CAMERA_INTRINSICS = {
    "fx": 904.0499267578125,
    "fy": 903.094970703125,
    "cx": 649.605712890625,
    "cy": 387.4659423828125,
}

try:
    from ..grounding_sam_api import GroundingSAMAPI
    from ..utils.bbox_utils import keep_only_lower_row_boxes, nms_xyxy_numpy
    from ..utils.depth_utils import (
        estimate_object_contact_depth,
        load_camera_intrinsics_from_file,
        load_depth_in_meters,
        median_depth_window,
        pixel_to_camera_xyz,
    )
    from ..utils.experiment5 import (
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        EXPERIMENT5_TARGET_KEYWORDS,
        map_label_to_experiment5_target,
        normalize_text_label,
    )
    from ..utils.plane_estimation import (
        assign_global_planes_to_layers,
        depth_bbox_to_points,
        detect_cabinet_planes_from_depth,
        estimate_layer_height_hms,
        estimate_layer_height_robust_fusion,
        fit_plane_tls,
        merge_plane_candidates_by_height,
    )
    from ..utils.empty_space import (
        find_empty_spaces_by_layer,
        find_largest_free_rectangle,
        find_top_k_free_rectangles,
        select_placement_point_from_3d_clearance,
        uv_to_xz_on_plane,
    )
    from ..utils.shelf_geometry import (
        assign_object_to_layer,
        cluster_horizontal_lines,
        compute_shelf_layers,
        detect_shelf_lines,
        generate_equal_layers,
    )
    from ..utils.visualization import (
        LAYER_COLORS,
        draw_3d_overlay,
        draw_annotated_image,
        draw_experiment5_image,
        draw_shelf_lines_image,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
        visualize_3d,
    )
except ImportError:
    from grounding_sam_api import GroundingSAMAPI
    from utils.bbox_utils import keep_only_lower_row_boxes, nms_xyxy_numpy
    from utils.depth_utils import (
        estimate_object_contact_depth,
        load_camera_intrinsics_from_file,
        load_depth_in_meters,
        median_depth_window,
        pixel_to_camera_xyz,
    )
    from utils.experiment5 import (
        EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
        EXPERIMENT5_TARGET_KEYWORDS,
        map_label_to_experiment5_target,
        normalize_text_label,
    )
    from utils.plane_estimation import (
        assign_global_planes_to_layers,
        depth_bbox_to_points,
        detect_cabinet_planes_from_depth,
        estimate_layer_height_hms,
        estimate_layer_height_robust_fusion,
        fit_plane_tls,
        merge_plane_candidates_by_height,
    )
    from utils.empty_space import (
        find_empty_spaces_by_layer,
        find_largest_free_rectangle,
        find_top_k_free_rectangles,
        select_placement_point_from_3d_clearance,
        uv_to_xz_on_plane,
    )
    from utils.shelf_geometry import (
        assign_object_to_layer,
        cluster_horizontal_lines,
        compute_shelf_layers,
        detect_shelf_lines,
        generate_equal_layers,
    )
    from utils.visualization import (
        LAYER_COLORS,
        draw_3d_overlay,
        draw_annotated_image,
        draw_experiment5_image,
        draw_shelf_lines_image,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
        visualize_3d,
    )


def predict_dino_boxes(
    api: GroundingSAMAPI,
    image_path: str,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, Any]:
    """Run Grounding DINO only and return xyxy boxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img_h, img_w = image.shape[:2]

    prompt_clean = prompt if prompt.endswith(".") else prompt + "."
    _, image_transformed = load_image(image_path)
    boxes_raw, confidences_raw, labels_raw = predict(
        model=api.grounding_model,
        image=image_transformed,
        caption=prompt_clean.lower(),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=api.device,
    )

    if len(boxes_raw) == 0:
        return {"boxes": [], "confidences": [], "labels": []}

    scale = torch.tensor(
        [img_w, img_h, img_w, img_h],
        dtype=boxes_raw.dtype,
        device=boxes_raw.device,
    )
    boxes_abs = boxes_raw * scale
    boxes_xyxy = box_convert(
        boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy"
    ).detach().cpu().numpy()
    confidences = confidences_raw.detach().cpu().numpy().astype(np.float32)
    labels = [str(lb).strip() for lb in labels_raw]

    return {
        "boxes": boxes_xyxy.tolist(),
        "confidences": confidences.tolist(),
        "labels": labels,
    }
