#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared protocol/data defaults for detect+grasp split pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


CMD_PING = "ping"
CMD_QUIT = "quit"
CMD_INIT = "init"
CMD_DETECT = "detect"
CMD_GRASP = "grasp"


DEFAULT_INTRINSICS = {
    "fx": 901.2042236328125,
    "fy": 901.2002563476562,
    "cx": 656.2393188476562,
    "cy": 366.75244140625,
    "width": 1280,
    "height": 720,
}

DEFAULT_CAMERA_TO_EE_TRANSFORM = {
    "translation": [-0.12429490071577348, 0.003792799494851079, 0.008257624372834817],
    "quaternion": [-0.08294401079502539, 0.0836091386689819, -0.6966879101730573, 0.7076409915880956],
}

DEFAULT_EE_TO_BASE_TRANSFORM = {
    "translation": [0.058, 0.0, 0.351],
    "quaternion": [0.0, 0.884, 0.0, 0.468],
}

DEFAULT_SEGMENTER_KWARGS = {
    "output_dir": "/tmp/object_clouds",
    "voxel_size": 0.01,
    "table_remove_thresh": 0.01,
    "depth_range": [0.1, 1.0],
    "bbox_horizontal_shrink_ratio": 0.12,
    "save_clouds_to_disk": False,
    "segmentation_method": "bbox",  # 'bbox' | 'sam2_mask'
    "sam2_checkpoint": "",
    "sam2_config": "",
    "box_threshold": 0.35,
    "text_threshold": 0.25,
}

DEFAULT_ESTIMATOR_CONFIG = {
    "grasp_threshold": 0.80,
    "num_grasps": 200,
    "return_topk": True,
    "topk_num_grasps": 20,
    "filter_collisions": True,
    "collision_threshold": 0.001,
    "collision_local_radius_m": 0.20,
    "collision_scene_sampling_mode": "legacy_random",
    "enable_z_angle_filter": True,
    "max_z_angle_deg": 30.0,
    "enable_x_region_orientation_filter": True,
    "x_region_positive_threshold": 0.1,
    "x_region_negative_threshold": -0.1,
    "x_region_positive_max_angle_deg": 100.0,
    "x_region_negative_min_angle_deg": 80.0,
    "enable_y_axis_angle_filter": True,
    "min_y_axis_angle_deg": 90.0,
    "enable_world_z_axis_angle_filter": False,
    "min_world_z_axis_angle_deg": 95.0,
}


@dataclass
class UnifiedProfile:
    """One profile drives both detect and grasp hubs."""

    category: str = "default"

    # Detection / segmentation side.
    text_prompt: str = "bottle."
    target_object_policy: str = "top_conf"  # top_conf | index
    target_object_index: int = 0
    segmenter_kwargs: dict[str, Any] = field(
        default_factory=lambda: deepcopy(DEFAULT_SEGMENTER_KWARGS)
    )

    # Grasp side.
    estimator_config: dict[str, Any] = field(
        default_factory=lambda: deepcopy(DEFAULT_ESTIMATOR_CONFIG)
    )

    # Runtime behavior.
    enable_visualization: bool = True
    enable_tableware_pca_visualization: bool = False
    tableware_pca_text_prompt: str = "fork. spoon."
    tableware_pca_depth_range: list[float] = field(default_factory=lambda: [0.1, 0.8])
    tableware_pca_grasp_tilt_x_deg: float = -30.0
    tableware_pca_visualization_max_points: int = 120000
    num_visualize_grasps: int = 10
    block_after_visualization: bool = False


def default_profile_dict() -> dict[str, Any]:
    return {
        "category": "default",
        "text_prompt": "bottle.",
        "target_object_policy": "top_conf",
        "target_object_index": 0,
        "segmenter_kwargs": deepcopy(DEFAULT_SEGMENTER_KWARGS),
        "estimator_config": deepcopy(DEFAULT_ESTIMATOR_CONFIG),
        "enable_visualization": True,
        "enable_tableware_pca_visualization": False,
        "tableware_pca_text_prompt": "fork. spoon.",
        "tableware_pca_depth_range": [0.1, 0.8],
        "tableware_pca_grasp_tilt_x_deg": -30.0,
        "tableware_pca_visualization_max_points": 120000,
        "num_visualize_grasps": 10,
        "block_after_visualization": False,
    }


def default_frame_transforms_dict() -> dict[str, Any]:
    return {
        "camera_to_ee_transform": deepcopy(DEFAULT_CAMERA_TO_EE_TRANSFORM),
        "ee_to_base_transform": deepcopy(DEFAULT_EE_TO_BASE_TRANSFORM),
    }


def ok(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "data": data or {}}


def error(message: str, *, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "error": {"message": message, "detail": detail}}
