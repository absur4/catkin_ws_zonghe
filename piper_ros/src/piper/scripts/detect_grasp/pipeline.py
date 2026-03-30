#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public API for the new in-memory detect+grasp pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from .orchestrator import DetectGraspOrchestrator
    from .protocol import (
        DEFAULT_CAMERA_TO_EE_TRANSFORM,
        DEFAULT_EE_TO_BASE_TRANSFORM,
        DEFAULT_INTRINSICS,
        default_profile_dict,
    )
except ImportError:  # script mode
    from orchestrator import DetectGraspOrchestrator
    from protocol import (
        DEFAULT_CAMERA_TO_EE_TRANSFORM,
        DEFAULT_EE_TO_BASE_TRANSFORM,
        DEFAULT_INTRINSICS,
        default_profile_dict,
    )


_ANSI_BLUE = "\033[34m"
_ANSI_RESET = "\033[0m"


@dataclass
class DetectGraspConfig:
    # Input.
    category: str = "default"
    input_source: str = "rgbd_files"  # camera | rgbd_files | rgbd_arrays
    rgb_path: str = ""
    depth_path: str = ""
    depth_value_in_meters: bool = False
    depth_scale: float = 1000.0
    intrinsics: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_INTRINSICS))

    # Runtime profile setup.
    default_profile: dict[str, Any] = field(default_factory=default_profile_dict)
    category_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_override: dict[str, Any] = field(default_factory=dict)
    grasp_mode: str = "graspgen"  # graspgen | tableware_pca
    camera_to_ee_transform: dict[str, Any] = field(
        default_factory=lambda: deepcopy(DEFAULT_CAMERA_TO_EE_TRANSFORM)
    )
    ee_to_base_transform: dict[str, Any] = field(
        default_factory=lambda: deepcopy(DEFAULT_EE_TO_BASE_TRANSFORM)
    )

    # Environment.
    conda_exe: str = "conda"
    detect_env_name: str = "dsam2"
    grasp_env_name: str = "GraspGen"
    verbose: bool = True


class DetectGraspRunner:
    """Persistent runner that keeps both hub workers alive."""

    def __init__(self, config: DetectGraspConfig | None = None):
        self.config = config or DetectGraspConfig()
        self._orchestrator = DetectGraspOrchestrator(
            conda_exe=self.config.conda_exe,
            detect_env_name=self.config.detect_env_name,
            grasp_env_name=self.config.grasp_env_name,
            verbose=self.config.verbose,
            default_profile=self.config.default_profile,
            category_profiles=self.config.category_profiles,
        )

    def start(self) -> None:
        self._orchestrator.start()

    def run(
        self,
        *,
        category: str | None = None,
        input_source: str | None = None,
        rgb_path: str | None = None,
        depth_path: str | None = None,
        color_image: np.ndarray | None = None,
        depth_image: np.ndarray | None = None,
        intrinsics: dict[str, Any] | None = None,
        depth_value_in_meters: bool | None = None,
        depth_scale: float | None = None,
        profile_override: dict[str, Any] | None = None,
        grasp_mode: str | None = None,
        camera_to_ee_transform: Any | None = None,
        ee_to_base_transform: Any | None = None,
    ) -> dict[str, Any]:
        category = self.config.category if category is None else category
        input_source = self.config.input_source if input_source is None else input_source
        rgb_path = self.config.rgb_path if rgb_path is None else rgb_path
        depth_path = self.config.depth_path if depth_path is None else depth_path
        intrinsics = self.config.intrinsics if intrinsics is None else intrinsics
        depth_value_in_meters = (
            self.config.depth_value_in_meters
            if depth_value_in_meters is None
            else depth_value_in_meters
        )
        depth_scale = self.config.depth_scale if depth_scale is None else depth_scale
        grasp_mode = self.config.grasp_mode if grasp_mode is None else grasp_mode
        camera_to_ee_transform = (
            self.config.camera_to_ee_transform
            if camera_to_ee_transform is None
            else camera_to_ee_transform
        )
        ee_to_base_transform = (
            self.config.ee_to_base_transform
            if ee_to_base_transform is None
            else ee_to_base_transform
        )

        merged_override = dict(self.config.profile_override)
        if profile_override:
            merged_override.update(profile_override)

        if self.config.verbose:
            print(
                _ANSI_BLUE
                + 
                "[DetectGrasp] 模型已就绪，开始完整推理 "
                f"(mode={grasp_mode}, category={category}, input={input_source})"
                + _ANSI_RESET
            )

        out = self._orchestrator.run(
            category=category,
            input_source=input_source,
            rgb_path=rgb_path,
            depth_path=depth_path,
            color_image=color_image,
            depth_image=depth_image,
            intrinsics=intrinsics,
            depth_value_in_meters=depth_value_in_meters,
            depth_scale=depth_scale,
            profile_override=merged_override,
            grasp_mode=grasp_mode,
            camera_to_ee_transform=camera_to_ee_transform,
            ee_to_base_transform=ee_to_base_transform,
        )

        best_pose = out.get("best_grasp_pose")

        # 从 result_payload 中提取 top5 抓姿（按置信度全局降序排序后取前5）。
        result_data = out.get("result_payload", {}).get("result", {})
        all_grasp_poses = list(result_data.get("grasp_poses", []) or [])
        all_grasp_confs = np.asarray(result_data.get("grasp_conf", []) or [], dtype=np.float64)

        pair_count = min(len(all_grasp_poses), int(all_grasp_confs.shape[0]))
        top5_grasp_poses: list[np.ndarray] = []
        top5_grasp_confs: list[float] = []
        if pair_count > 0:
            order = np.argsort(-all_grasp_confs[:pair_count])
            top_idx = order[: min(5, pair_count)]
            top5_grasp_poses = [
                np.asarray(all_grasp_poses[int(i)], dtype=np.float64) for i in top_idx
            ]
            top5_grasp_confs = [float(all_grasp_confs[int(i)]) for i in top_idx]

        return {
            "result_json_path": out.get("result_json_path", None),
            "best_grasp_pose": None
            if best_pose is None
            else np.asarray(best_pose, dtype=np.float64),
            "best_grasp_conf": out.get("best_grasp_conf"),
            "status": out.get("status", "ok"),
            "message": out.get("message", ""),
            "top5_grasp_poses": top5_grasp_poses,
            "top5_grasp_confs": top5_grasp_confs,
            "timing": out.get("timing", {}),
            "result_payload": out.get("result_payload", {}),
            "grasp_mode": out.get("grasp_mode", grasp_mode),
        }

    def stop(self) -> None:
        self._orchestrator.stop()

    def __enter__(self) -> "DetectGraspRunner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def run_pipeline(
    config: DetectGraspConfig | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Single-shot API. For repeated calls, use DetectGraspRunner."""
    with DetectGraspRunner(config) as runner:
        return runner.run(**kwargs)
