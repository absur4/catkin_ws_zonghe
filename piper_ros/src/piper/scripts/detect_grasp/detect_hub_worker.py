#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent detect hub worker (DINO + SAM2), with in-memory IPC."""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _configure_warning_filters() -> None:
    """Suppress known low-signal third-party warnings without hiding real errors."""
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"None of the inputs have requires_grad=True\. Gradients will be None",
        category=UserWarning,
    )


_configure_warning_filters()

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_segment_clouds_npz import _load_rgbd_from_files
from ipc import recv_message, send_message
from object_pc_segment import ObjectPointCloudSegmenter
from protocol import CMD_DETECT, CMD_INIT, CMD_PING, CMD_QUIT, error, ok

try:
    from tableware_grasp_detect import (
        FIXED_GRASP_CENTER_MODE as _TW_FIXED_GRASP_CENTER_MODE,
        FIXED_PCA_MODE as _TW_FIXED_PCA_MODE,
        TablewareGraspDetector,
        visualize_results as tableware_visualize_results,
        visualize_results_3d as tableware_visualize_results_3d,
    )
except Exception:  # pragma: no cover - optional mode dependency
    _TW_FIXED_GRASP_CENTER_MODE = "axis_midpoint_min_depth"
    _TW_FIXED_PCA_MODE = "3d"
    TablewareGraspDetector = None
    tableware_visualize_results = None
    tableware_visualize_results_3d = None


_PROTO_OUT = sys.stdout.buffer
_PROTO_IN = sys.stdin.buffer


class _StdoutToStderr(io.TextIOBase):
    """Route print() output away from protocol stdout."""

    def write(self, s):
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()


sys.stdout = _StdoutToStderr()


_MODEL_LEVEL_SEGMENTER_KEYS = {
    "gdino_config",
    "gdino_checkpoint",
    "sam2_config",
    "sam2_checkpoint",
    "device",
}

_TABLEWARE_PCA_DEFAULT_TEXT_PROMPT = "fork. spoon."
_TABLEWARE_PCA_DEFAULT_DEPTH_RANGE = (0.1, 0.8)
_TABLEWARE_PCA_DEFAULT_GRASP_TILT_X_DEG = -30.0
_TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT = 120000
_TABLEWARE_PCA_FIXED_MODE = str(_TW_FIXED_PCA_MODE)
_TABLEWARE_PCA_FIXED_CENTER_MODE = str(_TW_FIXED_GRASP_CENTER_MODE)


class DetectHub:
    def __init__(self):
        self.segmenter: ObjectPointCloudSegmenter | None = None
        self._model_signature: tuple[tuple[str, str], ...] | None = None
        self._camera_cache_dir = _THIS_DIR / "camera_rgbd_cache"
        self._camera_cache_dir.mkdir(parents=True, exist_ok=True)
        self._camera_rgb_path = self._camera_cache_dir / "camera_latest_color.png"
        self._camera_depth_path = self._camera_cache_dir / "camera_latest_depth.npy"

    def _cache_camera_rgbd(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> tuple[str, str]:
        """将相机帧缓存为文件，供 grasp 进程按 rgbd_files 方式读取。"""
        color_tmp = self._camera_cache_dir / "camera_latest_color.tmp.png"
        depth_tmp = self._camera_cache_dir / "camera_latest_depth.tmp.npy"

        ok = cv2.imwrite(str(color_tmp), color_image)
        if not ok:
            raise RuntimeError(f"Failed to write camera RGB cache: {color_tmp}")

        with open(depth_tmp, "wb") as f:
            np.save(f, np.asarray(depth_image, dtype=np.float32), allow_pickle=False)

        os.replace(str(color_tmp), str(self._camera_rgb_path))
        os.replace(str(depth_tmp), str(self._camera_depth_path))
        return str(self._camera_rgb_path), str(self._camera_depth_path)

    @staticmethod
    def _normalize_segmenter_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(profile.get("segmenter_kwargs", {}))
        if "text_prompt" in profile:
            kwargs["text_prompt"] = profile["text_prompt"]
        if isinstance(kwargs.get("depth_range"), list):
            kwargs["depth_range"] = tuple(kwargs["depth_range"])
        # 允许模板中用空字符串/纯空白表示“使用代码默认路径”。
        for path_key in (
            "gdino_config",
            "gdino_checkpoint",
            "sam2_config",
            "sam2_checkpoint",
            "output_dir",
        ):
            val = kwargs.get(path_key, None)
            if isinstance(val, str):
                stripped = val.strip()
                if stripped == "":
                    kwargs.pop(path_key, None)
                else:
                    kwargs[path_key] = stripped
        kwargs["save_clouds_to_disk"] = False
        return kwargs

    @staticmethod
    def _extract_model_signature(kwargs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        signature_items = []
        for key in sorted(_MODEL_LEVEL_SEGMENTER_KEYS):
            signature_items.append((key, str(kwargs.get(key, ""))))
        return tuple(signature_items)

    def _ensure_segmenter(self, profile: dict[str, Any], need_camera: bool) -> None:
        kwargs = self._normalize_segmenter_kwargs(profile)

        if self.segmenter is None:
            self.segmenter = ObjectPointCloudSegmenter(
                **kwargs,
                use_camera=need_camera,
            )
            self._model_signature = self._extract_model_signature(kwargs)
            return

        assert self._model_signature is not None
        new_signature = self._extract_model_signature(kwargs)
        if new_signature != self._model_signature:
            raise RuntimeError(
                "Detect hub model-level config changed after initialization. "
                "This worker keeps loaded DINO/SAM models fixed."
            )

        # Runtime parameter updates (no model reload).
        runtime_keys = [
            "text_prompt",
            "voxel_size",
            "plane_dist_thresh",
            "table_remove_thresh",
            "depth_range",
            "box_threshold",
            "text_threshold",
            "bbox_horizontal_shrink_ratio",
            "segmentation_method",
            "visual_preset",
        ]
        for key in runtime_keys:
            if key in kwargs and hasattr(self.segmenter, key):
                setattr(self.segmenter, key, kwargs[key])

        if "sam2_config" in kwargs:
            self.segmenter._sam2_config = kwargs["sam2_config"]
        if "sam2_checkpoint" in kwargs:
            self.segmenter._sam2_checkpoint = kwargs["sam2_checkpoint"]

        if need_camera and not self.segmenter.use_camera:
            self.segmenter.use_camera = True
            self.segmenter._init_realsense()

    def _get_rgbd(
        self,
        input_cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
        assert self.segmenter is not None

        input_source = input_cfg.get("input_source", "camera")
        if input_source == "camera":
            color_image, depth_image, intrinsics = self.segmenter.capture_rgbd()
            return color_image, depth_image, intrinsics, input_source

        if input_source == "rgbd_files":
            rgb_path = str(input_cfg.get("rgb_path", "")).strip()
            depth_path = str(input_cfg.get("depth_path", "")).strip()
            color_image, depth_image, intrinsics = _load_rgbd_from_files(
                rgb_path=rgb_path,
                depth_path=depth_path,
                depth_value_in_meters=input_cfg.get("depth_value_in_meters", False),
                depth_scale=input_cfg.get("depth_scale", 1000.0),
                intrinsics=input_cfg.get("intrinsics", {}),
            )
            return color_image, depth_image, intrinsics, input_source

        if input_source == "rgbd_arrays":
            color_image = np.asarray(input_cfg.get("color_image"))
            depth_image = np.asarray(input_cfg.get("depth_image"))
            intrinsics = dict(input_cfg.get("intrinsics", {}))
            if color_image.ndim != 3 or color_image.shape[2] != 3:
                raise ValueError(f"color_image shape must be (H,W,3), got {color_image.shape}")
            if depth_image.ndim != 2:
                raise ValueError(f"depth_image shape must be (H,W), got {depth_image.shape}")
            if color_image.shape[:2] != depth_image.shape[:2]:
                raise ValueError(
                    f"RGB/Depth size mismatch: {color_image.shape[:2]} vs {depth_image.shape[:2]}"
                )
            intrinsics["width"] = color_image.shape[1]
            intrinsics["height"] = color_image.shape[0]
            return color_image, depth_image.astype(np.float32), intrinsics, input_source

        raise ValueError(f"Unsupported input_source: {input_source}")

    @staticmethod
    def _select_target(
        confidences: list[float],
        policy: str,
        index: int,
    ) -> int:
        if len(confidences) == 0:
            raise RuntimeError("No objects detected by detect hub")

        if policy == "index":
            if index < 0 or index >= len(confidences):
                raise IndexError(
                    f"target_object_index {index} out of range for {len(confidences)} results"
                )
            return int(index)

        if policy == "top_conf":
            return int(np.argmax(np.asarray(confidences, dtype=np.float32)))

        raise ValueError(f"Unsupported target_object_policy: {policy}")

    def run_init(self, request: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        profile = dict(request.get("profile", {}))
        need_camera = bool(request.get("need_camera", False))
        self._ensure_segmenter(profile, need_camera=need_camera)
        t1 = time.perf_counter()
        assert self.segmenter is not None
        return {
            "message": "detect segmenter preloaded",
            "need_camera": need_camera,
            "device": str(getattr(self.segmenter, "device", "unknown")),
            "timing": {"init_s": float(t1 - t0)},
        }

    @staticmethod
    def _table_normal_from_plane_model(plane_model: np.ndarray) -> np.ndarray:
        normal = np.asarray(plane_model[:3], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            raise RuntimeError("Invalid table plane normal")
        if normal[2] > 0:
            normal = -normal
        return normal / np.linalg.norm(normal)

    @staticmethod
    def _run_tableware_pca(
        selected_mask: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict[str, Any],
        table_normal: np.ndarray,
        *,
        grasp_tilt_x_deg: float,
    ) -> dict[str, Any]:
        if TablewareGraspDetector is None:
            raise RuntimeError(
                "tableware_pca mode unavailable: failed to import tableware_grasp_detect"
            )

        object_points, _ = TablewareGraspDetector.extract_object_points(
            selected_mask,
            depth_image,
            intrinsics,
        )
        if object_points.shape[0] < 10:
            raise RuntimeError(
                f"Selected object has too few points for tableware_pca: {object_points.shape[0]}"
            )

        # 与 tableware_grasp_detect.main 同步:
        # 中心用 2D PCA，主轴用 3D PCA（固定策略）。
        center, _, center_2d, _ = TablewareGraspDetector.pca_analysis_2d(
            selected_mask,
            depth_image,
            intrinsics,
        )
        _, longest_axis, _, _ = TablewareGraspDetector.pca_analysis_3d(object_points)
        longest_axis = TablewareGraspDetector.orient_axis_near_to_far(
            center,
            longest_axis,
            object_points,
        )

        grasp_center = TablewareGraspDetector.compute_grasp_center(
            center,
            longest_axis,
            object_points,
            mode="axis_midpoint",
        )
        depth_values = TablewareGraspDetector.smooth_mask_depth_values(
            depth_image,
            selected_mask,
            kernel_size=5,
        )
        if depth_values.size > 0:
            grasp_center = np.asarray(grasp_center, dtype=np.float64).copy()
            z_new = float(np.percentile(depth_values, 5))
            z_old = float(grasp_center[2])

            if z_old > 1e-6:
                fx, fy = intrinsics["fx"], intrinsics["fy"]
                cx, cy = intrinsics["cx"], intrinsics["cy"]
                u = fx * grasp_center[0] / z_old + cx
                v = fy * grasp_center[1] / z_old + cy
                grasp_center[0] = (u - cx) * z_new / fx
                grasp_center[1] = (v - cy) * z_new / fy
            grasp_center[2] = z_new

        position, quaternion, rot_mat = TablewareGraspDetector.compute_grasp_pose(
            grasp_center,
            longest_axis,
            table_normal,
            grasp_tilt_x_deg=float(grasp_tilt_x_deg),
        )
        transform_mat = TablewareGraspDetector.compose_transform_matrix(position, rot_mat)

        return {
            "position": np.asarray(position, dtype=np.float64),
            "quaternion": np.asarray(quaternion, dtype=np.float64),
            "rotation_matrix": np.asarray(rot_mat, dtype=np.float64),
            "transform_matrix": np.asarray(transform_mat, dtype=np.float64),
            "center_2d": None
            if center_2d is None
            else np.asarray(center_2d, dtype=np.float64),
            "longest_axis": np.asarray(longest_axis, dtype=np.float64),
            "pca_mode": _TABLEWARE_PCA_FIXED_MODE,
            "grasp_center_mode": _TABLEWARE_PCA_FIXED_CENTER_MODE,
            "grasp_tilt_x_deg": float(grasp_tilt_x_deg),
        }

    @staticmethod
    def _apply_tableware_pca_defaults(profile: dict[str, Any]) -> dict[str, Any]:
        """对齐 tableware_grasp_detect.main() 的 tableware_pca 默认参数。"""
        prof = dict(profile)
        prof.setdefault("tableware_pca_text_prompt", _TABLEWARE_PCA_DEFAULT_TEXT_PROMPT)
        prof.setdefault("tableware_pca_depth_range", list(_TABLEWARE_PCA_DEFAULT_DEPTH_RANGE))
        prof.setdefault("tableware_pca_grasp_tilt_x_deg", _TABLEWARE_PCA_DEFAULT_GRASP_TILT_X_DEG)
        prof.setdefault(
            "tableware_pca_visualization_max_points",
            _TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT,
        )

        prompt = str(prof.get("text_prompt", "")).strip()
        if prompt in ("", "bottle.", "tableware."):
            prof["text_prompt"] = str(prof["tableware_pca_text_prompt"])

        seg_kwargs = dict(prof.get("segmenter_kwargs", {}))
        # 与 tableware_grasp_detect.main 对齐：tableware_pca 固定走 sam2_mask 流程。
        if str(seg_kwargs.get("segmentation_method", "sam2_mask")) != "sam2_mask":
            print(
                "[INFO] tableware_pca 模式强制使用 segmentation_method='sam2_mask'，"
                f"忽略当前值: {seg_kwargs.get('segmentation_method')}"
            )
        seg_kwargs["segmentation_method"] = "sam2_mask"
        depth_range = prof.get("tableware_pca_depth_range", list(_TABLEWARE_PCA_DEFAULT_DEPTH_RANGE))
        if isinstance(depth_range, tuple):
            depth_range = list(depth_range)
        current_depth_range = seg_kwargs.get("depth_range", None)
        if current_depth_range in (None, [0.1, 1.0], (0.1, 1.0)):
            seg_kwargs["depth_range"] = depth_range
        prof["segmenter_kwargs"] = seg_kwargs
        return prof

    @staticmethod
    def _detect_and_segment_like_tableware(
        segmenter: ObjectPointCloudSegmenter,
        color_image: np.ndarray,
    ) -> tuple[np.ndarray, list[str], list[float], np.ndarray]:
        """复用 tableware_grasp_detect.detect_and_segment 路径，保证行为一致。"""
        if TablewareGraspDetector is None:
            raise RuntimeError(
                "tableware_pca mode unavailable: failed to import tableware_grasp_detect"
            )
        if segmenter.sam2_predictor is None:
            segmenter._lazy_load_sam2(segmenter._sam2_config, segmenter._sam2_checkpoint)

        class _Proxy:
            pass

        proxy = _Proxy()
        proxy.grounding_model = segmenter.grounding_model
        proxy.text_prompt = str(segmenter.text_prompt)
        proxy.box_threshold = float(segmenter.box_threshold)
        proxy.text_threshold = float(segmenter.text_threshold)
        proxy.device = str(segmenter.device)
        proxy.sam2_predictor = segmenter.sam2_predictor

        masks, class_names, confidences, boxes = TablewareGraspDetector.detect_and_segment(
            proxy, color_image
        )
        masks = np.asarray(masks)
        boxes = np.asarray(boxes)
        return masks, list(class_names), [float(c) for c in confidences], boxes

    @staticmethod
    def _estimate_table_normal_like_tableware(
        segmenter: ObjectPointCloudSegmenter,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict[str, Any],
    ) -> np.ndarray:
        """复用 tableware_grasp_detect 的桌面拟合流程，保证法向计算一致。"""
        if TablewareGraspDetector is None:
            raise RuntimeError(
                "tableware_pca mode unavailable: failed to import tableware_grasp_detect"
            )

        points, colors, _ = TablewareGraspDetector.generate_point_cloud(
            depth_image, intrinsics, color_image
        )

        class _PlaneProxy:
            pass

        proxy = _PlaneProxy()
        proxy.depth_range = tuple(segmenter.depth_range)
        proxy.voxel_size = float(segmenter.voxel_size)
        proxy.plane_distance_threshold = float(segmenter.plane_dist_thresh)

        _, table_normal, _ = TablewareGraspDetector.fit_table_plane(proxy, points, colors)
        return np.asarray(table_normal, dtype=np.float64)

    @staticmethod
    def _build_skip_output(
        *,
        status: str,
        message: str,
        grasp_mode: str,
        input_source: str,
        intrinsics: dict[str, Any],
        segmentation_method: str,
        timing: dict[str, float],
        candidates: list[dict[str, Any]] | None = None,
        selected_index: int | None = None,
        selected_class_name: str | None = None,
        selected_confidence: float | None = None,
        selected_bbox: list[int] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "grasp_mode": grasp_mode,
            "input_source": input_source,
            "source_input_mode": input_source,
            "intrinsics": dict(intrinsics),
            "segmentation_method_used": segmentation_method,
            "selected_index": selected_index,
            "selected_class_name": selected_class_name,
            "selected_confidence": 0.0 if selected_confidence is None else float(selected_confidence),
            "selected_num_points": 0,
            "selected_bbox": [] if selected_bbox is None else list(selected_bbox),
            "num_candidates": len(candidates or []),
            "candidates": candidates or [],
            "timing": {
                "segmenter_update_s": float(timing.get("segmenter_update_s", 0.0)),
                "rgbd_input_s": float(timing.get("rgbd_input_s", 0.0)),
                "detect_core_s": float(timing.get("detect_core_s", 0.0)),
                "mask_extract_s": float(timing.get("mask_extract_s", 0.0)),
                "tableware_pca_s": float(timing.get("tableware_pca_s", 0.0)),
                "pack_response_s": float(timing.get("pack_response_s", 0.0)),
                "camera_cache_s": float(timing.get("camera_cache_s", 0.0)),
                "total_s": float(timing.get("total_s", 0.0)),
            },
        }

    def _maybe_visualize_tableware_pca(
        self,
        *,
        profile: dict[str, Any],
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: dict[str, Any],
        selected_mask: np.ndarray,
        class_name: str,
        confidence: float,
        tableware_result: dict[str, Any],
    ) -> None:
        if not bool(profile.get("enable_tableware_pca_visualization", False)):
            return
        if tableware_visualize_results is None or tableware_visualize_results_3d is None:
            print("[WARN] tableware_pca visualization unavailable: failed to import visualizers")
            return
        max_vis_points = _TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT
        try:
            max_vis_points = int(
                profile.get(
                    "tableware_pca_visualization_max_points",
                    _TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT,
                )
            )
        except (TypeError, ValueError):
            max_vis_points = _TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT
        if max_vis_points <= 0:
            max_vis_points = _TABLEWARE_PCA_VIS_MAX_POINTS_DEFAULT
        assert self.segmenter is not None

        vis_item = {
            "mask": np.asarray(selected_mask, dtype=bool),
            "class_name": str(class_name),
            "confidence": float(confidence),
            "center_2d": tableware_result.get("center_2d"),
            "position": tableware_result.get("position"),
            "rotation_matrix": tableware_result.get("rotation_matrix"),
        }

        # 2D overlay visualization.
        vis_img = tableware_visualize_results(
            color_image,
            [vis_item],
            intrinsics,
            save_path=None,
        )
        win_name = "Tableware Grasp Detection"
        cv2.imshow(win_name, vis_img)
        block_after_vis = bool(profile.get("block_after_visualization", False))
        cv2.waitKey(0 if block_after_vis else 1)
        if block_after_vis:
            cv2.destroyWindow(win_name)

        # 3D scene + grasp visualization (Open3D window blocks until closed).
        points_map, depth_valid = self.segmenter.depth_to_organized_cloud(depth_image, intrinsics)
        valid_mask = np.asarray(depth_valid, dtype=bool) & (np.asarray(depth_image) > 0)
        valid_flat = np.flatnonzero(valid_mask.reshape(-1))
        valid_total = int(valid_flat.size)
        if valid_total <= 0:
            print("[WARN] tableware_pca visualization skipped: no valid scene points")
            return
        if valid_total > max_vis_points:
            rng = np.random.default_rng(42)
            keep = rng.choice(valid_total, size=max_vis_points, replace=False)
            valid_flat = valid_flat[keep]
            print(
                "[INFO] tableware_pca 3D可视化点云下采样: "
                f"{valid_total} -> {max_vis_points}"
            )

        flat_points = points_map.reshape(-1, 3)
        flat_colors = color_image.reshape(-1, 3)
        scene_points = np.asarray(flat_points[valid_flat], dtype=np.float64)
        scene_colors = flat_colors[valid_flat][:, ::-1].astype(np.float64) / 255.0
        tableware_visualize_results_3d(scene_points, scene_colors, [vis_item])

    def run_detect(self, request: dict[str, Any]) -> dict[str, Any]:
        profile = dict(request.get("profile", {}))
        input_cfg = dict(request.get("input", {}))
        grasp_mode = str(request.get("grasp_mode", "graspgen"))
        if grasp_mode not in ("graspgen", "tableware_pca"):
            raise ValueError(f"Unsupported grasp_mode: {grasp_mode}")
        if grasp_mode == "tableware_pca":
            profile = self._apply_tableware_pca_defaults(profile)
        need_camera = input_cfg.get("input_source", "camera") == "camera"
        t0 = time.perf_counter()

        self._ensure_segmenter(profile, need_camera=need_camera)
        assert self.segmenter is not None
        t1 = time.perf_counter()

        color_image, depth_image, intrinsics, input_source = self._get_rgbd(input_cfg)
        t2 = time.perf_counter()

        segmentation_method = self.segmenter.segmentation_method
        non_table_mask = None
        table_normal = None
        try:
            if grasp_mode == "tableware_pca":
                depth_valid = depth_image > 0
                table_normal = self._estimate_table_normal_like_tableware(
                    self.segmenter,
                    color_image,
                    depth_image,
                    intrinsics,
                )
            else:
                if segmentation_method == "sam2_mask":
                    # sam2_mask 模式常规只保留有效深度约束。
                    depth_valid = depth_image > 0
                else:
                    points_map, depth_valid = self.segmenter.depth_to_organized_cloud(
                        depth_image,
                        intrinsics,
                    )
                    non_table_mask, _ = self.segmenter.fit_and_remove_table(points_map, depth_valid)
        except RuntimeError as exc:
            t_scene = time.perf_counter()
            return self._build_skip_output(
                status="scene_invalid",
                message=str(exc),
                grasp_mode=grasp_mode,
                input_source=input_source,
                intrinsics=intrinsics,
                segmentation_method=segmentation_method,
                timing={
                    "segmenter_update_s": float(t1 - t0),
                    "rgbd_input_s": float(t2 - t1),
                    "detect_core_s": 0.0,
                    "mask_extract_s": 0.0,
                    "tableware_pca_s": 0.0,
                    "pack_response_s": 0.0,
                    "camera_cache_s": 0.0,
                    "total_s": float(t_scene - t0),
                },
            )

        if grasp_mode == "tableware_pca":
            masks, class_names, confidences, boxes_xyxy = self._detect_and_segment_like_tableware(
                self.segmenter, color_image
            )
            t3 = time.perf_counter()
            if len(class_names) == 0:
                return self._build_skip_output(
                    status="no_object",
                    message="No objects detected",
                    grasp_mode=grasp_mode,
                    input_source=input_source,
                    intrinsics=intrinsics,
                    segmentation_method=segmentation_method,
                    timing={
                        "segmenter_update_s": float(t1 - t0),
                        "rgbd_input_s": float(t2 - t1),
                        "detect_core_s": float(t3 - t2),
                        "mask_extract_s": 0.0,
                        "tableware_pca_s": 0.0,
                        "pack_response_s": 0.0,
                        "camera_cache_s": 0.0,
                        "total_s": float(t3 - t0),
                    },
                )
            object_masks: list[np.ndarray] = []
            for mask in np.asarray(masks, dtype=bool):
                object_masks.append(mask & depth_valid)
            t4 = time.perf_counter()
        else:
            boxes_xyxy, class_names, confidences = self.segmenter.detect_objects(color_image)
            t3 = time.perf_counter()

            if len(boxes_xyxy) == 0:
                return self._build_skip_output(
                    status="no_object",
                    message="No objects detected",
                    grasp_mode=grasp_mode,
                    input_source=input_source,
                    intrinsics=intrinsics,
                    segmentation_method=segmentation_method,
                    timing={
                        "segmenter_update_s": float(t1 - t0),
                        "rgbd_input_s": float(t2 - t1),
                        "detect_core_s": float(t3 - t2),
                        "mask_extract_s": 0.0,
                        "tableware_pca_s": 0.0,
                        "pack_response_s": 0.0,
                        "camera_cache_s": 0.0,
                        "total_s": float(t3 - t0),
                    },
                )

            object_masks = []

            if segmentation_method == "sam2_mask":
                if self.segmenter.sam2_predictor is None:
                    self.segmenter._lazy_load_sam2(
                        self.segmenter._sam2_config,
                        self.segmenter._sam2_checkpoint,
                    )

                self.segmenter.sam2_predictor.set_image(color_image)
                input_boxes = np.array(boxes_xyxy)

                autocast_ctx = nullcontext()
                try:
                    autocast_ctx = torch.autocast(
                        device_type=self.segmenter.device,
                        dtype=torch.bfloat16,
                    )
                except Exception:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    masks, _, _ = self.segmenter.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                masks = masks.astype(bool)

                for mask in masks:
                    object_masks.append(mask & depth_valid)

            else:
                _, width = non_table_mask.shape
                adjusted_boxes = self.segmenter._apply_horizontal_bbox_shrink(
                    boxes_xyxy,
                    width,
                    self.segmenter.bbox_horizontal_shrink_ratio,
                )
                for x1, y1, x2, y2 in adjusted_boxes:
                    mask = np.zeros_like(non_table_mask, dtype=bool)
                    mask[y1:y2, x1:x2] = True
                    mask &= non_table_mask
                    object_masks.append(mask)
            t4 = time.perf_counter()

        if len(boxes_xyxy) == 0:
            return self._build_skip_output(
                status="no_object",
                message="No objects detected",
                grasp_mode=grasp_mode,
                input_source=input_source,
                intrinsics=intrinsics,
                segmentation_method=segmentation_method,
                timing={
                    "segmenter_update_s": float(t1 - t0),
                    "rgbd_input_s": float(t2 - t1),
                    "detect_core_s": float(t3 - t2),
                    "mask_extract_s": 0.0,
                    "tableware_pca_s": 0.0,
                    "pack_response_s": 0.0,
                    "camera_cache_s": 0.0,
                    "total_s": float(t3 - t0),
                },
            )

        target_policy = profile.get("target_object_policy", "top_conf")
        target_index = int(profile.get("target_object_index", 0))
        selected_index = self._select_target(
            confidences=confidences,
            policy=target_policy,
            index=target_index,
        )

        selected_mask = object_masks[selected_index]
        selected_points = int(np.count_nonzero(selected_mask))

        # 传输优化：mask 按 bit 打包，减少跨进程 IPC 负担。
        selected_mask_flat = selected_mask.reshape(-1).astype(np.uint8)
        selected_mask_packed = np.packbits(selected_mask_flat)

        candidates = []
        for i, (name, conf, box, mask) in enumerate(
            zip(class_names, confidences, boxes_xyxy, object_masks)
        ):
            candidates.append(
                {
                    "index": i,
                    "class_name": str(name),
                    "confidence": float(conf),
                    "bbox": np.asarray(box, dtype=np.int32).tolist(),
                    "num_points": int(np.count_nonzero(mask)),
                }
            )
        if selected_points == 0:
            t5 = time.perf_counter()
            return self._build_skip_output(
                status="no_valid_points",
                message="Selected object has zero points",
                grasp_mode=grasp_mode,
                input_source=input_source,
                intrinsics=intrinsics,
                segmentation_method=segmentation_method,
                timing={
                    "segmenter_update_s": float(t1 - t0),
                    "rgbd_input_s": float(t2 - t1),
                    "detect_core_s": float(t3 - t2),
                    "mask_extract_s": float(t4 - t3),
                    "tableware_pca_s": 0.0,
                    "pack_response_s": 0.0,
                    "camera_cache_s": 0.0,
                    "total_s": float(t5 - t0),
                },
                candidates=candidates,
                selected_index=selected_index,
                selected_class_name=str(class_names[selected_index]),
                selected_confidence=float(confidences[selected_index]),
                selected_bbox=np.asarray(boxes_xyxy[selected_index], dtype=np.int32).tolist(),
            )

        tableware_result = None
        tableware_pca_s = 0.0
        if grasp_mode == "tableware_pca":
            tt0 = time.perf_counter()
            try:
                if table_normal is None:
                    raise RuntimeError("tableware_pca requires table plane estimation")
                tableware_result = self._run_tableware_pca(
                    selected_mask=selected_mask,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    table_normal=table_normal,
                    grasp_tilt_x_deg=float(
                        profile.get(
                            "tableware_pca_grasp_tilt_x_deg",
                            _TABLEWARE_PCA_DEFAULT_GRASP_TILT_X_DEG,
                        )
                    ),
                )
                try:
                    self._maybe_visualize_tableware_pca(
                        profile=profile,
                        color_image=color_image,
                        depth_image=depth_image,
                        intrinsics=intrinsics,
                        selected_mask=selected_mask,
                        class_name=str(class_names[selected_index]),
                        confidence=float(confidences[selected_index]),
                        tableware_result=tableware_result,
                    )
                except Exception as vis_exc:
                    print(f"[WARN] tableware_pca visualization skipped: {vis_exc}")
                tt1 = time.perf_counter()
                tableware_pca_s = float(tt1 - tt0)
            except RuntimeError as exc:
                t_fail = time.perf_counter()
                return self._build_skip_output(
                    status="tableware_failed",
                    message=str(exc),
                    grasp_mode=grasp_mode,
                    input_source=input_source,
                    intrinsics=intrinsics,
                    segmentation_method=segmentation_method,
                    timing={
                        "segmenter_update_s": float(t1 - t0),
                        "rgbd_input_s": float(t2 - t1),
                        "detect_core_s": float(t3 - t2),
                        "mask_extract_s": float(t4 - t3),
                        "tableware_pca_s": float(t_fail - tt0),
                        "pack_response_s": 0.0,
                        "camera_cache_s": 0.0,
                        "total_s": float(t_fail - t0),
                    },
                    candidates=candidates,
                    selected_index=selected_index,
                    selected_class_name=str(class_names[selected_index]),
                    selected_confidence=float(confidences[selected_index]),
                    selected_bbox=np.asarray(boxes_xyxy[selected_index], dtype=np.int32).tolist(),
                )
        transfer_input_source = input_source
        camera_cache_s = 0.0
        camera_rgb_path = ""
        camera_depth_path = ""
        if input_source == "camera" and grasp_mode == "graspgen":
            tc0 = time.perf_counter()
            camera_rgb_path, camera_depth_path = self._cache_camera_rgbd(
                color_image=color_image,
                depth_image=depth_image,
            )
            tc1 = time.perf_counter()
            camera_cache_s = float(tc1 - tc0)
            transfer_input_source = "rgbd_files"
            print(f"[INFO] 相机帧缓存: {camera_rgb_path}, {camera_depth_path}")

        out = {
            "status": "ok",
            "input_source": transfer_input_source,
            "source_input_mode": input_source,
            "intrinsics": intrinsics,
            "segmentation_method_used": segmentation_method,
            "selected_mask_packed": selected_mask_packed,
            "selected_mask_shape": list(selected_mask.shape),
            "selected_index": selected_index,
            "selected_class_name": str(class_names[selected_index]),
            "selected_confidence": float(confidences[selected_index]),
            "selected_num_points": selected_points,
            "selected_bbox": np.asarray(boxes_xyxy[selected_index], dtype=np.int32).tolist(),
            "num_candidates": len(candidates),
            "candidates": candidates,
            "grasp_mode": grasp_mode,
            "timing": {
                "segmenter_update_s": float(t1 - t0),
                "rgbd_input_s": float(t2 - t1),
                "detect_core_s": float(t3 - t2),
                "mask_extract_s": float(t4 - t3),
                "tableware_pca_s": tableware_pca_s,
                "pack_response_s": 0.0,
                "camera_cache_s": camera_cache_s,
                "total_s": 0.0,
            },
        }

        # graspgen 需要将 RGBD 传递给抓取进程；tableware_pca 在 detect 进程内闭环计算，不传整帧数据。
        if grasp_mode == "graspgen":
            if transfer_input_source == "rgbd_files":
                if input_source == "rgbd_files":
                    out["rgb_path"] = str(input_cfg.get("rgb_path", "")).strip()
                    out["depth_path"] = str(input_cfg.get("depth_path", "")).strip()
                    out["depth_value_in_meters"] = bool(
                        input_cfg.get("depth_value_in_meters", False)
                    )
                    out["depth_scale"] = float(input_cfg.get("depth_scale", 1000.0))
                else:
                    out["rgb_path"] = camera_rgb_path
                    out["depth_path"] = camera_depth_path
                    out["depth_value_in_meters"] = True
                    out["depth_scale"] = 1.0
            else:
                out["color_image"] = color_image
                out["depth_image"] = depth_image
        if tableware_result is not None:
            out["tableware_pca"] = tableware_result

        t6 = time.perf_counter()
        out["timing"]["pack_response_s"] = float(t6 - t4)
        out["timing"]["total_s"] = float(t6 - t0)

        return out

    def close(self):
        if self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None


def main() -> int:
    hub = DetectHub()
    try:
        while True:
            try:
                req = recv_message(_PROTO_IN)
            except EOFError:
                break

            cmd = req.get("cmd")
            if cmd == CMD_QUIT:
                send_message(_PROTO_OUT, ok({"message": "detect hub quitting"}))
                break
            if cmd == CMD_PING:
                send_message(_PROTO_OUT, ok({"pong": "detect"}))
                continue
            if cmd == CMD_INIT:
                try:
                    out = hub.run_init(req)
                    send_message(_PROTO_OUT, ok(out))
                except Exception as exc:
                    send_message(
                        _PROTO_OUT,
                        error(str(exc), detail=traceback.format_exc()),
                    )
                continue
            if cmd != CMD_DETECT:
                send_message(_PROTO_OUT, error(f"Unsupported cmd: {cmd}"))
                continue

            try:
                out = hub.run_detect(req)
                send_message(_PROTO_OUT, ok(out))
            except Exception as exc:
                send_message(
                    _PROTO_OUT,
                    error(str(exc), detail=traceback.format_exc()),
                )
    finally:
        hub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
