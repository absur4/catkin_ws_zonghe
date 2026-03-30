#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent grasp hub worker (GraspGen), with in-memory IPC."""

from __future__ import annotations

import io
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _configure_warning_filters() -> None:
    """Suppress known low-signal third-party warnings without hiding real errors."""
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.clone\(\)\.detach\(\).*",
        category=UserWarning,
    )


_configure_warning_filters()

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from graspgen_pc_grasp_estimator import (
    GraspGenPointCloudConfig,
    GraspGenPointCloudEstimator,
    get_gripper_info,
)
from ipc import recv_message, send_message
from protocol import (
    CMD_GRASP,
    CMD_INIT,
    CMD_PING,
    CMD_QUIT,
    DEFAULT_CAMERA_TO_EE_TRANSFORM,
    DEFAULT_EE_TO_BASE_TRANSFORM,
    error,
    ok,
)

try:
    from run_grasp_from_npz import _as_jsonable as as_jsonable_like_old
    from run_grasp_from_npz import _visualize_like_demo_scene
except Exception:  # pragma: no cover - optional visualization fallback
    as_jsonable_like_old = None
    _visualize_like_demo_scene = None


_PROTO_OUT = sys.stdout.buffer
_PROTO_IN = sys.stdin.buffer


class _StdoutToStderr(io.TextIOBase):
    """Route print() output away from protocol stdout."""

    def write(self, s):
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()


sys.stdout = _StdoutToStderr()


def _load_rgbd_from_files(
    rgb_path: str,
    depth_path: str,
    depth_value_in_meters: bool,
    depth_scale: float,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """从 RGBD 文件读取图像，逻辑与旧流程保持一致。"""
    rgb_path = str(rgb_path).strip()
    depth_path = str(depth_path).strip()
    if not rgb_path or not depth_path:
        raise ValueError("rgbd_files mode requires non-empty rgb_path and depth_path")

    color_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"Failed to load RGB image: {rgb_path}")

    depth_ext = Path(depth_path).suffix.lower()
    if depth_ext == ".npy":
        depth_image = np.load(depth_path)
    else:
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Failed to load depth image: {depth_path}")

    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
    depth_image = depth_image.astype(np.float32)
    if not depth_value_in_meters:
        depth_image = depth_image / float(depth_scale)

    h, w = color_image.shape[:2]
    if depth_image.shape[:2] != (h, w):
        raise ValueError(
            f"RGB/Depth size mismatch: RGB={color_image.shape[:2]}, Depth={depth_image.shape[:2]}"
        )

    intr = dict(intrinsics)
    intr["width"] = w
    intr["height"] = h
    return color_image, depth_image, intr


def _depth_to_organized_cloud(
    depth_image: np.ndarray, intrinsics: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """将深度图转换为有组织点云，逻辑与 object_pc_segment 保持一致。"""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    h, w = depth_image.shape

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_map = np.stack([x, y, z], axis=-1)
    depth_valid = z > 0
    return points_map, depth_valid


def _unpack_mask(detect_output: dict[str, Any]) -> np.ndarray:
    """将压缩传输的掩码还原为 bool(H, W)。"""
    if "selected_mask_packed" in detect_output and "selected_mask_shape" in detect_output:
        packed = np.asarray(detect_output["selected_mask_packed"], dtype=np.uint8).reshape(-1)
        shape = tuple(detect_output["selected_mask_shape"])
        total = int(np.prod(shape))
        bits = np.unpackbits(packed, count=total)
        return bits.reshape(shape).astype(bool)

    # 兼容旧字段
    return np.asarray(detect_output["selected_mask"], dtype=bool)


def _as_jsonable_fallback(result: dict[str, Any]) -> dict[str, Any]:
    grasp_info = result.get("grasp_info", {})
    grasp_conf = grasp_info.get("grasp_conf", result.get("grasp_conf", []))
    grasp_poses = grasp_info.get("grasp_poses", result.get("grasp_poses", []))
    return {
        "gripper_name": result.get("gripper_name"),
        "best_grasp_pose": None
        if result.get("best_grasp_pose") is None
        else np.asarray(result["best_grasp_pose"]).tolist(),
        "best_grasp_conf": result.get("best_grasp_conf"),
        "num_grasps": int(np.asarray(grasp_conf).shape[0]),
        "grasp_conf": np.asarray(grasp_conf).tolist(),
        "grasp_poses": np.asarray(grasp_poses).tolist(),
    }


def _rotation_matrix_from_quaternion_xyzw(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Quaternion must have 4 elements [x,y,z,w], got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small")
    x, y, z, w = (q / norm).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_matrix_from_xyz_q(translation: Any, quaternion: Any) -> np.ndarray:
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if t.shape[0] != 3:
        raise ValueError(f"Translation must have 3 elements [x,y,z], got {t.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_matrix_from_quaternion_xyzw(quaternion)
    transform[:3, 3] = t
    return transform


def _normalize_transform_input(
    transform: Any,
    *,
    default_transform: dict[str, Any],
    name: str,
) -> np.ndarray:
    source = default_transform if transform is None else transform

    if isinstance(source, dict):
        if "matrix" in source:
            matrix = np.asarray(source["matrix"], dtype=np.float64)
        else:
            translation = source.get("translation", source.get("xyz"))
            quaternion = source.get("quaternion", source.get("q"))
            if translation is None or quaternion is None:
                raise ValueError(
                    f"{name} must provide either 'matrix' or ('translation'/'xyz' + "
                    "'quaternion'/'q')"
                )
            matrix = _transform_matrix_from_xyz_q(translation, quaternion)
    else:
        matrix = np.asarray(source, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform matrix, got shape={matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _apply_world_z_axis_filter_to_result(
    result: dict[str, Any],
    *,
    enable_filter: bool,
    min_angle_deg: float,
    camera_to_ee_transform: Any,
    ee_to_base_transform: Any,
) -> dict[str, Any]:
    if not enable_filter:
        result["world_z_axis_filter"] = {
            "enabled": False,
            "min_world_z_axis_angle_deg": float(min_angle_deg),
        }
        return result

    grasp_info = result.setdefault("grasp_info", {})
    grasp_poses = np.asarray(grasp_info.get("grasp_poses", []), dtype=np.float64)
    grasp_conf = np.asarray(grasp_info.get("grasp_conf", []), dtype=np.float64)
    total = int(grasp_poses.shape[0]) if grasp_poses.ndim == 3 else 0

    if total == 0:
        result["best_grasp_pose"] = None
        result["best_grasp_conf"] = None
        result["world_z_axis_filter"] = {
            "enabled": True,
            "min_world_z_axis_angle_deg": float(min_angle_deg),
            "kept": 0,
            "total": 0,
        }
        return result

    if grasp_poses.shape != (total, 4, 4):
        raise ValueError(f"Unexpected grasp_poses shape: {grasp_poses.shape}")
    if grasp_conf.shape[0] != total:
        raise ValueError(
            f"grasp_conf length mismatch: len(conf)={grasp_conf.shape[0]}, grasps={total}"
        )

    ee_T_cam = _normalize_transform_input(
        camera_to_ee_transform,
        default_transform=DEFAULT_CAMERA_TO_EE_TRANSFORM,
        name="camera_to_ee_transform",
    )
    base_T_ee = _normalize_transform_input(
        ee_to_base_transform,
        default_transform=DEFAULT_EE_TO_BASE_TRANSFORM,
        name="ee_to_base_transform",
    )
    base_T_cam = base_T_ee @ ee_T_cam
    world_grasps = np.matmul(base_T_cam[None, :, :], grasp_poses)

    world_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    grasp_local_z_in_world = world_grasps[:, :3, 2]
    dot_products = np.clip(np.dot(grasp_local_z_in_world, world_z_axis), -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(dot_products))
    keep_mask = angles_deg >= float(min_angle_deg)

    kept_poses = grasp_poses[keep_mask]
    kept_conf = grasp_conf[keep_mask]
    kept = int(kept_poses.shape[0])

    print(
        "[World-Z angle filter] "
        f"keep={kept}/{total} (min={float(min_angle_deg):.2f}°)"
    )
    if angles_deg.size > 0:
        print(
            "[World-Z angle filter] "
            f"angle range wrt world +Z: {angles_deg.min():.2f}° - {angles_deg.max():.2f}°"
        )

    grasp_info["grasp_poses"] = kept_poses
    grasp_info["grasp_conf"] = kept_conf
    if kept == 0:
        result["best_grasp_pose"] = None
        result["best_grasp_conf"] = None
    else:
        best_idx = int(np.argmax(kept_conf))
        result["best_grasp_pose"] = kept_poses[best_idx]
        result["best_grasp_conf"] = float(kept_conf[best_idx])

    result["world_z_axis_filter"] = {
        "enabled": True,
        "min_world_z_axis_angle_deg": float(min_angle_deg),
        "kept": kept,
        "total": total,
        "camera_to_ee_transform": ee_T_cam.tolist(),
        "ee_to_base_transform": base_T_ee.tolist(),
    }
    return result


class GraspHub:
    def __init__(self):
        self.estimator: GraspGenPointCloudEstimator | None = None
        self._gripper_config_signature: str | None = None

    @staticmethod
    def _sanitize_runtime_cfg(cfg: GraspGenPointCloudConfig) -> None:
        """Guard dangerous runtime values to avoid worker OOM/crash."""
        try:
            max_points = int(cfg.max_points)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid estimator_config.max_points: {cfg.max_points}") from exc

        # max_points <= 0 is treated as disabling random point-count sampling.
        if max_points <= 0:
            print(
                f"[INFO] estimator_config.max_points={cfg.max_points}; "
                "random point-count sampling disabled"
            )
        cfg.max_points = max_points

    def _ensure_estimator(self, profile: dict[str, Any]) -> None:
        estimator_overrides = dict(profile.get("estimator_config", {}))
        # 允许模板中用空字符串表示“使用默认 gripper_config”。
        gripper_cfg = estimator_overrides.get("gripper_config", None)
        if isinstance(gripper_cfg, str) and gripper_cfg.strip() == "":
            estimator_overrides.pop("gripper_config", None)
        elif isinstance(gripper_cfg, str):
            estimator_overrides["gripper_config"] = gripper_cfg.strip()

        if self.estimator is None:
            cfg = GraspGenPointCloudConfig(**estimator_overrides)
            self._sanitize_runtime_cfg(cfg)
            self.estimator = GraspGenPointCloudEstimator(cfg)
            self._gripper_config_signature = cfg.gripper_config
            return

        assert self._gripper_config_signature is not None
        requested_gripper_config = estimator_overrides.get(
            "gripper_config",
            self._gripper_config_signature,
        )
        if requested_gripper_config != self._gripper_config_signature:
            raise RuntimeError(
                "Grasp hub gripper_config changed after initialization. "
                "This worker keeps loaded GraspGen model fixed."
            )

        cfg = self.estimator.cfg
        for key, value in estimator_overrides.items():
            if key == "gripper_config":
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        if not cfg.return_topk:
            cfg.topk_num_grasps = -1
        if cfg.return_topk and cfg.topk_num_grasps == -1:
            cfg.topk_num_grasps = 100

        if cfg.filter_collisions and self.estimator.gripper_collision_mesh is None:
            self.estimator.gripper_collision_mesh = get_gripper_info(
                self.estimator.gripper_name
            ).collision_mesh

        if "random_seed" in estimator_overrides:
            self.estimator.rng = np.random.default_rng(cfg.random_seed)
        self._sanitize_runtime_cfg(cfg)

    def run_init(self, request: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        profile = dict(request.get("profile", {}))
        self._ensure_estimator(profile)
        t1 = time.perf_counter()
        assert self.estimator is not None
        return {
            "message": "grasp estimator preloaded",
            "gripper_name": self.estimator.gripper_name,
            "timing": {"init_s": float(t1 - t0)},
        }

    def run_grasp(self, request: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        profile = dict(request.get("profile", {}))
        detect_output = dict(request.get("detect_output", {}))

        self._ensure_estimator(profile)
        assert self.estimator is not None
        t1 = time.perf_counter()

        input_source = detect_output.get("input_source", "unknown")
        intrinsics = dict(detect_output.get("intrinsics", {}))
        if (
            input_source == "rgbd_files"
            and "rgb_path" in detect_output
            and "depth_path" in detect_output
        ):
            color_image, depth_image, intrinsics = _load_rgbd_from_files(
                rgb_path=detect_output["rgb_path"],
                depth_path=detect_output["depth_path"],
                depth_value_in_meters=bool(detect_output.get("depth_value_in_meters", False)),
                depth_scale=float(detect_output.get("depth_scale", 1000.0)),
                intrinsics=intrinsics,
            )
        else:
            color_image = np.asarray(detect_output["color_image"])
            depth_image = np.asarray(detect_output["depth_image"], dtype=np.float32)

        selected_mask = _unpack_mask(detect_output)
        t2 = time.perf_counter()

        points_map, depth_valid = _depth_to_organized_cloud(
            depth_image,
            intrinsics,
        )
        scene_pc = points_map[depth_valid]
        scene_color = color_image[depth_valid][:, ::-1].astype(np.uint8)

        object_valid = selected_mask & depth_valid
        object_pc = points_map[object_valid]
        object_color = color_image[object_valid][:, ::-1].astype(np.uint8)

        source_meta = {
            "selected_index": int(detect_output.get("selected_index", 0)),
            "num_candidates": int(detect_output.get("num_candidates", 0)),
            "class_name": detect_output.get("selected_class_name"),
            "confidence": float(detect_output.get("selected_confidence", 0.0)),
            "bbox": np.asarray(detect_output.get("selected_bbox", [])).tolist(),
            "intrinsics": intrinsics,
            "policy": profile.get("target_object_policy", "top_conf"),
            "input_source": detect_output.get("input_source", "unknown"),
            "result_json_path": None,
        }

        if object_pc.shape[0] == 0:
            t3 = time.perf_counter()
            return {
                "status": "no_valid_points",
                "message": "Selected mask yields zero object points",
                "result_json_path": None,
                "best_grasp_pose": None,
                "best_grasp_conf": None,
                "timing": {
                    "estimator_update_s": float(t1 - t0),
                    "input_prepare_s": float(t2 - t1),
                    "cloud_build_s": float(t3 - t2),
                    "inference_s": 0.0,
                    "pack_response_s": 0.0,
                    "total_s": float(t3 - t0),
                },
                "result_payload": {
                    "source": source_meta,
                    "estimator_config": asdict(self.estimator.cfg),
                    "result": {},
                },
            }
        t3 = time.perf_counter()

        result = self.estimator.estimate_grasps(
            scene_pc=scene_pc,
            object_pc=object_pc,
            scene_color=scene_color,
            object_color=object_color,
        )
        estimator_cfg = dict(profile.get("estimator_config", {}))
        frame_transforms = dict(request.get("frame_transforms", {}))
        result = _apply_world_z_axis_filter_to_result(
            result,
            enable_filter=bool(
                estimator_cfg.get(
                    "enable_world_z_axis_angle_filter",
                    getattr(self.estimator.cfg, "enable_world_z_axis_angle_filter", False),
                )
            ),
            min_angle_deg=float(
                estimator_cfg.get(
                    "min_world_z_axis_angle_deg",
                    getattr(self.estimator.cfg, "min_world_z_axis_angle_deg", 95.0),
                )
            ),
            camera_to_ee_transform=frame_transforms.get("camera_to_ee_transform"),
            ee_to_base_transform=frame_transforms.get("ee_to_base_transform"),
        )
        t4 = time.perf_counter()

        enable_visualization = bool(profile.get("enable_visualization", True))
        if enable_visualization and _visualize_like_demo_scene is not None:
            _visualize_like_demo_scene(
                result,
                num_visualize_grasps=int(profile.get("num_visualize_grasps", 10)),
                block_after_visualization=bool(
                    profile.get("block_after_visualization", False)
                ),
            )

        as_jsonable = as_jsonable_like_old or _as_jsonable_fallback
        result_payload = {
            "source": source_meta,
            "estimator_config": asdict(self.estimator.cfg),
            "world_z_axis_filter": dict(result.get("world_z_axis_filter", {})),
            "result": as_jsonable(result),
        }
        t5 = time.perf_counter()

        return {
            "status": "ok",
            "message": "",
            "result_json_path": None,
            "best_grasp_pose": result.get("best_grasp_pose"),
            "best_grasp_conf": result.get("best_grasp_conf"),
            "timing": {
                "estimator_update_s": float(t1 - t0),
                "input_prepare_s": float(t2 - t1),
                "cloud_build_s": float(t3 - t2),
                "inference_s": float(t4 - t3),
                "pack_response_s": float(t5 - t4),
                "total_s": float(t5 - t0),
            },
            "result_payload": result_payload,
        }


def main() -> int:
    hub = GraspHub()
    while True:
        try:
            req = recv_message(_PROTO_IN)
        except EOFError:
            break

        cmd = req.get("cmd")
        if cmd == CMD_QUIT:
            send_message(_PROTO_OUT, ok({"message": "grasp hub quitting"}))
            break
        if cmd == CMD_PING:
            send_message(_PROTO_OUT, ok({"pong": "grasp"}))
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
        if cmd != CMD_GRASP:
            send_message(_PROTO_OUT, error(f"Unsupported cmd: {cmd}"))
            continue

        try:
            out = hub.run_grasp(req)
            send_message(_PROTO_OUT, ok(out))
        except Exception as exc:
            send_message(
                _PROTO_OUT,
                error(str(exc), detail=traceback.format_exc()),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
