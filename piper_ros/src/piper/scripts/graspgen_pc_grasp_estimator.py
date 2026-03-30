#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GraspGen scene grasp estimation without file-based intermediate JSON/PLY.

Implementation is intentionally aligned with:
- GraspGen/my_scripts/rgbd_mask_to_scene_json.py (point-cloud preprocessing)
- GraspGen/scripts/demo_scene_pc.py (grasp inference + collision filtering flow)

Difference:
- This module passes data in memory instead of writing/reading intermediate files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh.transformations as tra

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
GRASPGEN_ROOT = REPO_ROOT / "GraspGen"
if str(GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_ROOT))

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.point_cloud_utils import (
    filter_colliding_grasps,
    point_cloud_outlier_removal_with_color,
)


@dataclass
class GraspGenPointCloudConfig:
    """In-memory config replacing argparse flags."""

    # Model config / checkpoints.
    gripper_config: str = str(
        GRASPGEN_ROOT / "models/checkpoints/graspgen_robotiq_2f_140.yml"
    )

    # Preprocessing settings (aligned with rgbd_mask_to_scene_json.py).
    voxel_size: float = 0.0
    max_points: int = 30000
    obj_ratio: float = 0.1
    disable_stratified_sampling: bool = True
    disable_obj_outlier_removal: bool = False
    obj_outlier_threshold: float = 0.014
    obj_outlier_k: int = 20
    obj_mask_match_decimals: int = 6

    # Depth filtering (aligned with rgbd_mask_to_scene_json.py --max_depth_m).
    max_depth_m: float = 0.8

    # Grasp inference settings (aligned with demo_scene_pc.py defaults).
    grasp_threshold: float = 0.80
    num_grasps: int = 200
    return_topk: bool = True
    topk_num_grasps: int = 20

    # Collision filtering settings (aligned with demo_scene_pc.py defaults).
    filter_collisions: bool = True
    collision_threshold: float = 0.001
    max_scene_points: int = 8192
    collision_local_radius_m: float = 0.20
    collision_scene_sampling_mode: str = "legacy_random"

    # demo_scene_pc.py defines this argument but currently comments filtering out.
    enable_z_angle_filter: bool = True
    max_z_angle_deg: float = 30.0
    enable_x_region_orientation_filter: bool = True
    x_region_positive_threshold: float = 0.1
    x_region_negative_threshold: float = -0.1
    x_region_positive_max_angle_deg: float = 100.0
    x_region_negative_min_angle_deg: float = 80.0
    enable_y_axis_angle_filter: bool = True
    min_y_axis_angle_deg: float = 90.0
    enable_world_z_axis_angle_filter: bool = False
    min_world_z_axis_angle_deg: float = 95.0

    # demo_scene_pc.py VIZ_BOUNDS used for scene filtering.
    viz_bounds_min: tuple[float, float, float] = (-1.5, -1.25, -0.15)
    viz_bounds_max: tuple[float, float, float] = (1.5, 1.25, 2.0)

    # Reproducible random sampling.
    random_seed: int | None = 42


def _to_xyz(pc: np.ndarray, name: str) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {pc.shape}")
    if pc.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    return pc


def _to_rgb(colors: np.ndarray | None, n: int, name: str) -> np.ndarray:
    if colors is None:
        return np.zeros((n, 3), dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {colors.shape}")
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def _voxel_downsample(
    pc: np.ndarray, pc_color: np.ndarray, obj_mask: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same behavior as rgbd_mask_to_scene_json.py."""
    if voxel_size <= 0 or pc.shape[0] == 0:
        return pc, pc_color, obj_mask

    voxel_coords = np.floor(pc / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(voxel_coords, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return pc[keep_idx], pc_color[keep_idx], obj_mask[keep_idx]


def _build_obj_mask_from_subset(
    scene_pc: np.ndarray, object_pc: np.ndarray, decimals: int
) -> np.ndarray:
    """Build scene-level obj_mask when object_pc is a subset of scene_pc."""
    if object_pc.shape[0] == 0:
        return np.zeros(scene_pc.shape[0], dtype=np.uint8)

    scale = float(10**max(decimals, 0))
    scene_q = np.round(scene_pc * scale).astype(np.int64)
    obj_q = np.round(object_pc * scale).astype(np.int64)

    dtype_xyz = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    scene_keys = scene_q.view(dtype_xyz).reshape(-1)
    obj_keys = np.unique(obj_q.view(dtype_xyz).reshape(-1))

    return np.isin(scene_keys, obj_keys).astype(np.uint8)


def _process_grasps_for_visualization(
    pc: np.ndarray, grasps: np.ndarray, grasp_conf: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same centering transform flow as demo_scene_pc.py."""
    _ = grasp_conf  # kept for parity with demo_scene_pc.py signature/flow
    grasps[:, 3, 3] = 1
    t_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, t_subtract_pc_mean)
    grasps_centered = np.array(
        [t_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )
    return pc_centered, grasps_centered, t_subtract_pc_mean


def _filter_grasps_by_z_angle(
    grasps: np.ndarray, grasp_conf: np.ndarray, max_angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    base_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    max_angle_rad = np.deg2rad(max_angle_deg)
    grasp_z_axes = grasps[:, :3, 2]
    dot_products = np.dot(grasp_z_axes, base_z_axis)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    mask = angles < max_angle_rad

    print(f"Z-angle filter: {mask.sum()}/{len(grasps)} grasps with z-axis angle < {max_angle_deg}°")
    if len(angles) > 0:
        print(f"Angles range: {np.rad2deg(angles.min()):.2f}° - {np.rad2deg(angles.max()):.2f}°")

    return grasps[mask], grasp_conf[mask]


def _filter_grasps_by_x_region_orientation(
    grasps: np.ndarray,
    grasp_conf: np.ndarray,
    *,
    x_positive_threshold: float,
    x_negative_threshold: float,
    positive_region_max_angle_deg: float,
    negative_region_min_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按抓姿原点 x 位置做朝向过滤。"""
    world_x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    grasp_z_axes = grasps[:, :3, 2]
    dot_products = np.dot(grasp_z_axes, world_x_axis)
    angles_deg = np.rad2deg(np.arccos(np.clip(dot_products, -1.0, 1.0)))

    origins_x = grasps[:, 0, 3]
    pos_region = origins_x > float(x_positive_threshold)
    neg_region = origins_x < float(x_negative_threshold)
    mid_region = ~(pos_region | neg_region)

    mask = np.ones(len(grasps), dtype=bool)
    mask[pos_region] = angles_deg[pos_region] <= float(positive_region_max_angle_deg)
    mask[neg_region] = angles_deg[neg_region] >= float(negative_region_min_angle_deg)

    print(
        "[X-region orientation filter] "
        f"keep={int(mask.sum())}/{len(mask)} "
        f"(x>{x_positive_threshold}: {int(np.count_nonzero(pos_region))}, "
        f"x<{x_negative_threshold}: {int(np.count_nonzero(neg_region))}, "
        f"mid: {int(np.count_nonzero(mid_region))})"
    )
    if len(angles_deg) > 0:
        print(
            "[X-region orientation filter] "
            f"angle range wrt world +X: {angles_deg.min():.2f}° - {angles_deg.max():.2f}°"
        )

    return grasps[mask], grasp_conf[mask]


def _filter_grasps_by_world_y_angle(
    grasps: np.ndarray, grasp_conf: np.ndarray, min_angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """过滤抓姿：局部 z 轴与世界/基坐标 +Y 的夹角必须 >= min_angle_deg。"""
    world_y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    grasp_z_axes = grasps[:, :3, 2]
    dot_products = np.dot(grasp_z_axes, world_y_axis)
    angles_deg = np.rad2deg(np.arccos(np.clip(dot_products, -1.0, 1.0)))
    mask = angles_deg >= float(min_angle_deg)

    print(
        "[Y-axis angle filter] "
        f"keep={int(mask.sum())}/{len(mask)} (min={float(min_angle_deg):.2f}°)"
    )
    if len(angles_deg) > 0:
        print(
            "[Y-axis angle filter] "
            f"angle range wrt world +Y: {angles_deg.min():.2f}° - {angles_deg.max():.2f}°"
        )

    return grasps[mask], grasp_conf[mask]


class GraspGenPointCloudEstimator:
    """In-memory scene grasp estimator aligned with demo_scene_pc.py."""

    def __init__(self, cfg: GraspGenPointCloudConfig | None = None):
        self.cfg = cfg or GraspGenPointCloudConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        if not self.cfg.return_topk:
            self.cfg.topk_num_grasps = -1
        if self.cfg.return_topk and self.cfg.topk_num_grasps == -1:
            self.cfg.topk_num_grasps = 100

        grasp_cfg = load_grasp_cfg(self.cfg.gripper_config)
        self.gripper_name = grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(grasp_cfg)

        self.gripper_collision_mesh = None
        if self.cfg.filter_collisions:
            self.gripper_collision_mesh = get_gripper_info(
                self.gripper_name
            ).collision_mesh

    def _preprocess_like_rgbd_mask_to_scene_json(
        self,
        scene_pc: np.ndarray,
        object_pc: np.ndarray,
        scene_color: np.ndarray | None,
    ) -> dict[str, Any]:
        """Mirror rgbd_mask_to_scene_json.py but with in-memory point clouds."""
        pc = _to_xyz(scene_pc, "scene_pc")
        pc_color = _to_rgb(scene_color, pc.shape[0], "scene_color")
        object_pc = _to_xyz(object_pc, "object_pc")

        # Depth filtering: keep points with z > 0 and z <= max_depth_m
        # (aligned with rgbd_mask_to_scene_json.py)
        if self.cfg.max_depth_m > 0:
            depth_valid = (pc[:, 2] > 0) & (pc[:, 2] <= self.cfg.max_depth_m)
            pc = pc[depth_valid]
            pc_color = pc_color[depth_valid]
            # Also filter object_pc by the same depth range
            obj_depth_valid = (object_pc[:, 2] > 0) & (object_pc[:, 2] <= self.cfg.max_depth_m)
            object_pc = object_pc[obj_depth_valid]

        obj_mask = _build_obj_mask_from_subset(
            pc, object_pc, self.cfg.obj_mask_match_decimals
        )

        pc, pc_color, obj_mask = _voxel_downsample(
            pc, pc_color, obj_mask, self.cfg.voxel_size
        )
        max_points = int(self.cfg.max_points)

        if (
            max_points > 0
            and pc.shape[0] > max_points
            and not self.cfg.disable_stratified_sampling
        ):
            obj_idx = np.where(obj_mask == 1)[0]
            bg_idx = np.where(obj_mask == 0)[0]

            obj_target = int(max_points * self.cfg.obj_ratio)
            obj_target = max(0, min(obj_target, max_points))
            bg_target = max_points - obj_target

            obj_keep = (
                self.rng.choice(obj_idx, min(len(obj_idx), obj_target), replace=False)
                if len(obj_idx) > 0
                else np.array([], dtype=int)
            )
            bg_keep = (
                self.rng.choice(bg_idx, min(len(bg_idx), bg_target), replace=False)
                if len(bg_idx) > 0
                else np.array([], dtype=int)
            )

            keep = np.concatenate([obj_keep, bg_keep])
            if keep.size == 0:
                keep = self.rng.choice(pc.shape[0], max_points, replace=False)

            pc = pc[keep]
            pc_color = pc_color[keep]
            obj_mask = obj_mask[keep]
        elif max_points > 0 and pc.shape[0] > max_points:
            keep = self.rng.choice(pc.shape[0], max_points, replace=False)
            pc = pc[keep]
            pc_color = pc_color[keep]
            obj_mask = obj_mask[keep]

        obj_pc = pc[obj_mask == 1]
        obj_pc_color = pc_color[obj_mask == 1]

        if obj_pc.shape[0] == 0:
            raise RuntimeError(
                "No object points after preprocessing. "
                "Check object_pc/scene_pc consistency and obj_mask_match_decimals."
            )

        if (
            not self.cfg.disable_obj_outlier_removal
            and obj_pc.shape[0] > self.cfg.obj_outlier_k * 2
        ):
            obj_pc_t, _, obj_pc_color_t, _ = point_cloud_outlier_removal_with_color(
                torch.from_numpy(obj_pc),
                torch.from_numpy(obj_pc_color),
                threshold=self.cfg.obj_outlier_threshold,
                K=self.cfg.obj_outlier_k,
            )
            obj_pc = obj_pc_t.cpu().numpy()
            obj_pc_color = obj_pc_color_t.cpu().numpy().astype(np.uint8)

        return {
            "object_info": {"pc": obj_pc, "pc_color": obj_pc_color},
            "scene_info": {
                "full_pc": np.expand_dims(pc, axis=0),  # keep same shape semantics
                "img_color": pc_color,
                "obj_mask": obj_mask,
            },
            "grasp_info": {"grasp_poses": np.empty((0, 4, 4)), "grasp_conf": np.empty(0)},
        }

    def estimate_grasps(
        self,
        scene_pc: np.ndarray,
        object_pc: np.ndarray,
        scene_color: np.ndarray | None = None,
        object_color: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Estimate grasps with the same processing flow as demo_scene_pc.py."""
        _ = object_color  # not used in reference flow; object color comes from scene+obj_mask

        data = self._preprocess_like_rgbd_mask_to_scene_json(
            scene_pc=scene_pc,
            object_pc=object_pc,
            scene_color=scene_color,
        )

        obj_pc = np.array(data["object_info"]["pc"])
        obj_pc_color = np.array(data["object_info"]["pc_color"])
        xyz_scene = np.array(data["scene_info"]["full_pc"])[0]
        xyz_scene_color = np.array(data["scene_info"]["img_color"])
        xyz_seg = np.array(data["scene_info"]["obj_mask"]).reshape(-1)

        # Guard against downstream model errors when object cloud becomes empty.
        if obj_pc.shape[0] == 0:
            data["grasp_info"]["grasp_poses"] = np.empty((0, 4, 4), dtype=np.float64)
            data["grasp_info"]["grasp_conf"] = np.empty((0,), dtype=np.float32)
            data["best_grasp_pose"] = None
            data["best_grasp_conf"] = None
            data["gripper_name"] = self.gripper_name
            data["scene_info"]["collision_scene_pc"] = xyz_scene
            data["scene_info"]["collision_scene_color"] = xyz_scene_color
            return data

        # Same as demo_scene_pc.py: remove object from scene for collision checking.
        xyz_scene = xyz_scene[xyz_seg != 1]
        xyz_scene_color = xyz_scene_color[xyz_seg != 1]

        # Same bound filtering as demo_scene_pc.py.
        viz_min = np.array(self.cfg.viz_bounds_min, dtype=np.float32)
        viz_max = np.array(self.cfg.viz_bounds_max, dtype=np.float32)
        mask_within_bounds = np.all((xyz_scene > viz_min), axis=1)
        mask_within_bounds = np.logical_and(
            mask_within_bounds, np.all((xyz_scene < viz_max), axis=1)
        )
        xyz_scene = xyz_scene[mask_within_bounds]
        xyz_scene_color = xyz_scene_color[mask_within_bounds]

        grasps_t, grasp_conf_t = GraspGenSampler.run_inference(
            obj_pc,
            self.grasp_sampler,
            grasp_threshold=self.cfg.grasp_threshold,
            num_grasps=self.cfg.num_grasps,
            topk_num_grasps=self.cfg.topk_num_grasps,
            min_grasps=5 * (self.cfg.topk_num_grasps if self.cfg.topk_num_grasps > 0 else 5),
            max_tries=5,
            remove_outliers=False,
        )

        if len(grasps_t) == 0:
            data["grasp_info"]["grasp_poses"] = np.empty((0, 4, 4), dtype=np.float64)
            data["grasp_info"]["grasp_conf"] = np.empty((0,), dtype=np.float32)
            data["best_grasp_pose"] = None
            data["best_grasp_conf"] = None
            data["gripper_name"] = self.gripper_name
            data["scene_info"]["collision_scene_pc"] = xyz_scene
            data["scene_info"]["collision_scene_color"] = xyz_scene_color
            return data

        grasp_conf = grasp_conf_t.cpu().numpy()
        grasps = grasps_t.cpu().numpy()
        grasps[:, 3, 3] = 1

        _, grasps_centered, t_center = _process_grasps_for_visualization(
            obj_pc, grasps, grasp_conf
        )
        xyz_scene_centered = tra.transform_points(xyz_scene, t_center)

        collision_free_grasps = grasps_centered
        collision_free_conf = grasp_conf
        collision_free_mask = np.ones(len(grasps_centered), dtype=bool)

        if self.cfg.filter_collisions and self.gripper_collision_mesh is not None:
            mode = str(getattr(self.cfg, "collision_scene_sampling_mode", "legacy_random"))
            mode = mode.strip().lower()
            if mode not in {"legacy_random", "centroid_radius_nearest"}:
                mode = "legacy_random"

            xyz_scene_for_collision = xyz_scene_centered
            if mode == "centroid_radius_nearest" and self.cfg.collision_local_radius_m > 0:
                # Use object centroid (centered at origin) as local region anchor.
                distances = np.linalg.norm(xyz_scene_centered, axis=1)
                xyz_scene_for_collision = xyz_scene_centered[
                    distances <= float(self.cfg.collision_local_radius_m)
                ]

            if xyz_scene_for_collision.shape[0] == 0:
                # Keep all grasps if collision scene is empty after filtering.
                collision_free_mask = np.ones(len(grasps_centered), dtype=bool)
            else:
                if (
                    self.cfg.max_scene_points > 0
                    and len(xyz_scene_for_collision) > self.cfg.max_scene_points
                ):
                    keep_n = int(self.cfg.max_scene_points)
                    if mode == "centroid_radius_nearest":
                        # Keep nearest points to object centroid.
                        distances = np.linalg.norm(xyz_scene_for_collision, axis=1)
                        nearest_idx = np.argpartition(distances, keep_n - 1)[:keep_n]
                        xyz_scene_downsampled = xyz_scene_for_collision[nearest_idx]
                    else:
                        # Legacy behavior: random downsample globally.
                        indices = self.rng.choice(
                            len(xyz_scene_for_collision), keep_n, replace=False
                        )
                        xyz_scene_downsampled = xyz_scene_for_collision[indices]
                else:
                    xyz_scene_downsampled = xyz_scene_for_collision

                print(
                    "[Collision] "
                    f"mode={mode}, "
                    f"scene_points={len(xyz_scene_centered)}, "
                    f"local_points={len(xyz_scene_for_collision)}, "
                    f"check_points={len(xyz_scene_downsampled)}, "
                    f"max_scene_points={self.cfg.max_scene_points}, "
                    f"local_radius_m={self.cfg.collision_local_radius_m}"
                )

                collision_free_mask = filter_colliding_grasps(
                    scene_pc=xyz_scene_downsampled,
                    grasp_poses=grasps_centered,
                    gripper_collision_mesh=self.gripper_collision_mesh,
                    collision_threshold=self.cfg.collision_threshold,
                )

            collision_free_grasps = grasps_centered[collision_free_mask]
            collision_free_conf = grasp_conf[collision_free_mask]

        if collision_free_grasps.shape[0] == 0:
            data["grasp_info"]["grasp_poses"] = np.empty((0, 4, 4), dtype=np.float64)
            data["grasp_info"]["grasp_conf"] = np.empty((0,), dtype=np.float32)
            data["best_grasp_pose"] = None
            data["best_grasp_conf"] = None
            data["gripper_name"] = self.gripper_name
            data["collision_free_mask"] = collision_free_mask
            data["scene_info"]["collision_scene_pc"] = xyz_scene
            data["scene_info"]["collision_scene_color"] = xyz_scene_color
            return data

        if self.cfg.enable_z_angle_filter:
            z_filtered_grasps, z_filtered_conf = _filter_grasps_by_z_angle(
                collision_free_grasps, collision_free_conf, self.cfg.max_z_angle_deg
            )
        else:
            z_filtered_grasps, z_filtered_conf = collision_free_grasps, collision_free_conf

        if z_filtered_grasps.shape[0] == 0:
            data["grasp_info"]["grasp_poses"] = np.empty((0, 4, 4), dtype=np.float64)
            data["grasp_info"]["grasp_conf"] = np.empty((0,), dtype=np.float32)
            data["best_grasp_pose"] = None
            data["best_grasp_conf"] = None
            data["gripper_name"] = self.gripper_name
            data["collision_free_mask"] = collision_free_mask
            data["scene_info"]["collision_scene_pc"] = xyz_scene
            data["scene_info"]["collision_scene_color"] = xyz_scene_color
            return data

        # Same pose restoration as demo_scene_pc.py: inverse(T_center) @ grasp_centered
        t_center_inv = tra.inverse_matrix(t_center)
        grasp_poses = np.array([t_center_inv @ g for g in z_filtered_grasps])
        grasp_conf_after_z = z_filtered_conf

        if self.cfg.enable_x_region_orientation_filter:
            grasp_poses, grasp_conf_after_z = _filter_grasps_by_x_region_orientation(
                grasp_poses,
                grasp_conf_after_z,
                x_positive_threshold=self.cfg.x_region_positive_threshold,
                x_negative_threshold=self.cfg.x_region_negative_threshold,
                positive_region_max_angle_deg=self.cfg.x_region_positive_max_angle_deg,
                negative_region_min_angle_deg=self.cfg.x_region_negative_min_angle_deg,
            )

        if self.cfg.enable_y_axis_angle_filter:
            grasp_poses, grasp_conf_after_z = _filter_grasps_by_world_y_angle(
                grasp_poses,
                grasp_conf_after_z,
                self.cfg.min_y_axis_angle_deg,
            )

        if grasp_poses.shape[0] == 0:
            data["grasp_info"]["grasp_poses"] = np.empty((0, 4, 4), dtype=np.float64)
            data["grasp_info"]["grasp_conf"] = np.empty((0,), dtype=np.float32)
            data["best_grasp_pose"] = None
            data["best_grasp_conf"] = None
            data["gripper_name"] = self.gripper_name
            data["collision_free_mask"] = collision_free_mask
            data["scene_info"]["collision_scene_pc"] = xyz_scene
            data["scene_info"]["collision_scene_color"] = xyz_scene_color
            return data

        best_idx = int(np.argmax(grasp_conf_after_z))
        best_grasp_pose = grasp_poses[best_idx]
        best_grasp_conf = float(grasp_conf_after_z[best_idx])

        data["grasp_info"]["grasp_poses"] = grasp_poses
        data["grasp_info"]["grasp_conf"] = grasp_conf_after_z
        data["best_grasp_pose"] = best_grasp_pose
        data["best_grasp_conf"] = best_grasp_conf
        data["gripper_name"] = self.gripper_name
        data["collision_free_mask"] = collision_free_mask
        data["scene_info"]["collision_scene_pc"] = xyz_scene
        data["scene_info"]["collision_scene_color"] = xyz_scene_color
        return data

    def estimate_from_segment_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convenience API for outputs from object_pc_segment.py."""
        required = ["scene_point_cloud", "point_cloud"]
        for key in required:
            if key not in result:
                raise KeyError(f"Missing key in segment result: {key}")

        return self.estimate_grasps(
            scene_pc=result["scene_point_cloud"],
            object_pc=result["point_cloud"],
            scene_color=result.get("scene_point_color"),
            object_color=result.get("point_cloud_color"),
        )


def estimate_grasps_from_point_clouds(
    scene_pc: np.ndarray,
    object_pc: np.ndarray,
    scene_color: np.ndarray | None = None,
    object_color: np.ndarray | None = None,
    cfg: GraspGenPointCloudConfig | None = None,
) -> dict[str, Any]:
    """Convenience wrapper."""
    estimator = GraspGenPointCloudEstimator(cfg)
    return estimator.estimate_grasps(
        scene_pc=scene_pc,
        object_pc=object_pc,
        scene_color=scene_color,
        object_color=object_color,
    )
