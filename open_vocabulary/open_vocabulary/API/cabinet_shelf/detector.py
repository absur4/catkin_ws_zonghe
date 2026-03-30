#!/usr/bin/env python3
"""Public detector class for cabinet shelf tasks."""

from typing import Any, Dict, Optional

try:
    from .runtime import (
        DEFAULT_CAMERA_INTRINSICS,
        GroundingSAMAPI,
        assign_global_planes_to_layers,
        depth_bbox_to_points,
        detect_cabinet_planes_from_depth,
        draw_3d_overlay,
        draw_annotated_image,
        draw_experiment5_image,
        draw_shelf_lines_image,
        estimate_layer_height_hms,
        estimate_layer_height_robust_fusion,
        estimate_object_contact_depth,
        find_empty_spaces_by_layer,
        fit_plane_tls,
        load_depth_in_meters,
        median_depth_window,
        pixel_to_camera_xyz,
        predict_dino_boxes,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
        select_placement_point_from_3d_clearance,
        uv_to_xz_on_plane,
        visualize_3d,
    )
    from .runtime import merge_plane_candidates_by_height
    from .workflows import detect, detect_3d, detect_experiment5
except ImportError:
    from cabinet_shelf.runtime import (
        DEFAULT_CAMERA_INTRINSICS,
        GroundingSAMAPI,
        assign_global_planes_to_layers,
        depth_bbox_to_points,
        detect_cabinet_planes_from_depth,
        draw_3d_overlay,
        draw_annotated_image,
        draw_experiment5_image,
        draw_shelf_lines_image,
        estimate_layer_height_hms,
        estimate_layer_height_robust_fusion,
        estimate_object_contact_depth,
        find_empty_spaces_by_layer,
        fit_plane_tls,
        load_depth_in_meters,
        median_depth_window,
        pixel_to_camera_xyz,
        predict_dino_boxes,
        save_colored_plane_cloud,
        save_plane_cloud_with_placements,
        select_placement_point_from_3d_clearance,
        uv_to_xz_on_plane,
        visualize_3d,
        merge_plane_candidates_by_height,
    )
    from cabinet_shelf.workflows import detect, detect_3d, detect_experiment5


class CabinetShelfDetector:
    """柜子分层检测器：检测物品并判断其位于柜子第几层"""

    def __init__(
        self,
        gsam_api: Optional[GroundingSAMAPI] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.api = gsam_api if gsam_api is not None else GroundingSAMAPI(**kwargs)
        self.camera_intrinsics = camera_intrinsics or dict(DEFAULT_CAMERA_INTRINSICS)

    def _predict_dino_boxes(self, image_path: str, prompt: str, box_threshold: float, text_threshold: float) -> Dict[str, Any]:
        return predict_dino_boxes(
            api=self.api,
            image_path=image_path,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    @staticmethod
    def _load_depth_in_meters(depth_path: str, depth_scale: float = 0.001):
        return load_depth_in_meters(depth_path, depth_scale)

    @staticmethod
    def _median_depth_window(depth_m, u, v, window=5):
        return median_depth_window(depth_m, u, v, window)

    @staticmethod
    def _pixel_to_camera_xyz(u, v, z_m, intrinsics):
        return pixel_to_camera_xyz(u, v, z_m, intrinsics)

    def _estimate_object_contact_depth(self, depth_m, bbox_xyxy):
        return estimate_object_contact_depth(depth_m, bbox_xyxy)

    def _estimate_layer_height_hms(self, depth_m, x1, y1, x2, y2, occupied_mask=None,
                                    sample_step=3, iterations=300, inlier_tol_m=0.012,
                                    min_inliers=120):
        return estimate_layer_height_hms(
            depth_m, x1, y1, x2, y2, self.camera_intrinsics,
            occupied_mask=occupied_mask, sample_step=sample_step,
            iterations=iterations, inlier_tol_m=inlier_tol_m, min_inliers=min_inliers,
        )

    @staticmethod
    def _fit_plane_tls(points_xyz, up_axis):
        return fit_plane_tls(points_xyz, up_axis)

    def _estimate_layer_height_robust_fusion(self, depth_m, x1, y1, x2, y2,
                                              occupied_mask=None, sample_step=2,
                                              inlier_tol_m=0.012, min_inliers=120,
                                              fuse_with_hms=True, **kwargs):
        return estimate_layer_height_robust_fusion(
            depth_m, x1, y1, x2, y2, self.camera_intrinsics,
            occupied_mask=occupied_mask, sample_step=sample_step,
            inlier_tol_m=inlier_tol_m, min_inliers=min_inliers,
            fuse_with_hms=fuse_with_hms, **kwargs,
        )

    def _depth_bbox_to_points(self, depth_m, bbox_xyxy, sample_step=1,
                               min_depth_m=0.15, max_depth_m=3.5):
        return depth_bbox_to_points(
            depth_m, bbox_xyxy, self.camera_intrinsics,
            sample_step=sample_step, min_depth_m=min_depth_m, max_depth_m=max_depth_m,
        )

    @staticmethod
    def _merge_plane_candidates_by_height(candidates, merge_tol_m=0.03):
        return merge_plane_candidates_by_height(candidates, merge_tol_m)

    def _detect_cabinet_planes_from_depth(self, depth_m, cabinet_bbox, method="robust_fusion",
                                           return_filtered_cloud=False):
        return detect_cabinet_planes_from_depth(
            depth_m, cabinet_bbox, self.camera_intrinsics,
            method=method, return_filtered_cloud=return_filtered_cloud,
        )

    @staticmethod
    def _assign_global_planes_to_layers(planes, shelf_layers):
        return assign_global_planes_to_layers(planes, shelf_layers)

    def _uv_to_xz_on_plane(self, u, v, plane_y_m):
        return uv_to_xz_on_plane(u, v, plane_y_m, self.camera_intrinsics)

    def _select_placement_point_from_3d_clearance(self, depth_roi, occ, layer_plane_y,
                                                    layer_plane_mask, rect_local, roi_origin):
        return select_placement_point_from_3d_clearance(
            depth_roi, occ, layer_plane_y, layer_plane_mask, rect_local, roi_origin,
            self.camera_intrinsics,
        )

    def _find_empty_spaces_by_layer(self, depth_m, cabinet_bbox, shelf_layers, obstacle_boxes, **kwargs):
        return find_empty_spaces_by_layer(
            depth_m, cabinet_bbox, shelf_layers, obstacle_boxes, self.camera_intrinsics, **kwargs
        )

    @staticmethod
    def _draw_shelf_lines_image(image, cabinet_bbox, shelf_y, shelf_layers):
        return draw_shelf_lines_image(image, cabinet_bbox, shelf_y, shelf_layers)

    @staticmethod
    def _draw_annotated_image(image, cabinet_bbox, shelf_y, shelf_layers, objects_info):
        return draw_annotated_image(image, cabinet_bbox, shelf_y, shelf_layers, objects_info)

    @staticmethod
    def _draw_experiment5_image(image, cabinet_bbox, shelf_layers, target_objects,
                                 empty_spaces, plane_segments=None):
        return draw_experiment5_image(
            image, cabinet_bbox, shelf_layers, target_objects, empty_spaces, plane_segments
        )

    @staticmethod
    def _draw_3d_overlay(image, surfaces, placements, intrinsics):
        return draw_3d_overlay(image, surfaces, placements, intrinsics)

    @staticmethod
    def _save_colored_plane_cloud(surfaces, save_path):
        return save_colored_plane_cloud(surfaces, save_path)

    @staticmethod
    def _save_plane_cloud_with_placements(surfaces, placements, save_path,
                                           marker_radius_m=0.012, marker_points=180):
        return save_plane_cloud_with_placements(
            surfaces, placements, save_path, marker_radius_m, marker_points
        )

    @staticmethod
    def _visualize_3d(all_points, surfaces, placements):
        return visualize_3d(all_points, surfaces, placements)

    def detect(self, *args, **kwargs):
        return detect(self, *args, **kwargs)

    def detect_experiment5(self, *args, **kwargs):
        return detect_experiment5(self, *args, **kwargs)

    def detect_3d(self, *args, **kwargs):
        return detect_3d(self, *args, **kwargs)
