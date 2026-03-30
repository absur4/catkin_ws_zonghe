from .shelf_geometry import (
    detect_shelf_lines,
    cluster_horizontal_lines,
    compute_shelf_layers,
    assign_object_to_layer,
    generate_equal_layers,
)

from .pointcloud_geometry import (
    HorizontalSurface,
    PlacementPoint,
    load_and_preprocess_pointcloud,
    hms_ransac,
    reconstruct_surface_bounds,
    compute_placement_points,
    find_largest_free_rectangle,
    project_points_to_image,
    surfaces_to_shelf_y,
    detect_horizontal_surfaces,
)

from .bbox_utils import (
    bbox_iou_xyxy,
    nms_xyxy_numpy,
    keep_only_lower_row_boxes,
)

from .depth_utils import (
    load_camera_intrinsics_from_file,
    load_depth_in_meters,
    median_depth_window,
    pixel_to_camera_xyz,
    estimate_object_contact_depth,
)

from .plane_estimation import (
    fit_plane_tls,
    estimate_layer_height_hms,
    estimate_layer_height_robust_fusion,
    depth_bbox_to_points,
    merge_plane_candidates_by_height,
    detect_cabinet_planes_from_depth,
    assign_global_planes_to_layers,
)

from .empty_space import (
    find_largest_free_rectangle as find_largest_free_rectangle_2d,
    find_top_k_free_rectangles,
    uv_to_xz_on_plane,
    select_placement_point_from_3d_clearance,
    find_empty_spaces_by_layer,
)

from .visualization import (
    LAYER_COLORS,
    draw_shelf_lines_image,
    draw_annotated_image,
    draw_experiment5_image,
    draw_3d_overlay,
    save_colored_plane_cloud,
    save_plane_cloud_with_placements,
    visualize_3d,
)

from .experiment5 import (
    EXPERIMENT5_DEFAULT_OBJECT_PROMPT,
    EXPERIMENT5_TARGET_KEYWORDS,
    normalize_text_label,
    map_label_to_experiment5_target,
)
