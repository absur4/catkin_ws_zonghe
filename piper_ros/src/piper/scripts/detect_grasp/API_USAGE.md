# detect_grasp API 使用指南

## 1. 推荐配置方式：单个 JSON

将默认参数和类别参数放到一个文件中，例如 `profile_bundle.json`：

```json
{
  "default_profile": {
    "text_prompt": "bottle.",
    "target_object_policy": "top_conf",
    "target_object_index": 0,
    "segmenter_kwargs": {
      "segmentation_method": "sam2_mask",
      "bbox_horizontal_shrink_ratio": 0.1,
      "depth_range": [0.1, 1.0],
      "voxel_size": 0.01,
      "table_remove_thresh": 0.01,
      "box_threshold": 0.35,
      "text_threshold": 0.25
    },
    "estimator_config": {
      "grasp_threshold": 0.8,
      "num_grasps": 200,
      "topk_num_grasps": 20,
      "collision_threshold": 0.001,
      "max_z_angle_deg": 30.0,
      "enable_y_axis_angle_filter": true,
      "min_y_axis_angle_deg": 90.0
    },
    "enable_visualization": true,
    "enable_tableware_pca_visualization": false,
    "tableware_pca_text_prompt": "fork. spoon.",
    "tableware_pca_depth_range": [0.1, 0.8],
    "tableware_pca_grasp_tilt_x_deg": -30.0,
    "tableware_pca_visualization_max_points": 120000,
    "num_visualize_grasps": 10,
    "block_after_visualization": false
  },
  "category_profiles": {
    "cola": {
      "text_prompt": "cola can.",
      "segmenter_kwargs": {
        "bbox_horizontal_shrink_ratio": 0.08
      },
      "estimator_config": {
        "grasp_threshold": 0.82
      }
    },
    "bottle": {
      "text_prompt": "bottle.",
      "segmenter_kwargs": {
        "segmentation_method": "bbox"
      }
    }
  }
}
```

运行时合并规则：

`最终参数 = default_profile + category_profiles[category] + profile_override(单次调用)`

## 2. CLI 使用

```bash
python3 sg_pkg/scripts/detect_grasp/run_pipeline.py \
  --category cola \
  --input_source rgbd_files \
  --rgb_path /path/color.png \
  --depth_path /path/depth.png \
  --profile_bundle_json /path/profile_bundle.json
```

兼容旧方式：仍可使用 `--default_profile_json` 和 `--category_profiles_json`。

## 3. Python API 使用

```python
import json
from sg_pkg.scripts.detect_grasp.pipeline import DetectGraspConfig, DetectGraspRunner

with open('/path/profile_bundle.json', 'r', encoding='utf-8') as f:
    bundle = json.load(f)

cfg = DetectGraspConfig(
    detect_env_name='dsam2',
    grasp_env_name='GraspGen',
    input_source='rgbd_files',
    rgb_path='/path/color.png',
    depth_path='/path/depth.png',
    default_profile=bundle.get('default_profile', {}),
    category_profiles=bundle.get('category_profiles', {}),
)

with DetectGraspRunner(cfg) as runner:
    result_cola = runner.run(category='cola')
    result_bottle = runner.run(category='bottle')

    # 单次调用临时覆盖
    result_snack = runner.run(
        category='snack',
        profile_override={
            'estimator_config': {
                'grasp_threshold': 0.72,
                'topk_num_grasps': 25
            }
        }
    )
```

## 4. 哪些参数可以修改

可热切换（每次请求可改，不重载模型）：

- 顶层 profile 字段：
  - `text_prompt`
  - `target_object_policy`
  - `target_object_index`
  - `enable_visualization`
  - `enable_tableware_pca_visualization`
  - `tableware_pca_text_prompt`
  - `tableware_pca_depth_range`
  - `tableware_pca_grasp_tilt_x_deg`
  - `tableware_pca_visualization_max_points`
  - `num_visualize_grasps`
  - `block_after_visualization`
- `segmenter_kwargs` 运行参数：
  - `segmentation_method`
  - `voxel_size`
  - `plane_dist_thresh`
  - `table_remove_thresh`
  - `depth_range`
  - `box_threshold`
  - `text_threshold`
  - `bbox_horizontal_shrink_ratio`
  - `visual_preset`
- `estimator_config` 运行参数：
  - `voxel_size`
  - `max_points`
  - `obj_ratio`
  - `disable_stratified_sampling`
  - `disable_obj_outlier_removal`
  - `obj_outlier_threshold`
  - `obj_outlier_k`
  - `obj_mask_match_decimals`
  - `max_depth_m`
  - `grasp_threshold`
  - `num_grasps`
  - `return_topk`
  - `topk_num_grasps`
  - `filter_collisions`
  - `collision_threshold`
  - `collision_local_radius_m`
  - `collision_scene_sampling_mode`
  - `max_scene_points`
  - `enable_z_angle_filter`
  - `max_z_angle_deg`
  - `enable_x_region_orientation_filter`
  - `x_region_positive_threshold`
  - `x_region_negative_threshold`
  - `x_region_positive_max_angle_deg`
  - `x_region_negative_min_angle_deg`
  - `enable_y_axis_angle_filter`
  - `min_y_axis_angle_deg`
  - `enable_world_z_axis_angle_filter`
  - `min_world_z_axis_angle_deg`
  - `viz_bounds_min`
  - `viz_bounds_max`
  - `random_seed`

另外，`runner.run(...)` 在 `graspgen` 模式下可传入两段外参（不传则使用内置默认值）：

- `camera_to_ee_transform`（支持 `{"translation"/"xyz", "quaternion"/"q"}` 或 4x4 矩阵）
- `ee_to_base_transform`（支持 `{"translation"/"xyz", "quaternion"/"q"}` 或 4x4 矩阵）

不可热切换（worker 启动后固定；改动会报错）：

- detect hub: `gdino_config`, `gdino_checkpoint`, `sam2_config`, `sam2_checkpoint`, `device`
- grasp hub: `gripper_config`

## 5. 输出字段

- `best_grasp_pose`
- `best_grasp_conf`
- `result_json_path`（当前不落盘，固定 `None`）
- `result_payload`（完整内存结果）

## 6. 参考示例（纯代码调用 API）

示例文件不使用命令行参数，提供可直接在代码中调用的函数：`run_multi_category_grasp(...)`。

调用方式示例：

```python
from sg_pkg.scripts.detect_grasp.example_api_usage import run_multi_category_grasp

results = run_multi_category_grasp(
    profile_bundle_path='/path/profile_bundle.json',
    rgb_path='/path/color.png',
    depth_path='/path/depth.png',
    categories=['cola', 'bottle'],
)
```

文件见：[example_api_usage.py](/home/h/PPC/src/sg_pkg/scripts/detect_grasp/example_api_usage.py)
