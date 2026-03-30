# detect_grasp 配置说明（README）

本文档用于说明如何配置 `profile_bundle_template.json`，以及如何在 `detect_grasp` 模块中生效。

## 1. 配置文件位置与作用

- 模板文件：`profile_bundle_template.json`
- 作用：统一管理检测（detect）和抓取（grasp）的运行参数
- 结构：
  - `default_profile`：全类别默认参数
  - `category_profiles`：按类别覆盖参数（例如 `plate`、`cutlery`）

最终生效规则：

`最终配置 = default_profile + category_profiles[category] + 单次调用 profile_override`

---

## 2. 快速开始

### 2.1 最小可用步骤

1. 复制模板：
   ```bash
   cp profile_bundle_template.json profile_bundle.json
   ```
2. 修改 `profile_bundle.json` 中你关心的类别配置
3. 运行：
   ```bash
   python3 run_pipeline.py \
     --category plate \
     --input_source rgbd_files \
     --rgb_path /home/songfei/catkin_ws/src/robocup_perception/scripts/color.png \
     --depth_path /home/songfei/catkin_ws/src/robocup_perception/scripts/depth.npy \
     --profile_bundle_json /home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/detect_grasp/profile_bundle.json \
     --detect_env dsam2 \
     --grasp_env GraspGen
   ```

### 2.2 在 ROS 状态机中使用

`main_scripts/multi_round_grasp_smach.py` 里通过参数 `~profile_bundle_path` 读取配置文件。

建议在 launch 中设置：

```xml
<param name="profile_bundle_path" value="/abs/path/to/profile_bundle.json"/>
<param name="detect_env_name" value="dsam2"/>
<param name="grasp_env_name" value="GraspGen"/>
```

---

## 3. 顶层字段说明

### 3.1 `default_profile`

所有类别共用的默认配置。建议把“稳定参数”放这里。

常用字段：

- `text_prompt`：检测文本提示词（GroundingDINO）
- `target_object_policy`：目标选择策略
  - `top_conf`：选置信度最高
  - `index`：按 `target_object_index` 指定
- `segmenter_kwargs`：检测/分割相关参数
- `estimator_config`：抓姿生成与过滤参数
- `enable_visualization`：是否可视化 graspgen 结果
- `enable_tableware_pca_visualization`：是否可视化 tableware_pca 结果

### 3.2 `category_profiles`

按类别局部覆盖。只写差异项，不必重复所有参数。

示例：

```json
"category_profiles": {
  "plate": {
    "text_prompt": "plate.",
    "segmenter_kwargs": {
      "depth_range": [0.15, 1.2]
    },
    "estimator_config": {
      "grasp_threshold": 0.78
    }
  }
}
```

---

## 4. `segmenter_kwargs`（检测侧）

常用参数：

- `segmentation_method`：
  - `bbox`：仅用 bbox（速度快）
  - `sam2_mask`：SAM2 掩码（更精细）
- `depth_range`：深度范围（米），用于过滤无关点
- `box_threshold` / `text_threshold`：检测阈值
- `bbox_horizontal_shrink_ratio`：bbox 水平收缩比例
- `voxel_size`：点云体素大小
- `table_remove_thresh` / `plane_dist_thresh`：桌面剔除相关阈值
- `visual_preset`：相机预设（相机模式下）

模型路径相关（worker 启动后固定，不建议运行中改）：

- `gdino_config`
- `gdino_checkpoint`
- `sam2_config`
- `sam2_checkpoint`
- `device`

> 说明：这些“模型级参数”在 detect worker 初始化后会锁定，若后续改动可能触发报错（需重启 runner）。

---

## 5. `estimator_config`（抓取侧）

常用参数：

- `grasp_threshold`：抓姿置信度阈值
- `num_grasps`：采样抓姿数量
- `topk_num_grasps`：返回 Top-K 抓姿数
- `filter_collisions` / `collision_threshold`：碰撞过滤
- `max_depth_m`：参与抓取的最大深度
- `enable_z_angle_filter` / `max_z_angle_deg`：抓姿与世界 z 轴角度过滤
- `enable_x_region_orientation_filter`：按 x 区域方向过滤
- `enable_y_axis_angle_filter`：按 y 轴角度过滤
- `enable_world_z_axis_angle_filter`：基于外参做世界坐标 z 轴过滤

模型路径相关（worker 启动后固定）：

- `gripper_config`

> 说明：`gripper_config` 改动会触发 grasp worker 重新初始化需求，常规建议固定后使用。

---

## 6. tableware_pca 模式专项

当 `grasp_mode=tableware_pca` 时，配置重点：

- `tableware_pca_text_prompt`
- `tableware_pca_depth_range`
- `tableware_pca_grasp_tilt_x_deg`
- `enable_tableware_pca_visualization`

注意：该模式下会强制走 `sam2_mask` 分割流程。

---

## 7. 常见配置建议

### 7.1 盘子（plate）

- `text_prompt`: `"plate."`
- `segmenter_kwargs.depth_range`: 可适当缩窄，减少背景干扰
- `estimator_config.grasp_threshold`: 0.75~0.85 之间根据场景调

### 7.2 餐具（fork/spoon/knife）

- 推荐 `grasp_mode=tableware_pca`
- `tableware_pca_text_prompt`: `"fork. spoon. knife."`
- 如倾斜抓取不稳定，可减小 `tableware_pca_grasp_tilt_x_deg` 的绝对值

---

## 8. 迁移到新主机时必须确认

1. 两个 conda 环境存在且可用：
   - 检测环境：`dsam2`
   - 抓取环境：`GraspGen`
2. `conda` 可执行文件可被找到（或传绝对路径）
3. Grounded-SAM-2 路径可用（`GSAM2_ROOT`）
4. `GraspGen` 代码与模型文件完整
5. 如果用 `input_source=camera`：
   - `pyrealsense2` 安装正常
   - 相机权限与驱动正常

---

## 9. 配置排错清单

- 报 `No objects detected`：
  - 检查 `text_prompt`、`box_threshold`、`text_threshold`
  - 检查输入图像是否正确
- 报 `no_valid_points`：
  - 检查 `depth_range` 是否过窄
  - 检查 depth 单位设置（米 or 毫米）
- 报模型路径错误：
  - 检查 `GSAM2_ROOT`、`gripper_config`、checkpoint/config 路径
- 报 worker 初始化后配置变化：
  - 重启 `DetectGraspRunner`，避免在运行中改模型级参数

---

## 10. 推荐实践

- 把稳定配置固化在 `default_profile`
- 每个类别只在 `category_profiles` 写差异
- 现场临时调参使用 `profile_override`，验证后再回写配置文件
- 新场景先用 `rgbd_files` 离线调好，再切到 `camera` 在线运行
