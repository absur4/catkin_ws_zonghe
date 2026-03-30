# robocup_perception

语义感知模块，封装 Grounded-SAM-2 视觉检测功能为 ROS 服务。

## 服务列表

| 服务 | 类型 | 说明 |
|------|------|------|
| `/detect_objects` | DetectObjects | 使用 Grounding DINO + SAM 2 检测并分割物体，返回坐标已转换至 `base_link` 帧 |
| `/detect_shelf` | DetectShelf | 检测柜子层板结构，返回 `ShelfResult`（含各层 `ShelfLayer`） |
| `/compute_grasp_pose` | ComputeGraspPose | 计算 Approach-Grasp-Retreat 三点抓取姿态 |
| `/compute_place_pose` | ComputePlacePose | 计算柜子放置位置（按相似性分组） |
| `/classify_object` | ClassifyObject | 根据物品名称分类并返回目的地 |

## 物品分类规则（rulebook2026 §5.2）

分类规则定义于 `config/classification_rules.yaml`：

| 类别 | 关键词 | 目的地 |
|------|--------|--------|
| `cleanable` | cup, mug, plate, dish, bowl, spoon, fork, knife | `dishwasher` |
| `trash` | *(Setup Days 通过 rosparam 动态注入)* | `trash_bin` |
| `other`（默认） | 其余所有物品 | `cabinet` |

### 赛前注入垃圾类关键词

```bash
# 启动前设置垃圾品类（Setup Days）
rosparam set /object_classifier/trash_keywords "[bottle]"
```

## TF 坐标变换

`/detect_objects` 返回的 `pose.position` 已通过 TF 从 `camera_color_optical_frame` 变换到 `base_link`。如需修改相机帧名称：

```bash
# 在 launch 文件中传入参数
<param name="camera_frame_id" value="camera_color_optical_frame"/>
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/camera_params.yaml` | 相机内参（fx, fy, cx, cy, depth_scale） |
| `config/classification_rules.yaml` | 物品分类规则（三类：cleanable/trash/other） |

## 依赖

- `open_vocabulary` 包（Grounded-SAM-2 API）
- RealSense D435 相机
- GPU / CUDA（推荐）
- `tf2_geometry_msgs`

## 使用

```bash
# 启动所有感知服务
roslaunch robocup_perception perception_system.launch

# 单独启动物体检测服务
rosrun robocup_perception object_detector_server.py

# 测试分类（cup → dishwasher）
rosservice call /classify_object "object_name: 'cup'"

# 动态设置垃圾类后测试
rosparam set /object_classifier/trash_keywords "[bottle]"
rosservice call /classify_object "object_name: 'bottle'"
```
