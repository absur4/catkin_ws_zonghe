# robocup_manipulation

机械臂控制模块，封装 Piper 机械臂和 MoveIt 运动规划为 ROS 动作服务器。

## 功能

- **抓取动作服务** (`/pick_object`): 6 阶段完整抓取流程
- **放置动作服务** (`/place_object`): 4 阶段完整放置流程
- MoveIt 接口封装
- 夹爪控制器

## 动作接口

### `/pick_object` (PickObject.action)

| 字段 | 类型 | 说明 |
|------|------|------|
| Goal: `target_object` | DetectedObject | 抓取目标 |
| Goal: `grasp_strategy` | string | `"auto"` / `"top_down"` / `"side"` |
| Feedback: `current_phase` | string | 当前阶段（接近/抓取/抬升/撤退） |
| Feedback: `progress` | float32 | 0.0 ~ 1.0 |
| Result: `success` | bool | 是否成功 |
| Result: `picked_pose` | Pose | 实际抓取位姿 |

### `/place_object` (PlaceObject.action)

| 字段 | 类型 | 说明 |
|------|------|------|
| Goal: `target_pose` | Pose | 放置目标位置 |
| Goal: `place_strategy` | string | `"gentle"`（轻柔放置）/ `"fast"`（洗碗机等快速放置） |
| Feedback: `current_phase` | string | 当前阶段（接近/下降/释放/撤退） |
| Result: `success` | bool | 是否成功 |
| Result: `placed_pose` | Pose | 实际放置位姿 |

## 抓取流程（6 阶段）

```
1. 计算抓取姿态  → 调用 /compute_grasp_pose
2. 接近物体      → 移动到 pre_grasp（物体上方）
3. 打开夹爪      → 张开到 gripper_width + 2cm 余量
4. 下降抓取      → 笛卡尔路径下降到 grasp 位置
5. 闭合夹爪      → 食物类 40N，其他 80N
6. 抬升撤退      → 抬升到 post_grasp，返回 transport 姿态
```

## 放置流程（4 阶段）

```
1. 接近放置点  → 移动到目标上方 15cm（普通规划）
2. 下降        → 笛卡尔路径下降到目标位置
3. 释放        → 打开夹爪 7cm
4. 撤退        → 抬升 15cm，返回 home 姿态
```

## 依赖

- `piper_ros` 包（Piper 机械臂驱动）
- MoveIt
- `piper_with_gripper_moveit` 配置

## 使用

```bash
# 启动机械臂系统
roslaunch robocup_manipulation manipulation_system.launch
```
