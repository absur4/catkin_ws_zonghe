# Pick & Place 流程演练指南（Dry Run Guide）

## 1. 概述

`pick_place_dry_run.py` 是 `pick_place_task.py` 的**轻量化演练版本**。

原始状态机需要相机、机械臂、感知服务（`/detect_objects`、`/classify_object`、`/pick_object`、`/place_object`）全部在线才能运行。本脚本的目标是：

> **在只有底盘导航正常的情况下，跑通完整 SMACH 状态机流程，验证状态转换逻辑和导航路径。**

感知、抓取、放置全部替换为终端打印（`rospy.loginfo`）；真实的 `/navigate_to_location` action 调用完整保留。

---

## 2. 代码结构

所有类内联在单一文件中，无需额外 Python 包：

| 类 / 函数 | 职责 | 模式 |
|-----------|------|------|
| `MockDetectedObject` | 模拟 `robocup_msgs/DetectedObject` 消息结构 | 辅助类 |
| `InitSystem` | 系统初始化 | **模拟**（直接返回 `initialized`）|
| `NavigateToKitchen` | 导航到厨房 | **真实**（`/navigate_to_location` action）|
| `AssessScene` | 评估桌面，检测物品 | **模拟**（返回 `MOCK_TABLE_OBJECTS`）|
| `SelectTarget` | 选择下一件物品并分类 | **模拟**（本地关键词分类，不调用 `/classify_object`）|
| `ExecutePick` | 执行抓取 | **模拟**（打印 6 个阶段）|
| `NavigateToDest` | 导航到目的地（dishwasher / trash_bin / cabinet）| **真实**（`/navigate_to_location` action）|
| `PerceiveDest` | 感知目的地，计算放置姿态 | **模拟**（返回默认 Pose）|
| `ExecutePlace` | 执行放置 | **模拟**（打印 4 个阶段，含洗碗机门提示）|
| `NavigateBackToKitchen` | 放置后返回厨房桌边 | **真实**（`/navigate_to_location` action）|
| `ServeBreakfast._navigate_to()` | 早餐任务中的导航 | **真实**（`/navigate_to_location` action）|
| `ServeBreakfast._compute_placement_pose()` | 计算桌面放置位姿 | **真实**（纯数学，无 ROS 调用）|
| `ServeBreakfast._detect_item()` | 检测早餐物品 | **模拟**（返回 `MOCK_BREAKFAST_OBJECTS`）|
| `ServeBreakfast._pick_item()` | 抓取早餐物品 | **模拟**（打印 6 个阶段）|
| `ServeBreakfast._place_item()` | 放置早餐物品 | **模拟**（打印 4 个阶段）|
| `TaskCompleted` | 打印统计信息 | **真实**（完全复制，无硬件调用）|
| `create_cleanup_loop()` | 构建内层清理循环状态机 | 完全复制 + 新增 `NAVIGATE_BACK_TO_KITCHEN` 状态 |
| `create_main_state_machine()` | 构建外层主状态机 | 完全复制 |
| `main()` | 初始化 ROS 节点、执行状态机 | 完全复制（节点名改为 `pick_place_dry_run`）|

---

## 3. 保留 vs 替换对照表

### 保留（真实 ROS 调用）

| 状态 / 方法 | 说明 |
|-------------|------|
| `NavigateToKitchen.execute()` | 向 `/navigate_to_location` 发送 goal，等待 result |
| `NavigateToDest.execute()` | 同上，目标地点由 `destination` userdata 决定 |
| `NavigateBackToKitchen.execute()` | 放置完成后返回 `kitchen`，为下一件物品就位 |
| `ServeBreakfast._navigate_to()` | 同上，用于早餐任务中各个地点间的移动 |
| `ServeBreakfast._compute_placement_pose()` | 纯数学：根据 index、pair_offsets 计算桌面 Pose |

### 替换为 print（不发 ROS 消息）

| 原始逻辑 | 干跑替换内容 |
|---------|-------------|
| `InitSystem`：`wait_for_service` × 4，检查 action server × 3 | 打印 `模拟检查... OK`，直接返回 `initialized` |
| `AssessScene`：等待相机 + 调用 `/detect_objects` | 打印模拟物品列表，返回 `MOCK_TABLE_OBJECTS` 实例 |
| `SelectTarget`：调用 `/classify_object` | 本地关键词匹配（`CLEANABLE_KEYWORDS` + rosparam `trash_keywords`）|
| `ExecutePick`：等待 `/pick_object` + send_goal | 打印 6 个抓取阶段，返回 `pick_succeeded`，`grasp_pose=None` |
| `PerceiveDest`：等待相机 + `/detect_shelf` + `/compute_place_pose` | 打印跳过提示，返回默认 `Pose`，返回 `perception_done` |
| `ExecutePlace`：TTS pub + `/place_object` | 打印（含洗碗机门提示），返回 `place_succeeded` |
| `ServeBreakfast._detect_item()` | 从 `MOCK_BREAKFAST_OBJECTS` 返回对应物品 |
| `ServeBreakfast._pick_item()` | 打印 6 个抓取阶段，返回 `True` |
| `ServeBreakfast._place_item()` | 打印 4 个放置阶段，返回 `True` |

---

## 4. 启动方法

### 前提条件

只需底盘 + 导航系统在线：

```bash
roslaunch robocup_navigation navigation_system.launch
```

不需要：相机、机械臂、感知服务、`/pick_object`、`/place_object`。

### 运行干跑脚本

```bash
rosrun robocup_executive pick_place_dry_run.py
```

如需启用早餐任务（默认跳过），在启动参数中设置：

```bash
rosrun robocup_executive pick_place_dry_run.py _enable_breakfast:=true
```

---

## 5. 如何修改模拟物品位置

### 5.1 桌面清理物品（`MOCK_TABLE_OBJECTS`）

在 `pick_place_dry_run.py` 顶部找到：

```python
MOCK_TABLE_OBJECTS = [
    ('cup',   0.85, 0.55,  0.10, 0.30),
    ('plate', 0.72, 0.60, -0.05, 0.28),
    ('apple', 0.65, 0.58,  0.20, 0.32),
]
```

**每行格式**：`('物品名', 置信度, x坐标, y坐标, z坐标)`

| 参数 | 含义 | 单位 |
|------|------|------|
| 物品名 | 类别名称（影响分类规则） | — |
| 置信度 | 模拟检测置信度，0.0 ~ 1.0 | — |
| x | 机器人正前方距离（到达 `kitchen` 后，`base_link` 帧） | 米 |
| y | 左正右负（机器人左侧为正） | 米 |
| z | 桌面高度 | 米 |

**示例**：将 cup 修改到桌面正前方 0.6m、偏左 0.1m、高度 0.3m：

```python
('cup', 0.85, 0.60, 0.10, 0.30),
```

**分类规则**（影响目的地）：
- `cup / mug / plate / dish / bowl / spoon / fork / knife` → `dishwasher`（可清洗）
- rosparam `/object_classifier/trash_keywords` 中的词 → `trash_bin`
- 其余 → `cabinet`

### 5.2 早餐物品（`MOCK_BREAKFAST_OBJECTS`）

在 `MockDetectedObject` 类定义之后找到：

```python
MOCK_BREAKFAST_OBJECTS = {
    'bowl':   MockDetectedObject('bowl',   0.88, 0.50,  0.00, 0.30),
    'spoon':  MockDetectedObject('spoon',  0.82, 0.52,  0.08, 0.30),
    'cereal': MockDetectedObject('cereal', 0.76, 0.50, -0.05, 0.35),
    'milk':   MockDetectedObject('milk',   0.79, 0.52,  0.05, 0.38),
}
```

字典键为物品名（与 `ServeBreakfast.breakfast_items` 中的 `name` 对应）。坐标是机器人到达各自 `source` 位置（`kitchen_surface` 或 `cabinet`）后的 `base_link` 帧坐标。

**添加物品**：直接在 `MOCK_TABLE_OBJECTS` 列表中增加一行。

**增减早餐物品**：修改 `MOCK_BREAKFAST_OBJECTS` 字典，同时修改 `ServeBreakfast.breakfast_items` 列表（两处保持一致）。

---

## 6. 可自定义项汇总

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `MOCK_TABLE_OBJECTS` | 文件顶部常量 | 模拟桌面物品及位置 |
| `CLEANABLE_KEYWORDS` | 文件顶部常量 | 决定哪些物品送洗碗机 |
| `MOCK_BREAKFAST_OBJECTS` | `MockDetectedObject` 类之后 | 模拟早餐物品及位置 |
| `~enable_breakfast` | rosparam | `true` 才会执行早餐流程（默认 `false`）|
| `~breakfast/item_spacing` | rosparam | 桌面摆放间距（默认 0.15m）|
| `/object_classifier/trash_keywords` | rosparam | 垃圾关键词列表（默认空）|

---

## 7. SMACH 可视化

在另一个终端运行：

```bash
rosrun smach_viewer smach_viewer.py
```

可实时观察状态机当前所处状态和状态转换。introspection 服务器的命名空间为 `/PICK_PLACE_TASK`。

---

## 8. 预期输出流程

```
[DRY RUN] 初始化系统 → 打印 7 条模拟检查 OK
↓
导航到厨房（真实导航，等待 /navigate_to_location）
↓
[DRY RUN] 评估场景 → 打印 3 个模拟物品
↓
内层循环（3 件物品，每轮）：
  └── [DRY RUN] 选择物品 → 本地分类 → 打印目的地
  └── [DRY RUN] 抓取 → 打印 6 个阶段
  └── 真实导航到目的地（dishwasher / trash_bin / cabinet）
  └── [DRY RUN] 感知目的地 → 返回默认 Pose
  └── [DRY RUN] 放置 → 打印 4 个阶段（dishwasher 时打印开门提示）
  └── 真实导航返回 kitchen（NAVIGATE_BACK_TO_KITCHEN）
  └── → SELECT_TARGET 处理下一件（直到无物品退出循环）
↓
（若 enable_breakfast=true）准备早餐：
  └── 真实导航到 kitchen_surface → [DRY RUN] 检测 bowl → [DRY RUN] 抓取 → 真实导航到 dining_table → [DRY RUN] 放置
  └── 真实导航到 kitchen_surface → [DRY RUN] 检测 spoon → ...
  └── 真实导航到 cabinet → [DRY RUN] 检测 cereal → ...
  └── 真实导航到 cabinet → [DRY RUN] 检测 milk → ...
↓
任务完成 → 打印统计信息
```
