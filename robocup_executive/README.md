# robocup_executive

SMACH 状态机核心模块，实现 Pick and Place 任务的高层控制逻辑。

---

## 目录结构

```
robocup_executive/
├── README.md                        # 本文档
├── DRY_RUN_GUIDE.md                 # 流程演练脚本使用指南
├── CMakeLists.txt                   # ROS 包构建配置
├── package.xml                      # 包依赖声明
│
├── config/
│   └── task_config.yaml             # 任务全局参数（超时、物品数、早餐配置等）
│
├── launch/
│   ├── pick_place_task.launch       # 仅启动状态机节点
│   └── system_bringup.launch        # 启动完整系统（导航+感知+机械臂+状态机）
│
├── scripts/                         # 可执行入口脚本（rosrun 直接调用）
│   ├── pick_place_task.py           # 正式任务主入口：组装并运行完整 SMACH 状态机
│   ├── pick_place_dry_run.py        # 演练版：保留真实导航，其余全部 print 模拟
│   └── breakfast_task.py            # 占位符（早餐独立任务，暂未实现）
│
└── src/robocup_executive/           # Python 包（供 scripts/ 导入）
    ├── __init__.py                  # 包标识（空文件）
    ├── task_data.py                 # TaskUserData 数据类：定义所有 SMACH userdata 字段
    └── states/                      # 每个 SMACH 状态对应一个独立文件
        ├── __init__.py              # 统一导出所有状态类
        ├── init_system.py           # InitSystem：检查服务/action 可用性
        ├── navigate_to_kitchen.py   # NavigateToKitchen：导航到厨房桌边（真实 action）
        ├── assess_scene.py          # AssessScene：调用 /detect_objects，感知公示（Rule #11）
        ├── select_target.py         # SelectTarget：选下一件物品，调用 /classify_object 分类
        ├── execute_pick.py          # ExecutePick：调用 /pick_object action 执行抓取
        ├── navigate_to_dest.py      # NavigateToDest：导航到分类决定的目的地
        ├── perceive_dest.py         # PerceiveDest：检测柜子层板，计算放置位姿
        ├── execute_place.py         # ExecutePlace：调用 /place_object；洗碗机时 TTS 通知 referee
        ├── navigate_back_to_kitchen.py  # NavigateBackToKitchen：放置后返回桌边准备下一件
        ├── serve_breakfast.py       # ServeBreakfast：完整早餐子流程（nav+detect+pick+place ×4）
        └── task_completed.py        # TaskCompleted：输出抓取/放置统计信息
```

### 关键文件说明

| 文件 | 作用 |
|------|------|
| `scripts/pick_place_task.py` | **正式比赛入口**。调用 `create_cleanup_loop()` 和 `create_main_state_machine()` 组装双层 SMACH，启动 introspection 服务器后执行 |
| `scripts/pick_place_dry_run.py` | **演练入口**。所有状态内联在单文件中；导航真实发送，感知/抓取/放置全部 print。详见 `DRY_RUN_GUIDE.md` |
| `scripts/breakfast_task.py` | 占位符，仅做节点初始化，尚未实现 |
| `src/robocup_executive/task_data.py` | 定义 `TaskUserData` dataclass，列出全部 userdata 字段及默认值，作为文档参考（运行时由 SMACH userdata 机制传递，不直接实例化） |
| `src/robocup_executive/states/__init__.py` | 统一 import 所有状态类，`pick_place_task.py` 只需 `from robocup_executive.states import ...` |
| `config/task_config.yaml` | 任务级参数：超时、物品数上限、早餐物品列表、导航/抓取/放置各阶段超时 |
| `launch/pick_place_task.launch` | 仅启动状态机，适合单独调试（需手动确保其他节点已启动） |
| `launch/system_bringup.launch` | 一键拉起全系统（包含导航、感知、机械臂、状态机） |
| `DRY_RUN_GUIDE.md` | 演练脚本的完整说明：代码结构、启动方法、如何修改模拟物品位置 |

---

## 状态机结构

### 外层状态机

```
INIT_SYSTEM
    → NAVIGATE_TO_KITCHEN
        → ASSESS_SCENE
            → TABLE_CLEANUP_LOOP（内层循环，处理桌面所有物品）
                → SERVE_BREAKFAST（可选，规则书 §5.2 早餐准备）
                    → TASK_COMPLETED
```

### TABLE_CLEANUP_LOOP（内层循环）

```
SELECT_TARGET → EXECUTE_PICK → NAVIGATE_TO_DEST → PERCEIVE_DEST → EXECUTE_PLACE
      ↑                                                                    |
      |←←←←←←←←←←←← NAVIGATE_BACK_TO_KITCHEN ←←←←←←←←←←←←←←←←←←←←←←←|
                        （循环直到无物品退出）
```

**注意**：每次放置完成后，机器人通过 `NAVIGATE_BACK_TO_KITCHEN` 返回厨房桌边，才能继续抓取下一件物品。

---

## 状态说明

| 状态 | 文件 | 说明 |
|------|------|------|
| `InitSystem` | `states/init_system.py` | 检查 `/detect_objects`、`/classify_object`、`/compute_grasp_pose`、`/compute_place_pose` 服务及三个 action server 可用性 |
| `NavigateToKitchen` | `states/navigate_to_kitchen.py` | 向 `/navigate_to_location` 发送 `target_location="kitchen"`，等待到达 |
| `AssessScene` | `states/assess_scene.py` | 等待 RGB-D 图像，调用 `/detect_objects`，输出感知公示日志（Rule #11） |
| `SelectTarget` | `states/select_target.py` | 按 index 选下一件物品，调用 `/classify_object` 获取 `category` 和 `destination` |
| `ExecutePick` | `states/execute_pick.py` | 调用 `/pick_object` action（`grasp_strategy="auto"`），读取 `result.picked_pose` |
| `NavigateToDest` | `states/navigate_to_dest.py` | 向 `/navigate_to_location` 发送 `target_location=destination`（dishwasher / trash_bin / cabinet） |
| `PerceiveDest` | `states/perceive_dest.py` | destination 为 cabinet 时调用 `/detect_shelf` 检测层板；调用 `/compute_place_pose` 计算放置位姿 |
| `ExecutePlace` | `states/execute_place.py` | 调用 `/place_object` action；destination 为 dishwasher 时通过 `/tts/say` 通知 referee 开门（Rule #4） |
| `NavigateBackToKitchen` | `states/navigate_back_to_kitchen.py` | 放置完成后，向 `/navigate_to_location` 发送 `target_location="kitchen"`，返回桌边 |
| `ServeBreakfast` | `states/serve_breakfast.py` | 按序处理 bowl→spoon（kitchen_surface）、cereal→milk（cabinet），每件：nav→detect→pick→nav→place |
| `TaskCompleted` | `states/task_completed.py` | 打印 `objects_picked_count`、`objects_placed_count`、`failed_objects` 统计信息 |

---

## 配置

### `config/task_config.yaml` 关键参数

| 参数 | 值 | 说明 |
|------|----|------|
| `task.total_timeout` | 420.0 s | 规则书最大时长 7 分钟 |
| `task.max_objects` | 7 | 桌面固定 7 件物品 |
| `task.enable_breakfast_serving` | true | 默认开启早餐准备 |
| `task.communicate_perception` | true | 感知公示开关（Rule #11） |
| `breakfast.items[bowl].location` | `kitchen_surface` | 规则书：bowl/spoon 在厨房台面 |
| `breakfast.items[milk].location` | `cabinet` | 规则书：cereal/milk 在柜子内 |

### 如何修改物品目的地

物品目的地由 **`robocup_perception/config/classification_rules.yaml`** 统一控制，共三类：

| 类别 | 目的地 | 修改方式 |
|------|--------|----------|
| `cleanable` | `dishwasher` | 在 `classification_rules.yaml` 的 `cleanable.keywords` 列表中添加/删除物品名 |
| `trash` | `trash_bin` | 运行时执行：`rosparam set /object_classifier/trash_keywords "[bottle]"` |
| `other`（默认） | `cabinet` | 不在以上两类中的物品自动归入 |

**示例**：让 `plate` 去柜子而非洗碗机——在 `classification_rules.yaml` 中将 `plate` 从 `cleanable.keywords` 删除即可。

### 赛前配置（Setup Days）

```bash
# 设置当场次垃圾类（示例：本场 bottle 为垃圾）
rosparam set /object_classifier/trash_keywords "[bottle]"
```

---

## 使用

```bash
# 1. 启动完整系统（导航 + 感知 + 机械臂 + 状态机）
roslaunch robocup_executive system_bringup.launch

# 2. 或分步启动（调试用）
roslaunch robocup_navigation navigation_system.launch
roslaunch robocup_perception perception_system.launch
roslaunch robocup_manipulation manipulation_system.launch
rosrun robocup_executive pick_place_task.py

# 3. 演练模式（只需底盘，感知/抓取/放置全部 print）
roslaunch robocup_navigation navigation_system.launch
rosrun robocup_executive pick_place_dry_run.py
# 启用早餐流程：
rosrun robocup_executive pick_place_dry_run.py _enable_breakfast:=true

# 4. 可视化状态机
rosrun smach_viewer smach_viewer.py
```
