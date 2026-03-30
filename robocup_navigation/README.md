# robocup_navigation

底盘导航模块，封装 tracer_ros 导航功能为 ROS 动作服务器。

## 功能

- **导航动作服务** (`/navigate_to_location`): 按名称导航到预定义位置
- 位置管理器（从 `config/locations.yaml` 加载）
- 集成 move_base 和 AMCL 定位

## 预定义位置（`config/locations.yaml`）

| 名称 | 说明 |
|------|------|
| `kitchen` | 厨房桌面（拾取区） |
| `dishwasher` | 洗碗机前方 |
| `cabinet` | 储物柜前方 |
| `trash_bin` | 垃圾桶旁 |
| `kitchen_surface` | 厨房台面（存放 bowl/spoon） |
| `dining_table` | 餐桌（早餐摆放目标位置） |
| `home` | 机器人初始位置 |

> **注意**：`locations.yaml` 中的坐标为占位值，需根据实际地图调整。

## 依赖

- `tracer_ros` 包（Tracer 底盘驱动）
- `move_base`
- `amcl`

## 使用

```bash
# 启动导航系统
roslaunch robocup_navigation navigation_system.launch

# 发送导航目标
rostopic pub /navigate_to_location/goal robocup_msgs/NavigateToLocationActionGoal \
  "{goal: {target_location: 'kitchen_surface', timeout: 60.0}}"
```
