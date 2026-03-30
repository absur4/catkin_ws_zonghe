# Tracer + MID360 切换到 Fast-LIO 定位方案（详细步骤）

本文档用于替代 `tracer_ros/readme_sf.md` 中的“雷达 + hector/gmapping”流程，改为 **Fast-LIO 定位/里程计** 方案。

适用环境（按你当前工程）：
- ROS1 Noetic + Catkin 工作空间：`/home/songfei/catkin_ws`
- 底盘包：`/home/songfei/catkin_ws/src/tracer_ros`
- 雷达驱动：`/home/songfei/catkin_ws/src/livox_ros_driver2`
- 雷达型号：MID360（IP 示例：`192.168.1.167`）

---

## 0. 先说明：Fast-LIO“定位”含义

Fast-LIO 本质是激光-惯导里程计（LIO），输出高频位姿与局部地图，常用于：
- 机器人实时定位（相对起点）
- 给导航/控制提供更稳定的里程计

它不是传统 2D AMCL 全局重定位。如果你后面需要“已知地图上的全局定位”，可再叠加全局定位模块。

---

## 1. 一次性准备（只做一次）

### 1.1 安装基础依赖

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-pcl-ros \
  ros-noetic-tf \
  ros-noetic-tf2-ros \
  ros-noetic-geometry-msgs \
  ros-noetic-nav-msgs \
  ros-noetic-sensor-msgs \
  ros-noetic-visualization-msgs \
  libeigen3-dev libgoogle-glog-dev libgflags-dev
```

### 1.2 安装/确认 Livox-SDK2

如果你已经按旧文档装过，可以跳过；否则：

```bash
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 1.3 编译 livox_ros_driver2

```bash
cd /home/songfei/catkin_ws/src/livox_ros_driver2
./build.sh ROS1
```

### 1.4 获取 Fast-LIO

```bash
cd /home/songfei/catkin_ws/src
git clone --recursive https://github.com/hku-mars/FAST_LIO.git
cd /home/songfei/catkin_ws
catkin_make
source devel/setup.bash
```

建议把 `source` 写入 `~/.bashrc`：

```bash
echo "source /home/songfei/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 2. 每次上电前检查（非常关键）

### 2.1 给雷达网口配置静态 IP（临时方式）

先确认网卡名（你示例是 `eno1`）：

```bash
ifconfig
```

设置主机 IP（与你现在配置一致）：

```bash
sudo ifconfig eno1 192.168.1.50
ifconfig eno1
ping 192.168.1.167
```

如果 `ping` 有 `time=...` 返回，说明网络正常。

### 2.2 检查 MID360 配置文件

文件：`/home/songfei/catkin_ws/src/livox_ros_driver2/config/MID360_config.json`

重点检查两类字段：
1. `host_net_info` 下面所有 `*_ip` 都是你的电脑地址（示例 `192.168.1.50`）
2. `lidar_configs[0].ip` 是雷达真实地址（示例 `192.168.1.167`）

### 2.3 配置 livox_ros_driver2 输出格式给 Fast-LIO

文件：`/home/songfei/catkin_ws/src/livox_ros_driver2/launch_ROS1/msg_MID360.launch`

把这两个参数设为：

```xml
<arg name="xfer_format" default="1"/>
<arg name="multi_topic" default="0"/>
```

说明：
- `xfer_format=1`：发布 Livox 自定义点云（Fast-LIO 常用）
- `multi_topic=0`：单雷达单话题

改完后回到工作空间重新编译：

```bash
cd /home/songfei/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 3. 处理底盘里程计 TF 冲突（建议做）

Fast-LIO 会发布自己的位姿/TF；`tracer_base` 也可能发布 `odom -> base_footprint`。
如果两者同时发布，会导致 TF 冲突、RViz 抖动、定位跳变。

你的当前文件：`/home/songfei/catkin_ws/src/tracer_ros/tracer_base/launch/tracer_base.launch`
中有：

```xml
<param name="publish_odom_tf" type="bool" value="true" />
```

建议改为：

```xml
<param name="publish_odom_tf" type="bool" value="false" />
```

然后重新编译：

```bash
cd /home/songfei/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 4. 配置 Fast-LIO（雷达话题、IMU、外参）

### 4.1 找到 Fast-LIO 配置文件

通常在：

`/home/songfei/catkin_ws/src/FAST_LIO/config/`

不同版本文件名可能不同（如 `avia.yaml` / `mid360.yaml`）。

你需要确认并设置以下参数（字段名按你本地 yaml 为准）：

1. 点云话题：`/livox/lidar`
2. IMU 话题：`/livox/imu`
3. 雷达类型：Livox（常见是 `lidar_type: 1`）
4. 雷达到机体外参（`extrinsic_T` / `extrinsic_R`）

### 4.2 外参如何填

先用“可运行默认值”，再精调：

- 若雷达安装方向与机体坐标轴一致：
  - `extrinsic_R` 设单位阵
- `extrinsic_T = [x, y, z]` 填雷达坐标系原点相对 `base_link` 的米制偏移

示例（仅示意，不可直接照搬）：

```yaml
extrinsic_T: [0.25, 0.0, 0.18]
extrinsic_R: [1,0,0, 0,1,0, 0,0,1]
```

---

## 5. 标准启动顺序（建议开 4 个终端）

每个终端都先执行：

```bash
source /home/songfei/catkin_ws/devel/setup.bash
```

### 终端 A：启动底盘驱动

首次配置 CAN（仅首次）：

```bash
rosrun tracer_bringup setup_can2usb.bash
```

日常上电：

```bash
rosrun tracer_bringup bringup_can2usb.bash
roslaunch tracer_bringup tracer_robot_base.launch
```

### 终端 B：启动 Livox 驱动（MID360）

```bash
roslaunch livox_ros_driver2 msg_MID360.launch
```

### 终端 C：启动 Fast-LIO

先看你本机有哪些 launch：

```bash
ls /home/songfei/catkin_ws/src/FAST_LIO/launch
```

常见命令（按你实际文件名选一个）：

```bash
roslaunch fast_lio mapping_avia.launch
```

或（若存在）：

```bash
roslaunch fast_lio mapping_mid360.launch
```

### 终端 D：RViz 可视化

```bash
rviz
```

RViz 里建议添加：
- `TF`
- `PointCloud2`（看 Fast-LIO 发布的地图/当前帧）
- `Path`
- `Odometry`

Fixed Frame 一般设为 Fast-LIO 的世界系（常见 `map` 或 `camera_init`，以你实际 TF 为准）。

---

## 6. 联调自检清单（按顺序排查）

### 6.1 雷达话题类型必须正确

```bash
rostopic type /livox/lidar
rostopic hz /livox/lidar
rostopic hz /livox/imu
```

目标现象：
- `/livox/lidar` 有稳定频率
- `/livox/imu` 有稳定频率

### 6.2 TF 树必须无冲突

```bash
rosrun tf view_frames
```

检查是否存在多个节点同时发布同一对变换（特别是 `odom -> base_*`）。

### 6.3 Fast-LIO 必须有持续位姿输出

```bash
rostopic list | rg -i "odom|path|cloud|lio"
```

若没有输出：
1. 先看终端 C 报错（话题名不匹配最常见）
2. 检查第 2.3 节 `xfer_format` 是否是 `1`
3. 检查第 4 节配置里的点云/IMU 话题是否和驱动一致

---

## 7. 常见问题与处理

### 问题 1：RViz 有点云但 Fast-LIO 不动

常见原因：
- Fast-LIO 订阅的话题名不对
- 点云类型不匹配（`xfer_format` 配错）
- 外参错误导致发散

处理顺序：
1. `rostopic type /livox/lidar` 确认类型
2. 对照 Fast-LIO yaml 的 `lid_topic` / `imu_topic`
3. 外参先用单位旋转 + 粗略平移，确认跑通后再精调

### 问题 2：画面抖动、位姿跳

常见原因：TF 冲突。

处理：
- 按第 3 节关闭 `tracer_base` 的 `publish_odom_tf`
- 确保只保留一套主里程计 TF 发布者

### 问题 3：每次开机都要手动设 IP

这是你当前“临时 ifconfig”方式的正常表现。后续可用 NetworkManager 或 netplan 做永久静态 IP。

---

## 8. 推荐的最终运行口令（最小可用）

1. 终端 A

```bash
source /home/songfei/catkin_ws/devel/setup.bash
rosrun tracer_bringup bringup_can2usb.bash
roslaunch tracer_bringup tracer_robot_base.launch
```

2. 终端 B

```bash
source /home/songfei/catkin_ws/devel/setup.bash
sudo ifconfig eno1 192.168.1.50
roslaunch livox_ros_driver2 msg_MID360.launch
```

3. 终端 C

```bash
source /home/songfei/catkin_ws/devel/setup.bash
roslaunch fast_lio mapping_avia.launch
```

4. 终端 D

```bash
source /home/songfei/catkin_ws/devel/setup.bash
rviz
```

---

## 9. 你接下来建议做的两件事

1. 先按本文档跑通“静止 + 慢速直线 + 原地转向”三组测试，观察轨迹是否连续。
2. 跑通后，再做外参精调（尤其是雷达安装角度），定位效果会明显更稳。

