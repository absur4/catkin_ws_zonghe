# Tracer + MID360 + IMU 融合定位（3_14 方案）

本方案**不修改任何原有文件**，只新增以 `3_14` 结尾的新文件。
目标：在你现有 AMCL + DWA 架构下，用 MID360 内置 IMU 融合底盘里程计，降低转弯漂移。

---

## 1. 新增文件清单（均已带 3_14 后缀）

- `src/tracer_ros/tracer_base/launch/tracer_base_3_14.launch`
- `src/tracer_ros/tracer_bringup/launch/tracer_robot_base_3_14.launch`
- `src/tracer_ros/tracer_nav/param/ekf_odom_imu_3_14.yaml`
- `src/tracer_ros/tracer_nav/launch/ekf_odom_imu_3_14.launch`
- `src/tracer_ros/tracer_nav/launch/nav_dwa_3_14.launch`
- `src/robocup_navigation/launch/navigation_dwa_3_14.launch`

---

## 2. 关键设计说明

### 2.1 底盘 TF 不再进入 /tf
`tracer_base_node` 会发布 `odom -> base_footprint` 的 TF，但原代码无法关掉。
本方案通过 **重映射 /tf** 到 `/tf_raw_3_14` 来隔离底盘 TF：
- 这样不会污染导航 TF 树
- EKF 统一发布 `odom -> base_footprint`

### 2.2 里程计输入话题
底盘里程计改为：
- 话题：`/wheel_odom_3_14`
- frame_id：`wheel_odom_3_14`
- child_frame_id：`base_footprint`

EKF 使用：
- `/wheel_odom_3_14` + `/livox/imu`

---

## 3. 启动顺序（4 个终端）

每个终端先执行：
```bash
source /home/songfei/catkin_ws/devel/setup.bash
```

### 终端 A：启动底盘（3_14 版本）
```bash
rosrun tracer_bringup bringup_can2usb.bash
roslaunch tracer_bringup tracer_robot_base_3_14.launch
```

### 终端 B：启动 Livox MID360
```bash
sudo ifconfig eno1 192.168.1.50
ifconfig eno1进行确认是不是上面的地址

```

### 终端 C：启动导航 + EKF（3_14 版本）
```bash
roslaunch robocup_navigation navigation_dwa_3_14.launch
```
rosrun robocup_execute pick_place_dry_run_90cm.py


实际：
navigation_dwa_vision.launch,pick_place_run_vision.py,realsense_camera.launch rs_camera.launch

### 终端 D：RViz（可选）
```bash
rviz
```

Fixed Frame 建议设为 `map`。

---

## 4. 自检要点

1. `rostopic list | grep wheel_odom_3_14`
2. `rostopic echo /wheel_odom_3_14 -n 1`
3. `rostopic echo /livox/imu -n 1`
4. `rosrun tf tf_echo odom base_footprint`

若 TF 正常，转弯时漂移会明显减小。

---

## 5. 常见问题

- 如果 `robot_localization` 未安装：
```bash
sudo apt install ros-noetic-robot-localization
```

- 如果 TF 抖动或重复：
检查是否仍旧启动了旧的 `tracer_base.launch` 或 `nav_dwa.launch`。

