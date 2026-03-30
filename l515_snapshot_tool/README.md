# L515 单次拍照脚本

## 文件说明
- `capture_l515_once.py`：自动启动 `realsense2_camera/l515.launch`，抓取一帧彩色图和深度图并保存。
- `output/`：默认输出目录。

## 使用前提
1. 已正确安装并配置 ROS Noetic、RealSense ROS 驱动、`l515.launch`。
2. 已在当前终端执行过：
   ```bash
   source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.bash
   ```
3. `l515.launch` 中应启用：
   - `/camera/color/image_raw`
   - `/camera/aligned_depth_to_color/image_raw`

## 最直接用法
```bash
cd <你的工作空间中放置本脚本的目录>
python3 capture_l515_once.py
```

## 若相机节点已手动启动
```bash
python3 capture_l515_once.py --skip-launch
```

## 自定义输出目录
```bash
python3 capture_l515_once.py --output-dir /home/songfei/catkin_ws/output
```

## 默认输出文件
- `YYYYmmdd_HHMMSS_color.png`
- `YYYYmmdd_HHMMSS_depth.png`
- `YYYYmmdd_HHMMSS_depth_vis.png`
- `YYYYmmdd_HHMMSS_depth.npy`

## 常见注意点
- 如果脚本提示超时，优先检查：
  - `rostopic list | grep camera`
  - `l515.launch` 里是否启用了 `align_depth`
  - 话题名是否与默认值一致
- 若你的深度话题不是 `/camera/aligned_depth_to_color/image_raw`，可用 `--depth-topic` 手动改。
