# API README

本目录当前主要有 3 个直接可用的脚本：

- [`closed_set_object_detector.py`](/Users/zhanghanyu/Desktop/objects/open_vocabulary/API/closed_set_object_detector.py)
- [`cabinet_layers_2d.py`](/Users/zhanghanyu/Desktop/objects/open_vocabulary/API/cabinet_layers_2d.py)
- [`cabinet_planes_placement_3d.py`](/Users/zhanghanyu/Desktop/objects/open_vocabulary/API/cabinet_planes_placement_3d.py)

## 文件作用

### 1. `closed_set_object_detector.py`
闭集物体检测与分类脚本。

作用：
- 用本地 `Grounding DINO` 先检测候选框
- 用 `SigLIP` 提取 crop 特征
- 用 `ProKeR` 做 training-free 少样本分类
- 用 support 特征白化后的 `log-likelihood` 过滤异常框
- 最后再对 `plate.` 和 `spoon.` 做覆盖检测

适用场景：
- 你已经有固定类别集合
- support 图片放在按类别分目录的 `train_root`
- 希望在测试图上输出类别标签、置信度和可视化结果

### 2. `cabinet_layers_2d.py`
只做 2D 柜子检测和分层。

作用：
- 检测柜子 2D 框
- 在柜子 ROI 内检测层板线
- 输出柜子分层结果和 2D 可视化

适用场景：
- 只想知道柜子在图中的位置和每层边界
- 不需要 3D 平面和放置点
- 不需要识别任何物品

### 3. `cabinet_planes_placement_3d.py`
只做和柜子相关的 3D 平面与放置点分析。

作用：
- 先做 2D 柜子检测和分层
- 再利用 RGB-D 建立柜子内部水平平面
- 在平面上计算可放置点
- 输出 2D overlay、PLY 点云和 JSON

适用场景：
- 需要知道柜子每层对应的 3D 支撑面
- 需要为后续抓取/放置规划提供可放置点
- 不关心柜子里已有物品的类别识别

## 整体流程

### A. 闭集检测流程 `closed_set_object_detector.py`
1. 用 `Grounding DINO` 根据 `--objects` 做开放词汇检测
2. 对每个检测框做中心收缩 crop
3. 用 `SigLIP` 提取图像特征
4. 用 support 集和文本特征构建 `ProKeR` kernel 分类器
5. 对分类 crop 计算 support 特征空间下的白化 `log-likelihood`
6. 删除白化分数过低的框
7. 再额外跑 `plate.` 和 `spoon.` 检测，覆盖最终结果

### B. 2D 柜子分层流程 `cabinet_layers_2d.py`
1. 用 `Grounding DINO` 检测柜子 ROI
2. 在柜子区域内做层板线检测
3. 对候选横线做聚类与清理
4. 生成每层的 2D 边界
5. 输出 `shelf_lines.jpg` 和 `shelf_results.json`

### C. 3D 平面与放置点流程 `cabinet_planes_placement_3d.py`
1. 先执行 2D 柜子 ROI 和分层
2. 读取深度图并转换成米
3. 在柜子 ROI 内重建点云
4. 检测水平平面，支持 `robust` / `hms` / `robust_fusion`
5. 对每个平面重建边界
6. 在平面上建立占用栅格，计算最大空闲区域
7. 输出放置点、平面 PLY 和 2D overlay

## 使用的技术

### `closed_set_object_detector.py`
- 检测：`Grounding DINO`
- 图像编码：`SigLIP`
- 少样本分类：`ProKeR`
- support 增强：随机裁剪 + 随机翻转
- 拒识/过滤：support 特征白化 + `log-likelihood`
- 结果后处理：NMS、重叠框过滤、`plate/spoon` override

### `cabinet_layers_2d.py`
- 柜体检测：`Grounding DINO`
- 2D 结构分析：Canny / Hough / RANSAC / DBSCAN 风格的层板线提取
- 几何输出：柜子 ROI、层板线、每层上下边界

### `cabinet_planes_placement_3d.py`
- 深度转点云：相机内参反投影
- 平面检测：`robust planar patches`、`HMS-RANSAC`、`robust_fusion`
- 平面建模：TLS 平面拟合、平面边界重建
- 放置点评估：占用栅格 + 最大空闲矩形 / 距离变换

## 运行命令

### 1. `closed_set_object_detector.py`

推荐命令：

```bash
/opt/anaconda3/bin/conda run -n IND python /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/closed_set_object_detector.py \
  --image /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/photos/test/cabit/c_2_Color.png \
  --train-root /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/photos/train \
  --objects "milk. cup. plate. apple. hot dog. crop." \
  --device mps \
  --siglip-model google/siglip-base-patch16-224 \
  --support-cache-dir /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/.cache/closed_set_support \
  --augment-epoch 1 \
  --box-threshold 0.28 \
  --text-threshold 0.20 \
  --proker-lambda 0.1 \
  --whitening-log-like-threshold -9000 \
  --classify-crop-shrink 0.15 \
  --overlap-containment-ratio 0.8 \
  --overlap-area-ratio 1.35 \
  --override-prompts plate. spoon. \
  --override-box-threshold 0.45 \
  --override-iou 0.3 \
  --override-containment 0.6 \
  --output /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/output/c_2_siglip.jpg \
  --json /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/output/c_2_siglip.json
```

输出：
- 标注图
- JSON 检测结果
- 每个框的类别、分类置信度、白化分数

### 2. `cabinet_layers_2d.py`

```bash
/opt/anaconda3/bin/conda run -n IND python /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/cabinet_layers_2d.py \
  --image /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/rgb_depth/rgb/rgb_000000.png \
  --output /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/output/cabinet_2d_rgb000000 \
  --cabinet-prompt "cabinet." \
  --camera-param-file /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/rgb_depth/result.txt
```

输出：
- `shelf_lines.jpg`
- `shelf_results.json`

### 3. `cabinet_planes_placement_3d.py`

```bash
/opt/anaconda3/bin/conda run -n IND python /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/cabinet_planes_placement_3d.py \
  --image /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/rgb_depth/rgb/rgb_000000.png \
  --depth /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/rgb_depth/depth/depth_000000.png \
  --camera-param-file /Users/zhanghanyu/Desktop/objects/photo_rgb_depth/rgb_depth/result.txt \
  --output /Users/zhanghanyu/Desktop/objects/open_vocabulary/API/output/cabinet_3d_rgb000000 \
  --cabinet-prompt "cabinet." \
  --plane-method robust_fusion
```

输出：
- `shelf_lines.jpg`
- `cabinet_3d_overlay.jpg`
- `cabinet_planes_colored.ply`
- `cabinet_planes_with_placements.ply`
- `cabinet_planes_placements.json`

## 输出解释

### `closed_set_object_detector.py`
- `classification_confidence`
  分类置信度
- `whitening_log_like`
  白化后 log-likelihood，越小越异常
- `accepted_detections`
  过滤后保留下来的结果

### `cabinet_layers_2d.py`
- `cabinet.bbox`
  柜子 2D 框
- `shelf_lines_y`
  层板线 y 坐标
- `shelf_layers`
  每层的上下边界

### `cabinet_planes_placement_3d.py`
- `planes`
  柜子内部平面
- `placement_points`
  推荐放置点
- `surface_area_m2`
  平面近似面积
- `rect_bounds_3d_m`
  可放置空闲矩形的 3D 边界

## 注意

- `closed_set_object_detector.py` 需要 `train_root` 按类别分目录
- `SigLIP` 需要 `sentencepiece`
- `cabinet_planes_placement_3d.py` 需要 RGB 和深度已经对齐
- 相机内参可以从 `result.txt` 读取，也可以用 `--fx --fy --cx --cy` 手动覆盖
- `open3d` 缺失时，3D 平面/PLY 相关功能不能运行
