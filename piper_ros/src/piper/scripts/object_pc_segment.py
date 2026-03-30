#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物品点云截取模块 (Object Point Cloud Segmentation)

高效方案:
  1. L515 采集 RGB + Depth
  2. 深度图 → 有组织点云 (H,W,3)
  3. 体素下采样 → RANSAC 拟合桌面 → 去除桌面点云
  4. Grounding DINO 检测 → 2D bbox
  5. bbox 切片 + 非桌面掩码 → 物品独立点云
  6. 保存为 .ply 文件
"""

import sys
import os
import time
import numpy as np
import cv2
import open3d as o3d

# ── Grounded-SAM-2 路径注入（仅用 Grounding DINO，不加载 SAM2） ──
GSAM2_ROOT = os.environ.get("GSAM2_ROOT", "/home/h/Grounded-SAM-2")
sys.path.insert(0, GSAM2_ROOT)

import torch
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import (
    load_model as load_gdino_model,
    Model as GroundingDINOModelAPI,
    predict as gdino_predict,
)

# ── SAM2 (条件加载) ──
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

# ── 默认路径 ──
DEFAULT_GDINO_CONFIG = os.path.join(
    GSAM2_ROOT, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_GDINO_CHECKPOINT = os.path.join(
    GSAM2_ROOT, "gdino_checkpoints/groundingdino_swint_ogc.pth"
)
DEFAULT_TEXT_PROMPT = "fork. knife. spoon. chopsticks. plate. bowl. cup."
DEFAULT_OUTPUT_DIR = "/tmp/object_clouds"


# ═══════════════════════════════════════════════════════════════
#  ObjectPointCloudSegmenter
# ═══════════════════════════════════════════════════════════════
class ObjectPointCloudSegmenter:
    """桌面物品点云截取器 (高效版)

    只加载 Grounding DINO (不加载 SAM2)，利用 2D bbox + 桌面去除
    实现快速物品点云分割。

    Parameters
    ----------
    text_prompt : str
        Grounding DINO 文本提示，类别间以 ". " 分隔并以 "." 结尾。
    output_dir : str
        点云保存目录。
    voxel_size : float
        体素下采样边长 (米)，仅用于平面拟合加速。
    plane_dist_thresh : float
        RANSAC 平面拟合的距离阈值 (米)。
    table_remove_thresh : float
        桌面去除距离阈值 (米)，点到桌面距离小于此值则判为桌面点。
    depth_range : tuple (min_m, max_m)
        深度预筛选范围 (米)，过滤地面等远处的点。
    box_threshold, text_threshold : float
        Grounding DINO 检测阈值。
    bbox_horizontal_shrink_ratio : float
        bbox 水平方向总缩减比例。比如 0.1 表示总宽度缩减 10%，左右各缩 5%。
    save_clouds_to_disk : bool
        是否将点云保存为 .ply。默认 False，仅返回内存中的点云结果。
    use_camera : bool
        是否初始化并使用 RealSense 相机。默认 True。
    device : str
        "cuda" 或 "cpu"。
    """

    def __init__(
        self,
        text_prompt=DEFAULT_TEXT_PROMPT,
        output_dir=DEFAULT_OUTPUT_DIR,
        voxel_size=0.01,
        plane_dist_thresh=0.01,
        table_remove_thresh=0.02,
        depth_range=(0.1, 1.0),
        box_threshold=0.35,
        text_threshold=0.25,
        bbox_horizontal_shrink_ratio=0.0,
        save_clouds_to_disk=False,
        segmentation_method="bbox",      # "bbox" 或 "sam2_mask"
        use_camera=True,
        visual_preset=5,
        gdino_config=DEFAULT_GDINO_CONFIG,
        gdino_checkpoint=DEFAULT_GDINO_CHECKPOINT,
        sam2_predictor=None,             # 可以从外部传入已经初始化的 SAM2
        sam2_config="",
        sam2_checkpoint="",
        device=None,
    ):
        self.text_prompt = text_prompt
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        self.plane_dist_thresh = plane_dist_thresh
        self.table_remove_thresh = table_remove_thresh
        self.depth_range = depth_range
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.bbox_horizontal_shrink_ratio = bbox_horizontal_shrink_ratio
        self.save_clouds_to_disk = save_clouds_to_disk
        self.segmentation_method = segmentation_method
        self.use_camera = use_camera
        self.visual_preset = visual_preset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._pipeline_started = False
        self.sam2_predictor = sam2_predictor

        os.makedirs(self.output_dir, exist_ok=True)

        # 加载 Grounding DINO
        print("[INFO] 加载 Grounding DINO 模型 ...")
        self.grounding_model = load_gdino_model(
            model_config_path=gdino_config,
            model_checkpoint_path=gdino_checkpoint,
            device=self.device,
        )
        print("[INFO] Grounding DINO 模型加载完成")

        # 若初始就指定了用 SAM2 分割且未从外部传入，则在内部直接加载
        if self.segmentation_method == "sam2_mask" and self.sam2_predictor is None:
            self._lazy_load_sam2(sam2_config, sam2_checkpoint)

        # 保存备用配置以便后续动态加载
        self._sam2_config = sam2_config
        self._sam2_checkpoint = sam2_checkpoint

        # 初始化 RealSense pipeline（长驻，避免重复 start/stop）
        if self.use_camera:
            self._init_realsense()

    # ─────────────────────────────────────────────
    #  RealSense 初始化 & 释放
    # ─────────────────────────────────────────────
    def _init_realsense(self):
        """初始化 RealSense pipeline 并预热（仅在构造时调用一次）。"""
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)
        self._pipeline_started = True
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        if (
            self.visual_preset is not None
            and depth_sensor.supports(rs.option.visual_preset)
        ):
            depth_sensor.set_option(rs.option.visual_preset, self.visual_preset)
            preset_val = depth_sensor.get_option(rs.option.visual_preset)
            print(f"[INFO] Visual Preset set to: {int(preset_val)}")
        elif self.visual_preset is not None:
            print("[WARN] visual_preset option is not supported on this sensor")

        # 获取深度缩放因子
        self.depth_scale = depth_sensor.get_depth_scale()

        # 预热：跳过前 30 帧（自动曝光稳定）
        print("[INFO] RealSense 预热中 ...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("[INFO] RealSense 就绪")

    def _lazy_load_sam2(self, config_path, checkpoint_path):
        """延迟加载 SAM2 模型，只有在用到时且尚未加载才会触发。"""
        if self.sam2_predictor is not None:
            return
            
        if not HAS_SAM2:
            raise ImportError("segmentation_method='sam2_mask' 需要安装 sam2")
            
        if not checkpoint_path or not config_path:
            # 若无外部指定，则取默认位置（依赖环境变量推导）
            gsam_root = os.environ.get("GSAM2_ROOT", "/home/h/Grounded-SAM-2")
            checkpoint_path = os.path.join(gsam_root, "checkpoints/sam2.1_hiera_large.pt")
            config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"

        print(f"[INFO] 延迟加载 SAM2 模型: {os.path.basename(checkpoint_path)} ...")
        sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("[INFO] SAM2 模型延迟加载完成")

    def close(self):
        """释放 RealSense pipeline。"""
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            return

        if not getattr(self, "_pipeline_started", False):
            self.pipeline = None
            return

        try:
            pipeline.stop()
            print("[INFO] RealSense pipeline 已释放")
        except RuntimeError as e:
            if "cannot be called before start" not in str(e):
                raise
        finally:
            self._pipeline_started = False
            self.pipeline = None

    def __del__(self):
        self.close()

    # ─────────────────────────────────────────────
    #  1. RealSense L515 采集（快速取帧）
    # ─────────────────────────────────────────────
    def capture_rgbd(self):
        """从已初始化的 pipeline 获取一帧对齐后的 RGB + Depth。

        Returns
        -------
        color_image : np.ndarray (H, W, 3), uint8, BGR
        depth_image : np.ndarray (H, W), float32, 单位米
        intrinsics : dict  {"fx", "fy", "cx", "cy", "width", "height"}
        """
        if not self.use_camera or not hasattr(self, "pipeline"):
            raise RuntimeError("当前实例未启用相机，请使用外部 RGBD 输入流程")

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("未能从 L515 获取有效帧")

        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = {
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height,
        }

        depth_image = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        )
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, intrinsics

    # ─────────────────────────────────────────────
    #  2. 深度图 → 有组织点云 (H, W, 3)
    # ─────────────────────────────────────────────
    @staticmethod
    def depth_to_organized_cloud(depth_image, intrinsics):
        """将深度图转为有组织点云，保持 (H, W, 3) 形状以便 bbox 直接切片。

        Returns
        -------
        points_map : np.ndarray (H, W, 3), float32
        depth_valid : np.ndarray (H, W), bool
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        h, w = depth_image.shape

        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_map = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        depth_valid = z > 0

        return points_map, depth_valid

    # ─────────────────────────────────────────────
    #  3. 拟合桌面 & 去除桌面点
    # ─────────────────────────────────────────────
    def fit_and_remove_table(self, points_map, depth_valid):
        """RANSAC 拟合桌面平面，返回非桌面点的掩码。

        Parameters
        ----------
        points_map : np.ndarray (H, W, 3)
        depth_valid : np.ndarray (H, W), bool

        Returns
        -------
        non_table_mask : np.ndarray (H, W), bool  非桌面且有效深度的点
        plane_model : np.ndarray (4,)  [a, b, c, d]
        """
        h, w = depth_valid.shape

        # 深度范围预筛选 (仅用于平面拟合)
        z = points_map[:, :, 2]
        z_min, z_max = self.depth_range
        range_mask = depth_valid & (z >= z_min) & (z <= z_max)
        filtered_pts = points_map[range_mask]  # (N, 3)

        if len(filtered_pts) < 100:
            raise RuntimeError(
                f"深度范围 [{z_min}, {z_max}]m 内点数不足 ({len(filtered_pts)})，请调整 depth_range"
            )

        # 构建 Open3D 点云 + 体素下采样
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_pts)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        print(f"[INFO] 体素下采样: {len(pcd.points)} → {len(pcd_down.points)} 点")

        # RANSAC
        plane_model, _ = pcd_down.segment_plane(
            distance_threshold=self.plane_dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
        plane_model = np.array(plane_model, dtype=np.float64)  # [a, b, c, d]
        normal = plane_model[:3]
        d = plane_model[3]

        print(f"[INFO] 桌面方程: {normal[0]:.4f}x + {normal[1]:.4f}y + "
              f"{normal[2]:.4f}z + {d:.4f} = 0")

        # 向量化计算所有像素到桌面的距离
        # dist = |ax + by + cz + d| / ||normal||
        norm_len = np.linalg.norm(normal)
        dist_map = np.abs(
            points_map[:, :, 0] * normal[0]
            + points_map[:, :, 1] * normal[1]
            + points_map[:, :, 2] * normal[2]
            + d
        ) / norm_len  # (H, W)

        table_mask = dist_map < self.table_remove_thresh  # 桌面点
        non_table_mask = depth_valid & (~table_mask)

        table_count = np.count_nonzero(table_mask & depth_valid)
        remain_count = np.count_nonzero(non_table_mask)
        print(f"[INFO] 桌面去除: 去除 {table_count} 个桌面点, 剩余 {remain_count} 个非桌面点")

        return non_table_mask, plane_model

    # ─────────────────────────────────────────────
    #  4. Grounding DINO 物体检测
    # ─────────────────────────────────────────────
    def detect_objects(self, color_image):
        """Grounding DINO 检测，返回 xyxy 像素坐标的 bbox。

        Returns
        -------
        boxes_xyxy : np.ndarray (N, 4), int, [x1, y1, x2, y2]
        class_names : list[str]
        confidences : list[float]
        """
        image_tensor = GroundingDINOModelAPI.preprocess_image(color_image)
        h, w = color_image.shape[:2]

        boxes, confidences, labels = gdino_predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if len(boxes) == 0:
            print("[WARN] 未检测到任何物体")
            return np.array([]).reshape(0, 4), [], []

        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(
            boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy"
        ).numpy().astype(int)

        # 限制在图像范围内
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

        class_names = labels
        conf_list = confidences.numpy().tolist()

        print(f"[INFO] 检测到 {len(class_names)} 个物体: "
              f"{list(zip(class_names, [f'{c:.2f}' for c in conf_list]))}")

        return boxes_xyxy, class_names, conf_list

    @staticmethod
    def _apply_horizontal_bbox_shrink(
        boxes_xyxy, width, bbox_horizontal_shrink_ratio
    ):
        """Apply horizontal bbox shrink with boundary-safe clamping."""
        if not 0.0 <= bbox_horizontal_shrink_ratio < 1.0:
            raise ValueError(
                "bbox_horizontal_shrink_ratio 必须满足 0 <= ratio < 1"
            )

        adjusted_boxes = []
        for x1, y1, x2, y2 in boxes_xyxy:
            box_width = x2 - x1
            shrink_each_side = int(
                round(box_width * bbox_horizontal_shrink_ratio / 2.0)
            )
            x1_adj = min(max(x1 + shrink_each_side, 0), width)
            x2_adj = min(max(x2 - shrink_each_side, 0), width)

            if x2_adj <= x1_adj:
                center_x = int(round((x1_adj + x2_adj) / 2.0))
                x1_adj = max(center_x, 0)
                x2_adj = min(center_x + 1, width)

            adjusted_boxes.append((x1_adj, y1, x2_adj, y2))

        return adjusted_boxes

    # ─────────────────────────────────────────────
    #  5. bbox 裁剪 → 物品点云
    # ─────────────────────────────────────────────
    @staticmethod
    def segment_object_clouds(
        points_map,
        non_table_mask,
        boxes_xyxy,
        bbox_horizontal_shrink_ratio=0.0,
    ):
        """根据 bbox 从有组织点云中切片取出物品点云。

        Parameters
        ----------
        points_map : np.ndarray (H, W, 3)
        non_table_mask : np.ndarray (H, W), bool
        boxes_xyxy : np.ndarray (N, 4), int
        bbox_horizontal_shrink_ratio : float
            bbox 水平方向总缩减比例。0.1 表示左右各缩 5%。

        Returns
        -------
        clouds : list[np.ndarray]  每个元素为 (M, 3) 的物品点云
        """
        if not 0.0 <= bbox_horizontal_shrink_ratio < 1.0:
            raise ValueError(
                "bbox_horizontal_shrink_ratio 必须满足 0 <= ratio < 1"
            )

        _, width = non_table_mask.shape
        adjusted_boxes = ObjectPointCloudSegmenter._apply_horizontal_bbox_shrink(
            boxes_xyxy, width, bbox_horizontal_shrink_ratio
        )

        clouds = []
        for x1, y1, x2, y2 in adjusted_boxes:
            crop = points_map[y1:y2, x1:x2]           # (h, w, 3) 切片
            valid = non_table_mask[y1:y2, x1:x2]      # (h, w) bool
            obj_cloud = crop[valid]                   # (M, 3)
            clouds.append(obj_cloud)
        return clouds

    @staticmethod
    def segment_object_clouds_with_colors(
        points_map,
        color_image,
        non_table_mask,
        boxes_xyxy,
        bbox_horizontal_shrink_ratio=0.0,
    ):
        """Extract object point clouds and RGB colors from bbox slices in one pass."""
        _, width = non_table_mask.shape
        adjusted_boxes = ObjectPointCloudSegmenter._apply_horizontal_bbox_shrink(
            boxes_xyxy, width, bbox_horizontal_shrink_ratio
        )

        clouds = []
        cloud_colors = []
        for x1, y1, x2, y2 in adjusted_boxes:
            valid = non_table_mask[y1:y2, x1:x2]
            cloud = points_map[y1:y2, x1:x2][valid]
            color = color_image[y1:y2, x1:x2][:, :, ::-1][valid].astype(np.uint8)
            clouds.append(cloud)
            cloud_colors.append(color)
        return clouds, cloud_colors

    # ─────────────────────────────────────────────
    #  6. 保存点云 (.ply)
    # ─────────────────────────────────────────────
    def save_clouds(self, clouds, class_names, confidences):
        """将每个物品点云保存为 .ply 文件。

        Returns
        -------
        saved_paths : list[str]
        """
        saved_paths = []
        for i, (cloud, name, conf) in enumerate(
            zip(clouds, class_names, confidences)
        ):
            if len(cloud) == 0:
                print(f"  [SKIP] {name}_{i}: 无有效点")
                saved_paths.append("")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)

            filename = f"{name}_{i}_conf{conf:.2f}.ply"
            filepath = os.path.join(self.output_dir, filename)
            o3d.io.write_point_cloud(filepath, pcd)
            saved_paths.append(filepath)
            print(f"  [SAVED] {filepath}  ({len(cloud)} 点)")

        return saved_paths

    # ─────────────────────────────────────────────
    #  完整流程
    # ─────────────────────────────────────────────
    def run(self):
        """执行完整的物品点云截取流程 (通过内置深度相机获取图像)。
        
        Returns
        -------
        results : list[dict]
            每个元素包含检测结果字典。
        """
        t_total = time.time()
        print("=" * 60)
        print("  物品点云截取 开始")
        print("=" * 60)

        # ── 1. 采集 ──
        t0 = time.time()
        print("\n[Step 1] 采集 RGB + Depth ...")
        color_image, depth_image, intrinsics = self.capture_rgbd()
        print(f"  耗时: {time.time() - t0:.3f}s")

        return self.process_rgbd(color_image, depth_image, intrinsics, t_total=t_total)

    def process_rgbd(self, color_image, depth_image, intrinsics, t_total=None):
        """处理给定的 RGB 图、深度图和相机内参，进行物体检测和点云截取。

        Returns
        -------
        results : list[dict]
            每个元素:
            - "color_image": np.ndarray (H, W, 3), uint8, BGR
            - "depth_image": np.ndarray (H, W), float32, 米
            - "intrinsics": dict
            - "scene_point_cloud": np.ndarray (N, 3)
            - "scene_point_color": np.ndarray (N, 3), uint8, RGB
            - "class_name": str
            - "confidence": float
            - "bbox": np.ndarray (4,) [x1,y1,x2,y2]
            - "point_cloud": np.ndarray (M, 3)
            - "point_cloud_color": np.ndarray (M, 3), uint8, RGB
            - "num_points": int
            - "saved_path": str | None
        """
        if t_total is None:
            t_total = time.time()

        # # ── DEBUG: 显示 RGB 和 Depth ──
        # depth_colormap = cv2.applyColorMap(
        #     cv2.convertScaleAbs(depth_image, alpha=255.0 / depth_image.max()),
        #     cv2.COLORMAP_JET,
        # )
        # # 统一尺寸后拼接
        # h, w = color_image.shape[:2]
        # depth_colormap = cv2.resize(depth_colormap, (w, h))
        # combined = np.hstack([color_image, depth_colormap])
        # cv2.imshow("DEBUG: RGB (left) | Depth (right)", combined)
        # print("[DEBUG] 按任意键继续 ...")

        # print(intrinsics)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # ── DEBUG: 生成并显示整幅场景点云 ──
        # t0 = time.time()
        # print("\n[DEBUG] 生成整幅场景点云 ...")
        # points_map, depth_valid = self.depth_to_organized_cloud(
        #     depth_image, intrinsics
        # )
        # z = points_map[:, :, 2]
        # debug_mask = depth_valid & np.isfinite(z)

        # debug_points = points_map[debug_mask]
        # debug_colors = color_image[debug_mask][:, ::-1].astype(np.float32) / 255.0

        # debug_pcd = o3d.geometry.PointCloud()
        # debug_pcd.points = o3d.utility.Vector3dVector(debug_points)
        # debug_pcd.colors = o3d.utility.Vector3dVector(debug_colors)

        # print(f"[DEBUG] 点云生成完成: {len(debug_points)} 点")
        # print(f"[DEBUG] 耗时: {time.time() - t0:.3f}s")
        # print("[DEBUG] 关闭 Open3D 窗口后继续 ...")
        # o3d.visualization.draw_geometries(
        #     [debug_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
        #     window_name="DEBUG: Full Scene Point Cloud",
        # )

        # ── 2. 有组织点云 ──
        t0 = time.time()
        print("\n[Step 2] 生成有组织点云 ...")
        points_map, depth_valid = self.depth_to_organized_cloud(
            depth_image, intrinsics
        )
        print(f"  点云形状: {points_map.shape}, 有效点: {np.count_nonzero(depth_valid)}")
        print(f"  耗时: {time.time() - t0:.3f}s")

        # ── 3. 桌面拟合 & 去除 ──
        t0 = time.time()
        print("\n[Step 3] RANSAC 桌面拟合 & 去除 ...")
        non_table_mask, plane_model = self.fit_and_remove_table(
            points_map, depth_valid
        )
        print(f"  耗时: {time.time() - t0:.3f}s")

        # # ── DEBUG: 可视化去桌面后的点云 ──
        # t0 = time.time()
        # print("\n[DEBUG] 可视化去桌面后的点云 ...")
        # debug_points = points_map[non_table_mask]
        # debug_colors = color_image[non_table_mask][:, ::-1].astype(np.float32) / 255.0

        # debug_pcd = o3d.geometry.PointCloud()
        # debug_pcd.points = o3d.utility.Vector3dVector(debug_points)
        # debug_pcd.colors = o3d.utility.Vector3dVector(debug_colors)

        # print(f"[DEBUG] 非桌面点数: {len(debug_points)}")
        # print(f"[DEBUG] 耗时: {time.time() - t0:.3f}s")
        # print("[DEBUG] 关闭 Open3D 窗口后继续 ...")
        # o3d.visualization.draw_geometries(
        #     [debug_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
        #     window_name="DEBUG: Non-table Point Cloud",
        # )

        # ── 4. 物体检测 ──
        t0 = time.time()
        print("\n[Step 4] Grounding DINO 检测 ...")
        boxes_xyxy, class_names, confidences = self.detect_objects(color_image)
        print(f"  耗时: {time.time() - t0:.3f}s")

        if len(boxes_xyxy) == 0:
            print("[WARN] 未检测到物体，流程结束")
            return []

        # ── 5. 物体点云分割 (bbox 裁剪 或 SAM2 掩码) ──
        t0 = time.time()
        print(f"\n[Step 5] 点云提取 ({self.segmentation_method}) ...")
        
        clouds = []
        cloud_colors = []

        if self.segmentation_method == "sam2_mask":
            # 引入 SAM2 提取精细 mask
            import torch
            
            # 若 SAM2 尚未加载则现在加载
            if self.sam2_predictor is None:
                self._lazy_load_sam2(self._sam2_config, self._sam2_checkpoint)
                
            self.sam2_predictor.set_image(color_image)
            
            # 准备 box (xyxy 像素坐标，Numpy 转 Tensor)
            input_boxes = np.array(boxes_xyxy)
            
            # 使用 autocast 提速
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            
            # masks 形状处理 (N, 1, H, W) -> (N, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.astype(bool)

            for i, mask in enumerate(masks):
                # 只保留含有有效深度 (且在设定范围内) 的掩码像素
                valid = mask & depth_valid
                cloud = points_map[valid]
                color = color_image[valid][:, ::-1].astype(np.uint8)
                clouds.append(cloud)
                cloud_colors.append(color)
                print(f"  [SAM2] 物体 #{i} 掩码提取完成，得到 {len(cloud)} 个有效点")

            print(f"  SAM2 掩码处理耗时: {time.time() - t0:.3f}s")

        else:
            # 原本的 bbox 粗略提取模式
            clouds, cloud_colors = self.segment_object_clouds_with_colors(
                points_map,
                color_image,
                non_table_mask,
                boxes_xyxy,
                bbox_horizontal_shrink_ratio=self.bbox_horizontal_shrink_ratio,
            )


        # # ── DEBUG: 显示原始点云与分割出的物品点云 ──
        # t0 = time.time()
        # print("\n[DEBUG] 可视化原始点云与物品点云 ...")

        # vis_geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]

        # scene_points = points_map[depth_valid]
        # scene_pcd = o3d.geometry.PointCloud()
        # scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        # scene_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        # vis_geometries.append(scene_pcd)

        # palette = np.array(
        #     [
        #         [1.0, 0.2, 0.2],
        #         [0.2, 0.8, 0.2],
        #         [0.2, 0.4, 1.0],
        #         [1.0, 0.7, 0.2],
        #         [0.8, 0.2, 1.0],
        #         [0.2, 0.9, 0.9],
        #     ],
        #     dtype=np.float64,
        # )

        # for i, cloud in enumerate(clouds):
        #     if len(cloud) == 0:
        #         print(f"[DEBUG] 物品 {i}: 无有效点，跳过显示")
        #         continue

        #     obj_pcd = o3d.geometry.PointCloud()
        #     obj_pcd.points = o3d.utility.Vector3dVector(cloud)
        #     obj_pcd.paint_uniform_color(palette[i % len(palette)].tolist())
        #     vis_geometries.append(obj_pcd)
        #     print(f"[DEBUG] 物品 {i}: {len(cloud)} 点")

        # print(f"[DEBUG] 原始点云: {len(scene_points)} 点")
        # print(f"[DEBUG] 耗时: {time.time() - t0:.3f}s")
        # print("[DEBUG] 灰色为原始点云，彩色为分割出的物品点云")
        # print("[DEBUG] 关闭 Open3D 窗口后继续 ...")
        # o3d.visualization.draw_geometries(
        #     vis_geometries,
        #     window_name="DEBUG: Scene Cloud + Segmented Object Clouds",
        # )

        saved_paths = [None] * len(clouds)
        if self.save_clouds_to_disk:
            # ── 6. 保存 ──
            t0 = time.time()
            print("\n[Step 6] 保存点云 ...")
            saved_paths = self.save_clouds(clouds, class_names, confidences)
            print(f"  耗时: {time.time() - t0:.3f}s")

        # ── 汇总 ──
        scene_point_cloud = points_map[depth_valid]
        scene_point_color = color_image[depth_valid][:, ::-1].astype(np.uint8)
        results = []
        for i, (cloud, cloud_color, name, conf, box) in enumerate(
            zip(clouds, cloud_colors, class_names, confidences, boxes_xyxy)
        ):
            path = saved_paths[i]
            results.append({
                "color_image": color_image,
                "depth_image": depth_image,
                "intrinsics": intrinsics,
                "scene_point_cloud": scene_point_cloud,
                "scene_point_color": scene_point_color,
                "class_name": name,
                "confidence": conf,
                "bbox": box,
                "point_cloud": cloud,
                "point_cloud_color": cloud_color,
                "num_points": len(cloud),
                "saved_path": path,
            })

        print(f"\n{'=' * 60}")
        print(f"  完成! 共 {len(results)} 个物品, 总耗时: {time.time() - t_total:.3f}s")
        if self.save_clouds_to_disk:
            print(f"  点云保存至: {self.output_dir}")
        else:
            print("  点云未落盘，仅返回内存结果")
        print(f"{'=' * 60}")

        return results


# ═══════════════════════════════════════════════════════════════
#  独立运行入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    segmenter = ObjectPointCloudSegmenter(
        text_prompt="bottle.",
        output_dir=DEFAULT_OUTPUT_DIR,
        voxel_size=0.01,
        table_remove_thresh=0.02,
        depth_range=(0.1, 1.0),
        bbox_horizontal_shrink_ratio=0.12,
        save_clouds_to_disk=False,
    )

    results = segmenter.run()

    for r in results:
        print(f"\n{r['class_name']}: {r['num_points']} 点")
