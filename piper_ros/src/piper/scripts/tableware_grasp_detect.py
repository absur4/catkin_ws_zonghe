#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
餐具抓姿识别模块 (Tableware Grasp Pose Detection)

功能流程:
  1. RealSense L515 采集 RGB + Depth
  2. Grounded SAM2 物体检测与掩码分割
  3. Open3D RANSAC 桌面平面拟合
  4. PCA 分析 (2D/3D 可选) 获取物体中心和最长轴
  5. 构建垂直于桌面的六维抓取姿态
"""

import sys
import os
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

# ── Grounded-SAM-2 路径注入 ──
GSAM2_ROOT = os.environ.get("GSAM2_ROOT", "/home/h/Grounded-SAM-2")
sys.path.insert(0, GSAM2_ROOT)

import torch
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import (
    load_model as load_gdino_model,
    load_image as load_gdino_image,
    predict as gdino_predict,
)

DEFAULT_TEXT_PROMPT = "fork. knife. spoon. chopsticks. plate. bowl."
DEFAULT_SAM2_CHECKPOINT = os.path.join(GSAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def resolve_sam2_config_path(config_rel_path):
    """兼容 Grounded-SAM-2 新旧目录结构，解析 SAM2 config 绝对路径。"""
    candidates = [
        os.path.join(GSAM2_ROOT, config_rel_path),
        os.path.join(GSAM2_ROOT, "sam2", config_rel_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


DEFAULT_SAM2_CONFIG_PATH = resolve_sam2_config_path(DEFAULT_SAM2_CONFIG)
DEFAULT_GDINO_CONFIG = os.path.join(
    GSAM2_ROOT, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_GDINO_CHECKPOINT = os.path.join(
    GSAM2_ROOT, "gdino_checkpoints/groundingdino_swint_ogc.pth"
)
FIXED_PCA_MODE = "3d"
FIXED_GRASP_CENTER_MODE = "axis_midpoint_min_depth"


# ═══════════════════════════════════════════════════════════════
#  TablewareGraspDetector
# ═══════════════════════════════════════════════════════════════
class TablewareGraspDetector:
    """桌面餐具六维抓姿检测器

    Parameters
    ----------
    text_prompt : str
        Grounding DINO 的文本提示，每个类别以 ". " 分隔并以 "." 结尾。
    pca_mode : str
        兼容旧接口，传入值会被忽略。
        当前算法固定为“2D PCA 计算中心 + 3D PCA 计算主轴”。
    sam2_checkpoint, sam2_config, gdino_config, gdino_checkpoint : str
        模型权重和配置路径，默认使用 Grounded-SAM-2 原位路径。
    box_threshold, text_threshold : float
        Grounding DINO 的检测置信度阈值。
    plane_distance_threshold : float
        RANSAC 平面拟合的距离阈值 (米)。
    grasp_center_mode : str
        兼容旧接口，当前算法固定为 "axis_midpoint_min_depth"，传入值会被忽略。
    grasp_tilt_x_deg : float
        抓姿绕局部 x 轴旋转角度（度）。
        0 表示末端 z 轴垂直于桌面；非 0 可控制斜向抓取。
    depth_range : tuple (min_m, max_m)
        用于预筛选点云的深度范围 (米)，过滤掉地面等远处的点。
    enable_result_visualization : bool
        是否可视化识别结果（2D/3D）。默认 False。
    device : str
        "cuda" 或 "cpu"。
    """

    def __init__(
        self,
        text_prompt=DEFAULT_TEXT_PROMPT,
        pca_mode=FIXED_PCA_MODE,
        sam2_checkpoint=None,
        sam2_config=None,
        gdino_config=None,
        gdino_checkpoint=None,
        sam2_predictor=None,
        grounding_model=None,
        box_threshold=0.35,
        text_threshold=0.25,
        plane_distance_threshold=0.01,
        grasp_center_mode=FIXED_GRASP_CENTER_MODE,
        grasp_tilt_x_deg=0.0,
        voxel_size=0.01,
        depth_range=(0.1, 1.0),
        visual_preset=5,
        enable_result_visualization=False,
        device=None,
    ):
        self.text_prompt = text_prompt
        if pca_mode != FIXED_PCA_MODE:
            print(
                f"[INFO] pca_mode 已固定为 '{FIXED_PCA_MODE}'，忽略传入值: {pca_mode}"
            )
        if grasp_center_mode != FIXED_GRASP_CENTER_MODE:
            print(
                "[INFO] grasp_center_mode 已固定为 "
                f"'{FIXED_GRASP_CENTER_MODE}'，忽略传入值: {grasp_center_mode}"
            )
        self.pca_mode = FIXED_PCA_MODE
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.plane_distance_threshold = plane_distance_threshold
        self.grasp_center_mode = FIXED_GRASP_CENTER_MODE
        self.grasp_tilt_x_deg = float(grasp_tilt_x_deg)
        self.voxel_size = voxel_size
        self.depth_range = depth_range
        self.visual_preset = visual_preset
        self.enable_result_visualization = bool(enable_result_visualization)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载或接收外部模型 ──
        if sam2_predictor is not None and grounding_model is not None:
            print("[INFO] 使用外部传入的预加载模型 (SAM2 & Grounding DINO)")
            self.sam2_predictor = sam2_predictor
            self.grounding_model = grounding_model
        else:
            if None in (sam2_checkpoint, sam2_config, gdino_config, gdino_checkpoint):
                raise ValueError("未提供预加载模型，必须传入完整的 checkpoint 和 config 路径。")
            self._load_models(sam2_checkpoint, sam2_config, gdino_config, gdino_checkpoint)

        self.pipeline = None
        self._pipeline_started = False
        # 初始化 RealSense pipeline（长驻，避免重复 start/stop）
        self._init_realsense()

    # ─────────────────────────────────────────────
    #  模型加载
    # ─────────────────────────────────────────────
    def _load_models(self, sam2_ckpt, sam2_cfg, gdino_cfg, gdino_ckpt):
        """加载 SAM2 和 Grounding DINO 模型"""
        print("[INFO] 加载 SAM2 模型...")
        sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        print("[INFO] 加载 Grounding DINO 模型...")
        self.grounding_model = load_gdino_model(
            model_config_path=gdino_cfg,
            model_checkpoint_path=gdino_ckpt,
            device=self.device,
        )
        print("[INFO] 模型加载完成")

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

        self.depth_scale = depth_sensor.get_depth_scale()

        print("[INFO] RealSense 预热中 ...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("[INFO] RealSense 就绪")

    def close(self):
        """释放 RealSense pipeline。"""
        if not getattr(self, "_pipeline_started", False):
            return
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
                print("[INFO] RealSense pipeline 已释放")
        except RuntimeError:
            pass
        finally:
            self._pipeline_started = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ─────────────────────────────────────────────
    #  1. RealSense L515 图像采集
    # ─────────────────────────────────────────────
    def capture_rgbd(self):
        """从 L515 相机获取对齐后的 RGB 图像、深度图和相机内参。

        Returns
        -------
        color_image : np.ndarray, shape (H, W, 3), dtype uint8, BGR 格式
        depth_image : np.ndarray, shape (H, W), dtype float32, 单位米
        intrinsics : dict  {"fx", "fy", "cx", "cy", "width", "height"}
        """
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
    #  2. 点云生成
    # ─────────────────────────────────────────────
    @staticmethod
    def generate_point_cloud(depth_image, intrinsics, color_image=None):
        """将深度图反投影为 3D 点云。

        Parameters
        ----------
        depth_image : np.ndarray (H, W), float32, 单位米
        intrinsics : dict  {"fx", "fy", "cx", "cy"}
        color_image : np.ndarray (H, W, 3), uint8, BGR, 可选

        Returns
        -------
        points : np.ndarray (N, 3)
        colors : np.ndarray (N, 3), float [0,1]  (如果提供了 color_image)
        valid_mask : np.ndarray (H, W), bool, 有效像素标记
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        h, w = depth_image.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image
        valid = z > 0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        points = points[valid_flat]

        colors = None
        if color_image is not None:
            c = color_image[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, [0,1]
            colors = c.reshape(-1, 3)[valid_flat]

        return points, colors, valid   #有效点云、每点颜色、有效掩码

    # ─────────────────────────────────────────────
    #  3. 桌面平面拟合 (Open3D RANSAC)
    # ─────────────────────────────────────────────
    def fit_table_plane(self, points, colors=None):
        """使用 Open3D RANSAC 拟合桌面平面。

        对输入点云按深度范围预筛选，避免地面干扰。

        Parameters
        ----------
        points : np.ndarray (N, 3)
        colors : np.ndarray (N, 3), 可选

        Returns
        -------
        plane_model : np.ndarray (4,)  [a, b, c, d]，ax+by+cz+d=0
        table_normal : np.ndarray (3,)  指向相机侧的法向量
        inlier_indices : np.ndarray     桌面内点索引
        """
        # 深度范围筛选
        z_min, z_max = self.depth_range
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        filtered_points = points[mask]

        if len(filtered_points) < 100:
            raise RuntimeError(
                f"深度范围 [{z_min}, {z_max}]m 内的点云过少 ({len(filtered_points)} 个)，"
                "请调整 depth_range 参数"
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        # 体素下采样（加速 RANSAC）
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        print(f"[INFO] 体素下采样: {len(pcd.points)} → {len(pcd_down.points)} 点 "
              f"(voxel_size={self.voxel_size}m)")

        # RANSAC 平面拟合
        plane_model, inlier_idx = pcd_down.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        plane_model = np.array(plane_model)  # [a, b, c, d]

        # 确保法向量指向相机侧（即 z 分量为负，因为相机看向 +z 方向）
        table_normal = plane_model[:3].copy()
        if table_normal[2] > 0:
            table_normal = -table_normal
            plane_model = -plane_model

        table_normal = table_normal / np.linalg.norm(table_normal)

        print(f"[INFO] 桌面方程: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + "
              f"{plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
        print(f"[INFO] 桌面法向量 (指向相机): {table_normal}")
        print(f"[INFO] 桌面内点数: {len(inlier_idx)}")

        return plane_model, table_normal, np.array(inlier_idx)

    # ─────────────────────────────────────────────
    #  4. Grounded SAM2 检测与分割
    # ─────────────────────────────────────────────
    def detect_and_segment(self, color_image):
        """使用 Grounded SAM2 进行物体检测与掩码分割。

        Parameters
        ----------
        color_image : np.ndarray (H, W, 3), BGR

        Returns
        -------
        masks : np.ndarray (N, H, W), bool
        class_names : list[str]
        confidences : list[float]
        boxes : np.ndarray (N, 4), xyxy 格式
        """
        # 保存临时图片（Grounding DINO 的 load_image 需要路径）
        tmp_path = "/tmp/_gsam2_tmp_input.jpg"
        cv2.imwrite(tmp_path, color_image)

        image_source, image_tensor = load_gdino_image(tmp_path)
        h, w, _ = image_source.shape

        # Grounding DINO 检测
        boxes, confidences, labels = gdino_predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if len(boxes) == 0:
            print("[WARN] Grounding DINO 未检测到任何物体")
            return np.array([]), [], [], np.array([])

        # 转换 box 格式: cxcywh → xyxy (像素坐标)
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(
            boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy"
        ).numpy()

        # SAM2 掩码分割
        self.sam2_predictor.set_image(image_source)

        # 设置 autocast
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # (N, 1, H, W) → (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        class_names = labels
        confidences_list = confidences.numpy().tolist()

        print(f"[INFO] 检测到 {len(class_names)} 个物体: "
              f"{list(zip(class_names, [f'{c:.2f}' for c in confidences_list]))}")

        return masks.astype(bool), class_names, confidences_list, input_boxes

    # ─────────────────────────────────────────────
    #  5. 物体点云提取
    # ─────────────────────────────────────────────
    @staticmethod
    def extract_object_points(mask, depth_image, intrinsics):
        """根据掩码从深度图中提取物体的 3D 点集。

        Parameters
        ----------
        mask : np.ndarray (H, W), bool
        depth_image : np.ndarray (H, W), float32, 米
        intrinsics : dict

        Returns
        -------
        object_points : np.ndarray (M, 3)
        pixel_coords : np.ndarray (M, 2)  对应的 (u, v) 像素坐标
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        valid = mask & (depth_image > 0)
        vs, us = np.where(valid)
        z = depth_image[vs, us]
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        object_points = np.stack([x, y, z], axis=-1)
        pixel_coords = np.stack([us, vs], axis=-1)

        return object_points, pixel_coords

    # ─────────────────────────────────────────────
    #  6a. PCA 分析 — 3D 模式
    # ─────────────────────────────────────────────
    @staticmethod
    def pca_analysis_3d(object_points):
        """在物体 3D 点云上做 PCA。

        Returns
        -------
        center : np.ndarray (3,)  物体中心
        longest_axis : np.ndarray (3,)  最长轴方向 (单位向量)
        eigenvalues : np.ndarray (3,)   三个特征值 (降序)
        eigenvectors : np.ndarray (3,3) 三个特征向量 (列向量, 降序)
        """
        center = object_points.mean(axis=0)
        centered = object_points - center
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 降序排列，最大方差的方向对应最长轴
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        longest_axis = eigenvectors[:, 0]  # 第一主成分
        longest_axis = longest_axis / np.linalg.norm(longest_axis)

        return center, longest_axis, eigenvalues, eigenvectors

    # ─────────────────────────────────────────────
    #  6b. PCA 分析 — 2D 模式
    # ─────────────────────────────────────────────
    @staticmethod
    def pca_analysis_2d(mask, depth_image, intrinsics):
        """在掩码像素坐标上做 2D PCA，然后反投影到 3D。

        1. 对掩码中有效像素的 (u, v) 坐标做 PCA
        2. 得到 2D 中心和 2D 最长轴方向
        3. 通过深度图将中心反投影为 3D 坐标
        4. 将 2D 最长轴方向扩展为 3D（在桌面附近近似有效）

        Returns
        -------
        center_3d : np.ndarray (3,)
        longest_axis_3d : np.ndarray (3,)
        center_2d : np.ndarray (2,)  像素坐标中心
        axis_2d : np.ndarray (2,)    像素坐标最长轴方向
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        valid = mask & (depth_image > 0)
        vs, us = np.where(valid)

        if len(us) < 3:
            raise RuntimeError("掩码中有效像素不足，无法进行 PCA 分析")

        # 2D PCA
        coords_2d = np.stack([us, vs], axis=-1).astype(np.float64)
        center_2d = coords_2d.mean(axis=0)
        centered = coords_2d - center_2d
        cov_2d = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        axis_2d = eigenvectors[:, 0]
        axis_2d = axis_2d / np.linalg.norm(axis_2d)

        # 将 2D 中心反投影到 3D
        cu, cv = center_2d
        cu_int, cv_int = int(round(cu)), int(round(cv))
        # 取中心附近区域的平均深度（更鲁棒）
        half_win = 3
        h, w = depth_image.shape
        r_min = max(0, cv_int - half_win)
        r_max = min(h, cv_int + half_win + 1)
        c_min = max(0, cu_int - half_win)
        c_max = min(w, cu_int + half_win + 1)
        patch = depth_image[r_min:r_max, c_min:c_max]
        valid_patch = patch[patch > 0]
        if len(valid_patch) == 0:
            raise RuntimeError("中心像素处无有效深度值")
        z_center = float(np.median(valid_patch))

        center_3d = np.array([
            (cu - cx) * z_center / fx,
            (cv - cy) * z_center / fy,
            z_center,
        ])

        # 将 2D 轴方向转为 3D：沿轴方向取另一个点，反投影后求方向
        # 使用轴两端的像素进行反投影
        du, dv = axis_2d
        end_u = cu + du * 50  # 沿轴方向偏移 50 像素
        end_v = cv + dv * 50

        # 端点处取深度
        eu_int, ev_int = int(round(end_u)), int(round(end_v))
        eu_int = np.clip(eu_int, 0, w - 1)
        ev_int = np.clip(ev_int, 0, h - 1)
        r_min_e = max(0, ev_int - half_win)
        r_max_e = min(h, ev_int + half_win + 1)
        c_min_e = max(0, eu_int - half_win)
        c_max_e = min(w, eu_int + half_win + 1)
        patch_e = depth_image[r_min_e:r_max_e, c_min_e:c_max_e]
        valid_patch_e = patch_e[patch_e > 0]
        z_end = float(np.median(valid_patch_e)) if len(valid_patch_e) > 0 else z_center

        end_3d = np.array([
            (end_u - cx) * z_end / fx,
            (end_v - cy) * z_end / fy,
            z_end,
        ])

        longest_axis_3d = end_3d - center_3d
        norm = np.linalg.norm(longest_axis_3d)
        if norm < 1e-6:
            longest_axis_3d = np.array([1.0, 0.0, 0.0])
        else:
            longest_axis_3d = longest_axis_3d / norm

        return center_3d, longest_axis_3d, center_2d, axis_2d

    @staticmethod
    def orient_axis_near_to_far(center, axis, object_points):
        """将轴方向统一为“由近及远”（点云到相机距离增大）的方向。"""
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis / axis_norm

        if object_points is None or len(object_points) < 6:
            if axis[2] < 0:
                axis = -axis
            return axis

        pts = np.asarray(object_points, dtype=np.float64)
        rel = pts - np.asarray(center, dtype=np.float64)
        proj = rel @ axis

        if (proj.max() - proj.min()) < 1e-6:
            if axis[2] < 0:
                axis = -axis
            return axis

        order = np.argsort(proj)
        k = max(3, int(0.2 * len(order)))
        near_pts = pts[order[:k]]
        far_pts = pts[order[-k:]]

        near_dist = np.linalg.norm(near_pts, axis=1).mean()
        far_dist = np.linalg.norm(far_pts, axis=1).mean()

        if far_dist < near_dist:
            axis = -axis
        elif abs(far_dist - near_dist) < 1e-6:
            near_z = near_pts[:, 2].mean()
            far_z = far_pts[:, 2].mean()
            if far_z < near_z:
                axis = -axis

        return axis

    @staticmethod
    def compute_grasp_center(center, axis, object_points, mode="axis_midpoint"):
        """计算抓取中心。

        Parameters
        ----------
        center : np.ndarray (3,)
            原始中心（通常为质心）
        axis : np.ndarray (3,)
            最长轴单位向量
        object_points : np.ndarray (N, 3)
            物体点云
        mode : str
            "axis_midpoint" 或 "centroid"。
            "axis_midpoint_min_depth" 在 process_frame 中单独处理：
            先取 axis_midpoint，再用掩码平滑深度的第 5 百分位覆盖 z。
        """
        center = np.asarray(center, dtype=np.float64)
        if mode == "centroid":
            return center.copy()

        if object_points is None or len(object_points) < 2:
            return center.copy()

        axis = np.asarray(axis, dtype=np.float64)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-9:
            return center.copy()
        axis = axis / norm_axis

        pts = np.asarray(object_points, dtype=np.float64)
        proj = (pts - center) @ axis
        t_min = float(np.min(proj))
        t_max = float(np.max(proj))
        midpoint = center + 0.5 * (t_min + t_max) * axis

        return midpoint

    @staticmethod
    def smooth_mask_depth_values(depth_image, mask, kernel_size=5):
        """对掩码区域深度做平滑，返回掩码内有效深度值。"""
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        valid = mask & (depth_image > 0) & np.isfinite(depth_image)
        if not np.any(valid):
            return np.array([], dtype=np.float32)

        work = depth_image.astype(np.float32).copy()
        fill_depth = float(np.median(work[valid]))
        work[~valid] = fill_depth

        smoothed = cv2.GaussianBlur(work, (kernel_size, kernel_size), 0)
        return smoothed[valid]

    # ─────────────────────────────────────────────
    #  7. 六维抓姿构建
    # ─────────────────────────────────────────────
    @staticmethod
    def compute_grasp_pose(center, longest_axis, table_normal, grasp_tilt_x_deg=0.0):
        """构建垂直于桌面的六维抓取姿态。

        姿态定义:
                    - approach (z 轴): 桌面法向量取反，指向桌面 (从上往下抓)
                    - longest_axis 投影到桌面上作为 y 轴（主轴方向，近→远）
                    - x 轴由 y × z 得到，满足右手系
                    - 最后可绕局部 x 轴旋转 grasp_tilt_x_deg（度）

        Parameters
        ----------
        center : np.ndarray (3,)
        longest_axis : np.ndarray (3,)
        table_normal : np.ndarray (3,)  指向相机侧
        grasp_tilt_x_deg : float
            绕抓手局部 x 轴旋转角度（度），正方向满足右手定则。

        Returns
        -------
        position : np.ndarray (3,)  抓取位置 (相机坐标系)
        quaternion : np.ndarray (4,)  [qx, qy, qz, qw] 抓取姿态
        rotation_matrix : np.ndarray (3,3)
        """
        # approach 方向: 法向量取反 → 指向桌面
        z_axis = -table_normal
        z_axis = z_axis / np.linalg.norm(z_axis)

        # y 轴: longest_axis 投影到桌面上 (去除法向量分量)
        y_axis = longest_axis - np.dot(longest_axis, table_normal) * table_normal
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            arbitrary = np.array([0.0, 1.0, 0.0])
            y_axis = arbitrary - np.dot(arbitrary, table_normal) * table_normal
            norm_y = np.linalg.norm(y_axis)
        y_axis = y_axis / norm_y

        # x 轴: y × z，确保 x × y = z
        x_axis = np.cross(y_axis, z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            arbitrary = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(arbitrary, z_axis)
            norm_x = np.linalg.norm(x_axis)
        x_axis = x_axis / norm_x

        # 正交化，减小数值误差
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 构建旋转矩阵 [x | y | z] (列向量)
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # 绕局部 x 轴旋转（可选）
        if abs(grasp_tilt_x_deg) > 1e-9:
            theta = np.deg2rad(grasp_tilt_x_deg)
            c, s = np.cos(theta), np.sin(theta)
            r_local_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ])
            rotation_matrix = rotation_matrix @ r_local_x

        # 确保是合法旋转矩阵 (det = +1)
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] = -rotation_matrix[:, 0]

        # 转四元数 [qx, qy, qz, qw]
        rot = Rotation.from_matrix(rotation_matrix)
        quaternion = rot.as_quat()  # [x, y, z, w]

        return center.copy(), quaternion, rotation_matrix

    @staticmethod
    def compose_transform_matrix(position, rotation_matrix):
        """由平移和 3x3 旋转矩阵构建 4x4 齐次变换矩阵。"""
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
        transform[:3, 3] = np.asarray(position, dtype=np.float64)
        return transform

    # ─────────────────────────────────────────────
    #  完整流程
    # ─────────────────────────────────────────────
    def process_frame(self, color_image, depth_image, intrinsics):
        """处理单帧 RGBD 数据并输出检测/抓姿结果。

        每个物体结果中包含 4x4 齐次变换矩阵: result["transform_matrix"]。
        """
        self.last_color_image = color_image.copy()
        self.last_intrinsics = dict(intrinsics)

        print("\n[Step 2] 生成点云 ...")
        points, colors, _ = self.generate_point_cloud(depth_image, intrinsics, color_image)
        self.last_points = points.copy()
        self.last_point_colors = None if colors is None else colors.copy()
        print(f"  有效点数: {len(points)}")

        print("\n[Step 3] RANSAC 桌面平面拟合 ...")
        _, table_normal, _ = self.fit_table_plane(points, colors)

        print("\n[Step 4] Grounded SAM2 物体检测与分割 ...")
        masks, class_names, confidences, boxes = self.detect_and_segment(color_image)
        if len(masks) == 0:
            print("[WARN] 未检测到任何物体，流程结束")
            return []

        print("\n[Step 5] PCA 分析 (中心: 2D, 主轴: 3D) + 抓姿构建 ...")
        print(f"  抓取中心模式: {self.grasp_center_mode}")
        results = []

        for i, (mask, cls_name, conf, box) in enumerate(
            zip(masks, class_names, confidences, boxes)
        ):
            print(f"\n  ── 物体 {i}: {cls_name} ({conf:.2f}) ──")
            try:
                obj_pts, _ = self.extract_object_points(mask, depth_image, intrinsics)
                if len(obj_pts) < 10:
                    print(f"  [SKIP] 物体点数过少 ({len(obj_pts)})")
                    continue

                center, _, center_2d, _ = self.pca_analysis_2d(
                    mask, depth_image, intrinsics
                )
                _, longest_axis, _, _ = self.pca_analysis_3d(obj_pts)

                longest_axis = self.orient_axis_near_to_far(center, longest_axis, obj_pts)

                grasp_center = self.compute_grasp_center(
                    center, longest_axis, obj_pts, mode="axis_midpoint"
                )
                valid_depth_values = self.smooth_mask_depth_values(
                    depth_image,
                    mask,
                    kernel_size=5,
                )
                if valid_depth_values.size > 0:
                    grasp_center = grasp_center.copy()
                    z_new = float(np.percentile(valid_depth_values, 5))
                    z_old = float(grasp_center[2])

                    if z_old > 1e-6:
                        fx, fy = intrinsics["fx"], intrinsics["fy"]
                        cx, cy = intrinsics["cx"], intrinsics["cy"]

                        u = fx * grasp_center[0] / z_old + cx
                        v = fy * grasp_center[1] / z_old + cy

                        grasp_center[0] = (u - cx) * z_new / fx
                        grasp_center[1] = (v - cy) * z_new / fy

                    grasp_center[2] = z_new
                else:
                    print("  [WARN] 掩码区域无有效深度，回退到 axis_midpoint")
                position, quaternion, rot_mat = self.compute_grasp_pose(
                    grasp_center, longest_axis, table_normal, grasp_tilt_x_deg=self.grasp_tilt_x_deg
                )
                transform_mat = self.compose_transform_matrix(position, rot_mat)
                print(f"  抓姿位置: {position}")
                print("  抓姿 4x4 齐次矩阵:")
                print(np.array2string(transform_mat, separator=", "))

                results.append({
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "box": np.asarray(box, dtype=np.float64),
                    "position": position,
                    "quaternion": quaternion,
                    "rotation_matrix": rot_mat,
                    "transform_matrix": transform_mat,
                    "center_2d": center_2d,
                    "longest_axis": longest_axis,
                    "mask": mask,
                })
            except Exception as e:
                print(f"  [ERROR] 处理失败: {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"  检测完成，共得到 {len(results)} 个抓姿")
        print(f"{'=' * 60}")
        return results

    def run(self):
        """执行完整的餐具抓姿检测流程。

        Returns
        -------
        results : list[dict]
            每个元素包含:
            - "class_name": str
            - "confidence": float
            - "box": np.ndarray (4,)  [x1, y1, x2, y2]
            - "position": np.ndarray (3,)
            - "quaternion": np.ndarray (4,)  [qx, qy, qz, qw]
            - "rotation_matrix": np.ndarray (3,3)
            - "transform_matrix": np.ndarray (4,4)
            - "center_2d": np.ndarray (2,) 或 None  (2D 模式时)
            - "longest_axis": np.ndarray (3,)
            - "mask": np.ndarray (H, W), bool
        """
        print("=" * 60)
        print("  餐具抓姿检测开始")
        print("=" * 60)

        # ── Step 1: 采集图像 ──
        print("\n[Step 1] 采集 RGB + Depth ...")
        color_image, depth_image, intrinsics = self.capture_rgbd()
        self.last_color_image = color_image.copy()
        self.last_intrinsics = dict(intrinsics)
        print(f"  图像尺寸: {color_image.shape[:2]}, 深度范围: "
              f"[{depth_image[depth_image > 0].min():.3f}, {depth_image[depth_image > 0].max():.3f}] m")

        return self.process_frame(color_image, depth_image, intrinsics)


# ═══════════════════════════════════════════════════════════════
#  可视化辅助函数
# ═══════════════════════════════════════════════════════════════
def visualize_results(color_image, results, intrinsics, save_path=None):
    """在 RGB 图像上可视化检测结果（掩码 + 六维抓姿投影）。

    Parameters
    ----------
    color_image : np.ndarray (H, W, 3), BGR
    results : list[dict]  run() 的输出
    intrinsics : dict  {"fx", "fy", "cx", "cy", "width", "height"}
    save_path : str, 可选, 保存路径
    """
    vis = color_image.copy()

    overlay = vis.copy()
    DEBUG_DRAW_2D = True
    colors_list = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (255, 128, 0),
    ]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    def project_point(point_3d):
        x, y, z = point_3d
        if z <= 1e-6:
            return None
        u = int(round(fx * x / z + cx))
        v = int(round(fy * y / z + cy))
        return (u, v)

    def draw_debug_marker(canvas, pt, name, color, offset=(6, -6)):
        if pt is None:
            return
        cv2.circle(canvas, pt, 4, color, -1)
        tx = pt[0] + offset[0]
        ty = pt[1] + offset[1]
        cv2.putText(
            canvas,
            name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for i, res in enumerate(results):
        color = colors_list[i % len(colors_list)]
        mask = res["mask"]

        # 半透明掩码
        overlay[mask] = (
            np.array(overlay[mask], dtype=np.float32) * 0.5
            + np.array(color, dtype=np.float32) * 0.5
        ).astype(np.uint8)

        # 标签
        if res["center_2d"] is not None:
            cu, cv = int(res["center_2d"][0]), int(res["center_2d"][1])
        else:
            ys, xs = np.where(mask)
            cu, cv = int(xs.mean()), int(ys.mean())
        center2d_px = (cu, cv)

        label = f'{res["class_name"]} {res["confidence"]:.2f}'
        cv2.putText(overlay, label, (cu, cv - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(overlay, (cu, cv), 5, color, -1)

        grasp_center = np.asarray(res["position"], dtype=np.float64)
        rot_mat = np.asarray(res["rotation_matrix"], dtype=np.float64)

        axis_length = max(0.03, min(0.12, grasp_center[2] * 0.25))
        gripper_half_width = axis_length * 0.35
        approach_backoff = axis_length * 0.6

        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]
        main_axis = np.asarray(res.get("longest_axis", y_axis), dtype=np.float64)
        main_axis_norm = np.linalg.norm(main_axis)
        if main_axis_norm < 1e-9:
            main_axis = y_axis.copy()
        else:
            main_axis = main_axis / main_axis_norm

        center_px = project_point(grasp_center)
        x_end_px = project_point(grasp_center + x_axis * axis_length)
        y_end_px = project_point(grasp_center + y_axis * axis_length)
        z_end_px = project_point(grasp_center + z_axis * axis_length)
        main_axis_half = axis_length * 0.8
        main_near_px = project_point(grasp_center - main_axis * main_axis_half)
        main_far_px = project_point(grasp_center + main_axis * main_axis_half)

        jaw_left_px = project_point(grasp_center - y_axis * gripper_half_width)
        jaw_right_px = project_point(grasp_center + y_axis * gripper_half_width)
        pregrasp_px = project_point(grasp_center - z_axis * approach_backoff)

        if center_px is not None:
            cv2.circle(overlay, center_px, 6, (255, 255, 255), -1)
            cv2.circle(overlay, center_px, 4, color, -1)

        if center_px is not None and x_end_px is not None:
            cv2.arrowedLine(overlay, center_px, x_end_px, (0, 0, 255), 2, tipLength=0.2)
        if center_px is not None and y_end_px is not None:
            cv2.arrowedLine(overlay, center_px, y_end_px, (0, 255, 0), 2, tipLength=0.2)
        if center_px is not None and z_end_px is not None:
            cv2.arrowedLine(overlay, center_px, z_end_px, (255, 0, 0), 2, tipLength=0.2)

        if jaw_left_px is not None and jaw_right_px is not None:
            cv2.line(overlay, jaw_left_px, jaw_right_px, (0, 255, 255), 2)
            cv2.circle(overlay, jaw_left_px, 4, (0, 255, 255), -1)
            cv2.circle(overlay, jaw_right_px, 4, (0, 255, 255), -1)

        if pregrasp_px is not None and center_px is not None:
            cv2.arrowedLine(
                overlay, pregrasp_px, center_px, (255, 255, 0), 2, tipLength=0.2
            )

        if main_near_px is not None and main_far_px is not None:
            cv2.line(overlay, main_near_px, main_far_px, (255, 0, 255), 2)
            cv2.arrowedLine(
                overlay, main_near_px, main_far_px, (255, 0, 255), 2, tipLength=0.25
            )
            cv2.circle(overlay, main_near_px, 4, (0, 255, 255), -1)
            cv2.circle(overlay, main_far_px, 4, (255, 0, 255), -1)

        if DEBUG_DRAW_2D:
            draw_debug_marker(overlay, center2d_px, "center_2d", (0, 255, 0), offset=(6, 14))
            draw_debug_marker(overlay, center_px, "grasp_center", (255, 255, 255))
            draw_debug_marker(overlay, jaw_left_px, "jaw_l", (0, 255, 255), offset=(6, -10))
            draw_debug_marker(overlay, jaw_right_px, "jaw_r", (0, 255, 255), offset=(6, 14))
            draw_debug_marker(overlay, x_end_px, "x", (0, 0, 255))
            draw_debug_marker(overlay, y_end_px, "y", (0, 255, 0))
            draw_debug_marker(overlay, z_end_px, "z", (255, 0, 0))
            draw_debug_marker(overlay, pregrasp_px, "pregrasp", (255, 255, 0), offset=(6, 14))
            draw_debug_marker(overlay, main_near_px, "main_near", (0, 255, 255), offset=(6, -10))
            draw_debug_marker(overlay, main_far_px, "main_far", (255, 0, 255), offset=(6, 14))

    vis = overlay

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"[INFO] 可视化结果已保存至: {save_path}")

    return vis


def visualize_results_3d(points, colors, results):
    """用 Open3D 可视化场景点云和六维抓姿。"""
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        scene_pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        scene_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(scene_pcd)

    pose_colors = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.4, 1.0],
            [1.0, 0.7, 0.2],
            [0.8, 0.2, 1.0],
            [0.2, 0.9, 0.9],
        ],
        dtype=np.float64,
    )

    for i, res in enumerate(results):
        center = np.asarray(res["position"], dtype=np.float64)
        rot_mat = np.asarray(res["rotation_matrix"], dtype=np.float64)
        pose_color = pose_colors[i % len(pose_colors)]

        axis_length = max(0.03, min(0.12, center[2] * 0.25))
        gripper_depth = axis_length * 0.5
        gripper_half_width = axis_length * 0.35
        finger_length = axis_length * 0.45

        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]
        main_axis = np.asarray(res.get("longest_axis", y_axis), dtype=np.float64)
        main_axis_norm = np.linalg.norm(main_axis)
        if main_axis_norm < 1e-9:
            main_axis = y_axis.copy()
        else:
            main_axis = main_axis / main_axis_norm

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
        frame.rotate(rot_mat, center=np.zeros(3))
        frame.translate(center)
        geometries.append(frame)

        pregrasp_center = center - z_axis * gripper_depth
        jaw_left = center - y_axis * gripper_half_width
        jaw_right = center + y_axis * gripper_half_width
        jaw_left_back = pregrasp_center - y_axis * gripper_half_width
        jaw_right_back = pregrasp_center + y_axis * gripper_half_width
        jaw_left_tip = jaw_left + z_axis * finger_length
        jaw_right_tip = jaw_right + z_axis * finger_length

        gripper_points = np.vstack([
            pregrasp_center,
            jaw_left_back,
            jaw_right_back,
            jaw_left,
            jaw_right,
            jaw_left_tip,
            jaw_right_tip,
        ])
        gripper_lines = np.array(
            [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
            ],
            dtype=np.int32,
        )
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(gripper_points)
        line_set.lines = o3d.utility.Vector2iVector(gripper_lines)
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(pose_color, (len(gripper_lines), 1))
        )
        geometries.append(line_set)

        main_axis_half = axis_length * 0.8
        main_near = center - main_axis * main_axis_half
        main_far = center + main_axis * main_axis_half

        main_axis_lines = o3d.geometry.LineSet()
        main_axis_lines.points = o3d.utility.Vector3dVector(np.vstack([main_near, main_far]))
        main_axis_lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        main_axis_lines.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 1.0]], dtype=np.float64))
        geometries.append(main_axis_lines)

        near_marker = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.08)
        near_marker.paint_uniform_color([1.0, 1.0, 0.0])
        near_marker.translate(main_near)
        geometries.append(near_marker)

        far_marker = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.08)
        far_marker.paint_uniform_color([1.0, 0.0, 1.0])
        far_marker.translate(main_far)
        geometries.append(far_marker)

    print("[INFO] 打开 3D 抓姿可视化窗口，关闭后继续 ...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Grasp Poses",
    )


# ═══════════════════════════════════════════════════════════════
#  独立运行入口
# ═══════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    required_paths = {
        "SAM2 checkpoint": DEFAULT_SAM2_CHECKPOINT,
        "SAM2 config": resolve_sam2_config_path(DEFAULT_SAM2_CONFIG),
        "GroundingDINO config": DEFAULT_GDINO_CONFIG,
        "GroundingDINO checkpoint": DEFAULT_GDINO_CHECKPOINT,
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 不存在: {path}")

    print("[INFO] 正在加载 SAM2 ...")
    sam2_model = build_sam2(DEFAULT_SAM2_CONFIG, DEFAULT_SAM2_CHECKPOINT, device=device)
    global_sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("[INFO] 正在加载 Grounding DINO ...")
    global_gdino_model = load_gdino_model(
        model_config_path=DEFAULT_GDINO_CONFIG,
        model_checkpoint_path=DEFAULT_GDINO_CHECKPOINT,
        device=device,
    )
    print("[INFO] 模型加载完成")

    detector = TablewareGraspDetector(
        sam2_predictor=global_sam2_predictor,
        grounding_model=global_gdino_model,
        text_prompt="fork. spoon.",
        grasp_tilt_x_deg=-40,  # 可调：绕局部 x 轴旋转，控制斜向抓取
        depth_range=(0.1, 0.8),  # 根据实际桌面高度调整
        device=device,
    )

    try:
        results = detector.run()

        # 可视化结果
        if hasattr(detector, "last_color_image") and hasattr(detector, "last_intrinsics"):
            vis_image = visualize_results(
                detector.last_color_image,
                results,
                detector.last_intrinsics,
            )
            cv2.imshow("Tableware Grasp Detection", vis_image)
            print("[INFO] 按任意键关闭可视化窗口 ...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if hasattr(detector, "last_points"):
            visualize_results_3d(
                detector.last_points,
                getattr(detector, "last_point_colors", None),
                results,
            )

        for i, r in enumerate(results):
            print(f"\n物体 {i}: {r['class_name']}")
            print("  4x4 齐次矩阵:")
            print(np.array2string(r["transform_matrix"], separator=", "))
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
