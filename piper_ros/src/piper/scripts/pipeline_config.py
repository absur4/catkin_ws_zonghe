#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一管线配置 (Unified Pipeline Configuration)

所有输入源、算法参数、路径配置在此文件中集中管理。
上层代码通过修改此配置并调用 run_pipeline_cross_env.run_pipeline() 启动管线。
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = str(SCRIPT_DIR.parent / "pc_npz")


@dataclass
class PipelineConfig:
    """抓取检测管线完整配置。

    修改此处的参数即可控制整个管线的行为。
    """

    # ── 输入源 ──
    input_source: str = "rgbd_files"            # "camera" (RealSense 实时) | "rgbd_files" (文件输入)
    rgb_path: str = "/home/h/PPC/src/GraspGen/rs_data/yinliao_shu/color.png"                      # rgbd_files 模式: RGB 图像路径
    depth_path: str = "/home/h/PPC/src/GraspGen/rs_data/yinliao_shu/depth.png"                    # rgbd_files 模式: Depth 图像路径 (.png/.npy)
    depth_value_in_meters: bool = False     
    depth_scale: float = 1000.0             

    # ── 相机内参 (rgbd_files 模式使用) ──
    intrinsics: dict = field(default_factory=lambda: {
        "fx": 901.2042236328125,
        "fy": 901.2002563476562,
        "cx": 656.2393188476562,
        "cy": 366.75244140625,
        "width": 1280,
        "height": 720,
    })

    # ── 物体选择策略 ──
    target_object_policy: str = "top_conf"  # "top_conf" (最高置信度) | "index" (指定索引)
    target_object_index: int = 0

    # ── 检测目标（常用，单独提升） ──
    text_prompt: str = "bottle."              # Grounding DINO 文本提示，类别以 ". " 分隔

    # ── 分割器参数 (ObjectPointCloudSegmenter 构造参数) ──
    segmenter_kwargs: dict = field(default_factory=lambda: {
        "output_dir": "/tmp/object_clouds",
        "voxel_size": 0.01,
        "table_remove_thresh": 0.01,
        "depth_range": [0.1, 1.0],          # JSON 序列化需用 list，传入时自动转 tuple
        "bbox_horizontal_shrink_ratio": 0.12,
        "save_clouds_to_disk": False,
        "segmentation_method": "bbox",  # 'bbox' 或 'sam2_mask'
        "sam2_checkpoint": "",          # 从外部传入，或让底层自行寻找
        "sam2_config": "",
    })

    # ── 抓取估计器参数 (GraspGenPointCloudConfig 的覆盖项) ──
    # 只需包含想覆盖的字段，其余使用 GraspGenPointCloudConfig 类默认值。
    # 可覆盖字段参见 graspgen_pc_grasp_estimator.py 中 GraspGenPointCloudConfig 定义。
    estimator_config: dict = field(default_factory=lambda: {
        "grasp_threshold": 0.80,
        "num_grasps": 200,
        "return_topk": True,
        "topk_num_grasps": 20,
        "filter_collisions": True,
        "collision_threshold": 0.001,
        "collision_local_radius_m": 0.20,
        "collision_scene_sampling_mode": "legacy_random",
        "enable_z_angle_filter": True,
        "max_z_angle_deg": 30.0,
        "enable_y_axis_angle_filter": True,
        "min_y_axis_angle_deg": 90.0,
    })

    # ── 可视化 ──
    enable_visualization: bool = True
    num_visualize_grasps: int = 10
    block_after_visualization: bool = False

    # ── Conda 环境 ──
    conda_exe: str = "conda"
    seg_env_name: str = "dsam2"
    grasp_env_name: str = "GraspGen"

    # ── 工作目录与文件名 ──
    work_dir: str = DEFAULT_WORK_DIR
    npz_filename: str = "scene_object_clouds.npz"
    result_json_filename: str = "grasp_result.json"

    @property
    def npz_path(self) -> str:
        """NPZ 中间文件的完整路径。"""
        return str(Path(self.work_dir) / self.npz_filename)

    @property
    def result_json_path(self) -> str:
        """抓取结果 JSON 的完整路径。"""
        return str(Path(self.work_dir) / self.result_json_filename)

    def to_json(self, path: str):
        """序列化配置到 JSON 文件（用于跨 conda 环境传递）。"""
        d = asdict(self)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """从 JSON 文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)
