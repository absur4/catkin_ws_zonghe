#!/usr/bin/env python3
"""
Grounding DINO API
用于检测图像中指定物体的边界框

使用示例:
    from grounding_dino_api import GroundingDINOAPI

    api = GroundingDINOAPI()
    results = api.detect(
        image_path="input/image.jpg",
        text_prompt="cat. dog.",
        output_path="output/result.jpg"
    )
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "Grounded-SAM-2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "grounding_dino"))

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# 导入配置
try:
    from config import (
        GROUNDING_DINO_CONFIG as DEFAULT_CONFIG,
        GROUNDING_DINO_CHECKPOINT as DEFAULT_CHECKPOINT,
        BOX_THRESHOLD as DEFAULT_BOX_THRESHOLD,
        TEXT_THRESHOLD as DEFAULT_TEXT_THRESHOLD
    )
except ImportError:
    # 如果无法导入配置，使用硬编码路径
    DEFAULT_CONFIG = os.path.join(
        PROJECT_ROOT,
        "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    DEFAULT_CHECKPOINT = os.path.join(
        PROJECT_ROOT,
        "gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    DEFAULT_BOX_THRESHOLD = 0.35
    DEFAULT_TEXT_THRESHOLD = 0.25


class GroundingDINOAPI:
    """Grounding DINO 物体检测 API"""

    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        box_threshold=0.35,
        text_threshold=0.25,
        device=None
    ):
        """
        初始化 Grounding DINO API

        Args:
            config_path: 配置文件路径，默认使用项目配置
            checkpoint_path: 模型权重路径，默认使用项目权重
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配置信度阈值
            device: 计算设备 ('cuda', 'mps', 'cpu')，默认自动选择
        """
        # 设置默认路径（从配置文件读取）
        if config_path is None:
            config_path = DEFAULT_CONFIG
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.box_threshold = box_threshold if box_threshold is not None else DEFAULT_BOX_THRESHOLD
        self.text_threshold = text_threshold if text_threshold is not None else DEFAULT_TEXT_THRESHOLD

        print(f"正在加载 Grounding DINO 模型...")
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=self.device
        )

        print("模型加载完成！")

    def detect(
        self,
        image_path,
        text_prompt,
        output_path=None,
        box_threshold=None,
        text_threshold=None
    ):
        """
        检测图像中的指定物体

        Args:
            image_path: 输入图像路径
            text_prompt: 要检测的物体文本提示，多个物体用". "分隔，如 "cat. dog."
            output_path: 输出图像路径（可选），如果提供则保存标注图像
            box_threshold: 边界框置信度阈值（可选）
            text_threshold: 文本匹配置信度阈值（可选）

        Returns:
            dict: 包含检测结果的字典
                - boxes: 边界框列表 [[x1, y1, x2, y2], ...]
                - confidences: 置信度列表
                - labels: 标签列表
                - image_path: 输入图像路径
                - output_path: 输出图像路径（如果保存）
        """
        # 使用实例阈值或传入阈值
        box_thresh = box_threshold if box_threshold is not None else self.box_threshold
        text_thresh = text_threshold if text_threshold is not None else self.text_threshold

        # 确保文本提示以点结尾
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'

        print(f"正在处理图像: {image_path}")
        print(f"检测物体: {text_prompt}")

        # 加载图像
        image_source, image = load_image(image_path)

        # 执行检测
        boxes, confidences, labels = predict(
            model=self.model,
            image=image,
            caption=text_prompt.lower(),  # Grounding DINO 需要小写
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            device=self.device
        )

        # 转换坐标
        h, w, _ = image_source.shape
        boxes_np = boxes.cpu().numpy() * np.array([w, h, w, h])

        # 转换为 xyxy 格式
        from torchvision.ops import box_convert
        boxes_xyxy = box_convert(
            boxes=torch.from_numpy(boxes_np),
            in_fmt="cxcywh",
            out_fmt="xyxy"
        ).numpy()

        confidences_np = confidences.cpu().numpy()

        print(f"检测到 {len(boxes_xyxy)} 个物体")

        # 如果提供输出路径，保存标注图像
        if output_path is not None:
            self._save_annotated_image(
                image_source,
                boxes_xyxy,
                confidences_np,
                labels,
                output_path
            )

        return {
            "boxes": boxes_xyxy.tolist(),
            "confidences": confidences_np.tolist(),
            "labels": labels,
            "image_path": image_path,
            "output_path": output_path
        }

    def _save_annotated_image(self, image, boxes, confidences, labels, output_path):
        """保存标注后的图像"""
        # 如果 output_path 是目录，自动生成文件名
        if os.path.isdir(output_path) or output_path.endswith('/'):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "annotated_image.jpg")
        else:
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        # 在图像上绘制边界框
        annotated_image = image.copy()

        for box, conf, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = map(int, box)

            # 绘制边界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 保存图像
        cv2.imwrite(output_path, annotated_image)
        print(f"标注图像已保存至: {output_path}")


# 命令行使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounding DINO 物体检测 API")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--prompt", required=True, help="检测物体文本提示，如 'cat. dog.'")
    parser.add_argument("--output", required=True, help="输出图像路径")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="边界框阈值")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="文本阈值")

    args = parser.parse_args()

    # 创建 API 实例
    api = GroundingDINOAPI(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    # 执行检测
    results = api.detect(
        image_path=args.image,
        text_prompt=args.prompt,
        output_path=args.output
    )

    print(f"\n检测结果:")
    print(f"- 检测到 {len(results['boxes'])} 个物体")
    print(f"- 标签: {results['labels']}")
    print(f"- 置信度: {[f'{c:.3f}' for c in results['confidences']]}")