#!/usr/bin/env python3
"""
Grounding SAM API
结合 Grounding DINO 和 SAM 2，用于检测和分割图像中的指定物体

使用示例:
    from grounding_sam_api import GroundingSAMAPI

    api = GroundingSAMAPI()
    results = api.segment(
        image_path="input/image.jpg",
        text_prompt="cat. dog.",
        output_dir="output/"
    )
"""

import os
import sys
import json
import torch
import cv2
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "Grounded-SAM-2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "grounding_dino"))

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

# 导入配置
try:
    from config import (
        SAM2_CHECKPOINT as DEFAULT_SAM2_CHECKPOINT,
        SAM2_CONFIG as DEFAULT_SAM2_CONFIG,
        GROUNDING_DINO_CONFIG as DEFAULT_DINO_CONFIG,
        GROUNDING_DINO_CHECKPOINT as DEFAULT_DINO_CHECKPOINT,
        BOX_THRESHOLD as DEFAULT_BOX_THRESHOLD,
        TEXT_THRESHOLD as DEFAULT_TEXT_THRESHOLD
    )
except ImportError:
    # 如果无法导入配置，使用硬编码路径
    DEFAULT_SAM2_CHECKPOINT = os.path.join(
        PROJECT_ROOT, "checkpoints/sam2.1_hiera_large.pt"
    )
    DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEFAULT_DINO_CONFIG = os.path.join(
        PROJECT_ROOT,
        "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    DEFAULT_DINO_CHECKPOINT = os.path.join(
        PROJECT_ROOT, "gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    DEFAULT_BOX_THRESHOLD = 0.35
    DEFAULT_TEXT_THRESHOLD = 0.25


class GroundingSAMAPI:
    """Grounding SAM (DINO + SAM2) 物体检测和分割 API"""

    def __init__(
        self,
        sam2_checkpoint=None,
        sam2_config=None,
        grounding_dino_config=None,
        grounding_dino_checkpoint=None,
        box_threshold=0.35,
        text_threshold=0.25,
        device=None
    ):
        """
        初始化 Grounding SAM API

        Args:
            sam2_checkpoint: SAM2 模型权重路径
            sam2_config: SAM2 配置文件路径
            grounding_dino_config: Grounding DINO 配置文件路径
            grounding_dino_checkpoint: Grounding DINO 权重路径
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配置信度阈值
            device: 计算设备 ('cuda', 'mps', 'cpu')
        """
        # 设置默认路径（从配置文件读取）
        if sam2_checkpoint is None:
            sam2_checkpoint = DEFAULT_SAM2_CHECKPOINT
        if sam2_config is None:
            sam2_config = DEFAULT_SAM2_CONFIG
        if grounding_dino_config is None:
            grounding_dino_config = DEFAULT_DINO_CONFIG
        if grounding_dino_checkpoint is None:
            grounding_dino_checkpoint = DEFAULT_DINO_CHECKPOINT

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

        print(f"正在加载 Grounding SAM 模型...")
        print(f"使用设备: {self.device}")

        # 加载 SAM 2 模型
        print("加载 SAM 2 模型...")
        # 需要切换到项目根目录以正确加载配置
        original_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)

        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        os.chdir(original_dir)

        # 加载 Grounding DINO 模型
        print("加载 Grounding DINO 模型...")
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=self.device
        )

        print("所有模型加载完成！")

    def segment(
        self,
        image_path,
        text_prompt,
        output_dir=None,
        save_mask=True,
        save_annotated=True,
        save_json=True,
        box_threshold=None,
        text_threshold=None
    ):
        """
        分割图像中的指定物体

        Args:
            image_path: 输入图像路径
            text_prompt: 要检测的物体文本提示，如 "cat. dog."
            output_dir: 输出目录路径（可选）
            save_mask: 是否保存分割掩码
            save_annotated: 是否保存标注图像
            save_json: 是否保存 JSON 结果
            box_threshold: 边界框置信度阈值（可选）
            text_threshold: 文本匹配置信度阈值（可选）

        Returns:
            dict: 包含分割结果的字典
                - boxes: 边界框列表
                - confidences: 置信度列表
                - labels: 标签列表
                - masks: 分割掩码
                - image_path: 输入图像路径
                - output_files: 输出文件路径字典
        """
        # 使用实例阈值或传入阈值
        box_thresh = box_threshold if box_threshold is not None else self.box_threshold
        text_thresh = text_threshold if text_threshold is not None else self.text_threshold

        # 确保文本提示以点结尾
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'

        print(f"\n{'='*50}")
        print(f"正在处理图像: {image_path}")
        print(f"检测物体: {text_prompt}")
        print(f"{'='*50}\n")

        # 加载图像
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape

        # Step 1: Grounding DINO 检测
        print("Step 1: 使用 Grounding DINO 检测物体...")
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt.lower(),
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            device=self.device
        )

        # 转换边界框格式
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        print(f"检测到 {len(input_boxes)} 个物体")

        if len(input_boxes) == 0:
            print("未检测到任何物体！")
            return {
                "boxes": [],
                "confidences": [],
                "labels": [],
                "masks": None,
                "image_path": image_path,
                "output_files": {}
            }

        # Step 2: SAM 2 分割
        print("\nStep 2: 使用 SAM 2 生成分割掩码...")
        self.sam2_predictor.set_image(image_source)

        # 设置自动混合精度
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # 转换掩码形状
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        print(f"生成了 {len(masks)} 个分割掩码")

        # 准备结果
        confidences_np = confidences.numpy().tolist()
        class_names = labels

        output_files = {}

        # 保存结果
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            # 保存掩码
            if save_mask and len(masks) > 0:
                mask_path = os.path.join(output_dir, "segmentation_mask.png")
                self._save_mask(masks[0], mask_path)
                output_files['mask'] = mask_path

            # 保存标注图像
            if save_annotated:
                annotated_path = os.path.join(output_dir, "annotated_image.jpg")
                self._save_annotated_image(
                    image_path, input_boxes, masks, confidences_np, class_names, annotated_path
                )
                output_files['annotated'] = annotated_path

            # 保存 JSON 结果
            if save_json:
                json_path = os.path.join(output_dir, "results.json")
                self._save_json_results(
                    image_path, input_boxes, masks, confidences_np, class_names, w, h, json_path
                )
                output_files['json'] = json_path

        print(f"\n{'='*50}")
        print("处理完成！")
        print(f"{'='*50}\n")

        return {
            "boxes": input_boxes.tolist(),
            "confidences": confidences_np,
            "labels": class_names,
            "masks": masks,
            "image_path": image_path,
            "output_files": output_files
        }

    def _save_mask(self, mask, output_path):
        """保存分割掩码"""
        from PIL import Image

        # 转换掩码为 uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # 保存为图像
        pil_image = Image.fromarray(mask_uint8)
        pil_image_1bit = pil_image.convert("1")
        pil_image_1bit.save(output_path)

        print(f"分割掩码已保存至: {output_path}")

    def _save_annotated_image(self, img_path, boxes, masks, confidences, labels, output_path):
        """保存标注后的图像"""
        img = cv2.imread(img_path)

        # 创建检测对象
        class_ids = np.array(list(range(len(labels))))
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        # 创建标签
        detection_labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(labels, confidences)
        ]

        # 绘制边界框
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        # 绘制标签
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=detection_labels
        )

        # 绘制掩码
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # 保存
        cv2.imwrite(output_path, annotated_frame)
        print(f"标注图像已保存至: {output_path}")

    def _save_json_results(self, img_path, boxes, masks, confidences, labels, w, h, output_path):
        """保存 JSON 格式的结果"""
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        # 转换掩码为 RLE 格式
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        # 构建结果
        results = {
            "image_path": img_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "confidence": conf,
                }
                for class_name, box, mask_rle, conf in zip(labels, boxes, mask_rles, confidences)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        # 保存
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"JSON 结果已保存至: {output_path}")


# 命令行使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounding SAM 物体分割 API")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--prompt", required=True, help="检测物体文本提示，如 'cat. dog.'")
    parser.add_argument("--output", required=True, help="输出目录路径")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="边界框阈值")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="文本阈值")

    args = parser.parse_args()

    # 创建 API 实例
    api = GroundingSAMAPI(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    # 执行分割
    results = api.segment(
        image_path=args.image,
        text_prompt=args.prompt,
        output_dir=args.output
    )

    print(f"\n分割结果:")
    print(f"- 检测到 {len(results['boxes'])} 个物体")
    print(f"- 标签: {results['labels']}")
    print(f"- 置信度: {[f'{c:.3f}' for c in results['confidences']]}")
    print(f"- 输出文件: {list(results['output_files'].keys())}")