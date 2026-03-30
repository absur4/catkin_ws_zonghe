#!/usr/bin/env python3
"""
Batch runner for the local closed-set detector.
"""

import argparse
import json
import os
from pathlib import Path

from closed_set_object_detector import ClosedSetObjectDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Batch run local closed-set detector")
    parser.add_argument(
        "--test-root",
        default="/Users/zhanghanyu/Desktop/objects/photo_rgb_depth/photos/test",
        help="测试图片根目录",
    )
    parser.add_argument(
        "--train-root",
        default="/Users/zhanghanyu/Desktop/objects/photo_rgb_depth/photos/train",
        help="按类别分目录的 support 图像目录",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/zhanghanyu/Desktop/objects/open_vocabulary/API/output",
        help="批量输出目录",
    )
    parser.add_argument(
        "--objects",
        default="milk. cup. plate. apple. hot dog. crop.",
        help="Grounding DINO 检测词",
    )
    parser.add_argument("--device", default="mps", help="cpu / mps / cuda")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument(
        "--support-cache-dir",
        default="/Users/zhanghanyu/Desktop/objects/open_vocabulary/API/.cache/closed_set_support",
        help="增强后的 support 特征缓存目录",
    )
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--proker-beta", type=float, default=None)
    parser.add_argument("--proker-lambda", type=float, default=0.1)
    parser.add_argument("--augment-epoch", type=int, default=1)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--whitening-log-like-threshold", type=float, default=-9000.0)
    parser.add_argument("--classify-crop-shrink", type=float, default=0.15)
    parser.add_argument("--overlap-containment-ratio", type=float, default=0.8)
    parser.add_argument("--overlap-area-ratio", type=float, default=1.35)
    parser.add_argument("--override-prompts", nargs="*", default=["plate.", "spoon."])
    parser.add_argument("--override-box-threshold", type=float, default=0.45)
    parser.add_argument("--override-text-threshold", type=float, default=None)
    parser.add_argument("--override-iou", type=float, default=0.3)
    parser.add_argument("--override-containment", type=float, default=0.6)
    args = parser.parse_args()

    test_root = Path(args.test_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ClosedSetObjectDetector(
        train_root=args.train_root,
        device=args.device,
        siglip_model_name=args.siglip_model,
        support_cache_dir=args.support_cache_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        proker_beta=args.proker_beta,
        proker_lambda=args.proker_lambda,
        augment_epoch=args.augment_epoch,
        min_similarity=args.min_similarity,
        whitening_log_like_threshold=args.whitening_log_like_threshold,
    )

    summaries = []
    for image_path in iter_images(test_root):
        rel = image_path.relative_to(test_root)
        stem = image_path.stem
        rel_parent = rel.parent
        target_dir = output_dir / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        output_path = target_dir / f"{stem}_siglip.jpg"
        json_path = target_dir / f"{stem}_siglip.json"

        result = detector.detect_and_classify(
            image_path=str(image_path),
            objects=args.objects,
            output_path=str(output_path),
            save_json_path=str(json_path),
            classify_crop_shrink=args.classify_crop_shrink,
            overlap_containment_ratio=args.overlap_containment_ratio,
            overlap_area_ratio=args.overlap_area_ratio,
            override_prompts=args.override_prompts,
            override_box_threshold=args.override_box_threshold,
            override_text_threshold=args.override_text_threshold,
            override_iou=args.override_iou,
            override_containment=args.override_containment,
        )
        summary = {
            "image": str(image_path),
            "accepted": len(result["accepted_detections"]),
            "all_detections": len(result["detections"]),
            "elapsed_time_sec": result["elapsed_time_sec"],
            "output": str(output_path),
            "json": str(json_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": str(summary_path), "count": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
