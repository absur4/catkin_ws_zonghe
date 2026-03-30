#!/usr/bin/env python3
"""
API 使用示例

演示如何使用 Grounding DINO 和 Grounding SAM API
"""

import os
from grounding_dino_api import GroundingDINOAPI
from grounding_sam_api import GroundingSAMAPI


def example_grounding_dino():
    """Grounding DINO API 使用示例"""
    print("\n" + "="*60)
    print("示例 1: 使用 Grounding DINO API 检测物体")
    print("="*60 + "\n")

    # 创建 API 实例
    api = GroundingDINOAPI()

    # 检测物体
    results = api.detect(
        image_path="../Grounded-SAM-2/bag.jpg",
        text_prompt="bag.",
        output_path="example_output/dino_result.jpg"
    )

    # 打印结果
    print(f"\n检测到 {len(results['boxes'])} 个物体:")
    for i, (label, conf, box) in enumerate(zip(results['labels'], results['confidences'], results['boxes'])):
        print(f"  {i+1}. {label} (置信度: {conf:.3f})")
        print(f"     边界框: {box}")


def example_grounding_sam():
    """Grounding SAM API 使用示例"""
    print("\n" + "="*60)
    print("示例 2: 使用 Grounding SAM API 检测和分割物体")
    print("="*60 + "\n")

    # 创建 API 实例
    api = GroundingSAMAPI()

    # 检测并分割物体
    results = api.segment(
        image_path="../Grounded-SAM-2/bag.jpg",
        text_prompt="bag.",
        output_dir="example_output/sam_results/"
    )

    # 打印结果
    print(f"\n检测到 {len(results['boxes'])} 个物体:")
    for i, (label, conf) in enumerate(zip(results['labels'], results['confidences'])):
        print(f"  {i+1}. {label} (置信度: {conf:.3f})")

    print(f"\n生成的文件:")
    for key, path in results['output_files'].items():
        print(f"  - {key}: {path}")


def example_custom_parameters():
    """使用自定义参数的示例"""
    print("\n" + "="*60)
    print("示例 3: 使用自定义参数")
    print("="*60 + "\n")

    # 创建 API 实例，使用更高的阈值
    api = GroundingDINOAPI(
        box_threshold=0.5,  # 更高的阈值，减少误检
        text_threshold=0.3
    )

    results = api.detect(
        image_path="../Grounded-SAM-2/bag.jpg",
        text_prompt="bag.",
        output_path="example_output/dino_high_threshold.jpg"
    )

    print(f"使用更高阈值检测到 {len(results['boxes'])} 个物体")


def example_multiple_objects():
    """检测多个物体类别的示例"""
    print("\n" + "="*60)
    print("示例 4: 检测多个物体类别")
    print("="*60 + "\n")

    api = GroundingSAMAPI()

    # 检测多个物体（需要有相应的测试图像）
    # 这里只是示例代码
    results = api.segment(
        image_path="../Grounded-SAM-2/bag.jpg",
        text_prompt="bag. person. bottle.",  # 多个物体用 . 分隔
        output_dir="example_output/multiple_objects/"
    )

    print(f"检测到 {len(results['boxes'])} 个物体")
    print(f"类别: {set(results['labels'])}")


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("example_output", exist_ok=True)

    print("\n" + "="*60)
    print("Open Vocabulary Detection API 使用示例")
    print("="*60)

    try:
        # 运行示例
        example_grounding_dino()
        example_grounding_sam()
        example_custom_parameters()
        example_multiple_objects()

        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("\n查看输出文件: example_output/\n")

    except FileNotFoundError as e:
        print(f"\n⚠️  注意: 找不到测试图像文件")
        print(f"错误: {e}")
        print("\n请确保 bag.jpg 存在于 ../Grounded-SAM-2/ 目录下")
        print("或者修改示例代码中的图像路径\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()