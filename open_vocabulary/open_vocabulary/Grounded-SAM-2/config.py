#!/usr/bin/env python3
"""
Grounded-SAM-2 项目配置文件
所有路径和参数配置都在这里统一管理
"""

import os
import torch

# ============================================================================
# 项目路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型权重路径
SAM2_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints/sam2.1_hiera_large.pt")
GROUNDING_DINO_CHECKPOINT = os.path.join(PROJECT_ROOT, "gdino_checkpoints/groundingdino_swint_ogc.pth")

# 模型配置文件路径
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # 相对路径，由 hydra 加载
GROUNDING_DINO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

# 输出目录
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Grounding DINO 子项目路径
GROUNDING_DINO_ROOT = os.path.join(PROJECT_ROOT, "grounding_dino")

# ============================================================================
# 模型参数配置
# ============================================================================

# 检测阈值
BOX_THRESHOLD = 0.35        # 边界框置信度阈值
TEXT_THRESHOLD = 0.25       # 文本匹配置信度阈值

# 设备配置
def get_device():
    """
    自动选择最优计算设备
    优先级: CUDA (NVIDIA GPU) > MPS (Mac GPU) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

# ============================================================================
# 默认示例配置
# ============================================================================

# 默认测试图像
DEFAULT_TEST_IMAGE = os.path.join(PROJECT_ROOT, "bag.jpg")

# 默认文本提示
DEFAULT_TEXT_PROMPT = "bag."

# ============================================================================
# 输出设置
# ============================================================================

# 是否保存 JSON 结果
DUMP_JSON_RESULTS = True

# 是否保存分割掩码
SAVE_MASK = True

# 是否保存标注图像
SAVE_ANNOTATED = True

# ============================================================================
# 高级设置
# ============================================================================

# 是否使用混合精度计算
USE_AUTOCAST = True

# 是否启用 TF32（仅 CUDA，Ampere 架构及以上）
ENABLE_TF32 = True

# ============================================================================
# 工具函数
# ============================================================================

def get_config_summary():
    """获取配置摘要"""
    summary = f"""
{'='*60}
Grounded-SAM-2 配置摘要
{'='*60}

项目路径:
  - 项目根目录: {PROJECT_ROOT}
  - SAM2 权重: {SAM2_CHECKPOINT}
  - DINO 权重: {GROUNDING_DINO_CHECKPOINT}
  - 输出目录: {DEFAULT_OUTPUT_DIR}

模型参数:
  - Box 阈值: {BOX_THRESHOLD}
  - Text 阈值: {TEXT_THRESHOLD}
  - 计算设备: {DEVICE}

输出设置:
  - 保存 JSON: {DUMP_JSON_RESULTS}
  - 保存掩码: {SAVE_MASK}
  - 保存标注: {SAVE_ANNOTATED}

{'='*60}
"""
    return summary

def print_config():
    """打印配置信息"""
    print(get_config_summary())

def validate_paths():
    """验证关键路径是否存在"""
    errors = []

    # 检查权重文件
    if not os.path.exists(SAM2_CHECKPOINT):
        errors.append(f"SAM2 权重文件不存在: {SAM2_CHECKPOINT}")

    if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
        errors.append(f"Grounding DINO 权重文件不存在: {GROUNDING_DINO_CHECKPOINT}")

    # 检查配置文件
    if not os.path.exists(GROUNDING_DINO_CONFIG):
        errors.append(f"Grounding DINO 配置文件不存在: {GROUNDING_DINO_CONFIG}")

    # 检查 SAM2 配置（相对路径，需要在项目根目录下）
    sam2_config_full = os.path.join(PROJECT_ROOT, SAM2_CONFIG)
    if not os.path.exists(sam2_config_full):
        errors.append(f"SAM2 配置文件不存在: {sam2_config_full}")

    if errors:
        print("⚠️  配置验证失败，发现以下问题：")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ 配置验证通过，所有关键文件都存在")
        return True

# ============================================================================
# 导出配置字典（方便其他模块导入）
# ============================================================================

CONFIG = {
    # 路径
    'project_root': PROJECT_ROOT,
    'sam2_checkpoint': SAM2_CHECKPOINT,
    'sam2_config': SAM2_CONFIG,
    'grounding_dino_checkpoint': GROUNDING_DINO_CHECKPOINT,
    'grounding_dino_config': GROUNDING_DINO_CONFIG,
    'output_dir': DEFAULT_OUTPUT_DIR,
    'grounding_dino_root': GROUNDING_DINO_ROOT,

    # 参数
    'box_threshold': BOX_THRESHOLD,
    'text_threshold': TEXT_THRESHOLD,
    'device': DEVICE,

    # 默认值
    'default_test_image': DEFAULT_TEST_IMAGE,
    'default_text_prompt': DEFAULT_TEXT_PROMPT,

    # 输出设置
    'dump_json': DUMP_JSON_RESULTS,
    'save_mask': SAVE_MASK,
    'save_annotated': SAVE_ANNOTATED,

    # 高级设置
    'use_autocast': USE_AUTOCAST,
    'enable_tf32': ENABLE_TF32,
}

# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print_config()
    print("\n正在验证配置...")
    validate_paths()