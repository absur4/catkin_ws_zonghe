# 文件名: mask_utils.py

import numpy as np
from PIL import Image
from typing import Union

def save_numpy_mask_as_1bit(mask_array: np.ndarray, output_path: str) -> bool:
    """
    将一个 NumPy 数组转换为 1-bit 的 PIL 图像并保存。

    这个函数接收一个通常形状为 (1, H, W) 或 (H, W) 的 NumPy 掩码数组，
    将其转换为纯黑白（1-bit）的图像，并保存到指定路径。

    参数:
        mask_array (np.ndarray): 输入的 NumPy 掩码数组。
                                 数组的值应为 0 和 1。
        output_path (str): 图像的输出保存路径，例如 'path/to/mask.png'。

    返回:
        bool: 如果成功保存则返回 True，否则返回 False。
    """
    try:
        # --- 数据预处理 ---
        # 1. 检查输入是否为 NumPy 数组
        if not isinstance(mask_array, np.ndarray):
            raise TypeError("输入必须是一个 NumPy 数组。")

        # 2. 降维：将 (1, H, W) 转换为 (H, W)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1:
            processed_mask = np.squeeze(mask_array, axis=0)
        elif mask_array.ndim == 2:
            processed_mask = mask_array
        else:
            raise ValueError(f"不支持的数组形状: {mask_array.shape}。只接受 (H, W) 或 (1, H, W)。")

        # 3. 将值从 0/1 映射到 0/255，并确保数据类型为 uint8
        mask_for_pil = (processed_mask * 255).astype(np.uint8)

        # --- 图像转换与保存 ---
        # 1. 从 NumPy 数组创建 PIL 灰度图像 ('L' 模式)
        pil_image_gray = Image.fromarray(mask_for_pil)

        # 2. 将灰度图像转换为 1-bit 模式 ('1' 模式)
        pil_image_1bit = pil_image_gray.convert("1")

        # 3. 保存图像
        pil_image_1bit.save(output_path)
        
        print(f"成功生成 1-bit 掩码图，并保存到: {output_path}")
        return True

    except (ValueError, TypeError, IOError) as e:
        print(f"错误: 处理或保存掩码时发生问题: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

# --- 测试代码块 ---
# 这部分代码只有在直接运行 `python mask_utils.py` 时才会执行
# 当其他文件 `import mask_utils` 时，这部分不会执行
if __name__ == '__main__':
    print("--- 正在测试 mask_utils.py ---")
    
    # 1. 创建一个 (1, 720, 1280) 的测试掩码
    print("\n测试案例 1: 形状为 (1, 720, 1280)")
    test_mask_3d = np.zeros((1, 720, 1280), dtype=np.uint8)
    test_mask_3d[0, 200:500, 400:900] = 1  # 创建一个白色矩形区域
    save_numpy_mask_as_1bit(test_mask_3d, 'test_mask_3d.png')

    # 2. 创建一个 (480, 640) 的测试掩码
    print("\n测试案例 2: 形状为 (480, 640)")
    test_mask_2d = np.zeros((480, 640), dtype=np.uint8)
    test_mask_2d[100:200, 150:350] = 1 # 创建一个白色矩形区域
    save_numpy_mask_as_1bit(test_mask_2d, 'test_mask_2d.png')

    print("\n--- 测试完成 ---")