'''
Author: ZhangLan
Date: 2025-05-09 08:42:01
LastEditors: ZhangLan
LastEditTime: 2025-05-24 19:29:57
Description: file content
'''
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import openslide

# 设置路径
source_dir = "H:/svs-data/data-1-914"
output_dir = "G:/patches/"
# target_image_path = "3-10_0.303.png"  # 一张参考标准图像

os.makedirs(output_dir, exist_ok=True)

# 判断背景和遮挡的函数
def check_background_and_occlusion(img, background_threshold=0.4, occlusion_threshold=0.4):
    # 转换为灰度图像并进行二值化
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)

    # 计算背景占比：背景为白色区域
    background_area = np.sum(binary_img == 255)  # 白色背景
    total_area = binary_img.size
    background_ratio = background_area / total_area

    # 计算遮挡部分：非白色区域
    non_background_area = total_area - background_area
    occlusion_ratio = non_background_area / total_area

    # 判断是否丢弃该图像
    if background_ratio > background_threshold or occlusion_ratio < occlusion_threshold:
        return False  # 丢弃图像
    return True  # 保留图像

# 图像处理函数
def normalize_and_resize(img_path, output_dir, size=(512, 512)):
    try:
        slide = openslide.OpenSlide(img_path)

        # 获取各层级图像信息
        # print("Level dimensions:", slide.level_dimensions)
        # print("Level downsamples:", slide.level_downsamples)

        # 获取文件名并使用文件名作为文件夹名称
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        output_folder = os.path.join(output_dir, base_filename)
        os.makedirs(output_folder, exist_ok=True)

        # 对于每一层次，进行裁剪并保存图像
        # for level in range(slide.level_count):
        level = 0
        level_width, level_height = slide.level_dimensions[level]
        tile_size = 512  # 每个小图的尺寸

        import torch

        for x in range(0, level_width, tile_size):
            for y in range(0, level_height, tile_size):
                # 读取指定位置的图像区域
                img = slide.read_region((x, y), level, (tile_size, tile_size))
                img = img.convert("RGB")  # 确保是 RGB 图像

                # 转换为 numpy 数组
                img_array = np.array(img)

                # 判断图像是否符合要求
                if not check_background_and_occlusion(img_array):
                    continue  # 丢弃该图像

                # 转换为 PyTorch tensor，并调整维度 (H, W, C) -> (C, H, W)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # 可加归一化如 / 255.0

                # 重设大小（可选，推荐先 resize 再转 tensor）
                img_resized = cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()

                # 保存为 .pt 文件
                output_tensor_path = os.path.join(output_folder, f"{level}_x{x}_y{y}.pt")
                torch.save(img_tensor, output_tensor_path)


                    # print(f"Processed {img_path} level {level} at x={x}, y={y} and saved to {output_img_path}")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 批量处理所有图片
for filename in tqdm(os.listdir(source_dir)):
    if filename.lower().endswith(('.svs', '.jpg', '.jpeg', '.tif')):
        src_path = os.path.join(source_dir, filename)
        normalize_and_resize(src_path, output_dir)
