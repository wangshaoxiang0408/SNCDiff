'''
Author: ZhangLan
Date: 2025-05-19 15:43:42
LastEditors: ZhangLan
LastEditTime: 2025-05-19 15:49:04
Description: file content
'''
import os
from pathlib import Path
from skimage import io, color
import numpy as np

def get_lab_mean_std(image_lab):
    mean = image_lab.mean(axis=(0, 1))
    std = image_lab.std(axis=(0, 1))
    return mean, std

def reinhard_normalization(source, target):
    source_lab = color.rgb2lab(source)
    target_lab = color.rgb2lab(target)
    source_mean, source_std = get_lab_mean_std(source_lab)
    target_mean, target_std = get_lab_mean_std(target_lab)
    normalized_lab = (source_lab - source_mean) / source_std
    normalized_lab = normalized_lab * target_std + target_mean
    normalized_rgb = np.clip(color.lab2rgb(normalized_lab), 0, 1)
    return (normalized_rgb * 255).astype(np.uint8)

def batch_process_images(input_dir, output_dir, reference_image_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载参考图像
    target = io.imread(reference_image_path)
    if target.max() > 1:
        target = target / 255.0  # 归一化

    # 遍历输入目录
    for img_path in input_dir.glob("**/*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        print(f"Processing {img_path}")
        src = io.imread(img_path)
        if src.max() > 1:
            src = src / 255.0  # 归一化

        try:
            normalized = reinhard_normalization(src, target)
            relative_path = img_path.relative_to(input_dir)
            save_path = output_dir / relative_path.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(save_path, normalized)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# 示例调用
if __name__ == "__main__":
    batch_process_images(
        input_dir="G:/3/reinhard/yuan",      # 原图像文件夹路径
        output_dir="G:/3/reinhard/normalized",       # 输出文件夹路径
        reference_image_path="F:/ranse/normalpicture.png"   # 目标参考图像路径
    )
