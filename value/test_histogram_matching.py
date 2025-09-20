'''
Author: ZhangLan
Date: 2025-05-19 16:37:40
LastEditors: ZhangLan
LastEditTime: 2025-05-19 16:42:05
Description: file content
'''
import numpy as np
from skimage import io, exposure
from pathlib import Path

def histogram_matching(source, reference):
    matched = np.zeros_like(source)
    for i in range(source.shape[2]):  # 遍历每个通道
        matched[:, :, i] = exposure.match_histograms(source[:, :, i], reference[:, :, i])
    return matched.astype(np.uint8)


def batch_histogram_matching(input_dir, output_dir, reference_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = io.imread(reference_path)
    if reference.dtype != np.uint8:
        reference = (reference * 255).astype(np.uint8)

    for img_path in input_dir.glob("**/*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        try:
            print(f"Processing {img_path.name}")
            img = io.imread(img_path)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            matched_img = histogram_matching(img, reference)

            save_path = output_dir / img_path.relative_to(input_dir)
            save_path = save_path.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            io.imsave(save_path, matched_img)
        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")

if __name__ == "__main__":
    batch_histogram_matching(
        input_dir="G:/3/reinhard/yuan",           # 源图像文件夹
        output_dir="G:/3/reinhard/hist_match",    # 输出文件夹
        reference_path="F:/ranse/normalpicture.png"  # 参考图像路径
    )
