'''
Author: ZhangLan
Date: 2025-05-19 15:52:45
LastEditors: ZhangLan
LastEditTime: 2025-05-19 16:26:40
Description: file content
'''
import numpy as np
from skimage import io
from pathlib import Path
from scipy.linalg import svd
import os

def rgb2od(I):
    I = I.astype(np.float32)
    I = np.clip(I, 1, 255)  # 避免 log(0) 和太亮
    return -np.log(I / 255)

def od2rgb(OD):
    OD = np.clip(OD, 0, np.inf)
    return np.clip(255 * np.exp(-OD), 0, 255).astype(np.uint8)


def get_stain_matrix(I, thresh=0.8, angular_percentile=99):
    OD = rgb2od(I).reshape((-1, 3))
    OD = OD[~np.any(OD < thresh, axis=1)]

    _, _, V = svd(OD, full_matrices=False)
    V = V[:2, :].T

    projections = np.dot(OD, V)
    angles = np.arctan2(projections[:, 1], projections[:, 0])
    min_angle = np.percentile(angles, 100 - angular_percentile)
    max_angle = np.percentile(angles, angular_percentile)

    v1 = np.dot(V, [np.cos(min_angle), np.sin(min_angle)])
    v2 = np.dot(V, [np.cos(max_angle), np.sin(max_angle)])

    return np.array([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)]).T  # shape (3,2)

def get_concentrations(I, stain_matrix):
    OD = rgb2od(I).reshape((-1, 3))
    return np.linalg.lstsq(stain_matrix, OD.T, rcond=None)[0]  # shape (2, N)

def normalize(I, target):
    stain_matrix_src = get_stain_matrix(I)
    stain_matrix_tar = get_stain_matrix(target)

    C_src = get_concentrations(I, stain_matrix_src)
    C_tar = get_concentrations(target, stain_matrix_tar)

    # 取目标图浓度的 99 分位数作为缩放标准
    maxC_tar = np.percentile(C_tar, 99, axis=1)
    maxC_src = np.percentile(C_src, 99, axis=1)

    # 保持染色浓度的分布一致性
    scaling = maxC_tar / (maxC_src + 1e-3)
    C_norm = C_src * scaling[:, None]

    OD_norm = np.dot(stain_matrix_tar, C_norm).T
    OD_norm = np.clip(OD_norm.reshape(I.shape), 0.2, 5)

    return od2rgb(OD_norm)


# ------------------------------
# 批量处理
# ------------------------------
def batch_macenko(input_dir, output_dir, reference_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载参考图像
    target = io.imread(reference_path)
    for path in input_dir.glob("**/*"):
        if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        try:
            print(f"Processing {path.name}")
            img = io.imread(path)
            norm_img = normalize(img, target)
            save_path = output_dir / path.relative_to(input_dir)
            save_path = save_path.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(save_path, norm_img)
        except Exception as e:
            print(f"Failed on {path.name}: {e}")

# 示例调用
if __name__ == "__main__":
    batch_macenko(
        input_dir="G:/3/reinhard/yuan",      # 原图像文件夹路径
        output_dir="G:/3/reinhard/normalized",       # 输出文件夹路径
        reference_path="F:/ranse/normalpicture.png"   # 目标参考图像路径
    )
