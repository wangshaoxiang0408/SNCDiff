import numpy as np
from skimage import io
from pathlib import Path
from sklearn.decomposition import NMF
import os

def rgb2od(I):
    I = I.astype(np.float32)
    I = np.clip(I, 1, 255)  # 避免 log(0) 和太亮
    return -np.log(I / 255)

def od2rgb(OD):
    OD = np.clip(OD, 0, np.inf)
    return np.clip(255 * np.exp(-OD), 0, 255).astype(np.uint8)

def get_stain_matrix_nmf(I, n_components=2):
    OD = rgb2od(I).reshape((-1, 3))
    # 阈值过滤，去除过亮区域
    OD = OD[~np.any(OD < 0.15, axis=1)]
    if OD.shape[0] == 0:
        print("Warning: OD is empty after thresholding.")
        return None

    model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
    W = model.fit_transform(OD)
    H = model.components_
    # 返回颜色基矩阵，shape (3, n_components)
    return H.T

def get_concentrations(I, stain_matrix):
    if stain_matrix is None:
        return None
    OD = rgb2od(I).reshape((-1, 3))
    if OD.shape[0] == 0:
        print("Warning: OD empty in get_concentrations.")
        return None
    # 用非负最小二乘拟合浓度矩阵
    concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)
    return concentrations

def normalize(I, target, maxC=1.5):
    stain_matrix_src = get_stain_matrix_nmf(I)
    stain_matrix_tar = get_stain_matrix_nmf(target)

    if stain_matrix_src is None or stain_matrix_tar is None:
        print("Warning: stain matrix is None, skipping normalization.")
        return I

    C = get_concentrations(I, stain_matrix_src)
    if C is None:
        print("Warning: concentrations are None, skipping normalization.")
        return I

    maxC_src = np.percentile(C, 99, axis=1)
    maxC_src[maxC_src == 0] = 1  # 防止除零

    C_normalized = C * (maxC / maxC_src[:, None])
    OD_norm = np.dot(stain_matrix_tar, C_normalized).T
    I_norm = od2rgb(OD_norm.reshape(I.shape))
    return I_norm

def batch_vahadane(input_dir, output_dir, reference_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = io.imread(reference_path)
    if target.dtype != np.uint8:
        target = (target * 255).astype(np.uint8)

    for path in input_dir.glob("**/*"):
        if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        try:
            print(f"Processing {path.name}")
            img = io.imread(path)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            norm_img = normalize(img, target)
            save_path = output_dir / path.relative_to(input_dir)
            save_path = save_path.with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(save_path, norm_img)
        except Exception as e:
            print(f"Failed on {path.name}: {e}")

if __name__ == "__main__":
    batch_vahadane(
        input_dir="G:/3/reinhard/yuan",            # 输入文件夹
        output_dir="G:/3/reinhard/normalized",     # 输出文件夹
        reference_path="F:/ranse/normalpicture.png" # 参考图像路径
    )
