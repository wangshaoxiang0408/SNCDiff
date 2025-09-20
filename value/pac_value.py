import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_image_pixels(path, sample_per_image=1000):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to read {path}")
        return np.empty((0, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3)
    if len(pixels) > sample_per_image:
        idx = np.random.choice(len(pixels), sample_per_image, replace=False)
        pixels = pixels[idx]
    return pixels

def read_all_images_from_folder(folder, sample_per_image=1000):
    all_pixels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            path = os.path.join(folder, filename)
            pixels = read_image_pixels(path, sample_per_image)
            if pixels.size > 0:
                all_pixels.append(pixels)
    if all_pixels:
        return np.vstack(all_pixels)
    else:
        return np.empty((0, 3))

def reduce_dimensionality(pixels, method='pca', n_components=3):
    if len(pixels) == 0:
        return np.empty((0, n_components))
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(pixels)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000, random_state=42)
        reduced = reducer.fit_transform(pixels)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    return reduced

def plot_boxplot_first_dimension(data_dict, method='pca', colors=None, median_color='red', median_linewidth=2):
    """
    只画第一个维度的箱线图，支持指定颜色和中位线样式
    data_dict: dict of {label: pixels}
    colors: dict of {label: color_code}
    median_color: 中位线颜色
    median_linewidth: 中位线宽度
    """
    reduced_data = {}
    for label, pixels in data_dict.items():
        reduced_data[label] = reduce_dimensionality(pixels, method=method)
    
    labels = list(reduced_data.keys())
    data_to_plot = [reduced_data[label][:, 0] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
    
    if colors:
        for patch, label in zip(box['boxes'], labels):
            patch.set_facecolor(colors.get(label, 'lightgray'))

    # 设置中位线颜色和粗细
    for median in box['medians']:
        median.set(color=median_color, linewidth=median_linewidth)

    ax.set_title(f"{method.upper()} Boxplot")
    ax.set_ylabel("Value")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return reduced_data


def calc_distances_to_target(reduced_data, target_label='Target'):
    """
    计算每个组和 Target 的欧氏距离和 MSE
    返回 dict: {label: {'euclidean': val, 'mse': val}}
    """
    distances = {}
    target_data = reduced_data.get(target_label)
    if target_data is None or len(target_data) == 0:
        print("No valid target data to compare.")
        return distances
    
    target_mean = np.mean(target_data, axis=0)
    
    for label, data in reduced_data.items():
        if label == target_label:
            continue
        if len(data) == 0:
            distances[label] = {'euclidean': np.nan, 'mse': np.nan}
            continue
        mean_vec = np.mean(data, axis=0)
        euclidean = np.linalg.norm(mean_vec - target_mean)
        mse = np.mean((mean_vec - target_mean)**2)
        distances[label] = {'euclidean': euclidean, 'mse': mse}
    return distances


if __name__ == "__main__":
    folder_paths = {
        'Source': 'G:\\3\\reinhard\\yuan',
        'Reinhard': 'G:\\3\\reinhard\\reinhard',
        # 'Macenko': 'G:\\3\\reinhard\\macenko',
        'Vahadane': 'G:\\3\\reinhard\\hist_match',
        'CycleGAN': 'G:\\3\\reinhard\\normalized',
        'SSCD': 'G:\\3\\reinhard\\sscd'
    }
    target_path = 'normalpicture.png'

    # 读取数据
    data = {}
    for label, folder in folder_paths.items():
        data[label] = read_all_images_from_folder(folder, sample_per_image=1000)
    data['Target'] = read_image_pixels(target_path, sample_per_image=2000)

    # 定义颜色字典
    colors = {
        'Source': '#1f77b4',      # 蓝色
        'Reinhard': '#2ca02c',    # 橙色
        # 'Macenko': '#2ca02c',     # 绿色
        'Vahadane': '#ff7f0e',    # 红色
        'CycleGAN': '#9467bd',    # 紫色
        'SSCD': '#8c564b',        # 棕色
        'Target': '#e377c2'       # 粉色
    }

    # PCA降维 + 只画第一个维度箱线图 + 计算距离
    print("PCA Dimension 1 Boxplot:")
    reduced_pca = plot_boxplot_first_dimension(data, method='pca', colors=colors, median_color='red', median_linewidth=2)
    distances_pca = calc_distances_to_target(reduced_pca, target_label='Target')
    print("PCA欧氏距离和MSE：")
    for label, dist in distances_pca.items():
        print(f"{label}: Euclidean={dist['euclidean']:.4f}, MSE={dist['mse']:.4f}")

    # t-SNE降维 + 只画第一个维度箱线图 + 计算距离
    print("\nt-SNE Dimension 1 Boxplot:")
    # reduced_tsne = plot_boxplot_first_dimension(data, method='tsne', colors=colors)
    # distances_tsne = calc_distances_to_target(reduced_tsne, target_label='Target')
    # print("t-SNE欧氏距离和MSE：")
    # for label, dist in distances_tsne.items():
    #     print(f"{label}: Euclidean={dist['euclidean']:.4f}, MSE={dist['mse']:.4f}")
