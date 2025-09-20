import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF
from mpl_toolkits.mplot3d import Axes3D

def rgb_to_od(rgb):
    rgb = rgb.astype(np.float32)
    rgb[rgb == 0] = 1
    return -np.log(rgb / 255.0)

def extract_stain_vectors(img, resize=(200, 200)):
    img = img.resize(resize)
    pixels = np.array(img).reshape(-1, 3)
    od_pixels = rgb_to_od(pixels)
    nmf = NMF(n_components=2, init='random', random_state=42, max_iter=500)
    W = nmf.fit_transform(od_pixels)
    H = nmf.components_
    return pixels, H

def draw_vector(ax, v, label, color):
    origin = np.zeros(3)
    ax.quiver(*origin, *(v * 255), color=color, linewidth=2, arrow_length_ratio=0.05)
    ax.text(*(v * 255 * 1.1), label, color=color, fontsize=10)

def plot_multiple_images(image_paths, labels=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        pixels, H = extract_stain_vectors(img)
        color = colors[i % len(colors)]

        # RGB 点云
        ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
    c='none', edgecolors=pixels / 255.0,  # 使用边缘色替代填充色
    alpha=0.3, s=3, marker='o', label=f"{labels[i] if labels else f'Image {i+1}'}")

        # stain 向量
        H1 = H[0] / np.linalg.norm(H[0])
        H2 = H[1] / np.linalg.norm(H[1])
        draw_vector(ax, H1, f"Eff-H {i+1}", color)
        draw_vector(ax, H2, f"Eff-E {i+1}", color)

    # 设置坐标轴
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title("Multi-image Stain Vectors and RGB Point Cloud")

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ========= 用法示例 =========
image_paths = [
    "G:/3/pre/1332327/0_x6144_y22016_input.jpg",
    "G:/3/pre/1332327/0_x6144_y22016_output.jpg",
    "F:/ranse/normalpicture.png"
]
labels = ["Original image", "Stained images standardized by SSCD", "Standardized staining images"]

plot_multiple_images(image_paths, labels)
