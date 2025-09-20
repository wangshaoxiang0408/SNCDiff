import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_image_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_dominant_colors(img, num_colors=10, sample_size=10000):
    pixels = img.reshape(-1, 3)
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_ / 255.0
    counts = np.bincount(kmeans.labels_)
    counts = counts / np.sum(counts)
    return colors, counts

def plot_color_sphere(ax, color, radius, resolution=20):
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = radius * np.cos(u) * np.sin(v) + color[0]
    y = radius * np.sin(u) * np.sin(v) + color[1]
    z = radius * np.cos(v) + color[2]
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color=color,
                    linewidth=0, antialiased=False, shade=True, alpha=0.95)

def beautify_3d_axis(ax):
    ax.set_xlim(1, 0)
    ax.set_ylim(1, 0)
    ax.set_zlim(0, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel('G', labelpad=8)
    ax.set_ylabel('R', labelpad=8)
    ax.set_zlabel('B', labelpad=8)
    ax.view_init(elev=30, azim=135)
    ax.grid(False)
    ax.set_facecolor('white')

def plot_3d_color_balls(image_paths, titles, num_colors=20):
    fig = plt.figure(figsize=(22, 4), facecolor='white')

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        img = read_image_rgb(path)
        colors, counts = extract_dominant_colors(img, num_colors)

        ax = fig.add_subplot(1, len(image_paths), i + 1, projection='3d')
        beautify_3d_axis(ax)
        ax.set_title(title, fontsize=12)

        # 调整球大小（比例缩放）
        for color, count in zip(colors, counts):
            plot_color_sphere(ax, color=color, radius=0.25 * count**0.5)

    plt.tight_layout()
    plt.show()

# 示例图像路径和标题
image_paths = [
    "F:/ranse/mutil-picture/yuan.png",
    "F:/ranse/mutil-picture/nor.png",
    "F:/ranse/mutil-picture/1.png",
    "F:/ranse/mutil-picture/2.png",
    "F:/ranse/mutil-picture/3.png",
    "F:/ranse/mutil-picture/4.png"
]
titles = [
    "(a) Source",
    "(b) Target",
    "(c) SSCD",
    "(d) hist_match",
    "(e) rehind",
    "(f) Mack"
]

plot_3d_color_balls(image_paths, titles, num_colors=20)
