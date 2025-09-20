import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ==== 参数设定 ====
folder = "G:\\3\\pre\\1332327"  # 替换为你图像文件夹路径
resize_dim = (200, 200)

def rgb_to_od(rgb):
    rgb = rgb.astype(np.float32)
    rgb[rgb == 0] = 1
    return -np.log(rgb / 255.0)

def od_to_rgb(od):
    return (np.exp(-od) * 255).clip(0, 255).astype(np.uint8)

def extract_snfm_stains(img):
    img = img.resize(resize_dim).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    od = rgb_to_od(pixels)
    nmf = NMF(n_components=2, init='random', random_state=0, max_iter=500)
    W = nmf.fit_transform(od)
    H = nmf.components_
    return normalize(H, axis=1)

def extract_macenko_stains(img, alpha=0.1, thresh=0.8):
    img = img.resize(resize_dim).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    od = rgb_to_od(pixels)
    od = od[np.all(od > 0, axis=1)]  # remove background
    pca = PCA(n_components=2)
    pca.fit(od)
    V = pca.components_
    return normalize(V, axis=1)

# ==== 扫描文件 ====
pairs = []
for fname in os.listdir(folder):
    if fname.endswith('_input.jpg'):
        base = fname.replace('_input.jpg', '')
        in_path = os.path.join(folder, f"{base}_input.jpg")
        out_path = os.path.join(folder, f"{base}_output.jpg")
        if os.path.exists(out_path):
            pairs.append((base, in_path, out_path))

# ==== 比较向量 ====
results = {
    "H": {"SNMF": [], "Output": []},
    "E": {"SNMF": [], "Output": []},
}

for base, in_img_path, out_img_path in pairs:
    in_img = Image.open(in_img_path)
    out_img = Image.open(out_img_path)

    macenko_H, macenko_E = extract_macenko_stains(in_img)
    snmf_H, snmf_E = extract_snfm_stains(in_img)
    out_H, out_E = extract_snfm_stains(out_img)  # 用同样的 SNMF 提取 output 图像

    results["H"]["SNMF"].append(cosine_similarity([macenko_H], [snmf_H])[0, 0])
    results["E"]["SNMF"].append(cosine_similarity([macenko_E], [snmf_E])[0, 0])
    results["H"]["Output"].append(cosine_similarity([macenko_H], [out_H])[0, 0])
    results["E"]["Output"].append(cosine_similarity([macenko_E], [out_E])[0, 0])

# ==== 画图 ====
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, stain in enumerate(["H", "E"]):
    axs[i].boxplot([results[stain]["SNMF"], results[stain]["Output"]],
                   labels=["SNMF (Input)", "SNMF (Output)"])
    axs[i].set_title(f"{stain} correlation to Macenko")
    axs[i].set_ylabel("Cosine similarity")

plt.suptitle("Stain Vector Comparison (vs. Macenko)")
plt.tight_layout()
plt.show()
