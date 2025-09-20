import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def rgb2od(I):
    I = I.astype(np.float32)
    I[I == 0] = 1  # 防止 log(0)
    return -np.log((I + 1e-8) / 255.0)

def compute_rRMSE(original, reconstructed, eps=1e-8):
    assert original.shape == reconstructed.shape, "图像尺寸不一致"
    mse = np.mean((original - reconstructed) ** 2)
    denom = np.mean(original ** 2)
    rrmse = np.sqrt(mse / (denom + eps))
    return rrmse

def compare_outputs_to_original(folder_path, original_name='original.jpg'):
    # 加载原图并转换为 OD
    original_path = os.path.join(folder_path, original_name)
    original_img = np.array(Image.open(original_path).convert('RGB'))
    original_od = rgb2od(original_img)

    rrmse_results = []

    # 遍历所有 _output.jpg 文件
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('_output.jpg'):
            output_path = os.path.join(folder_path, fname)
            output_img = np.array(Image.open(output_path).convert('RGB'))
            output_od = rgb2od(output_img)

            if original_od.shape != output_od.shape:
                print(f"跳过：{fname}（尺寸不匹配）")
                continue

            rrmse = compute_rRMSE(original_od, output_od)
            rrmse_results.append((fname, rrmse))
            print(f"{fname}: rRMSE = {rrmse:.6f}")

    if rrmse_results:
        avg_rrmse = np.mean([x[1] for x in rrmse_results])
        print(f"\n平均 rRMSE（共 {len(rrmse_results)} 张图像）: {avg_rrmse:.6f}")
    else:
        print("未找到任何有效的 _output.jpg 文件")

    return rrmse_results


if __name__ == "__main__":
    folder = 'G:\\3\\pre\\1332327'  # 包含 original.jpg 和多个 *_output.jpg 的文件夹
    compare_outputs_to_original(folder, original_name='original.jpg')
