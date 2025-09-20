'''
Author: ZhangLan
Date: 2025-06-10 15:40:05
LastEditors: ZhangLan
LastEditTime: 2025-06-10 17:33:48
Description: file content
'''
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calculate_rsnr(ref_img, noisy_img):
    ref_img = ref_img.astype(np.float64)
    noisy_img = noisy_img.astype(np.float64)
    noise = ref_img - noisy_img
    signal_energy = np.sum(ref_img ** 2)
    noise_energy = np.sum(noise ** 2)
    if noise_energy == 0:
        return float('inf')
    rsnr = 10 * np.log10(signal_energy / noise_energy + 1e-10)
    return rsnr
def calculate_ssim(img1, img2, data_range=255):
    return ssim(img1, img2, data_range=data_range)

def pearson_correlation(img1, img2):
    # 将图像转为numpy数组并展平
    img1 = np.array(img1).flatten().astype(float)
    img2 = np.array(img2).flatten().astype(float)
    
    # 计算均值
    mean1, mean2 = np.mean(img1), np.mean(img2)
    
    # 计算协方差和标准差
    covariance = np.sum((img1 - mean1) * (img2 - mean2))
    std1 = np.sqrt(np.sum((img1 - mean1) ** 2))
    std2 = np.sqrt(np.sum((img2 - mean2) ** 2))
    
    # 避免除零
    if std1 * std2 == 0:
        return 0
    return covariance / (std1 * std2)

def compare_folders(folder_ref, folder_noisy):
    # 获取两个文件夹的文件列表
    files_ref = sorted(os.listdir(folder_ref))
    files_noisy = sorted(os.listdir(folder_noisy))
    
    # 确保文件名一致（简单匹配，按文件名排序）
    common_files = files_ref
    if not common_files:
        print("错误：两个文件夹没有相同文件名的图片！")
        return
    
    rsnr_results = {}
    ssim_results = {}
    pc_results = {}
    
    for filename in common_files:
        # 读取参考图像和噪声图像
        img_ref = np.array(Image.open(os.path.join(folder_ref, filename)).convert('L'))  # 转为灰度图
        filename_no_ext = os.path.splitext(filename)[0]  # 去掉后缀，得到 "example"
        new_filename = f"{filename_no_ext}.jpg"  # 改为 "example.jpg"
        img_noisy = np.array(Image.open(os.path.join(folder_noisy, new_filename)).convert('L'))
        ref_img_normalized = img_ref / 255.0
        noisy_img_normalized = img_noisy / 255.0
        rsnr = calculate_rsnr(ref_img_normalized, noisy_img_normalized)
        # 计算 RSNR
        # rsnr = calculate_rsnr(img_ref, img_noisy)
        rsnr_results[filename] = rsnr
        print(f"{filename}: RSNR = {rsnr:.2f} dB")
        ssim_val = calculate_ssim(ref_img_normalized, noisy_img_normalized)
        ssim_results[filename] = ssim_val
        print(f"{filename}: SSIM = {ssim_val:.4f}")
        
        pc = pearson_correlation(ref_img_normalized, noisy_img_normalized)
        pc_results[filename] = pc
        print(f"{filename}: SSIM = {pc:.4f}")
        
    
    # 计算统计信息
    rsnr_values = list(rsnr_results.values())
    avg_rsnr = np.mean(rsnr_values)
    min_rsnr = min(rsnr_values)
    max_rsnr = max(rsnr_values)
    
    print("\n--- 统计结果 ---")
    print(f"平均 RSNR: {avg_rsnr:.2f} dB")
    print(f"最低 RSNR: {min_rsnr:.2f} dB (最差质量)")
    print(f"最高 RSNR: {max_rsnr:.2f} dB (最好质量)")
    avg_ssim = np.mean(list(ssim_results.values()))
    print(f"\n平均 SSIM: {avg_ssim:.4f}")
    avg_pc = np.mean(list(pc_results.values()))
    print(f"\n平均 pc: {avg_pc:.4f}")

# 示例：对比两个文件夹
folder_ref = "G:\\3\\reinhard\\yuan"    # 参考图像（干净图像）文件夹
folder_noisy = "G:\\3\\reinhard\\normalized"      # 噪声/失真图像文件夹
compare_folders(folder_ref, folder_noisy)