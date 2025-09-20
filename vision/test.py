'''
Author: ZhangLan
Date: 2024-03-13 13:06:06
LastEditors: ZhangLan
LastEditTime: 2024-03-14 11:13:19
Description: file content
'''
#可视化傅里叶滤波的效果

import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



from scipy.ndimage import gaussian_filter
def remove_noise(data, kernel_size=3, sigma=0.1):
  # 使用高斯滤波器去除泊松噪声
  data_noise_free = gaussian_filter(data, sigma=sigma)
  return data_noise_free

def save_as_png(data, output_dir,a):
    # picture =[]
    # 遍历数据并保存每个切片为 PNG 文件
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # 将数据调整成 PIL Image 能够接受的格式
        slice_data_scaled = np.uint8(slice_data / (slice_data.max()+0.0000001) * 255)
        # slice_data_scaled = np.uint8(slice_data / slice_data.max() * 255)
        # 将数据转换为 PIL Image 对象
        slice_image = Image.fromarray(slice_data_scaled)
        # slice_image = Image.fromarray(np.uint8(color_map[slice_data_scaled]))
    
        # 保存图像为 PNG 文件
        slice_image.save(f'{output_dir}/{a}_slice_{i}.png')

def low_pass_filter(image, cutoff):
    # 计算图像的傅里叶变换
    freq_image = np.fft.fft2(image)
    
    # 计算图像的空间谱函数
    space_spectrum = np.abs(freq_image)
    
    # 对空间谱函数进行低通滤波
    filtered_space_spectrum = space_spectrum.copy()
    filtered_space_spectrum[cutoff:] = 0
    
    # 计算低通滤波后的图像
    filtered_image = np.fft.ifft2(filtered_space_spectrum)
    
    return filtered_image.real



def change(data):
    # 读取 .nii.gz 文件
    path = f'D:/data/Brast/HGG/c/c/{data}'
    image = nib.load(path+'.nii.gz')
    # 获取图像数据
    data = image.get_fdata()
    data = np.array(data, dtype=np.float64)
    # 获取图像头信息
    header = image.header

    # 应用低通滤波器
    cutoff = 100  # 设置滤波器的截止频率
    filtered_data = remove_noise(data)
    # filtered_data = fourier_filter(data, low_cutoff=0.001, high_cutoff=300.0)
    filtered_data = np.array(filtered_data, dtype=np.float64)
    # 打印滤波后的数据
    print("Filtered data:")
    print(filtered_data.shape)
    # path_out = os.path.join('D:/data/Brast/HGG/c/c/t1')
    save_as_png(data, path,'y')
    save_as_png(filtered_data, path,'f')
change('t2')


