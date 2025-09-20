'''
Author: ZhangLan
Date: 2024-03-14 11:13:50
LastEditors: ZhangLan
LastEditTime: 2024-03-14 11:26:15
Description: file content
'''
import os
import numpy as np
import imageio
# 指定文件夹路径
folder_path = 'D:/data/ktest/mask/'
output_path = 'D:/data/ktest/datas/'

# 获取文件夹下所有文件名
file_list = os.listdir(folder_path)


def change (npy_file,out_path):
    

    # 读取npy文件
    # npy_file = 'path/to/your/npy_file.npy'
    npy_data = np.load(npy_file)
    has_non_zero_element = np.any(npy_data)
    print(has_non_zero_element)
    print(npy_data.shape)
    
    # npy_data = npy_data[:,:,2]
    print(npy_data.shape)

    # 将npy数据转换为0-255范围
    npy_data = np.clip(npy_data, 0, 255).astype(np.uint8)
    
    

    # 将npy数据转换为png文件
    png_file = out_path
    imageio.imwrite(png_file, npy_data)


# 遍历文件夹下所有文件
for file_name in file_list:
    # 获取文件绝对路径
    file_path = os.path.join(folder_path, file_name)
    new_name = file_name.split('.')[0] + '.jpg'
    print(new_name)
    out_path = os.path.join(output_path, new_name)
    print (file_name)
    print(out_path)
    change(file_path,out_path)

    # 调用文件
    # print(f'Calling {file_path}')
    # # 这里可以添加您想要执行的操作，例如读取文件内容或执行其他操作

    # # 添加一个打印语句
    # print(f'{file_path} called')
