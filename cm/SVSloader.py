'''
Author: ZhangLan
Date: 2025-05-09 08:24:26
LastEditors: ZhangLan
LastEditTime: 2025-05-16 09:52:26
Description: file content
'''
import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
import cv2 



class SVSDataset(torch.utils.data.Dataset):

    def __init__(self, image_paths, transform, df=None, mode='train'):
        self.image_paths = image_paths
        self.transform = transform
        self.df = df#.copy()
        # self.df['ID'] = self.df['ID'].astype(str)
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pt_path = self.image_paths[idx]
        image = torch.load(pt_path)

        # 预处理
        if image.max() > 1:
            image = image.float() / 255.0
        if image.dim() == 3 and image.size(0) != 3:
            image = image.permute(2, 0, 1)
        # image = image.unsqueeze(0)

        # 双视图增强
        # x = self.transform(image)
       
        if self.mode == 'train':
            return image, pt_path
        else:
            # 提取 ID
            
            return image, pt_path