"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append("../")  #给系统添加路径
sys.path.append("./")
import torchvision.utils as vutils
import numpy as np
import torch as th
import torch.distributed as dist

from cm.utils import staple

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
import torchvision.transforms as transforms
from cm.SVSloader import SVSDataset
from skimage import color
from PIL import Image
from skimage import color
# import matplotlib.pyplot as plt 
# from pathlib import Path



def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img



def get_all_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".pt", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    return image_paths



import numpy as np





def main():
    args = create_argparser().parse_args()
    
    # args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    # logger.configure()
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")
    if args.data_name == 'SVS':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = SVSDataset(get_all_image_paths(args.data_dir), transform_train, mode='train')
        args.in_ch = 3
    
    else:
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ", args.data_dir)
        # ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    data_list = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True,drop_last = True)
    data = iter(data_list)

    # dist_util.setup_dist()
    # logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
        
    if args.multi_gpu:
        model = th.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    model.eval()
    dice = []
    iou = []
    cal_iou = []
    cal_dice = []
    
    for _ in range(len(data)):
        try:
                b, path = next(data)
        except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                # data_iter = iter(self.dataloader)
                b, path = next(data)                    
       
        # c = th.randn_like(b)   #定义了一个和数据一样大的
        # img = th.cat((b, c), dim=1)     #add a noise channel$
        img = b
        if args.data_name == 'SVS':
            new_path = []
            for i in range(len(path)):
                slice_fd=path[i].split("\\")[-2]
                slice_ID=path[i].split("\\")[-1].split('.')[0]
                new_path.append(os.path.join(args.out_dir, slice_fd, slice_ID))
       
        
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        # slice_ID = 'ce'
        logger.log("sampling...")
        if args.sampler == "multistep":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        masklist = []
        callist = []
        yu = []
        # dice_value = 0
        # iou_value = 0

        # all_images = []
        # all_labels = []
        generator = get_generator(args.generator, args.num_samples, args.seed)
        
        for i in range(args.num_ensemble): 
            

    # while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            output, sample, cal = karras_sample(
                diffusion,
                model,
                (args.batch_size, 3, args.image_size, args.image_size),img = img,
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
                dyed=load_png_with_shifted_crops(args.dyed, crop_size=args.crop_size, batch=args.batch_size, shift_step=20),
                name_path = new_path
            )
           

    dist.barrier()
    logger.log("sampling complete")


def load_png_with_shifted_crops(image_path, crop_size=512, batch=8, shift_step=20):
    """
    加载 PNG 图像，从中心点附近滑动裁剪 batch 个 patch，每个 patch 大小为 crop_size，
    返回 shape 为 [batch, 3, crop_size, crop_size] 的张量。
    
    参数:
        image_path (str): 图像路径
        crop_size (int): 裁剪 patch 大小（正方形）
        A (int): 要生成的 patch 数量
        shift_step (int): 每次移动中心点的步长（像素）

    返回:
        torch.Tensor: 大小为 [A, 3, crop_size, crop_size] 的 float32 张量
    """
    image = Image.open(image_path).convert('RGB')
    W, H = image.size

    cx, cy = W // 2, H // 2  # 原始中心点
    offsets = []

    # 构造 A 个平移的中心偏移位置（围绕中心，呈网格或螺旋）
    for i in range(batch):
        dx = ((i % 3) - 1) * shift_step  # -step, 0, +step 循环
        dy = ((i // 3) - 1) * shift_step
        offsets.append((dx, dy))

    crops = []
    for dx, dy in offsets:
        new_cx, new_cy = cx + dx, cy + dy
        left = max(0, new_cx - crop_size // 2)
        upper = max(0, new_cy - crop_size // 2)
        right = min(W, left + crop_size)
        lower = min(H, upper + crop_size)
        crop = image.crop((left, upper, right, lower))

        # 若裁剪不满 crop_size，进行 padding
        if crop.size != (crop_size, crop_size):
            padded = Image.new("RGB", (crop_size, crop_size))
            padded.paste(crop, (0, 0))
            crop = padded

        tensor = transforms.ToTensor()(crop)
        crops.append(tensor)

    return th.stack(crops, dim=0)  # [A, 3, crop_size, crop_size]



def create_argparser():
    defaults = dict(
        data_name='SVS',
        data_dir="normalized_images",
        multi_gpu=None,
        training_mode="consistency_distillation",
        gpu_dev="0",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        sampler="onestep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=0.1,
        steps=40,
        model_path="F:/ranse/textpicture/model018000.pt",# model411000.pt 0.8160 model489000.pt 0.8712   model522000 0.8939   target_
        seed=42,
        ts="0,22,39",
        num_ensemble=1,
        out_dir='G:/cncdiff',
        dyed="normalpicture.png",
        crop_size=512
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
