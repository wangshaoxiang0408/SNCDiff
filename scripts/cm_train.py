"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("../")  #给系统添加路径
sys.path.append("./")

from cm import dist_util, logger
# from cm.image_datasets import load_data
from cm.SVSloader import SVSDataset
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy
import torchvision.transforms as transforms
import torch
import os
from PIL import Image


def get_all_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".pt", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    return image_paths

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

    return torch.stack(crops, dim=0)  # [A, 3, crop_size, crop_size]

def main():
    args = create_argparser().parse_args()

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

    data_list = torch.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True)
    data = iter(data_list)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode="adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=200,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {num_params}')
    model.load_state_dict(torch.load('F:/ranse/textpicture/target_model023000.pt'))
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()
        
    if args.multi_gpu:
        model = torch.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
        
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )
    num_params = sum(p.numel() for p in target_model.parameters())
    print(f'模型2参数数量: {num_params}')
    if args.multi_gpu:
        model_t = torch.nn.DataParallel(
            model_t, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model_t.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.load_state_dict(torch.load('F:/ranse/textpicture/target_model023000.pt'))
        target_model.to(dist_util.dev())

    # target_model.to(dist_util.dev())
    # target_model.load_state_dict(torch.load('0505/target_model020000.pt'))
    target_model.train()

    # dist_util.sync_params(target_model.parameters())
    # dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        dataloader=data_list,
        batch_size=8,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dyed=load_png_with_shifted_crops(args.dyed, crop_size=args.crop_size, batch=args.batch_size, shift_step=20),
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='SVS',
        data_dir="normalized_images",
        schedule_sampler="uniform",
        out_dir = "textpi",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        gpu_dev="0",
        multi_gpu=None,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        target_ema_mode = "adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=150,
        training_mode="consistency_training",
        total_training_steps =100000,
        # ema_rate="0.9999,0.99994,0.9999432189950708" ,
        log_interval=10,
        save_interval=1000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dyed="normalpicture.png",
        crop_size=512
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

