import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from model import DCGAN
import argparse


def load_checkpoint(model_path, device):
    """加载训练好的模型"""
    dcgan = DCGAN(device=device)
    epoch = dcgan.load_models(model_path)
    print(f"已加载第 {epoch} 轮的模型")
    return dcgan


def generate_images(dcgan, num_images=64, output_dir="./generated"):
    """生成图像并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图像
    fake_images = dcgan.generate_images(num_images)
    
    # 保存网格图像
    vutils.save_image(
        fake_images,
        f'{output_dir}/generated_grid.png',
        normalize=True,
        nrow=8
    )
    
    # 保存单独的图像
    for i in range(min(num_images, 16)):  # 最多保存16张单独图像
        vutils.save_image(
            fake_images[i],
            f'{output_dir}/generated_{i:03d}.png',
            normalize=True
        )
    
    print(f"已生成 {num_images} 张图像，保存在 {output_dir}")
    return fake_images


def interpolate_between_images(dcgan, num_steps=10, output_dir="./interpolation"):
    """在两个随机点之间进行插值，展示生成器的连续性"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成两个随机噪声向量
    z1 = dcgan.generate_noise(1)
    z2 = dcgan.generate_noise(1)
    
    interpolated_images = []
    
    for i in range(num_steps):
        # 线性插值
        alpha = i / (num_steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # 生成图像
        with torch.no_grad():
            img = dcgan.netG(z_interp)
            interpolated_images.append(img)
    
    # 保存插值序列
    all_images = torch.cat(interpolated_images, dim=0)
    vutils.save_image(
        all_images,
        f'{output_dir}/interpolation_sequence.png',
        normalize=True,
        nrow=num_steps
    )
    
    print(f"插值序列已保存在 {output_dir}")


def create_animation(dcgan, num_frames=100, output_dir="./animation"):
    """创建随机游走动画"""
    os.makedirs(output_dir, exist_ok=True)
    
    frames = []
    current_z = dcgan.generate_noise(1)
    
    for i in range(num_frames):
        # 添加小的随机噪声实现平滑变化
        noise_step = torch.randn_like(current_z) * 0.1
        current_z = current_z + noise_step
        
        # 生成图像
        with torch.no_grad():
            img = dcgan.netG(current_z)
            
        # 转换为PIL图像
        img_pil = transforms.ToPILImage()(
            (img.squeeze().cpu() + 1) / 2  # 反归一化到[0,1]
        )
        frames.append(img_pil)
        
        # 保存单帧
        img_pil.save(f'{output_dir}/frame_{i:03d}.png')
    
    # 创建GIF动画
    frames[0].save(
        f'{output_dir}/generation_animation.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 每帧100ms
        loop=0
    )
    
    print(f"动画已保存在 {output_dir}")


def evaluate_model(model_path, args):
    """评估和可视化模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    dcgan = load_checkpoint(model_path, device)
    dcgan.netG.eval()
    
    # 生成图像
    print("生成随机图像...")
    generate_images(dcgan, args.num_images, f"{args.output_dir}/generated")
    
    # 插值展示
    print("生成插值序列...")
    interpolate_between_images(dcgan, args.interpolation_steps, f"{args.output_dir}/interpolation")
    
    # 创建动画
    if args.create_animation:
        print("创建动画...")
        create_animation(dcgan, args.animation_frames, f"{args.output_dir}/animation")
    
    print("评估完成!")


def main():
    parser = argparse.ArgumentParser(description='DCGAN推理和评估脚本')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--output_dir', default='./evaluation_results', help='输出目录')
    parser.add_argument('--num_images', type=int, default=64, help='生成图像数量')
    parser.add_argument('--interpolation_steps', type=int, default=10, help='插值步数')
    parser.add_argument('--create_animation', action='store_true', help='是否创建动画')
    parser.add_argument('--animation_frames', type=int, default=50, help='动画帧数')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始评估
    evaluate_model(args.model_path, args)


if __name__ == "__main__":
    main()
