import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import numpy as np
import matplotlib.pyplot as plt
from model import DCGAN
import argparse


class DCGANTrainer:
    """DCGAN训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.create_directories()
        
        # 初始化数据加载器
        self.dataloader = self.create_dataloader()
        
        # 初始化模型
        self.dcgan = DCGAN(
            nz=args.nz, 
            ngf=args.ngf, 
            ndf=args.ndf, 
            nc=args.nc, 
            device=self.device
        )
        
        # 初始化优化器
        self.optimizerD = optim.Adam(
            self.dcgan.netD.parameters(), 
            lr=args.lr, 
            betas=(args.beta1, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.dcgan.netG.parameters(), 
            lr=args.lr, 
            betas=(args.beta1, 0.999)
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 用于可视化的固定噪声
        self.fixed_noise = self.dcgan.generate_noise(64)
        
        # 训练历史
        self.G_losses = []
        self.D_losses = []
    
    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(f"{self.args.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.args.output_dir}/samples", exist_ok=True)
        os.makedirs(f"{self.args.output_dir}/logs", exist_ok=True)
    
    def create_dataloader(self):
        """创建数据加载器"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
        ])
        
        # 加载数据集
        dataset = ImageFolder(root=self.args.dataroot, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size,
            shuffle=True, 
            num_workers=self.args.workers
        )
        
        print(f"数据集大小: {len(dataset)}")
        print(f"批次数量: {len(dataloader)}")
        
        return dataloader
    
    def train_discriminator(self, real_batch):
        """训练判别器"""
        self.dcgan.netD.zero_grad()
        
        # 训练真实图像
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
        
        output = self.dcgan.netD(real_batch).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        # 训练生成图像
        noise = self.dcgan.generate_noise(batch_size)
        fake = self.dcgan.netG(noise)
        label.fill_(0.)
        
        output = self.dcgan.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        self.optimizerD.step()
        
        return errD.item(), D_x, D_G_z1
    
    def train_generator(self, batch_size):
        """训练生成器"""
        self.dcgan.netG.zero_grad()
        
        label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
        noise = self.dcgan.generate_noise(batch_size)
        fake = self.dcgan.netG(noise)
        
        output = self.dcgan.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        self.optimizerG.step()
        
        return errG.item(), D_G_z2
    
    def save_samples(self, epoch):
        """保存生成的样本"""
        with torch.no_grad():
            fake = self.dcgan.netG(self.fixed_noise).detach().cpu()
            
        # 保存图像网格
        vutils.save_image(
            fake, 
            f'{self.args.output_dir}/samples/fake_samples_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=8
        )
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{self.args.output_dir}/logs/loss_curves.png')
        plt.close()
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        
        iters = 0
        for epoch in range(self.args.num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                
                # 训练判别器
                errD, D_x, D_G_z1 = self.train_discriminator(data[0])
                
                # 训练生成器
                errG, D_G_z2 = self.train_generator(data[0].size(0))
                
                # 记录损失
                self.G_losses.append(errG)
                self.D_losses.append(errD)
                
                # 输出训练状态
                if i % 50 == 0:
                    print(f'[{epoch}/{self.args.num_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {errD:.4f} Loss_G: {errG:.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
                # 保存样本
                if (iters % 500 == 0) or ((epoch == self.args.num_epochs-1) and (i == len(self.dataloader)-1)):
                    self.save_samples(epoch)
                
                iters += 1
            
            # 保存检查点
            if epoch % 10 == 0:
                self.dcgan.save_models(f"{self.args.output_dir}/checkpoints", epoch)
                
            # 更新损失图
            if epoch % 5 == 0:
                self.plot_losses()
        
        # 保存最终模型
        self.dcgan.save_models(f"{self.args.output_dir}/checkpoints", "final")
        self.plot_losses()
        
        print("训练完成!")


def main():
    parser = argparse.ArgumentParser(description='DCGAN训练脚本')
    parser.add_argument('--dataroot', default="./dataset", help='数据集根目录路径')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--image_size', type=int, default=64, help='输入图像大小')
    parser.add_argument('--nc', type=int, default=3, help='输入图像通道数')
    parser.add_argument('--nz', type=int, default=100, help='噪声向量大小')
    parser.add_argument('--ngf', type=int, default=64, help='生成器特征图大小')
    parser.add_argument('--ndf', type=int, default=64, help='判别器特征图大小')
    parser.add_argument('--num_epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
    parser.add_argument('--output_dir', default='./output', help='输出目录')
    
    args = parser.parse_args()
    
    # 打印参数
    print("训练参数:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # 开始训练
    trainer = DCGANTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
