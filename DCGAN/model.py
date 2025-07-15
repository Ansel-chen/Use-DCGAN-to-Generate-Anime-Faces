import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """DCGAN生成器网络"""
    
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        Args:
            nz: 噪声向量维度 (latent vector size)
            ngf: 生成器特征图数量 (generator feature map size)
            nc: 输出图像通道数 (number of channels)
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        # 反卷积层序列
        # 输入: nz x 1 x 1
        self.main = nn.Sequential(
            # 第一层: nz -> ngf*8
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            
            # 第二层: ngf*8 -> ngf*4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            
            # 第三层: ngf*4 -> ngf*2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            
            # 第四层: ngf*2 -> ngf
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: ngf x 32 x 32
            
            # 输出层: ngf -> nc
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: nc x 64 x 64
        )
    
    def forward(self, input):
        """前向传播"""
        return self.main(input)


class Discriminator(nn.Module):
    """DCGAN判别器网络"""
    
    def __init__(self, nc=3, ndf=64):
        """
        Args:
            nc: 输入图像通道数
            ndf: 判别器特征图数量
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        # 卷积层序列
        self.main = nn.Sequential(
            # 输入层: nc -> ndf
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            
            # 第二层: ndf -> ndf*2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            
            # 第三层: ndf*2 -> ndf*4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            
            # 第四层: ndf*4 -> ndf*8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            
            # 输出层: ndf*8 -> 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )
    
    def forward(self, input):
        """前向传播"""
        return self.main(input).view(-1, 1).squeeze(1)


def weights_init(m):
    """
    自定义权重初始化函数
    从DCGAN论文中: 所有权重都从均值为0，标准差为0.02的正态分布中随机初始化
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    """DCGAN主类，包含生成器和判别器"""
    
    def __init__(self, nz=100, ngf=64, ndf=64, nc=3, device='cuda'):
        """
        Args:
            nz: 噪声向量维度
            ngf: 生成器特征图数量
            ndf: 判别器特征图数量
            nc: 图像通道数
            device: 计算设备
        """
        self.nz = nz
        self.device = device
        
        # 初始化网络
        self.netG = Generator(nz, ngf, nc).to(device)
        self.netD = Discriminator(nc, ndf).to(device)
        
        # 应用权重初始化
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        print("生成器网络结构:")
        print(self.netG)
        print("\n判别器网络结构:")
        print(self.netD)
    
    def generate_noise(self, batch_size):
        """生成随机噪声"""
        return torch.randn(batch_size, self.nz, 1, 1, device=self.device)
    
    def generate_images(self, num_images=64):
        """生成图像"""
        with torch.no_grad():
            noise = self.generate_noise(num_images)
            fake_images = self.netG(noise)
            return fake_images
    
    def save_models(self, checkpoint_dir, epoch):
        """保存模型"""
        torch.save({
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
            'epoch': epoch
        }, f'{checkpoint_dir}/dcgan_epoch_{epoch}.pth')
    
    def load_models(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(checkpoint['generator_state_dict'])
        self.netD.load_state_dict(checkpoint['discriminator_state_dict'])
        return checkpoint['epoch']


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建DCGAN实例
    dcgan = DCGAN(device=device)
    
    # 测试生成器
    noise = dcgan.generate_noise(4)
    fake_images = dcgan.netG(noise)
    print(f"生成的图像尺寸: {fake_images.shape}")
    
    # 测试判别器
    real_images = torch.randn(4, 3, 64, 64).to(device)
    output = dcgan.netD(real_images)
    print(f"判别器输出尺寸: {output.shape}")
