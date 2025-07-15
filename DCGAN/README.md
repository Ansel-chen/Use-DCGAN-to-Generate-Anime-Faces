# PyTorch DCGAN 项目

这是一个基于PyTorch实现的深度卷积生成对抗网络(DCGAN)项目，可以根据您提供的图片数据集生成新的图像。

## 项目结构

```
DCCGAN/
├── model.py              # DCGAN模型定义（生成器和判别器）
├── train.py              # 训练脚本
├── evaluate.py           # 模型评估和图像生成脚本
├── prepare_dataset.py    # 数据集准备工具
├── requirements.txt      # 依赖包列表
├── README.md            # 项目说明文档
└── output/              # 训练输出目录（自动创建）
    ├── checkpoints/     # 模型检查点
    ├── samples/         # 训练过程中的样本
    └── logs/           # 训练日志和损失曲线
```

## 环境设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证PyTorch安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

## 使用步骤

### 第一步：准备数据集

使用数据集准备工具将您的图片整理成训练格式：

```bash
python prepare_dataset.py
```

或者手动创建数据集结构：
```
dataset/
└── images/
    ├── img_000001.jpg
    ├── img_000002.jpg
    └── ...
```

**数据集要求：**
- 图片格式：支持 JPG、PNG、BMP、TIFF、WebP
- 图片大小：建议至少64x64像素（训练时会自动调整）
- 数量：建议至少1000张图片以获得好的效果
- 内容：同一类型的图片（如人脸、动物、风景等）

### 第二步：开始训练

```bash
python train.py --dataroot ./dataset --num_epochs 100 --batch_size 64
```

**训练参数说明：**
- `--dataroot`: 数据集路径
- `--num_epochs`: 训练轮数（推荐100-200）
- `--batch_size`: 批次大小（根据GPU内存调整）
- `--image_size`: 输出图像大小（默认64x64）
- `--lr`: 学习率（默认0.0002）
- `--output_dir`: 输出目录（默认./output）

### 第三步：生成图像

训练完成后，使用评估脚本生成新图像：

```bash
python evaluate.py --model_path ./output/checkpoints/dcgan_final.pth --num_images 64
```

## 模型架构

### 生成器 (Generator)
- 输入：100维随机噪声向量
- 输出：64x64x3 彩色图像
- 架构：5层转置卷积 + 批归一化 + ReLU/Tanh激活

### 判别器 (Discriminator)
- 输入：64x64x3 彩色图像
- 输出：真实/生成的二分类概率
- 架构：5层卷积 + 批归一化 + LeakyReLU激活

## 训练过程监控

训练过程中会自动保存：

1. **损失曲线图** (`output/logs/loss_curves.png`)
   - 监控生成器和判别器的损失变化

2. **生成样本** (`output/samples/`)
   - 每500次迭代保存一次生成的图像网格

3. **模型检查点** (`output/checkpoints/`)
   - 每10轮保存一次模型状态

## 训练技巧

### 1. 数据预处理
- 图像归一化到[-1, 1]范围
- 使用中心裁剪保持宽高比
- 数据增强（可选）

### 2. 训练稳定性
- 使用标签平滑
- 生成器和判别器的学习率平衡
- 批归一化层的使用

### 3. 超参数调优
```python
# 推荐的训练参数
learning_rate = 0.0002
beta1 = 0.5
batch_size = 64
nz = 100  # 噪声维度
ngf = 64  # 生成器特征数
ndf = 64  # 判别器特征数
```

## 常见问题

### Q1: 训练不稳定，损失震荡严重
**解决方案：**
- 降低学习率（0.0001）
- 增加批次大小
- 检查数据集质量

### Q2: 生成的图像质量差
**解决方案：**
- 增加训练轮数
- 改进数据集（更多样本，更高质量）
- 调整网络架构参数

### Q3: 模式坍塌（生成图像过于相似）
**解决方案：**
- 使用不同的损失函数
- 调整训练频率比例
- 增加噪声维度

### Q4: GPU内存不足
**解决方案：**
- 减小批次大小
- 减小图像尺寸
- 使用梯度累积

## 评估指标

### 1. 视觉质量评估
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- 人工评估

### 2. 训练稳定性
- 损失曲线平稳性
- 生成样本多样性
- 模式坍塌检测

## 扩展功能

### 1. 条件生成（Conditional GAN）
可以扩展为条件GAN，根据标签生成特定类别的图像。

### 2. 渐进式训练（Progressive GAN）
逐步增加图像分辨率，生成更高质量的图像。

### 3. 样式迁移
结合样式迁移技术，生成特定风格的图像。

## 参考资料

- [DCGAN论文](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN教程](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GAN训练技巧](https://github.com/soumith/ganhacks)

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。
