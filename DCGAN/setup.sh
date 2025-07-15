#!/bin/bash

# DCGAN 快速启动脚本

echo "=== PyTorch DCGAN 项目启动脚本 ==="

# 检查Python环境
echo "检查Python环境..."
python --version

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 检查PyTorch安装
echo "检查PyTorch安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 创建必要的目录
echo "创建项目目录..."
mkdir -p dataset/images
mkdir -p output/checkpoints
mkdir -p output/samples
mkdir -p output/logs

echo "环境设置完成！"
echo ""
echo "下一步："
echo "1. 将您的图片放入 dataset/images/ 目录"
echo "2. 运行: python train.py --dataroot ./dataset --num_epochs 50"
echo "3. 训练完成后运行: python evaluate.py --model_path ./output/checkpoints/dcgan_final.pth"
