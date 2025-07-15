import os
from datasets import load_dataset
from pathlib import Path
import requests
import zipfile
from PIL import Image
import io

def download_anime_faces_dataset():
    """
    从Hugging Face下载anime-faces数据集到~/datasets文件夹
    """
    # 设置数据集保存路径
    dataset_path = Path.home() / "datasets" / "anime-faces"
    
    # 创建目录（如果不存在）
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载anime-faces数据集到: {dataset_path}")
    
    # 尝试不同的数据集名称
    dataset_names = [
        "huggan/anime-faces",
        "anime-faces", 
        "huggan/AFHQv2",
        "sayakpaul/anime-faces-dataset"
    ]
    
    for dataset_name in dataset_names:
        try:
            print(f"尝试下载数据集: {dataset_name}")
            # 从Hugging Face下载数据集
            dataset = load_dataset(dataset_name, cache_dir=str(dataset_path))
            
            print(f"数据集下载完成！")
            print(f"数据集信息: {dataset}")
            print(f"训练集大小: {len(dataset['train'])}")
            
            # 保存数据集到本地
            dataset.save_to_disk(str(dataset_path / "processed"))
            print(f"数据集已保存到: {dataset_path / 'processed'}")
            
            return dataset
            
        except Exception as e:
            print(f"尝试 {dataset_name} 失败: {e}")
            continue
    
    # 如果所有Hugging Face数据集都失败，尝试其他方法
    print("尝试从其他源下载...")
    return download_alternative_dataset(dataset_path)

def download_alternative_dataset(dataset_path):
    """
    备选下载方法 - 从GitHub或其他源下载
    """
    try:
        # 可以尝试其他数据源
        print("正在寻找备选数据源...")
        
        # 这里可以添加其他数据下载逻辑
        # 例如从GitHub、Kaggle等下载
        
        return None
        
    except Exception as e:
        print(f"备选下载方法失败: {e}")
        return None

def check_network_connection():
    """
    检查网络连接
    """
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    # 检查网络连接
    if not check_network_connection():
        print("网络连接失败，请检查网络设置")
        exit(1)
    
    # 检查是否安装了必要的包
    try:
        import datasets
        import huggingface_hub
        import requests
        from PIL import Image
    except ImportError as e:
        print("请先安装必要的依赖包:")
        print("pip install datasets huggingface_hub requests pillow")
        exit(1)
    
    # 下载数据集
    dataset = download_anime_faces_dataset()
    
    if dataset is not None:
        print("数据集下载成功！")
    else:
        print("数据集下载失败。您可以尝试:")
        print("1. 检查网络连接")
        print("2. 更新datasets库: pip install --upgrade datasets")
        print("3. 手动从 https://huggingface.co/datasets/huggan/anime-faces 下载")