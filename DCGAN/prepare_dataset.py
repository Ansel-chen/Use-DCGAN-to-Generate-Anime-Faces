import os
import shutil


def prepare_dataset(source_dir, target_dir="./dataset", max_images=None):
    """
    准备训练数据集
    
    Args:
        source_dir: 原始图片目录
        target_dir: 目标数据集目录
        max_images: 最大图片数量限制
    """
    
    # 创建目标目录结构
    # DCGAN需要的目录结构: dataset/class_name/images
    class_dir = os.path.join(target_dir, "images")
    os.makedirs(class_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    copied_count = 0
    
    print(f"开始从 {source_dir} 复制图片到 {target_dir}")
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名
            _, ext = os.path.splitext(file.lower())
            if ext in supported_formats:
                source_path = os.path.join(root, file)
                
                # 创建新的文件名（避免重名）
                new_filename = f"img_{copied_count:06d}{ext}"
                target_path = os.path.join(class_dir, new_filename)
                
                try:
                    # 复制文件
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
                    
                    if copied_count % 100 == 0:
                        print(f"已复制 {copied_count} 张图片...")
                    
                    # 检查是否达到最大数量
                    if max_images and copied_count >= max_images:
                        print(f"已达到最大图片数量限制: {max_images}")
                        break
                        
                except Exception as e:
                    print(f"复制文件 {source_path} 时出错: {e}")
        
        # 如果达到最大数量，跳出外层循环
        if max_images and copied_count >= max_images:
            break
    
    print(f"数据集准备完成！")
    print(f"总共复制了 {copied_count} 张图片到 {target_dir}")
    print(f"数据集结构: {target_dir}/images/")
    
    return target_dir, copied_count


def validate_dataset(dataset_dir):
    """验证数据集结构"""
    
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录 {dataset_dir} 不存在")
        return False
    
    # 查找图片文件
    image_count = 0
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in supported_formats:
                image_count += 1
    
    if image_count == 0:
        print(f"错误: 在 {dataset_dir} 中没有找到有效的图片文件")
        return False
    
    print(f"数据集验证通过！找到 {image_count} 张图片")
    return True


def get_dataset_info(dataset_dir):
    """获取数据集信息"""
    
    if not os.path.exists(dataset_dir):
        return None
    
    info = {
        'total_images': 0,
        'formats': {},
        'subdirs': []
    }
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for root, dirs, files in os.walk(dataset_dir):
        # 记录子目录
        relative_root = os.path.relpath(root, dataset_dir)
        if relative_root != '.':
            info['subdirs'].append(relative_root)
        
        # 统计图片
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in supported_formats:
                info['total_images'] += 1
                info['formats'][ext] = info['formats'].get(ext, 0) + 1
    
    return info


def main():
    """数据集准备工具的主函数"""
    
    print("=== DCGAN 数据集准备工具 ===")
    
    # 获取用户输入
    source_dir = "/home/chen/datasets/anime_faces"  # 默认源目录
    
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 {source_dir} 不存在")
        return
    
    target_dir = input("请输入目标数据集目录 (默认: ./dataset): ").strip()
    if not target_dir:
        target_dir = "./dataset"
    
    max_images_input = input("请输入最大图片数量 (回车表示无限制): ").strip()
    max_images = None
    if max_images_input:
        try:
            max_images = int(max_images_input)
        except ValueError:
            print("无效的数字，将不限制图片数量")
    
    # 准备数据集
    result_dir, count = prepare_dataset(source_dir, target_dir, max_images)
    
    # 验证数据集
    if validate_dataset(result_dir):
        print("\n数据集准备成功！")
        
        # 显示数据集信息
        info = get_dataset_info(result_dir)
        if info:
            print(f"\n数据集信息:")
            print(f"  总图片数: {info['total_images']}")
            print(f"  图片格式分布:")
            for fmt, count in info['formats'].items():
                print(f"    {fmt}: {count} 张")
            
            print(f"\n现在您可以使用以下命令开始训练:")
            print(f"python train.py --dataroot {result_dir} --num_epochs 25")
    
    else:
        print("数据集准备失败！")


if __name__ == "__main__":
    main()
