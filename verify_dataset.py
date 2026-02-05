# -*-coding:utf-8 -*-
'''
验证数据集文件是否匹配且可正常读取
'''
import os
import cv2
import numpy as np

# 设置数据集路径
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lung_split_dataset")
images_dir = os.path.join(dataset_path, "Training_Images")
labels_dir = os.path.join(dataset_path, "Training_Labels")

print(f"数据集路径: {dataset_path}")
print(f"图像目录: {images_dir}")
print(f"标签目录: {labels_dir}")

# 检查目录是否存在
if not os.path.exists(images_dir):
    print(f"错误: 图像目录不存在: {images_dir}")
    exit(1)

if not os.path.exists(labels_dir):
    print(f"错误: 标签目录不存在: {labels_dir}")
    exit(1)

# 获取所有图像和标签文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(".tif")]
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".tif")]

print(f"\n=== 文件数量检查 ===")
print(f"图像文件数量: {len(image_files)}")
print(f"标签文件数量: {len(label_files)}")

if len(image_files) == 0:
    print("错误: 未找到任何图像文件!")
    exit(1)

if len(label_files) == 0:
    print("错误: 未找到任何标签文件!")
    exit(1)

if len(image_files) != len(label_files):
    print(f"警告: 图像和标签文件数量不匹配! {len(image_files)} != {len(label_files)}")

# 检查文件名是否匹配
print(f"\n=== 文件名匹配检查 ===")
image_ids = set([os.path.splitext(f)[0] for f in image_files])
label_ids = set([os.path.splitext(f)[0] for f in label_files])

missing_in_labels = image_ids - label_ids
missing_in_images = label_ids - image_ids

if missing_in_labels:
    print(f"错误: 以下图像ID在标签中缺失: {sorted(missing_in_labels)}")
else:
    print("✓ 所有图像ID在标签中都存在")

if missing_in_images:
    print(f"警告: 以下标签ID在图像中缺失: {sorted(missing_in_images)}")
else:
    print("✓ 所有标签ID在图像中都存在")

# 测试读取几个文件
print(f"\n=== 文件读取测试 ===")
test_files = sorted(image_files)[:3]  # 测试前3个文件

for i, image_file in enumerate(test_files):
    image_path = os.path.join(images_dir, image_file)
    label_file = image_file.replace(".tif", ".tif")
    label_path = os.path.join(labels_dir, label_file)
    
    try:
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"✗ 无法读取图像文件: {image_path}")
        else:
            print(f"✓ 成功读取图像: {image_file}, 形状: {image.shape}, 数据类型: {image.dtype}")
        
        # 读取标签
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            print(f"✗ 无法读取标签文件: {label_path}")
        else:
            print(f"✓ 成功读取标签: {label_file}, 形状: {label.shape}, 数据类型: {label.dtype}")
            # 检查标签中的唯一值
            unique_values = np.unique(label)
            print(f"  标签{label_file}中的唯一值: {unique_values}")
    
    except Exception as e:
        print(f"✗ 读取文件时出错: {str(e)}")

print(f"\n=== 验证完成 ===")