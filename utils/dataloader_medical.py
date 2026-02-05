import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
# 假设 preprocess_input 和 cvtColor 函数在 utils 模块中定义
# 如果没有，需要添加这些函数的实现
from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, image_suffix, label_suffix,
                 color_map=None):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix

        # 三分类颜色映射表（默认：白红蓝）
        self.color_map = color_map or {
            0: [255, 255, 255],
            1: [255, 0, 0],
            2: [0, 0, 255]
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        if self.train:
            jpg = Image.open(
                os.path.join(os.path.join(self.dataset_path, "Training_Images"), name + "." + self.image_suffix))
            png = Image.open(
                os.path.join(os.path.join(self.dataset_path, "Training_Labels"), name + "." + self.label_suffix))
        else:
            jpg = Image.open(
                os.path.join(os.path.join(self.dataset_path, "Val_Images"), name + "." + self.image_suffix))
            png = Image.open(
                os.path.join(os.path.join(self.dataset_path, "Val_Labels"), name + "." + self.label_suffix))

        # -------------------------------#
        #   数据增强
        # -------------------------------#
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)

        # -------------------------------------------------------#
        #   三分类标签处理：假设标签灰度值为0、128、255分别对应三个类别
        # -------------------------------------------------------#
        modify_png = np.zeros_like(png)
        modify_png[png == 0] = 0  # 标签0：黑色区域
        modify_png[png == 128] = 1  # 标签1：灰色区域
        modify_png[png == 200] = 2  # 标签2：白色区域

        seg_labels = modify_png
        # 生成one-hot编码，注意num_classes+1（包含背景）
        seg_labels = np.eye(self.num_classes + 1)[seg_labels.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, modify_png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # resize image
        rand_jit1 = self.rand(1 - jitter, 1 + jitter)
        rand_jit2 = self.rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def visualize_label(self, index=0):
        """可视化三分类标签映射效果"""
        if index >= self.length:
            raise IndexError(f"索引超出范围，最大索引为{self.length - 1}")

        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 加载标签图像
        if self.train:
            label_path = os.path.join(os.path.join(self.dataset_path, "Training_Labels"), name + "." + self.label_suffix)
        else:
            label_path = os.path.join(os.path.join(self.dataset_path, "Val_Labels"), name + "." + self.label_suffix)

        label = Image.open(label_path)
        gray_label = np.array(label)

        # 确保标签值映射一致
        color_map = {
            0: [255, 255, 255],
            1: [255, 0, 0],
            2: [0, 0, 255]
        }

        # 创建彩色标签图像
        color_png = np.zeros((gray_label.shape[0], gray_label.shape[1], 3), dtype=np.uint8)
        for cls_id, color in color_map.items():
            color_png[gray_label == cls_id] = color

        # 保存可视化结果
        vis_dir = os.path.join(os.path.dirname(label_path), "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{name}_label_vis.jpg")

        # 将RGB图像转换为BGR格式（OpenCV保存图像时需要）
        color_bgr = cv2.cvtColor(color_png, cv2.COLOR_RGB2BGR)
        cv2.imwrite(vis_path, color_bgr)

        return color_png


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels