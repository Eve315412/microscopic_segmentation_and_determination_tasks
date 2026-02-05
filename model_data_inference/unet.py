import colorsys
import copy
import time
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image


class Unet(object):
    _defaults = {
        "model_path": 'model_data/unet_vgg_voc.pth',
        "num_classes": 3,
        "backbone": "resnet50",
        "input_shape": [512, 512],
        "mix_type": 0,
        "cuda": True,
        "use_amp": False  # 新增：是否使用混合精度推理
    }

    def __init__(self, **kwargs):
        # 初始化默认参数
        self.__dict__.update(self._defaults)
        # 更新用户提供的参数
        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        print(f"使用设备: {self.device}")

        # 初始化颜色映射
        self.init_colors()
        # 加载模型
        self.generate()

    def init_colors(self):
        """初始化类别颜色映射"""
        if self.num_classes <= 3:
            # 三分类专用颜色映射
            self.colors = [(0, 0, 0), (128, 128, 128), (200, 200, 200)]
        else:
            # 自动生成颜色映射
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        print(f"已初始化{self.num_classes}类颜色映射")

    def generate(self):
        """加载并初始化模型"""
        try:
            # 创建模型
            self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
            # 模型初始化
            if self.backbone == "vgg16" or self.backbone == "resnet50":
                print(f"使用{self.backbone}作为骨干网络")

            # 安全加载模型权重
            print(f"正在加载模型: {self.model_path}")
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
                # 移除weights_only参数以兼容旧版PyTorch
            )

            # 处理可能的权重不匹配问题
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

            if len(pretrained_dict) == 0:
                raise ValueError("未找到匹配的模型权重，请检查模型路径和结构")

            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
            print(f"成功加载{len(pretrained_dict)}/{len(model_dict)}个权重参数")

            # 模型设置为评估模式
            self.net = self.net.eval()
            self.net = self.net.to(self.device)

            # 混合精度推理设置
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
                print("已启用混合精度推理")

        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        except KeyError as e:
            raise KeyError(f"模型权重不匹配: {str(e)}，请检查模型结构和类别数设置")
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def detect_image(self, image, display=True):
        """
        对输入图像进行分割检测

        参数:
            image: 输入图像(PIL Image)
            display: 是否显示分割结果

        返回:
            分割结果图像(PIL Image)
        """
        start_time = time.time()

        # 图像预处理
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h, orininal_w = np.array(image).shape[:2]

        # 调整图像大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 预处理
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        # 模型推理
        with torch.no_grad():
            # 转换为张量
            images = torch.from_numpy(image_data)
            images = images.to(self.device)

            # 混合精度推理
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pr = self.net(images)[0]
            else:
                pr = self.net(images)[0]

            # 计算分割结果
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 调整结果尺寸到原始图像大小
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            if cv2 is not None:
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            else:
                pr = self._resize_array(pr, (orininal_w, orininal_h))
            pr = pr.argmax(axis=-1)

        # 处理分割结果
        if self.mix_type == 0:
            # 半透明叠加
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            result = Image.fromarray(np.uint8(seg_img))
            result = Image.blend(old_img, result, 0.7)
        elif self.mix_type == 1:
            # 纯分割图
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            result = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            # 仅显示分割区域
            seg_img = (np.expand_dims(pr > 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            result = Image.fromarray(np.uint8(seg_img))
        else:
            raise ValueError(f"不支持的mix_type: {self.mix_type}，支持0、1、2")

        # 显示推理时间
        if display:
            end_time = time.time()
            print(f"推理时间: {end_time - start_time:.2f}秒")

            # 显示类别分布
            class_counts = np.bincount(pr.flatten())
            if len(class_counts) < self.num_classes:
                class_counts = np.pad(class_counts, (0, self.num_classes - len(class_counts)), 'constant')

            for i in range(self.num_classes):
                percentage = (class_counts[i] / (orininal_h * orininal_w)) * 100
                print(f"类别{i} ({self.colors[i]}): {class_counts[i]}像素, {percentage:.2f}%")

        return result

    def detect_image_ui(self, image):
        """
        用于UI展示的分割结果生成，固定使用白、蓝、红三色映射

        参数:
            image: 输入图像(PIL Image)

        返回:
            分割结果图像(PIL Image)，颜色映射为：
            - 类别0: 白色 (255,255,255)
            - 类别1: 蓝色 (0,0,255)
            - 类别2: 红色 (255,0,0)
        """
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.device)

            # 支持混合精度推理
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pr = self.net(images)[0]
            else:
                pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            if cv2 is not None:
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            else:
                pr = self._resize_array(pr, (orininal_w, orininal_h))
            pr = pr.argmax(axis=-1)

        # 转换为指定颜色的图像（白、蓝、红）
        rgb_img = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
        rgb_img[pr == 0] = [255, 255, 255]  # 白色
        rgb_img[pr == 1] = [0, 0, 255]  # 蓝色
        rgb_img[pr == 2] = [255, 0, 0]  # 红色

        image = Image.fromarray(rgb_img)
        return image

    def save_class_images(self, folder):
        """
        保存各个类别的分割图像（白、蓝、红三色掩码）

        参数:
            folder: 保存图像的目标文件夹路径
        """
        try:
            import os  # 导入os模块用于文件夹操作
            os.makedirs(folder, exist_ok=True)  # 确保文件夹存在，不存在则创建

            # 读取检测结果（默认从"images/tmp/single_result.jpg"读取，需提前生成）
            img = cv2.imread("images/tmp/single_result.jpg")  # 以BGR格式读取（OpenCV默认）
            if img is None:
                raise FileNotFoundError("未找到检测结果图像，请先运行detect_image_ui生成结果")

            # 保存背景图像（白色）
            bg_mask = np.zeros_like(img)
            bg_mask[np.all(img == [255, 255, 255], axis=-1)] = [255, 255, 255]  # 匹配白色
            cv2.imwrite(f"{folder}/background_mask.jpg", bg_mask)

            # 保存类别1图像（蓝色）
            class1_mask = np.zeros_like(img)
            class1_mask[np.all(img == [255, 0, 0], axis=-1)] = [255, 0, 0]  # 注意OpenCV中蓝色为[255,0,0]（BGR）
            cv2.imwrite(f"{folder}/class1_mask.jpg", class1_mask)

            # 保存类别2图像（红色）
            class2_mask = np.zeros_like(img)
            class2_mask[np.all(img == [0, 0, 255], axis=-1)] = [0, 0, 255]  # 注意OpenCV中红色为[0,0,255]（BGR）
            cv2.imwrite(f"{folder}/class2_mask.jpg", class2_mask)

            print(f"类别图像已成功保存到 {folder}")
        except Exception as e:
            print(f"保存类别图像时出错: {str(e)}")

    def get_segmentation_mask(self, image):
        """获取分割掩码"""
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]

        # 调整图像大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 预处理
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        # 模型推理
        with torch.no_grad():
            # 转换为张量
            images = torch.from_numpy(image_data)
            images = images.to(self.device)

            # 混合精度推理
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pr = self.net(images)[0]
            else:
                pr = self.net(images)[0]

            # 计算分割结果
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 调整结果尺寸到原始图像大小
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            if cv2 is not None:
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            else:
                pr = self._resize_array(pr, (orininal_w, orininal_h))
            mask = pr.argmax(axis=-1)

        return mask

    def get_class_bounding_boxes(self, image, min_area=100):
        """获取每个类别的边界框"""
        mask = self.get_segmentation_mask(image)
        bounding_boxes = {i: [] for i in range(self.num_classes)}

        for class_id in range(1, self.num_classes):  # 跳过背景类别
            # 创建类别掩码
            class_mask = np.uint8(mask == class_id) * 255
            if np.sum(class_mask) == 0:
                continue

            # 查找轮廓
            if cv2 is not None:
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours = self._find_components(class_mask)

            for contour in contours:
                # 计算轮廓面积
                area = self._contour_area(contour)
                if area < min_area:
                    continue

                # 获取边界框
                x, y, w, h = self._bounding_rect(contour)
                bounding_boxes[class_id].append((x, y, x + w, y + h, area))

        return bounding_boxes

    def _resize_array(self, arr, size):
        w, h = size
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.float32))
            img = img.resize((w, h), Image.BILINEAR)
            return np.array(img)
        else:
            chans = []
            for c in range(arr.shape[2]):
                img = Image.fromarray(arr[..., c].astype(np.float32))
                img = img.resize((w, h), Image.BILINEAR)
                chans.append(np.array(img))
            return np.stack(chans, axis=2)

    def _find_components(self, binary_mask):
        h, w = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        comps = []
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] and not visited[i, j]:
                    minx, miny, maxx, maxy = j, i, j, i
                    area = 0
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        x, y = stack.pop()
                        area += 1
                        if y < minx: minx = y
                        if y > maxx: maxx = y
                        if x < miny: miny = x
                        if x > maxy: maxy = x
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < h and 0 <= ny < w and binary_mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
                    comps.append(((minx, miny, maxx, maxy), area))
        return comps

    def _contour_area(self, contour):
        if cv2 is not None:
            return cv2.contourArea(contour)
        else:
            rect, area = contour if isinstance(contour, tuple) else (None, 0)
            return int(area)

    def _bounding_rect(self, contour):
        if cv2 is not None:
            x, y, w, h = cv2.boundingRect(contour)
            return x, y, w, h
        else:
            (minx, miny, maxx, maxy), _ = contour
            return minx, miny, maxx - minx + 1, maxy - miny + 1
