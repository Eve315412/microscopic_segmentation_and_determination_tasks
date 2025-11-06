import sys
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# 导入model/unet.py中的Unet类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.unet import Unet

try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False
    print("警告: OpenCV未找到，将使用替代方法实现部分功能")

try:
    from skimage import filters, measure, morphology
    from skimage.color import rgb2gray
    skimage_available = True
    print("scikit-image已成功导入")
except ImportError:
    skimage_available = False
    print("警告: scikit-image未找到，将使用NumPy和PIL实现部分功能")
    # 定义rgb2gray的简单替代实现
    def rgb2gray(image):
        if len(image.shape) == 3:
            # 简单的灰度转换公式
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return image
    # 定义一个空的measure对象，避免导入错误
    class DummyMeasure:
        pass
    measure = DummyMeasure()

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 简单的U-Net分割模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # 编码器
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # 解码器
        self.up4 = self.up_conv(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = self.up_conv(128, 64)
        self.dec2 = self.conv_block(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # 解码器
        dec4 = self.up4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        out = self.out(dec2)
        
        # 上采样到原始大小
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return torch.sigmoid(out)

class ImageSegmentationApp:
    """图像分割与颗粒分析主应用程序"""
    def __init__(self, root):
        self.root = root
        self.root.title("图像分割与颗粒分析")
        self.root.geometry("1000x500")
        
        # 初始化变量
        self.original_image = None
        self.segmented_image = None
        self.stats = None
        self.worker_thread = None
        self.running = False
        
        # 深度学习模型相关
        self.model = None
        self.weight_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建主界面
        self.create_widgets()
        
        # 自动加载预训练模型
        self.auto_load_model()
    
    def create_widgets(self):
        """创建用户界面组件，优化图像区域占比"""
        # 顶部按钮栏（保持不变）
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.load_button = tk.Button(button_frame, text="加载图像", command=self.load_image, width=15)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.segment_button = tk.Button(button_frame, text="执行分割", command=self.start_segmentation, width=15, state=tk.DISABLED)
        self.segment_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="保存结果", command=self.save_results, width=15, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # 进度标签
        self.status_label = tk.Label(button_frame, text="就绪")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 1. 模型和权重说明（保持不变）
        model_info_frame = ttk.LabelFrame(self.root, text="模型和权重信息")
        model_info_frame.pack(fill=tk.X, padx=10, pady=5)
        # （此处省略模型信息文本框代码，保持不变）

        # 关键修改：使用PanedWindow实现左右布局，方便调整图像与控制面板占比
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)  # 填充整个窗口并允许扩展

        # 左侧控制面板（缩小占比）
        control_frame = ttk.LabelFrame(main_paned, text="控制面板")
        # 添加到PanedWindow，权重设为1（占比小）
        main_paned.add(control_frame, weight=1)

        # 右侧图像与结果区域（放大占比）
        right_frame = ttk.Frame(main_paned)
        # 添加到PanedWindow，权重设为4（占比大，是控制面板的4倍）
        main_paned.add(right_frame, weight=4)

        # 2. 原图和分割图像（放在右侧区域，占比更大）
        image_frame = ttk.LabelFrame(right_frame, text="图像显示")
        # 关键：让图像区域填充右侧空间并优先扩展
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图像对比区域
        compare_frame = ttk.Frame(image_frame)
        compare_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 原始图像区域
        original_frame = ttk.LabelFrame(compare_frame, text="原始图像")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=12)
        self.original_image_label = tk.Label(original_frame, text="原始图像将显示在这里", relief=tk.SUNKEN)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)  # 图像标签填满区域
        
        # 分割图像区域
        segmented_frame = ttk.LabelFrame(compare_frame, text="分割结果")
        segmented_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=12)
        self.segmented_image_label = tk.Label(segmented_frame, text="分割结果将显示在这里", relief=tk.SUNKEN)
        self.segmented_image_label.pack(fill=tk.BOTH, expand=True)  # 图像标签填满区域
        
        # 3. 计算结果（放在右侧区域下方）
        results_frame = ttk.LabelFrame(right_frame, text="计算结果")
        results_frame.pack(fill=tk.X, padx=5, pady=5)  # 仅水平填充，不抢占图像区域的垂直空间
        self.stats_text = scrolledtext.ScrolledText(results_frame, height=6, wrap=tk.WORD)  # 适当减小高度
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.insert(tk.END, "请先加载图像并执行分割")
        self.stats_text.config(state=tk.DISABLED)

        # 标签页控件（放在左侧控制面板内）
        self.notebook = ttk.Notebook(control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分割参数标签页
        self.segment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.segment_tab, text="分割参数")
        
        # 结果分析标签页
        self.analysis_tab = ttk.Frame(self.notebook)

        
        # 配置标签页内容（保持不变）
        self.setup_segment_tab()
        self.setup_analysis_tab()
    
    def setup_segment_tab(self):
        """设置分割参数标签页"""
        # 分割方法组（直接使用深度学习模型）
        method_frame = ttk.LabelFrame(self.segment_tab, text="分割方法")
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 分割方法选择
        self.segment_method_var = tk.StringVar(value="深度学习模型")
        segment_methods = ["深度学习模型"]
        method_combo = ttk.Combobox(method_frame, textvariable=self.segment_method_var, values=segment_methods, state="readonly", width=15)
        method_combo.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 显示模型状态标签（不再需要加载按钮）
        self.weight_path_var = tk.StringVar(value="正在加载模型...")
        self.weight_path_label = tk.Label(method_frame, textvariable=self.weight_path_var, anchor=tk.W)
        self.weight_path_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 深度学习分割参数组
        self.model_frame = ttk.LabelFrame(self.segment_tab, text="深度学习参数")
        self.model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 置信度阈值
        conf_frame = ttk.Frame(self.model_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(conf_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.pack(side=tk.LEFT, padx=5)
        conf_entry = ttk.Entry(conf_frame, textvariable=self.confidence_var, width=5)
        conf_entry.pack(side=tk.LEFT, padx=5)
    
    def setup_analysis_tab(self):
        """设置结果分析标签页"""
        # 创建图表区域
        self.figure_frame = ttk.Frame(self.analysis_tab)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图表
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # update_threshold_options方法已移除，不再需要传统方法
    
    def update_method_options(self, event):
        """根据选择的分割方法更新选项"""
        # 直接显示模型参数面板（只使用深度学习模型）
        self.threshold_frame.pack_forget()
        self.model_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def auto_load_model(self):
        """自动加载预训练模型"""
        # 预训练模型路径
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "epoch100-loss0.071.pth")
        
        try:
            # 检查模型文件是否存在
            if os.path.exists(model_path):
                # 使用model/unet.py中的Unet类创建实例
                self.model = Unet(model_path=model_path, num_classes=3, backbone='vgg16', cuda=torch.cuda.is_available())
                self.weight_path = model_path
                
                # 更新UI
                self.weight_path_var.set(f"已加载: {os.path.basename(model_path)}")
                print(f"成功加载预训练模型: {model_path}")
            else:
                # 如果模型文件不存在，使用默认的Unet实例（模拟模式）
                self.model = Unet(model_path="default_model", num_classes=3, backbone='vgg16', cuda=torch.cuda.is_available())
                self.weight_path = "default_model"
                self.weight_path_var.set("已加载默认模型")
                print(f"预训练模型文件不存在，使用默认模型: {model_path}")
        except Exception as e:
            self.weight_path_var.set(f"模型加载失败: {str(e)}")
            print(f"加载模型时出错: {str(e)}")
            self.model = None
            self.weight_path = None
    
    def load_weights(self):
        """加载深度学习模型权重文件"""
        # 这个方法可以保留，但在自动加载模型后使用较少
        file_path = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=[("PyTorch权重文件", "*.pth;*.pt")]
        )
        
        if file_path:
            try:
                # 使用model/unet.py中的Unet类创建实例
                self.model = Unet(model_path=file_path, num_classes=3, backbone='vgg16', cuda=torch.cuda.is_available())
                
                self.weight_path = file_path
                
                # 更新UI
                self.weight_path_var.set(f"已加载: {os.path.basename(file_path)}")
            except Exception as e:
                self.weight_path_var.set(f"模型加载失败: {str(e)}")
                print(f"加载权重文件时出错: {str(e)}")
                self.model = None
                self.weight_path = None
    
    def load_image(self):
        """加载图像文件，支持多种图像格式"""
        # 扩展支持的图像格式列表
        supported_formats = [
            "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif",
            "*.gif", "*.webp", "*.ppm", "*.pgm", "*.pbm", "*.pnm",
            "*.svg", "*.eps", "*.ps", "*.im", "*.dib", "*.jp2"
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[
                ("所有支持的图像文件", ";".join(supported_formats)),
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg;*.jpeg"),
                ("BMP图像", "*.bmp"),
                ("TIFF图像", "*.tiff;*.tif"),
                ("GIF图像", "*.gif"),
                ("WebP图像", "*.webp"),
                ("SVG矢量图", "*.svg"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 使用PIL作为主要的图像加载工具，支持更多格式
                pil_image = Image.open(file_path)
                
                # 转换为RGB格式（处理透明通道和灰度图）
                if pil_image.mode == 'RGBA':
                    # 处理透明通道，白色背景
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    rgb_image = background
                elif pil_image.mode == 'P':
                    # 调色板模式
                    rgb_image = pil_image.convert('RGB')
                elif pil_image.mode == 'L':
                    # 灰度图转RGB
                    rgb_image = pil_image.convert('RGB')
                else:
                    # 其他模式直接转RGB
                    rgb_image = pil_image.convert('RGB')
                
                # 转换为numpy数组
                self.original_image = np.array(rgb_image)
                
                # 如果OpenCV可用且需要使用OpenCV功能，转换为BGR格式
                if cv2_available:
                    # 注意：display_image方法期望RGB格式
                    # 这里保持为RGB格式，只在需要OpenCV处理时临时转换
                    pass
                
                # 显示图像
                self.display_image(self.original_image, self.original_image_label)
                
                # 启用分割按钮
                self.segment_button.config(state=tk.NORMAL)
                
                # 重置分割结果
                self.segmented_image = None
                self.stats = None
                self.segmented_image_label.config(text="分割结果将显示在这里")
                self.save_button.config(state=tk.DISABLED)
                
                # 重置统计信息
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, "请执行分割以查看统计信息")
                self.stats_text.config(state=tk.DISABLED)
                
                messagebox.showinfo("成功", f"图像加载成功: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
                print(f"加载图像错误详情: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    
    def start_segmentation(self):
        """开始图像分割处理"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        # 禁用按钮防止重复点击
        self.segment_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在处理...")
        
        # 检查是否已加载模型
        if self.model is None:
            messagebox.showwarning("警告", "请先加载深度学习模型权重文件")
            self.segment_button.config(state=tk.NORMAL)
            self.load_button.config(state=tk.NORMAL)
            self.status_label.config(text="就绪")
            return
            
        # 收集参数（只保留深度学习模型参数）
        params = {
            'segment_method': self.segment_method_var.get(),
            'confidence_threshold': self.confidence_var.get()
        }
        
        # 启动分割线程
        self.running = True
        self.worker_thread = threading.Thread(target=self.perform_segmentation, args=(params,))
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # 检查线程完成状态
        self.root.after(100, self.check_segmentation_complete)
    
    def perform_segmentation(self, params):
        """执行图像分割"""
        try:
            # 直接使用深度学习模型进行分割（移除传统方法）
            self._perform_deep_learning_segmentation(params)
        except Exception as e:
            self.error_message = str(e)
    
    # _perform_traditional_segmentation方法已移除，不再需要传统分割功能
    
    def _perform_deep_learning_segmentation(self, params):
        """使用深度学习模型进行分割"""
        if self.model is None:
            raise Exception("请先加载模型权重文件")
        
        # 预处理图像
        img = self.original_image.copy()
        
        # 转换为PIL Image格式
        img_pil = Image.fromarray(img)
        
        # 使用model/unet.py中的detect_image_ui方法进行分割
        # 该方法返回四个值：segmentation_map, labels, red_pixels, blue_pixels
        segmentation_map, labels, pred_red_pixels, pred_blue_pixels = self.model.detect_image_ui(img_pil)
        
        # 直接使用返回的彩色分割图，无需转换
        segmented_rgb = segmentation_map
        
        # 计算统计信息
        total_area = img.shape[0] * img.shape[1]
        blue_pixels = np.sum(np.all(segmented_rgb == [0, 0, 255], axis=2))  # 蓝色像素数(轨迹)
        red_pixels = np.sum(np.all(segmented_rgb == [255, 0, 0], axis=2))  # 红色像素数(颗粒)
        white_pixels = np.sum(np.all(segmented_rgb == [255, 255, 255], axis=2))  # 白色像素数(背景)
        
        # 计算颗粒占比：红色像素数/(红色像素数+蓝色像素数)
        particle_ratio = (red_pixels / (red_pixels + blue_pixels) * 100) if (red_pixels + blue_pixels) > 0 else 0
        
        # 设置结果
        self.segmented_image = segmented_rgb
        self.stats = {
            'particle_count': 0,  # 暂时设为0，需要根据模型输出调整
            'particle_area': red_pixels,
            'total_area': total_area,
            'particle_ratio': particle_ratio,
            'red_pixels': red_pixels,
            'blue_pixels': blue_pixels,
            'white_pixels': white_pixels,
            'properties': []
        }
    
    def check_segmentation_complete(self):
        """检查分割是否完成"""
        if self.worker_thread.is_alive():
            self.root.after(100, self.check_segmentation_complete)
        else:
            self.on_segmentation_complete()
    
    def on_segmentation_complete(self):
        """分割完成后的处理"""
        # 重新启用按钮
        self.segment_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)
        self.status_label.config(text="就绪")
        
        # 检查是否有错误
        if hasattr(self, 'error_message'):
            messagebox.showerror("错误", f"分割过程中出错: {self.error_message}")
            delattr(self, 'error_message')
            return
        
        if self.segmented_image is not None and self.stats is not None:
            # 显示分割结果
            self.display_image(self.segmented_image, self.segmented_image_label)
            
            # 更新统计信息
            self.update_stats_display()
            
            # 生成图表
            self.generate_charts()
            
            # 启用保存按钮
            self.save_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("成功", "图像分割完成")
    
    def display_image(self, image, label):
        """在标签上显示图像"""
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        
        # 获取标签大小
        label_width = label.winfo_width()
        label_height = label.winfo_height()
        
        # 如果标签还没有实际大小，使用默认大小
        if label_width < 100 or label_height < 100:
            label_width = 400
            label_height = 300
        
        # 计算调整后的图像大小，保持宽高比
        image_ratio = pil_image.width / pil_image.height
        label_ratio = label_width / label_height
        
        if image_ratio > label_ratio:
            # 图像更宽，以宽度为准
            new_width = label_width
            new_height = int(new_width / image_ratio)
        else:
            # 图像更高，以高度为准
            new_height = label_height
            new_width = int(new_height * image_ratio)
        
        # 调整图像大小
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为Tkinter可用的图像
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # 保存图像引用，防止被垃圾回收
        label.image = tk_image
        
        # 显示图像
        label.config(image=tk_image)
    
    def update_stats_display(self):
        """更新统计信息显示"""
        if self.stats:
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            
            # 显示统计信息
            self.stats_text.insert(tk.END, "===== 计算结果 =====\n")
            self.stats_text.insert(tk.END, f"颗粒数量: {self.stats.get('particle_count', 0)}\n")
            self.stats_text.insert(tk.END, f"颗粒面积: {self.stats.get('particle_area', 0)} 像素\n")
            self.stats_text.insert(tk.END, f"总区域: {self.stats.get('total_area', 0)} 像素\n")
            self.stats_text.insert(tk.END, f"颗粒占比: {self.stats.get('particle_ratio', 0):.2f}%\n")
            
            # 显示像素统计
            self.stats_text.insert(tk.END, f"\n===== 像素统计 =====\n")
            self.stats_text.insert(tk.END, f"红色像素(颗粒): {self.stats.get('red_pixels', 0)}\n")
            self.stats_text.insert(tk.END, f"蓝色像素(轨迹): {self.stats.get('blue_pixels', 0)}\n")
            self.stats_text.insert(tk.END, f"白色像素(背景): {self.stats.get('white_pixels', 0)}\n")
            
            self.stats_text.config(state=tk.DISABLED)
    
    def generate_charts(self):
        """生成分析图表"""
        if not self.stats or not self.stats['properties']:
            return
        
        # 清除现有图表
        self.figure.clear()
        
        # 提取数据
        areas = [prop.area for prop in self.stats['properties'] if prop.area >= self.min_area_var.get()]
        
        if areas:
            # 面积分布直方图
            ax1 = self.figure.add_subplot(121)
            ax1.hist(areas, bins=20, alpha=0.7, color='blue')
            ax1.set_xlabel('颗粒面积 (像素)')
            ax1.set_ylabel('数量')
            ax1.set_title('颗粒面积分布')
            ax1.grid(True, alpha=0.3)
            
            # 面积累积分布图
            ax2 = self.figure.add_subplot(122)
            sorted_areas = sorted(areas, reverse=True)
            cumulative = np.cumsum(sorted_areas) / np.sum(sorted_areas) * 100
            ax2.plot(range(1, len(cumulative) + 1), cumulative, 'r-')
            ax2.set_xlabel('颗粒数量 (从大到小)')
            ax2.set_ylabel('累积面积百分比 (%)')
            ax2.set_title('颗粒累积面积分布')
            ax2.grid(True, alpha=0.3)
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '未检测到符合条件的颗粒', 
                   horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_axis_off()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def save_results(self):
        """保存分割结果和统计信息"""
        if not self.segmented_image:
            messagebox.showwarning("警告", "没有可保存的分割结果")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*")]
        )
        
        if file_path:
            try:
                # 转换为BGR格式以保存
                bgr_image = cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_image)
                
                # 如果有统计信息，保存为文本文件
                if self.stats:
                    # 生成文本文件名
                    txt_path = '.'.join(file_path.split('.')[:-1]) + '_stats.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write("颗粒分析统计信息\n")
                        f.write("======================\n")
                        f.write(f"颗粒数量: {self.stats['particle_count']}\n")
                        f.write(f"颗粒总面积: {self.stats['particle_area']} 像素\n")
                        f.write(f"图像总面积: {self.stats['total_area']} 像素\n")
                        f.write(f"颗粒占比: {self.stats['particle_ratio']:.2f}%\n")
                
                messagebox.showinfo("成功", "结果已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存结果时出错: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
