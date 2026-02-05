import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch_no_val
import os.path as osp
from config import *
from utils.iou_utils import calculate_mean_iou

MODEL_NAME = "unet_vgg16"
SAVE_PATH = osp.join(PROJECT_ROOT, "runs", MODEL_NAME)


def train_main(dataset_path=DATASET_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
               save_path=SAVE_PATH, num_classes=NUM_CLASSES, gpu_index=0, force_gpu=True):
    try:
        if not os.path.isabs(dataset_path):
            full_dataset_path = os.path.join(PROJECT_ROOT, dataset_path)
            if not os.path.exists(full_dataset_path):
                raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        else:
            full_dataset_path = dataset_path
            if not os.path.exists(full_dataset_path):
                raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        images_dir = osp.join(full_dataset_path, "Training_Images")
        labels_dir = osp.join(full_dataset_path, "Training_Labels")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {osp.join(dataset_path, 'Training_Images')}")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"标签目录不存在: {osp.join(dataset_path, 'Training_Labels')}")

        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception as e:
                raise RuntimeError(f"设置 GPU 设备失败 (gpu_index={gpu_index}): {e}")
            device = torch.device(f"cuda:{gpu_index}")
            gpu_name = torch.cuda.get_device_name(gpu_index)
            print(f"使用设备: {device} ({gpu_name})")
        else:
            if force_gpu:
                raise RuntimeError("未检测到可用的 CUDA GPU，请确认已安装支持 CUDA 的 PyTorch 与显卡驱动。")
            device = torch.device("cpu")
            print(f"使用设备: {device}")

        print(f"数据集路径: {dataset_path}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数: {epochs}")
        print(f"类别数量: {num_classes}")

    except Exception as e:
        print(f"参数检查错误: {e}")
        return

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    input_shape = [512, 512]
    dice_loss = True
    focal_loss = True

    cls_weights = np.ones([num_classes], np.float32)
    if num_classes >= 3:
        cls_weights[2] = 3.0

    num_workers = 0

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"已创建保存路径: {save_path}")
    except Exception as e:
        print(f"创建保存路径失败: {e}")
        return

    try:
        backbone = "vgg16"
        pretrained = True
        model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
        if not pretrained:
            weights_init(model)

        custom_weights_path = os.path.join(PROJECT_ROOT, "pretrained", "unet.pth")
        if os.path.exists(custom_weights_path):
            try:
                weights = torch.load(custom_weights_path, map_location=device)
                if isinstance(weights, dict) and 'state_dict' in weights:
                    model.load_state_dict(weights['state_dict'])
                    print(f"成功加载自定义权重: {custom_weights_path} (包含state_dict)")
                else:
                    model.load_state_dict(weights)
                    print(f"成功加载自定义权重: {custom_weights_path}")
            except Exception as e:
                print(f"加载自定义权重失败: {e}")
                print("将继续使用默认预训练权重")
        else:
            print(f"自定义权重文件不存在: {custom_weights_path}")
            print("将继续使用默认预训练权重")

        model_train = model.train().to(device)
        loss_history = LossHistory(save_path, val_loss_flag=False)
        print(f"模型加载成功，backbone: {backbone}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("提示: 如果是预训练模型问题，可以尝试设置pretrained=False")
        return

    try:
        train_images_dir = osp.join(full_dataset_path, "Training_Images")
        train_lines = [x.split(".")[0] for x in os.listdir(train_images_dir) if x.endswith(IMAGE_SUFFIX)]

        print(f"找到 {len(train_lines)} 张训练图像")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, full_dataset_path,
                                    IMAGE_SUFFIX, LABEL_SUFFIX)
        use_pin_memory = device.type == 'cuda'
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=use_pin_memory, drop_last=True, collate_fn=unet_dataset_collate)
        epoch_step = len(train_lines) // batch_size

        if epoch_step == 0:
            raise ValueError("无法加载数据集，请检查：\n1.是否是数据集过小。\n2.原图和标签数量及名称是否一致")

        print(f"数据加载成功，每轮迭代步数: {epoch_step}")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    miou_history = []
    miou_txt_path = os.path.join(save_path, 'miou_history.txt')

    try:
        with open(miou_txt_path, 'w') as f:
            f.write("Epoch,mIoU\n")
        print(f"已初始化mIoU记录文件: {miou_txt_path}")
    except Exception as e:
        print(f"创建mIoU记录文件失败: {e}")
        return

    try:
        for epoch in range(epochs):
            print(f"\n开始训练轮次 {epoch + 1}/{epochs}")
            try:
                fit_one_epoch_no_val(
                    model_train, model, loss_history, optimizer, epoch, epoch_step, gen, epochs, device,
                    dice_loss, focal_loss, cls_weights, num_classes, save_path=save_path, model_save_gap=10,
                    use_amp=(device.type == 'cuda')
                )

                try:
                    mean_ious = calculate_mean_iou(model, gen, device, num_classes)
                    mIoU = np.nanmean(mean_ious)
                    miou_history.append(mIoU)
                    print(f"Epoch {epoch + 1}/{epochs} - mIoU: {mIoU:.4f}")
                    print(f"各类别IoU: {[f'{iou:.4f}' for iou in mean_ious]}")

                    with open(miou_txt_path, 'a') as f:
                        f.write(f"{epoch + 1},{mIoU:.6f}\n")
                except Exception as e:
                    print(f"计算mIoU时出错: {e}")

                lr_scheduler.step()

            except KeyboardInterrupt:
                print("\n训练被用户中断")
                try:
                    torch.save(model.state_dict(), osp.join(save_path, f'interrupted_epoch{epoch + 1}.pth'))
                    print(f"已保存中断状态到 interrupted_epoch{epoch + 1}.pth")
                    if miou_history:
                        plot_miou_curve(miou_history, save_path)
                except:
                    pass
                return
            except Exception as e:
                print(f"训练轮次 {epoch + 1} 出错: {e}")
                continue
    except Exception as e:
        print(f"训练过程发生严重错误: {e}")
        return

    try:
        if miou_history:
            plot_miou_curve(miou_history, save_path)
            print(f"训练完成，mIoU数据已保存至 {miou_txt_path}")
            print(f"标签2权重已设置为 {cls_weights[2]}，模型训练时会优先关注该类别分割精度")
        else:
            print("训练完成，但没有mIoU历史记录可供绘制")
    except Exception as e:
        print(f"绘制mIoU曲线时出错: {e}")
        print(f"训练已完成，但无法绘制mIoU曲线")


def plot_miou_curve(miou_history, save_path):
    epochs = range(1, len(miou_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, miou_history, marker='o', linestyle='-', color='blue',
             linewidth=2, label='Mean IoU (mIoU)')

    plt.title('Mean Intersection over Union (mIoU) vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('mIoU', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    miou_curve_path = osp.join(save_path, 'mIoU_curve.png')
    plt.savefig(miou_curve_path, dpi=300)
    plt.close()
    print(f"mIoU曲线已保存至 {miou_curve_path}")


if __name__ == '__main__':
    print("===== 医学图像分割训练程序 ====\n")
    try:
        train_main(epochs=EPOCHS)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n===== 训练程序结束 =====")

