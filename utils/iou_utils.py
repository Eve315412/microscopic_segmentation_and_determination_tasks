# iou_utils.py
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp


# 计算单个图像的IoU
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.flatten()
    target = target.flatten()

    # 对每个类别计算IoU
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        # 计算交集和并集
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        # 避免除零
        if union == 0:
            ious.append(float('nan'))  # 或0，根据需求
        else:
            ious.append(float(intersection) / float(union))

    return ious


# 计算数据集的平均IoU - 修改版本
def calculate_mean_iou(model, dataloader, device, num_classes):
    model.eval()
    all_ious = []

    with torch.no_grad():
        for batch in dataloader:
            # 处理可能包含多个元素的批次
            # 假设图像在第一个位置，标签在第二个位置
            images = batch[0]
            targets = batch[1]

            # 转换为PyTorch张量并移至设备
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float().to(device)
            else:
                images = images.to(device)

            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets).long().to(device)
            else:
                targets = targets.to(device)

            # 获取预测结果
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 转换为numpy数组进行计算
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            # 计算每个样本的IoU
            for i in range(len(preds)):
                ious = calculate_iou(preds[i], targets[i], num_classes)
                all_ious.append(ious)

    # 计算每个类别的平均IoU
    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)  # 忽略NaN值

    return mean_ious


# 绘制IoU曲线并保存
def plot_iou_curve(iou_history, save_path, num_classes):
    epochs = range(1, len(iou_history) + 1)

    plt.figure(figsize=(10, 6))


    # 绘制平均IoU曲线
    mean_iou = [np.nanmean(epoch_iou) for epoch_iou in iou_history]
    plt.plot(epochs, mean_iou, marker='s', linestyle='--', color='black', label='Mean IoU')

    plt.title('Intersection over Union (IoU) by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    iou_path = osp.join(save_path, 'iou_curve.png')
    plt.savefig(iou_path)
    plt.close()

    print(f"IoU curve saved to {iou_path}")