import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def dice_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap="OrRd")
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2), ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    plt.close()


def compute_mIoU(
    gt_dir, pred_dir, png_name_list, num_classes, name_classes,
    image_suffix, label_suffix, color_tolerance=10  # 新增：颜色容差参数，默认10（0-255范围）
):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    gt_imgs = [join(gt_dir, x + "." + image_suffix) for x in png_name_list]
    pred_imgs = [join(pred_dir, x + "." + label_suffix) for x in png_name_list]

    # --------------------------修改1：颜色映射包含容差范围--------------------------
    # 格式：(中心颜色(R,G,B), 类别)，后续会自动计算容差范围内的颜色
    color_map_with_tolerance = [
        ((255, 255, 255), 0),  # 背景（白色），容差范围内的近白色均匹配
        ((0, 0, 255), 1),      # 类别1（蓝色）
        ((255, 0, 0), 2)       # 类别2（红色）
    ]
    # ------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # 辅助函数：将三通道RGB图像转为单通道类别索引（0/1/2）
    # 辅助函数：将三通道RGB图像转为单通道类别索引（0/1/2），支持颜色容差
    def rgb_to_class(rgb_img):
        h, w, _ = rgb_img.shape
        class_img = np.zeros((h, w), dtype=np.uint8)  # 默认为0（可根据需求设为背景或未知）

        for (center_r, center_g, center_b), cls in color_map_with_tolerance:
            # 计算每个通道的容差范围：[center - tolerance, center + tolerance]
            # 确保范围在0-255之间（避免越界）
            r_min = max(0, center_r - color_tolerance)
            r_max = min(255, center_r + color_tolerance)
            g_min = max(0, center_g - color_tolerance)
            g_max = min(255, center_g + color_tolerance)
            b_min = max(0, center_b - color_tolerance)
            b_max = min(255, center_b + color_tolerance)

            # 生成当前类别的掩码：三通道均在容差范围内
            mask = (
                    (rgb_img[:, :, 0] >= r_min) & (rgb_img[:, :, 0] <= r_max) &
                    (rgb_img[:, :, 1] >= g_min) & (rgb_img[:, :, 1] <= g_max) &
                    (rgb_img[:, :, 2] >= b_min) & (rgb_img[:, :, 2] <= b_max)
            )
            class_img[mask] = cls  # 匹配的区域赋值为当前类别
        return class_img

    dice_total = []
    for ind in range(len(gt_imgs)):
        # --------------------------核心修改2：读取并转换三通道标签--------------------------
        try:
            # 强制转为RGB模式，避免灰度图/RGBA格式导致的通道问题
            pred_rgb = np.array(Image.open(pred_imgs[ind]).convert('RGB'))
            label_rgb = np.array(Image.open(gt_imgs[ind]).convert('RGB'))
        except Exception as e:
            print(f"读取图像失败：{e}，跳过该样本")
            continue

        # 检查尺寸是否匹配（宽高必须相同）
        if pred_rgb.shape != label_rgb.shape:
            print(f'Skipping: 尺寸不匹配 {pred_imgs[ind]} vs {gt_imgs[ind]}')
            continue

        # 将RGB转为类别索引（0/1/2）
        pred = rgb_to_class(pred_rgb)
        label = rgb_to_class(label_rgb)
        # ----------------------------------------------------------------------------------

        # --------------------------核心修改3：删除255的硬编码转换--------------------------
        # 三通道标签中255是颜色分量，无需转换（原代码针对单通道标签的特殊处理）
        # ----------------------------------------------------------------------------------

        # --------------------------核心修改4：逐类别计算Dice系数--------------------------
        class_dice = []
        for cls in range(num_classes):
            # 提取当前类别的掩码
            pred_cls = (pred == cls).astype(np.float32)
            label_cls = (label == cls).astype(np.float32)
            # 处理空类别（预测和标签均无该类像素时，Dice记为1.0）
            if (pred_cls.sum() + label_cls.sum()) < 1e-6:
                class_dice.append(1.0)
            else:
                dice = (2. * (pred_cls * label_cls).sum() + 1e-5) / (pred_cls.sum() + label_cls.sum() + 1e-5)
                class_dice.append(dice)
        dice_total.append(np.mean(class_dice))  # 单张图像的平均Dice
        # ----------------------------------------------------------------------------------

        # 累加混淆矩阵
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # 中间输出
        if ind > 0 and ind % 10 == 0:
            print(
                f'{ind}/{len(gt_imgs)}: mIou-{100 * np.nanmean(per_class_iu(hist)):.2f}%; mPA-{100 * np.nanmean(per_class_PA_Recall(hist)):.2f}%; Accuracy-{100 * per_Accuracy(hist):.2f}%')

    # 计算全局指标
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    dice_final = np.nanmean(np.array(dice_total)) if dice_total else np.nan

    # 逐类别输出结果
    for ind_class in range(num_classes):
        print(
            f'===>{name_classes[ind_class]}: Iou-{IoUs[ind_class] * 100:.2f}%; Recall-{PA_Recall[ind_class] * 100:.2f}%; Precision-{Precision[ind_class] * 100:.2f}%')
    print(
        f'===> mIoU: {np.nanmean(IoUs) * 100:.2f}%; mPA: {np.nanmean(PA_Recall) * 100:.2f}%; Accuracy: {per_Accuracy(hist) * 100:.2f}%; Dice score: {dice_final * 100:.3f}%')
    return np.array(hist, int), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) if val >= 1.0 else " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)
    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, plt_show=False,
                 class_names=[]):
    os.makedirs(miou_out_path, exist_ok=True)
    draw_plot_func(IoUs, name_classes, f"mIoU = {np.nanmean(IoUs) * 100:.2f}%", "Intersection over Union",
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size, plt_show)
    draw_plot_func(PA_Recall, name_classes, f"mPA = {np.nanmean(PA_Recall) * 100:.2f}%", "Pixel Accuracy",
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size, plt_show)
    draw_plot_func(PA_Recall, name_classes, f"mRecall = {np.nanmean(PA_Recall) * 100:.2f}%", "Recall",
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size, plt_show)
    draw_plot_func(Precision, name_classes, f"mPrecision = {np.nanmean(Precision) * 100:.2f}%", "Precision",
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size, plt_show)

    # 绘制混淆矩阵热力图
    heat_maps = hist
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    heat_maps_float = np.divide(heat_maps, heat_maps_sum, out=np.zeros_like(heat_maps, dtype=float),
                                where=heat_maps_sum != 0)
    show_heatmaps("Confusion Matrix", class_names, class_names, heat_maps_float,
                  os.path.join(miou_out_path, "Confusion_matrix.png"))