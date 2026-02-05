import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import os.path as osp

def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes, save_path, model_save_gap=10, use_amp=False):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    model_train.train()
    scaler = GradScaler(enabled=use_amp)
    # print('Start Train')
    with tqdm(total=epoch_step, desc=f'Train-Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model_train(imgs)
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    if dice_loss:
                        loss = loss + Dice_loss(outputs, labels)
            else:
                outputs = model_train(imgs)
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    # print('Finish Train')
    model_train.eval()
    # print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Val-Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

                if use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model_train(imgs)
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                        if dice_loss:
                            loss = loss + Dice_loss(outputs, labels)
                else:
                    outputs = model_train(imgs)
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    if dice_loss:
                        loss = loss + Dice_loss(outputs, labels)
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)
                val_loss += loss.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'dice_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
    current_epoch = epoch + 1
    if current_epoch % model_save_gap == 0:
        torch.save(model.state_dict(),
                   osp.join(save_path, 'epoch%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1))))

    # print('Finish Validation')
    # print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    # print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    # torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
    # (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, device, dice_loss,
                         focal_loss, cls_weights, num_classes, save_path, model_save_gap, use_amp=False):
    """
    :param model_train:   训练模式的模型
    :param model:         模型结构
    :param loss_history:  损失历史记录
    :param optimizer:     优化器
    :param epoch:         训练的轮数，当前的epoch
    :param epoch_step:    训练的长度
    :param gen:           训练的数据集
    :param Epoch:         总的训练轮数
    :param device:        训练的设备
    :param dice_loss:     是否开启dice loss
    :param focal_loss:    是否开启focal loss
    :param cls_weights:   训练的类别权重
    :param num_classes:   总的训练的类别数目
    :param save_path:     模型需要保存的路径
    :return:
    """
    total_loss = 0
    total_f_score = 0
    scaler = GradScaler(enabled=use_amp)
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                imgs = imgs.to(device, non_blocking=True)
                pngs = pngs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model_train(imgs)
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    if dice_loss:
                        loss = loss + Dice_loss(outputs, labels)
            else:
                outputs = model_train(imgs)
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)
            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_f_score += _f_score.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'dice_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    loss_history.append_loss(total_loss / (epoch_step + 1))
    current_epoch = epoch + 1
    if current_epoch % model_save_gap == 0:
        torch.save(model.state_dict(), osp.join(save_path, 'epoch%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1))))
