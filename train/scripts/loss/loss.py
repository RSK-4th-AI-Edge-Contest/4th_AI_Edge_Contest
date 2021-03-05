#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1e-10):
        n_classes = inputs.size()[1]

        if self.weight is None:
            self.weight = [1.] * n_classes

        inputs = F.softmax(inputs, dim=1)

        loss = torch.zeros(n_classes, dtype=torch.float, device=inputs.device)
        for class_ind, w in enumerate(self.weight):
            dice_target = (targets == class_ind).float()
            dice_output = inputs[:, class_ind, ...]

            num_preds = dice_target.long().sum()

            if num_preds == 0:
                loss[class_ind] = 0.
            else:
                intersection = (dice_output * dice_target).sum()
                dice = 2. * (intersection + smooth) / (dice_output.sum() + dice_target.sum() + smooth)
                loss[class_ind] = (1. - dice) * w

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()


class IoULoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(IoULoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1e-10):
        n_classes = inputs.size()[1]

        if self.weight is None:
            self.weight = [1.] * n_classes

        inputs = F.softmax(inputs, dim=1)

        loss = torch.zeros(n_classes, dtype=torch.float, device=inputs.device)
        for class_ind, w in enumerate(self.weight):
            jaccard_target = (targets == class_ind).float()
            jaccard_output = inputs[:, class_ind, ...]

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[class_ind] = 0.
            else:
                intersection = (jaccard_output * jaccard_target).sum()
                total = (jaccard_output + jaccard_target).sum()
                union = total - intersection
                iou = (intersection + smooth) / (union + smooth)
                loss[class_ind] = (1. - iou) * w

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-10):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target, self.smooth)
        return CE_loss + dice_loss


class MyCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(MyCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        n_classes = inputs.size()[1]

        inputs = inputs.permute(0,2,3,1).contiguous()
        inputs = inputs.view(-1, n_classes)
        targets = targets.view(-1)
        targets_onehot = F.one_hot(targets)

        if self.weight is None:
            weight = torch.ones(targets.size()[0], dtype=torch.float, device=inputs.device)
        else:
            weight = self.weight[targets]

        loss = weight * (-targets_onehot * inputs).sum(dim=1)

        if self.weight is not None:
            return loss.sum() / weight.sum()

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()


class InverseFrequencyLoss(nn.Module):
    def __init__(self):
        super(InverseFrequencyLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        n_classes = inputs.size()[1]

        inputs = inputs.permute(0,2,3,1).contiguous()
        inputs = inputs.view(-1, n_classes)
        targets = targets.view(-1)
        targets_onehot = F.one_hot(targets)

        num_perclass = []
        for class_ind in range(n_classes):
            cnt = (targets == class_ind).sum().float().item()
            if cnt:
                num_perclass.append(1. / cnt)
            else:
                num_perclass.append(1.)

        num_perclass = torch.tensor(num_perclass)
        weight = num_perclass[targets]
        loss = weight * (-targets_onehot * inputs).sum(dim=1)

        return loss.sum() / weight.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        n_classes = inputs.size()[1]

        inputs = inputs.permute(0,2,3,1).contiguous()
        inputs = inputs.view(-1, n_classes)
        targets = targets.view(-1)
        targets_onehot = F.one_hot(targets)

        if self.weight is None:
            weight = torch.ones(targets.size()[0], dtype=torch.float, device=inputs.device)
        else:
            weight = self.weight[targets]

        loss = -weight * (targets_onehot * torch.pow(1. - inputs, self.gamma) *  torch.log(inputs)).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()


if __name__ == '__main__':
    import torch
    torch.manual_seed(0)

    GPU = False
    device = torch.device("cuda" if GPU else "cpu")

    pred = 2 * torch.rand(2, 10, 100, 100) - 1
    mask = torch.randint(0, 10, (2, 100, 100))
    pred = pred.to(device)
    mask = mask.to(device)

    weight = torch.rand(10)

    criterion1 = IoULoss()
    criterion2 = DiceLoss()
    criterion3 = CE_DiceLoss()
    criterion4 = nn.CrossEntropyLoss()
    criterion5 = MyCrossEntropyLoss()
    criterion6 = FocalLoss()
    criterion7 = InverseFrequencyLoss()

    iou_loss = criterion1(pred, mask)
    dice_loss = criterion2(pred, mask)
    ce_dice_loss = criterion3(pred, mask)
    ce_loss = criterion4(pred, mask)
    my_ce_loss = criterion5(pred, mask)
    focal_loss = criterion6(pred, mask)
    inverse_frequency_loss = criterion7(pred, mask)

    print(f'IoU_loss: {iou_loss.item()}')
    print(f'Dice_loss: {dice_loss.item()}')
    print(f'CE_loss: {ce_loss.item()}')
    print(f'CE_Dice_loss: {ce_dice_loss.item()}')
    print(f'my_CE_loss: {my_ce_loss.item()}')
    print(f'Focal_loss: {focal_loss.item()}')
    print(f'Inverse_Frequency_loss: {inverse_frequency_loss.item()}')
