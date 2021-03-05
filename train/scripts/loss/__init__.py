#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import CrossEntropyLoss

from scripts.loss.loss import DiceLoss, CE_DiceLoss, FocalLoss, IoULoss, InverseFrequencyLoss
from scripts.loss.lovasz_losses import lovasz_softmax

key2loss = {
    "CrossEntropy": CrossEntropyLoss,
    "Dice": DiceLoss,
    "CE_Dice": CE_DiceLoss,
    "Focal": FocalLoss,
    "IoU": IoULoss,
    "InverseFrequency": InverseFrequencyLoss,
    "Lovasz": lovasz_softmax
    }


def get_criterion(cfg):
    loss_name = cfg.criterion['type']

    if loss_name not in key2loss:
        raise NotImplementedError(f'Loss {loss_name} is not implemented.')
    
    print(f'Loss function: {loss_name}')
    
    if loss_name == 'Lovasz':
        return lovasz_softmax
        
    return key2loss[loss_name]()