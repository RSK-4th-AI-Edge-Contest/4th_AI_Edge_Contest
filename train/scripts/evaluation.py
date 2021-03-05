#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def calculate_iou(pred, gt, n_class):
    iou = np.zeros((n_class,), dtype=np.float32)
    smooth = 1e-10

    for i in range(n_class):
        pred_ind = pred == i
        gt_ind = gt == i

        intersection = pred_ind & gt_ind
        union = pred_ind | gt_ind
        
        if gt_ind.sum():
            iou[i] = intersection.sum() / (union.sum() + smooth)
        else:
            iou[i] = np.nan
    
    return iou
