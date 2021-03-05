#!/usr/bin/env python
# -*- coding: utf-8 -*-

from albumentations import Compose, HorizontalFlip, RandomBrightness, RandomCrop, RandomScale

def get_augmentations(cfg):

    processes = []

    if cfg.augmentation['random_scale']['is_applied']:
        processes.append(RandomScale(**cfg.augmentation['random_scale']['params']))
    
    if cfg.augmentation['random_crop']['is_applied']:
        processes.append(RandomCrop(**cfg.augmentation['random_crop']['params']))

    if cfg.augmentation['LRflip']['is_applied']:
        processes.append(HorizontalFlip(**cfg.augmentation['LRflip']['params']))
    
    if cfg.augmentation['brightness_shift']['is_applied']:
        processes.append(RandomBrightness(**cfg.augmentation['brightness_shift']['params']))
    
    return Compose(processes)
