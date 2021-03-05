#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.hardnet_pytorch import HarDNet

key2model = {
    'HarDNet': HarDNet
}


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict


def get_model(cfg):
    model_name = cfg.model['type']
    in_channels = cfg.model['in_channels']
    num_classes = cfg.num_classes
    
    if model_name not in key2model:
        raise NotImplementedError(f'Model {model_name} is not implemented.')

    print(f'Model: {model_name}')

    if pretrained == 'cityscapes':
        state = torch.load('hardnet70_cityscapes_model.pkl')['model_state']
        state_dict = convert_state_dict(state)
        model = key2model[model_name](in_channels=3, out_channels=19)
        model.load_state_dict(state_dict)
        model.finalConv = nn.Conv2d(in_channels=48, out_channels=num_classes, kernel_size=1,
                                    stride=1, padding=0, bias=True)
        return model

    return key2model[model_name](in_channels=in_channels, out_channels=num_classes)
    