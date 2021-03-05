#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim import Adam, AdamW, SGD
from torch_optimizer import RAdam

key2opt = {
    "Adam": Adam,
    "AdamW": AdamW,
    "RAdam": RAdam,
    "SGD": SGD
    }


def get_optimizer(model_params, cfg):
    opt_type = cfg.optimizer['type']
    params = cfg.optimizer['params']
    
    if opt_type not in key2opt:
        raise NotImplementedError(f'Optimizer {opt_type} is not implemented.')

    if opt_type == "SGD":
        print('Optimizer: SGD(momentum)')
        return key2opt[opt_type](model_params, **params)
    
    params.pop('momentum')

    print(f'Optimizer: {opt_type}')

    return key2opt[opt_type](model_params, **params)
    