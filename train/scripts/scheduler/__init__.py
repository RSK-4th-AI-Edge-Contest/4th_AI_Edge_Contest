#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scripts.scheduler.scheduler import CosineAnnealingWarmUpRestarts


def get_scheduler(optimizer, cfg):
    scheduler_type = cfg.scheduler['type']
    params = cfg.scheduler['params']

    if model_name not in key2model:
        raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented.')

    print(f'Scheduler: {scheduler_type}')
    return CosineAnnealingWarmUpRestarts(optimizer, **params)

