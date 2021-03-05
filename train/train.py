#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import glob
import os
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.config import Config
from scripts.ai_edge_contest_loader import SetEdgeContestData
from scripts.trainer import Trainer

learning_condition_filepath = 'configs/learning_condition.yaml'

random.seed(0)
torch.manual_seed(0)


def main():
    today = datetime.date.today().strftime('%Y%m%d')
    num_log_data = len(glob.glob(f'logs/{today}_??'))
    log_dir = f'logs/{today}_{num_log_data:02}'

    if not os.path.exists(f'{log_dir}/models'):
        os.makedirs(f'{log_dir}/models')

    writer = SummaryWriter(log_dir=log_dir)
    cfg = Config(learning_condition_filepath)
    cfg.save_dir = log_dir

    train_dataset = SetEdgeContestData(cfg, split='train')
    val_dataset = SetEdgeContestData(cfg, split='val')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.training['train_batch_size'],
                                  shuffle=True,
                                  num_workers=cfg.training['num_workers'])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.training['val_batch_size'],
                                shuffle=False,
                                num_workers=cfg.training['num_workers'])

    trainer = Trainer(writer, cfg)
    trainer.fit(train_dataloader, val_dataloader)
    writer.close()


if __name__ == "__main__":
    main()
