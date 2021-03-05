#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import get_model
from scripts.optimizer import get_optimizer
from scripts.scheduler import get_scheduler
from scripts.loss import get_criterion
from scripts.evaluation import calculate_iou

class Trainer:
    def __init__(self, writer, cfg):
        self.writer = writer
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = get_model(cfg).to(self.device)
        self.opt = get_optimizer(self.model.parameters(), cfg)
        self.scheduler = get_scheduler(self.opt, cfg)
        self.criterion = get_criterion(cfg)

        self.class_names = self.cfg.class_names
        self.num_classes = self.cfg.num_classes

    def fit(self, train_loader, val_loader):
        best_miou = -1.
        for i in tqdm(range(self.cfg.epoch), desc='Epoch'):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f'[train] {i+1}')
            self.model.train()
            for j, (x, t) in pbar:
                x = x.to(self.device)
                t = t.to(self.device)

                self.opt.zero_grad()
                y = self.model(x)

                loss = self.criterion(y, t)
                loss.backward()
                self.opt.step()

                pred = y.data.max(1)[1]
                acc = pred.eq(t).sum().item() / torch.numel(pred)

                self.writer.add_scalar('train/loss', loss.item(), i * len(train_loader) + j)
                self.writer.add_scalar('train/acc', acc, i * len(train_loader) + j)

                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc:.3f}'})

            pbar.close()

            # validate
            self.model.eval()
            iou = np.zeros((len(val_loader), self.num_classes), dtype=np.float32)
            val_loss_sum = 0
            val_acc_sum = 0
            for k, (x, t) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False, desc=f'[validation] {i+1}'):
                x = x.to(self.device)
                t = t.to(self.device)

                pred = self.model(x)

                currrent_loss = self.criterion(pred, t)
                val_loss_sum += currrent_loss.item()

                pred = pred.data.max(1)[1]
                val_acc_sum += pred.eq(t).sum().item() / torch.numel(pred)

                pred = pred.cpu().numpy().astype(np.uint8)
                t = t.cpu().numpy().astype(np.uint8)

                iou[k] = calculate_iou(pred, t, self.num_classes)

            self.writer.add_scalar('val/loss', val_loss_sum / len(val_loader), i+1)
            self.writer.add_scalar('val/acc', val_acc_sum / len(val_loader), i+1)

            iou_per_image = np.nanmean(iou[:, :-1], axis=1) 
            miou = np.nanmean(iou_per_image)
            class_iou = np.nanmean(iou, axis=0)
            for name, score in zip(self.class_names, class_iou):
                self.writer.add_scalars('IoU', {name: score}, i+1)
            self.writer.add_scalars('IoU', {'mIoU': miou}, i+1)

            if miou > best_miou:
                best_miou = miou
                torch.save(self.model.state_dict(), f'{self.cfg.save_dir}/models/epoch_{i+1}.pt')

            self.scheduler.step()
        
        print(f'total {len(train_loader) * self.cfg.epoch} loop finish.')
