#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml


class Config:
    def __init__(self, config_filepath):

        self.config_filepath = config_filepath
        with open(self.config_filepath, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.model = cfg['model']
        self.data = cfg['data']
        self.training = cfg['training']
        self.augmentation = cfg['augmentation']
        self.optimizer = cfg['optimizer']
        self.scheduler = cfg['scheduler']
        self.criterion = cfg['criterion']

        self.class_names = ['Car',
                            'Pedestrian',
                            'Signal',
                            'Lane',
                            'Other']
        self.num_classes = len(self.class_names)
        self.save_dir = None