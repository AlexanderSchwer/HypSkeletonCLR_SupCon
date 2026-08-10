#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    StepLR,
)

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

import geoopt as gt


LR_SCHEDULERS = (
    'auto',
    'none',
    'multistep',
    'step',
    'exponential',
    'cosine',
    'cosine_warm_restarts',
)


def add_lr_scheduler_args(parser):
    parser.add_argument('--lr_scheduler', default='auto', choices=LR_SCHEDULERS,
                        help='learning-rate scheduler: auto maps step to multistep')
    parser.add_argument('--lr_milestones', type=int, default=[], nargs='+',
                        help='epoch milestones for MultiStepLR; defaults to step when empty')
    parser.add_argument('--lr_step_size', type=int, default=250,
                        help='step size in epochs for StepLR')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='multiplicative decay factor for StepLR, MultiStepLR, and ExponentialLR')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='linear warmup epochs before the selected scheduler')
    parser.add_argument('--lr_warmup_start_factor', type=float, default=0.001,
                        help='initial LR factor for LinearLR warmup')
    parser.add_argument('--lr_eta_min', type=float, default=None,
                        help='minimum learning rate for cosine schedulers; null defaults to 0.01 * base_lr')
    parser.add_argument('--lr_t_max', type=int, default=None,
                        help='T_max for CosineAnnealingLR; null defaults to min(50, num_epoch - lr_warmup_epochs)')
    parser.add_argument('--lr_t_0', type=int, default=10,
                        help='initial cycle length for CosineAnnealingWarmRestarts')
    parser.add_argument('--lr_t_mult', type=int, default=1,
                        help='cycle length multiplier for CosineAnnealingWarmRestarts')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class PT_Processor(Processor):
    """
        Processor for Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RSGD':
            self.optimizer = gt.optim.RiemannianSGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.lr_scheduler = self._build_lr_scheduler()
        self.lr = self._current_lr()
        
    def adjust_lr_scheduler(self):
        self.adjust_lr()

    def adjust_lr(self):
        self.lr = self._current_lr()

    def after_train_epoch(self, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _build_lr_scheduler(self):
        scheduler_name = self._lr_scheduler_name()
        warmup_epochs = max(0, int(getattr(self.arg, 'lr_warmup_epochs', 0)))
        main_scheduler = self._build_main_lr_scheduler(scheduler_name, warmup_epochs)

        if warmup_epochs == 0:
            return main_scheduler

        warmup = LinearLR(
            self.optimizer,
            start_factor=float(self.arg.lr_warmup_start_factor),
            total_iters=warmup_epochs,
        )
        if main_scheduler is None:
            return warmup
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_epochs],
        )

    def _build_main_lr_scheduler(self, scheduler_name, warmup_epochs):
        if scheduler_name == 'none':
            return None

        if scheduler_name == 'multistep':
            milestones = self._lr_milestones(warmup_epochs)
            if not milestones:
                return None
            return MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'step':
            step_size = max(1, int(self.arg.lr_step_size))
            return StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'exponential':
            return ExponentialLR(
                self.optimizer,
                gamma=float(self.arg.lr_gamma),
            )

        if scheduler_name == 'cosine':
            eta_min = self._lr_eta_min()
            t_max = self._lr_t_max(warmup_epochs)
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            schedule_epochs = self._lr_schedule_epochs(warmup_epochs)
            if t_max >= schedule_epochs:
                return scheduler
            eta_min_factor = eta_min / float(self.arg.base_lr)
            return SequentialLR(
                self.optimizer,
                schedulers=[
                    scheduler,
                    LambdaLR(self.optimizer, lr_lambda=lambda _: eta_min_factor),
                ],
                milestones=[t_max],
            )

        if scheduler_name == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, int(self.arg.lr_t_0)),
                T_mult=max(1, int(self.arg.lr_t_mult)),
                eta_min=self._lr_eta_min(),
            )

        raise ValueError('Unknown lr_scheduler: {}'.format(scheduler_name))

    def _lr_scheduler_name(self):
        scheduler_name = str(getattr(self.arg, 'lr_scheduler', 'auto')).lower()
        scheduler_name = scheduler_name.replace('-', '_')

        if scheduler_name == 'auto':
            if getattr(self.arg, 'step', None):
                return 'multistep'
            return 'none'

        if scheduler_name not in LR_SCHEDULERS:
            raise ValueError('Unknown lr_scheduler: {}'.format(scheduler_name))
        return scheduler_name

    def _lr_milestones(self, warmup_epochs):
        milestones = list(getattr(self.arg, 'lr_milestones', None) or [])
        if not milestones:
            milestones = list(getattr(self.arg, 'step', None) or [])
        if warmup_epochs == 0:
            return [int(value) for value in milestones]
        return [max(1, int(value) - warmup_epochs) for value in milestones if int(value) > warmup_epochs]

    def _lr_eta_min(self):
        eta_min = getattr(self.arg, 'lr_eta_min', None)
        if eta_min is None:
            return float(self.arg.base_lr) * 0.01
        return float(eta_min)

    def _lr_t_max(self, warmup_epochs):
        t_max = getattr(self.arg, 'lr_t_max', None)
        if t_max is not None:
            return max(1, int(t_max))
        return min(50, self._lr_schedule_epochs(warmup_epochs))

    def _lr_schedule_epochs(self, warmup_epochs):
        return max(1, int(self.arg.num_epoch) - warmup_epochs)

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            output, target = self.model(data1, data2)
            loss = self.loss(output, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        add_lr_scheduler_args(parser)
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
