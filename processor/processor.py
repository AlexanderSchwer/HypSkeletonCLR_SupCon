#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import os
import numpy as np
import random
import math

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
from .work_dir import (
    WORK_DIR_LAYOUTS,
    WORK_DIR_MODES,
    build_canonical_work_dir,
    prepare_training_work_dir,
)
from tensorboardX import SummaryWriter


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.configure_work_dir()
        self.init_environment()

        if self.arg.phase == 'train':
            self.train_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'val'), 'val')
        else:
            self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'test'), 'test')

        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

        self.global_step = 0

    def configure_work_dir(self):
        if self.arg.phase != 'train':
            return

        requested_work_dir = self.arg.work_dir
        if self.arg.work_dir_layout == 'canonical' and self.arg.work_dir_mode != 'resume':
            self.arg.work_dir = build_canonical_work_dir(
                self.arg,
                processor_name=self.__class__.__name__)

        self.arg.work_dir = prepare_training_work_dir(
            self.arg.work_dir,
            mode=self.arg.work_dir_mode,
            run_id=self.arg.run_id,
            run_subdir=self.arg.run_subdir)
        if self.arg.work_dir != os.path.normpath(requested_work_dir):
            print('Training output directory: {}'.format(self.arg.work_dir))

    def train_log_writer(self, epoch):
        self.train_writer.add_scalar('batch_loss', self.iter_info['loss'], self.global_step)
        self.train_writer.add_scalar('lr', self.lr, self.global_step)
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

    def eval_log_writer(self, epoch):
        self.val_writer.add_scalar('eval_loss', self.eval_info['eval_mean_loss'], epoch)
        self.val_writer.add_scalar('current_result', self.current_result, epoch)
        self.val_writer.add_scalar('best_result', self.best_result, epoch)

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        self.data_loader = dict()

        if self.arg.train_feeder_args:
            train_feeder = import_class(self.arg.train_feeder)
            train_feeder_args = self._resolve_feeder_paths(self.arg.train_feeder_args)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_feeder(**train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                pin_memory=True,    # set True when memory is abundant
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,
                worker_init_fn=init_seed)
                
        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)
            test_feeder_args = self._resolve_feeder_paths(self.arg.test_feeder_args)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_feeder(**test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False,
                worker_init_fn=init_seed)

    def _resolve_single_dataset_path(self, path):
        if not isinstance(path, str) or not path:
            return path
        normalized = os.path.expandvars(os.path.expanduser(path))
        if os.path.isabs(normalized):
            return normalized
        if self.arg.dataset_root:
            root = os.path.expandvars(os.path.expanduser(self.arg.dataset_root))
            return os.path.normpath(os.path.join(root, normalized))
        return normalized

    def _resolve_feeder_paths(self, feeder_args):
        resolved_args = dict(feeder_args)
        for key in ('data_path', 'label_path'):
            if key in resolved_args:
                resolved_args[key] = self._resolve_single_dataset_path(resolved_args[key])
        return resolved_args

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_eval_info(self):
        for k, v in self.eval_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('eval', self.meta_info['iter'], self.eval_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['train_mean_loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.eval_info['test_mean_loss'] = 1
        self.show_eval_info()

    def after_train_epoch(self, epoch):
        pass

    def print_networks(self, net, print_flag=False):
        self.io.print_log('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if print_flag:
            self.io.print_log(net)
        self.io.print_log('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        self.io.print_log('-----------------------------------------------')

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)

        # training phase
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.meta_info['iter'] = self.global_step
            self.best_result = 0.0
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch + 1

                # training
                self.io.print_log('Training epoch: {}'.format(epoch + 1))
                self.train(epoch + 1)
                self.after_train_epoch(epoch + 1)

                # save model
                if self.arg.save_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if self.arg.eval_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch + 1))
                    self.test(epoch + 1)
                    self.io.print_log("current %.2f%%, best %.2f%%" % 
                                            (self.current_result, self.best_result))

                    # save best model
                    filename = 'epoch%.3d_acc%.2f_model.pt' % (epoch + 1, self.current_result)
                    self.io.save_model(self.model, filename)
                    if self.current_result >= self.best_result:
                        filename = 'best_model.pt'
                        self.io.save_model(self.model, filename)

        # test phase
        elif self.arg.phase == 'test':
            
            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            self.best_result = 0.0

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test(1)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('--work_dir_layout', default='canonical', choices=WORK_DIR_LAYOUTS,
                            help='canonical derives the experiment directory from run arguments; manual uses work_dir as given')
        parser.add_argument('--work_dir_mode', default='auto', choices=WORK_DIR_MODES,
                            help='auto creates a unique run directory; error fails if work_dir exists; resume uses work_dir exactly')
        parser.add_argument('--run_id', default=None,
                            help='optional run id used below work_dir/run_subdir when work_dir_mode is auto')
        parser.add_argument('--run_subdir', default='runs',
                            help='subdirectory below work_dir for automatic run directories')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#epoch)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#epoch)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
        parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
        parser.add_argument('--num_worker', type=int, default=0, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--dataset_root', default='', help='root directory for relative data_path/label_path in feeder args')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable

        return parser
