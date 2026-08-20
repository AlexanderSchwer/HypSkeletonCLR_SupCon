#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor, add_lr_scheduler_args
from .pretrain_skeletonclr import SkeletonCLR_Processor

import wandb

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

class SkeletonCLR_3views_Processor(SkeletonCLR_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_joint_value = []
        loss_motion_value = []
        loss_bone_value = []
        sink_loss_value = []
        hier_loss_value = []
        cluster_distance_diagnostics = self._new_cluster_distance_diagnostics()
        embedding_snapshot = (
            self._new_embedding_snapshot(loader)
            if self._should_plot_embeddings(epoch)
            else None
        )

        if self._wandb_ok:
            try:
                wandb.watch(self.model)
            except Exception as exc:
                self._wandb_ok = False
                print(f"W&B disabled during watch due to error: {exc}")
        # wandb.watch(self.model, log="all") # for logging of parameters panels

        for [data1, data2], label in loader:
            self.global_step += 1

            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            model_output = self.model(data1, data2)
            output, output_motion, output_bone, target, features_sup, cluster_pack = (
                self._parse_model_output(model_output)
            )
            self._accumulate_embedding_snapshot(
                embedding_snapshot,
                features_sup,
                label,
                output,
            )

            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output.size(0))
            else:
                self.model.update_ptr(output.size(0))
            loss_joint = self.loss(output, target)
            loss_motion = self.loss(output_motion, target)
            loss_bone = self.loss(output_bone, target)

            loss_base = loss_joint + loss_motion + loss_bone
            loss = loss_base
            loss_sink, loss_hier, cluster_metrics = self._compute_cluster_losses(cluster_pack)
            self._accumulate_cluster_distance_diagnostics(
                cluster_distance_diagnostics,
                cluster_pack,
            )
            sink_weight = self._ramp_weight(
                self.arg.lambda_sink,
                self.arg.cluster_warmup_steps,
                self.arg.cluster_ramp_steps,
            )
            hier_weight = self._ramp_weight(
                self.arg.lambda_hier,
                self.arg.hier_warmup_steps,
                self.arg.hier_ramp_steps,
            )
            if loss_sink is not None:
                loss = loss + sink_weight * loss_sink
            if loss_hier is not None:
                loss = loss + hier_weight * loss_hier

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_base'] = loss_base.data.item()
            self.iter_info['loss_joint'] = loss_joint.data.item()
            self.iter_info['loss_motion'] = loss_motion.data.item()
            self.iter_info['loss_bone'] = loss_bone.data.item()
            self.iter_info['lambda_sink_effective'] = sink_weight
            self.iter_info['lambda_hier_effective'] = hier_weight
            self.iter_info.update(cluster_metrics)
            if loss_sink is not None:
                self.iter_info['loss_sink'] = loss_sink.data.item()
                sink_loss_value.append(self.iter_info['loss_sink'])
            elif 'loss_sink' in self.iter_info:
                del self.iter_info['loss_sink']

            if loss_hier is not None:
                self.iter_info['loss_hier'] = loss_hier.data.item()
                hier_loss_value.append(self.iter_info['loss_hier'])
            elif 'loss_hier' in self.iter_info:
                del self.iter_info['loss_hier']

            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            loss_joint_value.append(self.iter_info['loss_joint'])
            loss_motion_value.append(self.iter_info['loss_motion'])
            loss_bone_value.append(self.iter_info['loss_bone'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_writer.add_scalar('batch_loss_joint', self.iter_info['loss_joint'], self.global_step)
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)

            if self.global_step % self.arg.log_interval == 0:
                # Log metrics to wandb
                payload = {
                    "loss": loss.data.item(),
                    "loss_base": loss_base.data.item(),
                    "loss_joint": loss_joint.data.item(),
                    "loss_motion": loss_motion.data.item(),
                    "loss_bone": loss_bone.data.item(),
                    "lambda_sink_effective": sink_weight,
                    "lambda_hier_effective": hier_weight,
                    "learning_rate": self.lr,
                    "epoch": epoch}
                payload.update(cluster_metrics)
                if loss_sink is not None:
                    payload["loss_sink"] = loss_sink.data.item()
                if loss_hier is not None:
                    payload["loss_hier"] = loss_hier.data.item()
                self._safe_wandb_log(payload, step=self.global_step)
            
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_joint']= np.mean(loss_joint_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        if sink_loss_value:
            self.epoch_info['train_mean_loss_sink'] = np.mean(sink_loss_value)
            self.train_writer.add_scalar('loss_sink', self.epoch_info['train_mean_loss_sink'], epoch)
        if hier_loss_value:
            self.epoch_info['train_mean_loss_hier'] = np.mean(hier_loss_value)
            self.train_writer.add_scalar('loss_hier', self.epoch_info['train_mean_loss_hier'], epoch)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_joint', self.epoch_info['train_mean_loss_joint'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)

        # Log epoch-level mean loss
        epoch_payload = {
            "train_mean_loss": np.mean(loss_value),
            "train_mean_loss_joint": np.mean(loss_joint_value),
            "train_mean_loss_motion": np.mean(loss_motion_value),
            "train_mean_loss_bone": np.mean(loss_bone_value),
            "learning_rate": self.lr,
            "epoch": epoch}
        if sink_loss_value:
            epoch_payload["train_mean_loss_sink"] = np.mean(sink_loss_value)
        if hier_loss_value:
            epoch_payload["train_mean_loss_hier"] = np.mean(hier_loss_value)
        self._safe_wandb_log(epoch_payload, step=self.global_step)

        self._log_cluster_distance_diagnostics(epoch, cluster_distance_diagnostics)

        if embedding_snapshot is not None:
            self._render_embedding_snapshot(epoch, embedding_snapshot)

        self.show_epoch_info()

    @staticmethod
    def _parse_model_output(model_output):
        if not isinstance(model_output, (list, tuple)):
            raise ValueError("Model output must be tuple/list")
        if len(model_output) == 4:
            output, output_motion, output_bone, target = model_output
            return output, output_motion, output_bone, target, None, None
        if len(model_output) == 6:
            output, output_motion, output_bone, target, features_sup, cluster_pack = model_output
            return output, output_motion, output_bone, target, features_sup, cluster_pack
        raise ValueError(f"Unexpected number of outputs from model: {len(model_output)}")

    @staticmethod
    def get_parser(add_help=False):
        parser = SkeletonCLR_Processor.get_parser(add_help=add_help)
        parser.set_defaults(view='all')
        return parser
