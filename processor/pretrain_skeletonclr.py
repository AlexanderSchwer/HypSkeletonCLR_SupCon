#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import os
import subprocess
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor, add_lr_scheduler_args
from .wandb_utils import init_wandb_from_work_dir

from tools.losses import SupConLoss
from tools.hyperbolic_hierarchy import (
    prototype_affinity_hyp,
    update_affinity_ema,
    sample_triplets_from_affinity,
    hierarchy_triplet_loss_hyp,
)
from tools.pseudo_labeling import pseudo_label_mask_from_posteriors
from tools.hyperbolic_embedding_plot import (
    DEFAULT_NEGATIVE_DISTANCE_SAMPLES,
    default_hierarchy_plot_classes,
    format_class_hierarchies,
    render_embedding_diagnostics,
)
from tools.action_label_hierarchy import hierarchy_leaf_ids

import wandb

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

CONTRASTIVE_MODES = ('augmentation', 'supervised', 'pseudo_hard', 'pseudo_soft')


class SkeletonCLR_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_affinity = None
        self.contrastive_schedule = self._normalize_contrastive_schedule()
        self.arg.contrastive_schedule = self.contrastive_schedule
        
        # Initialize wandb run
        self._wandb_ok = not self.arg.wandb_disabled
        self._wandb_run = None
        self._wandb_run_dir = None
        if self._wandb_ok:
            try:
                self._wandb_run = init_wandb_from_work_dir(
                    self.arg,
                    job_type="pretrain",
                    config=vars(self.arg),
                )
                if self._wandb_run is not None and self._wandb_run.dir:
                    self._wandb_run_dir = os.path.dirname(self._wandb_run.dir)
            except Exception as exc:
                self._wandb_ok = False
                print(f"W&B disabled during init due to error: {exc}")

        self.criterion = SupConLoss(temperature=self.arg.temperature, curvature=self.arg.curvature)

    def start(self):
        try:
            return super().start()
        finally:
            self._finish_and_sync_wandb()

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
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
        
        '''
        label_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0,
                        10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 3, 16: 3, 17: 0, 18: 0, 19: 0,
                        20: 0, 21: 3, 22: 0, 23: 1, 24: 0, 25: 1, 26: 1, 27: 0, 28: 0, 29: 0,
                        30: 0, 31: 0, 32: 0, 33: 0, 34: 2, 35: 2, 36: 0, 37: 0, 38: 0, 39: 0,
                        40: 2, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 0, 49: 0,
                        50: 1, 51: 0, 52: 0, 53: 0, 54: 3, 55: 0, 56: 0, 57: 0, 58: 1, 59: 1}
        '''
        '''
        label_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1,
                        10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 16: 0, 17: 1, 18: 0, 19: 1,
                        20: 0, 21: 1, 22: 0, 23: 1, 24: 0, 25: 1, 26: 0, 27: 1, 28: 0, 29: 1,
                        30: 0, 31: 1, 32: 0, 33: 1, 34: 0, 35: 1, 36: 0, 37: 1, 38: 0, 39: 1,
                        40: 0, 41: 1, 42: 0, 43: 1, 44: 0, 45: 1, 46: 0, 47: 1, 48: 0, 49: 1,
                        50: 0, 51: 1, 52: 0, 53: 1, 54: 0, 55: 1, 56: 0, 57: 1, 58: 0, 59: 1}
        '''

        for [data1, data2], label in loader:
            self.global_step += 1

            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            #data1 = poincare_ball.expmap0(data1)
            #data2 = poincare_ball.expmap0(data2)

            if self.arg.view == 'joint':
                pass
            elif self.arg.view == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
            elif self.arg.view == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
                
                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                
                data1 = bone1
                data2 = bone2
            else:
                raise ValueError

            # forward
            model_output = self.model(data1, data2)
            output, target, features_sup, cluster_pack = self._parse_model_output(model_output)
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

            loss_base, contrastive_metrics = self._compute_scheduled_contrastive_loss(
                epoch=epoch,
                features_sup=features_sup,
                cluster_pack=cluster_pack,
                output=output,
                target=target,
                labels=label,
            )
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
            self.iter_info.update(contrastive_metrics)
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
            self.show_iter_info()
            self.meta_info['iter'] += 1

            if self.global_step % self.arg.log_interval == 0:
                # Log metrics to wandb
                payload = {
                    "loss": loss.data.item(),
                    "loss_base": loss_base.data.item(),
                    "contrastive_mode": contrastive_metrics["contrastive_mode"],
                    "lambda_sink_effective": sink_weight,
                    "lambda_hier_effective": hier_weight,
                    "learning_rate": self.lr,
                    "epoch": epoch}
                payload.update(cluster_metrics)
                payload.update(contrastive_metrics)
                if loss_sink is not None:
                    payload["loss_sink"] = loss_sink.data.item()
                if loss_hier is not None:
                    payload["loss_hier"] = loss_hier.data.item()
                self._safe_wandb_log(payload, step=self.global_step)
            
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        if sink_loss_value:
            self.epoch_info['train_mean_loss_sink'] = np.mean(sink_loss_value)
            self.train_writer.add_scalar('loss_sink', self.epoch_info['train_mean_loss_sink'], epoch)
        if hier_loss_value:
            self.epoch_info['train_mean_loss_hier'] = np.mean(hier_loss_value)
            self.train_writer.add_scalar('loss_hier', self.epoch_info['train_mean_loss_hier'], epoch)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)

        print(f"Contrastive mode: {self._format_active_contrastive_modes(epoch)}")
        
        # Log epoch-level mean loss
        epoch_payload = {
            "train_mean_loss": np.mean(loss_value),
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
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--temperature', type=float, default=0.07, help='the temperature used in supervised training loss')
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincaré ball')
        parser.add_argument('--contrastive_mode', default=None, choices=CONTRASTIVE_MODES, help='fallback contrastive objective when contrastive_schedule is omitted')
        parser.add_argument('--contrastive_schedule', default=None, help='list of epoch phases with mode, start_epoch, end_epoch, and optional transition_epochs')
        parser.add_argument('--lambda_sink', type=float, default=1.0, help='maximum weight for Sinkhorn clustering loss')
        parser.add_argument('--cluster_warmup_steps', type=int, default=1000, help='MoCo-only iterations before Sinkhorn loss')
        parser.add_argument('--cluster_ramp_steps', type=int, default=2000, help='iterations used to ramp Sinkhorn loss weight')
        parser.add_argument('--cluster_distance_log_interval', type=int, default=1, help='log cluster distance diagnostics every N epochs; 0 disables')
        parser.add_argument('--cluster_distance_matrix_max_clusters', type=int, default=20, help='maximum number of clusters for full inter-cluster distance matrix logging')
        parser.add_argument('--lambda_hier', type=float, default=0.1, help='maximum weight for hyperbolic hierarchy loss')
        parser.add_argument('--hier_update_interval', type=int, default=200, help='interval of iterations for hierarchy loss')
        parser.add_argument('--hier_warmup_steps', type=int, default=3000, help='warmup iterations before hierarchy loss')
        parser.add_argument('--hier_ramp_steps', type=int, default=2000, help='iterations used to ramp hierarchy loss weight')
        parser.add_argument('--hier_triplets', type=int, default=512, help='number of hierarchy triplets sampled each update')
        parser.add_argument('--hier_margin', type=float, default=0.05, help='triplet margin for hierarchy loss')
        parser.add_argument('--affinity_momentum', type=float, default=0.9, help='EMA momentum for cluster affinity')
        parser.add_argument('--affinity_temperature', type=float, default=1.0, help='temperature for prototype affinity')
        parser.add_argument('--lambda_aug', type=float, default=1.0, help='positive weight for augmentation pairs in SupCon modes')
        parser.add_argument('--lambda_pseudo', type=float, default=None, help='maximum positive weight for pseudo-label pairs in pseudo SupCon modes')
        parser.add_argument('--lambda_pseudo_supcon', type=float, default=None, help='deprecated alias for lambda_pseudo')
        parser.add_argument('--pseudo_supcon_warmup_steps', type=int, default=0, help='iterations before pseudo-label positive weights are enabled')
        parser.add_argument('--pseudo_supcon_ramp_steps', type=int, default=0, help='iterations used to ramp pseudo-label positive weights')
        parser.add_argument('--pseudo_supcon_confidence_threshold', type=float, default=0.8, help='minimum prototype posterior confidence for pseudo-label positives')
        parser.add_argument('--pseudo_supcon_assignment_source', default='q_k', choices=['p_q', 'p_k', 'q_k', 'p_mean'], help='cluster posterior used to form pseudo labels')
        parser.add_argument('--wandb_offline', type=str2bool, default=False, help='log W&B offline and automatically sync the run when the script exits')
        parser.add_argument('--wandb_disabled', type=str2bool, default=False, help='disable W&B init, logging, finishing, and sync completely')
        parser.add_argument('--embedding_plot_interval', type=int, default=0, help='render embedding diagnostic plots every N epochs; 0 disables live plotting')
        parser.add_argument('--embedding_plot_max_samples', type=int, default=1024, help='maximum epoch samples retained for each embedding plot')
        parser.add_argument('--embedding_plot_methods', default=['logmap_pca_disk', 'hyp_tsne'], nargs='+', help='projection methods: pca, svd, tsne, logmap_pca, logmap_pca_disk, logmap_tsne, hyp_tsne')
        parser.add_argument('--embedding_plot_selected_labels', type=int, default=[], nargs='+', help='class labels to plot in embedding diagnostics; use -1 for all classes; empty uses hierarchy defaults')
        parser.add_argument('--embedding_plot_color_by', default=['class_group'], nargs='+', choices=['class', 'class_group'], help='color embedding diagnostics by one or more modes: class or class_group')
        parser.add_argument('--embedding_plot_class_groups', action=DictAction, default=dict(), help='mapping from group names to class-label lists')
        parser.add_argument('--embedding_plot_class_names', action=DictAction, default=dict(), help='mapping from class labels to semantic class names')
        parser.add_argument('--embedding_plot_reference_hierarchy', default='', help='stored hierarchy name used to fill embedding_plot_class_groups and embedding_plot_class_names when omitted, e.g. hypskeletonclr_ward')
        parser.add_argument('--embedding_plot_hierarchy', type=str2bool, default=True, help='render class-prototype hierarchy diagnostics with embedding plots')
        parser.add_argument('--embedding_plot_hierarchy_linkages', default=['ward_tangent'], nargs='+', help='class hierarchy linkages to render: ward_tangent, single_hyperbolic, complete_hyperbolic, average_hyperbolic, weighted_hyperbolic, or all')
        parser.add_argument('--embedding_plot_hierarchy_log_max_merges', type=int, default=20, help='maximum hierarchy merge rows written to the training log; 0 logs all')
        
        # endregion yapf: enable

        return parser

    @staticmethod
    def _parse_model_output(model_output):
        if not isinstance(model_output, (list, tuple)):
            raise ValueError("Model output must be tuple/list")
        if len(model_output) == 3:
            output, target, features_sup = model_output
            return output, target, features_sup, None
        if len(model_output) == 4:
            output, target, features_sup, cluster_pack = model_output
            return output, target, features_sup, cluster_pack
        raise ValueError(f"Unexpected number of outputs from model: {len(model_output)}")

    def _compute_cluster_losses(self, cluster_pack):
        if cluster_pack is None:
            return None, None, {}
        if not isinstance(cluster_pack, dict):
            raise ValueError("cluster_pack must be a dict when provided")

        p_q = cluster_pack.get("p_q", None)
        q_k = cluster_pack.get("q_k", None)
        proto_h = cluster_pack.get("proto_h", None)

        if q_k is None:
            return None, None, {}

        if p_q is not None:
            log_prob = p_q.clamp_min(1e-12).log()
        else:
            return None, None, {}

        q_k = q_k.detach()
        loss_sink = None
        if self.global_step >= self.arg.cluster_warmup_steps:
            loss_sink = -(q_k * log_prob).sum(dim=1).mean()

        loss_hier = None
        hierarchy_triplet_accuracy = None
        if proto_h is not None:
            batch_affinity = prototype_affinity_hyp(
                proto_h.detach(),
                curvature=self.arg.curvature,
                temperature=self.arg.affinity_temperature,
            )
            self.cluster_affinity = update_affinity_ema(
                self.cluster_affinity,
                batch_affinity,
                momentum=self.arg.affinity_momentum,
            )

        if (
            self.cluster_affinity is not None
            and proto_h is not None
            and self.global_step >= self.arg.hier_warmup_steps
            and self.global_step % self.arg.hier_update_interval == 0
        ):
            triplets = sample_triplets_from_affinity(self.cluster_affinity, self.arg.hier_triplets)
            if triplets.numel() > 0:
                loss_hier = hierarchy_triplet_loss_hyp(
                    proto_h,
                    triplets,
                    curvature=self.arg.curvature,
                    margin=self.arg.hier_margin,
                )
                hierarchy_triplet_accuracy = self._hierarchy_triplet_accuracy(
                    proto_h, triplets
                )

        metrics = self._cluster_metrics(
            q_k,
            proto_h,
            self.cluster_affinity,
            hierarchy_triplet_accuracy,
        )
        return loss_sink, loss_hier, metrics

    def _compute_scheduled_contrastive_loss(
        self,
        epoch,
        features_sup,
        cluster_pack=None,
        output=None,
        target=None,
        labels=None,
        stream_outputs=None,
    ):
        active_phases = self._active_contrastive_phases(epoch)
        total_loss = None
        metrics = {
            "contrastive_mode": self._format_contrastive_phases(active_phases),
            "contrastive_schedule_weight_sum": sum(weight for _, weight in active_phases),
        }

        for phase_index, (phase, phase_weight) in enumerate(active_phases):
            phase_loss, phase_metrics = self._compute_contrastive_mode_loss(
                mode=phase["mode"],
                phase=phase,
                features_sup=features_sup,
                cluster_pack=cluster_pack,
                output=output,
                target=target,
                labels=labels,
                stream_outputs=stream_outputs,
            )
            weighted_loss = float(phase_weight) * phase_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

            metrics[f"contrastive_phase_{phase_index}_mode"] = phase["mode"]
            metrics[f"contrastive_phase_{phase_index}_weight"] = float(phase_weight)
            metrics[f"contrastive_phase_{phase_index}_loss"] = phase_loss.detach().item()
            metrics[f"contrastive_weight_{phase['mode']}"] = (
                metrics.get(f"contrastive_weight_{phase['mode']}", 0.0)
                + float(phase_weight)
            )

            if len(active_phases) == 1:
                metrics.update(phase_metrics)
            else:
                for key, value in phase_metrics.items():
                    if key == "contrastive_mode":
                        continue
                    metrics[f"contrastive_phase_{phase_index}_{key}"] = value
                    if key in ("loss_joint", "loss_motion", "loss_bone"):
                        metrics[key] = value

        return total_loss, metrics

    def _compute_contrastive_mode_loss(
        self,
        features_sup,
        cluster_pack=None,
        output=None,
        target=None,
        labels=None,
        stream_outputs=None,
        mode=None,
        phase=None,
    ):
        mode = mode or getattr(self.arg, 'contrastive_mode', None) or 'augmentation'
        phase = phase or {}
        lambda_aug = float(self._phase_value(phase, 'lambda_aug', self.arg.lambda_aug))
        metrics = {
            "contrastive_mode": mode,
            "lambda_aug_effective": lambda_aug,
            "lambda_pseudo_effective": 0.0,
        }

        if mode == "augmentation":
            if stream_outputs is None:
                if output is None or target is None:
                    raise ValueError("augmentation mode requires output and target")
                return self.loss(output, target), metrics

            losses = [self.loss(stream_output, target) for stream_output in stream_outputs]
            if len(losses) == 3:
                metrics.update({
                    "loss_joint": losses[0].detach().item(),
                    "loss_motion": losses[1].detach().item(),
                    "loss_bone": losses[2].detach().item(),
                })
            return sum(losses), metrics

        if features_sup is None:
            raise ValueError(f"{mode} mode requires SupCon features from the model")

        if mode == "supervised":
            if labels is None:
                raise ValueError("supervised mode requires dataset labels")
            return self.criterion(features_sup, labels), metrics

        if cluster_pack is None:
            raise ValueError(f"{mode} mode requires cluster_enabled=True and cluster outputs")
        if not isinstance(cluster_pack, dict):
            raise ValueError("cluster_pack must be a dict when provided")

        assignment_source = self._phase_value(
            phase,
            'pseudo_supcon_assignment_source',
            self.arg.pseudo_supcon_assignment_source,
        )
        posteriors = self._pseudo_supcon_posteriors(cluster_pack, assignment_source)
        if posteriors is None:
            raise ValueError(
                f"{mode} mode requires posterior source "
                f"{assignment_source!r}"
            )
        if posteriors.size(0) != features_sup.size(0):
            raise ValueError(
                "pseudo SupCon posteriors and features must have matching batch "
                f"sizes, got {posteriors.size(0)} and {features_sup.size(0)}"
            )

        lambda_pseudo_max = self._phase_value(phase, 'lambda_pseudo', None)
        if lambda_pseudo_max is None:
            lambda_pseudo_max = self._max_lambda_pseudo()

        pseudo_weight = self._ramp_weight(
            lambda_pseudo_max,
            self._phase_value(
                phase,
                'pseudo_supcon_warmup_steps',
                self.arg.pseudo_supcon_warmup_steps,
            ),
            self._phase_value(
                phase,
                'pseudo_supcon_ramp_steps',
                self.arg.pseudo_supcon_ramp_steps,
            ),
        )
        mask, pseudo_labels, confidence, confident = pseudo_label_mask_from_posteriors(
            posteriors,
            confidence_threshold=self._phase_value(
                phase,
                'pseudo_supcon_confidence_threshold',
                self.arg.pseudo_supcon_confidence_threshold,
            ),
            mode=mode,
            lambda_aug=lambda_aug,
            lambda_pseudo=pseudo_weight,
        )
        metrics.update(
            self._pseudo_supcon_metrics(mask, pseudo_labels, confidence, confident)
        )
        metrics["lambda_pseudo_effective"] = pseudo_weight

        return self.criterion(features_sup, mask=mask), metrics

    def _pseudo_supcon_posteriors(self, cluster_pack, source=None):
        source = source or self.arg.pseudo_supcon_assignment_source
        if source == 'p_mean':
            p_q = cluster_pack.get('p_q', None)
            p_k = cluster_pack.get('p_k', None)
            if p_q is None or p_k is None:
                return None
            return 0.5 * (p_q.detach() + p_k.detach())

        posteriors = cluster_pack.get(source, None)
        if posteriors is None:
            return None
        return posteriors.detach()

    @classmethod
    def _valid_contrastive_modes(cls):
        return CONTRASTIVE_MODES

    def _normalize_contrastive_schedule(self):
        schedule = getattr(self.arg, 'contrastive_schedule', None)
        fallback_mode = getattr(self.arg, 'contrastive_mode', None)

        if schedule is None:
            schedule = [{
                "mode": fallback_mode or "augmentation",
                "start_epoch": 1,
                "end_epoch": int(self.arg.num_epoch),
            }]
        elif fallback_mode is not None:
            raise ValueError(
                "Use either contrastive_schedule or contrastive_mode, not both"
            )
        elif isinstance(schedule, str):
            parsed_schedule = yaml.safe_load(schedule)
            schedule = parsed_schedule

        if isinstance(schedule, dict):
            schedule = [schedule]
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("contrastive_schedule must be a non-empty list")

        normalized = []
        for index, phase in enumerate(schedule):
            if not isinstance(phase, dict):
                raise ValueError("each contrastive_schedule phase must be a dict")
            normalized_phase = dict(phase)
            mode = normalized_phase.get("mode")
            if mode not in self._valid_contrastive_modes():
                raise ValueError(
                    "contrastive_schedule phase {} has invalid mode {!r}; "
                    "expected one of {}".format(
                        index,
                        mode,
                        self._valid_contrastive_modes(),
                    )
                )

            if "start_epoch" not in normalized_phase or "end_epoch" not in normalized_phase:
                raise ValueError(
                    "each contrastive_schedule phase needs start_epoch and end_epoch"
                )
            start_epoch = int(normalized_phase["start_epoch"])
            end_epoch = int(normalized_phase["end_epoch"])
            transition_epochs = int(normalized_phase.get("transition_epochs", 0))

            if start_epoch < 1:
                raise ValueError("contrastive_schedule start_epoch must be >= 1")
            if end_epoch < start_epoch:
                raise ValueError("contrastive_schedule end_epoch must be >= start_epoch")
            if transition_epochs < 0:
                raise ValueError("contrastive_schedule transition_epochs must be >= 0")
            if transition_epochs > (end_epoch - start_epoch + 1):
                raise ValueError(
                    "contrastive_schedule transition_epochs must not exceed phase length"
                )
            if index == 0 and transition_epochs:
                raise ValueError("first contrastive_schedule phase cannot transition in")

            normalized_phase["mode"] = mode
            normalized_phase["start_epoch"] = start_epoch
            normalized_phase["end_epoch"] = end_epoch
            normalized_phase["transition_epochs"] = transition_epochs
            normalized.append(normalized_phase)

        normalized.sort(key=lambda item: item["start_epoch"])
        expected_start = 1
        max_epoch = int(self.arg.num_epoch)
        for phase in normalized:
            if phase["start_epoch"] != expected_start:
                raise ValueError(
                    "contrastive_schedule must cover epochs without gaps or overlaps; "
                    f"expected start_epoch {expected_start}, got {phase['start_epoch']}"
                )
            expected_start = phase["end_epoch"] + 1
            if phase["start_epoch"] > max_epoch:
                raise ValueError("contrastive_schedule starts after num_epoch")
            if expected_start > max_epoch:
                break
        if expected_start <= max_epoch:
            raise ValueError(
                "contrastive_schedule must cover all epochs through num_epoch; "
                f"missing epochs {expected_start}-{max_epoch}"
            )

        return normalized

    def _active_contrastive_phases(self, epoch):
        schedule = getattr(self, "contrastive_schedule", None)
        if schedule is None:
            schedule = self._normalize_contrastive_schedule()

        for index, phase in enumerate(schedule):
            if phase["start_epoch"] <= epoch <= phase["end_epoch"]:
                transition_epochs = phase.get("transition_epochs", 0)
                if index > 0 and transition_epochs > 0:
                    transition_end = phase["start_epoch"] + transition_epochs - 1
                    if epoch <= transition_end:
                        progress = (
                            epoch - phase["start_epoch"] + 1
                        ) / float(transition_epochs + 1)
                        previous_phase = schedule[index - 1]
                        return [
                            (previous_phase, 1.0 - progress),
                            (phase, progress),
                        ]
                return [(phase, 1.0)]

        raise ValueError(f"No contrastive_schedule phase covers epoch {epoch}")

    def _format_active_contrastive_modes(self, epoch):
        return self._format_contrastive_phases(self._active_contrastive_phases(epoch))

    @staticmethod
    def _format_contrastive_phases(active_phases):
        if len(active_phases) == 1:
            return active_phases[0][0]["mode"]
        return "+".join(
            "{}:{:.3f}".format(phase["mode"], weight)
            for phase, weight in active_phases
        )

    @staticmethod
    def _phase_value(phase, key, default):
        if phase is not None and key in phase:
            return phase[key]
        return default

    def _max_lambda_pseudo(self):
        if self.arg.lambda_pseudo is not None:
            return self.arg.lambda_pseudo
        if self.arg.lambda_pseudo_supcon is not None:
            return self.arg.lambda_pseudo_supcon
        return 1.0

    @staticmethod
    def _pseudo_supcon_metrics(mask, pseudo_labels, confidence, confident):
        with torch.no_grad():
            batch_size = mask.size(0)
            identity = torch.eye(batch_size, dtype=torch.bool, device=mask.device)
            extra_positive_pairs = ((mask > 0) & ~identity).sum()
            possible_extra_pairs = max(1, batch_size * (batch_size - 1))
            active_clusters = (
                int(pseudo_labels[confident].unique().numel())
                if confident.any()
                else 0
            )
            return {
                "pseudo_supcon_confident_ratio": confident.float().mean().item(),
                "pseudo_supcon_confidence_mean": confidence.mean().item(),
                "pseudo_supcon_active_clusters": active_clusters,
                "pseudo_supcon_extra_positive_pairs": int(extra_positive_pairs.item()),
                "pseudo_supcon_extra_positive_density": (
                    extra_positive_pairs.float() / float(possible_extra_pairs)
                ).item(),
            }

    def _ramp_weight(self, maximum, warmup_steps, ramp_steps):
        if self.global_step < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return float(maximum)
        progress = min(1.0, (self.global_step - warmup_steps) / float(ramp_steps))
        return float(maximum) * progress

    def _cluster_metrics(
        self,
        q_k,
        proto_h,
        cluster_affinity=None,
        hierarchy_triplet_accuracy=None,
    ):
        with torch.no_grad():
            assignment_entropy = -(
                q_k * q_k.clamp_min(1e-12).log()
            ).sum(dim=1).mean()
            usage = q_k.sum(dim=0)
            usage = usage / usage.sum().clamp_min(1e-12)
            metrics = {
                "assignment_entropy": assignment_entropy.item(),
                "cluster_usage_min": usage.min().item(),
                "cluster_usage_max": usage.max().item(),
            }
            if proto_h is not None:
                manifold = gt.PoincareBall(self.arg.curvature)
                proto_depth = manifold.dist0(proto_h)
                metrics.update({
                    "prototype_depth_mean": proto_depth.mean().item(),
                    "prototype_depth_max": proto_depth.max().item(),
                })
            if cluster_affinity is not None:
                affinity = cluster_affinity.clamp_min(1e-12)
                metrics["affinity_entropy"] = (
                    -(cluster_affinity * affinity.log()).sum().item()
                )
            if hierarchy_triplet_accuracy is not None:
                metrics["hierarchy_triplet_accuracy"] = hierarchy_triplet_accuracy
            return metrics

    def _new_cluster_distance_diagnostics(self):
        return {
            "sample_count": 0,
            "num_clusters": None,
            "intra_sum": None,
            "intra_sq_sum": None,
            "intra_count": None,
            "proto_h": None,
        }

    def _accumulate_cluster_distance_diagnostics(self, diagnostics, cluster_pack):
        if diagnostics is None or cluster_pack is None:
            return
        dist_k_proto = cluster_pack.get("dist_k_proto", None)
        assign_k = cluster_pack.get("assign_k", None)
        proto_h = cluster_pack.get("proto_h", None)
        if dist_k_proto is None or assign_k is None or proto_h is None:
            return

        with torch.no_grad():
            dist_k_proto = dist_k_proto.detach()
            assign_k = assign_k.detach().long()
            assigned_dist = dist_k_proto.gather(1, assign_k.view(-1, 1)).view(-1)
            finite_mask = torch.isfinite(assigned_dist)
            if not finite_mask.any():
                diagnostics["proto_h"] = proto_h.detach()
                return

            assigned_dist = assigned_dist[finite_mask]
            assign_k = assign_k[finite_mask]
            num_clusters = int(proto_h.size(0))
            device = assigned_dist.device

            if diagnostics["num_clusters"] != num_clusters:
                diagnostics["num_clusters"] = num_clusters
                diagnostics["intra_sum"] = torch.zeros(num_clusters, dtype=torch.float64)
                diagnostics["intra_sq_sum"] = torch.zeros(num_clusters, dtype=torch.float64)
                diagnostics["intra_count"] = torch.zeros(num_clusters, dtype=torch.long)

            diagnostics["sample_count"] += int(assigned_dist.numel())
            diagnostics["proto_h"] = proto_h.detach().cpu()
            diagnostics["intra_sum"] += torch.bincount(
                assign_k.cpu(),
                weights=assigned_dist.double().cpu(),
                minlength=num_clusters,
            )
            diagnostics["intra_sq_sum"] += torch.bincount(
                assign_k.cpu(),
                weights=assigned_dist.double().pow(2).cpu(),
                minlength=num_clusters,
            )
            diagnostics["intra_count"] += torch.bincount(
                assign_k.cpu(),
                minlength=num_clusters,
            )

    def _log_cluster_distance_diagnostics(self, epoch, diagnostics):
        interval = int(self.arg.cluster_distance_log_interval)
        if interval <= 0 or epoch % interval != 0:
            return
        if diagnostics is None or diagnostics["sample_count"] == 0:
            return

        proto_h = diagnostics["proto_h"]
        if proto_h is None:
            return

        counts = diagnostics["intra_count"]
        sums = diagnostics["intra_sum"]
        sq_sums = diagnostics["intra_sq_sum"]
        if counts is None or sums is None or sq_sums is None:
            return

        self.io.print_log(
            "Cluster distance diagnostics (epoch {}, samples={}):".format(
                epoch,
                diagnostics["sample_count"],
            )
        )
        self.io.print_log("Intra-cluster distances to assigned prototype:")
        self.io.print_log("\tcluster | samples | mean | std")
        for cluster_id in range(int(diagnostics["num_clusters"])):
            count = int(counts[cluster_id].item())
            if count == 0:
                self.io.print_log("\t{:>7} | {:>7} | {:>6} | {:>6}".format(cluster_id, 0, "nan", "nan"))
                continue
            mean = sums[cluster_id].item() / count
            variance = max(0.0, sq_sums[cluster_id].item() / count - mean ** 2)
            self.io.print_log(
                "\t{:>7} | {:>7} | {:>6.4f} | {:>6.4f}".format(
                    cluster_id,
                    count,
                    mean,
                    math.sqrt(variance),
                )
            )

        inter_matrix = self._inter_cluster_distance_matrix(proto_h)
        if inter_matrix is None:
            return
        finite_inter = inter_matrix[torch.isfinite(inter_matrix)]
        finite_inter = finite_inter[finite_inter > 0]
        if finite_inter.numel() > 0:
            self.io.print_log(
                "Inter-cluster prototype distance summary: min={:.4f}, mean={:.4f}, max={:.4f}".format(
                    finite_inter.min().item(),
                    finite_inter.mean().item(),
                    finite_inter.max().item(),
                )
            )

        max_clusters = int(self.arg.cluster_distance_matrix_max_clusters)
        if max_clusters > 0 and inter_matrix.size(0) <= max_clusters:
            self.io.print_log("Inter-cluster prototype distance matrix:")
            self.io.print_log(self._format_distance_matrix(inter_matrix))

    def _inter_cluster_distance_matrix(self, proto_h):
        if proto_h is None or proto_h.numel() == 0:
            return None
        with torch.no_grad():
            manifold = gt.PoincareBall(self.arg.curvature)
            proto_h = proto_h.to(self.dev)
            return manifold.dist(proto_h.unsqueeze(1), proto_h.unsqueeze(0)).cpu()

    @staticmethod
    def _format_distance_matrix(matrix):
        matrix = matrix.detach().cpu()
        size = matrix.size(0)
        header = "\tcluster | " + " ".join("{:>7}".format(index) for index in range(size))
        rows = [header]
        for row_index in range(size):
            values = " ".join(
                "{:>7.3f}".format(matrix[row_index, col_index].item())
                for col_index in range(size)
            )
            rows.append("\t{:>7} | {}".format(row_index, values))
        return "\n".join(rows)

    def _hierarchy_triplet_accuracy(self, proto_h, triplets):
        with torch.no_grad():
            manifold = gt.PoincareBall(self.arg.curvature)
            anchor = proto_h[triplets[:, 0]]
            positive = proto_h[triplets[:, 1]]
            negative = proto_h[triplets[:, 2]]

            def lca_depth(x, y):
                return 0.5 * (
                    manifold.dist0(x) + manifold.dist0(y) - manifold.dist(x, y)
                )

            positive_depth = lca_depth(anchor, positive)
            negative_depth = torch.maximum(
                lca_depth(anchor, negative),
                lca_depth(positive, negative),
            )
            return (positive_depth > negative_depth).float().mean().item()

    def _should_plot_embeddings(self, epoch):
        interval = int(self.arg.embedding_plot_interval)
        if interval <= 0:
            return False
        if epoch == 1:
            return True
        return epoch % interval == 0

    def _new_embedding_snapshot(self, loader):
        max_negatives = DEFAULT_NEGATIVE_DISTANCE_SAMPLES
        try:
            loader_len = max(1, len(loader))
        except TypeError:
            loader_len = 1
        negatives_per_batch = int(math.ceil(max_negatives / float(loader_len))) if max_negatives else 0
        return {
            "embeddings": [],
            "labels": [],
            "positive_distances": [],
            "negative_distances": [],
            "sample_count": 0,
            "negative_count": 0,
            "max_samples": max(0, int(self.arg.embedding_plot_max_samples)),
            "max_negatives": max_negatives,
            "negatives_per_batch": negatives_per_batch,
            "dataset_size": self._loader_dataset_size(loader),
            "split_name": "train",
        }

    def _accumulate_embedding_snapshot(self, snapshot, features_sup, label, output):
        if snapshot is None:
            return
        with torch.no_grad():
            self._accumulate_embedding_points(snapshot, features_sup, label)
            positive_distances, negative_distances = self._contrastive_distances_from_output(output)
            if positive_distances is not None:
                snapshot["positive_distances"].append(positive_distances.cpu())
            if negative_distances is not None:
                remaining = snapshot["max_negatives"] - snapshot["negative_count"]
                take = min(snapshot["negatives_per_batch"], remaining, negative_distances.numel())
                if take > 0:
                    indices = torch.randint(
                        negative_distances.numel(),
                        (take,),
                        device=negative_distances.device,
                    )
                    snapshot["negative_distances"].append(negative_distances[indices].cpu())
                    snapshot["negative_count"] += take

    def _accumulate_embedding_points(self, snapshot, features_sup, label):
        remaining = snapshot["max_samples"] - snapshot["sample_count"]
        if remaining <= 0 or features_sup is None:
            return

        if features_sup.dim() == 3:
            embeddings = features_sup[:, 0, :]
            labels = label
        else:
            embeddings = features_sup
            labels = label

        if embeddings.size(0) > remaining:
            indices = torch.randperm(embeddings.size(0), device=embeddings.device)[:remaining]
            embeddings = embeddings[indices]
            labels = labels[indices]

        snapshot["embeddings"].append(embeddings.detach().cpu())
        snapshot["labels"].append(labels.detach().cpu())
        snapshot["sample_count"] += embeddings.size(0)

    def _contrastive_distances_from_output(self, output):
        if output is None or output.dim() != 2 or output.size(1) < 2:
            return None, None
        temperature = self._model_temperature()
        distances = (-output.detach() * temperature).clamp_min(0)
        return distances[:, 0], distances[:, 1:].reshape(-1)

    def _model_temperature(self):
        model = self._unwrap_model()
        return float(getattr(model, "T", self.arg.temperature))

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _current_cluster_centroids(self):
        model = self._unwrap_model()
        if not getattr(model, "cluster_enabled", False) or not hasattr(model, "proto_tan"):
            return None
        with torch.no_grad():
            proto_tan = model.proto_tan.detach()
            proto_norm = proto_tan.norm(dim=1, keepdim=True).clamp_min(1e-12)
            proto_tan = proto_tan * (torch.tanh(proto_norm) / proto_norm)
            manifold = gt.PoincareBall(c=float(self.arg.curvature))
            proto_h = manifold.projx(manifold.expmap0(proto_tan))
        return proto_h.cpu().numpy()

    def _embedding_plot_methods(self):
        methods = self.arg.embedding_plot_methods
        if isinstance(methods, str):
            return [method.strip() for method in methods.split(",") if method.strip()]
        return list(methods)

    def _embedding_plot_selected_labels(self):
        if not self.arg.embedding_plot_selected_labels:
            labels = self._reference_hierarchy_leaf_ids(self.arg.embedding_plot_reference_hierarchy)
            if labels:
                return labels
            return default_hierarchy_plot_classes()
        return self.arg.embedding_plot_selected_labels

    @staticmethod
    def _reference_hierarchy_leaf_ids(reference_hierarchy):
        if not reference_hierarchy:
            return []
        return hierarchy_leaf_ids(reference_hierarchy)

    def _render_embedding_snapshot(self, epoch, snapshot):
        if not snapshot["embeddings"]:
            print(f"Skipping embedding diagnostics for epoch {epoch}: no samples collected.")
            return

        embeddings = torch.cat(snapshot["embeddings"], dim=0).numpy()
        labels = torch.cat(snapshot["labels"], dim=0).numpy()
        positive_distances = (
            torch.cat(snapshot["positive_distances"], dim=0).numpy()
            if snapshot["positive_distances"]
            else None
        )
        negative_distances = (
            torch.cat(snapshot["negative_distances"], dim=0).numpy()
            if snapshot["negative_distances"]
            else None
        )
        output_dir = os.path.join(self.arg.work_dir, "embedding_plots")

        try:
            paths = render_embedding_diagnostics(
                embeddings,
                labels,
                output_dir=output_dir,
                epoch=epoch,
                curvature=self.arg.curvature,
                centroids=self._current_cluster_centroids(),
                positive_distances=positive_distances,
                negative_distances=negative_distances,
                projection_methods=self._embedding_plot_methods(),
                selected_labels=self._embedding_plot_selected_labels(),
                color_by=self.arg.embedding_plot_color_by,
                class_groups=self.arg.embedding_plot_class_groups,
                class_names=self.arg.embedding_plot_class_names,
                class_hierarchy=self.arg.embedding_plot_reference_hierarchy,
                dataset_size=snapshot.get("dataset_size"),
                split_name=snapshot.get("split_name"),
                render_class_hierarchy=self.arg.embedding_plot_hierarchy,
                hierarchy_linkages=self.arg.embedding_plot_hierarchy_linkages,
            )
        except Exception as exc:
            print(f"Embedding diagnostics failed for epoch {epoch}: {exc}")
            return

        if paths:
            self.io.print_log(
                "Saved embedding diagnostics for epoch {} to {}".format(epoch, output_dir)
            )
            self._safe_wandb_image_log(paths)

        self._log_class_hierarchies(epoch, embeddings, labels)

    def _log_class_hierarchies(self, epoch, embeddings, labels):
        if not self.arg.embedding_plot_hierarchy:
            return
        try:
            table = format_class_hierarchies(
                embeddings,
                labels,
                curvature=self.arg.curvature,
                selected_labels=self._embedding_plot_selected_labels(),
                class_groups=self.arg.embedding_plot_class_groups,
                class_names=self.arg.embedding_plot_class_names,
                class_hierarchy=self.arg.embedding_plot_reference_hierarchy,
                linkage_methods=self.arg.embedding_plot_hierarchy_linkages,
                max_merges=self.arg.embedding_plot_hierarchy_log_max_merges,
            )
        except Exception as exc:
            print(f"Class hierarchy logging failed for epoch {epoch}: {exc}")
            return
        self.io.print_log(table)

    def _safe_wandb_image_log(self, paths):
        if not self._wandb_ok:
            return
        payload = {}
        for path in paths:
            key = self._wandb_image_series_key(path)
            payload[f"embedding_diagnostics/{key}"] = wandb.Image(path)
        self._safe_wandb_log(payload, step=self.global_step)

    @staticmethod
    def _wandb_image_series_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "epoch" and parts[1].isdigit():
            return "_".join(parts[2:])
        return stem

    @staticmethod
    def _loader_dataset_size(loader):
        try:
            return len(loader.dataset)
        except (AttributeError, TypeError):
            return None

    def _safe_wandb_log(self, data, step=None):
        if not self._wandb_ok:
            return
        try:
            if step is None:
                wandb.log(data)
            else:
                wandb.log(data, step=step)
        except Exception as exc:
            self._wandb_ok = False
            print(f"W&B logging disabled after error: {exc}")

    def _finish_and_sync_wandb(self):
        if self._wandb_run is None:
            return

        try:
            self._wandb_run.finish()
        except Exception as exc:
            print(f"W&B run finalization failed: {exc}")

        if not self.arg.wandb_offline or not self._wandb_run_dir:
            return

        print(f"Syncing offline W&B run: {self._wandb_run_dir}")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wandb",
                    "sync",
                    "--include-offline",
                    self._wandb_run_dir,
                ],
                check=True,
            )
        except Exception as exc:
            print(
                "Automatic W&B sync failed. The offline run remains available at "
                f"{self._wandb_run_dir}: {exc}"
            )
