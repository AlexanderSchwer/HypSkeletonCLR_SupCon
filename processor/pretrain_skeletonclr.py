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
from .pretrain import PT_Processor

from tools.losses import SupConLoss
from tools.hyperbolic_hierarchy import (
    prototype_affinity_hyp,
    update_affinity_ema,
    sample_triplets_from_affinity,
    hierarchy_triplet_loss_hyp,
)
from tools.hyperbolic_embedding_plot import (
    DEFAULT_NEGATIVE_DISTANCE_SAMPLES,
    render_embedding_diagnostics,
)

import wandb

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 

class SkeletonCLR_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_affinity = None
        
        # Initialize wandb run
        self._wandb_ok = not self.arg.wandb_disabled
        self._wandb_run = None
        self._wandb_run_dir = None
        if self._wandb_ok:
            try:
                mode = "offline" if self.arg.wandb_offline else "online"
                self._wandb_run = wandb.init(
                    project="HypSkeletonCLR_SupCon",
                    mode=mode,
                )
                if self._wandb_run is not None and self._wandb_run.dir:
                    self._wandb_run_dir = os.path.dirname(self._wandb_run.dir)
                model_args = self.arg.model_args if isinstance(self.arg.model_args, dict) else {}
                wandb.config.update({
                    "learning_rate": self.arg.base_lr,
                    "optimizer": self.arg.optimizer,
                    "weight_decay": self.arg.weight_decay,
                    "nesterov": self.arg.nesterov,
                    "num_epochs": self.arg.num_epoch,
                    "sup_epoch": self.arg.sup_epoch,
                    "temperature": self.arg.temperature,
                    "curvature": self.arg.curvature,
                    "cluster_enabled": bool(model_args.get("cluster_enabled", False)),
                    "num_clusters": model_args.get("num_clusters", None),
                    "sinkhorn_tau": model_args.get("sinkhorn_tau", None),
                    "sinkhorn_iters": model_args.get("sinkhorn_iters", None),
                    "sinkhorn_eps": model_args.get("sinkhorn_eps", None),
                    "lambda_sink": self.arg.lambda_sink,
                    "lambda_hier": self.arg.lambda_hier,
                    "cluster_warmup_steps": self.arg.cluster_warmup_steps,
                    "cluster_ramp_steps": self.arg.cluster_ramp_steps,
                    "hier_warmup_steps": self.arg.hier_warmup_steps,
                    "hier_ramp_steps": self.arg.hier_ramp_steps,
                    "embedding_plot_interval": self.arg.embedding_plot_interval,
                    "embedding_plot_max_samples": self.arg.embedding_plot_max_samples,
                    "embedding_plot_methods": self.arg.embedding_plot_methods,
                })
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

            if epoch < self.arg.sup_epoch:
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
            else:
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                
                #loss_unsup = self.loss(output, target)
                
                try:
                    label_sup = torch.tensor([label_mapping[int(l)] for l in label], device=label.device)
                except NameError:
                    label_sup = label

                loss_sup = self.criterion(features_sup, label_sup)
                
                # new loss function: scaled sum of unsupervised and supervised loss
                #alpha = (epoch - self.arg.sup_epoch) / (self.arg.num_epoch - self.arg.sup_epoch)
                alpha = 1.0
                #loss = (1 - alpha) * loss_unsup + alpha * loss_sup
                loss = loss_sup

            loss_base = loss
            loss_sink, loss_hier, cluster_metrics = self._compute_cluster_losses(cluster_pack)
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
                    "lambda_sink_effective": sink_weight,
                    "lambda_hier_effective": hier_weight,
                    #"supervised_loss": loss_sup.data.item(),
                    #"unsupervised_loss": loss_unsup.data.item(),
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
        if sink_loss_value:
            self.epoch_info['train_mean_loss_sink'] = np.mean(sink_loss_value)
            self.train_writer.add_scalar('loss_sink', self.epoch_info['train_mean_loss_sink'], epoch)
        if hier_loss_value:
            self.epoch_info['train_mean_loss_hier'] = np.mean(hier_loss_value)
            self.train_writer.add_scalar('loss_hier', self.epoch_info['train_mean_loss_hier'], epoch)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)

        if epoch < self.arg.sup_epoch:
            alpha = 0
            print(f"Scaling of Loss Functions -> Unsupervised: {1 - alpha:.4f}, Supervised: {alpha:.4f}")
        else:
            print(f"Scaling of Loss Functions -> Unsupervised: {1 - alpha:.4f}, Supervised: {alpha:.4f}")
        
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
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--sup_epoch', type=int, default=1e6, help='the starting epoch of supervised training')
        parser.add_argument('--temperature', type=float, default=0.07, help='the temperature used in supervised training loss')
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincaré ball')
        parser.add_argument('--lambda_sink', type=float, default=1.0, help='maximum weight for Sinkhorn clustering loss')
        parser.add_argument('--cluster_warmup_steps', type=int, default=1000, help='MoCo-only iterations before Sinkhorn loss')
        parser.add_argument('--cluster_ramp_steps', type=int, default=2000, help='iterations used to ramp Sinkhorn loss weight')
        parser.add_argument('--lambda_hier', type=float, default=0.1, help='maximum weight for hyperbolic hierarchy loss')
        parser.add_argument('--hier_update_interval', type=int, default=200, help='interval of iterations for hierarchy loss')
        parser.add_argument('--hier_warmup_steps', type=int, default=3000, help='warmup iterations before hierarchy loss')
        parser.add_argument('--hier_ramp_steps', type=int, default=2000, help='iterations used to ramp hierarchy loss weight')
        parser.add_argument('--hier_triplets', type=int, default=512, help='number of hierarchy triplets sampled each update')
        parser.add_argument('--hier_margin', type=float, default=0.05, help='triplet margin for hierarchy loss')
        parser.add_argument('--affinity_momentum', type=float, default=0.9, help='EMA momentum for cluster affinity')
        parser.add_argument('--affinity_temperature', type=float, default=1.0, help='temperature for prototype affinity')
        parser.add_argument('--wandb_offline', type=str2bool, default=False, help='log W&B offline and automatically sync the run when the script exits')
        parser.add_argument('--wandb_disabled', type=str2bool, default=False, help='disable W&B init, logging, finishing, and sync completely')
        parser.add_argument('--embedding_plot_interval', type=int, default=0, help='render embedding diagnostic plots every N epochs; 0 disables live plotting')
        parser.add_argument('--embedding_plot_max_samples', type=int, default=1024, help='maximum epoch samples retained for each embedding plot')
        parser.add_argument('--embedding_plot_methods', default=['poincare', 'logmap_pca_disk', 'hyp_tsne'], nargs='+', help='projection methods: poincare, logmap_pca, logmap_pca_disk, logmap_tsne, hyp_tsne')
        
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
            )
        except Exception as exc:
            print(f"Embedding diagnostics failed for epoch {epoch}: {exc}")
            return

        if paths:
            self.io.print_log(
                "Saved embedding diagnostics for epoch {} to {}".format(epoch, output_dir)
            )
            self._safe_wandb_image_log(paths)

    def _safe_wandb_image_log(self, paths):
        if not self._wandb_ok:
            return
        payload = {}
        for path in paths:
            key = os.path.splitext(os.path.basename(path))[0]
            payload[f"embedding_diagnostics/{key}"] = wandb.Image(path)
        self._safe_wandb_log(payload, step=self.global_step)

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
