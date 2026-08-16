#!/usr/bin/env python
# pylint: disable=W0201
import argparse
import os

import geoopt as gt
import numpy as np
import torch
import torch.nn.functional as F

from torchlight import DictAction, str2bool

from .processor import Processor
from .pretrain import PT_Processor, add_lr_scheduler_args
from tools.hyperbolic_embedding_plot import (
    DEFAULT_HYP_TSNE_CHUNK_SIZE,
    DEFAULT_HYP_TSNE_EXAGGERATION_ITER,
    DEFAULT_HYP_TSNE_ITER,
    DEFAULT_HYP_TSNE_PERPLEXITY,
    DEFAULT_STANDALONE_PLOT_CLASSES,
    format_class_hierarchies,
    render_embedding_diagnostics,
)


class SkeletonCLR_Plotting(PT_Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poincare_ball = gt.PoincareBall(self.arg.curvature)

        self.all_features = []
        self.all_labels = []

    def train(self, epoch):
        self.model.eval()
        loader = self.data_loader['train']
        dataset_size = self._loader_dataset_size(loader)
        features = []
        labels = []

        for data, label in loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                latent_features = self.model.encoder_q(data)
                latent_features = F.normalize(latent_features, dim=1)
                latent_features = self.poincare_ball.projx(
                    self.poincare_ball.expmap0(latent_features)
                )
                features.append(latent_features.cpu().numpy())
            labels.append(label.cpu().numpy())

        self.all_labels = np.concatenate(labels)
        self.all_features = np.concatenate(features)

        print("all_features shape:", self.all_features.shape)
        print("all_labels shape:", self.all_labels.shape)
        print("Generating embedding diagnostics...")

        output_dir = os.path.join(self.arg.work_dir, "embedding_plots")
        paths = render_embedding_diagnostics(
            self.all_features,
            self.all_labels,
            output_dir=output_dir,
            epoch=epoch,
            curvature=self.arg.curvature,
            projection_methods=self._plot_methods(),
            selected_labels=self._plot_selected_labels(),
            color_by=self.arg.plot_color_by,
            class_groups=self.arg.plot_class_groups,
            class_names=self.arg.plot_class_names,
            dataset_size=dataset_size,
            split_name="train",
            render_class_hierarchy=self.arg.plot_hierarchy,
            hierarchy_linkages=self.arg.plot_hierarchy_linkages,
            hyp_tsne_perplexity=self.arg.plot_hyp_tsne_perplexity,
            hyp_tsne_chunk_size=self.arg.plot_hyp_tsne_chunk_size,
            hyp_tsne_exaggeration_iter=self.arg.plot_hyp_tsne_exaggeration_iter,
            hyp_tsne_iter=self.arg.plot_hyp_tsne_iter,
            hyp_tsne_verbose=self.arg.plot_hyp_tsne_verbose,
        )

        for path in paths:
            print(f"Plot saved as {path}.")

        if self.arg.plot_hierarchy:
            print(
                format_class_hierarchies(
                    self.all_features,
                    self.all_labels,
                    curvature=self.arg.curvature,
                    selected_labels=self._plot_selected_labels(),
                    class_groups=self.arg.plot_class_groups,
                    class_names=self.arg.plot_class_names,
                    linkage_methods=self.arg.plot_hierarchy_linkages,
                    max_merges=self.arg.plot_hierarchy_log_max_merges,
                )
            )

    def _plot_methods(self):
        methods = self.arg.plot_methods
        if isinstance(methods, str):
            return [method.strip() for method in methods.split(",") if method.strip()]
        return list(methods)

    def _plot_selected_labels(self):
        if not self.arg.plot_selected_labels:
            return list(DEFAULT_STANDALONE_PLOT_CLASSES)
        return self.arg.plot_selected_labels

    @staticmethod
    def _loader_dataset_size(loader):
        try:
            return len(loader.dataset)
        except (AttributeError, TypeError):
            return None

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
        parser.add_argument('--sup_epoch', type=int, default=1e6, help='the starting epoch of supervised training')
        parser.add_argument('--temperature', type=float, default=0.07, help='the temperature used in supervised training loss')
        parser.add_argument('--curvature', type=float, default=1.0, help='the curvature of the Poincare ball')
        parser.add_argument('--plot_methods', default=['tsne', 'hyp_tsne'], nargs='+', help='projection methods: pca, svd, tsne, logmap_pca, logmap_pca_disk, logmap_tsne, hyp_tsne')
        parser.add_argument('--plot_selected_labels', type=int, default=list(DEFAULT_STANDALONE_PLOT_CLASSES), nargs='+', help='class labels to plot; use -1 for all classes')
        parser.add_argument('--plot_color_by', default=['class'], nargs='+', choices=['class', 'class_group'], help='color plots by one or more modes: class or class_group')
        parser.add_argument('--plot_class_groups', action=DictAction, default=dict(), help='mapping from group names to class-label lists')
        parser.add_argument('--plot_class_names', action=DictAction, default=dict(), help='mapping from class labels to semantic class names')
        parser.add_argument('--plot_hierarchy', type=str2bool, default=True, help='render class-prototype hierarchy diagnostics')
        parser.add_argument('--plot_hierarchy_linkages', default=['ward_tangent'], nargs='+', help='class hierarchy linkages to render: ward_tangent, single_hyperbolic, complete_hyperbolic, average_hyperbolic, weighted_hyperbolic, or all')
        parser.add_argument('--plot_hierarchy_log_max_merges', type=int, default=20, help='maximum hierarchy merge rows printed by standalone plotting; 0 prints all')
        parser.add_argument('--plot_hyp_tsne_perplexity', type=float, default=DEFAULT_HYP_TSNE_PERPLEXITY, help='perplexity for hyperbolic t-SNE plotting')
        parser.add_argument('--plot_hyp_tsne_chunk_size', type=int, default=DEFAULT_HYP_TSNE_CHUNK_SIZE, help='chunk size for Poincare distance computation')
        parser.add_argument('--plot_hyp_tsne_exaggeration_iter', type=int, default=DEFAULT_HYP_TSNE_EXAGGERATION_ITER, help='early exaggeration iterations for hyperbolic t-SNE')
        parser.add_argument('--plot_hyp_tsne_iter', type=int, default=DEFAULT_HYP_TSNE_ITER, help='main gradient descent iterations for hyperbolic t-SNE')
        parser.add_argument('--plot_hyp_tsne_verbose', type=int, default=1, help='verbosity for hyperbolic t-SNE')

        # endregion yapf: enable

        return parser
