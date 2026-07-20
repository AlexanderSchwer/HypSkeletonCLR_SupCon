#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse

from scipy.sparse import csr_matrix

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
from .pretrain import PT_Processor

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 


def _get_hyperbolic_tsne_classes():
    try:
        from hyperbolicTSNE import HyperbolicTSNE, SequentialOptimizer
    except ImportError as exc:
        raise ImportError(
            "hyperbolicTSNE is required for method='hyp_tsne'. "
            "Install the bundled package from tools/hyperbolic-tsne."
        ) from exc

    if hasattr(HyperbolicTSNE, "_validate_data"):
        return HyperbolicTSNE, SequentialOptimizer

    class HyperbolicTSNECompat(HyperbolicTSNE):
        def _validate_data(self, X, **kwargs):
            return check_array(X, **kwargs)

    return HyperbolicTSNECompat, SequentialOptimizer


def _effective_perplexity(n_samples, perplexity):
    if n_samples < 4:
        raise ValueError("Hyperbolic t-SNE needs at least 4 samples after filtering.")
    if perplexity <= 0:
        raise ValueError("Hyperbolic t-SNE perplexity must be positive.")
    max_perplexity = max(1.0, (n_samples - 1) / 3.0)
    return min(float(perplexity), max_perplexity)


def _poincare_knn_distances(features, curvature, perplexity, chunk_size=256):
    if chunk_size < 1:
        raise ValueError("Hyperbolic t-SNE distance chunk size must be positive.")
    n_samples = features.shape[0]
    n_neighbors = min(n_samples - 1, int(3 * perplexity + 1))
    ball = gt.PoincareBall(c=curvature)
    points = torch.as_tensor(features, dtype=torch.float32)
    points = ball.projx(points)

    indices = []
    data = []
    indptr = [0]
    all_points = points.unsqueeze(0)

    with torch.no_grad():
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = points[start:end].unsqueeze(1)
            distances = ball.dist(chunk, all_points)
            rows = torch.arange(end - start)
            cols = torch.arange(start, end)
            distances[rows, cols] = float("inf")

            knn_distances, knn_indices = torch.topk(
                distances,
                k=n_neighbors,
                dim=1,
                largest=False,
                sorted=True,
            )
            data.extend(knn_distances.cpu().numpy().astype(np.float32).ravel())
            indices.extend(knn_indices.cpu().numpy().astype(np.int32).ravel())
            for _ in range(end - start):
                indptr.append(indptr[-1] + n_neighbors)

    distance_matrix = csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int32)),
        shape=(n_samples, n_samples),
    )
    distance_matrix.sort_indices()
    return distance_matrix


def _hyperbolic_tsne(features, n_components, random_state, curvature,
                     perplexity=30, chunk_size=256, exaggeration_iter=250,
                     gradient_descent_iter=750, verbose=1):
    HyperbolicTSNE, SequentialOptimizer = _get_hyperbolic_tsne_classes()
    requested_perplexity = perplexity
    perplexity = _effective_perplexity(features.shape[0], perplexity)
    if perplexity != requested_perplexity:
        print(
            f"Adjusted hyperbolic t-SNE perplexity from {requested_perplexity} "
            f"to {perplexity} for {features.shape[0]} samples."
        )
    distance_matrix = _poincare_knn_distances(
        features,
        curvature=curvature,
        perplexity=perplexity,
        chunk_size=chunk_size,
    )
    opt_params = SequentialOptimizer.sequence_poincare(
        exaggeration_its=exaggeration_iter,
        gradientDescent_its=gradient_descent_iter,
        learning_rate_main=35,
        vanilla=True,
        exact=False,
        area_split=False,
        n_iter_check=10,
        size_tol=0.999,
    )
    reducer = HyperbolicTSNE(
        n_components=n_components,
        metric="precomputed",
        hd_params={"perplexity": perplexity},
        opt_params=opt_params,
        random_state=random_state,
        verbose=verbose,
    )
    return reducer.fit_transform((distance_matrix, None))


def visualize_latent_space(features, labels, method='pca', n_components=2, random_state=42,
                           save_path=None, selected_labels=None, curvature=1.0,
                           hyp_tsne_perplexity=30,
                           hyp_tsne_chunk_size=256,
                           hyp_tsne_exaggeration_iter=250,
                           hyp_tsne_iter=750,
                           hyp_tsne_verbose=1):

    features = np.asarray(features)
    labels = np.asarray(labels)

    if selected_labels is not None:
        mask = np.isin(labels, selected_labels)
        features = features[mask]
        labels = labels[mask]
        print(f"Selected {len(labels)} samples for labels {selected_labels}.")

    if method in ('pca', 'svd', 'tsne'):
        print(f"Normalizing features for {method}...")
        features = normalize(features, axis=1)
        print(f"Features normalized.")

    '''
    label_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0,
                        10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 3, 16: 3, 17: 0, 18: 0, 19: 0,
                        20: 0, 21: 3, 22: 0, 23: 1, 24: 0, 25: 1, 26: 1, 27: 0, 28: 0, 29: 0,
                        30: 0, 31: 0, 32: 0, 33: 0, 34: 2, 35: 2, 36: 0, 37: 0, 38: 0, 39: 0,
                        40: 2, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 0, 49: 0,
                        50: 1, 51: 0, 52: 0, 53: 0, 54: 3, 55: 0, 56: 0, 57: 0, 58: 1, 59: 1}
    labels = torch.tensor([label_mapping[int(l)] for l in labels])
    #'''
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
        
    if method == 'hyp_tsne':
        reduced_features = _hyperbolic_tsne(
            features,
            n_components=n_components,
            random_state=random_state,
            curvature=curvature,
            perplexity=hyp_tsne_perplexity,
            chunk_size=hyp_tsne_chunk_size,
            exaggeration_iter=hyp_tsne_exaggeration_iter,
            gradient_descent_iter=hyp_tsne_iter,
            verbose=hyp_tsne_verbose,
        )
    else:
        reduced_features = reducer.fit_transform(features)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", len(np.unique(labels)))
    #palette = sns.color_palette("hls", len(np.unique(labels)))
    ax = sns.scatterplot(
        x=reduced_features[:, 0],
        y=reduced_features[:, 1],
        hue=labels,
        palette=palette,
        legend=True,
        alpha=0.7
    )
    if method == 'pca':
        plt.title(f"PCA of Model Output Features")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    
    elif method == 'svd':
        plt.title(f"SVD of Model Output Features")
        plt.xlabel("Singular Value 1")
        plt.ylabel("Singular Value 2")
    
    elif method == 'tsne':
        plt.title(f"t-SNE of Model Output Features")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

    elif method == 'hyp_tsne':
        plt.title(f"Hyperbolic t-SNE of Model Output Features")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        circle = plt.Circle((0, 0), radius=1, edgecolor="black", facecolor="none")
        ax.add_patch(circle)
        ax.axis("square")

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}.")
    
    plt.close()

class SkeletonCLR_Plotting(PT_Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.poincare_ball = gt.PoincareBall(self.arg.curvature)

        self.all_features = []
        self.all_labels = []

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        features = []
        labels = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            with torch.no_grad():
                latent_features = self.model.encoder_q(data)
                latent_features = F.normalize(latent_features, dim=1)
                latent_features = self.poincare_ball.expmap0(latent_features)
                features.append(latent_features.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        self.all_labels = np.concatenate(labels)
        self.all_features = np.concatenate(features)

        print("all_features shape:", self.all_features.shape)
        print("all_labels shape:", self.all_labels.shape)

        print("Generating plots with model output features...")
        save_path_svd = f"latent_space_svd.png"
        save_path_pca = f"latent_space_pca.png"
        save_path_tsne = f"latent_space_tsne.png"
        save_path_hyp_tsne = f"latent_space_hyperbolic_tsne.png"

        #selected_labels = None
        #selected_labels = [0, 5, 11, 17, 23, 26, 34, 35, 43, 54]
        #selected_labels = [5, 6, 9, 13, 14, 25, 39, 42, 50, 54]
        #selected_labels = [5, 13, 14, 25, 27, 39, 42, 50, 52, 54]
        selected_labels = [5, 11, 13, 14, 25, 27, 39, 42, 50, 54]

        #visualize_latent_space(self.all_features, self.all_labels, method='svd', save_path=save_path_svd, selected_labels=selected_labels)
        #visualize_latent_space(self.all_features, self.all_labels, method='pca', save_path=save_path_pca, selected_labels=selected_labels)
        visualize_latent_space(self.all_features, self.all_labels, method='tsne', save_path=save_path_tsne, selected_labels=selected_labels)
        visualize_latent_space(
            self.all_features,
            self.all_labels,
            method='hyp_tsne',
            save_path=save_path_hyp_tsne,
            selected_labels=selected_labels,
            curvature=self.arg.curvature,
            hyp_tsne_perplexity=self.arg.plot_hyp_tsne_perplexity,
            hyp_tsne_chunk_size=self.arg.plot_hyp_tsne_chunk_size,
            hyp_tsne_exaggeration_iter=self.arg.plot_hyp_tsne_exaggeration_iter,
            hyp_tsne_iter=self.arg.plot_hyp_tsne_iter,
            hyp_tsne_verbose=self.arg.plot_hyp_tsne_verbose,
        )
    
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
        parser.add_argument('--plot_hyp_tsne_perplexity', type=float, default=30.0, help='perplexity for hyperbolic t-SNE plotting')
        parser.add_argument('--plot_hyp_tsne_chunk_size', type=int, default=256, help='chunk size for Poincaré distance computation')
        parser.add_argument('--plot_hyp_tsne_exaggeration_iter', type=int, default=250, help='early exaggeration iterations for hyperbolic t-SNE')
        parser.add_argument('--plot_hyp_tsne_iter', type=int, default=750, help='main gradient descent iterations for hyperbolic t-SNE')
        parser.add_argument('--plot_hyp_tsne_verbose', type=int, default=1, help='verbosity for hyperbolic t-SNE')
        
        # endregion yapf: enable

        return parser
