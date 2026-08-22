import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import geoopt as gt
from tools.sinkhorn import sinkhorn_balanced_probabilities

class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, curvature=1.0,
                 cluster_enabled=False, num_clusters=120, sinkhorn_tau=0.1,
                 sinkhorn_iters=20, sinkhorn_eps=0.05, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.cluster_enabled = bool(cluster_enabled and pretrain)
        self.num_clusters = int(num_clusters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_eps = float(sinkhorn_eps)

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.c = curvature
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.c = curvature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            if self.cluster_enabled:
                if self.num_clusters <= 1:
                    raise ValueError("num_clusters must be > 1 when clustering is enabled")
                if self.sinkhorn_tau <= 0:
                    raise ValueError("sinkhorn_tau must be > 0")
                if self.sinkhorn_iters < 1:
                    raise ValueError("sinkhorn_iters must be >= 1")
                if self.sinkhorn_eps <= 0:
                    raise ValueError("sinkhorn_eps must be > 0")
                self.proto_tan = nn.Parameter(torch.empty(self.num_clusters, feature_dim))
                nn.init.normal_(self.proto_tan, std=0.1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index if keys.device.index is not None else 0
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        if not self.pretrain:
            return self.encoder_q(im_q)

        poincare_ball = gt.PoincareBall(self.c)

        # compute query features
        q_e = self.encoder_q(im_q)  # queries shape: [batch_size, feature_dim]
        q_e = F.normalize(q_e, dim=1)
        q_h = poincare_ball.expmap0(q_e) # shape: [batch_size, feature_dim]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # compute key features
            k_e = self.encoder_k(im_k)  # keys shape: [batch_size, feature_dim]
            k_e = F.normalize(k_e, dim=1)
            k_eucl = k_e.clone().detach()
            k_h = poincare_ball.expmap0(k_e) # shape: [batch_size, feature_dim]
        
        # compute contrastive scores
        # positive scores shape: [batch_size, 1]
        pos_scores = -poincare_ball.dist(q_h, k_h).unsqueeze(-1)

        # negative scores shape: [batch_size, queue_size]
        # transpose self.queue to match dimensions for pairwise comparison [feature_dim, queue_size]
        # expand q and queue to compute pairwise distances
        # compute all pairwise (negative) hyperbolic distances between q and queue
        neg_scores = -poincare_ball.dist(q_h.unsqueeze(1), poincare_ball.expmap0(self.queue.clone().detach().T))

        # scores shape: [batch_size, 1+queue_size]
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        
        # apply temperature
        scores /= self.T

        # labels: positive key indicators
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)

        # Combine q (query) and k (key) as two views of the same image
        features = torch.cat([q_h.unsqueeze(1), k_h.unsqueeze(1)], dim=1)  # features shape: [batch_size, n_views, feature_dim], with n_views=2 (q and k)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_eucl)

        if self.cluster_enabled:
            proto_norm = self.proto_tan.norm(dim=1, keepdim=True).clamp_min(1e-12)
            proto_tan = self.proto_tan * (torch.tanh(proto_norm) / proto_norm)
            proto_h = poincare_ball.expmap0(proto_tan)

            # Paper notation:
            # P_ij = p(y_i = j | x_i) are predicted posterior probabilities.
            # Q_ij = q(y_i = j | x_i) are the same probabilities after OT balancing.
            dist_q_proto = poincare_ball.dist(q_h.unsqueeze(1), proto_h.unsqueeze(0))
            dist_k_proto = poincare_ball.dist(k_h.unsqueeze(1), proto_h.unsqueeze(0))
            p_q = F.softmax(-dist_q_proto / self.sinkhorn_tau, dim=1)
            p_k = F.softmax(-dist_k_proto / self.sinkhorn_tau, dim=1)

            q_k = sinkhorn_balanced_probabilities(
                p_k.detach(),
                n_iters=self.sinkhorn_iters,
                exponent=self.sinkhorn_tau / self.sinkhorn_eps,
            )
            assign_k = torch.argmax(q_k, dim=1)

            cluster_pack = {
                "dist_q_proto": dist_q_proto,
                "dist_k_proto": dist_k_proto,
                "p_q": p_q,
                "p_k": p_k,
                "q_k": q_k,
                "proto_h": proto_h,
                "assign_k": assign_k,
            }
            return scores, labels, features, cluster_pack

        return scores, labels, features
        
