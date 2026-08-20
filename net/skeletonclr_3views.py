import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
# HYP: libraries
import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath 
from tools.sinkhorn import sinkhorn_balanced_probabilities

#import tools.hyptorch.pmath as pmath

class SkeletonCLR_3views(nn.Module):
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
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]


        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
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
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
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
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(feature_dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(feature_dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

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
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index if keys.device.index is not None else 0
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        gpu_index = keys.device.index if keys.device.index is not None else 0
        self.queue_motion[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        gpu_index = keys.device.index if keys.device.index is not None else 0
        self.queue_bone[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K


    def forward(self, im_q, im_k=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        # HYP: Initialize the Poincaré ball manifold
        poincare_ball = gt.PoincareBall(self.c)

        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

        if not self.pretrain:
            """
            Linear evaluation:
                perform same projections as in pretraining
            """
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone)) / 3.
            else:
                raise ValueError
        
        """
        Pretraining:
            project features to poincare ball in hyperbolic space
        """
        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # compute query features
        q_e = self.encoder_q(im_q)  # queries shape: [batch_size, feature_dim]
        q_e = F.normalize(q_e, dim=1)
        q = poincare_ball.expmap0(q_e) # shape: [batch_size, feature_dim]

        q_motion_e = self.encoder_q_motion(im_q_motion)
        q_motion_e = F.normalize(q_motion_e, dim=1)
        q_motion = poincare_ball.expmap0(q_motion_e)

        q_bone_e = self.encoder_q_bone(im_q_bone)
        q_bone_e = F.normalize(q_bone_e, dim=1)
        q_bone = poincare_ball.expmap0(q_bone_e)

        q_all_e = F.normalize((q_e + q_motion_e + q_bone_e) / 3.0, dim=1)
        q_all = poincare_ball.expmap0(q_all_e)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            # compute key features
            k_e = self.encoder_k(im_k)  # keys shape: [batch_size, feature_dim]
            k_e = F.normalize(k_e, dim=1)
            k_eucl = k_e.clone().detach()
            k = poincare_ball.expmap0(k_e) # shape: [batch_size, feature_dim]

            k_motion_e = self.encoder_k_motion(im_k_motion)
            k_motion_e = F.normalize(k_motion_e, dim=1)
            k_motion_eucl = k_motion_e.clone().detach()
            k_motion = poincare_ball.expmap0(k_motion_e)

            k_bone_e = self.encoder_k_bone(im_k_bone)
            k_bone_e = F.normalize(k_bone_e, dim=1)
            k_bone_eucl = k_bone_e.clone().detach()
            k_bone = poincare_ball.expmap0(k_bone_e)

            k_all_e = F.normalize((k_e + k_motion_e + k_bone_e) / 3.0, dim=1)
            k_all = poincare_ball.expmap0(k_all_e)
        
        # compute logits
        # positive logits shape: [batch_size, 1]
        l_pos = -poincare_ball.dist(q, k).unsqueeze(-1) 

        # negative logits shape: [batch_size, queue_size]
        # transpose self.queue to match dimensions for pairwise comparison [feature_dim, queue_size]
        # expand q and queue to compute pairwise distances
        # compute all pairwise (negative) hyperbolic distances between q and queue
        l_neg = -poincare_ball.dist(q.unsqueeze(1), poincare_ball.expmap0(self.queue.clone().detach().T))

        l_pos_motion = -poincare_ball.dist(q_motion, k_motion).unsqueeze(-1) 
        l_neg_motion = -poincare_ball.dist(q_motion.unsqueeze(1), poincare_ball.expmap0(self.queue_motion.clone().detach().T))
        
        l_pos_bone = -poincare_ball.dist(q_bone, k_bone).unsqueeze(-1)
        l_neg_bone = -poincare_ball.dist(q_bone.unsqueeze(1), poincare_ball.expmap0(self.queue_bone.clone().detach().T))
        
        # logits shape: [batch_size, 1+queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # dequeue and enqueue
        #self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue(k_eucl)
        self._dequeue_and_enqueue_motion(k_motion_eucl)
        self._dequeue_and_enqueue_bone(k_bone_eucl)

        features = torch.cat([q_all.unsqueeze(1), k_all.unsqueeze(1)], dim=1)

        cluster_pack = None
        if self.cluster_enabled:
            proto_norm = self.proto_tan.norm(dim=1, keepdim=True).clamp_min(1e-12)
            proto_tan = self.proto_tan * (torch.tanh(proto_norm) / proto_norm)
            proto_h = poincare_ball.expmap0(proto_tan)

            dist_q_proto = poincare_ball.dist(q_all.unsqueeze(1), proto_h.unsqueeze(0))
            dist_k_proto = poincare_ball.dist(k_all.unsqueeze(1), proto_h.unsqueeze(0))
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

        return logits, logits_motion, logits_bone, labels, features, cluster_pack
        
