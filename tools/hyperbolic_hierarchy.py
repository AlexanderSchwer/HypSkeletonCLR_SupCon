import torch
import torch.nn.functional as F
import geoopt as gt


@torch.no_grad()
def update_affinity_ema(aff_ema, q, momentum=0.9):
    """
    Update cluster-cluster affinity with exponential moving average.

    Args:
        aff_ema: [K, K] or None
        q: [B, K] soft assignments q(y=j|x_i)
        momentum: EMA factor
    Returns:
        [K, K] updated affinity
    """
    if q.dim() != 2:
        raise ValueError(f"q must be 2D [B, K], got shape {tuple(q.shape)}")
    if not (0.0 <= momentum < 1.0):
        raise ValueError("momentum must be in [0, 1)")

    batch_size, n_clusters = q.shape
    if batch_size == 0:
        raise ValueError("q batch dimension must be > 0")

    batch_aff = torch.matmul(q.t(), q) / float(batch_size)
    batch_aff = 0.5 * (batch_aff + batch_aff.t())
    batch_aff.fill_diagonal_(0.0)
    batch_aff = batch_aff / batch_aff.sum().clamp_min(1e-12)

    if aff_ema is None:
        return batch_aff.detach()

    if aff_ema.shape != (n_clusters, n_clusters):
        raise ValueError(
            f"aff_ema shape mismatch: expected {(n_clusters, n_clusters)}, got {tuple(aff_ema.shape)}"
        )

    updated = momentum * aff_ema + (1.0 - momentum) * batch_aff
    updated = 0.5 * (updated + updated.t())
    updated.fill_diagonal_(0.0)
    updated = updated / updated.sum().clamp_min(1e-12)
    return updated.detach()


@torch.no_grad()
def sample_triplets_from_affinity(aff, n_triplets):
    """
    Sample (anchor, positive, negative) from cluster affinity matrix.
    Higher affinity is preferred for positives and lower for negatives.
    """
    if aff is None:
        return torch.empty((0, 3), dtype=torch.long)
    if aff.dim() != 2 or aff.shape[0] != aff.shape[1]:
        raise ValueError(f"aff must be square [K, K], got shape {tuple(aff.shape)}")
    if n_triplets <= 0:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)

    n_clusters = aff.shape[0]
    if n_clusters < 3:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)

    aff_sym = 0.5 * (aff + aff.t())
    aff_sym = aff_sym.clone()
    aff_sym.fill_diagonal_(0.0)

    anchors = torch.randint(0, n_clusters, (n_triplets,), device=aff.device)
    triplets = []

    max_val = aff_sym.max().detach()

    for anchor in anchors:
        pos_prob = aff_sym[anchor].clone()
        pos_prob[anchor] = 0.0
        if pos_prob.sum() <= 0:
            continue
        pos = torch.multinomial((pos_prob / pos_prob.sum()).clamp_min(1e-12), 1).squeeze(0)

        neg_score = (max_val - aff_sym[anchor]).clamp_min(0.0)
        neg_score[anchor] = 0.0
        neg_score[pos] = 0.0
        if neg_score.sum() <= 0:
            continue
        neg = torch.multinomial((neg_score / neg_score.sum()).clamp_min(1e-12), 1).squeeze(0)

        triplets.append(torch.stack([anchor, pos, neg]))

    if not triplets:
        return torch.empty((0, 3), dtype=torch.long, device=aff.device)
    return torch.stack(triplets, dim=0)


def _lca_surrogate_hyp(x, y, manifold):
    """
    Surrogate for the LCA embedding z_ij in hyperbolic space.
    We average in the tangent space at the origin and map back.
    """
    tx = manifold.logmap0(x)
    ty = manifold.logmap0(y)
    t_mid = 0.5 * (tx + ty)
    return manifold.expmap0(t_mid)


def hierarchy_triplet_loss_hyp(proto_h, triplets, curvature, margin=0.05):
    """
    Hyperbolic hierarchical loss aligned with Sec. 3.2:
    maximize root-distance of positive-pair LCA over negative-pair LCAs
    with a softmax objective over (a,p), (a,n), (p,n).
    """
    if triplets.numel() == 0:
        return proto_h.new_zeros(())
    if proto_h.dim() != 2:
        raise ValueError(f"proto_h must be [K, D], got shape {tuple(proto_h.shape)}")
    if triplets.dim() != 2 or triplets.shape[1] != 3:
        raise ValueError(f"triplets must be [T, 3], got shape {tuple(triplets.shape)}")

    poincare_ball = gt.PoincareBall(curvature)

    anchors = triplets[:, 0]
    positives = triplets[:, 1]
    negatives = triplets[:, 2]

    anc = proto_h[anchors]
    pos = proto_h[positives]
    neg = proto_h[negatives]

    z_ap = _lca_surrogate_hyp(anc, pos, poincare_ball)
    z_an = _lca_surrogate_hyp(anc, neg, poincare_ball)
    z_pn = _lca_surrogate_hyp(pos, neg, poincare_ball)

    # In the paper objective, similar pairs should have LCA farther from origin.
    s_ap = poincare_ball.dist0(z_ap)
    s_an = poincare_ball.dist0(z_an) + margin
    s_pn = poincare_ball.dist0(z_pn) + margin

    scores = torch.stack([s_ap, s_an, s_pn], dim=1)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)

