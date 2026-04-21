import torch


@torch.no_grad()
def sinkhorn_balanced_transport(
    cost_matrix,
    n_iters=3,
    epsilon=0.05,
    sample_marginal=None,
    cluster_marginal=None,
):
    """
    Entropic optimal transport with balanced Sinkhorn iterations.

    Notation (matching the paper's pre-clustering view):
        C in R^{N x K} : transport cost matrix between samples x_i and clusters y_j
        K = exp(-C / epsilon) : Gibbs kernel
        a in Delta^N : sample marginal (default uniform 1/N)
        b in Delta^K : cluster marginal (default uniform 1/K)
        Pi = diag(u) K diag(v) : transport plan
        Q(y=j|x_i) = Pi_ij / a_i : per-sample soft assignment (rows sum to 1)

    Args:
        cost_matrix: [N, K]
        n_iters: number of Sinkhorn normalization iterations
        epsilon: entropy regularization strength
        sample_marginal: optional [N] source marginal a (sums to 1)
        cluster_marginal: optional [K] target marginal b (sums to 1)
    Returns:
        q: [N, K], rows sum to 1
    """
    if cost_matrix.dim() != 2:
        raise ValueError(f"cost_matrix must be 2D [N, K], got shape {tuple(cost_matrix.shape)}")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")

    n_samples, n_clusters = cost_matrix.shape
    if n_samples == 0 or n_clusters == 0:
        raise ValueError("cost_matrix must have non-zero dimensions")

    device = cost_matrix.device
    dtype = cost_matrix.dtype

    if sample_marginal is None:
        a = torch.full((n_samples,), 1.0 / float(n_samples), device=device, dtype=dtype)
    else:
        if sample_marginal.shape != (n_samples,):
            raise ValueError(
                f"sample_marginal must have shape [{n_samples}], got {tuple(sample_marginal.shape)}"
            )
        a = sample_marginal.to(device=device, dtype=dtype)
        a = a / a.sum().clamp_min(1e-12)

    if cluster_marginal is None:
        b = torch.full((n_clusters,), 1.0 / float(n_clusters), device=device, dtype=dtype)
    else:
        if cluster_marginal.shape != (n_clusters,):
            raise ValueError(
                f"cluster_marginal must have shape [{n_clusters}], got {tuple(cluster_marginal.shape)}"
            )
        b = cluster_marginal.to(device=device, dtype=dtype)
        b = b / b.sum().clamp_min(1e-12)

    # Stabilize exponentials: row-wise shift is absorbed by Sinkhorn scaling factors.
    scaled_cost = cost_matrix / epsilon
    scaled_cost = scaled_cost - scaled_cost.amin(dim=1, keepdim=True)
    kernel = torch.exp(-scaled_cost).clamp_min(1e-12)

    u = torch.ones((n_samples,), device=device, dtype=dtype)
    v = torch.ones((n_clusters,), device=device, dtype=dtype)

    for _ in range(n_iters):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.t() @ u).clamp_min(1e-12)

    transport_plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    q = transport_plan / a.unsqueeze(1).clamp_min(1e-12)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return q


@torch.no_grad()
def sinkhorn_balanced(logits, n_iters=3, eps=0.05, target_prior=None):
    """
    Backward-compatible wrapper around OT-form Sinkhorn.
    """
    cost_matrix = -logits
    return sinkhorn_balanced_transport(
        cost_matrix=cost_matrix,
        n_iters=n_iters,
        epsilon=eps,
        cluster_marginal=target_prior,
    )
