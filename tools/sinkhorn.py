import torch


@torch.no_grad()
def sinkhorn_balanced_probabilities(
    probability_matrix,
    n_iters=3,
    exponent=1.0,
    sample_marginal=None,
    cluster_marginal=None,
):
    """
    Balanced Sinkhorn iterations over predicted posterior probabilities.

    Notation (matching the paper's pre-clustering view):
        P in R^{N x K} : predicted posteriors, P_ij = p(y_i = j | x_i)
        P^lambda : probability kernel used by Sinkhorn
        a in Delta^N : sample marginal (default uniform 1/N)
        b in Delta^K : cluster marginal (default uniform 1/K)
        Pi = diag(u) P^lambda diag(v) : transport plan
        Q_ij = q(y_i = j | x_i) = Pi_ij / a_i : OT-balanced probabilities

    Args:
        probability_matrix: [N, K], non-negative predicted probabilities
        n_iters: number of Sinkhorn normalization iterations
        exponent: paper's lambda applied as P^lambda
        sample_marginal: optional [N] source marginal a (sums to 1)
        cluster_marginal: optional [K] target marginal b (sums to 1)
    Returns:
        q: [N, K], OT-balanced probabilities with rows summing to 1
    """
    if probability_matrix.dim() != 2:
        raise ValueError(
            f"probability_matrix must be 2D [N, K], got shape {tuple(probability_matrix.shape)}"
        )
    if exponent <= 0:
        raise ValueError("exponent must be > 0")
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")

    n_samples, n_clusters = probability_matrix.shape
    if n_samples == 0 or n_clusters == 0:
        raise ValueError("probability_matrix must have non-zero dimensions")

    device = probability_matrix.device
    dtype = probability_matrix.dtype

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

    if (probability_matrix < 0).any():
        raise ValueError("probability_matrix must be non-negative")

    # Row-wise normalization keeps the input explicit as probabilities. Any row-wise
    # factor introduced by this normalization is absorbed by the Sinkhorn scaling.
    p = probability_matrix / probability_matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
    kernel = p.clamp_min(1e-12).pow(exponent)

    u = torch.ones((n_samples,), device=device, dtype=dtype)
    v = torch.ones((n_clusters,), device=device, dtype=dtype)

    for _ in range(n_iters):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.t() @ u).clamp_min(1e-12)

    transport_plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    q = transport_plan / a.unsqueeze(1).clamp_min(1e-12)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return q
