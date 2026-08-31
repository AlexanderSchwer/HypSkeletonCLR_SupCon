import torch


PSEUDO_CONTRASTIVE_MODES = ("pseudo_hard", "pseudo_soft")


def augmentation_mask(batch_size, device=None, dtype=None, weight=1.0):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if dtype is None:
        dtype = torch.float32
    return torch.eye(batch_size, device=device, dtype=dtype) * float(weight)


def pseudo_label_mask_from_posteriors(
    posteriors,
    confidence_threshold=0.8,
    mode="pseudo_hard",
    lambda_aug=1.0,
    lambda_pseudo=1.0,
):
    """
    Build a weighted SupCon mask from prototype posterior probabilities.

    The diagonal represents augmentation positives for the same original clip.
    Off-diagonal weights come from hard or soft pseudo-label assignments.
    """
    _validate_posteriors(posteriors)
    _validate_pseudo_args(confidence_threshold, mode, lambda_aug, lambda_pseudo)

    with torch.no_grad():
        posteriors = posteriors.detach()
        batch_size = posteriors.size(0)
        confidence, pseudo_labels = posteriors.max(dim=1)
        confident = confidence >= float(confidence_threshold)

        identity = torch.eye(batch_size, dtype=torch.bool, device=posteriors.device)
        confident_pair = confident.view(-1, 1) & confident.view(1, -1)
        off_diagonal = ~identity

        mask = augmentation_mask(
            batch_size,
            device=posteriors.device,
            dtype=posteriors.dtype,
            weight=lambda_aug,
        )

        if mode == "pseudo_hard":
            same_cluster = pseudo_labels.view(-1, 1).eq(pseudo_labels.view(1, -1))
            pseudo_weights = same_cluster & confident_pair & off_diagonal
            mask = mask + float(lambda_pseudo) * pseudo_weights.to(posteriors.dtype)
        elif mode == "pseudo_soft":
            posterior_similarity = posteriors @ posteriors.T
            pseudo_weights = posterior_similarity * confident_pair.to(posteriors.dtype)
            mask = mask + float(lambda_pseudo) * pseudo_weights * off_diagonal.to(posteriors.dtype)
        else:
            raise ValueError(f"Unknown pseudo contrastive mode: {mode}")

    return mask, pseudo_labels, confidence, confident


def _validate_posteriors(posteriors):
    if posteriors.dim() != 2:
        raise ValueError(
            f"posteriors must be 2D [batch, clusters], got shape {tuple(posteriors.shape)}"
        )
    if posteriors.size(0) == 0 or posteriors.size(1) == 0:
        raise ValueError("posteriors must have non-zero batch and cluster dimensions")
    if not torch.is_floating_point(posteriors):
        raise ValueError("posteriors must be floating point probabilities")
    if not torch.isfinite(posteriors).all():
        raise ValueError("posteriors must be finite")
    if (posteriors < 0).any():
        raise ValueError("posteriors must be non-negative")


def _validate_pseudo_args(confidence_threshold, mode, lambda_aug, lambda_pseudo):
    if mode not in PSEUDO_CONTRASTIVE_MODES:
        raise ValueError(
            f"mode must be one of {PSEUDO_CONTRASTIVE_MODES}, got {mode!r}"
        )
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be in [0, 1]")
    if lambda_aug < 0:
        raise ValueError("lambda_aug must be non-negative")
    if lambda_pseudo < 0:
        raise ValueError("lambda_pseudo must be non-negative")
