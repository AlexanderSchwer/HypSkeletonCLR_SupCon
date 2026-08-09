import math
import os
from collections import OrderedDict

import geoopt as gt
import matplotlib
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_HIERARCHY_GROUPS = OrderedDict([
    ("dressing_wearing", (13, 14, 15, 16, 17, 18, 19, 20)),
    ("posture_balance_falling", (7, 8, 41, 42)),
    ("leg_dominant_dynamic", (23, 25, 26, 50)),
])
DEFAULT_PROJECTION_METHODS = ("logmap_pca_disk", "hyp_tsne")
DEFAULT_COLOR_BY = "hierarchy"
DEFAULT_RANDOM_STATE = 42
DEFAULT_NEGATIVE_DISTANCE_SAMPLES = 8192
DEFAULT_HYP_TSNE_PERPLEXITY = 30.0
DEFAULT_HYP_TSNE_CHUNK_SIZE = 256
DEFAULT_HYP_TSNE_EXAGGERATION_ITER = 250
DEFAULT_HYP_TSNE_ITER = 750
DEFAULT_HYP_TSNE_LEARNING_RATE = 0.1
DEFAULT_HYP_TSNE_BOUNDARY_FRACTION_LIMIT = 0.75
DEFAULT_HYP_TSNE_VERBOSE = 0
DEFAULT_ANNOTATE_CENTROIDS = True
DEFAULT_ANNOTATE_CENTROIDS_MAX = 80


def default_hierarchy_plot_classes():
    classes = []
    for group_classes in DEFAULT_HIERARCHY_GROUPS.values():
        classes.extend(group_classes)
    return sorted(set(classes))


def render_embedding_diagnostics(
    embeddings,
    labels,
    output_dir,
    epoch,
    curvature=1.0,
    centroids=None,
    positive_distances=None,
    negative_distances=None,
    projection_methods=None,
):
    os.makedirs(output_dir, exist_ok=True)

    embeddings = _as_numpy_2d(embeddings, "embeddings")
    labels = np.asarray(labels)
    finite_sample_mask = np.isfinite(embeddings).all(axis=1)
    if not finite_sample_mask.all():
        embeddings = embeddings[finite_sample_mask]
        labels = labels[finite_sample_mask]
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "embeddings and labels must contain the same number of samples: "
            f"{embeddings.shape[0]} != {labels.shape[0]}"
        )

    centroids = None if centroids is None else _as_numpy_2d(centroids, "centroids")
    if centroids is not None:
        finite_centroid_mask = np.isfinite(centroids).all(axis=1)
        centroids = centroids[finite_centroid_mask]
    embeddings = _project_points_to_ball(embeddings, curvature)
    if centroids is not None and centroids.size > 0:
        centroids = _project_points_to_ball(centroids, curvature)
    else:
        centroids = None

    selected_labels = default_hierarchy_plot_classes()
    if selected_labels:
        mask = np.isin(labels, selected_labels)
        if mask.any():
            embeddings = embeddings[mask]
            labels = labels[mask]

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings available for plotting after filtering.")

    methods = projection_methods or DEFAULT_PROJECTION_METHODS
    if isinstance(methods, str):
        methods = [item.strip() for item in methods.split(",") if item.strip()]

    color_values, color_title = _plot_color_values(
        labels,
        DEFAULT_COLOR_BY,
        DEFAULT_HIERARCHY_GROUPS,
    )
    created_paths = []

    for method in methods:
        try:
            sample_xy, centroid_xy, is_disk, disk_radius = _project_for_plot(
                embeddings,
                centroids,
                method=method,
                curvature=curvature,
                random_state=DEFAULT_RANDOM_STATE,
                hyp_tsne_perplexity=DEFAULT_HYP_TSNE_PERPLEXITY,
                hyp_tsne_chunk_size=DEFAULT_HYP_TSNE_CHUNK_SIZE,
                hyp_tsne_exaggeration_iter=DEFAULT_HYP_TSNE_EXAGGERATION_ITER,
                hyp_tsne_iter=DEFAULT_HYP_TSNE_ITER,
                hyp_tsne_verbose=DEFAULT_HYP_TSNE_VERBOSE,
            )
        except Exception as exc:
            print(f"Skipping embedding projection '{method}' for epoch {epoch}: {exc}")
            continue

        path = os.path.join(output_dir, f"epoch_{epoch:04d}_{method}.png")
        _plot_embedding_projection(
            sample_xy,
            color_values,
            color_title,
            path,
            title=f"Epoch {epoch} - {method}",
            centroid_xy=centroid_xy,
            is_disk=is_disk,
            disk_radius=disk_radius,
            annotate_centroids=DEFAULT_ANNOTATE_CENTROIDS,
            annotate_centroids_max=DEFAULT_ANNOTATE_CENTROIDS_MAX,
        )
        created_paths.append(path)

    radius_path = os.path.join(output_dir, f"epoch_{epoch:04d}_radius_histograms.png")
    _plot_radius_histograms(
        embeddings,
        centroids,
        curvature=curvature,
        epoch=epoch,
        save_path=radius_path,
    )
    created_paths.append(radius_path)

    if positive_distances is not None and negative_distances is not None:
        distance_path = os.path.join(
            output_dir, f"epoch_{epoch:04d}_contrastive_distance_histograms.png"
        )
        _plot_distance_histograms(
            positive_distances,
            negative_distances,
            epoch=epoch,
            save_path=distance_path,
        )
        created_paths.append(distance_path)

    return created_paths


def _as_numpy_2d(values, name):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {values.shape}")
    return values


def _project_points_to_ball(points, curvature):
    ball = gt.PoincareBall(c=float(curvature))
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32)
        tensor = ball.projx(tensor)
    return tensor.cpu().numpy()


def _project_for_plot(
    embeddings,
    centroids,
    method,
    curvature,
    random_state,
    hyp_tsne_perplexity,
    hyp_tsne_chunk_size,
    hyp_tsne_exaggeration_iter,
    hyp_tsne_iter,
    hyp_tsne_verbose,
):
    method = method.lower()
    n_samples = embeddings.shape[0]
    all_points = embeddings
    if centroids is not None:
        all_points = np.concatenate([embeddings, centroids], axis=0)

    if method in ("logmap_pca", "tangent_pca"):
        tangent = _logmap0(all_points, curvature)
        xy = _pca_2d(tangent, random_state=random_state)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, False, None

    if method in ("logmap_pca_disk", "pca_disk", "tangent_pca_disk"):
        tangent = _logmap0(all_points, curvature)
        xy_tangent = _pca_2d(tangent, random_state=random_state)
        xy = _expmap0(xy_tangent, curvature)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, True, _ball_radius(curvature)

    if method in ("tsne", "logmap_tsne", "tangent_tsne"):
        tangent = _logmap0(all_points, curvature)
        xy = _tsne_2d(tangent, random_state=random_state, perplexity=hyp_tsne_perplexity)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, False, None

    if method in ("hyp_tsne", "hyperbolic_tsne"):
        xy = _hyperbolic_tsne(
            all_points,
            n_components=2,
            random_state=random_state,
            curvature=curvature,
            perplexity=hyp_tsne_perplexity,
            chunk_size=hyp_tsne_chunk_size,
            exaggeration_iter=hyp_tsne_exaggeration_iter,
            gradient_descent_iter=hyp_tsne_iter,
            verbose=hyp_tsne_verbose,
        )
        xy = np.asarray(xy, dtype=np.float32)
        _raise_if_boundary_saturated(
            xy,
            radius=1.0,
            fraction_limit=DEFAULT_HYP_TSNE_BOUNDARY_FRACTION_LIMIT,
        )
        xy = _clip_to_radius(xy, radius=1.0)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, True, 1.0

    raise ValueError(
        "Unknown projection method '{}'. Supported methods: "
        "logmap_pca, logmap_pca_disk, logmap_tsne, hyp_tsne.".format(method)
    )


def _split_projection(xy, n_samples):
    sample_xy = xy[:n_samples]
    centroid_xy = xy[n_samples:] if xy.shape[0] > n_samples else None
    return sample_xy, centroid_xy


def _logmap0(points, curvature):
    ball = gt.PoincareBall(c=float(curvature))
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32)
        tensor = ball.logmap0(ball.projx(tensor))
    return tensor.cpu().numpy()


def _expmap0(points, curvature):
    ball = gt.PoincareBall(c=float(curvature))
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32)
        tensor = ball.expmap0(tensor)
        tensor = ball.projx(tensor)
    return tensor.cpu().numpy()


def _pca_2d(points, random_state):
    if points.shape[0] < 2:
        padded = np.zeros((points.shape[0], 2), dtype=np.float32)
        if points.shape[1] > 0:
            width = min(points.shape[1], 2)
            padded[:, :width] = points[:, :width]
        return padded
    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(points).astype(np.float32)


def _tsne_2d(points, random_state, perplexity):
    if points.shape[0] < 4:
        raise ValueError("t-SNE needs at least 4 points.")
    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=_effective_perplexity(points.shape[0], perplexity),
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(points).astype(np.float32)


def _plot_color_values(labels, color_by, class_groups):
    if color_by == "hierarchy":
        group_lookup = {}
        groups = class_groups or DEFAULT_HIERARCHY_GROUPS
        for group_name, group_classes in groups.items():
            for label in group_classes:
                group_lookup[int(label)] = group_name
        return np.asarray([group_lookup.get(int(label), "other") for label in labels]), "Hierarchy group"
    return labels, "Class"


def _plot_embedding_projection(
    sample_xy,
    color_values,
    color_title,
    save_path,
    title,
    centroid_xy=None,
    is_disk=False,
    disk_radius=None,
    annotate_centroids=True,
    annotate_centroids_max=80,
):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    unique_values = sorted(set(color_values), key=_legend_sort_key)
    cmap = plt.get_cmap("tab20", max(1, len(unique_values)))
    color_map = {value: cmap(index) for index, value in enumerate(unique_values)}

    for value in unique_values:
        mask = color_values == value
        ax.scatter(
            sample_xy[mask, 0],
            sample_xy[mask, 1],
            s=14,
            alpha=0.68,
            linewidths=0,
            color=color_map[value],
            label=str(value),
        )

    if centroid_xy is not None and centroid_xy.size > 0:
        ax.scatter(
            centroid_xy[:, 0],
            centroid_xy[:, 1],
            s=80,
            marker="X",
            color="#111111",
            edgecolors="white",
            linewidths=0.75,
            label="cluster centroid",
            zorder=5,
        )
        if annotate_centroids and centroid_xy.shape[0] <= annotate_centroids_max:
            for index, (x_coord, y_coord) in enumerate(centroid_xy):
                ax.text(
                    x_coord,
                    y_coord,
                    str(index),
                    fontsize=5,
                    color="#111111",
                    ha="center",
                    va="center",
                    zorder=6,
                )

    if is_disk:
        radius = float(disk_radius or 1.0)
        circle = plt.Circle((0, 0), radius=radius, edgecolor="#222222", facecolor="none", linewidth=1.0)
        ax.add_patch(circle)
        ax.set_xlim(-1.05 * radius, 1.05 * radius)
        ax.set_ylim(-1.05 * radius, 1.05 * radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Poincare x")
        ax.set_ylabel("Poincare y")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    ax.set_title(title)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    ax.legend(title=color_title, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _legend_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _plot_radius_histograms(embeddings, centroids, curvature, epoch, save_path):
    sample_norm = np.linalg.norm(embeddings, axis=1)
    sample_depth = _dist0(embeddings, curvature)

    centroid_norm = None
    centroid_depth = None
    if centroids is not None and centroids.size > 0:
        centroid_norm = np.linalg.norm(centroids, axis=1)
        centroid_depth = _dist0(centroids, curvature)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(sample_norm, bins=40, alpha=0.75, label="embeddings", color="#4C78A8")
    if centroid_norm is not None:
        axes[0].hist(centroid_norm, bins=25, alpha=0.55, label="centroids", color="#F58518")
    axes[0].axvline(_ball_radius(curvature), color="#222222", linestyle="--", linewidth=1.0, label="ball boundary")
    axes[0].set_title("Euclidean radius in Poincare ball")
    axes[0].set_xlabel("||x||")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    axes[1].hist(sample_depth, bins=40, alpha=0.75, label="embeddings", color="#4C78A8")
    if centroid_depth is not None:
        axes[1].hist(centroid_depth, bins=25, alpha=0.55, label="centroids", color="#F58518")
    axes[1].set_title("Hyperbolic radius from origin")
    axes[1].set_xlabel("d_H(0, x)")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Epoch {epoch} - radius diagnostics")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_histograms(positive_distances, negative_distances, epoch, save_path):
    positive = _finite_1d(positive_distances)
    negative = _finite_1d(negative_distances)
    if positive.size == 0 or negative.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bins = np.histogram_bin_edges(np.concatenate([positive, negative]), bins=60)
    ax.hist(negative, bins=bins, alpha=0.6, density=True, label="negative", color="#E45756")
    ax.hist(positive, bins=bins, alpha=0.7, density=True, label="positive", color="#4C78A8")
    ax.axvline(np.median(positive), color="#1F4E79", linestyle="--", linewidth=1.0, label="positive median")
    ax.axvline(np.median(negative), color="#8B1E1E", linestyle="--", linewidth=1.0, label="negative median")
    ax.set_title(f"Epoch {epoch} - contrastive hyperbolic distances")
    ax.set_xlabel("Hyperbolic distance")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _finite_1d(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return values[np.isfinite(values)]


def _dist0(points, curvature):
    ball = gt.PoincareBall(c=float(curvature))
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32)
        depth = ball.dist0(ball.projx(tensor))
    return depth.cpu().numpy()


def _ball_radius(curvature):
    return 1.0 / math.sqrt(float(curvature))


def _clip_to_radius(points, radius):
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    max_norm = radius * 0.999
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1e-12))
    return points * scale


def _raise_if_boundary_saturated(points, radius, fraction_limit):
    norms = np.linalg.norm(points, axis=1)
    finite_norms = norms[np.isfinite(norms)]
    if finite_norms.size == 0:
        raise ValueError("hyperbolic t-SNE returned no finite points.")
    boundary_fraction = np.mean(finite_norms >= radius * 0.98)
    if boundary_fraction > fraction_limit:
        raise ValueError(
            "hyperbolic t-SNE projection is boundary-saturated "
            f"({boundary_fraction:.1%} of points have radius >= {radius * 0.98:.3f}). "
            "The projection is likely not interpretable."
        )


def _get_hyperbolic_tsne_classes():
    try:
        from hyperbolicTSNE import HyperbolicTSNE, SequentialOptimizer
    except ImportError as exc:
        raise ImportError(
            "hyperbolicTSNE is required for method='hyp_tsne'. Install the bundled "
            "package with `python -m pip install -e tools/hyperbolic-tsne`."
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
        raise ValueError("t-SNE perplexity must be positive.")
    max_perplexity = max(1.0, (n_samples - 1) / 3.0)
    return min(float(perplexity), max_perplexity)


def _poincare_knn_distances(features, curvature, perplexity, chunk_size=256):
    if chunk_size < 1:
        raise ValueError("Hyperbolic t-SNE distance chunk size must be positive.")
    n_samples = features.shape[0]
    n_neighbors = min(n_samples - 1, int(3 * perplexity + 1))
    ball = gt.PoincareBall(c=float(curvature))
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
            # The bundled h-tSNE affinity code expects squared neighbor distances.
            knn_distances = knn_distances.pow(2)
            data.extend(knn_distances.cpu().numpy().astype(np.float32).ravel())
            indices.extend(knn_indices.cpu().numpy().astype(np.int32).ravel())
            for _ in range(end - start):
                indptr.append(indptr[-1] + n_neighbors)

    distance_matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n_samples, n_samples),
    )
    distance_matrix.sort_indices()
    return distance_matrix


def _hyperbolic_tsne(
    features,
    n_components,
    random_state,
    curvature,
    perplexity=30,
    chunk_size=256,
    exaggeration_iter=250,
    gradient_descent_iter=750,
    verbose=0,
):
    HyperbolicTSNE, SequentialOptimizer = _get_hyperbolic_tsne_classes()
    perplexity = _effective_perplexity(features.shape[0], perplexity)
    distance_matrix = _poincare_knn_distances(
        features,
        curvature=curvature,
        perplexity=perplexity,
        chunk_size=chunk_size,
    )
    opt_params = SequentialOptimizer.sequence_poincare(
        exaggeration_its=exaggeration_iter,
        gradientDescent_its=gradient_descent_iter,
        learning_rate_ex=DEFAULT_HYP_TSNE_LEARNING_RATE,
        learning_rate_main=DEFAULT_HYP_TSNE_LEARNING_RATE,
        vanilla=False,
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
