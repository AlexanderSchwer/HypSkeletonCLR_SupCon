import math
import os
from collections import OrderedDict

import geoopt as gt
import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array

from tools.action_label_hierarchy import (
    get_hierarchy,
    hierarchy_class_names,
    hierarchy_leaf_groups,
    hierarchy_to_nested_dict,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_HIERARCHY_GROUPS = OrderedDict([
    ("dressing_wearing", (13, 14, 15, 16, 17, 18, 19, 20)),
    ("posture_balance_falling", (7, 8, 41, 42)),
    ("leg_dominant_dynamic", (23, 25, 26, 50)),
])
DEFAULT_STANDALONE_PLOT_CLASSES = (5, 11, 13, 14, 25, 27, 39, 42, 50, 54)
DEFAULT_PROJECTION_METHODS = ("logmap_pca_disk", "hyp_tsne")
DEFAULT_COLOR_BY = "class_group"
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
DEFAULT_CLASS_HIERARCHY_LINKAGES = ("ward_tangent",)
SUPPORTED_CLASS_HIERARCHY_LINKAGES = (
    "ward_tangent",
    "single_hyperbolic",
    "complete_hyperbolic",
    "average_hyperbolic",
    "weighted_hyperbolic",
)
HYPERBOLIC_LINKAGE_METHODS = {
    "single_hyperbolic": "single",
    "complete_hyperbolic": "complete",
    "average_hyperbolic": "average",
    "weighted_hyperbolic": "weighted",
}


def default_hierarchy_plot_classes():
    classes = []
    for group_classes in DEFAULT_HIERARCHY_GROUPS.values():
        classes.extend(group_classes)
    return sorted(set(classes))


def parse_selected_labels(selected_labels, default=None):
    if selected_labels is None or (isinstance(selected_labels, str) and selected_labels == ""):
        selected_labels = default

    tokens = _selection_tokens(selected_labels)
    if not tokens:
        tokens = _selection_tokens(default)
    if not tokens:
        return None

    labels = []
    for token in tokens:
        if isinstance(token, str) and token.strip().lower() in ("all", "-1"):
            return None
        label = int(token)
        if label < 0:
            return None
        if label not in labels:
            labels.append(label)
    return labels


def _selection_tokens(selection):
    if selection is None:
        return []
    if isinstance(selection, str):
        return [token for token in selection.replace(",", " ").split() if token]
    try:
        return list(selection)
    except TypeError:
        return [selection]


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
    selected_labels=None,
    color_by=DEFAULT_COLOR_BY,
    class_groups=None,
    class_names=None,
    class_hierarchy=None,
    dataset_size=None,
    split_name=None,
    render_class_hierarchy=True,
    hierarchy_linkages=None,
    hyp_tsne_perplexity=DEFAULT_HYP_TSNE_PERPLEXITY,
    hyp_tsne_chunk_size=DEFAULT_HYP_TSNE_CHUNK_SIZE,
    hyp_tsne_exaggeration_iter=DEFAULT_HYP_TSNE_EXAGGERATION_ITER,
    hyp_tsne_iter=DEFAULT_HYP_TSNE_ITER,
    hyp_tsne_verbose=DEFAULT_HYP_TSNE_VERBOSE,
):
    os.makedirs(output_dir, exist_ok=True)

    embeddings = _as_numpy_2d(embeddings, "embeddings")
    labels = np.asarray(labels)
    collected_sample_count = int(embeddings.shape[0])
    class_groups = _normalize_class_groups(class_groups, class_hierarchy=class_hierarchy)
    class_names = _normalize_class_names(class_names, class_hierarchy=class_hierarchy)
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

    selected_labels = parse_selected_labels(
        selected_labels,
        default=default_hierarchy_plot_classes(),
    )
    if selected_labels is not None:
        mask = np.isin(labels, selected_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings available for plotting after filtering.")

    metadata_text = _plot_metadata_text(
        split_name=split_name,
        plotted_sample_count=int(embeddings.shape[0]),
        collected_sample_count=collected_sample_count,
        dataset_size=dataset_size,
        selected_labels=selected_labels,
    )

    methods = projection_methods or DEFAULT_PROJECTION_METHODS
    if isinstance(methods, str):
        methods = [item.strip() for item in methods.split(",") if item.strip()]

    color_modes = _normalize_color_modes(color_by)
    use_color_suffix = len(color_modes) > 1
    created_paths = []

    for method in methods:
        try:
            sample_xy, centroid_xy, is_disk, disk_radius = _project_for_plot(
                embeddings,
                centroids,
                method=method,
                curvature=curvature,
                random_state=DEFAULT_RANDOM_STATE,
                hyp_tsne_perplexity=hyp_tsne_perplexity,
                hyp_tsne_chunk_size=hyp_tsne_chunk_size,
                hyp_tsne_exaggeration_iter=hyp_tsne_exaggeration_iter,
                hyp_tsne_iter=hyp_tsne_iter,
                hyp_tsne_verbose=hyp_tsne_verbose,
            )
        except Exception as exc:
            print(f"Skipping embedding projection '{method}' for epoch {epoch}: {exc}")
            continue

        for color_mode in color_modes:
            color_values, color_title = _plot_color_values(
                labels,
                color_mode,
                class_groups,
                class_names,
            )
            color_suffix = f"_{color_mode}" if use_color_suffix else ""
            title_suffix = f" ({color_title.lower()})" if use_color_suffix else ""
            path = os.path.join(output_dir, f"epoch_{epoch:04d}_{method}{color_suffix}.png")
            _plot_embedding_projection(
                sample_xy,
                color_values,
                color_title,
                path,
                title=f"Epoch {epoch} - {method}{title_suffix}",
                metadata_text=metadata_text,
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
        metadata_text=metadata_text,
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
            metadata_text=metadata_text,
        )
        created_paths.append(distance_path)

    if render_class_hierarchy:
        for hierarchy_linkage in _normalize_hierarchy_linkages(hierarchy_linkages):
            try:
                hierarchy = compute_class_hierarchy(
                    embeddings,
                    labels,
                    curvature=curvature,
                    selected_labels=None,
                    linkage_method=hierarchy_linkage,
                )
                hierarchy_path = os.path.join(
                    output_dir,
                    f"epoch_{epoch:04d}_{hierarchy['linkage_method']}_class_hierarchy.png",
                )
                plot_class_hierarchy(
                    hierarchy,
                    save_path=hierarchy_path,
                    epoch=epoch,
                    class_groups=class_groups,
                    class_names=class_names,
                    metadata_text=metadata_text,
                )
                created_paths.append(hierarchy_path)
            except Exception as exc:
                print(
                    "Skipping class hierarchy '{}' for epoch {}: {}".format(
                        hierarchy_linkage,
                        epoch,
                        exc,
                    )
                )

        if class_hierarchy:
            try:
                reference_hierarchy_path = os.path.join(
                    output_dir,
                    f"epoch_{epoch:04d}_{_safe_plot_token(class_hierarchy)}_reference_class_hierarchy.png",
                )
                plot_class_hierarchy(
                    class_hierarchy,
                    save_path=reference_hierarchy_path,
                    epoch=epoch,
                    class_groups=class_groups,
                    class_names=class_names,
                    metadata_text=metadata_text,
                )
                created_paths.append(reference_hierarchy_path)
            except Exception as exc:
                print(
                    "Skipping reference class hierarchy '{}' for epoch {}: {}".format(
                        class_hierarchy,
                        epoch,
                        exc,
                    )
                )

    return created_paths


def compute_class_hierarchy(
    embeddings,
    labels,
    curvature=1.0,
    selected_labels=None,
    linkage_method="ward_tangent",
):
    linkage_method = _normalize_hierarchy_linkage(linkage_method)
    embeddings = _as_numpy_2d(embeddings, "embeddings")
    labels = np.asarray(labels)
    finite_sample_mask = np.isfinite(embeddings).all(axis=1)
    embeddings = embeddings[finite_sample_mask]
    labels = labels[finite_sample_mask]
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "embeddings and labels must contain the same number of samples: "
            f"{embeddings.shape[0]} != {labels.shape[0]}"
        )

    embeddings = _project_points_to_ball(embeddings, curvature)
    selected_labels = parse_selected_labels(selected_labels, default=None)
    if selected_labels is not None:
        mask = np.isin(labels, selected_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings available after filtering.")

    class_ids, class_counts, tangent_prototypes = _class_tangent_prototypes(
        embeddings,
        labels,
        curvature,
    )
    if len(class_ids) < 2:
        raise ValueError("Class hierarchy needs at least two classes.")

    poincare_prototypes = _expmap0(tangent_prototypes, curvature)
    distance_matrix = None
    if linkage_method == "ward_tangent":
        linkage_matrix = linkage(
            tangent_prototypes,
            method="ward",
            metric="euclidean",
            optimal_ordering=True,
        )
        distance_description = "Ward linkage distance in logmap tangent space"
    else:
        scipy_method = HYPERBOLIC_LINKAGE_METHODS[linkage_method]
        distance_matrix = _poincare_distance_matrix(poincare_prototypes, curvature)
        linkage_matrix = linkage(
            squareform(distance_matrix, checks=False),
            method=scipy_method,
            optimal_ordering=True,
        )
        distance_description = (
            f"{scipy_method} linkage over Poincare class-prototype distances"
        )

    return {
        "class_ids": class_ids,
        "class_counts": class_counts,
        "tangent_prototypes": tangent_prototypes,
        "poincare_prototypes": poincare_prototypes,
        "distance_matrix": distance_matrix,
        "linkage_matrix": linkage_matrix,
        "linkage_method": linkage_method,
        "distance_description": distance_description,
    }


def format_class_hierarchy(
    embeddings,
    labels,
    curvature=1.0,
    selected_labels=None,
    class_groups=None,
    class_names=None,
    class_hierarchy=None,
    linkage_method="ward_tangent",
    max_merges=20,
):
    hierarchy = compute_class_hierarchy(
        embeddings,
        labels,
        curvature=curvature,
        selected_labels=selected_labels,
        linkage_method=linkage_method,
    )
    class_groups = _normalize_class_groups(class_groups, class_hierarchy=class_hierarchy)
    class_names = _normalize_class_names(class_names, class_hierarchy=class_hierarchy)
    return _format_hierarchy_merge_table(
        hierarchy,
        class_groups=class_groups,
        class_names=class_names,
        max_merges=max_merges,
    )


def format_class_hierarchies(
    embeddings,
    labels,
    curvature=1.0,
    selected_labels=None,
    class_groups=None,
    class_names=None,
    class_hierarchy=None,
    linkage_methods=None,
    max_merges=20,
):
    tables = []
    for linkage_method in _normalize_hierarchy_linkages(linkage_methods):
        tables.append(
            format_class_hierarchy(
                embeddings,
                labels,
                curvature=curvature,
                selected_labels=selected_labels,
                class_groups=class_groups,
                class_names=class_names,
                class_hierarchy=class_hierarchy,
                linkage_method=linkage_method,
                max_merges=max_merges,
            )
        )
    return "\n".join(tables)


def plot_class_hierarchy(
    hierarchy,
    save_path,
    epoch=None,
    class_groups=None,
    class_names=None,
    class_hierarchy=None,
    metadata_text=None,
    title=None,
):
    if isinstance(hierarchy, str):
        class_hierarchy = hierarchy
    class_groups = _normalize_class_groups(class_groups, class_hierarchy=class_hierarchy)
    class_names = _normalize_class_names(class_names, class_hierarchy=class_hierarchy)
    plot_data = _hierarchy_to_dendrogram_data(hierarchy, class_names)
    _plot_class_hierarchy_dendrogram(
        plot_data,
        save_path=save_path,
        epoch=epoch,
        class_groups=class_groups,
        metadata_text=metadata_text,
        title=title,
    )


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

    if method in ("pca", "ambient_pca"):
        xy = _pca_2d(_normalize_rows(all_points), random_state=random_state)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, False, None

    if method in ("svd", "ambient_svd"):
        xy = _svd_2d(_normalize_rows(all_points), random_state=random_state)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, False, None

    if method in ("tsne", "ambient_tsne"):
        xy = _legacy_tsne_2d(_normalize_rows(all_points), random_state=random_state)
        sample_xy, centroid_xy = _split_projection(xy, n_samples)
        return sample_xy, centroid_xy, False, None

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

    if method in ("logmap_tsne", "tangent_tsne"):
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
        "pca, svd, tsne, logmap_pca, logmap_pca_disk, logmap_tsne, hyp_tsne.".format(method)
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


def _svd_2d(points, random_state):
    if points.shape[0] < 2:
        return _pca_2d(points, random_state=random_state)
    reducer = TruncatedSVD(n_components=2, random_state=random_state)
    return reducer.fit_transform(points).astype(np.float32)


def _legacy_tsne_2d(points, random_state):
    if points.shape[0] < 4:
        raise ValueError("t-SNE needs at least 4 points.")
    reducer = TSNE(n_components=2, random_state=random_state)
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


def _normalize_rows(points):
    return normalize(points, axis=1)


def _normalize_class_groups(class_groups, class_hierarchy=None):
    if class_groups:
        groups = class_groups
    elif class_hierarchy:
        groups = hierarchy_leaf_groups(class_hierarchy, identifier="class_id")
        if not groups:
            raise ValueError(
                f"Hierarchy '{class_hierarchy}' does not provide numeric class ids."
            )
    else:
        groups = DEFAULT_HIERARCHY_GROUPS
    normalized = OrderedDict()
    for group_name, group_classes in groups.items():
        labels = parse_selected_labels(group_classes, default=None)
        normalized[str(group_name)] = [] if labels is None else labels
    return normalized


def _normalize_class_names(class_names, class_hierarchy=None):
    if not class_names:
        return hierarchy_class_names(class_hierarchy) if class_hierarchy else {}
    if isinstance(class_names, (list, tuple)):
        return {
            index: str(name)
            for index, name in enumerate(class_names)
            if name is not None and str(name) != ""
        }
    return {
        int(class_id): str(name)
        for class_id, name in class_names.items()
        if name is not None and str(name) != ""
    }


def _normalize_color_by(color_by):
    color_by = str(color_by or DEFAULT_COLOR_BY).strip().lower()
    if color_by == "class":
        return "class"
    if color_by == "class_group":
        return "class_group"
    raise ValueError("color_by must be one of: class, class_group")


def _normalize_color_modes(color_by):
    tokens = _selection_tokens(color_by)
    if not tokens:
        tokens = [DEFAULT_COLOR_BY]

    modes = []
    for token in tokens:
        mode = _normalize_color_by(token)
        if mode not in modes:
            modes.append(mode)
    return modes


def _normalize_hierarchy_linkages(linkage_methods):
    tokens = _selection_tokens(linkage_methods)
    if not tokens:
        tokens = list(DEFAULT_CLASS_HIERARCHY_LINKAGES)

    normalized = []
    for token in tokens:
        if isinstance(token, str) and token.strip().lower() == "all":
            for method in SUPPORTED_CLASS_HIERARCHY_LINKAGES:
                if method not in normalized:
                    normalized.append(method)
            continue
        method = _normalize_hierarchy_linkage(token)
        if method not in normalized:
            normalized.append(method)
    return normalized


def _normalize_hierarchy_linkage(linkage_method):
    method = str(linkage_method or "").strip().lower().replace("-", "_")
    aliases = {
        "ward": "ward_tangent",
        "ward_tangent": "ward_tangent",
        "tangent_ward": "ward_tangent",
        "single": "single_hyperbolic",
        "single_hyperbolic": "single_hyperbolic",
        "hyperbolic_single": "single_hyperbolic",
        "complete": "complete_hyperbolic",
        "complete_hyperbolic": "complete_hyperbolic",
        "hyperbolic_complete": "complete_hyperbolic",
        "average": "average_hyperbolic",
        "average_hyperbolic": "average_hyperbolic",
        "hyperbolic_average": "average_hyperbolic",
        "weighted": "weighted_hyperbolic",
        "weighted_hyperbolic": "weighted_hyperbolic",
        "hyperbolic_weighted": "weighted_hyperbolic",
    }
    if method in aliases:
        return aliases[method]
    if method in ("centroid", "median"):
        raise ValueError(
            "centroid and median linkage require vector-space centroids and are not "
            "enabled for hyperbolic distance diagnostics."
        )
    raise ValueError(
        "Unsupported class hierarchy linkage '{}'. Supported values: {}.".format(
            linkage_method,
            ", ".join(SUPPORTED_CLASS_HIERARCHY_LINKAGES),
        )
    )


def _plot_color_values(labels, color_by, class_groups, class_names):
    if _normalize_color_by(color_by) == "class_group":
        group_lookup = {}
        for group_name, group_classes in class_groups.items():
            for label in group_classes:
                group_lookup[int(label)] = group_name
        return np.asarray([group_lookup.get(int(label), "other") for label in labels]), "Class group"
    return np.asarray([_class_display_name(int(label), class_names) for label in labels]), "Class"


def _class_display_name(label, class_names):
    if label in class_names:
        return f"{label}: {class_names[label]}"
    return str(label)


def _plot_metadata_text(
    split_name,
    plotted_sample_count,
    collected_sample_count,
    dataset_size,
    selected_labels,
):
    fields = []
    if split_name:
        fields.append(f"split={split_name}")
    fields.append(f"points={_format_count(plotted_sample_count)}")
    if collected_sample_count != plotted_sample_count:
        fields.append(f"collected={_format_count(collected_sample_count)}")
    if dataset_size is not None:
        fields.append(f"dataset={_format_count(dataset_size)}")
    if selected_labels is None:
        fields.append("classes=all")
    else:
        fields.append(f"classes={len(selected_labels)} selected")
    return " | ".join(fields)


def _format_count(value):
    return f"{int(value):,}"


def _plot_embedding_projection(
    sample_xy,
    color_values,
    color_title,
    save_path,
    title,
    metadata_text=None,
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
            s=36,
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

    ax.set_title(_title_with_metadata(title, metadata_text))
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    ax.legend(title=color_title, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _legend_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        text = str(value)
        try:
            return (0, int(text.split(":", 1)[0]))
        except ValueError:
            return (1, text)


def _title_with_metadata(title, metadata_text):
    if metadata_text:
        return f"{title}\n{metadata_text}"
    return title


def _plot_radius_histograms(embeddings, centroids, curvature, epoch, save_path, metadata_text=None):
    sample_norm = _finite_1d(np.linalg.norm(embeddings, axis=1))
    sample_depth = _finite_1d(_dist0(embeddings, curvature))

    centroid_norm = None
    centroid_depth = None
    if centroids is not None and centroids.size > 0:
        centroid_norm = _finite_1d(np.linalg.norm(centroids, axis=1))
        centroid_depth = _finite_1d(_dist0(centroids, curvature))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ball_radius = _ball_radius(curvature)
    norm_bins = np.linspace(0.0, ball_radius, 41)
    axes[0].hist(
        sample_norm,
        bins=norm_bins,
        alpha=0.78,
        label=f"embeddings (n={sample_norm.size})",
        color="#4C78A8",
        edgecolor="#1F4E79",
        linewidth=0.35,
    )
    if centroid_norm is not None:
        _draw_centroid_markers(axes[0], centroid_norm)
    axes[0].axvline(ball_radius, color="#222222", linestyle="--", linewidth=1.0, label="ball boundary")
    axes[0].set_xlim(0.0, ball_radius * 1.02)
    axes[0].set_title("Euclidean radius in Poincare ball")
    axes[0].set_xlabel("||x||")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False)

    depth_values = sample_depth
    if centroid_depth is not None:
        depth_values = np.concatenate([sample_depth, centroid_depth])
    depth_upper = _positive_axis_upper(depth_values)
    depth_bins = np.linspace(0.0, depth_upper, 41)
    axes[1].hist(
        sample_depth,
        bins=depth_bins,
        alpha=0.78,
        label=f"embeddings (n={sample_depth.size})",
        color="#4C78A8",
        edgecolor="#1F4E79",
        linewidth=0.35,
    )
    if centroid_depth is not None:
        _draw_centroid_markers(axes[1], centroid_depth)
    axes[1].set_xlim(0.0, depth_upper)
    axes[1].set_title("Hyperbolic radius from origin")
    axes[1].set_xlabel("d_H(0, x)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    axes[1].legend(frameon=False)

    fig.suptitle(_title_with_metadata(f"Epoch {epoch} - radius diagnostics", metadata_text))
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_centroid_markers(ax, values, color="#F58518", max_labels=30):
    values = _finite_1d(values)
    for index, value in enumerate(values):
        label = "centroids" if index == 0 else None
        ax.axvline(value, color=color, linewidth=1.2, alpha=0.9, label=label)
        if index < max_labels:
            ax.text(
                value,
                0.98,
                f"c{index}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=color,
            )


def _positive_axis_upper(values, fallback=1.0, padding=1.05):
    values = _finite_1d(values)
    if values.size == 0:
        return fallback
    upper = float(np.max(values))
    if upper <= 0:
        return fallback
    return upper * padding


def _plot_distance_histograms(
    positive_distances,
    negative_distances,
    epoch,
    save_path,
    metadata_text=None,
):
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
    ax.set_title(
        _title_with_metadata(f"Epoch {epoch} - contrastive hyperbolic distances", metadata_text)
    )
    ax.set_xlabel("Hyperbolic distance")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _class_tangent_prototypes(embeddings, labels, curvature):
    tangent = _logmap0(embeddings, curvature)
    class_ids = sorted(int(label) for label in np.unique(labels))
    class_counts = []
    prototypes = []
    for class_id in class_ids:
        mask = labels == class_id
        class_tangent = tangent[mask]
        if class_tangent.shape[0] == 0:
            continue
        class_counts.append(int(class_tangent.shape[0]))
        prototypes.append(class_tangent.mean(axis=0))

    if not prototypes:
        raise ValueError("No class prototypes could be computed.")

    return (
        np.asarray(class_ids, dtype=np.int64),
        np.asarray(class_counts, dtype=np.int64),
        np.asarray(prototypes, dtype=np.float32),
    )


def _hierarchy_to_dendrogram_data(hierarchy, class_names):
    if isinstance(hierarchy, str):
        record = get_hierarchy(hierarchy)
        return _reference_hierarchy_to_dendrogram_data(
            hierarchy_to_nested_dict(record["name"]),
            name=record["name"],
            class_names=class_names,
        )

    if not isinstance(hierarchy, dict):
        raise ValueError("hierarchy must be a stored hierarchy name or hierarchy dictionary.")

    if "linkage_matrix" in hierarchy:
        return _learned_hierarchy_to_dendrogram_data(hierarchy, class_names)

    if "tree" in hierarchy:
        return _reference_hierarchy_to_dendrogram_data(
            hierarchy["tree"],
            name=hierarchy.get("name", "reference"),
            class_names=class_names,
        )

    return _reference_hierarchy_to_dendrogram_data(
        hierarchy,
        name=hierarchy.get("name", "reference"),
        class_names=class_names,
    )


def _learned_hierarchy_to_dendrogram_data(hierarchy, class_names):
    class_ids = [int(class_id) for class_id in hierarchy["class_ids"]]
    class_counts = [int(class_count) for class_count in hierarchy["class_counts"]]
    linkage_matrix = np.asarray(hierarchy["linkage_matrix"], dtype=np.float64)
    return {
        "kind": "learned",
        "linkage_matrix": linkage_matrix,
        "leaf_labels": [
            _class_hierarchy_leaf_label(class_id, class_count, class_names)
            for class_id, class_count in zip(class_ids, class_counts)
        ],
        "leaf_class_ids": class_ids,
        "linkage_method": hierarchy["linkage_method"],
        "distance_description": hierarchy["distance_description"],
    }


def _reference_hierarchy_to_dendrogram_data(hierarchy, name, class_names):
    leaves = _reference_hierarchy_leaves(hierarchy)
    if len(leaves) < 2:
        raise ValueError("Class hierarchy plot needs at least two leaves.")

    leaf_indices = {
        id(leaf): index
        for index, leaf in enumerate(leaves)
    }
    leaf_labels = [
        _reference_hierarchy_leaf_label(leaf, class_names)
        for leaf in leaves
    ]
    leaf_class_ids = [
        None if leaf.get("class_id") is None else int(leaf["class_id"])
        for leaf in leaves
    ]
    linkage_rows = []
    leaf_count = len(leaves)

    def convert_node(node):
        children = list(node.get("children", []))
        if not children:
            return leaf_indices[id(node)], 1, 0.0

        child_clusters = [convert_node(child) for child in children]
        node_height = 1.0 + max(child_height for _, _, child_height in child_clusters)
        current_level = child_clusters
        merge_level = 0
        while len(current_level) > 1:
            next_level = []
            merge_height = node_height + 0.04 * merge_level
            for index in range(0, len(current_level), 2):
                if index + 1 >= len(current_level):
                    next_level.append(current_level[index])
                    continue
                left_id, left_count, _ = current_level[index]
                right_id, right_count, _ = current_level[index + 1]
                linkage_rows.append(
                    [
                        left_id,
                        right_id,
                        merge_height,
                        left_count + right_count,
                    ]
                )
                next_level.append(
                    (
                        leaf_count + len(linkage_rows) - 1,
                        left_count + right_count,
                        merge_height,
                    )
                )
            current_level = next_level
            merge_level += 1
        return current_level[0]

    convert_node(hierarchy)

    return {
        "kind": "reference",
        "name": str(name),
        "linkage_matrix": np.asarray(linkage_rows, dtype=np.float64),
        "leaf_labels": leaf_labels,
        "leaf_class_ids": leaf_class_ids,
        "distance_description": "Synthetic binary tree depth",
    }


def _plot_class_hierarchy_dendrogram(
    plot_data,
    save_path,
    epoch,
    class_groups,
    metadata_text=None,
    title=None,
):
    leaf_labels = plot_data["leaf_labels"]
    leaf_class_ids = plot_data["leaf_class_ids"]
    leaf_to_class = dict(zip(leaf_labels, leaf_class_ids))
    numeric_class_ids = [
        int(class_id)
        for class_id in leaf_class_ids
        if class_id is not None
    ]
    group_lookup = _class_group_lookup(class_groups)
    group_colors = _class_group_colors(numeric_class_ids, group_lookup)

    leaf_font_size = _class_hierarchy_leaf_font_size(len(leaf_labels))
    fig_width = max(12.0, 3.0 + 0.30 * len(leaf_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))
    dendrogram(
        plot_data["linkage_matrix"],
        labels=leaf_labels,
        orientation="top",
        ax=ax,
        leaf_font_size=leaf_font_size,
        leaf_rotation=90,
        color_threshold=0,
        above_threshold_color="#333333",
    )

    for tick in ax.get_xticklabels():
        class_id = leaf_to_class.get(tick.get_text())
        if class_id is None:
            continue
        group_name = group_lookup.get(int(class_id), "other")
        tick.set_color(group_colors[group_name])

    ax.set_title(
        _title_with_metadata(
            title or _dendrogram_plot_title(plot_data, epoch),
            metadata_text,
        ),
        fontsize=14,
    )
    ax.set_ylabel(plot_data["distance_description"], fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.6)

    if group_colors:
        handles = [
            Line2D([0], [0], color=color, lw=3, label=group_name)
            for group_name, color in group_colors.items()
        ]
        ax.legend(
            handles=handles,
            title="Class group",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=10,
            title_fontsize=11,
        )

    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0) if group_colors else None)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _dendrogram_plot_title(plot_data, epoch):
    prefix = f"Epoch {epoch} - " if epoch is not None else ""
    if plot_data["kind"] == "learned":
        return (
            f"{prefix}{_hierarchy_display_name(plot_data['linkage_method'])} "
            "class prototype hierarchy"
        )
    return f"{prefix}{plot_data['name']} reference class hierarchy"


def _class_hierarchy_leaf_label(class_id, class_count, class_names):
    return f"{_class_display_name(int(class_id), class_names)} (n={int(class_count)})"


def _reference_hierarchy_leaf_label(node, class_names):
    class_id = node.get("class_id")
    if class_id is None:
        return str(node.get("name", ""))
    return _class_display_name(int(class_id), class_names)


def _reference_hierarchy_leaves(node):
    children = list(node.get("children", []))
    if not children:
        return [node]
    leaves = []
    for child in children:
        leaves.extend(_reference_hierarchy_leaves(child))
    return leaves


def _safe_plot_token(value):
    token = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value)
    ).strip("_")
    return token or "hierarchy"


def _class_hierarchy_leaf_font_size(num_classes):
    if num_classes > 40:
        return 7
    if num_classes > 20:
        return 8
    return 9


def _class_group_lookup(class_groups):
    lookup = {}
    for group_name, group_classes in class_groups.items():
        for class_id in group_classes:
            lookup[int(class_id)] = str(group_name)
    return lookup


def _class_group_colors(class_ids, group_lookup):
    group_names = []
    for class_id in class_ids:
        group_name = group_lookup.get(int(class_id), "other")
        if group_name not in group_names:
            group_names.append(group_name)

    cmap = plt.get_cmap("tab10", max(1, len(group_names)))
    return {
        group_name: cmap(index)
        for index, group_name in enumerate(group_names)
    }


def _format_hierarchy_merge_table(hierarchy, class_groups, class_names, max_merges):
    linkage_matrix = hierarchy["linkage_matrix"]
    class_ids = hierarchy["class_ids"]
    members = {index: [int(class_id)] for index, class_id in enumerate(class_ids)}

    rows = []
    for merge_index, (left_id, right_id, distance, merged_leaf_count) in enumerate(linkage_matrix):
        left_id = int(left_id)
        right_id = int(right_id)
        left_members = members[left_id]
        right_members = members[right_id]
        merged_members = left_members + right_members
        rows.append(
            {
                "step": merge_index + 1,
                "left": _format_hierarchy_member_list(left_members, class_names),
                "right": _format_hierarchy_member_list(right_members, class_names),
                "distance": float(distance),
                "classes": int(merged_leaf_count),
                "groups": _format_hierarchy_groups(merged_members, class_groups),
            }
        )
        members[len(class_ids) + merge_index] = merged_members

    selected_rows = _select_hierarchy_rows(rows, max_merges=max_merges)
    lines = [
        "{} class hierarchy diagnostics:".format(
            _hierarchy_display_name(hierarchy["linkage_method"])
        ),
        "\t{}.".format(hierarchy["distance_description"]),
        "\tstep | merge | linkage_distance | classes | groups",
    ]
    for row in selected_rows:
        if row is None:
            skipped = len(rows) - len(selected_rows) + 1
            lines.append(f"\t... {skipped} merges omitted ...")
            continue
        lines.append(
            "\t{step:>4} | {left} + {right} | {distance:>8.4f} | {classes:>7} | {groups}".format(
                **row
            )
        )
    return "\n".join(lines)


def _hierarchy_display_name(linkage_method):
    return str(linkage_method).replace("_", " ").title()


def _select_hierarchy_rows(rows, max_merges):
    if max_merges is None or max_merges <= 0 or len(rows) <= max_merges:
        return rows
    head_count = max(1, max_merges // 2)
    tail_count = max(1, max_merges - head_count)
    return rows[:head_count] + [None] + rows[-tail_count:]


def _format_hierarchy_member_list(member_labels, class_names):
    labels = sorted(int(label) for label in member_labels)
    if len(labels) <= 3:
        return "{" + ", ".join(_class_display_name(label, class_names) for label in labels) + "}"
    return "{" + ", ".join(str(label) for label in labels[:3]) + f", ...; n={len(labels)}" + "}"


def _format_hierarchy_groups(member_labels, class_groups):
    lookup = _class_group_lookup(class_groups)
    groups = []
    for label in sorted(int(item) for item in member_labels):
        group = lookup.get(label, "other")
        if group not in groups:
            groups.append(group)
    return ", ".join(groups)


def _poincare_distance_matrix(points, curvature):
    ball = gt.PoincareBall(c=float(curvature))
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32)
        tensor = ball.projx(tensor)
        distances = ball.dist(tensor.unsqueeze(1), tensor.unsqueeze(0))
    matrix = distances.cpu().numpy().astype(np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


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
