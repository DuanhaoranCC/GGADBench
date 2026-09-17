"""Testing-time KDE, pseudo-label, and score fusion for TA-GGAD.

This module preserves the released implementation's target-label anomaly-count
oracle, hard-coded pseudo-normal RNG seed, and fixed exhaustive weight grid.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from torch import Tensor

SEARCH_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 0.99)


def to_numpy(value):
    """Official tensor-to-NumPy helper."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_score(anomaly_score):
    """Official epsilon-free min-max normalization."""
    return (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))


def compute_kde_distribution(scores, bins: int = 500):
    """Evaluate a Gaussian KDE on its own min/max grid."""
    if not isinstance(scores, np.ndarray):
        scores = scores.detach().cpu().numpy()
    kde = gaussian_kde(scores)
    x_values = np.linspace(scores.min(), scores.max(), bins)
    density = kde(x_values)
    density /= density.sum()
    return x_values, density


def average_train_kde(train_kde_list, x_grid):
    """Interpolate each source KDE onto ``x_grid`` and average densities."""
    densities = []
    for x_values, kde_values in train_kde_list:
        density = np.interp(x_grid, x_values, kde_values)
        density /= density.sum()
        densities.append(density)
    return np.mean(densities, axis=0)


def compute_js_between(train_kde_list, test_scores, bins: int = 500) -> float:
    """Official Jensen-Shannon distance between source KDEs and target scores."""
    test_scores = to_numpy(test_scores)
    x_min = min(min(x_values) for x_values, _ in train_kde_list)
    x_max = max(max(x_values) for x_values, _ in train_kde_list)
    x_min = min(x_min, test_scores.min())
    x_max = max(x_max, test_scores.max())
    x_grid = np.linspace(x_min, x_max, bins)

    train_density = average_train_kde(train_kde_list, x_grid)
    kde = gaussian_kde(test_scores)
    test_density = kde(x_grid)
    total = test_density.sum()
    if np.isfinite(total) and total > 0:
        test_density /= total
    else:
        # Same KDE and grid: the shared exponential scale cancels in p / sum(p).
        log_density = kde.logpdf(x_grid)
        if not np.isfinite(log_density).all():
            raise ValueError("TA-GGAD target KDE has non-finite log density")
        test_density = np.exp(log_density - log_density.max())
        test_density /= test_density.sum()
        warnings.warn(
            "TA-GGAD target KDE density normalization used log-domain arithmetic "
            f"(direct sum={total}, bins={bins}); KDE and grid unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(jensenshannon(train_density, test_density))


def fuse_scores_with_js(
    score_features: Tensor,
    train_query_scores_kde_list,
    train_mlp_score_kde_list,
    train_gcn_score_kde_list,
    bins: int = 500,
    method: str = "softmax",
    alpha: float = 3.0,
    eps: float = 1e-8,
):
    """Fuse query/MLP/GCN scores with ``softmax(-alpha * JS)`` weights.

    ``method`` and ``eps`` are retained because they are exposed by the release,
    although its active implementation does not use either argument.
    """
    del method, eps
    js_query = compute_js_between(train_query_scores_kde_list, score_features[:, 0], bins=bins)
    js_mlp = compute_js_between(train_mlp_score_kde_list, score_features[:, 1], bins=bins)
    js_gcn = compute_js_between(train_gcn_score_kde_list, score_features[:, 2], bins=bins)

    js_values = {"query": js_query, "mlp": js_mlp, "gcn": js_gcn}
    js_array = np.array([js_query, js_mlp, js_gcn])
    exponent = np.exp(-alpha * js_array)
    weights = exponent / exponent.sum()
    fused_score = (
        weights[0] * score_features[:, 0]
        + weights[1] * score_features[:, 1]
        + weights[2] * score_features[:, 2]
    )
    return {
        "js_values": {key: round(value, 4) for key, value in js_values.items()},
        "weights": np.round(weights, 4).tolist(),
        "fused_score": fused_score,
    }


def generate_pseudo_labels_for_each_score(
    score_features,
    true_labels,
    sort_order: str = "desc",
    num_anomaly: int = 10,
):
    """Mark the oracle-count top (or bottom) entries in every score column."""
    del true_labels
    pseudo_labels = torch.zeros_like(score_features, dtype=torch.long, device=score_features.device)
    if score_features.ndimension() == 1:
        score_features = score_features.unsqueeze(1)

    if sort_order == "desc":
        _, sorted_indices = torch.sort(score_features, dim=0, descending=True)
    else:
        _, sorted_indices = torch.sort(score_features, dim=0, descending=False)
    pseudo_labels[sorted_indices[:num_anomaly]] = 1
    return pseudo_labels


def select_anomalous_nodes_from_pseudo_labels(
    pseudo_labels: Tensor,
    true_labels,
    count: int = 2,
    normal_ratio: int = 2,
    seed: int = 42,
):
    """Vote anomalies and sample zero-vote pseudo normals.

    As in the release, ``true_labels`` is unused and the generator is always
    seeded with 42 even if a different ``seed`` argument is passed.
    """
    del true_labels, seed
    vote_counts = torch.sum(pseudo_labels, dim=1)
    anomaly_indices = torch.where(vote_counts >= count)[0]
    num_anomalies = len(anomaly_indices)

    normal_candidates = torch.where(vote_counts == 0)[0]
    num_normals = min(len(normal_candidates), num_anomalies * normal_ratio)
    generator = torch.Generator()
    generator.manual_seed(42)
    permutation = torch.randperm(len(normal_candidates), generator=generator)
    selected_normals = normal_candidates[permutation[:num_normals]]

    selected_indices = torch.cat((anomaly_indices, selected_normals), dim=0)
    selected_labels = torch.zeros_like(selected_indices, dtype=torch.long)
    selected_labels[: len(anomaly_indices)] = 1
    return selected_labels, selected_indices


def optimize_score_weights(
    score_features: Tensor,
    selected_pseudo_indices: Tensor,
    labels: Tensor,
    top_k: int = 3,
    search_grid: Sequence[float] = SEARCH_GRID,
):
    """Select the top pseudo-AUROC dimensions and exhaustively grid-search them."""
    score_norm = score_features
    selected_scores = score_norm[selected_pseudo_indices]

    with torch.no_grad():
        score_numpy = score_norm.cpu().numpy()
        selected_numpy = selected_scores.cpu().numpy()
        label_numpy = labels.cpu().numpy()

    dimension_auc = [
        roc_auc_score(label_numpy, selected_numpy[:, index])
        for index in range(selected_numpy.shape[1])
    ]
    top_dimensions = np.argsort(dimension_auc)[-top_k:]

    best_auc = 0
    best_weights = None
    for weights in itertools.product(search_grid, repeat=top_k):
        weights = np.array(weights)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        fused = (selected_numpy[:, top_dimensions] * weights).sum(axis=1)
        auc = roc_auc_score(label_numpy, fused)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    fused_score = torch.tensor(
        (score_numpy[:, top_dimensions] * best_weights).sum(axis=1),
        device=score_features.device,
        dtype=score_features.dtype,
    )
    return {
        "best_auc": round(best_auc, 4),
        "best_weights": np.round(best_weights, 4).tolist(),
        "best_dims": top_dimensions.tolist(),
        "fused_score": fused_score,
    }


def testing_time_adaptive_fusion(
    query_scores: Tensor,
    mlp_scores: Tensor,
    gcn_scores: Tensor,
    query_labels: Tensor,
    train_query_scores_kde_list,
    train_mlp_score_kde_list,
    train_gcn_score_kde_list,
    count_node: int = 1,
    normal_ratio: int = 2,
    bins: int = 500,
    alpha: float = 3.0,
):
    """Run the released target-time fusion, including label-count leakage.

    This convenience wrapper is intentionally named as an oracle protocol: it
    reads the number of positive labels in the target query set exactly as the
    official release does.
    """
    score_features = torch.stack((query_scores, mlp_scores, gcn_scores), dim=1)
    js_result = fuse_scores_with_js(
        score_features,
        train_query_scores_kde_list,
        train_mlp_score_kde_list,
        train_gcn_score_kde_list,
        bins=bins,
        alpha=alpha,
    )
    score_features = torch.cat((score_features, js_result["fused_score"].unsqueeze(1)), dim=1)

    num_anomaly = int(torch.sum(query_labels == 1).data)
    pseudo_query = generate_pseudo_labels_for_each_score(
        query_scores, query_labels, sort_order="desc", num_anomaly=num_anomaly
    )
    pseudo_mlp = generate_pseudo_labels_for_each_score(
        mlp_scores, query_labels, sort_order="desc", num_anomaly=num_anomaly
    )
    pseudo_gcn = generate_pseudo_labels_for_each_score(
        gcn_scores, query_labels, sort_order="desc", num_anomaly=num_anomaly
    )
    pseudo_all = torch.stack((pseudo_query, pseudo_mlp, pseudo_gcn), dim=1)
    selected_labels, selected_indices = select_anomalous_nodes_from_pseudo_labels(
        pseudo_all,
        query_labels,
        count=count_node,
        normal_ratio=normal_ratio,
    )
    optimized = optimize_score_weights(score_features, selected_indices, selected_labels)
    return {
        **optimized,
        "js_values": js_result["js_values"],
        "js_weights": js_result["weights"],
        "num_anomaly_oracle": num_anomaly,
        "pseudo_labels": selected_labels,
        "pseudo_indices": selected_indices,
        "score_features": score_features,
    }
