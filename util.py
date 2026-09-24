"""Shared random seeds, sparse graph operators, and evaluation metrics."""

import math
import os
import random

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from evaluation import metric_names, recall_at_k, aggregate_metrics


def set_seed(seed):
    """Seed CPU RNGs and only the currently selected CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        # Seed the CPU generator directly to avoid initializing every visible GPU.
        torch.default_generator.manual_seed(seed)
        requested_device = os.environ.get("GAD_BENCHMARK_DEVICE")
        if torch.cuda.is_available() and (
            requested_device is None or requested_device.startswith("cuda")
        ):
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def glorot(value):
    """Initialize a tensor with Glorot uniform weights."""
    if value is not None:
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)


def evaluate(labels, scores, mark=None, metrics=None):
    """Evaluate only selected query nodes; higher scores mean more anomalous."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if mark is not None:
        labels, scores = labels[mark], scores[mark]
    invalid = ~np.isfinite(scores)
    if invalid.any():
        raise ValueError(f"non-finite evaluation scores: {int(invalid.sum())}/{scores.size}")
    result = {}
    for name in metric_names(metrics):
        if name == "AUROC":
            result[name] = float(roc_auc_score(labels, scores))
        elif name == "AUPRC":
            result[name] = float(average_precision_score(labels, scores))
        else:
            result[name] = recall_at_k(labels, scores)
    return result


def remove_self_loop(adj):
    """Remove diagonal edges and return a CSR adjacency matrix."""
    a = sp.coo_matrix(adj)
    keep = a.row != a.col
    return sp.csr_matrix(
        (a.data[keep], (a.row[keep], a.col[keep])), shape=a.shape, dtype=np.float32
    )


def add_self_loop(adj):
    """Replace diagonal edges with one unit self-loop per node."""
    return (remove_self_loop(adj) + sp.eye(adj.shape[0], dtype=np.float32, format="csr")).tocsr()


def to_torch_sparse(adj):
    import torch

    adj = sp.coo_matrix(adj).astype(np.float32)
    idx = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    val = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(adj.shape)).coalesce()


def aggregate(per_target):
    """Aggregate selected metrics with population standard deviations."""
    return aggregate_metrics(per_target)
