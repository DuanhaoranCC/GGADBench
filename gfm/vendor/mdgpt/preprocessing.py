"""Label-free MDGPT input alignment and full-graph GCN normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from common.data import _atomic_save_npy, _content_fingerprint

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache"


def align_features(features, dim=8, seed=42, cache=True):
    """Uncentered SVD U Sigma = X V, deterministic right-vector sign convention.

    A small exact Gram eigendecomposition avoids a full N-by-D left-SVD for
    tall graphs. Wide sparse features use converged ARPACK singular vectors.
    No zero padding, random projection, graph smoothing or label dependence.
    """
    n, d = features.shape
    dim = int(dim)
    if dim <= 0 or dim > min(n, d):
        raise ValueError(
            f"MDGPT SVD dim={dim} exceeds input {features.shape}; choose a common smaller dimension (no zero padding)"
        )
    tag = f"mdgpt-svd-us-v1-d{dim}-s{seed}-tol1e-8-right-pivot-positive"
    path = CACHE_ROOT / "mdgpt_svd" / (_content_fingerprint(features, tag) + ".npy")
    if cache and path.exists():
        stored = np.load(path, allow_pickle=False)
        if stored.shape == (n, dim) and stored.dtype == np.float32 and np.isfinite(stored).all():
            return stored
    matrix = (
        sp.csr_matrix(features, dtype=np.float64) if sp.issparse(features) else np.asarray(features)
    )
    if not np.isfinite(matrix.data if sp.issparse(matrix) else matrix).all():
        raise ValueError("MDGPT input features contain NaN or infinity")
    if d <= 2048:
        if sp.issparse(matrix):
            gram = (matrix.T @ matrix).toarray()
        else:
            gram = np.zeros((d, d), dtype=np.float64)
            for start in range(0, n, 8192):
                block = np.asarray(matrix[start : start + 8192], dtype=np.float64)
                gram += block.T @ block
        _, vectors = scipy.linalg.eigh(gram, subset_by_index=(d - dim, d - 1))
        vectors = vectors[:, ::-1].copy()
    elif dim < min(n, d):
        _, values, vt = svds(
            matrix.astype(np.float64),
            k=dim,
            tol=1e-8,
            v0=np.random.RandomState(seed).normal(size=min(n, d)),
        )
        vectors = vt[np.argsort(values)[::-1]].T.copy()
    else:
        dense = matrix.toarray() if sp.issparse(matrix) else matrix
        _, _, vt = np.linalg.svd(dense, full_matrices=False)
        vectors = vt[:dim].T.copy()
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(dim)])
    vectors *= np.where(signs == 0, 1.0, signs)
    result = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, 8192):
        result[start : start + 8192] = matrix[start : start + 8192] @ vectors
    if not np.isfinite(result).all():
        raise FloatingPointError("MDGPT feature alignment produced non-finite values")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save_npy(path, result)
    return result


def normalize_graph(adjacency):
    """P = D^(-1/2) (A+I) D^(-1/2), row i aggregates original columns j.

    Preserve the benchmark input's direction, weights and existing self loops.
    Add one identity as in the standard GCN convention, without ARC exceptions.
    Directed graphs are an explicitly documented benchmark input extension.
    """
    a = sp.csr_matrix(adjacency, dtype=np.float32, copy=True)
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError("MDGPT requires a square finite nonnegative adjacency")
    a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    inverse = np.asarray(a.sum(axis=1)).reshape(-1).astype(np.float64) ** -0.5
    for start in range(0, a.shape[0], 4096):
        end = min(start + 4096, a.shape[0])
        left, right = a.indptr[start], a.indptr[end]
        rows = np.repeat(inverse[start:end], np.diff(a.indptr[start : end + 1]))
        a.data[left:right] *= rows * inverse[a.indices[left:right]]
    return a


def support_dependency(adjacency, features, support, num_layers):
    """Exact L-hop computation closure, with original full-graph weights.

    This is dead-computation elimination, not neighborhood sampling: every
    dependency of every support output is retained, without a fanout cap. Never
    renormalize the resulting matrix, and never use it for query evaluation.
    """
    support = np.asarray(support, dtype=np.int64)
    reached = np.zeros(adjacency.shape[0], dtype=bool)
    reached[support] = True
    frontier = np.unique(support)
    for _ in range(int(num_layers)):
        if not len(frontier):
            break
        following = np.zeros_like(reached)
        for start in range(0, len(frontier), 4096):
            following[adjacency[frontier[start : start + 4096]].indices] = True
        following &= ~reached
        reached |= following
        frontier = np.flatnonzero(following)
    nodes = np.flatnonzero(reached)
    if len(nodes) == adjacency.shape[0]:
        return adjacency, features, support.copy()
    reduced = adjacency[nodes][:, nodes].tocsr()
    return reduced, np.ascontiguousarray(features[nodes]), np.searchsorted(nodes, support)
