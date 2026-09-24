"""Shared deterministic node-feature adapter for gfm.

Projection width and normalization are selected per method in ``gfm.config``.
The disk cache reuses SVD and edge-smoothness computations across runs while
preserving the complete graph.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import GaussianRandomProjection

from common.data import x_svd

CACHE_VERSION = 2
CACHE = Path(__file__).resolve().parents[1] / "cache_features"
SMOOTH_EDGE_CHUNK = 1_000_000
VALID_NORMS = ("none", "center", "zscore", "arc_reorder")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _cache_path(name: str, feat, adj: sp.spmatrix, dim: int, norm: str) -> Path:
    raw_dim = int(feat.shape[1])
    return CACHE / (
        f"v{CACHE_VERSION}_{_safe_name(name)}_n{adj.shape[0]}_e{adj.nnz}_"
        f"raw{raw_dim}_svd{int(dim)}_{norm}.npy"
    )


def _arc_reorder(x: np.ndarray, adj: sp.spmatrix) -> np.ndarray:
    """Exact ARC smoothness ordering with bounded edge temporaries."""
    lo = x.min(0, keepdims=True)
    span = x.max(0, keepdims=True) - lo
    span[span == 0] = 1.0
    scaled = (x - lo) / span
    coo = sp.coo_matrix(adj)
    smooth = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, coo.nnz, SMOOTH_EDGE_CHUNK):
        end = min(start + SMOOTH_EDGE_CHUNK, coo.nnz)
        diff = scaled[coo.row[start:end]] - scaled[coo.col[start:end]]
        smooth += np.square(diff, dtype=np.float32).sum(0, dtype=np.float64)
    smooth /= max(coo.nnz, 1)
    return x[:, np.argsort(smooth, kind="stable")]


def _fixed_width_input(feat, dim: int):
    if int(feat.shape[1]) >= int(dim):
        return feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        return GaussianRandomProjection(n_components=256, random_state=0).fit_transform(feat)


def _output_width(feat, dim: int) -> int:
    input_width = 256 if int(feat.shape[1]) < int(dim) else int(feat.shape[1])
    return min(int(dim), int(feat.shape[0]), input_width)


def _compute(feat, adj: sp.spmatrix, dim: int, norm: str) -> np.ndarray:
    feat = _fixed_width_input(feat, dim)
    x = x_svd(feat, int(dim)).astype(np.float32, copy=False)
    if norm == "center":
        x = x - x.mean(0, keepdims=True)
    elif norm == "zscore":
        mean = x.mean(0, keepdims=True)
        std = x.std(0, keepdims=True)
        std[std == 0] = 1.0
        x = (x - mean) / std
    elif norm == "arc_reorder":
        x = _arc_reorder(x, adj)
    elif norm != "none":
        raise ValueError(f"unknown FEATURE_NORM={norm}; expected {VALID_NORMS}")
    return np.ascontiguousarray(x, dtype=np.float32)


def adapted_features(
    name: str, feat, adj: sp.spmatrix, dim: int, norm: str, use_cache: bool = True
) -> np.ndarray:
    norm = str(norm).lower()
    if norm == "smooth":
        norm = "arc_reorder"
    if norm not in VALID_NORMS:
        raise ValueError(f"unknown FEATURE_NORM={norm}; expected {VALID_NORMS}")
    adj = sp.csr_matrix(adj)
    path = _cache_path(name, feat, adj, dim, norm)
    if use_cache and path.exists():
        cached = np.load(path, allow_pickle=False)
        expected = (adj.shape[0], _output_width(feat, dim))
        if cached.shape == expected and cached.dtype == np.float32:
            return np.ascontiguousarray(cached)

    x = _compute(feat, adj, dim, norm)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
        np.save(tmp, x, allow_pickle=False)
        os.replace(tmp, path)
    return x
