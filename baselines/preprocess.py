"""Shared feature preprocessing for classic baselines."""

import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import GaussianRandomProjection

from common.data import CACHE, _atomic_save_npy, _load_valid_svd_cache, x_svd

_FEATURE_DIM = ContextVar("baseline_feature_dim", default=8)
_ADJACENCY = {}


@contextmanager
def feature_dimension(dim):
    token = _FEATURE_DIM.set(int(dim))
    try:
        yield
    finally:
        _FEATURE_DIM.reset(token)


def register_adjacency(feat, adj):
    _ADJACENCY[id(feat)] = adj


def _fixed_width_input(feat, dim):
    if int(feat.shape[1]) >= int(dim):
        return feat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        return GaussianRandomProjection(n_components=256, random_state=0).fit_transform(feat)


def _svd_cached(feat, name, dim):
    feat = _fixed_width_input(feat, dim)
    n = feat.shape[0]
    cache = CACHE / f"{name}_{n}_svd{dim}.npy"
    expected_shape = (int(n), min(int(dim), int(feat.shape[0]), int(feat.shape[1])))
    if cache.exists():
        cached = _load_valid_svd_cache(cache, expected_shape)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
    x = np.asarray(x_svd(feat, dim), dtype=np.float64)
    CACHE.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(cache, x)
    return np.asarray(x, dtype=np.float32)


def _arc_reorder(x, adj):
    lo = x.min(0, keepdims=True)
    span = x.max(0, keepdims=True) - lo
    scaled = (x - lo) / np.where(span == 0, 1.0, span)
    graph = sp.coo_matrix(adj)
    smooth = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, graph.nnz, 1_000_000):
        end = min(start + 1_000_000, graph.nnz)
        diff = scaled[graph.row[start:end]] - scaled[graph.col[start:end]]
        smooth += np.square(diff).sum(0)
    return x[:, np.argsort(smooth, kind="stable")]


def unify_features(feat, norm, name, dim=None, adj=None):
    dim = _FEATURE_DIM.get() if dim is None else int(dim)
    x = _svd_cached(feat, name, dim)
    norm = str(norm).lower()
    if norm == "none":
        return x
    if norm == "zscore":
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        return (x - mu) / np.where(sd == 0, 1.0, sd)
    if norm == "center":
        return x - x.mean(0, keepdims=True)
    if norm in ("arc", "arc_reorder"):
        adj = _ADJACENCY.get(id(feat)) if adj is None else adj
        if adj is None:
            raise ValueError(f"ARC preprocessing is missing adjacency for {name}")
        return _arc_reorder(x, adj)
    raise ValueError(f"unknown feature normalization {norm!r}")
