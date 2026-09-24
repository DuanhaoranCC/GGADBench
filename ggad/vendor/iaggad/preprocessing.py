"""Shared feature alignment and sparse normalization from IA-GGAD."""

import warnings

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import GaussianRandomProjection


def preprocess_features(features):
    """Row-normalize a sparse feature matrix and return a dense array."""
    rowsum = np.array(features.sum(1))
    with np.errstate(divide="ignore"):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return np.asarray(sp.diags(r_inv).dot(features).todense())


def feat_alignment(X, edge_src, edge_dst, dims):
    """Apply optional random projection, PCA, and smoothness ordering."""
    num_edges = len(edge_src)
    if X.shape[1] < dims:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataDimensionalityWarning)
            projection_input = X.cpu().numpy() if torch.is_tensor(X) else X
            X = GaussianRandomProjection(n_components=256, random_state=0).fit_transform(
                projection_input
            )
    Xt = PCA(n_components=dims, random_state=0).fit_transform(X)
    Xt = torch.FloatTensor(Xt)
    Xmin, Xmax = Xt.min(0).values, Xt.max(0).values
    Xs = (Xt - Xmin) / (Xmax - Xmin)
    es, ed = torch.as_tensor(edge_src), torch.as_tensor(edge_dst)
    smooth = torch.tensor(
        [torch.sum((Xs[es, k] - Xs[ed, k]) ** 2) / num_edges for k in range(Xt.shape[1])]
    )
    return Xt[:, torch.sort(smooth).indices].numpy()


def normalize_adj_sym(adj):
    """Apply symmetric degree normalization to a sparse adjacency."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide="ignore"):
        d = np.power(rowsum, -0.5).flatten()
    d[np.isinf(d)] = 0.0
    D = sp.diags(d)
    return adj.dot(D).transpose().dot(D).tocoo()


def _to_torch_sparse(m):
    m = sp.coo_matrix(m).astype(np.float32)
    idx = torch.from_numpy(np.vstack((m.row, m.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data), torch.Size(m.shape)).coalesce()
