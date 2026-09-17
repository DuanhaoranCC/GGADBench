"""Sparse graph views and normalization for UNPrompt."""

import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(adj):
    """Normalize the adjacency using the method-specific degree convention."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv = np.zeros_like(rowsum, dtype=np.float64)
    nz = rowsum > 0
    d_inv[nz] = np.power(rowsum[nz], -1)
    return sp.diags(d_inv).dot(adj).tocoo()


def official_loop_views(raw_adj):
    """Return the two raw adjacency views constructed by released UNPrompt."""
    adj = sp.csr_matrix(raw_adj).astype(np.float32)
    identity = sp.eye(adj.shape[0], format="csr")
    if np.all(adj.diagonal() > 0):
        with_loop = adj
        without_loop = adj - identity
    else:
        with_loop = adj + identity
        without_loop = adj
    return with_loop.tocsr(), without_loop.tocsr()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape)).coalesce()


def build_adjs(raw_adj):
    """Construct the graph views required by the model."""
    adj_withloop_won, adj_no_loop = official_loop_views(raw_adj)
    adj_withloop = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_withloop_won))
    adj_woself = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_no_loop))
    return adj_withloop, adj_withloop_won, adj_woself
