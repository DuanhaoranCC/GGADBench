"""Sparse graph operations used by ARC."""

import numpy as np
import scipy.sparse as sp
import torch


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    nz = rowsum > 0
    d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape)).coalesce()


def build_adj(raw_adj, name):
    if name in ["YelpChi", "Facebook"]:
        a = normalize_adj(raw_adj)
    else:
        a = normalize_adj(raw_adj + sp.eye(raw_adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(a)


def propagate(adj_norm, feat, num_hops):
    x_list = [feat]
    for _ in range(num_hops):
        x_list.append(torch.sparse.mm(adj_norm, x_list[-1]))
    return x_list
