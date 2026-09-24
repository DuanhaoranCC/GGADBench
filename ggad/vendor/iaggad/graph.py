"""Sparse graph propagation and affinity scoring for IA-GGAD."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from util import remove_self_loop


def _edge_dot(feat, r, c):
    return (feat[r] * feat[c]).sum(1)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    nz = rowsum > 0
    d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
    D = sp.diags(d_inv_sqrt)
    return adj.dot(D).transpose().dot(D).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape)).coalesce()


def normalize_graphconv_adj(adj):
    """Sparse matrix exactly matching DGL GraphConv(norm='both')."""
    matrix = sp.coo_matrix(adj)
    out_degree = np.asarray(matrix.sum(axis=1)).ravel()
    in_degree = np.asarray(matrix.sum(axis=0)).ravel()
    out_inv = np.zeros_like(out_degree, dtype=np.float64)
    in_inv = np.zeros_like(in_degree, dtype=np.float64)
    np.power(out_degree, -0.5, out=out_inv, where=out_degree > 0)
    np.power(in_degree, -0.5, out=in_inv, where=in_degree > 0)
    values = matrix.data * out_inv[matrix.row] * in_inv[matrix.col]
    return sp.coo_matrix((values, (matrix.col, matrix.row)), shape=matrix.shape)


def build_adj(raw_adj, name):
    a = (
        normalize_adj(raw_adj)
        if name in ["YelpChi", "Facebook"]
        else normalize_adj(raw_adj + sp.eye(raw_adj.shape[0]))
    )
    return sparse_mx_to_torch_sparse_tensor(a)


def propagate(adj_norm, feat, num_hops):
    x_list = [feat]
    for _ in range(num_hops):
        x_list.append(torch.sparse.mm(adj_norm, x_list[-1]))
    return x_list


def build_aff(raw_adj, device):
    """Build normalized and binary affinity graph views."""
    a = sp.csr_matrix(raw_adj).astype(np.float32)
    a.data[:] = 1.0
    a = remove_self_loop(a)
    n = a.shape[0]
    a_sl = (a + sp.eye(n, dtype=np.float32)).tocsr()
    norm = sparse_mx_to_torch_sparse_tensor(normalize_graphconv_adj(a_sl)).to(device)
    binary = sparse_mx_to_torch_sparse_tensor(a_sl).to(device)
    return norm, binary


def normalize_score(ano_score):
    return (ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score))


class my_GCN(nn.Module):

    def __init__(self, in_feats, h_feats):
        super(my_GCN, self).__init__()
        self.W1 = nn.Linear(in_feats, 2 * h_feats, bias=False)
        self.b1 = nn.Parameter(torch.zeros(2 * h_feats))
        self.W2 = nn.Linear(2 * h_feats, h_feats, bias=False)
        self.b2 = nn.Parameter(torch.zeros(h_feats))
        self.fc1 = nn.Linear(h_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, h_feats)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, norm_adj, x):
        h = F.relu(torch.sparse.mm(norm_adj, self.W1(x)) + self.b1)
        h = F.relu(torch.sparse.mm(norm_adj, self.W2(h)) + self.b2)
        return h


def max_message(feature, adj_sparse, max_edges=None):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    adj = adj_sparse.coalesce()
    row, col = adj.indices()[0], adj.indices()[1]
    val = adj.values()
    n = feature.size(0)
    E = row.size(0)
    if max_edges is not None and E > max_edges:
        sel = torch.randperm(E, device=feature.device)[:max_edges]
        row, col, val = row[sel], col[sel], val[sel]
        E = max_edges
    message = torch.zeros(n, device=feature.device)
    deg = torch.zeros(n, device=feature.device)

    CHUNK = 250_000 if E > 2_000_000 else 2_000_000
    if torch.is_grad_enabled():
        if E <= CHUNK:
            sim = (feature[row] * feature[col]).sum(1)
        else:

            sim = torch.cat(
                [
                    checkpoint(
                        _edge_dot,
                        feature,
                        row[s : s + CHUNK],
                        col[s : s + CHUNK],
                        use_reentrant=False,
                    )
                    for s in range(0, E, CHUNK)
                ]
            )
        sim = torch.nan_to_num(sim)
        message.index_add_(0, row, sim * val)  # Σ_j sim[i,j]·adj[i,j]
        deg.index_add_(0, col, val)
    else:
        for s in range(0, E, CHUNK):
            r = row[s : s + CHUNK]
            c = col[s : s + CHUNK]
            v = val[s : s + CHUNK]
            sim = torch.nan_to_num((feature[r] * feature[c]).sum(1))
            message.index_add_(0, r, sim * v)
            deg.index_add_(0, c, v)
    r_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    message = message * r_inv
    return -torch.sum(message), message
