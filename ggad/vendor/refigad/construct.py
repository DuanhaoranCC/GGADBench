"""REFIGAD synthetic feature construction."""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from common.data import EdgeList, edge_chunks

LARGE_THRESHOLD = 40000


def rank_score(score):
    rank_scores = torch.argsort(score).argsort()
    max_rank = torch.max(rank_scores)
    if max_rank == 0:
        return rank_scores.float()
    return rank_scores.float() / max_rank * 100


def _to_scipy_binary_adj(adj_norm):
    if isinstance(adj_norm, EdgeList):
        r, c = adj_norm.row.numpy(), adj_norm.col.numpy()
        adj_scipy = sp.csr_matrix((np.ones(r.shape[0], np.float32), (r, c)), shape=adj_norm.shape)
    elif torch.is_tensor(adj_norm):
        if adj_norm.is_sparse:
            adj_norm = adj_norm.coalesce()
            idx = adj_norm.indices().cpu().numpy()
            val = adj_norm.values().cpu().numpy()
            adj_scipy = sp.csr_matrix((val, (idx[0], idx[1])), shape=adj_norm.shape)
        else:
            adj_scipy = sp.csr_matrix(adj_norm.cpu().numpy())
    elif isinstance(adj_norm, np.ndarray):
        adj_scipy = sp.csr_matrix(adj_norm)
    else:
        adj_scipy = adj_norm
    adj_scipy.data = np.ones_like(adj_scipy.data)
    adj_scipy.setdiag(0)
    adj_scipy.eliminate_zeros()
    return adj_scipy


def conv_residual(conv_list):
    first = conv_list[0]
    h_embed_list = [h_i for h_i in conv_list[1:]]
    return (conv_list[-1] - first), h_embed_list[-1]


def get_global_sim(embed, mode="cos"):
    global_center = embed.mean(dim=0, keepdim=True)
    if mode == "dis":
        return torch.norm(embed - global_center, p=2, dim=1)
    norm_embed = F.normalize(embed, p=2, dim=1)
    norm_center = F.normalize(global_center, p=2, dim=1)
    return torch.mm(norm_embed, norm_center.t()).squeeze()


def get_neibour_sim(embed, adj, step=1, mode="dis", large_threshold=LARGE_THRESHOLD):
    is_large_graph = embed.size(0) > large_threshold
    if is_large_graph:
        device = embed.device
        n = embed.size(0)
        degree = torch.zeros(n, device=device)
        if mode == "cos":
            norm_embed = F.normalize(embed, p=2, dim=1)
            agg = torch.zeros_like(norm_embed)
            for r, c, val in edge_chunks(adj, device):
                agg.index_add_(0, r, norm_embed[c] * val.unsqueeze(1))
                degree.index_add_(0, r, val)
            neibour_sim = (norm_embed * agg).sum(dim=1) / (degree + 1e-8)
        else:  # dis
            numerator = torch.zeros(n, device=device)
            for r, c, val in edge_chunks(adj, device):
                numerator.index_add_(0, r, torch.norm(embed[r] - embed[c], p=2, dim=1))
                degree.index_add_(0, r, val)
            neibour_sim = numerator / (degree + 1e-8)
        return (neibour_sim - neibour_sim.min()) / (neibour_sim.max() - neibour_sim.min() + 1e-8)

    if step > 1:
        adj1 = adj.clone()
        for _ in range(step - 1):
            adj = torch.sparse.mm(adj, adj1)
    if adj.is_sparse:
        adj = adj.to_dense()
    adj = (adj > 0).float()
    adj.fill_diagonal_(0)
    if mode == "cos":
        norm_embed = F.normalize(embed, p=2, dim=1)
        sim_matrix = torch.mm(norm_embed, norm_embed.t())
        neibour_sim = (sim_matrix * adj).sum(dim=1) / (adj.sum(dim=1) + 1e-8)
    elif mode == "dis":
        dist_matrix = torch.cdist(embed, embed, p=2)
        neibour_sim = (dist_matrix * adj).sum(dim=1) / (adj.sum(dim=1) + 1e-8)
    return (neibour_sim - neibour_sim.min()) / (neibour_sim.max() - neibour_sim.min() + 1e-8)


def get_degree_centrality(adj_norm):
    device = adj_norm.device if torch.is_tensor(adj_norm) else torch.device("cpu")
    adj_scipy = _to_scipy_binary_adj(adj_norm)
    degree_np = np.array(adj_scipy.sum(axis=1)).flatten()
    return torch.tensor(degree_np, dtype=torch.float32, device=device)


def get_clustering_coefficient(adj_norm, block=50_000):
    device = adj_norm.device if torch.is_tensor(adj_norm) else torch.device("cpu")
    adj = _to_scipy_binary_adj(adj_norm)
    n = adj.shape[0]
    degree = np.array(adj.sum(axis=1)).flatten()
    triangles = np.zeros(n)
    for s in range(0, n, block):
        sl = slice(s, min(s + block, n))
        a_blk = adj[sl]
        triangles[sl] = np.array(a_blk.multiply(a_blk.dot(adj)).sum(axis=1)).flatten() / 2
    possible = degree * (degree - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = triangles / possible
        cc[np.isnan(cc)] = 0
        cc[np.isinf(cc)] = 0
    return torch.tensor(cc, dtype=torch.float32, device=device)


def construct_features(graph, large_threshold=LARGE_THRESHOLD):
    labels = graph.ano_labels
    _, h_embed = conv_residual(graph.conv_list)
    s_global = get_global_sim(h_embed, mode="cos")
    s_nb_cos = get_neibour_sim(
        graph.sim_conv[-2], graph.adj, step=1, mode="cos", large_threshold=large_threshold
    )
    s_nb_dis = get_neibour_sim(
        graph.sim_conv[-2], graph.adj, step=1, mode="dis", large_threshold=large_threshold
    )
    dev = labels.device
    degree = get_degree_centrality(graph.adj).to(dev)
    clustering = get_clustering_coefficient(graph.adj).to(dev)

    P_features = torch.cat(
        [
            rank_score(s_global).unsqueeze(1),
            rank_score(s_nb_cos).unsqueeze(1),
            rank_score(s_nb_dis).unsqueeze(1),
            rank_score(degree).unsqueeze(1),
            rank_score(clustering).unsqueeze(1),
        ],
        dim=1,
    )
    return P_features, labels
