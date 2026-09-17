"""AnomalyGFM graph views and residual prototype scoring."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from util import remove_self_loop, to_torch_sparse


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    nz = rowsum > 0
    d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_adjs(raw_adj, device):
    """Construct the graph views required by the model."""
    raw = sp.csr_matrix(raw_adj)
    n = raw.shape[0]
    adj_norm = normalize_adj(raw) + sp.eye(n)
    a = remove_self_loop(raw)
    rowsum = np.asarray(a.sum(1)).flatten()
    rinv = np.zeros_like(rowsum, dtype=np.float64)
    nz = rowsum > 0
    rinv[nz] = 1.0 / rowsum[nz]
    adj_resid = sp.diags(rinv).dot(a)
    return to_torch_sparse(adj_norm).to(device), to_torch_sparse(adj_resid).to(device)


def pretrain_residual(
    model, src_t, epochs, emb_dim, opt, b_xent, device, align_weight=0.1, train_rate=0.3
):
    model.train()
    idx_trains = []
    for _f, _a, _ar, labels in src_t:
        n = int(labels.shape[0])
        if train_rate < 1.0:
            idx = np.arange(n)
            np.random.shuffle(idx)
            idx_trains.append(torch.as_tensor(idx[: int(n * train_rate)], device=device))
        else:
            idx_trains.append(torch.arange(n, device=device))
    for _ in range(epochs):
        for (feat, adj, adj_resid, labels), it in zip(src_t, idx_trains):
            normal_raw = torch.randn(emb_dim, device=device)
            abnormal_raw = torch.randn(emb_dim, device=device)
            logits, _, _, emb_residual, normal_prompt, abnormal_prompt, _ = model(
                feat, adj, adj_resid, normal_raw, abnormal_raw
            )
            lt = labels[it]
            loss_bce = torch.mean(b_xent(torch.squeeze(logits[:, it]), lt))
            er = emb_residual[:, it, :]
            normal_proto = er[:, lt == 0, :]
            abnormal_proto = er[:, lt == 1, :]
            dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=2))
            dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=2))
            loss = loss_bce + torch.mean(dif_abnormal) + align_weight * torch.mean(dif_normal)
            opt.zero_grad()
            loss.backward()
            opt.step()


def residual_proto_score(model, feat, adj, adj_resid, emb_dim, device, beta):
    """Compute anomaly scores relative to the residual prototypes."""
    normal_raw = torch.randn(emb_dim, device=device)
    abnormal_raw = torch.randn(emb_dim, device=device)
    _, _, _, emb_residual, normal_prompt, abnormal_prompt, _ = model(
        feat, adj, adj_resid, normal_raw, abnormal_raw
    )
    abnormal_p = F.normalize(abnormal_prompt, p=2, dim=0)
    normal_p = F.normalize(normal_prompt, p=2, dim=0)
    er = F.normalize(torch.squeeze(emb_residual), p=2, dim=1)
    score_normal = torch.mm(er, normal_p.unsqueeze(1))
    score_abnormal = torch.mm(er, abnormal_p.unsqueeze(1))
    return torch.exp(score_abnormal) + beta * torch.exp(-score_normal)


def subgraph_proto_score(
    model, feat, edge_index, emb_dim, device, beta, subgraph_nodes=8, batch=200_000
):
    """Compute anomaly scores using subgraph prototypes."""
    from torch_cluster import random_walk

    row = torch.as_tensor(edge_index[0], device=device).long()
    col = torch.as_tensor(edge_index[1], device=device).long()
    n = feat.shape[1]
    feat2d = feat.squeeze(0)  # (N, d)
    normal_prompt = model.act(model.fc_normal_prompt(torch.randn(emb_dim, device=device)))
    abnormal_prompt = model.act(model.fc_abnormal_prompt(torch.randn(emb_dim, device=device)))

    walk_all = random_walk(
        row, col, torch.arange(n, device=device), walk_length=subgraph_nodes - 1, coalesced=False
    )
    scores = []
    for s in range(0, n, batch):
        walk = walk_all[s : s + batch]
        sub = feat2d[walk]
        eye = torch.eye(sub.size(1), device=device).unsqueeze(0).expand(sub.size(0), -1, -1)
        emb = model.gcn2(model.gcn1(sub, eye, sparse=False), eye, sparse=False)
        seed_emb = emb[:, 0, :]
        score_ab = F.cosine_similarity(seed_emb, abnormal_prompt.unsqueeze(0))  # (b,)
        score_n = F.cosine_similarity(seed_emb, normal_prompt.unsqueeze(0))
        scores.append(torch.exp(score_ab) + beta * torch.exp(-score_n))
    return torch.cat(scores).unsqueeze(1)
