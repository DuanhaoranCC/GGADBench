"""REFIGAD graph preparation and chunked propagation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from common.data import (
    BIG,
    EDGE_CHUNK,
    EdgeList,
    adj_sym_cond_scipy,
    chunk_propagate,
    load_source_marked,
    load_target_marked,
)
from ggad.vendor.iaggad.preprocessing import (
    _to_torch_sparse,
    normalize_adj_sym,
    preprocess_features,
)

ROWNORM = ["Amazon", "Amazon-all", "YelpChi", "YelpChi-all", "tolokers", "t_finance"]
NO_SELFLOOP = ["YelpChi", "Facebook"]
LARGE_THRESHOLD = 40000
STREAM_FULL_RANGE_THRESHOLD = 30000


def _calc_edge_sim(src, dst, x, metric, chunk=5_000_000):
    """Compute edge similarities in bounded memory chunks."""
    E = src.numel()
    sim = torch.empty(E, device=x.device)
    for s in range(0, E, chunk):
        a, b = x[src[s : s + chunk]], x[dst[s : s + chunk]]
        if metric == "cosine":
            a, b = F.normalize(a, p=2, dim=1), F.normalize(b, p=2, dim=1)
            sim[s : s + chunk] = ((a * b).sum(dim=1) + 1) / 2
        else:
            sim[s : s + chunk] = (a * b).sum(dim=1)  # dot
    if metric == "cosine":
        return sim
    return (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)


def _sim_conv(x, adj_norm, k, device, sim_metric="dot", large_threshold=LARGE_THRESHOLD):
    n = x.size(0)
    h_list = [x]
    is_large = n > large_threshold
    eye_idx = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    eye = torch.sparse_coo_tensor(eye_idx, torch.ones(n, device=device), (n, n)).coalesce()
    A = adj_norm.coalesce()
    indices, values = A.indices(), A.values()

    # The release uses the dense all-pairs similarity range through 40k nodes.
    # Stream that exact range for the 30k--40k interval to avoid a dense N x N
    # allocation while retaining the released values on every used edge.
    if STREAM_FULL_RANGE_THRESHOLD < n <= large_threshold:
        row, col = indices
        low = high = None
        if sim_metric == "dot":
            for start in range(0, n, 256):
                dots = x[start : start + 256] @ x.T
                current_low, current_high = dots.min(), dots.max()
                low = current_low if low is None else torch.minimum(low, current_low)
                high = current_high if high is None else torch.maximum(high, current_high)
        h_1 = x.clone()
        for start in range(0, row.numel(), 250_000):
            end = min(start + 250_000, row.numel())
            r, c, v = row[start:end], col[start:end], values[start:end]
            if sim_metric == "cosine":
                sim = (
                    (F.normalize(x[r], p=2, dim=1) * F.normalize(x[c], p=2, dim=1))
                    .sum(dim=1)
                    .add(1)
                    .div(2)
                )
            elif sim_metric == "dot":
                sim = ((x[r] * x[c]).sum(dim=1) - low) / (high - low + 1e-8)
            else:
                raise ValueError(f"Unknown sim_metric: {sim_metric}")
            sim = sim.masked_fill(r == c, 0.0)
            h_1.index_add_(0, r, (v * sim).unsqueeze(1) * x[c])
        h_list.append(h_1)
        for _ in range(k - 1):
            h_list.append(h_1)
        return h_list

    sim_mat = None
    if is_large:
        sim_values = _calc_edge_sim(indices[0], indices[1], x, sim_metric)
    else:
        if sim_metric == "cosine":
            xn = F.normalize(x, p=2, dim=1)
            sim_mat = (torch.mm(xn, xn.t()) + 1) / 2
        else:
            sim_mat = torch.mm(x, x.t())
            sim_mat = (sim_mat - sim_mat.min()) / (sim_mat.max() - sim_mat.min() + 1e-8)
        sim_mat.fill_diagonal_(0)
        sim_values = sim_mat[indices[0], indices[1]]

    weighted = values * sim_values
    A_w = (torch.sparse_coo_tensor(indices, weighted, (n, n)).coalesce() + eye).coalesce()
    h_list.append(torch.sparse.mm(A_w, x))

    for _ in range(k - 1):
        h_list.append(torch.sparse.mm(A_w, h_list[-1]))
    return h_list


def _sim_conv_big(adj_el, x, num_hops, device, sim_metric="dot", chunk=EDGE_CHUNK):
    row, col, val = adj_el.row, adj_el.col, adj_el.val
    E = adj_el.nnz
    gmin = gmax = None
    if sim_metric != "cosine":
        for s in range(0, E, chunk):
            r, c = row[s : s + chunk].to(device), col[s : s + chunk].to(device)
            d = (x[r] * x[c]).sum(1)
            gmin = d.min() if gmin is None else torch.minimum(gmin, d.min())
            gmax = d.max() if gmax is None else torch.maximum(gmax, d.max())
    h1 = x.clone()
    for s in range(0, E, chunk):
        r, c = row[s : s + chunk].to(device), col[s : s + chunk].to(device)
        v = val[s : s + chunk].to(device)
        if sim_metric == "cosine":
            a, b = F.normalize(x[r], p=2, dim=1), F.normalize(x[c], p=2, dim=1)
            sim = ((a * b).sum(1) + 1) / 2
        else:
            sim = ((x[r] * x[c]).sum(1) - gmin) / (gmax - gmin + 1e-8)
        h1.index_add_(0, r, ((v * sim).unsqueeze(1)) * x[c])
    out = [x, h1]
    for _ in range(num_hops - 1):
        out.append(h1)
    return out


def build_graph(name, num_hops, device, large_threshold=LARGE_THRESHOLD, target=False):
    if target:
        adj, feat_raw, label, mark = load_target_marked(name, hops=num_hops)
    else:
        adj, feat_raw, label, mark = load_source_marked(name)
    if name in ROWNORM:
        feat = preprocess_features(sp.lil_matrix(feat_raw))
    else:
        feat = np.asarray(
            feat_raw.todense() if sp.issparse(feat_raw) else feat_raw, dtype=np.float64
        )
    feat = np.ascontiguousarray(feat, np.float32)

    if name in BIG:
        conv_list = chunk_propagate(name, adj, feat, num_hops, device)
        x = conv_list[0]
        adj_out = EdgeList(adj_sym_cond_scipy(name, adj))
        sim_list = _sim_conv_big(adj_out, x, num_hops, device)
    else:
        x = torch.FloatTensor(feat).to(device)
        if name in NO_SELFLOOP:
            adj_norm = normalize_adj_sym(adj)
        else:
            adj_norm = normalize_adj_sym(adj + sp.eye(adj.shape[0]))
        adj_norm = _to_torch_sparse(adj_norm).to(device)
        conv_list = [x]  # propagated(k)
        for _ in range(num_hops):
            conv_list.append(torch.sparse.mm(adj_norm, conv_list[-1]))
        sim_list = _sim_conv(x, adj_norm, num_hops, device, large_threshold=large_threshold)
        adj_out = adj_norm

    return SimpleNamespace(
        name=name,
        feat=x,
        adj=adj_out,
        mark=mark,
        ano_labels=torch.tensor(label, dtype=torch.float).to(device),
        conv_list=conv_list,
        sim_conv=sim_list,
        n=int(label.shape[0]),
    )
