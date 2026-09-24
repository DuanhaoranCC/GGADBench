"""TAM (NeurIPS'23) strict baseline core.

This module ports the official TAM implementation:

- model: two-layer GCN LAMNet from official `model.py`;
- objective: maximize local affinity on iteratively truncated graphs;
- score: `1 - minmax(local affinity)` so larger means more anomalous.

Callers provide a graph and method-specific aligned features.
The training/scoring logic follows
`TAM/train.py`.
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from baselines.runtime import print_progress


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, seq.shape[-1])
        out = torch.mul(seq, sim)
        return torch.sum(out, 1)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = [self.f_k(h_pl, c)]
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        return torch.cat(tuple(scs))


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super().__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, 2 * n_h, activation)
        self.gcn2 = GCN(2 * n_h, n_h, activation)
        self.act = nn.PReLU()
        self.fc1 = nn.Linear(n_h, 2 * n_h, bias=False)
        self.fc2 = nn.Linear(n_h, 2 * n_h, bias=False)
        self.ReLU = nn.ReLU()
        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "avg":
            self.read = AvgReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout()
        else:
            raise ValueError(f"unknown readout: {readout}")

    def forward(self, seq, adj, sparse=False):
        feat = self.gcn1(seq, adj)
        feat = self.gcn2(feat, adj)
        feat1 = self.fc1(feat)
        feat2 = self.fc2(feat)
        return feat, feat1, feat2


def normalize_score(ano_score):
    lo, hi = np.min(ano_score), np.max(ano_score)
    den = hi - lo
    if den == 0:
        return np.zeros_like(ano_score)
    return (ano_score - lo) / den


def normalize_adj_tensor(raw_adj):
    adj = raw_adj[0, :, :]
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    return adj.unsqueeze(0)


def calc_distance(adj, seq, chunk_edges=2_000_000):
    """Official `calc_distance`, vectorized over nonzero adjacency entries."""
    nz = torch.argwhere(adj > 0)
    dis_array = torch.zeros_like(adj)
    for start in range(0, nz.shape[0], chunk_edges):
        idx = nz[start : start + chunk_edges]
        src, dst = idx[:, 0], idx[:, 1]
        dis = torch.sqrt(torch.sum((seq[src] - seq[dst]) * (seq[src] - seq[dst]), dim=1))
        dis_array[src, dst] = dis
    return dis_array


def graph_nsgt(dis_array, adj):
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    valid = dis_array_u[dis_array_u != 0]
    mean_dis = valid.mean() if valid.numel() else torch.tensor(0.0, device=adj.device)
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            vals = dis_array[i, node_index[:, 0]]
            max_dis = vals.max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(vals > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = 1 - adj
    sim_u_u = sim_u_u * adj_inverse
    sim_u_u_no_diag = torch.sum(sim_u_u, 1)
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    sim_u_u_no_diag = sim_u_u_no_diag * r_inv
    return torch.sum(sim_u_u_no_diag)


def max_message(feature, adj_matrix, normalize_inside=False):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    if normalize_inside:
        message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    return -torch.sum(message), message


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = torch.sum(sim_matrix, 1)
    return message * r_inv


def _scipy_to_torch_sparse(mx, device):
    mx = sp.coo_matrix(mx).astype(np.float32)
    idx = torch.from_numpy(np.vstack([mx.row, mx.col]).astype(np.int64))
    val = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(mx.shape), device=device).coalesce()


def _normalize_adj_sparse(adj):
    # Exact sparse counterpart of normalize_adj_tensor: that function computes
    # degrees along dim=0 and returns D @ A @ D (without transposing A).
    adj = sp.csr_matrix(adj).astype(np.float32)
    colsum = np.asarray(adj.sum(0)).flatten()
    d_inv_sqrt = np.zeros_like(colsum, dtype=np.float32)
    nz = colsum > 0
    d_inv_sqrt[nz] = np.power(colsum[nz], -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    return d.dot(adj).dot(d).tocoo()


def _distance_graph_sparse(adj, feat, chunk_edges=2_000_000):
    """Sparse representation of official ``calc_distance`` and its mean."""
    a = sp.csr_matrix(adj, dtype=np.float32)
    coo = a.tocoo()
    feat = np.asarray(feat, dtype=np.float32)
    distance = np.empty(coo.nnz, dtype=np.float32)
    for start in range(0, coo.nnz, chunk_edges):
        end = min(start + chunk_edges, coo.nnz)
        delta = feat[coo.row[start:end]] - feat[coo.col[start:end]]
        distance[start:end] = np.sqrt(np.sum(delta * delta, axis=1))
    dis = sp.csr_matrix((distance, (coo.row, coo.col)), shape=a.shape, dtype=np.float32)
    weighted = distance * coo.data
    valid = weighted[weighted != 0]
    mean_dis = float(valid.mean()) if valid.size else 0.0
    return dis, mean_dis


def _active_distance_mean(dis_graph, adj):
    current = sp.coo_matrix(adj, dtype=np.float32)
    if current.nnz == 0:
        return 0.0
    distances = np.asarray(dis_graph[current.row, current.col]).reshape(-1)
    weighted = distances * current.data
    valid = weighted[weighted != 0]
    return float(valid.mean()) if valid.size else 0.0


def _graph_nsgt_sparse(dis_graph, adj, mean_dis):
    """Exact sparse form of official graph_nsgt, including RNG call order."""
    a = sp.coo_matrix(adj, dtype=np.float32)
    if a.nnz == 0:
        return a.tocsr()
    distances = np.asarray(dis_graph[a.row, a.col]).reshape(-1)
    row_max = np.full(a.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(row_max, a.row, distances)
    eligible = row_max > mean_dis
    threshold = np.full(a.shape[0], np.inf, dtype=np.float32)
    eligible_rows = np.flatnonzero(eligible)
    if eligible_rows.size:
        random_values = np.random.random_sample(eligible_rows.size)
        threshold[eligible_rows] = (row_max[eligible_rows] - mean_dis) * random_values + mean_dis
    keep = distances <= threshold[a.row]
    cut = sp.csr_matrix((a.data[keep], (a.row[keep], a.col[keep])), shape=a.shape, dtype=np.float32)
    cut = (cut + cut.transpose()).tocsr()
    cut.data[cut.data > 1] = 1
    cut.eliminate_zeros()
    return cut


def _gcn_layer_sparse(layer, x, adj_norm):
    xw = layer.fc(x)
    out = torch.sparse.mm(adj_norm, xw)
    if layer.bias is not None:
        out = out + layer.bias
    return layer.act(out)


def _model_forward_sparse(model, x, adj_norm):
    feat = _gcn_layer_sparse(model.gcn1, x, adj_norm)
    feat = _gcn_layer_sparse(model.gcn2, feat, adj_norm)
    feat1 = model.fc1(feat)
    feat2 = model.fc2(feat)
    return feat, feat1, feat2


def inference_sparse(feature, adj_sparse, chunk_edges=2_000_000):
    feature = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-12)
    adj_sparse = adj_sparse.coalesce()
    row, col = adj_sparse.indices()
    val = adj_sparse.values()
    n = feature.size(0)
    message = torch.zeros(n, device=feature.device)
    degree = torch.zeros(n, device=feature.device)
    for start in range(0, row.numel(), chunk_edges):
        r = row[start : start + chunk_edges]
        c = col[start : start + chunk_edges]
        v = val[start : start + chunk_edges]
        sim = (feature[r] * feature[c]).sum(1) * v
        message.index_add_(0, r, sim)
        degree.index_add_(0, c, v)
    r_inv = torch.pow(degree, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    return message * r_inv


def _edge_dot(feat, r, c, v):
    return (feat[r] * feat[c]).sum(1) * v


def max_message_sparse(feature, adj_sparse, chunk_edges=500_000):
    """Sparse differentiable equivalent of TAM max_message.

    Used only for dense-edge source training where official dense NxN tensors
    are impossible.  Edges are all retained; checkpointing avoids keeping the
    full E×hidden gather graph for backward.
    """
    feature = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-12)
    adj_sparse = adj_sparse.coalesce()
    row, col = adj_sparse.indices()
    val = adj_sparse.values()
    n = feature.size(0)
    message = torch.zeros(n, device=feature.device)
    degree = torch.zeros(n, device=feature.device)
    grad = torch.is_grad_enabled()
    for start in range(0, row.numel(), chunk_edges):
        end = min(start + chunk_edges, row.numel())
        r = row[start:end]
        c = col[start:end]
        v = val[start:end]
        if grad:
            sim = checkpoint(_edge_dot, feature, r, c, v, use_reentrant=False)
        else:
            sim = (feature[r] * feature[c]).sum(1) * v
        message = message.index_add(0, r, sim)
        degree.index_add_(0, c, v)
    r_inv = torch.pow(degree, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = message * r_inv
    return -torch.sum(message), message


def reg_edge_sparse(emb, adj_sparse):
    """Exact O(E + N*D) form of TAM's dense complement regularizer.

    ``(Z Z.T * (1-A)).sum(1)`` is evaluated algebraically, without ever
    materializing either dense N×N matrix.
    """
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    neighbor_sum = torch.sparse.mm(adj_sparse, emb)
    all_sum = emb.sum(dim=0)
    sim_all = (emb * all_sum).sum(dim=1)
    sim_adj = (emb * neighbor_sum).sum(dim=1)
    row_sum = emb.shape[0] - torch.sparse.sum(adj_sparse, dim=1).to_dense()
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    return torch.sum((sim_all - sim_adj) * r_inv)


def _graph_tensors(adj, feat, device):
    feat = np.asarray(feat, dtype=np.float32)
    n = feat.shape[0]
    raw_adj_np = (sp.csr_matrix(adj) + sp.eye(n, dtype=np.float32)).toarray().astype(np.float32)
    features = torch.FloatTensor(feat[np.newaxis]).to(device)
    raw_features = torch.FloatTensor(feat[np.newaxis]).to(device)
    raw_adj = torch.FloatTensor(raw_adj_np[np.newaxis]).to(device)
    return features, raw_features, raw_adj


def init_ensemble(
    input_dim,
    device,
    *,
    embedding_dim=128,
    cutting=4,
    n_tree=3,
    negsamp_ratio=2,
    readout="avg",
    lr=1e-5,
    weight_decay=0.0,
):
    models, opts = [], []
    for _ in range(cutting * n_tree):
        model = Model(input_dim, embedding_dim, "prelu", negsamp_ratio, readout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        models.append(model)
        opts.append(opt)
    return models, opts


def train_ensemble_on_graph(
    models,
    opts,
    adj,
    feat,
    device,
    *,
    num_epoch=500,
    cutting=4,
    n_tree=3,
    lamda=0.0,
    large_variant=False,
    progress_label=None,
):
    """Train an existing TAM ensemble on one source graph.

    This follows official `train.py`: a fixed ensemble of K*T LAMNets is trained
    on the source graph's iteratively truncated adjacency matrices. Calling this
    repeatedly over multiple source graphs implements the benchmark multi-source
    protocol while keeping the official LAMNet/objective unchanged.
    """
    features, raw_features, raw_adj = _graph_tensors(adj, feat, device)
    all_cut_adj = torch.cat([raw_adj for _ in range(n_tree)])
    dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])

    index = 0
    progress_total = cutting * n_tree * num_epoch
    progress_every = max(1, progress_total // 20)
    started = time.perf_counter()
    for _ in range(cutting):
        for n_t in range(n_tree):
            cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :]).unsqueeze(0)
            adj_norm = normalize_adj_tensor(cut_adj)
            model, opt = models[index], opts[index]
            model.train()
            opt.zero_grad()
            for _epoch in range(num_epoch):
                node_emb, feat1, _feat2 = model.forward(features, adj_norm)
                loss, _ = max_message(
                    node_emb[0, :, :], raw_adj[0, :, :], normalize_inside=large_variant
                )
                reg_loss = reg_edge(feat1[0, :, :], raw_adj[0, :, :])
                loss = loss + lamda * reg_loss
                loss.backward()
                opt.step()
                completed = index * num_epoch + _epoch + 1
                if progress_label is not None and (
                    completed == 1 or completed % progress_every == 0 or completed == progress_total
                ):
                    print_progress("tam", progress_label, completed, progress_total, started)
            all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
            index += 1


def train_ensemble_on_graph_sparse(
    models,
    opts,
    adj,
    feat,
    device,
    *,
    num_epoch=500,
    cutting=4,
    n_tree=3,
    lamda=0.0,
    large_variant=False,
    progress_label=None,
):
    """Memory-safe but lossless TAM training for dense-edge source graphs.

    All official operations are retained: iterative randomized NSGT cutting,
    normalized GCN propagation, local-affinity loss and the non-edge
    regularizer.  Sparse algebra removes only dense N×N materialization.
    """
    feat = np.asarray(feat, dtype=np.float32)
    n = feat.shape[0]
    raw_adj_sp = (sp.csr_matrix(adj) + sp.eye(n, dtype=np.float32, format="csr")).astype(np.float32)
    dis_graph, _ = _distance_graph_sparse(raw_adj_sp, feat)
    x = torch.as_tensor(feat, dtype=torch.float32, device=device)
    raw_adj = _scipy_to_torch_sparse(raw_adj_sp, device)
    cut_graphs = [raw_adj_sp.copy() for _ in range(n_tree)]

    index = 0
    progress_total = cutting * n_tree * num_epoch
    progress_every = max(1, progress_total // 20)
    started = time.perf_counter()
    for _ in range(cutting):
        for n_t in range(n_tree):
            mean_dis = _active_distance_mean(dis_graph, cut_graphs[n_t])
            cut_graphs[n_t] = _graph_nsgt_sparse(dis_graph, cut_graphs[n_t], mean_dis)
            adj_norm = _scipy_to_torch_sparse(_normalize_adj_sparse(cut_graphs[n_t]), device)
            model, opt = models[index], opts[index]
            model.train()
            # Official train.py zeros gradients once per LAMNet, outside its
            # epoch loop.  Preserve that unusual but published behavior.
            opt.zero_grad()
            for _epoch in range(num_epoch):
                node_emb, feat1, _feat2 = _model_forward_sparse(model, x, adj_norm)
                loss, _ = max_message_sparse(node_emb, raw_adj, chunk_edges=500_000)
                if lamda:
                    loss = loss + lamda * reg_edge_sparse(feat1, raw_adj)
                loss.backward()
                opt.step()
                completed = index * num_epoch + _epoch + 1
                if progress_label is not None and (
                    completed == 1 or completed % progress_every == 0 or completed == progress_total
                ):
                    print_progress("tam", progress_label, completed, progress_total, started)
            index += 1


def prepare_tam_graph(adj, feat, device, sparse=False):
    """Prepare a target graph once for all TAM seed ensembles."""
    if not sparse:
        features, _raw_features, raw_adj = _graph_tensors(adj, feat, device)
        return {
            "sparse": False,
            "features": features,
            "raw_adj": raw_adj,
            "adj_norm": normalize_adj_tensor(raw_adj),
        }
    feat = np.asarray(feat, dtype=np.float32)
    n = feat.shape[0]
    raw_adj_sp = (sp.csr_matrix(adj) + sp.eye(n, dtype=np.float32, format="csr")).astype(np.float32)
    return {
        "sparse": True,
        "features": torch.as_tensor(feat, dtype=torch.float32, device=device),
        "raw_adj": _scipy_to_torch_sparse(raw_adj_sp, device),
        "adj_norm": _scipy_to_torch_sparse(_normalize_adj_sparse(raw_adj_sp), device),
    }


def score_with_prepared_ensemble(models, graph, *, cutting=4, n_tree=3, progress_label=None):
    """Score one ensemble without rebuilding target tensors/normalization."""
    features = graph["features"]
    raw_adj = graph["raw_adj"]
    adj_norm = graph["adj_norm"]
    index = 0
    progress_total = cutting * n_tree
    started = time.perf_counter()
    message_mean_list = []
    for _ in range(cutting):
        message_list = []
        for _n_t in range(n_tree):
            model = models[index]
            model.eval()
            with torch.no_grad():
                if graph["sparse"]:
                    node_emb, _feat1, _feat2 = _model_forward_sparse(model, features, adj_norm)
                    message_sum = inference_sparse(node_emb, raw_adj)
                else:
                    node_emb, _feat1, _feat2 = model.forward(features, adj_norm)
                    message_sum = inference(node_emb[0, :, :], raw_adj[0, :, :])
            message_list.append(torch.unsqueeze(message_sum, 0))
            index += 1
            if progress_label is not None:
                print_progress("tam", progress_label, index, progress_total, started)
        message_list = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(torch.unsqueeze(message_list, 0))
    message_mean = torch.mean(torch.cat(message_mean_list), 0)
    score = 1 - normalize_score(message_mean.detach().cpu().numpy())
    return np.asarray(score, dtype=np.float64)
