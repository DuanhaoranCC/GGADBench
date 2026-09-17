"""DR-GGAD source-transfer implementation using sparse PyTorch operators.

Features use ARC-style alignment to 64 dimensions, with conditional self-loops
and symmetric normalization. The model trains on source labels and performs
frozen zero-shot inference on target graphs.
"""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint

import ggad.config as C
from common.data import (
    BIG,
    DIMS,
    EDGE_CHUNK,
    NO_SELFLOOP,
    EdgeList,
    adj_sym_cond,
    adj_sym_cond_scipy,
    aggregate,
    chunk_propagate,
    column_degree_row_values,
    edge_chunks,
    exact_edge_spmm,
    load_aligned,
)
from util import evaluate, set_seed, to_torch_sparse

STREAM_GRAPHS = BIG | {"t_finance"}


def _normalize_score_torch(score):
    return (score - score.min()) / (score.max() - score.min() + 1e-8)


def _prototype_alignment_loss(residual_embed, labels, cluster_centers, margin=1):
    dists = torch.cdist(residual_embed, cluster_centers)
    # Official code names this min_dist but uses max over centers.  Keep it.
    min_dist, _ = torch.max(dists, dim=1)
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    normal_loss = min_dist[normal_mask].mean()
    if abnormal_mask.sum() > 0:
        abnormal_dists = dists[abnormal_mask]
        abnormal_loss = F.relu(margin - abnormal_dists).mean()
    else:
        abnormal_loss = torch.tensor(0.0, device=residual_embed.device)
    return normal_loss + abnormal_loss


def _edge_message(feat, r, c, v):
    return (feat[r] * feat[c]).sum(1) * v


def _max_message_sparse(feature, adj_matrix, chunk_edges=EDGE_CHUNK):
    """Sparse equivalent of official max_message without materializing NxN dense.

    Official logic:
      sim = cosine(feature) @ cosine(feature).T
      sim = sim * adj
      row_sum = torch.sum(adj, 0)
      message = torch.sum(sim, 1) * row_sum^{-1}
      loss = mean((1 - message)^2)

    Small graphs retain the sparse official-equivalent path. Dense/large source
    graphs use the same exact differentiable edge-list SpMM as IA-GGAD: forward
    and backward both stream every edge while keeping the full node state.
    """
    feat = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-12)
    n = feature.size(0)
    device = feature.device
    if torch.is_grad_enabled() and isinstance(adj_matrix, EdgeList):
        loss_val = column_degree_row_values(adj_matrix)
        aggregate_feat = exact_edge_spmm(adj_matrix, feat, val=loss_val, chunk=chunk_edges)
        message = torch.sum(feat * aggregate_feat, dim=1)
        return torch.mean((1 - message) ** 2), message
    message_sum = torch.zeros(n, device=device)
    col_sum = torch.zeros(n, device=device)
    grad = torch.is_grad_enabled()
    for r, c, v in edge_chunks(adj_matrix, device, chunk_edges):
        if grad:

            sim = checkpoint(_edge_message, feat, r, c, v, use_reentrant=False)
        else:
            sim = (feat[r] * feat[c]).sum(1) * v
        message_sum = message_sum.index_add(0, r, sim)
        col_sum.index_add_(0, c, v)
    r_inv = torch.pow(col_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = message_sum * r_inv
    return torch.mean((1 - message) ** 2), message


class DR(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=1024,
        num_layers=4,
        dropout_rate=0.2,
        activation="ELU",
        num_hops=2,
        alpha=0.1,
        k=1,
        edge_chunk=500_000,
        node_chunk=32_768,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.K = k
        self.alpha = alpha
        self.edge_chunk = edge_chunk
        self.node_chunk = node_chunk
        if num_layers > 0:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for _ in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))

        self.dropout = nn.Dropout(0.2)
        self.cluster_centers_dict = {}
        self.cluster_mlps = nn.ModuleDict()
        self._initialized_datasets = set()
        self.node_mlps = nn.Sequential(
            nn.Linear(in_feats, in_feats * 2),
            nn.Dropout(0.0),
            nn.Linear(in_feats * 2, in_feats),
            nn.BatchNorm1d(in_feats),
        )

    def compute_residual_prototypes(self, h, normal_idx, dataset_name):
        x_list = h.x_list
        first = x_list[0]
        residual_embed = torch.hstack([h_i - first for h_i in x_list[1:]])
        h_normal = residual_embed[normal_idx]
        # Strict official DRGGAD behavior. The released main.py fixes k=1,
        # meaning one normal residual prototype per source domain, but it still
        # obtains that prototype through sklearn KMeans.
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(h_normal.detach().cpu().numpy())
        self.cluster_centers_dict[dataset_name] = torch.as_tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=residual_embed.device
        )
        if dataset_name not in self._initialized_datasets:
            self.cluster_mlps[dataset_name] = nn.Linear(
                self.in_feats * self.num_hops, self.h_feats * self.num_hops
            ).to(h.x.device)
            self._initialized_datasets.add(dataset_name)

    def forward(self, h):
        x_list = list(h.x_list)
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
        first = x_list[0]
        residual_embed = torch.hstack([h_i - first for h_i in x_list[1:]])
        node_embed = self.node_mlps(h.x)
        return residual_embed, node_embed

    def computer_loss(self, h, name):
        residual_embed, node_embed = self.forward(h)
        centers = self.cluster_mlps[name](self.cluster_centers_dict[name])
        proto_loss = _prototype_alignment_loss(residual_embed, h.ano_labels, centers, margin=1)
        art_loss, _ = _max_message_sparse(node_embed, h.adj_ori, self.edge_chunk)
        return proto_loss + art_loss

    def tam_score_euclidean(self, adj, node_feat, skip_self_loops=False):
        # adj is a GPU torch.sparse (small graphs) or a CPU EdgeList (big targets);
        # edge_chunks streams (row,col,val) either way. Only the edge set is used.
        n = node_feat.size(0)
        device = node_feat.device
        score_sum = torch.zeros(n, device=device)
        degree = torch.zeros(n, device=device)
        for r, c, _v in edge_chunks(adj, device, self.edge_chunk):
            if skip_self_loops:
                keep = r != c
                r = r[keep]
                c = c[keep]
                if r.numel() == 0:
                    continue
            dist = torch.norm(node_feat[r] - node_feat[c], p=2, dim=1)
            score_sum.index_add_(0, r, dist)
            degree.index_add_(0, r, torch.ones_like(dist))
        anomaly_score = score_sum / (degree + 1e-8)
        return _normalize_score_torch(anomaly_score)

    def _forward_chunk(self, h, start, end):
        x_list = [x[start:end] for x in h.x_list]
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
        first = x_list[0]
        residual_embed = torch.hstack([h_i - first for h_i in x_list[1:]])
        node_embed = self.node_mlps(h.x[start:end])
        return residual_embed, node_embed

    @torch.no_grad()
    def _chunked_residual_score_and_node_embed(self, h, cluster_centers):
        n = h.n
        score_dis = torch.empty(n, dtype=torch.float32, device=h.x.device)
        node_embed = torch.empty((n, self.in_feats), dtype=torch.float32, device=h.x.device)
        for start in range(0, n, self.node_chunk):
            end = min(start + self.node_chunk, n)
            residual, node = self._forward_chunk(h, start, end)
            dists = torch.cdist(residual, cluster_centers)
            score_dis[start:end] = dists.max(dim=1)[0]
            node_embed[start:end] = node
        return _normalize_score_torch(score_dis), node_embed

    def get_all_projected_cluster_centers(self):
        projected = []
        for name, centers in self.cluster_centers_dict.items():
            projected.append(self.cluster_mlps[name](centers))
        return torch.cat(projected, dim=0)

    @torch.no_grad()
    def get_anomaly_score(self, h):
        cluster_centers = self.get_all_projected_cluster_centers()
        # For million-node targets, full forward would materialize
        # N x (num_hops*h_feats) residuals.  With T-Social and h_feats=1024
        # that is >40GB, so score nodes in chunks while preserving all nodes.
        if h.n > self.node_chunk:
            score_dis, node_embed = self._chunked_residual_score_and_node_embed(h, cluster_centers)
        else:
            residual_embed, node_embed = self.forward(h)
            dists = torch.cdist(residual_embed, cluster_centers)
            score_dis = _normalize_score_torch(dists.max(dim=1)[0])

        t_score_dis = self.tam_score_euclidean(h.adj, node_embed, skip_self_loops=True)

        _, tam_score_cos = _max_message_sparse(node_embed, h.adj_ori, self.edge_chunk)
        t_score_cos = 1 - _normalize_score_torch(tam_score_cos)

        return score_dis * self.alpha + (t_score_cos + t_score_dis) * (1 - self.alpha)


def _move_model(model, device):
    """Move registered state plus the official plain-dict prototype tensors."""
    model.to(device)
    model.cluster_centers_dict = {
        name: centers.to(device) for name, centers in model.cluster_centers_dict.items()
    }
    return model


def _adj_ori_cond(name, adj, device):
    a = adj if name in NO_SELFLOOP else adj + sp.eye(adj.shape[0])
    return to_torch_sparse(a).to(device)


def _graph(name, num_hops, device, target=False, edge_chunk=500_000):
    adj, feat, label, mark = load_aligned(name, target=target, hops=num_hops)
    if name in STREAM_GRAPHS:
        x_list = chunk_propagate(name, adj, feat, num_hops, device, chunk=edge_chunk)
        x = x_list[0]
        # Source training never reads h.adj. Avoid a second 42M-edge CPU table
        # for t_finance; target scoring still builds the exact normalized one.
        adj_norm = EdgeList(adj_sym_cond_scipy(name, adj)) if target else None
        adj_ori = EdgeList(adj if name in NO_SELFLOOP else adj + sp.eye(adj.shape[0]))
    else:
        adj_norm = adj_sym_cond(name, adj, device)
        adj_ori = _adj_ori_cond(name, adj, device)
        x = torch.from_numpy(np.ascontiguousarray(feat, np.float32)).to(device)
        x_list = [x]
        for _ in range(num_hops):
            x_list.append(torch.sparse.mm(adj_norm, x_list[-1]))
    return SimpleNamespace(
        name=name,
        x=x,
        x_list=x_list,
        adj=adj_norm,
        adj_ori=adj_ori,
        labels_np=np.asarray(label),
        mark=mark,
        ano_labels=torch.tensor(label, dtype=torch.float, device=device),
        n=len(label),
    )


def _initialize_random_prototype(model, device):
    """Create the minimum random prototype state required by DR-GGAD scoring.

    This is used only by the zero-update control.  It does not inspect source
    features or labels and therefore keeps the model genuinely untrained.
    """
    key = "no_train_random_init"
    input_dim = model.in_feats * model.num_hops
    output_dim = model.h_feats * model.num_hops
    bound = input_dim**-0.5
    model.cluster_centers_dict[key] = torch.empty(model.K, input_dim, device=device).uniform_(
        -bound, bound
    )
    model.cluster_mlps[key] = nn.Linear(input_dim, output_dim).to(device)
    model._initialized_datasets.add(key)


def run_drggad(sources, targets, seeds, epochs, device, train=True, target_evaluator=None):
    hp = dict(C.DRGGAD_HP)
    hp["epoch"] = epochs
    nh = hp["num_hops"]
    edge_chunk = hp.get("edge_chunk", 500_000)
    src_g = []
    if train:
        for source in sources:
            print(f"    [drggad/build-source] {source} begin", flush=True)
            src_g.append(_graph(source, nh, device, target=False, edge_chunk=edge_chunk))
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            print(f"    [drggad/build-source] {source} done", flush=True)
    per = {n: [] for n in targets}
    trained = []

    for seed in seeds:
        set_seed(seed)
        stage = "train" if train else "random-init"
        print(f"    [drggad/{stage}] seed={seed} sources={sources}", flush=True)
        model = DR(
            in_feats=DIMS,
            h_feats=hp["h_feats"],
            num_layers=hp["num_layers"],
            dropout_rate=hp["drop_rate"],
            activation=hp["activation"],
            num_hops=nh,
            alpha=hp["alpha"],
            k=hp["k"],
            edge_chunk=hp.get("edge_chunk", 500_000),
            node_chunk=hp.get("node_chunk", 32_768),
        ).to(device)
        if train:
            # Strict official order: optimizer is created before cluster_mlps are
            # dynamically initialized during epoch 0.
            opt = Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            model.train()
            for e in range(hp["epoch"]):
                for g in src_g:

                    if e == 0:
                        print(
                            f"    [drggad/prototype] seed={seed} source={g.name} begin", flush=True
                        )
                        model.compute_residual_prototypes(g, g.ano_labels == 0, g.name)
                        if str(device).startswith("cuda"):
                            torch.cuda.synchronize(device)
                        print(
                            f"    [drggad/prototype] seed={seed} source={g.name} done", flush=True
                        )
                    opt.zero_grad(set_to_none=True)
                    loss = model.computer_loss(g, g.name)
                    loss.backward()
                    opt.step()
                    # Expose a failed device at the exact source/epoch instead of
                    # letting an asynchronous CUDA error surface much later.
                    if isinstance(g.adj_ori, EdgeList) and str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    del loss
                if e == 0 or (e + 1) % 10 == 0 or e + 1 == hp["epoch"]:
                    print(f"    [drggad/train] seed={seed} epoch={e + 1}/{hp['epoch']}", flush=True)
        else:
            _initialize_random_prototype(model, device)
        model.eval()
        model.zero_grad(set_to_none=True)
        _move_model(model, "cpu")
        trained.append((seed, model))
        if train:
            del opt
        torch.cuda.empty_cache()
    for g in src_g:
        del g.x, g.x_list, g.adj, g.adj_ori, g.ano_labels
    src_g.clear()
    torch.cuda.empty_cache()

    for name in targets:
        try:
            g = _graph(name, nh, device, target=True, edge_chunk=edge_chunk)
        except RuntimeError as e:
            if name in BIG:
                raise
            print(f"    [drggad] {name} build skipped ({str(e)[:80]})")
            torch.cuda.empty_cache()
            continue
        for seed, model in trained:
            _move_model(model, device)
            set_seed(seed)
            with torch.no_grad():
                try:
                    scores = model.get_anomaly_score(g)
                except RuntimeError as e:
                    if name in BIG:
                        raise
                    print(f"    [drggad] {name} seed={seed} skipped ({str(e)[:80]})")
                    _move_model(model, "cpu")
                    torch.cuda.empty_cache()
                    continue
            score_array = scores.detach().cpu().numpy()
            per[name].append(
                target_evaluator(name, int(seed), g.labels_np, score_array, g.mark)
                if target_evaluator is not None
                else evaluate(g.labels_np, score_array, g.mark)
            )
            _move_model(model, "cpu")
            torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()
    return aggregate(per)
