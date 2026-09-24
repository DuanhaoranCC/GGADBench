"""GraphPrompt node-downstream runner for gfm.

The required official GraphPrompt logic is ported into this self-contained runner.

Reproduced core:
  * GIN encoder with sum aggregation, 3 hidden layers, BN/ReLU/dropout.
  * Node prompt layer FEATURE-WEIGHTED-SUM (learned feature-wise weight).
  * Prototype/center distance classifier:
        center = mean(prompted support embedding per class)
        distance = squared Euclidean distance to centers
        pred = log_softmax(-normalize(distance), dim=1)

Benchmark adaptation:
  * The encoder and prompt train on source labels, then target few-shot support
    centers define the classifier. The upstream code loads a pre-trained encoder.
  * Target evaluation keeps all `mark=True` nodes except support; scores are
    produced in chunks.  No eval-node sampling.
  * Propagation uses torch/scipy edge-index operations with edge chunks for
    large graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

BENCH_ROOT = Path(__file__).resolve().parents[2]

from common.data import EdgeList, load_source_marked, load_target_marked
from gfm.runners.ego_neighbor_residual import (
    STREAM_GRAPHS,
    EgoNeighborResidualEncoder,
    arc_operator,
    edge_spmm_transpose,
)
from gfm.runners.feature_adapter import adapted_features
from gfm.runners.few_shot import sample_class_support
from util import aggregate, evaluate, set_seed, to_torch_sparse


@dataclass
class GraphCtx:
    name: str
    x: np.ndarray
    adj: sp.csr_matrix
    labels: np.ndarray
    mark: np.ndarray


def _ctx(name: str, feature_dim: int, feature_norm: str, target: bool) -> GraphCtx:
    if target:
        adj, feat, labels, mark = load_target_marked(name)
    else:
        adj, feat, labels, mark = load_source_marked(name)
    adj = sp.csr_matrix(adj).astype(np.float32)
    return GraphCtx(
        name=name,
        x=adapted_features(name, feat, adj, feature_dim, feature_norm),
        adj=adj,
        labels=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
    )


class GINEncoder(nn.Module):
    """PyTorch port of official nodedownstream/gin.py::GIN."""

    def __init__(
        self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            din = in_dim if i == 0 else hidden_dim
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(din, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def _layer(self, x: torch.Tensor, adj_torch, layer: int, edge_chunk: int):
        # DGL GINConv(sum) aggregates predecessor messages, then applies MLP.
        if isinstance(adj_torch, EdgeList):
            # Exact full-edge forward/backward with bounded edge memory. This is
            # A.T @ x because adjacency rows are sources and columns are targets.
            agg = edge_spmm_transpose(adj_torch, x, edge_chunk)
        else:
            agg = torch.sparse.mm(adj_torch.transpose(0, 1), x)
        h = F.relu(self.mlps[layer](agg))
        h = self.bns[layer](h)
        h = self.dropout(h)
        return h

    def forward(self, x: torch.Tensor, adj_torch, edge_chunk: int = 2_000_000):
        xs = []
        h = x
        for i in range(self.num_layers):
            h = self._layer(h, adj_torch, i, edge_chunk)
            xs.append(h)
        return torch.cat(xs, dim=1)


class FeatureWeightedPrompt(nn.Module):
    """Official node_prompt_layer_feature_weighted_sum."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor):
        return h * self.weight


class GraphPromptModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        encoder_type: str = "native",
        enr_hp: dict | None = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "native":
            out_dim = hidden_dim * num_layers
            self.encoder = GINEncoder(in_dim, hidden_dim, num_layers, dropout)
        elif encoder_type == "ego_neighbor_residual":
            if enr_hp is None:
                raise ValueError("enr_hp is required for GraphPrompt ENR variant")
            self.encoder = EgoNeighborResidualEncoder(
                in_dim,
                int(enr_hp["hidden_dim"]),
                int(enr_hp["num_layers"]),
                int(enr_hp["num_hops"]),
                dropout=float(enr_hp["dropout"]),
                activation=enr_hp["activation"],
            )
            out_dim = self.encoder.output_dim
        else:
            raise ValueError(f"unknown encoder_type={encoder_type}")
        self.prompt = FeatureWeightedPrompt(out_dim)

    @staticmethod
    def propagate(h: torch.Tensor, adj_torch, edge_chunk: int):
        if isinstance(adj_torch, EdgeList):
            return edge_spmm_transpose(adj_torch, h, edge_chunk)
        return torch.sparse.mm(adj_torch.transpose(0, 1), h)

    def forward(
        self, x: torch.Tensor, adj_torch, edge_chunk: int, nhop_neighbour: int = 1, post_adj=None
    ):
        # Official node downstream pre_train():
        #   x, pred = GIN(...)
        #   pred = sigmoid(pred)
        #   for _ in range(nhop_neighbour): pred = adj @ pred
        h = torch.sigmoid(self.encoder(x, adj_torch, edge_chunk))
        post_adj = adj_torch if post_adj is None else post_adj
        for _ in range(int(nhop_neighbour)):
            h = self.propagate(h, post_adj, edge_chunk)
        return self.prompt(h)


def _center_embedding(emb: torch.Tensor, labels: torch.Tensor, num_classes: int = 2):
    centers = []
    for c in range(num_classes):
        m = labels == c
        if m.sum() == 0:
            centers.append(torch.zeros(emb.size(1), device=emb.device, dtype=emb.dtype))
        else:
            centers.append(emb[m].mean(0))
    return torch.stack(centers, dim=0)


def _distance_logits(emb: torch.Tensor, centers: torch.Tensor):
    # Official: distance2center -> -1 * F.normalize(distance, dim=1) -> log_softmax.
    dist = torch.cdist(emb, centers, p=2) ** 2
    logits = -F.normalize(dist, dim=1)
    return logits


def _adj_for(ctx: GraphCtx, device: str, big: bool = False, enr: bool = False):
    if enr:
        return arc_operator(ctx.name, ctx.adj, device, stream=big)
    if big:
        return EdgeList(ctx.adj)
    return to_torch_sparse(ctx.adj).to(device)


def _sample_class(
    ctx: GraphCtx,
    cls: int,
    k: int,
    rng: np.random.RandomState,
    exclude: Iterable[int] = (),
    *,
    reserve_query: bool = False,
) -> np.ndarray:
    return sample_class_support(
        ctx.labels, ctx.mark, cls, k, rng, exclude, reserve_query=reserve_query, dataset=ctx.name
    )


def _balanced_train_batch(ctxs: Sequence[GraphCtx], rng: np.random.RandomState, per_class: int):
    ctx = ctxs[int(rng.randint(0, len(ctxs)))]
    n = _sample_class(ctx, 0, per_class, rng)
    a = _sample_class(ctx, 1, per_class, rng)
    idx = np.concatenate([n, a])
    lab = np.concatenate([np.zeros(n.size, dtype=np.int64), np.ones(a.size, dtype=np.int64)])
    return ctx, idx, lab


def _train_one_seed(
    model: GraphPromptModel, ctxs: Sequence[GraphCtx], hp: dict, seed: int, device: str
):
    model.train()
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"], amsgrad=True
    )

    # Source graphs are kept full.  Dense source training can still be expensive;
    # this path does not drop source nodes/edges, it only samples labeled query
    # nodes for the prototype loss after full-graph embedding.
    prepared = []
    for ctx in ctxs:
        x = torch.from_numpy(ctx.x).float().to(device)
        is_enr = model.encoder_type == "ego_neighbor_residual"
        stream = ctx.name in STREAM_GRAPHS
        adj = _adj_for(ctx, device, big=stream, enr=is_enr)
        # Native GIN and its post-prompt propagation use the same raw operator.
        # Reuse the CPU edge table for dense graphs instead of duplicating it.
        post_adj = _adj_for(ctx, device, big=stream, enr=False) if is_enr else adj
        prepared.append((ctx, x, adj, post_adj))

    per_class = int(hp["batch_query_per_class"])
    for epoch in range(int(hp["epochs"])):
        ctx, idx_np, lab_np = _balanced_train_batch(ctxs, rng, per_class)
        x = next(v[1] for v in prepared if v[0] is ctx)
        adj = next(v[2] for v in prepared if v[0] is ctx)
        post_adj = next(v[3] for v in prepared if v[0] is ctx)
        emb = model(x, adj, int(hp["edge_chunk"]), int(hp.get("nhop_neighbour", 1)), post_adj)
        idx = torch.from_numpy(idx_np).long().to(device)
        lab = torch.from_numpy(lab_np).long().to(device)
        batch_emb = emb[idx]
        centers = _center_embedding(batch_emb, lab)
        logits = _distance_logits(batch_emb, centers)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        opt.step()
        if (epoch + 1) % max(int(hp["epochs"]) // 5, 1) == 0:
            print(
                f"    [graphprompt] seed={seed} epoch={epoch + 1}/{hp['epochs']} loss={loss.item():.4f}"
            )


@torch.no_grad()
def _embed_graph(model: GraphPromptModel, ctx: GraphCtx, hp: dict, device: str):
    model.eval()
    x = torch.from_numpy(ctx.x).float().to(device)
    big = ctx.name in STREAM_GRAPHS
    if big:
        print(
            f"    [graphprompt-full-large-target] {ctx.name}: eval={int(ctx.mark.sum())} "
            f"N={ctx.adj.shape[0]} E={ctx.adj.nnz} edge_chunk={hp['edge_chunk']} "
            f"query_chunk={hp['eval_query_batch']}"
        )
    adj = _adj_for(ctx, device, big=big, enr=model.encoder_type == "ego_neighbor_residual")
    post_adj = (
        _adj_for(ctx, device, big=big, enr=False)
        if model.encoder_type == "ego_neighbor_residual"
        else adj
    )
    # This is exact full-graph message passing.  `edge_chunk` bounds temporary
    # edge tensors; every node and edge still contributes.
    emb = model(x, adj, int(hp["edge_chunk"]), int(hp.get("nhop_neighbour", 1)), post_adj) * float(
        hp["scalar"]
    )
    return emb


@torch.no_grad()
def _score_target(
    model: GraphPromptModel, ctx: GraphCtx, hp: dict, shot: int, seed: int, device: str
):
    rng = np.random.RandomState(seed + 20_000)
    sup0 = _sample_class(ctx, 0, shot, rng, reserve_query=True)
    sup1 = _sample_class(ctx, 1, shot, rng, reserve_query=True)
    support = np.concatenate([sup0, sup1])
    support_labels = torch.tensor(
        [0] * len(sup0) + [1] * len(sup1), dtype=torch.long, device=device
    )
    support_set = set(int(i) for i in support)
    query = np.asarray(
        [i for i in np.where(ctx.mark)[0] if int(i) not in support_set], dtype=np.int64
    )
    if query.size == 0:
        raise ValueError(f"{ctx.name}: no query nodes after support selection")

    emb = _embed_graph(model, ctx, hp, device)
    centers = _center_embedding(emb[torch.from_numpy(support).long().to(device)], support_labels)
    big = ctx.name in STREAM_GRAPHS
    if big:
        # Keep the exact full embedding, but move it to host memory before
        # query scoring so only one query block returns to the accelerator.
        emb = emb.cpu()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    scores = np.empty(query.size, dtype=np.float32)
    qchunk = int(hp["eval_query_batch"])
    for s in range(0, query.size, qchunk):
        q = query[s : s + qchunk]
        qidx = torch.from_numpy(q).long()
        qe = emb[qidx].to(device) if big else emb[qidx.to(device)]
        logits = _distance_logits(qe, centers)
        scores[s : s + q.size] = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return evaluate(ctx.labels[query], scores)


def run_graphprompt(
    sources,
    targets,
    seeds,
    hp,
    device,
    shot=10,
    feature_dim=8,
    feature_norm="zscore",
    encoder_type="native",
    enr_hp=None,
):
    src_ctxs = [_ctx(n, feature_dim, feature_norm, target=False) for n in sources]
    per = {n: [] for n in targets}

    trained: List[Tuple[int, GraphPromptModel]] = []
    for seed in seeds:
        set_seed(seed)
        model = GraphPromptModel(
            feature_dim,
            int(hp["hidden_dim"]),
            int(hp["num_layers"]),
            float(hp["dropout"]),
            encoder_type,
            enr_hp,
        ).to(device)
        _train_one_seed(model, src_ctxs, hp, seed, device)
        model.eval()
        model.zero_grad(set_to_none=True)
        model.cpu()
        trained.append((seed, model))
        torch.cuda.empty_cache()

    for name in targets:
        ctx = _ctx(name, feature_dim, feature_norm, target=True)
        for seed, model in trained:
            model.to(device)
            try:
                set_seed(seed)
                per[name].append(_score_target(model, ctx, hp, shot, seed, device))
            finally:
                model.cpu()
                torch.cuda.empty_cache()
        del ctx
    return aggregate(per)
