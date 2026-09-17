"""SAMGPT runner for gfm.

The required official SAMGPT logic is ported into this self-contained runner.

Reproduced core:
  * GCN layer: Linear(no bias) -> normalized sparse adjacency propagation ->
    bias -> PReLU.
  * PrePrompt: per-source feature prompts and per-layer structure prompts.
  * GRAPHCL-style pretraining on source graphs.
  * Downstream prompt: composed pretext prompt + open prompt, with target
    few-shot class prototypes and cosine-similarity classifier.

Benchmark adaptation:
  * Node features come from the benchmark's shared SVD feature adapter.
  * Target uses k normal + k anomaly support nodes, then scores every remaining
    marked target node.  No eval-node sampling is used.
  * Large targets use exact edge streaming (`EdgeList` + `edge_chunk`) for
    sparse propagation and query chunks for scoring.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

BENCH_ROOT = Path(__file__).resolve().parents[2]

from common.data import NO_SELFLOOP, load_source_marked, load_target_marked
from gfm.runners.ego_neighbor_residual import (
    STREAM_GRAPHS,
    CompactEdgeList,
    EgoNeighborResidualEncoder,
    edge_spmm,
)
from gfm.runners.feature_adapter import adapted_features
from gfm.runners.few_shot import sample_class_support
from util import aggregate, evaluate, set_seed, to_torch_sparse


@dataclass
class GraphCtx:
    name: str
    x: np.ndarray
    adj: sp.csr_matrix
    adj_norm: sp.coo_matrix | None
    labels: np.ndarray
    mark: np.ndarray
    enr_adj_norm: sp.coo_matrix | None = None
    difference_adj_norm: sp.coo_matrix | None = None


def _normalize_transposed_sym(adj: sp.spmatrix, *, add_self_loops: bool) -> sp.coo_matrix:
    """Memory-bounded D^-1/2 A.T D^-1/2 with optional unit self-loops."""
    a = sp.csr_matrix(adj).astype(np.float32, copy=False)
    if add_self_loops:
        a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    a = a.tocoo()
    rowsum = np.asarray(a.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum > 0
    d_inv_sqrt[nz] = np.power(rowsum[nz], -0.5)
    # Algebraically identical to a.dot(D).T.dot(D), without constructing two
    # additional E-sized scipy sparse intermediates.  SAMGPT propagates with
    # the transposed normalized operator, hence output row=original col.
    weight = a.data.astype(np.float32, copy=False) * d_inv_sqrt[a.row] * d_inv_sqrt[a.col]
    return sp.coo_matrix((weight, (a.col, a.row)), shape=a.shape, dtype=np.float32)


def _normalize_adj_official(adj: sp.spmatrix) -> sp.coo_matrix:
    """SAMGPT utils/process.py::normalize_adj(adj + I)."""
    return _normalize_transposed_sym(adj, add_self_loops=True)


def _ctx(
    name: str, feature_dim: int, feature_norm: str, target: bool, *, build_native_norm: bool = True
) -> GraphCtx:
    if target:
        adj, feat, labels, mark = load_target_marked(name)
    else:
        adj, feat, labels, mark = load_source_marked(name)
    adj = sp.csr_matrix(adj).astype(np.float32, copy=False)
    return GraphCtx(
        name=name,
        x=adapted_features(name, feat, adj, feature_dim, feature_norm),
        adj=adj,
        adj_norm=(_normalize_adj_official(adj) if build_native_norm else None),
        labels=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
    )


def _spmm(adj_torch, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
    # Identity propagation is the ego-only branch used by the anomaly-aware
    # downstream adapter. Keeping it implicit avoids allocating an N-by-N
    # identity sparse tensor on large targets.
    if adj_torch is None:
        return x
    # The shared implementation is torch.sparse.mm for ordinary graphs and an
    # exact custom forward/backward for CPU-resident EdgeList chunks.
    return edge_spmm(adj_torch, x, edge_chunk)


class TextPrompt(nn.Module):
    """Port of SAMGPT layers/prompt.py::textprompt."""

    def __init__(self, hid_units: int, type_: str = "mul"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, hid_units))
        self.prompttype = type_
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompttype == "add":
            return x + self.weight
        if self.prompttype == "mul":
            return x * self.weight
        raise ValueError(f"unknown prompt type {self.prompttype}")


class WeightedPrompt(nn.Module):
    """Port of SAMGPT layers/prompt.py::weighted_prompt."""

    def __init__(self, weightednum: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, weightednum))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(xs) == self.weight.shape[1], "length must equal"
        out = torch.zeros_like(xs[0])
        for i, x in enumerate(xs):
            out = out + self.weight[0, i] * x
        return out


class ComposedToken(nn.Module):
    """Port of SAMGPT layers/prompt.py::composedtoken."""

    def __init__(self, texttokens: Sequence[torch.Tensor], type_: str = "mul"):
        super().__init__()
        self.register_buffer(
            "texttoken", torch.cat([t.detach().clone() for t in texttokens], dim=0)
        )
        self.prompt = WeightedPrompt(len(texttokens))
        self.type = type_

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        token = self.prompt([self.texttoken[i : i + 1] for i in range(self.texttoken.size(0))])
        if self.type == "add":
            return seq + token
        if self.type == "mul":
            return seq * token
        raise ValueError(f"unknown prompt type {self.type}")


class SAMGCNLayer(nn.Module):
    """Port of SAMGPT layers/gcn.py::GCN."""

    def __init__(self, in_ft: int, out_ft: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        self.bias = nn.Parameter(torch.zeros(out_ft)) if bias else None
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, seq: torch.Tensor, adj_torch, edge_chunk: int) -> torch.Tensor:
        h = self.fc(seq)
        out = _spmm(adj_torch, h, edge_chunk)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class SAMGcnLayers(nn.Module):
    """Port of SAMGPT models/gcnlayers.py::GcnLayers."""

    def __init__(self, n_in: int, n_h: int, layers: int):
        super().__init__()
        self.g_net = nn.ModuleList(
            [SAMGCNLayer(n_in if i == 0 else n_h, n_h) for i in range(layers)]
        )

    def forward(
        self,
        seq: torch.Tensor,
        adj_torch,
        edge_chunk: int,
        prompt_layers: Sequence[nn.Module] | None = None,
    ) -> torch.Tensor:
        h = seq
        for i, conv in enumerate(self.g_net):
            if i == 0:
                h = conv(h, adj_torch, edge_chunk)
            else:
                h = conv(h, adj_torch, edge_chunk) + h
            if prompt_layers is not None:
                h = prompt_layers[i](h)
        return h


class GraphCLHead(nn.Module):
    """Official GraphCL discriminator logic with a shared bilinear scorer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.f_k.weight.data)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def disc(self, c: torch.Tensor, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
        cx = c.reshape(1, -1).expand_as(h_pos)
        sc_pos = torch.squeeze(self.f_k(h_pos, cx), 1)
        sc_neg = torch.squeeze(self.f_k(h_neg, cx), 1)
        return torch.cat([sc_pos, sc_neg], dim=0)

    def forward(
        self,
        gcn: SAMGcnLayers,
        x: torch.Tensor,
        x_neg: torch.Tensor,
        adj,
        aug1,
        aug2,
        edge_chunk: int,
        prompt_layers: Sequence[nn.Module] | None = None,
    ) -> torch.Tensor:
        h0 = gcn(x, adj, edge_chunk)
        h1 = gcn(x, aug1, edge_chunk, prompt_layers)
        h3 = gcn(x, aug2, edge_chunk, prompt_layers)
        h2 = gcn(x_neg, adj, edge_chunk, prompt_layers)
        c1 = torch.sigmoid(h1.mean(0))
        c3 = torch.sigmoid(h3.mean(0))
        return self.disc(c1, h0, h2) + self.disc(c3, h0, h2)


class PrePrompt(nn.Module):
    """SAMGPT preprompt with feature/structure pretext prompts."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_pretrain: int,
        layers_num: int,
        type_: str,
        alpha: float,
        encoder_type: str = "native",
        enr_hp: dict | None = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.feature_prompt_layers = nn.ModuleList(
            [TextPrompt(feature_dim, type_) for _ in range(num_pretrain)]
        )
        if encoder_type == "native":
            self.encoder_depth = layers_num
            self.structure_dim = hidden_dim
            self.gcn = SAMGcnLayers(feature_dim, hidden_dim, layers_num)
            self.output_dim = hidden_dim
        elif encoder_type == "ego_neighbor_residual":
            if enr_hp is None:
                raise ValueError("enr_hp is required for SAMGPT ENR variant")
            self.structure_dim = int(enr_hp["hidden_dim"])
            self.encoder_depth = int(enr_hp["num_layers"])
            self.gcn = EgoNeighborResidualEncoder(
                feature_dim,
                self.structure_dim,
                int(enr_hp["num_layers"]),
                int(enr_hp["num_hops"]),
                dropout=float(enr_hp["dropout"]),
                activation=enr_hp["activation"],
            )
            self.output_dim = self.gcn.output_dim
        else:
            raise ValueError(f"unknown encoder_type={encoder_type}")
        self.structure_prompt_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [TextPrompt(self.structure_dim, type_) for _ in range(self.encoder_depth)]
                )
                for _ in range(num_pretrain)
            ]
        )
        self.graphcl = GraphCLHead(self.output_dim)
        self.alpha = float(alpha)

    def graphcl_loss(
        self, graph_id: int, x: torch.Tensor, adj, aug1, aug2, edge_chunk: int
    ) -> torch.Tensor:
        perm = torch.randperm(x.size(0), device=x.device)
        x_neg = x[perm]
        lbl = torch.cat(
            [torch.ones(x.size(0), device=x.device), torch.zeros(x.size(0), device=x.device)]
        )

        fea_x = self.feature_prompt_layers[graph_id](x)
        fea_neg = self.feature_prompt_layers[graph_id](x_neg)
        fea_logits = self.graphcl(self.gcn, fea_x, fea_neg, adj, aug1, aug2, edge_chunk, None)
        str_logits = self.graphcl(
            self.gcn, x, x_neg, adj, aug1, aug2, edge_chunk, self.structure_prompt_layers[graph_id]
        )
        logits = fea_logits + self.alpha * str_logits
        return F.binary_cross_entropy_with_logits(logits, lbl)

    def get_weights(self) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[float]]:
        fea = [p.weight.detach().clone() for p in self.feature_prompt_layers]
        struct = [
            [p.weight.detach().clone() for p in layers] for layers in self.structure_prompt_layers
        ]
        return fea, struct, [self.alpha]


def _run_encoder(
    gcn: nn.Module, x: torch.Tensor, adj, edge_chunk: int, prompt_layers=None
) -> torch.Tensor:
    """Use the low-peak ENR path only for target inference."""
    if (
        isinstance(gcn, EgoNeighborResidualEncoder)
        and not gcn.training
        and not torch.is_grad_enabled()
    ):
        return gcn.forward_inference(x, adj, edge_chunk, prompt_layers=prompt_layers)
    return gcn(x, adj, edge_chunk, prompt_layers)


class DownstreamPrompt(nn.Module):
    """Port of SAMGPT downprompt.py::downstreamprompt/downprompt for node task."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        structure_dim: int,
        layers_num: int,
        fea_pretext_weights: Sequence[torch.Tensor],
        str_pretext_weights: Sequence[Sequence[torch.Tensor]],
        alpha: float,
        beta: float,
        type_: str,
    ):
        super().__init__()
        self.composedprompt_fea = ComposedToken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList(
            [
                ComposedToken([pretext[i] for pretext in str_pretext_weights], type_)
                for i in range(layers_num)
            ]
        )
        self.open_prompt_fea = TextPrompt(feature_dim, type_)
        self.open_prompt_str = nn.ModuleList(
            [TextPrompt(structure_dim, type_) for _ in range(layers_num)]
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ave: torch.Tensor | None = None

    def embed(self, x: torch.Tensor, adj, gcn: SAMGcnLayers, edge_chunk: int) -> torch.Tensor:
        seq_fea = self.composedprompt_fea(x) + self.beta * self.open_prompt_fea(x)
        embed_fea = _run_encoder(gcn, seq_fea, adj, edge_chunk, None)
        del seq_fea
        composed_embed_str = _run_encoder(gcn, x, adj, edge_chunk, self.composedprompt_str)
        if not self.training and not torch.is_grad_enabled():
            # Inference-only in-place accumulation avoids retaining arithmetic
            # temporaries and a fourth full-node embedding.  The three encoder
            # branches and their coefficients are unchanged.
            embed_fea.add_(composed_embed_str, alpha=self.alpha)
            del composed_embed_str
            open_embed_str = _run_encoder(gcn, x, adj, edge_chunk, self.open_prompt_str)
            embed_fea.add_(open_embed_str, alpha=self.alpha * self.beta)
            del open_embed_str
            return embed_fea
        open_embed_str = _run_encoder(gcn, x, adj, edge_chunk, self.open_prompt_str)
        embed_str = composed_embed_str + self.beta * open_embed_str
        return embed_fea + self.alpha * embed_str

    def embed_to_cpu(
        self, x: torch.Tensor, adj, gcn: SAMGcnLayers, edge_chunk: int, transfer_chunk: int
    ) -> torch.Tensor:
        """Exact three-branch inference with only one full branch on GPU."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("embed_to_cpu requires eval mode and no_grad")

        seq_fea = self.composedprompt_fea(x) + self.beta * self.open_prompt_fea(x)
        branch = _run_encoder(gcn, seq_fea, adj, edge_chunk, None)
        out_cpu = branch.cpu()
        del branch, seq_fea

        def accumulate(prompt_layers, coefficient: float):
            branch_gpu = _run_encoder(gcn, x, adj, edge_chunk, prompt_layers)
            block_size = max(int(transfer_chunk), 1)
            for start in range(0, branch_gpu.size(0), block_size):
                end = min(start + block_size, branch_gpu.size(0))
                block = out_cpu[start:end].to(branch_gpu.device)
                block.add_(branch_gpu[start:end], alpha=coefficient)
                out_cpu[start:end].copy_(block.cpu())
            del branch_gpu

        accumulate(self.composedprompt_str, self.alpha)
        accumulate(self.open_prompt_str, self.alpha * self.beta)
        return out_cpu

    @staticmethod
    def _averageemb(
        labels: torch.Tensor, rawret: torch.Tensor, nb_classes: int = 2
    ) -> torch.Tensor:
        centers = []
        for c in range(nb_classes):
            m = labels == c
            if m.sum() == 0:
                centers.append(
                    torch.zeros(rawret.size(1), dtype=rawret.dtype, device=rawret.device)
                )
            else:
                centers.append(rawret[m].mean(0))
        return torch.stack(centers, dim=0)

    def probs_from_emb(
        self,
        emb: torch.Tensor,
        idx: torch.Tensor,
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> torch.Tensor:
        raw = emb[idx]
        if train:
            assert labels is not None
            self.ave = self._averageemb(labels, raw)
        if self.ave is None:
            raise RuntimeError("SAMGPT downstream centers are not initialized")
        sim = F.cosine_similarity(raw.unsqueeze(1), self.ave.unsqueeze(0), dim=-1)
        return F.softmax(sim, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        adj,
        gcn: SAMGcnLayers,
        edge_chunk: int,
        idx: torch.Tensor,
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> torch.Tensor:
        emb = self.embed(x, adj, gcn, edge_chunk)
        return self.probs_from_emb(emb, idx, labels, train)


class AnomalyAwareDownstreamPrompt(DownstreamPrompt):
    """SAMGPT domain prompts tuned by ego-neighbor inconsistency.

    The inherited composed prompts keep every source-domain feature/structure
    token and learn their target-specific mixture.  The same prompted encoder
    is evaluated with (1) a self-loop-free neighbor operator and (2) implicit
    identity propagation. Only SAMGPT's downstream prompt parameters are
    tuned; the pretrained encoder and source-domain tokens stay frozen.
    """

    def _branch(self, x: torch.Tensor, adj, gcn: SAMGcnLayers, edge_chunk: int) -> torch.Tensor:
        # Feature-token path: composed source-domain token + target open token.
        seq_fea = self.composedprompt_fea(x) + self.beta * self.open_prompt_fea(x)
        out = _run_encoder(gcn, seq_fea, adj, edge_chunk, None)
        del seq_fea

        # Structure-token paths retain the composed source-domain tokens at
        # every encoder layer and the target-specific open structure tokens.
        composed = _run_encoder(gcn, x, adj, edge_chunk, self.composedprompt_str)
        if not self.training and not torch.is_grad_enabled():
            out.add_(composed, alpha=self.alpha)
        else:
            out = out + self.alpha * composed
        del composed
        opened = _run_encoder(gcn, x, adj, edge_chunk, self.open_prompt_str)
        if not self.training and not torch.is_grad_enabled():
            out.add_(opened, alpha=self.alpha * self.beta)
        else:
            out = out + self.alpha * self.beta * opened
        del opened
        return out

    def embed_pair(
        self, x: torch.Tensor, neighbor_adj, gcn: SAMGcnLayers, edge_chunk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # `None` means identity propagation: the ego branch uses exactly the
        # same GCN weights and prompts but receives no neighbor messages.
        ego = self._branch(x, None, gcn, edge_chunk)
        neighbor = self._branch(x, neighbor_adj, gcn, edge_chunk)
        return ego, neighbor

    @staticmethod
    def branch_cosine(ego: torch.Tensor, neighbor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(ego[idx], neighbor[idx], dim=-1, eps=1e-12)

    def completion_loss(
        self, ego: torch.Tensor, neighbor: torch.Tensor, idx: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Anomaly-aware sign convention: normal agrees, anomaly disagrees."""
        cosine = self.branch_cosine(ego, neighbor, idx)
        signed = torch.where(labels == 0, -cosine, cosine)
        return signed.mean()


def _enr_norm(ctx: GraphCtx):
    if ctx.enr_adj_norm is None:
        ctx.enr_adj_norm = _normalize_transposed_sym(
            ctx.adj, add_self_loops=ctx.name not in NO_SELFLOOP
        )
    return ctx.enr_adj_norm


def _difference_neighbor_norm(ctx: GraphCtx):
    """SAMGPT symmetric normalization with self-loops removed.

    Only the ego/neighbor separation is borrowed from anomaly-aware prompt
    tuning; the propagation convention remains SAMGPT's own normalization.
    """
    if ctx.difference_adj_norm is None:
        a = ctx.adj.tocoo(copy=False)
        keep = a.row != a.col
        row = a.row[keep]
        col = a.col[keep]
        data = a.data[keep].astype(np.float32, copy=False)
        degree = np.bincount(row, weights=data, minlength=a.shape[0]).astype(np.float32)
        inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nz = degree > 0
        inv_sqrt[nz] = np.power(degree[nz], -0.5)
        weight = data * inv_sqrt[row] * inv_sqrt[col]
        ctx.difference_adj_norm = sp.coo_matrix(
            (weight, (col, row)), shape=a.shape, dtype=np.float32
        )
    return ctx.difference_adj_norm


def _adj_for(ctx: GraphCtx, device: str, big: bool = False, enr: bool = False):
    norm = _enr_norm(ctx) if enr else ctx.adj_norm
    if norm is None:
        raise RuntimeError(f"{ctx.name}: native SAMGPT adjacency was not prepared")
    return CompactEdgeList(norm) if big else to_torch_sparse(norm).to(device)


def _difference_adj_for(ctx: GraphCtx, device: str, big: bool = False):
    norm = _difference_neighbor_norm(ctx)
    return CompactEdgeList(norm) if big else to_torch_sparse(norm).to(device)


def _drop_aug_adj(
    ctx: GraphCtx,
    drop_percent: float,
    rng: np.random.RandomState,
    device: str,
    hp: dict,
    enr: bool = False,
):
    if ctx.adj.nnz > int(hp.get("aug_max_edges", 5_000_000)):
        # Official runner already reuses the full graph for dense sources;
        # EdgeList keeps that exact full operator off GPU for both native and
        # ENR encoders. The caller reuses one object for both augmented views.
        return _adj_for(ctx, device, big=ctx.name in STREAM_GRAPHS, enr=enr)
    if enr:
        raw = ctx.adj.tocoo()
        keep = rng.rand(raw.nnz) >= float(drop_percent)
        dropped = sp.csr_matrix((raw.data[keep], (raw.row[keep], raw.col[keep])), shape=raw.shape)
        norm = _normalize_transposed_sym(dropped, add_self_loops=ctx.name not in NO_SELFLOOP)
        return to_torch_sparse(norm).to(device)
    raw = (ctx.adj + sp.eye(ctx.adj.shape[0], dtype=np.float32, format="csr")).tocoo()
    keep = rng.rand(raw.nnz) >= float(drop_percent)
    # Keep self-loops stable so isolated nodes remain well-defined.
    keep = np.logical_or(keep, raw.row == raw.col)
    dropped = sp.csr_matrix((raw.data[keep], (raw.row[keep], raw.col[keep])), shape=raw.shape)
    return to_torch_sparse(
        _normalize_adj_official(dropped - sp.eye(dropped.shape[0], dtype=np.float32, format="csr"))
    ).to(device)


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


def _query_nodes(ctx: GraphCtx, support: np.ndarray) -> np.ndarray:
    """All marked non-support nodes without a million-element Python list."""
    mask = ctx.mark.copy()
    mask[support] = False
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _exact_support_dependency_subgraph(
    ctx: GraphCtx, support: np.ndarray, layers: int, enr: bool = False, difference: bool = False
):
    """Exact L-hop dependency subgraph for A @ X GCN support embeddings.

    SAMGPT's GCN computes `out[row] += A_norm[row, col] * x[col]`.  Therefore
    support rows depend on their nonzero columns; after L layers we need the
    transitive L-hop column closure.  No fanout cap and no random sampling are
    used, so support embeddings on this subgraph match full-graph embeddings up
    to floating-point accumulation order.
    """
    if enr and difference:
        raise ValueError("dependency operator cannot be both ENR and difference-style")
    if difference:
        csr = _difference_neighbor_norm(ctx).tocsr()
    else:
        norm = _enr_norm(ctx) if enr else ctx.adj_norm
        if norm is None:
            raise RuntimeError(f"{ctx.name}: native SAMGPT adjacency was not prepared")
        csr = norm.tocsr()
    seen = np.zeros(csr.shape[0], dtype=bool)
    frontier = np.asarray(support, dtype=np.int64)
    seen[frontier] = True
    for _ in range(int(layers)):
        if frontier.size == 0:
            break
        cols = csr[frontier].indices
        if cols.size == 0:
            break
        new = np.unique(cols[~seen[cols]])
        if new.size == 0:
            break
        seen[new] = True
        frontier = new
    nodes = np.where(seen)[0].astype(np.int64)
    rel = {int(n): i for i, n in enumerate(nodes)}
    support_rel = np.asarray([rel[int(i)] for i in support], dtype=np.int64)
    adj_sub = csr[nodes][:, nodes].tocoo().astype(np.float32)
    x_sub = np.ascontiguousarray(ctx.x[nodes], dtype=np.float32)
    return nodes, support_rel, x_sub, adj_sub


def _pretrain_one_seed(
    model: PrePrompt, ctxs: Sequence[GraphCtx], hp: dict, seed: int, device: str
):
    model.train()
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["pretrain_lr"]),
        weight_decay=float(hp["pretrain_weight_decay"]),
    )
    prepared = []
    for gid, ctx in enumerate(ctxs):
        x = torch.from_numpy(ctx.x).float().to(device)
        is_enr = model.encoder_type == "ego_neighbor_residual"
        stream = ctx.name in STREAM_GRAPHS
        adj = _adj_for(ctx, device, big=stream, enr=is_enr)
        if ctx.adj.nnz > int(hp.get("aug_max_edges", 5_000_000)):
            # The released dense-graph path uses the full graph for both views.
            # Reusing the very same EdgeList avoids three CPU copies of 42M edges.
            aug1 = adj
            aug2 = adj
        else:
            aug1 = _drop_aug_adj(ctx, float(hp["drop_percent"]), rng, device, hp, enr=is_enr)
            aug2 = _drop_aug_adj(ctx, float(hp["drop_percent"]), rng, device, hp, enr=is_enr)
        prepared.append((gid, ctx, x, adj, aug1, aug2))

    epochs = int(hp["pretrain_epochs"])
    for epoch in range(epochs):
        total = 0.0
        opt.zero_grad()
        for gid, _ctx, x, adj, aug1, aug2 in prepared:
            loss = model.graphcl_loss(gid, x, adj, aug1, aug2, int(hp["edge_chunk"]))
            total = total + loss
        total = total / max(len(prepared), 1)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        opt.step()
        if (epoch + 1) % max(epochs // 5, 1) == 0:
            print(
                f"    [samgpt-pretrain] seed={seed} epoch={epoch + 1}/{epochs} loss={float(total.item()):.4f}"
            )


def _build_downstream(
    model: PrePrompt, hp: dict, feature_dim: int, device: str
) -> DownstreamPrompt:
    fea, struct, combines = model.get_weights()
    return DownstreamPrompt(
        feature_dim=feature_dim,
        hidden_dim=int(hp["hid_units"]),
        structure_dim=model.structure_dim,
        layers_num=model.encoder_depth,
        fea_pretext_weights=[t.to(device) for t in fea],
        str_pretext_weights=[[t.to(device) for t in layers] for layers in struct],
        alpha=float(combines[0]),
        beta=float(hp["beta"]),
        type_=hp["combinetype"],
    ).to(device)


def _build_difference_downstream(
    model: PrePrompt, hp: dict, feature_dim: int, device: str
) -> AnomalyAwareDownstreamPrompt:
    fea, struct, combines = model.get_weights()
    return AnomalyAwareDownstreamPrompt(
        feature_dim=feature_dim,
        hidden_dim=model.output_dim,
        structure_dim=model.structure_dim,
        layers_num=model.encoder_depth,
        fea_pretext_weights=[t.to(device) for t in fea],
        str_pretext_weights=[[t.to(device) for t in layers] for layers in struct],
        alpha=float(combines[0]),
        beta=float(hp["beta"]),
        type_=hp["combinetype"],
    ).to(device)


def _train_target_downstream(
    down: DownstreamPrompt,
    gcn: SAMGcnLayers,
    ctx: GraphCtx,
    x: torch.Tensor | None,
    adj,
    support: np.ndarray,
    support_labels: torch.Tensor,
    hp: dict,
    device: str,
):
    steps = (
        int(hp["large_downstream_steps"])
        if ctx.name in STREAM_GRAPHS
        else int(hp["downstream_steps"])
    )
    idx = torch.from_numpy(support).long().to(device)
    train_x, train_adj, train_idx = x, adj, idx
    if ctx.name in STREAM_GRAPHS and steps > 0:
        is_enr = isinstance(gcn, EgoNeighborResidualEncoder)
        dependency_hops = gcn.num_hops if is_enr else int(hp["layers_num"])
        nodes, support_rel, x_sub, adj_sub = _exact_support_dependency_subgraph(
            ctx, support, dependency_hops, enr=is_enr
        )
        print(
            f"    [samgpt-large-support-subgraph] {ctx.name}: support={len(support)} "
            f"dep_nodes={len(nodes)} dep_edges={adj_sub.nnz} steps={steps}"
        )
        train_x = torch.from_numpy(x_sub).float().to(device)
        # A high-degree exact dependency closure can still contain millions of
        # edges. Stream it from compact CPU storage instead of materializing a
        # full int64 COO tensor on the accelerator for every tuning step.
        train_adj = CompactEdgeList(adj_sub)
        train_idx = torch.from_numpy(support_rel).long().to(device)
    elif train_x is None or train_adj is None:
        raise RuntimeError(f"{ctx.name}: full target tensors are required for downstream training")
    if steps <= 0:
        with torch.no_grad():
            emb = down.embed(train_x, train_adj, gcn, int(hp["edge_chunk"]))
            down.ave = down._averageemb(support_labels, emb[train_idx])
        return

    gcn.eval()
    down.train()
    old_requires_grad = [p.requires_grad for p in gcn.parameters()]
    for p in gcn.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(down.parameters(), lr=float(hp["downstream_lr"]))
    try:
        for step in range(steps):
            probs = down(
                train_x,
                train_adj,
                gcn,
                int(hp["edge_chunk"]),
                train_idx,
                support_labels,
                train=True,
            )
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), support_labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(down.parameters(), 8.0)
            opt.step()
            if (step + 1) % max(steps // 4, 1) == 0:
                print(
                    f"    [samgpt-down] {ctx.name} step={step + 1}/{steps} loss={loss.item():.4f}"
                )
    finally:
        for p, req in zip(gcn.parameters(), old_requires_grad):
            p.requires_grad_(req)


def _train_target_difference(
    down: AnomalyAwareDownstreamPrompt,
    gcn: SAMGcnLayers,
    ctx: GraphCtx,
    support: np.ndarray,
    support_labels: torch.Tensor,
    hp: dict,
    device: str,
):
    """Tune only SAMGPT prompts using labeled ego-neighbor agreement."""
    steps = (
        int(hp["large_downstream_steps"])
        if ctx.name in STREAM_GRAPHS
        else int(hp["downstream_steps"])
    )
    if steps <= 0:
        raise ValueError("samgpt_diff requires positive downstream prompt-tuning steps")

    # The loss touches only support nodes. Its exact L-hop dependency closure
    # preserves their full-graph neighbor embeddings without sampling nodes or
    # edges. Large closures keep the sparse operator on CPU and stream chunks.
    nodes, support_rel, x_sub, adj_sub = _exact_support_dependency_subgraph(
        ctx, support, int(hp["layers_num"]), difference=True
    )
    print(
        f"    [samgpt-diff-support-subgraph] {ctx.name}: support={len(support)} "
        f"dep_nodes={len(nodes)} dep_edges={adj_sub.nnz} steps={steps}"
    )
    train_x = torch.from_numpy(x_sub).float().to(device)
    train_adj = (
        CompactEdgeList(adj_sub)
        if ctx.name in STREAM_GRAPHS
        else to_torch_sparse(adj_sub).to(device)
    )
    train_idx = torch.from_numpy(support_rel).long().to(device)

    gcn.eval()
    down.train()
    old_requires_grad = [p.requires_grad for p in gcn.parameters()]
    for p in gcn.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(down.parameters(), lr=float(hp["downstream_lr"]))
    try:
        for step in range(steps):
            ego, neighbor = down.embed_pair(train_x, train_adj, gcn, int(hp["edge_chunk"]))
            loss = down.completion_loss(ego, neighbor, train_idx, support_labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(down.parameters(), 8.0)
            opt.step()
            if (step + 1) % max(steps // 4, 1) == 0:
                print(
                    f"    [samgpt-diff-down] {ctx.name} step={step + 1}/{steps} "
                    f"loss={loss.item():.4f}"
                )
    finally:
        for p, req in zip(gcn.parameters(), old_requires_grad):
            p.requires_grad_(req)


@torch.no_grad()
def _score_target_difference(
    model: PrePrompt, ctx: GraphCtx, hp: dict, shot: int, seed: int, feature_dim: int, device: str
):
    """Few-shot target adaptation followed by full marked-query scoring."""
    if model.encoder_type != "native":
        raise ValueError(
            "samgpt_diff adapts native SAMGPT; it is not stacked on the ENR encoder variant"
        )
    rng = np.random.RandomState(seed + 40_000)
    sup0 = _sample_class(ctx, 0, shot, rng, reserve_query=True)
    sup1 = _sample_class(ctx, 1, shot, rng, reserve_query=True)
    support = np.concatenate([sup0, sup1])
    query = _query_nodes(ctx, support)
    if query.size == 0:
        raise ValueError(f"{ctx.name}: no query nodes after support selection")

    big = ctx.name in STREAM_GRAPHS
    if big:
        print(
            f"    [samgpt-diff-full-large-target] {ctx.name}: "
            f"eval={int(ctx.mark.sum())} N={ctx.adj.shape[0]} E={ctx.adj.nnz} "
            f"edge_chunk={hp['edge_chunk']} query_chunk={hp['eval_query_batch']} "
            f"downstream_steps={hp['large_downstream_steps']}"
        )

    labels_sup = torch.tensor([0] * len(sup0) + [1] * len(sup1), dtype=torch.long, device=device)
    down = _build_difference_downstream(model, hp, feature_dim, device)
    torch.set_grad_enabled(True)
    try:
        _train_target_difference(down, model.gcn, ctx, support, labels_sup, hp, device)
    finally:
        torch.set_grad_enabled(False)

    down.eval()
    x = torch.from_numpy(ctx.x).float().to(device)
    neighbor_adj = _difference_adj_for(ctx, device, big=big)
    if big:
        # The two branches are independent.  Move ego to host before computing
        # neighbor so only one full N-by-H branch is ever resident on the GPU.
        ego = down._branch(x, None, model.gcn, int(hp["edge_chunk"])).cpu()
        neighbor = down._branch(x, neighbor_adj, model.gcn, int(hp["edge_chunk"]))
        del neighbor_adj, x
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        ego, neighbor = down.embed_pair(x, neighbor_adj, model.gcn, int(hp["edge_chunk"]))

    scores = np.empty(query.size, dtype=np.float32)
    qchunk = int(hp["eval_query_batch"])
    for s in range(0, query.size, qchunk):
        q = query[s : s + qchunk]
        idx = torch.from_numpy(q).long()
        # Low ego-neighbor agreement means high anomaly score; `-cosine` is
        # rank-equivalent to any affine 1-cosine anomaly score.
        if big:
            ego_q = ego[idx].to(device)
            neighbor_q = neighbor[idx.to(device)]
            cosine = F.cosine_similarity(ego_q, neighbor_q, dim=-1, eps=1e-12)
        else:
            cosine = down.branch_cosine(ego, neighbor, idx.to(device))
        scores[s : s + q.size] = (-cosine).detach().cpu().numpy()

    eval_labels = ctx.labels[query]
    del ego, neighbor, down
    if not big:
        del neighbor_adj, x
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return evaluate(eval_labels, scores)


@torch.no_grad()
def _score_target(
    model: PrePrompt, ctx: GraphCtx, hp: dict, shot: int, seed: int, feature_dim: int, device: str
):
    rng = np.random.RandomState(seed + 40_000)
    sup0 = _sample_class(ctx, 0, shot, rng, reserve_query=True)
    sup1 = _sample_class(ctx, 1, shot, rng, reserve_query=True)
    support = np.concatenate([sup0, sup1])
    query = _query_nodes(ctx, support)
    if query.size == 0:
        raise ValueError(f"{ctx.name}: no query nodes after support selection")

    is_enr = model.encoder_type == "ego_neighbor_residual"
    big = ctx.name in STREAM_GRAPHS
    if big:
        print(
            f"    [samgpt-full-large-target] {ctx.name}: eval={int(ctx.mark.sum())} "
            f"N={ctx.adj.shape[0]} E={ctx.adj.nnz} edge_chunk={hp['edge_chunk']} "
            f"query_chunk={hp['eval_query_batch']} downstream_steps={hp['large_downstream_steps']}"
        )

    labels_sup = torch.tensor([0] * len(sup0) + [1] * len(sup1), dtype=torch.long, device=device)
    down = _build_downstream(model, hp, feature_dim, device)

    # Temporarily leave no_grad so target downstream prompt training can follow
    # the configured steps. Large targets use the exact support-dependency
    # subgraph for backpropagation and still score all marked query nodes.
    torch.set_grad_enabled(True)
    try:
        if big:
            # Large-target adaptation uses the exact support dependency graph;
            # do not build the multi-gigabyte full EdgeList until adaptation is
            # complete and full-node inference actually starts.
            _train_target_downstream(
                down, model.gcn, ctx, None, None, support, labels_sup, hp, device
            )
        else:
            x = torch.from_numpy(ctx.x).float().to(device)
            adj = _adj_for(ctx, device, big=False, enr=is_enr)
            _train_target_downstream(down, model.gcn, ctx, x, adj, support, labels_sup, hp, device)
    finally:
        torch.set_grad_enabled(False)

    down.eval()
    if big:
        x = torch.from_numpy(ctx.x).float().to(device)
        adj = _adj_for(ctx, device, big=True, enr=is_enr)
        emb = down.embed_to_cpu(
            x, adj, model.gcn, int(hp["edge_chunk"]), int(hp["eval_query_batch"])
        )
        idx_sup = torch.from_numpy(support).long()
        down.ave = down._averageemb(labels_sup.cpu(), emb[idx_sup]).to(device)
    else:
        emb = down.embed(x, adj, model.gcn, int(hp["edge_chunk"]))
        idx_sup = torch.from_numpy(support).long().to(device)
        down.ave = down._averageemb(labels_sup, emb[idx_sup])
    # Propagation is complete; query scoring no longer needs the edge stream.
    del adj
    if big:
        del x
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    scores = np.empty(query.size, dtype=np.float32)
    qchunk = int(hp["eval_query_batch"])
    for s in range(0, query.size, qchunk):
        q = query[s : s + qchunk]
        idx = torch.from_numpy(q).long()
        if big:
            query_emb = emb[idx].to(device)
            probs = down.probs_from_emb(query_emb, torch.arange(query_emb.size(0), device=device))
        else:
            probs = down.probs_from_emb(emb, idx.to(device))
        scores[s : s + q.size] = probs[:, 1].detach().cpu().numpy()
    eval_labels = ctx.labels[query]
    del emb, down
    if not big:
        del x
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return evaluate(eval_labels, scores)


def run_samgpt(
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
    downstream_type="prototype",
):
    if downstream_type not in ("prototype", "ego_neighbor_difference"):
        raise ValueError(f"unknown SAMGPT downstream_type={downstream_type}")
    if downstream_type == "ego_neighbor_difference" and encoder_type != "native":
        raise ValueError(
            "SAMGPT difference prompt is a native-encoder adaptation, not an ENR stack"
        )
    source_needs_native_norm = encoder_type == "native"
    target_needs_native_norm = encoder_type == "native" and downstream_type == "prototype"
    src_ctxs = [
        _ctx(n, feature_dim, feature_norm, target=False, build_native_norm=source_needs_native_norm)
        for n in sources
    ]
    per = {n: [] for n in targets}
    trained: List[Tuple[int, PrePrompt]] = []

    for seed in seeds:
        set_seed(seed)
        model = PrePrompt(
            feature_dim=feature_dim,
            hidden_dim=int(hp["hid_units"]),
            num_pretrain=len(src_ctxs),
            layers_num=int(hp["layers_num"]),
            type_=hp["combinetype"],
            alpha=float(hp["alpha"]),
            encoder_type=encoder_type,
            enr_hp=enr_hp,
        ).to(device)
        _pretrain_one_seed(model, src_ctxs, hp, seed, device)
        model.eval()
        model.zero_grad(set_to_none=True)
        model.cpu()
        trained.append((seed, model))
        torch.cuda.empty_cache()

    for target_idx, name in enumerate(targets, start=1):
        print(
            f"    [samgpt-target {target_idx}/{len(targets)}] loading {name} "
            f"native_norm={target_needs_native_norm}",
            flush=True,
        )
        ctx = _ctx(
            name, feature_dim, feature_norm, target=True, build_native_norm=target_needs_native_norm
        )
        for seed_idx, (seed, model) in enumerate(trained, start=1):
            print(
                f"    [samgpt-target {target_idx}/{len(targets)}] {name} "
                f"seed={seed} ({seed_idx}/{len(trained)})",
                flush=True,
            )
            model.to(device)
            try:
                set_seed(seed)
                if downstream_type == "ego_neighbor_difference":
                    score = _score_target_difference(
                        model, ctx, hp, shot, seed, feature_dim, device
                    )
                else:
                    score = _score_target(model, ctx, hp, shot, seed, feature_dim, device)
                per[name].append(score)
            finally:
                model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
        del ctx
        gc.collect()
    return aggregate(per)
