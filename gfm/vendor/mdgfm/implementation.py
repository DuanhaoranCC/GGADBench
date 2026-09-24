"""Scalable MDGFM reproduction for the foundation model benchmark.

Paper / official-code components reproduced here:
  * Eq. (1)-(2): a shared cross-graph feature width followed by per-domain and
    shared multiplicative/additive tokens.
  * Eq. (3): a balance token over [X', A X'] for topology refinement.
  * Eq. (4), (13)-(15): mutual-information lower bounds between embeddings of
    the original and refined graph views.
  * Eq. (5)-(7): a source-token meta prompt, a target-specific prompt, frozen
    GCN transfer, and cosine class prototypes.
  * Eq. (10)-(11): sparse kNN, ReLU, symmetrization, self-loops, and symmetric
    degree normalization.

Benchmark adaptations are intentionally localized in this file:
  * Features use gfm's shared SVD8 adapter, as required for a fair
    comparison with SAMGPT and the other graph foundation models.
  * The released implementation's dense N-by-N learned adjacency is replaced
    by an algebraically equivalent sparse operator for Eq. (10)-(11).
  * Exact chunked cosine kNN is used on moderate graphs. Large graphs use the
    locality-sensitive random-projection candidate search stated by the paper;
    every node is retained and receives k neighbors.
  * Eq. (15) uses the paper-reported batch size (128 by default) as a stochastic
    anchor batch. Each selected anchor still uses every graph node in its
    contrastive denominator. This removes the released code's N-by-N matrix.
  * Large-target prompt tuning freezes one sparse topology and trains on the
    complete GCN dependency closure of the support nodes. After tuning, the
    full sparse target topology is rebuilt once and every marked non-support
    node is scored. There is no evaluation-node sampling.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from common.data import load_source_marked, load_target_marked
from gfm.runners.feature_adapter import adapted_features
from gfm.runners.few_shot import sample_class_support
from util import aggregate, evaluate, set_seed

_EPS = 1e-10


class _FixedCSRSpMM(torch.autograd.Function):
    """Exact CPU-resident CSR SpMM with a bounded accelerator edge chunk."""

    @staticmethod
    def forward(ctx, x, rows_cpu, indices_cpu, values_cpu, n, edge_chunk):
        ctx.rows_cpu = rows_cpu
        ctx.indices_cpu = indices_cpu
        ctx.values_cpu = values_cpu
        ctx.edge_chunk = int(edge_chunk)
        ctx.x_shape = tuple(x.shape)
        out = x.new_zeros((int(n), x.shape[1]))
        nnz = int(values_cpu.numel())
        for start in range(0, nnz, ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, nnz)
            rows = rows_cpu[start:end].to(device=x.device, dtype=torch.long)
            cols = indices_cpu[start:end].to(device=x.device, dtype=torch.long)
            vals = values_cpu[start:end].to(device=x.device, dtype=x.dtype).unsqueeze(1)
            out.index_add_(0, rows, x[cols] * vals)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = grad_out.new_zeros(ctx.x_shape)
        nnz = int(ctx.values_cpu.numel())
        for start in range(0, nnz, ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, nnz)
            rows = ctx.rows_cpu[start:end].to(device=grad_out.device, dtype=torch.long)
            cols = ctx.indices_cpu[start:end].to(device=grad_out.device, dtype=torch.long)
            vals = (
                ctx.values_cpu[start:end]
                .to(device=grad_out.device, dtype=grad_out.dtype)
                .unsqueeze(1)
            )
            grad_x.index_add_(0, cols, grad_out[rows] * vals)
        return grad_x, None, None, None, None, None


class _DifferentiableKNNSpMM(torch.autograd.Function):
    """SpMM for Eq. (10)-(11) retaining gradients to sparse edge weights."""

    @staticmethod
    def forward(ctx, x, cols, values, diagonal, k, edge_chunk):
        ctx.save_for_backward(x, cols, values, diagonal)
        ctx.k = int(k)
        ctx.edge_chunk = int(edge_chunk)
        out = x * diagonal.unsqueeze(1)
        flat_cols = cols.reshape(-1)
        flat_values = values.reshape(-1)
        for start in range(0, flat_values.numel(), ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, flat_values.numel())
            rows = torch.arange(start, end, device=x.device, dtype=torch.long).div(
                ctx.k, rounding_mode="floor"
            )
            dst = flat_cols[start:end]
            val = flat_values[start:end].unsqueeze(1)
            # A_sym=(A_sp+A_sp.T)/2 is represented without duplicating edges.
            out.index_add_(0, rows, x[dst] * val)
            out.index_add_(0, dst, x[rows] * val)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, cols, values, diagonal = ctx.saved_tensors
        grad_x = grad_out * diagonal.unsqueeze(1)
        grad_values = torch.empty_like(values).reshape(-1)
        flat_cols = cols.reshape(-1)
        flat_values = values.reshape(-1)
        for start in range(0, flat_values.numel(), ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, flat_values.numel())
            rows = torch.arange(start, end, device=x.device, dtype=torch.long).div(
                ctx.k, rounding_mode="floor"
            )
            dst = flat_cols[start:end]
            val = flat_values[start:end].unsqueeze(1)
            grad_x.index_add_(0, dst, grad_out[rows] * val)
            grad_x.index_add_(0, rows, grad_out[dst] * val)
            grad_values[start:end] = (grad_out[rows] * x[dst]).sum(1) + (
                grad_out[dst] * x[rows]
            ).sum(1)
        grad_diagonal = (grad_out * x).sum(1)
        return (
            grad_x,
            None,
            grad_values.reshape_as(values),
            grad_diagonal,
            None,
            None,
        )


class _FixedKNNSpMM(torch.autograd.Function):
    """CPU-resident fixed counterpart of :class:`_DifferentiableKNNSpMM`."""

    @staticmethod
    def forward(ctx, x, cols_cpu, values_cpu, diagonal_cpu, k, edge_chunk):
        ctx.cols_cpu = cols_cpu
        ctx.values_cpu = values_cpu
        ctx.diagonal_cpu = diagonal_cpu
        ctx.k = int(k)
        ctx.edge_chunk = int(edge_chunk)
        ctx.x_shape = tuple(x.shape)
        diagonal = diagonal_cpu.to(device=x.device, dtype=x.dtype)
        out = x * diagonal.unsqueeze(1)
        flat_cols = cols_cpu.reshape(-1)
        flat_values = values_cpu.reshape(-1)
        for start in range(0, flat_values.numel(), ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, flat_values.numel())
            rows = torch.arange(start, end, device=x.device, dtype=torch.long).div(
                ctx.k, rounding_mode="floor"
            )
            dst = flat_cols[start:end].to(device=x.device, dtype=torch.long)
            val = flat_values[start:end].to(device=x.device, dtype=x.dtype).unsqueeze(1)
            out.index_add_(0, rows, x[dst] * val)
            out.index_add_(0, dst, x[rows] * val)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        diagonal = ctx.diagonal_cpu.to(device=grad_out.device, dtype=grad_out.dtype)
        grad_x = grad_out * diagonal.unsqueeze(1)
        flat_cols = ctx.cols_cpu.reshape(-1)
        flat_values = ctx.values_cpu.reshape(-1)
        for start in range(0, flat_values.numel(), ctx.edge_chunk):
            end = min(start + ctx.edge_chunk, flat_values.numel())
            rows = torch.arange(start, end, device=grad_out.device, dtype=torch.long).div(
                ctx.k, rounding_mode="floor"
            )
            dst = flat_cols[start:end].to(device=grad_out.device, dtype=torch.long)
            val = (
                flat_values[start:end].to(device=grad_out.device, dtype=grad_out.dtype).unsqueeze(1)
            )
            grad_x.index_add_(0, dst, grad_out[rows] * val)
            grad_x.index_add_(0, rows, grad_out[dst] * val)
        return grad_x, None, None, None, None, None


class FixedCSRGraph:
    """Normalized original graph stored as compact CPU CSR arrays."""

    def __init__(self, adjacency: sp.spmatrix):
        csr = sp.csr_matrix(adjacency, dtype=np.float32)
        csr.sum_duplicates()
        csr.sort_indices()
        self.csr = csr
        row_counts = np.diff(csr.indptr)
        rows = np.repeat(np.arange(csr.shape[0], dtype=np.int32), row_counts)
        self.rows = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int32))
        self.indices = torch.from_numpy(np.ascontiguousarray(csr.indices, dtype=np.int32))
        self.values = torch.from_numpy(np.ascontiguousarray(csr.data, dtype=np.float32))
        self.shape = csr.shape

    def spmm(self, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
        return _FixedCSRSpMM.apply(
            x,
            self.rows,
            self.indices,
            self.values,
            self.shape[0],
            int(edge_chunk),
        )


class DifferentiableKNNGraph:
    """Paper Eq. (10)-(11) in O(Nk) storage with live edge gradients."""

    def __init__(self, cols: torch.Tensor, values: torch.Tensor, diagonal: torch.Tensor):
        self.cols = cols
        self.values = values
        self.diagonal = diagonal
        self.n = int(cols.shape[0])
        self.k = int(cols.shape[1])

    def spmm(self, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
        return _DifferentiableKNNSpMM.apply(
            x, self.cols, self.values, self.diagonal, self.k, int(edge_chunk)
        )

    def positive_edges(self):
        """Expanded sparse positive matrix used by Eq. (14)-(15)."""
        device = self.cols.device
        src = torch.arange(self.n, device=device).repeat_interleave(self.k)
        dst = self.cols.reshape(-1)
        val = self.values.reshape(-1)
        nodes = torch.arange(self.n, device=device)
        rows = torch.cat((src, dst, nodes))
        cols = torch.cat((dst, src, nodes))
        values = torch.cat((val, val, self.diagonal))
        order = torch.argsort(rows)
        rows = rows[order]
        cols = cols[order]
        values = values[order]
        counts = torch.bincount(rows, minlength=self.n)
        rowptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return cols, values, rowptr


class FixedKNNGraph:
    """CPU-resident O(Nk) refined graph for large-target inference."""

    def __init__(self, cols: np.ndarray, values: np.ndarray, diagonal: np.ndarray):
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        diagonal = np.ascontiguousarray(diagonal, dtype=np.float32)
        if cols.shape != values.shape:
            raise ValueError("fixed kNN columns and values must have equal shape")
        self.cols = torch.from_numpy(cols)
        self.values = torch.from_numpy(values)
        self.diagonal = torch.from_numpy(diagonal)
        self.n = int(cols.shape[0])
        self.k = int(cols.shape[1])
        self._csr: Optional[sp.csr_matrix] = None

    def spmm(self, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
        return _FixedKNNSpMM.apply(
            x, self.cols, self.values, self.diagonal, self.k, int(edge_chunk)
        )

    def to_csr(self) -> sp.csr_matrix:
        """Materialize only the O(Nk) sparse matrix, never an N-by-N dense one."""
        if self._csr is None:
            cols = self.cols.numpy()
            values = self.values.numpy()
            indptr = np.arange(0, (self.n + 1) * self.k, self.k, dtype=np.int64)
            directed = sp.csr_matrix(
                (values.reshape(-1), cols.reshape(-1), indptr),
                shape=(self.n, self.n),
                dtype=np.float32,
            )
            diagonal = sp.diags(self.diagonal.numpy(), format="csr", dtype=np.float32)
            self._csr = (directed + directed.transpose() + diagonal).tocsr()
            self._csr.sum_duplicates()
            self._csr.sort_indices()
        return self._csr


@dataclass
class BlendedGraph:
    original: object
    refined: object
    original_weight: float

    def spmm(self, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
        left = graph_spmm(self.original, x, edge_chunk)
        right = graph_spmm(self.refined, x, edge_chunk)
        return self.original_weight * left + (1.0 - self.original_weight) * right


def graph_spmm(graph, x: torch.Tensor, edge_chunk: int) -> torch.Tensor:
    return graph.spmm(x, int(edge_chunk))


def _normalized_original_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Paper-style D^-1/2(A+I)D^-1/2 without dense intermediates."""
    coo = sp.coo_matrix(adjacency, dtype=np.float32)
    off_diagonal = coo.row != coo.col
    a = sp.csr_matrix(
        (
            np.ones(int(off_diagonal.sum()), dtype=np.float32),
            (coo.row[off_diagonal], coo.col[off_diagonal]),
        ),
        shape=coo.shape,
        dtype=np.float32,
    )
    a.sum_duplicates()
    if a.nnz:
        a.data.fill(1.0)
    a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(a.sum(1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(degree)
    positive = degree > 0
    inv[positive] = np.power(degree[positive], -0.5)
    d = sp.diags(inv, format="csr", dtype=np.float32)
    normalized = (d @ a @ d).tocsr().astype(np.float32, copy=False)
    normalized.sum_duplicates()
    normalized.sort_indices()
    return normalized


class PromptToken(nn.Module):
    """Official textprompt: one learnable vector with add/mul composition."""

    def __init__(self, width: int, prompt_type: str = "mul"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, int(width)))
        self.prompt_type = str(prompt_type)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompt_type == "mul":
            return x * self.weight
        if self.prompt_type == "add":
            return x + self.weight
        raise ValueError(f"unknown MDGFM prompt type {self.prompt_type}")


class MDGFMGCNLayer(nn.Module):
    """Official GCN layer: Linear(no bias), propagation, bias, PReLU."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor, graph, edge_chunk: int) -> torch.Tensor:
        projected = self.fc(x)
        propagated = graph_spmm(graph, projected, edge_chunk)
        del projected
        if torch.is_grad_enabled():
            return self.activation(propagated + self.bias)
        # PReLU has one scalar parameter here. The in-place no-grad form is
        # mathematically identical and avoids another full N-by-hidden tensor.
        propagated.add_(self.bias)
        return F.leaky_relu_(propagated, negative_slope=float(self.activation.weight.item()))


class MDGFMGCN(nn.Module):
    """Three-layer residual GCN from official ``models/gcnlayers.py``."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MDGFMGCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(int(num_layers))
            ]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(int(num_layers))]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.num_layers = int(num_layers)

    def forward(
        self, x: torch.Tensor, graph, edge_chunk: int, pretrain_view: bool = False
    ) -> torch.Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            out = layer(h, graph, edge_chunk)
            if layer_idx:
                if torch.is_grad_enabled():
                    out = out + h
                else:
                    out.add_(h)
            if pretrain_view:
                out = self.batch_norms[layer_idx](out)
                out = self.dropout(out)
            h = out
        return F.elu(h) if pretrain_view else h


def _l2_normalize_numpy(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.ascontiguousarray(x / norm, dtype=np.float32)


def _exact_knn_cols_numpy(x: np.ndarray, k: int, query_chunk: int) -> np.ndarray:
    n = int(x.shape[0])
    cols = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, int(query_chunk)):
        end = min(start + int(query_chunk), n)
        similarity = x[start:end] @ x.T
        local = np.arange(end - start)
        similarity[local, np.arange(start, end)] = -np.inf
        chosen = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        chosen_values = np.take_along_axis(similarity, chosen, axis=1)
        order = np.argsort(-chosen_values, axis=1)
        cols[start:end] = np.take_along_axis(chosen, order, axis=1).astype(np.int32, copy=False)
    return cols


def _lsh_knn_cols_numpy(x: np.ndarray, k: int, hp: dict) -> np.ndarray:
    """Random-projection locality search with O(Nk) result storage.

    Each random projection induces a locality-sensitive ordering. Candidate
    neighbors are collected from bounded windows in those orderings, deduplicated,
    and re-ranked by exact cosine similarity. All N nodes are processed.
    """
    n, width = x.shape
    tables = max(int(hp.get("lsh_tables", 2)), 1)
    window = max(int(hp.get("lsh_window", 64)), k + 1)
    chunk = max(int(hp.get("knn_query_chunk", 1024)), 1)
    rng = np.random.RandomState(int(hp.get("gsl_search_seed", 0)))

    orders = []
    ranks = []
    for _ in range(tables):
        direction = rng.normal(size=width).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), _EPS)
        order = np.argsort(x @ direction).astype(np.int32, copy=False)
        rank = np.empty(n, dtype=np.int32)
        rank[order] = np.arange(n, dtype=np.int32)
        orders.append(order)
        ranks.append(rank)

    offsets = np.arange(-window, window + 1, dtype=np.int64)
    result = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = np.arange(start, end, dtype=np.int32)
        candidates = []
        for order, rank in zip(orders, ranks):
            positions = rank[rows].astype(np.int64)[:, None] + offsets[None, :]
            np.clip(positions, 0, n - 1, out=positions)
            candidates.append(order[positions])
        candidate = np.concatenate(candidates, axis=1)
        candidate.sort(axis=1)
        duplicate = np.zeros(candidate.shape, dtype=bool)
        duplicate[:, 1:] = candidate[:, 1:] == candidate[:, :-1]
        duplicate |= candidate == rows[:, None]
        safe = candidate.copy()
        safe[duplicate] = 0
        similarity = np.einsum("bd,bcd->bc", x[rows], x[safe], optimize=True)
        similarity[duplicate] = -np.inf
        chosen_pos = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        chosen_values = np.take_along_axis(similarity, chosen_pos, axis=1)
        chosen_order = np.argsort(-chosen_values, axis=1)
        chosen_pos = np.take_along_axis(chosen_pos, chosen_order, axis=1)
        result[start:end] = np.take_along_axis(candidate, chosen_pos, axis=1).astype(
            np.int32, copy=False
        )
    return result


def _search_cols_numpy(normalized_h: np.ndarray, k: int, hp: dict) -> np.ndarray:
    n = int(normalized_h.shape[0])
    if n <= int(hp.get("exact_knn_max_nodes", 12_000)):
        return _exact_knn_cols_numpy(normalized_h, k, int(hp.get("knn_query_chunk", 1024)))
    return _lsh_knn_cols_numpy(normalized_h, k, hp)


def _search_cols_live(h: torch.Tensor, k: int, hp: dict) -> torch.Tensor:
    """Choose non-differentiable kNN indices; live values are computed later."""
    n = int(h.shape[0])
    with torch.no_grad():
        normalized = F.normalize(h.detach(), dim=1, eps=1e-12)
        if n <= int(hp.get("exact_knn_max_nodes", 12_000)):
            pieces = []
            chunk = int(hp.get("knn_query_chunk", 1024))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                similarity = normalized[start:end] @ normalized.T
                local = torch.arange(end - start, device=h.device)
                similarity[local, torch.arange(start, end, device=h.device)] = -torch.inf
                pieces.append(torch.topk(similarity, k=k, dim=1).indices)
            return torch.cat(pieces, dim=0)
        normalized_np = np.ascontiguousarray(normalized.cpu().numpy(), dtype=np.float32)
        cols = _lsh_knn_cols_numpy(normalized_np, k, hp)
        return torch.from_numpy(cols).to(device=h.device, dtype=torch.long)


def _refined_from_cols(
    h: torch.Tensor, cols: torch.Tensor, hp: dict, training: bool
) -> DifferentiableKNNGraph:
    """Compute Eq. (10)-(11) weights for fixed top-k identities."""
    n, k = cols.shape
    normalized = F.normalize(h, dim=1, eps=1e-12)
    query_chunk = max(int(hp.get("knn_query_chunk", 1024)), 1)
    similarities = []
    for start in range(0, n, query_chunk):
        end = min(start + query_chunk, n)
        block_cols = cols[start:end]
        similarities.append((normalized[start:end, None, :] * normalized[block_cols]).sum(2))
    raw = F.relu(torch.cat(similarities, dim=0))
    out_degree = raw.sum(1)
    in_degree = torch.zeros(n, device=h.device, dtype=h.dtype).index_add(
        0, cols.reshape(-1), raw.reshape(-1)
    )
    degree = 1.0 + 0.5 * (out_degree + in_degree)
    inv = degree.clamp_min(_EPS).rsqrt()
    values = 0.5 * raw * inv[:, None] * inv[cols]
    dropout = float(hp.get("gsl_dropout", 0.5))
    if training and dropout > 0:
        values = F.dropout(values, p=dropout, training=True)
    diagonal = inv.square()
    return DifferentiableKNNGraph(cols, values, diagonal)


def _build_differentiable_refined(
    h: torch.Tensor, k: int, hp: dict, training: bool, cols: Optional[torch.Tensor] = None
):
    if h.shape[0] <= 1:
        raise ValueError("MDGFM graph structure learning requires at least two nodes")
    k = min(int(k), int(h.shape[0]) - 1)
    if cols is None:
        cols = _search_cols_live(h, k, hp)
    return _refined_from_cols(h, cols, hp, training), cols


@torch.no_grad()
def _build_fixed_refined(h: torch.Tensor, k: int, hp: dict) -> FixedKNNGraph:
    if h.shape[0] <= 1:
        raise ValueError("MDGFM graph structure learning requires at least two nodes")
    k = min(int(k), int(h.shape[0]) - 1)
    normalized = _l2_normalize_numpy(np.ascontiguousarray(h.detach().float().cpu().numpy()))
    cols = _search_cols_numpy(normalized, k, hp)
    rows = np.arange(normalized.shape[0], dtype=np.int64)[:, None]
    raw = np.einsum("nd,nkd->nk", normalized, normalized[cols], optimize=True).astype(
        np.float32, copy=False
    )
    np.maximum(raw, 0.0, out=raw)
    out_degree = raw.sum(1, dtype=np.float64)
    in_degree = np.bincount(
        cols.reshape(-1), weights=raw.reshape(-1), minlength=normalized.shape[0]
    )
    degree = 1.0 + 0.5 * (out_degree + in_degree)
    inv = np.power(np.maximum(degree, _EPS), -0.5).astype(np.float32)
    values = 0.5 * raw * inv[:, None] * inv[cols]
    diagonal = inv * inv
    del rows
    return FixedKNNGraph(cols, values, diagonal)


def _streaming_log_denominator(
    query: torch.Tensor, keys: torch.Tensor, temperature: float, key_chunk: int
) -> torch.Tensor:
    running = None
    for start in range(0, keys.shape[0], int(key_chunk)):
        end = min(start + int(key_chunk), keys.shape[0])
        logits = query @ keys[start:end].T / float(temperature)
        chunk_lse = torch.logsumexp(logits, dim=1)
        running = chunk_lse if running is None else torch.logaddexp(running, chunk_lse)
    if running is None:
        raise ValueError("contrastive key set is empty")
    return running


def _directional_alignment(
    query_all: torch.Tensor,
    key_all: torch.Tensor,
    positive_cols: torch.Tensor,
    positive_values: torch.Tensor,
    positive_rowptr: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float,
    key_chunk: int,
) -> torch.Tensor:
    query_all = F.normalize(query_all, dim=1, eps=1e-12)
    key_all = F.normalize(key_all, dim=1, eps=1e-12)
    query = query_all[anchors]
    log_denominator = _streaming_log_denominator(query, key_all, temperature, key_chunk)
    identity_logit = (query * key_all[anchors]).sum(1) / float(temperature)
    identity_loss = (log_denominator - identity_logit).mean()

    starts = positive_rowptr[anchors].detach().cpu().numpy()
    ends = positive_rowptr[anchors + 1].detach().cpu().numpy()
    graph_log_numerators = []
    for local, (start, end) in enumerate(zip(starts, ends)):
        cols = positive_cols[int(start) : int(end)]
        weights = positive_values[int(start) : int(end)]
        valid = weights > 0
        if bool(valid.any()):
            cols = cols[valid]
            weights = weights[valid]
            logits = (query[local : local + 1] * key_all[cols]).sum(1) / float(temperature)
            graph_log_numerators.append(
                torch.logsumexp(logits + torch.log(weights.clamp_min(_EPS)), dim=0)
            )
        else:
            graph_log_numerators.append(query.new_tensor(float(np.log(_EPS))))
    graph_log_numerator = torch.stack(graph_log_numerators)
    graph_loss = (log_denominator - graph_log_numerator).mean()
    return identity_loss + graph_loss


def alignment_loss(
    refined_embedding: torch.Tensor,
    original_embedding: torch.Tensor,
    refined_graph: DifferentiableKNNGraph,
    temperature: float = 0.2,
    anchor_batch_size: int = 128,
    key_chunk: int = 100_000,
) -> torch.Tensor:
    """Memory-bounded Eq. (13)-(15), symmetric in the two graph views."""
    n = int(refined_embedding.shape[0])
    if anchor_batch_size <= 0 or anchor_batch_size >= n:
        anchors = torch.arange(n, device=refined_embedding.device)
    else:
        anchors = (
            torch.randperm(n, device=refined_embedding.device)[: int(anchor_batch_size)]
            .sort()
            .values
        )
    cols, values, rowptr = refined_graph.positive_edges()
    # Official ``Calbound`` receives ``refinedadj.detach()``: the learned
    # topology selects and weights positives, but this positive-pair role must
    # not add a second gradient path into the graph learner.
    values = values.detach()
    forward = _directional_alignment(
        refined_embedding, original_embedding, cols, values, rowptr, anchors, temperature, key_chunk
    )
    reverse = _directional_alignment(
        original_embedding, refined_embedding, cols, values, rowptr, anchors, temperature, key_chunk
    )
    return 0.5 * (forward + reverse)


class MDGFMPretrain(nn.Module):
    """Variable-source generalization of the official fixed five-domain model."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_sources: int,
        num_layers: int,
        dropout: float,
        prompt_type: str,
    ):
        super().__init__()
        if num_sources <= 0:
            raise ValueError("MDGFM needs at least one source domain")
        self.feature_dim = int(feature_dim)
        self.prompt_type = str(prompt_type)
        self.domain_tokens = nn.ModuleList(
            [PromptToken(feature_dim, prompt_type) for _ in range(num_sources)]
        )
        self.shared_token = PromptToken(feature_dim, prompt_type)
        self.balance_tokens = nn.ModuleList(
            [PromptToken(2 * feature_dim, prompt_type) for _ in range(num_sources)]
        )
        self.gcn = MDGFMGCN(feature_dim, hidden_dim, num_layers, dropout)

    def source_features(self, source_idx: int, x: torch.Tensor) -> torch.Tensor:
        domain = F.relu(self.domain_tokens[source_idx](x))
        return self.shared_token(domain)

    def source_loss(
        self, source_idx: int, x: torch.Tensor, original_graph: FixedCSRGraph, k: int, hp: dict
    ) -> torch.Tensor:
        prompted = self.source_features(source_idx, x)
        topology = graph_spmm(original_graph, prompted, int(hp["edge_chunk"]))
        balanced = self.balance_tokens[source_idx](torch.cat((prompted, topology), dim=1))
        refined_graph, _ = _build_differentiable_refined(balanced, k, hp, training=self.training)
        refined_embedding = self.gcn(
            prompted, refined_graph, int(hp["edge_chunk"]), pretrain_view=True
        )
        original_embedding = self.gcn(
            prompted, original_graph, int(hp["edge_chunk"]), pretrain_view=True
        )
        return alignment_loss(
            refined_embedding,
            original_embedding,
            refined_graph,
            temperature=float(hp["alignment_temperature"]),
            anchor_batch_size=int(hp["alignment_batch_size"]),
            key_chunk=int(hp["contrastive_key_chunk"]),
        )

    def transfer_tokens(self):
        domains = torch.cat([token.weight.detach().clone() for token in self.domain_tokens], dim=0)
        shared = self.shared_token.weight.detach().clone()
        balances = torch.cat(
            [token.weight.detach().clone() for token in self.balance_tokens], dim=0
        )
        return domains, shared, balances


class MDGFMDownstreamPrompt(nn.Module):
    """Official dual-stream target prompt adapted to two GAD prototypes."""

    def __init__(
        self,
        source_tokens: torch.Tensor,
        shared_token: torch.Tensor,
        source_balance_tokens: torch.Tensor,
        feature_dim: int,
        prompt_type: str,
    ):
        super().__init__()
        self.register_buffer("source_tokens", source_tokens.detach().clone())
        self.register_buffer("shared_token", shared_token.detach().clone())
        self.register_buffer("source_balance_tokens", source_balance_tokens.detach().clone())
        self.prompt_type = str(prompt_type)
        self.meta_weights = nn.Parameter(torch.empty(1, int(source_tokens.shape[0])))
        self.specific_token = nn.Parameter(torch.empty(1, int(feature_dim)))
        self.fusion_weights = nn.Parameter(torch.empty(1, 2))
        self.balance_token = PromptToken(2 * feature_dim, prompt_type)
        nn.init.xavier_uniform_(self.meta_weights)
        nn.init.xavier_uniform_(self.specific_token)
        nn.init.xavier_uniform_(self.fusion_weights)
        # The released downstream constructor receives source balance prompts
        # but never uses them. A source-mean initialization preserves learned
        # Eq. (3) information and, critically, gives the exact large-target
        # dependency path a deterministic learned prompt before topology is
        # frozen. Full-target tuning may continue optimizing this parameter.
        with torch.no_grad():
            self.balance_token.weight.copy_(source_balance_tokens.mean(0, keepdim=True))

    def _apply_token(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        if self.prompt_type == "mul":
            return x * token
        if self.prompt_type == "add":
            return x + token
        raise ValueError(f"unknown MDGFM prompt type {self.prompt_type}")

    def prompt_features(self, x: torch.Tensor) -> torch.Tensor:
        composed = self.meta_weights @ self.source_tokens
        meta = self._apply_token(x, composed)
        # Official downstream ``sumtext`` is always multiplicative, even when
        # source-domain prompt composition uses the optional additive mode.
        meta = F.relu(meta) * self.shared_token
        # The released downstreamprompt always applies the specific prompt by
        # multiplication, irrespective of source-token composition type.
        specific = x * self.specific_token
        fused = self.fusion_weights[0, 0] * meta + self.fusion_weights[0, 1] * specific
        return F.elu(fused)

    def topology_features(
        self, prompted: torch.Tensor, original_graph: FixedCSRGraph, edge_chunk: int
    ) -> torch.Tensor:
        propagated = graph_spmm(original_graph, prompted, edge_chunk)
        return self.balance_token(torch.cat((prompted, propagated), dim=1))


@dataclass
class GraphContext:
    name: str
    x: np.ndarray
    original_csr: sp.csr_matrix
    original_graph: FixedCSRGraph
    labels: np.ndarray
    mark: np.ndarray


def _context(
    name: str, target: bool, feature_dim: int, feature_norm: str, hp: dict
) -> GraphContext:
    if target:
        adjacency, features, labels, mark = load_target_marked(name)
    else:
        adjacency, features, labels, mark = load_source_marked(name)
    adjacency = sp.csr_matrix(adjacency, dtype=np.float32)
    x = adapted_features(
        name,
        features,
        adjacency,
        feature_dim,
        feature_norm,
        use_cache=bool(hp.get("cache_features", True)),
    )
    original = _normalized_original_adjacency(adjacency)
    return GraphContext(
        name=name,
        x=np.ascontiguousarray(x, dtype=np.float32),
        original_csr=original,
        original_graph=FixedCSRGraph(original),
        labels=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
    )


def _knn_k(name: str, hp: dict) -> int:
    overrides = hp.get("knn_k_by_dataset", {})
    if name in overrides:
        return int(overrides[name])
    homophilic = {
        str(item).lower() for item in hp.get("homophilic_datasets", ("cora", "citeseer", "pubmed"))
    }
    if str(name).lower() in homophilic:
        return int(hp.get("homophilic_knn_k", 30))
    return int(hp.get("knn_k", 15))


def _copy_state(module: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _pretrain_one_seed(
    model: MDGFMPretrain, contexts: Sequence[GraphContext], hp: dict, seed: int, device: str
):
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["pretrain_lr"]),
        weight_decay=float(hp["pretrain_weight_decay"]),
    )
    source_x = [torch.from_numpy(ctx.x).to(device=device, dtype=torch.float32) for ctx in contexts]
    epochs = int(hp["pretrain_epochs"])
    patience = int(hp.get("pretrain_patience", 500))
    best_loss = float("inf")
    best_state = _copy_state(model)
    wait = 0

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for source_idx, (ctx, x) in enumerate(zip(contexts, source_x)):
            loss = model.source_loss(source_idx, x, ctx.original_graph, _knn_k(ctx.name, hp), hp)
            (loss / len(contexts)).backward()
            total += float(loss.detach().item()) / len(contexts)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp.get("clip_grad", 8.0)))
        optimizer.step()

        if total < best_loss:
            best_loss = total
            best_state = _copy_state(model)
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % max(epochs // 5, 1) == 0:
            print(
                f"    [mdgfm-pretrain] seed={seed} epoch={epoch + 1}/{epochs} " f"loss={total:.4f}",
                flush=True,
            )
        if wait >= patience:
            print(f"    [mdgfm-pretrain] early stop at epoch={epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    del source_x


def _class_prototypes(
    embedding: torch.Tensor, support_idx: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    selected = embedding[support_idx]
    centers = []
    for cls in (0, 1):
        mask = support_labels == cls
        if not bool(mask.any()):
            raise ValueError(f"MDGFM support has no class {cls}")
        centers.append(selected[mask].mean(0))
    return torch.stack(centers, dim=0)


def _prototype_logits(
    embedding: torch.Tensor,
    idx: torch.Tensor,
    support_idx: torch.Tensor,
    support_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    centers = _class_prototypes(embedding, support_idx, support_labels)
    logits = F.cosine_similarity(embedding[idx].unsqueeze(1), centers.unsqueeze(0), dim=-1)
    return logits / float(temperature)


def _dependency_subgraph(
    original: sp.csr_matrix, refined: sp.csr_matrix, support: np.ndarray, layers: int
):
    """Complete L-hop row dependency closure; no fanout or node sampling."""
    n = original.shape[0]
    seen = np.zeros(n, dtype=bool)
    frontier = np.unique(np.asarray(support, dtype=np.int64))
    seen[frontier] = True
    for _ in range(int(layers)):
        if frontier.size == 0:
            break
        original_cols = original[frontier].indices
        refined_cols = refined[frontier].indices
        if original_cols.size and refined_cols.size:
            candidates = np.concatenate((original_cols, refined_cols))
        elif original_cols.size:
            candidates = original_cols
        else:
            candidates = refined_cols
        if candidates.size == 0:
            break
        candidates = np.unique(candidates)
        new = candidates[~seen[candidates]]
        if new.size == 0:
            break
        seen[new] = True
        frontier = new
    nodes = np.flatnonzero(seen).astype(np.int64, copy=False)
    support_rel = np.searchsorted(nodes, np.asarray(support, dtype=np.int64))
    return nodes, support_rel


@torch.no_grad()
def _initial_fixed_target_graph(
    adapter: MDGFMDownstreamPrompt, ctx: GraphContext, k: int, hp: dict, device: str
):
    x = torch.from_numpy(ctx.x).to(device=device, dtype=torch.float32)
    prompted = adapter.prompt_features(x)
    topology = adapter.topology_features(prompted, ctx.original_graph, int(hp["edge_chunk"]))
    refined = _build_fixed_refined(topology, k, hp)
    del x, prompted, topology
    return refined


def _train_target_full(
    adapter: MDGFMDownstreamPrompt,
    gcn: MDGFMGCN,
    ctx: GraphContext,
    support: np.ndarray,
    support_labels: torch.Tensor,
    k: int,
    hp: dict,
    device: str,
):
    """Prompt tuning with live sparse weights and fixed top-k identities."""
    x = torch.from_numpy(ctx.x).to(device=device, dtype=torch.float32)
    support_idx = torch.from_numpy(support).to(device=device, dtype=torch.long)
    with torch.no_grad():
        prompted = adapter.prompt_features(x)
        topology = adapter.topology_features(prompted, ctx.original_graph, int(hp["edge_chunk"]))
        cols = _search_cols_live(topology, min(k, len(ctx.x) - 1), hp)
        del prompted, topology

    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(hp["downstream_lr"]))
    steps = int(hp["downstream_steps"])
    best = float("inf")
    best_state = _copy_state(adapter)
    refresh_interval = int(hp.get("target_knn_refresh_interval", 0))
    refresh_limit = int(hp.get("target_knn_refresh_max_nodes", 2_000))

    for step in range(steps):
        prompted = adapter.prompt_features(x)
        topology = adapter.topology_features(prompted, ctx.original_graph, int(hp["edge_chunk"]))
        if (
            refresh_interval > 0
            and len(ctx.x) <= refresh_limit
            and step > 0
            and step % refresh_interval == 0
        ):
            cols = _search_cols_live(topology, min(k, len(ctx.x) - 1), hp)
        refined = _refined_from_cols(topology, cols, hp, training=True)
        graph = BlendedGraph(ctx.original_graph, refined, float(hp["adjacency_mix"]))
        embedding = gcn(prompted, graph, int(hp["edge_chunk"]), pretrain_view=False)
        logits = _prototype_logits(
            embedding,
            support_idx,
            support_idx,
            support_labels,
            float(hp.get("prototype_temperature", 1.0)),
        )
        loss = F.cross_entropy(logits, support_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(hp.get("clip_grad", 8.0)))
        optimizer.step()
        value = float(loss.detach().item())
        if value < best:
            best = value
            best_state = _copy_state(adapter)
        if (step + 1) % max(steps // 4, 1) == 0:
            print(
                f"    [mdgfm-down] {ctx.name} step={step + 1}/{steps} " f"loss={value:.4f}",
                flush=True,
            )
    adapter.load_state_dict(best_state)
    del x, cols


def _train_target_dependency(
    adapter: MDGFMDownstreamPrompt,
    gcn: MDGFMGCN,
    ctx: GraphContext,
    support: np.ndarray,
    support_labels: torch.Tensor,
    k: int,
    hp: dict,
    device: str,
):
    """Large-target tuning on the exact closure of one fixed sparse topology."""
    initial_refined = _initial_fixed_target_graph(adapter, ctx, k, hp, device)
    refined_csr = initial_refined.to_csr()
    nodes, support_rel = _dependency_subgraph(
        ctx.original_csr, refined_csr, support, int(hp["layers_num"])
    )
    alpha = float(hp["adjacency_mix"])
    blended = (
        (alpha * ctx.original_csr[nodes][:, nodes] + (1.0 - alpha) * refined_csr[nodes][:, nodes])
        .tocsr()
        .astype(np.float32, copy=False)
    )
    graph = FixedCSRGraph(blended)
    x = torch.from_numpy(np.ascontiguousarray(ctx.x[nodes], dtype=np.float32)).to(device=device)
    support_idx = torch.from_numpy(support_rel).to(device=device, dtype=torch.long)
    trainable = [
        parameter
        for name, parameter in adapter.named_parameters()
        if not name.startswith("balance_token.")
    ]
    optimizer = torch.optim.Adam(trainable, lr=float(hp["downstream_lr"]))
    steps = int(hp["downstream_steps"])
    best = float("inf")
    best_state = _copy_state(adapter)
    print(
        f"    [mdgfm-large-support-closure] {ctx.name}: support={len(support)} "
        f"nodes={len(nodes)} edges={blended.nnz} steps={steps}",
        flush=True,
    )

    for step in range(steps):
        prompted = adapter.prompt_features(x)
        embedding = gcn(prompted, graph, int(hp["edge_chunk"]), pretrain_view=False)
        logits = _prototype_logits(
            embedding,
            support_idx,
            support_idx,
            support_labels,
            float(hp.get("prototype_temperature", 1.0)),
        )
        loss = F.cross_entropy(logits, support_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(hp.get("clip_grad", 8.0)))
        optimizer.step()
        value = float(loss.detach().item())
        if value < best:
            best = value
            best_state = _copy_state(adapter)
        if (step + 1) % max(steps // 4, 1) == 0:
            print(
                f"    [mdgfm-down-large] {ctx.name} step={step + 1}/{steps} " f"loss={value:.4f}",
                flush=True,
            )
    adapter.load_state_dict(best_state)
    del initial_refined, refined_csr, blended, graph, x


@torch.no_grad()
def _final_target_embedding(
    adapter: MDGFMDownstreamPrompt, gcn: MDGFMGCN, ctx: GraphContext, k: int, hp: dict, device: str
) -> np.ndarray:
    x = torch.from_numpy(ctx.x).to(device=device, dtype=torch.float32)
    prompted = adapter.prompt_features(x)
    topology = adapter.topology_features(prompted, ctx.original_graph, int(hp["edge_chunk"]))
    refined = _build_fixed_refined(topology, k, hp)
    del topology
    original_weight = float(hp["adjacency_mix"])
    blended_csr = (
        (original_weight * ctx.original_csr + (1.0 - original_weight) * refined.to_csr())
        .tocsr()
        .astype(np.float32, copy=False)
    )
    blended_csr.sum_duplicates()
    blended_csr.sort_indices()
    # Final inference has fixed prompts/topology. Combining the two sparse
    # operators first is exact and avoids holding two N-by-hidden propagation
    # outputs simultaneously on the accelerator.
    graph = FixedCSRGraph(blended_csr)
    del refined, blended_csr
    embedding = gcn(prompted, graph, int(hp["edge_chunk"]), pretrain_view=False)
    embedding_cpu = np.ascontiguousarray(embedding.float().cpu().numpy(), dtype=np.float32)
    del x, prompted, graph, embedding
    return embedding_cpu


def _score_from_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    query_chunk: int,
    temperature: float,
):
    support_labels = labels[support]
    centers = []
    for cls in (0, 1):
        centers.append(embedding[support[support_labels == cls]].mean(0))
    centers = _l2_normalize_numpy(np.stack(centers).astype(np.float32))
    scores = np.empty(query.size, dtype=np.float32)
    for start in range(0, query.size, int(query_chunk)):
        end = min(start + int(query_chunk), query.size)
        block = _l2_normalize_numpy(embedding[query[start:end]])
        logits = block @ centers.T / float(temperature)
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(1, keepdims=True)
        scores[start:end] = probabilities[:, 1]
    return evaluate(labels[query], scores)


def _score_target(
    model: MDGFMPretrain,
    ctx: GraphContext,
    hp: dict,
    shot: int,
    seed: int,
    feature_dim: int,
    device: str,
):
    rng = np.random.RandomState(seed + 50_000)
    normal = sample_class_support(
        ctx.labels, ctx.mark, 0, shot, rng, reserve_query=True, dataset=ctx.name
    )
    anomaly = sample_class_support(
        ctx.labels, ctx.mark, 1, shot, rng, reserve_query=True, dataset=ctx.name
    )
    support = np.concatenate((normal, anomaly)).astype(np.int64)
    query_mask = ctx.mark.copy()
    query_mask[np.unique(support)] = False
    query = np.flatnonzero(query_mask).astype(np.int64, copy=False)
    if query.size == 0:
        raise ValueError(f"{ctx.name}: no query nodes after MDGFM support selection")

    source_tokens, shared_token, source_balance_tokens = model.transfer_tokens()
    adapter = MDGFMDownstreamPrompt(
        source_tokens.to(device),
        shared_token.to(device),
        source_balance_tokens.to(device),
        feature_dim,
        model.prompt_type,
    ).to(device)
    support_labels = torch.tensor(
        [0] * len(normal) + [1] * len(anomaly), dtype=torch.long, device=device
    )
    k = _knn_k(ctx.name, hp)
    gcn = model.gcn
    gcn.eval()
    old_requires_grad = [parameter.requires_grad for parameter in gcn.parameters()]
    for parameter in gcn.parameters():
        parameter.requires_grad_(False)
    adapter.train()
    try:
        with torch.enable_grad():
            if len(ctx.x) <= int(hp.get("full_target_tune_max_nodes", 5_000)):
                _train_target_full(adapter, gcn, ctx, support, support_labels, k, hp, device)
            else:
                _train_target_dependency(adapter, gcn, ctx, support, support_labels, k, hp, device)
    finally:
        for parameter, required in zip(gcn.parameters(), old_requires_grad):
            parameter.requires_grad_(required)

    adapter.eval()
    embedding = _final_target_embedding(adapter, gcn, ctx, k, hp, device)
    result = _score_from_embedding(
        embedding,
        ctx.labels,
        support,
        query,
        int(hp["eval_query_batch"]),
        float(hp.get("prototype_temperature", 1.0)),
    )
    del embedding, adapter
    return result


def run_mdgfm(sources, targets, seeds, hp, device, shot=10, feature_dim=8, feature_norm="zscore"):
    """Run source pretraining and binary few-shot target adaptation."""
    if not sources:
        raise ValueError("MDGFM source list cannot be empty")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"MDGFM requested {device}, but CUDA is unavailable")
    os.environ["GAD_BENCHMARK_DEVICE"] = str(device)
    source_contexts = [
        _context(name, target=False, feature_dim=feature_dim, feature_norm=feature_norm, hp=hp)
        for name in sources
    ]
    trained: List[Tuple[int, MDGFMPretrain]] = []
    for seed in seeds:
        set_seed(int(seed))
        model = MDGFMPretrain(
            feature_dim=feature_dim,
            hidden_dim=int(hp["hid_units"]),
            num_sources=len(source_contexts),
            num_layers=int(hp["layers_num"]),
            dropout=float(hp["pretrain_dropout"]),
            prompt_type=str(hp["combinetype"]),
        ).to(device)
        _pretrain_one_seed(model, source_contexts, hp, int(seed), device)
        model.cpu()
        trained.append((int(seed), model))
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    per_target = {name: [] for name in targets}
    for target_idx, name in enumerate(targets, start=1):
        print(f"    [mdgfm-target {target_idx}/{len(targets)}] loading {name}", flush=True)
        context = _context(
            name, target=True, feature_dim=feature_dim, feature_norm=feature_norm, hp=hp
        )
        for seed_idx, (seed, model) in enumerate(trained, start=1):
            print(
                f"    [mdgfm-target {target_idx}/{len(targets)}] {name} "
                f"seed={seed} ({seed_idx}/{len(trained)})",
                flush=True,
            )
            model.to(device)
            try:
                set_seed(seed)
                per_target[name].append(
                    _score_target(model, context, hp, shot, seed, feature_dim, device)
                )
            finally:
                model.cpu()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
        del context
        gc.collect()
    return aggregate(per_target)
