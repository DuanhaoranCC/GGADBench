"""Paper-only TPCA-GAD reproduction for the second benchmark block.

The implementation follows Eqs. (1)--(12) from:
"TPCA-GAD: Topology preference-consistency aggregation for zero-shot graph
anomaly detection" (Expert Systems With Applications, 2026).

The paper does not release code and omits several executable details. Those
choices are centralized in ``ggad.config.TPCAGAD_HP``. In particular,
the default implementation uses a target-label-free 8-D SVD attribute adapter
and layer-specific topology-audit MLPs so that the consistency term in Eq. (12)
is meaningful. ``compatibility_mode='shared_static'`` preserves the literal
Eq. (5)/Algorithm 1 interpretation, for which that term is exactly zero.
"""

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint

import ggad.config as C
from common.data import CACHE_ROOT, aggregate, load_source_marked, load_target_marked, x_svd
from util import evaluate, set_seed

try:
    from numba import njit
except ImportError:  # exact Python fallback remains available
    njit = None


CACHE = CACHE_ROOT / "tpcagad"


def _binary_undirected_loop_free(adj) -> sp.csr_matrix:
    """Return the paper-reproduction structural graph.

    Input weights, directions, duplicate entries, and pre-existing self-loops
    are intentionally discarded. Equation (6) aggregates neighbors only.
    """
    graph = sp.coo_matrix(adj, dtype=np.float32)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("TPCA-GAD adjacency must be square")
    if not np.isfinite(graph.data).all():
        raise ValueError("TPCA-GAD adjacency must contain finite values")
    graph.eliminate_zeros()
    keep = graph.row != graph.col
    graph = sp.csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (graph.row[keep], graph.col[keep]),
        ),
        shape=graph.shape,
    )
    graph.sum_duplicates()
    if graph.nnz:
        graph.data.fill(1.0)
    graph = graph.maximum(graph.T).tocsr()
    if graph.nnz:
        graph.data.fill(1.0)
    graph.sum_duplicates()
    graph.sort_indices()
    return graph


def _core_numbers_impl(indptr, indices):
    """Exact Batagelj--Zaversnik core decomposition on a simple CSR graph."""
    n = len(indptr) - 1
    degree = (indptr[1:] - indptr[:-1]).astype(np.int64)
    if n == 0:
        return degree
    max_degree = int(degree.max())
    bins = np.zeros(max_degree + 1, dtype=np.int64)
    for v in range(n):
        bins[degree[v]] += 1

    start = 0
    for d in range(max_degree + 1):
        count = bins[d]
        bins[d] = start
        start += count

    position = np.empty(n, dtype=np.int64)
    vertices = np.empty(n, dtype=np.int64)
    for v in range(n):
        position[v] = bins[degree[v]]
        vertices[position[v]] = v
        bins[degree[v]] += 1

    for d in range(max_degree, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0

    for order in range(n):
        v = vertices[order]
        for edge in range(indptr[v], indptr[v + 1]):
            u = indices[edge]
            if degree[u] > degree[v]:
                du = degree[u]
                pu = position[u]
                pw = bins[du]
                w = vertices[pw]
                if u != w:
                    position[u] = pw
                    position[w] = pu
                    vertices[pu] = w
                    vertices[pw] = u
                bins[du] += 1
                degree[u] -= 1
    return degree


def _forward_csr_impl(indptr, indices, degree):
    """Orient each undirected edge by (degree, node id), preserving exactness."""
    n = len(indptr) - 1
    counts = np.zeros(n, dtype=np.int64)
    for u in range(n):
        du = degree[u]
        for edge in range(indptr[u], indptr[u + 1]):
            v = indices[edge]
            dv = degree[v]
            if du < dv or (du == dv and u < v):
                counts[u] += 1

    forward_indptr = np.zeros(n + 1, dtype=np.int64)
    for u in range(n):
        forward_indptr[u + 1] = forward_indptr[u] + counts[u]
    forward_indices = np.empty(forward_indptr[-1], dtype=np.int64)
    cursor = forward_indptr[:-1].copy()
    for u in range(n):
        du = degree[u]
        for edge in range(indptr[u], indptr[u + 1]):
            v = indices[edge]
            dv = degree[v]
            if du < dv or (du == dv and u < v):
                forward_indices[cursor[u]] = v
                cursor[u] += 1
    return forward_indptr, forward_indices


def _triangle_counts_impl(forward_indptr, forward_indices):
    """Count every triangle once by intersecting degree-oriented lists."""
    n = len(forward_indptr) - 1
    triangles = np.zeros(n, dtype=np.int64)
    for u in range(n):
        u_start = forward_indptr[u]
        u_end = forward_indptr[u + 1]
        for uv in range(u_start, u_end):
            v = forward_indices[uv]
            left = u_start
            right = forward_indptr[v]
            right_end = forward_indptr[v + 1]
            while left < u_end and right < right_end:
                a = forward_indices[left]
                b = forward_indices[right]
                if a == b:
                    triangles[u] += 1
                    triangles[v] += 1
                    triangles[a] += 1
                    left += 1
                    right += 1
                elif a < b:
                    left += 1
                else:
                    right += 1
    return triangles


if njit is not None:
    _core_numbers_kernel = njit(cache=False)(_core_numbers_impl)
    _forward_csr_kernel = njit(cache=False)(_forward_csr_impl)
    _triangle_counts_kernel = njit(cache=False)(_triangle_counts_impl)
else:
    _core_numbers_kernel = _core_numbers_impl
    _forward_csr_kernel = _forward_csr_impl
    _triangle_counts_kernel = _triangle_counts_impl


def _exact_core_numbers(adj: sp.csr_matrix) -> np.ndarray:
    indptr = np.asarray(adj.indptr, dtype=np.int64)
    indices = np.asarray(adj.indices, dtype=np.int64)
    return np.asarray(_core_numbers_kernel(indptr, indices), dtype=np.int64)


def _exact_clustering_coefficients(adj: sp.csr_matrix) -> np.ndarray:
    degree = np.diff(adj.indptr).astype(np.int64, copy=False)
    indptr = np.asarray(adj.indptr, dtype=np.int64)
    indices = np.asarray(adj.indices, dtype=np.int64)
    forward_indptr, forward_indices = _forward_csr_kernel(indptr, indices, degree)
    triangles = np.asarray(
        _triangle_counts_kernel(forward_indptr, forward_indices),
        dtype=np.float64,
    )
    denominator = degree.astype(np.float64) * np.maximum(degree - 1, 0)
    clustering = np.zeros(adj.shape[0], dtype=np.float64)
    valid = denominator > 0
    clustering[valid] = 2.0 * triangles[valid] / denominator[valid]
    return clustering.astype(np.float32)


def _graph_digest(adj: sp.csr_matrix, hp: Dict) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(adj.shape).encode())
    digest.update(str(hp["cache_version"]).encode())
    digest.update(str(hp["degree_normalization"]).encode())
    digest.update(str(hp["core_normalization"]).encode())
    digest.update(str(hp["pool"]).encode())
    digest.update(np.ascontiguousarray(adj.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(adj.indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _load_or_build_structural_inputs(
    name: str, adj, hp: Dict
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build exact Eq. (1)--(3) fixed inputs and cache label-free results."""
    print(f"    [tpcagad/structure] {name}: canonicalizing graph", flush=True)
    graph = _binary_undirected_loop_free(adj)
    cache_path: Optional[Path] = None
    if hp.get("cache", True):
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE / f"{_graph_digest(graph, hp)}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    local, pooled = cached["local"], cached["pooled"]
                for value in (local, pooled):
                    if (
                        value.shape != (graph.shape[0], 3)
                        or value.dtype != np.float32
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError("invalid structural cache array")
            except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile):
                print(f"    [tpcagad/structure] {name}: rebuilding invalid cache", flush=True)
            else:
                print(f"    [tpcagad/structure] {name}: cache hit", flush=True)
                return graph, local, pooled

    if njit is None and graph.nnz > int(hp["python_structure_max_edges"]):
        raise RuntimeError(
            f"TPCA-GAD exact structural preprocessing for {name} has "
            f"{graph.nnz:,} edges and requires Numba. Install a compatible "
            "Numba build in the experiment environment or copy the generated "
            "cache/tpcagad cache from another machine."
        )

    backend = "numba" if njit is not None else "python"
    print(
        f"    [tpcagad/structure] {name}: exact core numbers ({backend})",
        flush=True,
    )
    degree = np.diff(graph.indptr).astype(np.float32, copy=False)
    core = _exact_core_numbers(graph).astype(np.float32)
    print(
        f"    [tpcagad/structure] {name}: exact clustering coefficients ({backend})",
        flush=True,
    )
    clustering = _exact_clustering_coefficients(graph)

    if hp["degree_normalization"] != "graph_max":
        raise ValueError("TPCA-GAD currently supports degree_normalization='graph_max'")
    if hp["core_normalization"] != "graph_max":
        raise ValueError("TPCA-GAD currently supports core_normalization='graph_max'")
    degree_scale = float(degree.max()) if degree.size and degree.max() > 0 else 1.0
    core_scale = float(core.max()) if core.size and core.max() > 0 else 1.0
    local = np.column_stack((degree / degree_scale, clustering, core / core_scale)).astype(
        np.float32
    )

    if hp["pool"] != "mean_exclude_center":
        raise ValueError("TPCA-GAD currently supports pool='mean_exclude_center'")
    pooled = np.asarray(graph @ local, dtype=np.float32)
    nonzero = degree > 0
    pooled[nonzero] /= degree[nonzero, None]
    pooled[~nonzero] = 0.0

    if cache_path is not None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez(handle, local=local, pooled=pooled)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        print(f"    [tpcagad/structure] {name}: cache saved", flush=True)
    return graph, local, pooled


def _aligned_attributes(feat, hp: Dict) -> np.ndarray:
    dim = int(hp["attribute_dim"])
    aligned = np.asarray(
        x_svd(feat, dim, cache=bool(hp.get("cache", True))),
        dtype=np.float32,
    )
    if aligned.ndim == 1:
        aligned = aligned[:, None]
    if aligned.shape[1] < dim:
        padded = np.zeros((aligned.shape[0], dim), dtype=np.float32)
        padded[:, : aligned.shape[1]] = aligned
        aligned = padded
    elif aligned.shape[1] > dim:
        aligned = aligned[:, :dim]
    return np.ascontiguousarray(aligned, dtype=np.float32)


def _stratified_source_split(
    labels: np.ndarray,
    seed: int,
    fractions: Sequence[float] = (0.3, 0.1, 0.6),
    eligible: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded source-only 30/10/60 split with a deterministic rare-class fallback."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if (
        fractions.shape != (3,)
        or not np.isfinite(fractions).all()
        or (fractions <= 0).any()
        or not np.isclose(fractions.sum(), 1.0)
    ):
        raise ValueError("source_split must contain three positive finite fractions summing to one")
    indices = np.arange(labels.size, dtype=np.int64)
    if eligible is not None:
        indices = indices[np.asarray(eligible, dtype=bool).reshape(-1)]
    if indices.size < 3:
        raise ValueError("TPCA-GAD source split requires at least three labeled nodes")

    y = labels[indices]
    try:
        train_idx, remainder = train_test_split(
            indices,
            train_size=float(fractions[0]),
            random_state=int(seed),
            shuffle=True,
            stratify=y,
        )
        relative_val = float(fractions[1] / (fractions[1] + fractions[2]))
        val_idx, test_idx = train_test_split(
            remainder,
            train_size=relative_val,
            random_state=int(seed) + 1,
            shuffle=True,
            stratify=labels[remainder],
        )
        return (
            np.sort(train_idx.astype(np.int64)),
            np.sort(val_idx.astype(np.int64)),
            np.sort(test_idx.astype(np.int64)),
        )
    except ValueError:
        rng = np.random.RandomState(seed)
        groups = []
        for cls in np.unique(y):
            group = indices[y == cls].copy()
            rng.shuffle(group)
            n = group.size
            n_train = int(np.floor(n * fractions[0]))
            n_val = int(np.floor(n * fractions[1]))
            if n >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                if n_train + n_val >= n:
                    n_val = max(0, n - n_train - 1)
            groups.append(
                (group[:n_train], group[n_train : n_train + n_val], group[n_train + n_val :])
            )
        train_idx = np.concatenate([group[0] for group in groups])
        val_idx = np.concatenate([group[1] for group in groups])
        test_idx = np.concatenate([group[2] for group in groups])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


class CSRChunkGraph:
    """CPU-resident exact simple graph with row-complete edge chunks."""

    def __init__(self, adj: sp.csr_matrix):
        graph = sp.csr_matrix(adj)
        graph.sort_indices()
        self.indptr = np.asarray(graph.indptr, dtype=np.int64)
        self.indices = np.asarray(graph.indices, dtype=np.int64)
        self.shape = graph.shape
        self.n = int(graph.shape[0])
        self.nnz = int(graph.nnz)
        self.degree = np.diff(self.indptr).astype(np.int64, copy=False)

    def row_ranges(
        self, max_edges: int, max_nodes: Optional[int] = None
    ) -> Iterable[Tuple[int, int]]:
        max_edges = max(1, int(max_edges))
        max_nodes = self.n if max_nodes is None else max(1, int(max_nodes))
        start = 0
        while start < self.n:
            edge_limit = int(self.indptr[start]) + max_edges
            edge_end = int(np.searchsorted(self.indptr, edge_limit, side="right") - 1)
            end = min(self.n, start + max_nodes, max(start + 1, edge_end))
            yield start, end
            start = end

    def edge_arrays(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        edge_start = int(self.indptr[start])
        edge_end = int(self.indptr[end])
        counts = self.degree[start:end]
        rows = np.repeat(np.arange(end - start, dtype=np.int64), counts)
        cols = self.indices[edge_start:edge_end]
        return rows, cols

    def edge_tensors(self, start: int, end: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, cols = self.edge_arrays(start, end)
        return (
            torch.from_numpy(rows).to(device=device),
            torch.from_numpy(np.asarray(cols, dtype=np.int64)).to(device=device),
        )


@dataclass
class PreparedGraph:
    name: str
    adjacency: sp.csr_matrix
    chunks: CSRChunkGraph
    local: np.ndarray
    pooled: np.ndarray
    attributes: np.ndarray
    labels: np.ndarray
    mark: np.ndarray


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        if depth == 1:
            self.net = nn.Linear(input_dim, output_dim)
        elif depth == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.PReLU(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError("TPCA-GAD reproduction supports MLP depth 1 or 2")

    def forward(self, x):
        return self.net(x)


class TPCABlock(nn.Module):
    """One compatibility-weighted aggregation plus Eq. (7)--(8) residual."""

    def __init__(self, hidden_dim: int, fingerprint_dim: int):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = nn.PReLU(hidden_dim)
        self.gate = nn.Linear(hidden_dim + fingerprint_dim, hidden_dim)


class TPCAGAD(nn.Module):
    """TPCA-GAD architecture reconstructed from Eqs. (2)--(12)."""

    def __init__(self, hp: Optional[Dict] = None):
        super().__init__()
        cfg = _resolved_hp(hp)
        self.hp = cfg
        hidden = int(cfg["hidden_dim"])
        audit = int(cfg["audit_dim"])
        depth = int(cfg["mlp_depth"])
        self.num_layers = int(cfg["num_layers"])
        self.compatibility_mode = cfg["compatibility_mode"]

        self.context_mlp = _MLP(3, hidden, hidden, depth)
        self.fingerprint_dim = 3 + hidden
        self.attr_projection = _MLP(int(cfg["attribute_dim"]), hidden, hidden, 1)
        self.input_mlp = _MLP(self.fingerprint_dim + hidden, hidden, hidden, depth)
        self.structural_projection = _MLP(self.fingerprint_dim, hidden, hidden, depth)
        audit_count = 1 if self.compatibility_mode == "shared_static" else self.num_layers
        self.audit_mlps = nn.ModuleList(
            [_MLP(self.fingerprint_dim, audit, hidden, depth) for _ in range(audit_count)]
        )
        self.blocks = nn.ModuleList(
            [TPCABlock(hidden, self.fingerprint_dim) for _ in range(self.num_layers)]
        )
        self.classifier = nn.Linear(hidden, 1)

    def fingerprint(self, local: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        return torch.cat((local, self.context_mlp(pooled)), dim=1)

    def initial_state(self, phi: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        attr = self.attr_projection(attributes)
        return self.input_mlp(torch.cat((phi, attr), dim=1))

    def audit_embedding(self, phi: torch.Tensor, layer: int) -> torch.Tensor:
        index = 0 if self.compatibility_mode == "shared_static" else layer
        return self.audit_mlps[index](phi)

    def forward(
        self,
        local: torch.Tensor,
        pooled: torch.Tensor,
        attributes: torch.Tensor,
        graph: CSRChunkGraph,
        checkpoint_chunks: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        phi = self.fingerprint(local, pooled)
        h = self.initial_state(phi, attributes)
        structural = self.structural_projection(phi)
        audits = [self.audit_embedding(phi, layer) for layer in range(self.num_layers)]
        consistency_terms = []
        for layer, block in enumerate(self.blocks):
            message = block.message(h)
            previous = audits[layer - 1] if layer > 0 else None
            aggregated, consistency = _exact_tpca_aggregate(
                graph,
                message,
                audits[layer],
                previous,
                float(self.hp["tau"]),
                int(self.hp["edge_chunk"]),
                checkpoint_chunks,
            )
            aggregated = block.activation(aggregated)
            gate = torch.sigmoid(block.gate(torch.cat((aggregated, phi), dim=1)))
            h = (1.0 - gate) * aggregated + gate * structural
            if layer > 0:
                consistency_terms.append(consistency)
        if consistency_terms:
            consistency_loss = torch.stack(consistency_terms).mean()
        else:
            consistency_loss = h.new_zeros(())
        return h, audits, consistency_loss

    def anomaly_score_components(
        self, h: torch.Tensor, final_audit: torch.Tensor, graph: CSRChunkGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _score_components_from_embeddings(
            graph,
            h,
            final_audit,
            float(self.hp["tau"]),
            int(self.hp["edge_chunk"]),
        )

    def anomaly_scores(
        self, h: torch.Tensor, final_audit: torch.Tensor, graph: CSRChunkGraph
    ) -> torch.Tensor:
        representation, topology = self.anomaly_score_components(h, final_audit, graph)
        return _combine_score_components(representation, topology, float(self.hp["alpha"]))


def _chunk_aggregate_values(
    message: torch.Tensor,
    audit: torch.Tensor,
    previous_audit: Optional[torch.Tensor],
    graph: CSRChunkGraph,
    start: int,
    end: int,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = graph.edge_tensors(start, end, message.device)
    row_count = end - start
    neighbor_sum = message.new_zeros((row_count, message.shape[1]))
    denominator = message.new_zeros(row_count)
    consistency = message.new_zeros(())
    if col.numel():
        global_row = row + start
        delta = audit[global_row] - audit[col]
        log_compatibility = -(delta * delta).sum(1) / tau
        row_max = log_compatibility.new_full((row_count,), float("-inf"))
        row_max = row_max.scatter_reduce(
            0, row, log_compatibility, reduce="amax", include_self=True
        )
        normalized_compatibility = torch.exp(log_compatibility - row_max[row])
        neighbor_sum = neighbor_sum.index_add(
            0, row, message[col] * normalized_compatibility.unsqueeze(1)
        )
        denominator = denominator.index_add(0, row, normalized_compatibility)
        if previous_audit is not None:
            compatibility = torch.exp(log_compatibility)
            previous_delta = previous_audit[global_row] - previous_audit[col]
            previous_compatibility = torch.exp(-(previous_delta * previous_delta).sum(1) / tau)
            consistency = ((compatibility - previous_compatibility) ** 2).sum()
    # Eq. (6) aggregates N(i), which excludes the center node. Keep the
    # current message only as a finite fallback for isolated nodes.
    has_neighbor_weight = denominator > 0
    safe_denominator = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    neighbor_mean = neighbor_sum / safe_denominator.unsqueeze(1)
    output = torch.where(has_neighbor_weight.unsqueeze(1), neighbor_mean, message[start:end])
    return output, consistency


def _exact_tpca_aggregate(
    graph: CSRChunkGraph,
    message: torch.Tensor,
    audit: torch.Tensor,
    previous_audit: Optional[torch.Tensor],
    tau: float,
    edge_chunk: int,
    checkpoint_chunks: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Eq. (5)--(6) over row-complete CSR chunks.

    During training, checkpoint closures retain only CPU graph metadata and
    node-level tensors. Edge gathers and compatibility vectors are recomputed in
    backward, preventing an ``O(E * d)`` autograd footprint.
    """
    outputs = []
    consistency_parts = []
    for start, end in graph.row_ranges(edge_chunk):
        if checkpoint_chunks and torch.is_grad_enabled():
            if previous_audit is None:

                def chunk_fn(msg, aud, s=start, e=end):
                    return _chunk_aggregate_values(msg, aud, None, graph, s, e, tau)

                output, consistency = checkpoint(chunk_fn, message, audit, use_reentrant=False)
            else:

                def chunk_fn(msg, aud, prev, s=start, e=end):
                    return _chunk_aggregate_values(msg, aud, prev, graph, s, e, tau)

                output, consistency = checkpoint(
                    chunk_fn,
                    message,
                    audit,
                    previous_audit,
                    use_reentrant=False,
                )
        else:
            output, consistency = _chunk_aggregate_values(
                message, audit, previous_audit, graph, start, end, tau
            )
        outputs.append(output)
        consistency_parts.append(consistency)
    if outputs:
        result = torch.cat(outputs, dim=0)
        consistency = torch.stack(consistency_parts).sum() / max(graph.nnz, 1)
    else:
        result = message.new_empty((0, message.shape[1]))
        consistency = message.new_zeros(())
    return result, consistency


def _combine_score_components(representation, topology, alpha: float):
    """Combine the two Eq. (9) views without rerunning target inference."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"TPCA-GAD alpha must be finite and in [0, 1], got {alpha}")
    if representation.shape != topology.shape:
        raise ValueError("TPCA-GAD score components must have identical shapes")
    return alpha * representation + (1.0 - alpha) * topology


@torch.no_grad()
def _score_components_from_embeddings(
    graph: CSRChunkGraph,
    h: torch.Tensor,
    audit: torch.Tensor,
    tau: float,
    edge_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the exact representation/topology terms of raw Eq. (9)."""
    representation_scores = h.new_empty(graph.n)
    topology_scores = h.new_empty(graph.n)
    for start, end in graph.row_ranges(edge_chunk):
        row, col = graph.edge_tensors(start, end, h.device)
        row_count = end - start
        neighbor_sum = h.new_zeros((row_count, h.shape[1]))
        compatibility_sum = h.new_zeros(row_count)
        if col.numel():
            global_row = row + start
            neighbor_sum.index_add_(0, row, h[col])
            delta = audit[global_row] - audit[col]
            compatibility = torch.exp(-(delta * delta).sum(1) / tau)
            compatibility_sum.index_add_(0, row, compatibility)
        degree = torch.from_numpy(graph.degree[start:end].astype(np.float32, copy=False)).to(
            device=h.device
        )
        nonzero = degree > 0
        mean_neighbor = h[start:end].clone()
        mean_compatibility = h.new_ones(row_count)
        if nonzero.any():
            mean_neighbor[nonzero] = neighbor_sum[nonzero] / degree[nonzero].unsqueeze(1)
            mean_compatibility[nonzero] = compatibility_sum[nonzero] / degree[nonzero]
        representation_scores[start:end] = torch.linalg.vector_norm(
            h[start:end] - mean_neighbor, dim=1
        )
        topology_scores[start:end] = 1.0 - mean_compatibility
    return representation_scores, topology_scores


@torch.no_grad()
def _scores_from_embeddings(
    graph: CSRChunkGraph,
    h: torch.Tensor,
    audit: torch.Tensor,
    tau: float,
    alpha: float,
    edge_chunk: int,
) -> torch.Tensor:
    """Exact raw dual-view score from Eq. (9), excluding the self-message."""
    representation, topology = _score_components_from_embeddings(graph, h, audit, tau, edge_chunk)
    return _combine_score_components(representation, topology, alpha)


def _build_graph(name: str, target: bool, hp: Dict) -> PreparedGraph:
    if target:
        adj, feat, labels, mark = load_target_marked(name)
    else:
        adj, feat, labels, mark = load_source_marked(name)
    adjacency, local, pooled = _load_or_build_structural_inputs(name, adj, hp)
    attributes = _aligned_attributes(feat, hp)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    mark = np.asarray(mark, dtype=bool).reshape(-1)
    if not (
        adjacency.shape[0]
        == local.shape[0]
        == pooled.shape[0]
        == attributes.shape[0]
        == labels.size
        == mark.size
    ):
        raise ValueError(f"TPCA-GAD graph arrays disagree for {name}")
    print(
        f"    [tpcagad/graph] {name}: N={adjacency.shape[0]} "
        f"E={adjacency.nnz} F={attributes.shape[1]}",
        flush=True,
    )
    return PreparedGraph(
        name=name,
        adjacency=adjacency,
        chunks=CSRChunkGraph(adjacency),
        local=local,
        pooled=pooled,
        attributes=attributes,
        labels=labels,
        mark=mark,
    )


def _device_node_tensors(graph: PreparedGraph, device):
    return (
        torch.from_numpy(graph.local).to(device=device),
        torch.from_numpy(graph.pooled).to(device=device),
        torch.from_numpy(graph.attributes).to(device=device),
    )


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validation_metrics(
    model: TPCAGAD,
    source_tensors,
    source_splits,
) -> Tuple[float, float]:
    model.eval()
    aurocs = []
    bces = []
    with torch.no_grad():
        for (graph, local, pooled, attrs, labels), split in zip(source_tensors, source_splits):
            h, _, _ = model(local, pooled, attrs, graph.chunks, checkpoint_chunks=False)
            logits = model.classifier(h).squeeze(1)
            val_idx = split[1]
            val_tensor = torch.from_numpy(val_idx).to(device=logits.device)
            val_labels = labels[val_tensor]
            if torch.unique(val_labels).numel() < 2:
                raise ValueError(
                    f"TPCA-GAD source validation for {graph.name} requires both classes"
                )
            val_logits = logits[val_tensor]
            if not torch.isfinite(val_logits).all():
                raise FloatingPointError(f"TPCA-GAD non-finite validation logits on {graph.name}")
            probabilities = torch.sigmoid(val_logits).cpu().numpy()
            aurocs.append(float(roc_auc_score(val_labels.cpu().numpy(), probabilities)))
            bces.append(
                float(
                    F.binary_cross_entropy_with_logits(
                        logits[val_tensor], val_labels.float()
                    ).item()
                )
            )
    if not aurocs or not np.isfinite(aurocs).all() or not np.isfinite(bces).all():
        raise ValueError("TPCA-GAD source validation requires finite AUROC and loss")
    return float(np.mean(aurocs)), float(np.mean(bces))


def _train_seed(
    source_graphs: Sequence[PreparedGraph],
    seed: int,
    epochs: int,
    device,
    hp: Dict,
) -> TPCAGAD:
    set_seed(seed)
    model = TPCAGAD(hp).to(device)
    if epochs <= 0:
        return model.cpu()

    if not source_graphs:
        raise ValueError("TPCA-GAD training requires at least one source graph")
    source_tensors = []
    source_splits = []
    for graph in source_graphs:
        marked_labels = graph.labels[graph.mark]
        if not np.isin(marked_labels, (0, 1)).all():
            raise ValueError(f"TPCA-GAD source labels for {graph.name} must be binary")
        if np.unique(marked_labels).size != 2:
            raise ValueError(f"TPCA-GAD source training for {graph.name} requires both classes")
        local, pooled, attrs = _device_node_tensors(graph, device)
        labels = torch.from_numpy(graph.labels).to(device=device)
        source_tensors.append((graph, local, pooled, attrs, labels))
        split = _stratified_source_split(
            graph.labels,
            seed,
            hp["source_split"],
            eligible=graph.mark,
        )
        for split_name, indices in zip(("training", "validation"), split[:2]):
            if np.unique(graph.labels[indices]).size != 2:
                raise ValueError(
                    f"TPCA-GAD source {split_name} for {graph.name} requires both classes"
                )
        source_splits.append(split)

    optimizer = Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    best_auc = float("-inf")
    best_bce = float("inf")
    best_state = _clone_state_dict(model)
    report_every = max(1, epochs // 10)
    validation_interval = max(1, int(hp["validation_interval"]))

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for (graph, local, pooled, attrs, labels), split in zip(source_tensors, source_splits):
            train_idx = torch.from_numpy(split[0]).to(device=device)
            h, _, consistency = model(local, pooled, attrs, graph.chunks, checkpoint_chunks=True)
            logits = model.classifier(h).squeeze(1)
            task_loss = F.binary_cross_entropy_with_logits(
                logits[train_idx], labels[train_idx].float()
            )
            loss = task_loss + float(hp["consistency_lambda"]) * consistency
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"TPCA-GAD non-finite loss: seed={seed} "
                    f"epoch={epoch + 1} graph={graph.name} lr={hp['lr']}"
                )
            (loss / len(source_tensors)).backward()
            epoch_losses.append(float(loss.detach().cpu()))
        bad_gradient = next(
            (
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ),
            None,
        )
        if bad_gradient is not None:
            raise FloatingPointError(
                f"TPCA-GAD non-finite gradient: seed={seed} "
                f"epoch={epoch + 1} parameter={bad_gradient} lr={hp['lr']}"
            )
        optimizer.step()

        validate_now = epoch == 0 or epoch + 1 == epochs or (epoch + 1) % validation_interval == 0
        if validate_now:
            val_auc, val_bce = _validation_metrics(model, source_tensors, source_splits)
            if not np.isfinite(val_auc) or not np.isfinite(val_bce):
                raise ValueError("TPCA-GAD source validation requires finite AUROC and loss")
            if val_auc > best_auc or (np.isclose(val_auc, best_auc) and val_bce < best_bce):
                best_auc = val_auc
                best_bce = val_bce
                best_state = _clone_state_dict(model)
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % report_every == 0:
            val_text = f"{val_auc:.6f}" if validate_now else "skipped"
            print(
                f"    [tpcagad/train] seed={seed} epoch={epoch + 1}/{epochs} "
                f"loss={np.mean(epoch_losses):.6f} val_auroc={val_text}",
                flush=True,
            )

    model.load_state_dict(best_state)
    print(
        f"    [tpcagad/select] seed={seed} source-val AUROC={best_auc:.6f}",
        flush=True,
    )
    del source_tensors
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model.cpu()


def _node_ranges(n: int, chunk: int):
    for start in range(0, n, max(1, int(chunk))):
        yield start, min(start + max(1, int(chunk)), n)


@torch.no_grad()
def _streaming_phi(
    model: TPCAGAD, graph: PreparedGraph, start: int, end: int, device
) -> torch.Tensor:
    local = torch.from_numpy(graph.local[start:end]).to(device=device)
    pooled = torch.from_numpy(graph.pooled[start:end]).to(device=device)
    return model.fingerprint(local, pooled)


@torch.no_grad()
def _score_target_components_streaming(
    model: TPCAGAD, graph: PreparedGraph, device, hp: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact Eq. (9) components with node tensors resident on CPU."""
    n = graph.chunks.n
    hidden = int(hp["hidden_dim"])
    audit_dim = int(hp["audit_dim"])
    node_chunk = int(hp["node_chunk"])
    edge_chunk = int(hp["edge_chunk"])
    tau = float(hp["tau"])

    print(f"    [tpcagad/target] {graph.name}: streaming exact inference", flush=True)
    current_h = torch.empty((n, hidden), dtype=torch.float32)
    for start, end in _node_ranges(n, node_chunk):
        phi = _streaming_phi(model, graph, start, end, device)
        attrs = torch.from_numpy(graph.attributes[start:end]).to(device=device)
        current_h[start:end].copy_(model.initial_state(phi, attrs).cpu())

    final_audit = None
    for layer, block in enumerate(model.blocks):
        message_cpu = torch.empty((n, hidden), dtype=torch.float32)
        audit_cpu = torch.empty((n, audit_dim), dtype=torch.float32)
        for start, end in _node_ranges(n, node_chunk):
            h = current_h[start:end].to(device=device)
            phi = _streaming_phi(model, graph, start, end, device)
            message_cpu[start:end].copy_(block.message(h).cpu())
            audit_cpu[start:end].copy_(model.audit_embedding(phi, layer).cpu())
        del current_h

        next_h = torch.empty((n, hidden), dtype=torch.float32)
        for start, end in graph.chunks.row_ranges(edge_chunk, node_chunk):
            row_np, col_np = graph.chunks.edge_arrays(start, end)
            row = torch.from_numpy(row_np).to(device=device)
            col_cpu = torch.from_numpy(np.asarray(col_np, dtype=np.int64))
            row_count = end - start
            message_rows = message_cpu[start:end].to(device=device)
            audit_rows = audit_cpu[start:end].to(device=device)
            neighbor_sum = message_rows.new_zeros((row_count, hidden))
            denominator = message_rows.new_zeros(row_count)
            if col_cpu.numel():
                message_cols = message_cpu[col_cpu].to(device=device)
                audit_cols = audit_cpu[col_cpu].to(device=device)
                log_compatibility = -((audit_rows[row] - audit_cols) ** 2).sum(1) / tau
                row_max = log_compatibility.new_full((row_count,), float("-inf")).scatter_reduce(
                    0,
                    row,
                    log_compatibility,
                    reduce="amax",
                    include_self=True,
                )
                compatibility = torch.exp(log_compatibility - row_max[row])
                neighbor_sum.index_add_(0, row, message_cols * compatibility.unsqueeze(1))
                denominator.index_add_(0, row, compatibility)
            has_neighbor_weight = denominator > 0
            safe_denominator = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
            neighbor_mean = neighbor_sum / safe_denominator.unsqueeze(1)
            aggregated = block.activation(
                torch.where(has_neighbor_weight.unsqueeze(1), neighbor_mean, message_rows)
            )
            phi = _streaming_phi(model, graph, start, end, device)
            structural = model.structural_projection(phi)
            gate = torch.sigmoid(block.gate(torch.cat((aggregated, phi), dim=1)))
            next_h[start:end].copy_(((1.0 - gate) * aggregated + gate * structural).cpu())
        del message_cpu
        current_h = next_h
        if layer + 1 == model.num_layers:
            final_audit = audit_cpu
        else:
            del audit_cpu
        gc.collect()

    representation_scores = np.empty(n, dtype=np.float32)
    topology_scores = np.empty(n, dtype=np.float32)
    for start, end in graph.chunks.row_ranges(edge_chunk, node_chunk):
        row_np, col_np = graph.chunks.edge_arrays(start, end)
        row = torch.from_numpy(row_np).to(device=device)
        col_cpu = torch.from_numpy(np.asarray(col_np, dtype=np.int64))
        row_count = end - start
        h_rows = current_h[start:end].to(device=device)
        audit_rows = final_audit[start:end].to(device=device)
        neighbor_sum = h_rows.new_zeros((row_count, hidden))
        compatibility_sum = h_rows.new_zeros(row_count)
        if col_cpu.numel():
            h_cols = current_h[col_cpu].to(device=device)
            audit_cols = final_audit[col_cpu].to(device=device)
            neighbor_sum.index_add_(0, row, h_cols)
            compatibility = torch.exp(-((audit_rows[row] - audit_cols) ** 2).sum(1) / tau)
            compatibility_sum.index_add_(0, row, compatibility)
        degree = torch.from_numpy(graph.chunks.degree[start:end].astype(np.float32, copy=False)).to(
            device=device
        )
        nonzero = degree > 0
        mean_neighbor = h_rows.clone()
        mean_compatibility = h_rows.new_ones(row_count)
        if nonzero.any():
            mean_neighbor[nonzero] = neighbor_sum[nonzero] / degree[nonzero].unsqueeze(1)
            mean_compatibility[nonzero] = compatibility_sum[nonzero] / degree[nonzero]
        representation_scores[start:end] = (
            torch.linalg.vector_norm(h_rows - mean_neighbor, dim=1).cpu().numpy()
        )
        topology_scores[start:end] = (1.0 - mean_compatibility).cpu().numpy()
    return representation_scores, topology_scores


@torch.no_grad()
def _score_target_streaming(model: TPCAGAD, graph: PreparedGraph, device, hp: Dict) -> np.ndarray:
    representation, topology = _score_target_components_streaming(model, graph, device, hp)
    return _combine_score_components(representation, topology, hp["alpha"])


@torch.no_grad()
def _score_target_components(
    model: TPCAGAD, graph: PreparedGraph, device, hp: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    streaming = graph.chunks.n >= int(hp["stream_node_threshold"]) or graph.chunks.nnz >= int(
        hp["stream_edge_threshold"]
    )
    if streaming:
        return _score_target_components_streaming(model, graph, device, hp)
    local, pooled, attrs = _device_node_tensors(graph, device)
    h, audits, _ = model(local, pooled, attrs, graph.chunks, checkpoint_chunks=False)
    representation, topology = model.anomaly_score_components(h, audits[-1], graph.chunks)
    return representation.cpu().numpy(), topology.cpu().numpy()


@torch.no_grad()
def _score_target(model: TPCAGAD, graph: PreparedGraph, device, hp: Dict) -> np.ndarray:
    representation, topology = _score_target_components(model, graph, device, hp)
    return _combine_score_components(representation, topology, hp["alpha"])


def _resolved_hp(hp: Optional[Dict]) -> Dict:
    cfg = dict(C.TPCAGAD_HP)
    if hp:
        cfg.update(hp)
    for key in (
        "attribute_dim",
        "hidden_dim",
        "audit_dim",
        "num_layers",
        "mlp_depth",
        "fingerprint_hops",
        "edge_chunk",
        "node_chunk",
        "stream_node_threshold",
        "stream_edge_threshold",
        "validation_interval",
        "python_structure_max_edges",
    ):
        try:
            value = float(cfg[key])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"TPCA-GAD {key} must be a positive integer") from error
        if (
            isinstance(cfg[key], (bool, np.bool_))
            or not np.isfinite(value)
            or value <= 0
            or not value.is_integer()
        ):
            raise ValueError(f"TPCA-GAD {key} must be a positive integer")
        cfg[key] = int(value)
    if cfg["mlp_depth"] not in (1, 2):
        raise ValueError("TPCA-GAD reproduction supports MLP depth 1 or 2")
    fractions = np.asarray(cfg["source_split"], dtype=np.float64)
    if (
        fractions.shape != (3,)
        or not np.isfinite(fractions).all()
        or (fractions <= 0).any()
        or not np.isclose(fractions.sum(), 1.0)
    ):
        raise ValueError("source_split must contain three positive finite fractions summing to one")
    for key in ("degree_normalization", "core_normalization"):
        if cfg[key] != "graph_max":
            raise ValueError(f"TPCA-GAD requires {key}='graph_max'")
    if int(cfg["fingerprint_hops"]) != 1:
        raise ValueError("TPCA-GAD paper configuration requires fingerprint_hops=1")
    if cfg["pool"] != "mean_exclude_center":
        raise ValueError("TPCA-GAD reproduction requires pool='mean_exclude_center'")
    if cfg["compatibility_mode"] not in {"layer_specific", "shared_static"}:
        raise ValueError("TPCA-GAD compatibility_mode must be 'layer_specific' or 'shared_static'")
    if cfg["activation"] != "PReLU":
        raise ValueError("TPCA-GAD paper configuration requires activation='PReLU'")
    if cfg["consistency_reduction"] != "mean_edges_layers":
        raise ValueError("TPCA-GAD reproduction requires consistency_reduction='mean_edges_layers'")
    if cfg["multi_source_reduction"] != "macro_mean":
        raise ValueError(
            "TPCA-GAD multi-source adaptation requires multi_source_reduction='macro_mean'"
        )
    if int(cfg["validation_interval"]) <= 0:
        raise ValueError("TPCA-GAD validation_interval must be positive")
    if int(cfg["python_structure_max_edges"]) <= 0:
        raise ValueError("TPCA-GAD python_structure_max_edges must be positive")
    if cfg["selection_metric"] != "source_val_macro_auroc":
        raise ValueError("TPCA-GAD reproduction requires selection_metric='source_val_macro_auroc'")
    if cfg["score_mode"] != "raw_eq9":
        raise ValueError("TPCA-GAD reproduction requires score_mode='raw_eq9'")
    if cfg["base_graph"] != "binary_undirected_loop_free":
        raise ValueError("TPCA-GAD reproduction requires base_graph='binary_undirected_loop_free'")
    if float(cfg["isolated_score"]) != 0.0:
        raise ValueError("TPCA-GAD reproduction requires isolated_score=0.0")
    for key in ("hidden_dim", "audit_dim", "num_layers", "edge_chunk", "node_chunk"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"TPCA-GAD {key} must be positive")
    for key in ("tau", "lr"):
        if not np.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"TPCA-GAD {key} must be finite and positive")
    for key in ("consistency_lambda", "weight_decay"):
        if not np.isfinite(float(cfg[key])) or float(cfg[key]) < 0:
            raise ValueError(f"TPCA-GAD {key} must be finite and non-negative")
    _combine_score_components(np.zeros(1), np.zeros(1), float(cfg["alpha"]))
    return cfg


def _prepare_seed_models(
    sources,
    seeds,
    epochs: int,
    device,
    train: bool,
    cfg: Dict,
) -> List[TPCAGAD]:
    source_graphs: List[PreparedGraph] = []
    if train:
        if not sources:
            raise ValueError("TPCA-GAD training requires at least one source graph")
        source_graphs = [_build_graph(name, target=False, hp=cfg) for name in sources]
    else:
        print(
            "    [tpcagad/no-train] zero optimizer steps; source graphs are not loaded",
            flush=True,
        )

    models = []
    for seed in seeds:
        if train:
            model = _train_seed(source_graphs, int(seed), int(epochs), device, cfg)
        else:
            set_seed(int(seed))
            model = TPCAGAD(cfg).cpu()
        models.append(model)

    del source_graphs
    gc.collect()
    return models


def run_tpcagad(
    sources,
    targets,
    seeds,
    epochs,
    device,
    train=True,
    hp=None,
    target_evaluator=None,
):
    """Train source-only TPCA-GAD models and evaluate frozen target scores."""
    cfg = _resolved_hp(hp)
    print(
        f"    [tpcagad/version] {cfg['implementation_version']} "
        f"mode={cfg['compatibility_mode']}",
        flush=True,
    )
    models = _prepare_seed_models(sources, seeds, int(epochs), device, bool(train), cfg)

    per_target = {name: [] for name in targets}
    for target in targets:
        print(f"    [tpcagad/target] {target}: build", flush=True)
        graph = _build_graph(target, target=True, hp=cfg)
        for seed, model in zip(seeds, models):
            scores = _score_target(model, graph, device, cfg)
            metrics = (
                target_evaluator(target, int(seed), graph.labels, scores, graph.mark)
                if target_evaluator is not None
                else evaluate(graph.labels, scores, graph.mark)
            )
            per_target[target].append(metrics)
            print(
                f"    [tpcagad/target] {target} seed={seed} "
                f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f}",
                flush=True,
            )
            model.cpu()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del graph
        gc.collect()
        print(f"    [tpcagad/target] {target}: released", flush=True)
    return aggregate(per_target)


__all__ = [
    "CSRChunkGraph",
    "PreparedGraph",
    "TPCABlock",
    "TPCAGAD",
    "run_tpcagad",
]
