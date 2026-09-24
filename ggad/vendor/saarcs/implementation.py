"""Paper-based SAARCS implementation for source-to-target anomaly detection.

The implementation follows Eqs. (1)--(12) from:
"Multi-dimensional Adaptive Mix-hop Contextual Learning Framework for
Universal Graph Anomaly Detection" (AAAI 2026).

Implementation choices are configured in ``ggad.config.SAARCS_HP``.
Hop attention is node-wise, differing from the printed quadratic Eq. (6).
The encoder uses projected features, reordered using a separate min-max copy
for Eqs. (2)--(4); ``alignment_output`` can select the standardized values.
"""

from __future__ import annotations

from evaluation import format_metrics

import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from torch import nn
from torch.optim import Adam

import ggad.config as C
from common.data import (
    CACHE_ROOT,
    NO_SELFLOOP,
    ROWNORM,
    aggregate,
    load_source_marked,
    load_target_marked,
    x_svd,
)
from util import evaluate, set_seed

CACHE = CACHE_ROOT / "saarcs"
PROJECTION_CACHE = CACHE_ROOT / "saarcs_projection"


def _binary_undirected_loop_free(adj) -> sp.csr_matrix:
    """Return the binary, undirected, loop-free graph used by Eqs. (2)--(5)."""
    graph = sp.coo_matrix(adj, dtype=np.float32).copy()
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


def _binary_input_graph(adj) -> sp.csr_matrix:
    """Binarize the supplied adjacency without inventing reverse edges.

    Preserve input direction, including asymmetric Questions/Tolokers edges,
    following ARC's input convention.
    """
    graph = sp.csr_matrix(adj, dtype=np.float32).copy()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data.fill(1.0)
    graph.sort_indices()
    return graph


def _binary_input_loop_free(adj) -> sp.csr_matrix:
    """Preserve supplied edge directions while excluding non-relational loops."""
    graph = _binary_input_graph(adj)
    diagonal = graph.diagonal()
    if np.any(diagonal):
        graph = (graph - sp.diags(diagonal, format="csr")).tocsr()
        graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _symmetric_normalized(adj: sp.csr_matrix, add_self_loops: bool) -> sp.csr_matrix:
    """Construct D^-1/2 A^T D^-1/2 with an explicit loop policy.

    The benchmark edge convention stores a relation as ``A[source, target]``.
    Transposing therefore aggregates source messages into their destination,
    matching the released ARC predecessor's ``normalize_adj`` implementation.
    The distinction is invisible on symmetric graphs but material on directed
    Questions/Tolokers inputs.
    """
    graph = sp.csr_matrix(adj, dtype=np.float32).copy()
    if add_self_loops:
        # Some files already contain diagonal entries. Blindly adding I would
        # turn them into weight two, contradicting the paper's binary A.
        graph = graph.maximum(sp.eye(graph.shape[0], dtype=np.float32, format="csr")).tocsr()
    graph.sum_duplicates()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32)
    inverse = np.zeros_like(degree)
    nonzero = degree > 0
    inverse[nonzero] = degree[nonzero] ** -0.5
    scale = sp.diags(inverse, dtype=np.float32, format="csr")
    normalized = (scale @ graph.T @ scale).tocsr().astype(np.float32)
    normalized.sum_duplicates()
    normalized.sort_indices()
    return normalized


def _symmetric_normalized_with_self_loops(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Compatibility wrapper for the conventional A + I normalization."""
    return _symmetric_normalized(adj, add_self_loops=True)


class CSRChunkGraph:
    """CPU-resident CSR graph exposing row-complete exact edge chunks."""

    def __init__(self, adj: sp.csr_matrix):
        graph = sp.csr_matrix(adj)
        graph.sort_indices()
        self.indptr = np.asarray(graph.indptr, dtype=np.int64)
        self.indices = np.asarray(graph.indices, dtype=np.int64)
        self.n = int(graph.shape[0])
        self.nnz = int(graph.nnz)
        self.shape = graph.shape
        self.degree = np.diff(self.indptr).astype(np.int64, copy=False)

    def row_ranges(self, max_edges: int) -> Iterable[Tuple[int, int]]:
        max_edges = max(1, int(max_edges))
        start = 0
        while start < self.n:
            edge_limit = int(self.indptr[start]) + max_edges
            edge_end = int(np.searchsorted(self.indptr, edge_limit, side="right") - 1)
            end = min(self.n, max(start + 1, edge_end))
            yield start, end
            start = end

    def edge_arrays(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        edge_start = int(self.indptr[start])
        edge_end = int(self.indptr[end])
        counts = self.degree[start:end]
        rows = np.repeat(np.arange(end - start, dtype=np.int64), counts)
        cols = self.indices[edge_start:edge_end]
        return rows, cols


@dataclass
class PreparedGraph:
    name: str
    adjacency: sp.csr_matrix
    normalized_adjacency: sp.csr_matrix
    chunks: CSRChunkGraph
    features: np.ndarray
    labels: np.ndarray
    mark: np.ndarray


def _projection_fingerprint(feat, dim: int, projection: str) -> str:
    """Content key for SAARCS-specific sparse/random projection caches."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{projection}:{int(dim)}:{tuple(feat.shape)}".encode("utf-8"))
    if sp.issparse(feat):
        values = sp.csr_matrix(feat)
        values.sum_duplicates()
        for array in (values.indptr, values.indices, values.data):
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        digest.update(np.ascontiguousarray(feat).tobytes())
    return digest.hexdigest()


def _graph_fingerprint(graph: CSRChunkGraph) -> str:
    """Content fingerprint for the binary adjacency used by alignment."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{graph.shape}:{graph.nnz}".encode("utf-8"))
    for array in (graph.indptr, graph.indices):
        values = np.ascontiguousarray(array)
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _project_features(
    feat, dim: int, cache: bool, projection: str = "svd_random_adapter"
) -> np.ndarray:
    """Apply the paper's target-label-free dataset-specific adapter.

    A deterministic Gaussian adapter is used when the raw width is smaller
    than the shared width, allowing every projected column to carry information.
    High-dimensional sparse inputs use randomized truncated SVD without ever
    materializing an N x d dense raw matrix. Dense reduction keeps the exact
    cached SVD used by the rest of the benchmark.
    """
    if projection not in {"svd", "svd_random_adapter"}:
        raise ValueError(f"unsupported SAARCS projection={projection!r}")

    raw_dim = int(feat.shape[1])
    cache_path = None
    if cache and (raw_dim < dim or sp.issparse(feat)):
        PROJECTION_CACHE.mkdir(parents=True, exist_ok=True)
        fingerprint = _projection_fingerprint(feat, dim, projection)
        cache_path = PROJECTION_CACHE / f"{projection}_{fingerprint}.npy"
        if cache_path.exists():
            projected = np.asarray(np.load(cache_path), dtype=np.float32)
        else:
            projected = None
    else:
        projected = None

    if projected is None and raw_dim < dim and projection == "svd_random_adapter":
        rng = np.random.RandomState(0)
        adapter = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(raw_dim, dim)).astype(np.float32)
        projected = np.asarray(feat @ adapter, dtype=np.float32)
    elif projected is None and sp.issparse(feat) and raw_dim >= dim:
        reducer = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=7, random_state=0)
        projected = np.asarray(reducer.fit_transform(feat), dtype=np.float32)
    elif projected is None:
        projected = np.asarray(x_svd(feat, dim, cache=cache), dtype=np.float32)

    if cache_path is not None and not cache_path.exists():
        np.save(cache_path, projected)
    if projected.ndim == 1:
        projected = projected[:, None]
    if projected.shape[1] < dim:
        padded = np.zeros((projected.shape[0], dim), dtype=np.float32)
        padded[:, : projected.shape[1]] = projected
        projected = padded
    elif projected.shape[1] > dim:
        projected = projected[:, :dim]
    return np.ascontiguousarray(projected, dtype=np.float32)


def _row_normalize_features(feat):
    """Apply the released ARC input convention before dataset projection."""
    row_sum = np.asarray(feat.sum(axis=1)).reshape(-1).astype(np.float64)
    inverse = np.zeros_like(row_sum)
    nonzero = row_sum != 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    if sp.issparse(feat):
        return (sp.diags(inverse, format="csr") @ sp.csr_matrix(feat)).tocsr()
    return np.asarray(feat, dtype=np.float64) * inverse[:, None]


def _canonicalize_projection_signs(projected: np.ndarray) -> np.ndarray:
    """Remove the arbitrary per-column sign of an independently fitted SVD."""
    projected = np.asarray(projected, dtype=np.float32)
    if projected.shape[0] == 0 or projected.shape[1] == 0:
        return np.ascontiguousarray(projected, dtype=np.float32)
    pivots = np.argmax(np.abs(projected), axis=0)
    signs = np.sign(projected[pivots, np.arange(projected.shape[1])])
    signs[signs == 0] = 1.0
    return np.ascontiguousarray(projected * signs[None, :], dtype=np.float32)


def _minmax_standardize(features: np.ndarray) -> np.ndarray:
    """Column-wise min-max scaling that preserves Eq. (3)'s dispersion signal."""
    minimum = features.min(axis=0, keepdims=True)
    maximum = features.max(axis=0, keepdims=True)
    width = maximum - minimum
    standardized = np.zeros_like(features, dtype=np.float32)
    nonconstant = width.reshape(-1) > 0
    if nonconstant.any():
        standardized[:, nonconstant] = (features[:, nonconstant] - minimum[:, nonconstant]) / width[
            :, nonconstant
        ]
    return np.ascontiguousarray(standardized, dtype=np.float32)


def _composite_spatial_sort(
    projected: np.ndarray,
    graph: CSRChunkGraph,
    alpha: float,
    edge_chunk: int,
    alignment_output: str = "projected",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement Eqs. (2)--(4) and return sorted features, scores, order."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SAARCS spatial_alpha must be in [0,1], got {alpha}")
    standardized = _minmax_standardize(projected)
    local_sum = np.zeros(standardized.shape[1], dtype=np.float64)
    for start, end in graph.row_ranges(edge_chunk):
        row, col = graph.edge_arrays(start, end)
        if col.size:
            global_row = row + start
            difference = standardized[global_row] - standardized[col]
            local_sum += np.sum(difference * difference, axis=0, dtype=np.float64)
    local = local_sum / max(graph.nnz, 1)
    global_variance = np.var(standardized, axis=0, dtype=np.float64)
    composite = alpha * local + (1.0 - alpha) * global_variance
    order = np.argsort(-composite, kind="mergesort").astype(np.int64)
    if alignment_output == "projected":
        encoder_input = projected
    elif alignment_output == "standardized":
        encoder_input = standardized
    else:
        raise ValueError(
            "SAARCS alignment_output must be 'projected' or 'standardized', "
            f"got {alignment_output!r}"
        )
    aligned = np.ascontiguousarray(encoder_input[:, order], dtype=np.float32)
    return aligned, composite.astype(np.float32), order


def _alignment_cache_path(name: str, feat, graph: CSRChunkGraph, hp: Dict) -> Path:
    identity = {
        "version": hp["cache_version"],
        "name": str(name),
        "nodes": graph.n,
        "edges": graph.nnz,
        "raw_shape": tuple(int(value) for value in feat.shape),
        "raw_feature_content": _projection_fingerprint(
            feat, int(hp["feature_dim"]), hp["projection"]
        ),
        "graph_content": _graph_fingerprint(graph),
        "feature_dim": int(hp["feature_dim"]),
        "spatial_alpha": float(hp["spatial_alpha"]),
        "standardization": hp["standardization"],
        "projection": hp["projection"],
        "projection_sign": hp["projection_sign"],
        "alignment_output": hp["alignment_output"],
        "base_graph": hp["base_graph"],
        "raw_feature_norm": hp["raw_feature_norm"],
    }
    digest = hashlib.blake2b(
        repr(sorted(identity.items())).encode("utf-8"), digest_size=10
    ).hexdigest()
    return CACHE / f"{name}_{digest}.npy"


def _aligned_features(name: str, feat, adjacency: sp.csr_matrix, hp: Dict) -> np.ndarray:
    chunks = CSRChunkGraph(adjacency)
    cache_path = None
    if bool(hp.get("cache", True)):
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path = _alignment_cache_path(name, feat, chunks, hp)
        if cache_path.exists():
            aligned = np.load(cache_path)
            expected = (adjacency.shape[0], int(hp["feature_dim"]))
            if aligned.shape == expected:
                print(f"    [saarcs/alignment] {name}: cache hit", flush=True)
                return np.ascontiguousarray(aligned, dtype=np.float32)

    if hp["projection"] not in {"svd", "svd_random_adapter"}:
        raise ValueError("SAARCS projection must be 'svd' or 'svd_random_adapter'")
    if hp["standardization"] != "minmax":
        raise ValueError("SAARCS reproduction currently supports standardization='minmax'")
    if hp["raw_feature_norm"] == "arc_conditional":
        projection_input = _row_normalize_features(feat) if name in ROWNORM else feat
    elif hp["raw_feature_norm"] == "none":
        projection_input = feat
    else:
        raise ValueError(
            "SAARCS raw_feature_norm must be 'arc_conditional' or 'none', "
            f"got {hp['raw_feature_norm']!r}"
        )
    projected = _project_features(
        projection_input,
        int(hp["feature_dim"]),
        bool(hp.get("cache", True)),
        hp["projection"],
    )
    random_expansion = hp["projection"] == "svd_random_adapter" and int(
        projection_input.shape[1]
    ) < int(hp["feature_dim"])
    if hp["projection_sign"] == "max_abs_positive":
        if not random_expansion:
            projected = _canonicalize_projection_signs(projected)
    elif hp["projection_sign"] != "none":
        raise ValueError(
            "SAARCS projection_sign must be 'max_abs_positive' or 'none', "
            f"got {hp['projection_sign']!r}"
        )
    aligned, scores, order = _composite_spatial_sort(
        projected,
        chunks,
        float(hp["spatial_alpha"]),
        int(hp["edge_chunk"]),
        hp["alignment_output"],
    )
    print(
        f"    [saarcs/alignment] {name}: top feature ranks="
        f"{order[:min(5, order.size)].tolist()} "
        f"scores={scores[order[:min(5, order.size)]].round(6).tolist()}",
        flush=True,
    )
    if cache_path is not None:
        np.save(cache_path, aligned)
        print(f"    [saarcs/alignment] {name}: cache saved", flush=True)
    return aligned


def _build_graph(name: str, target: bool, hp: Dict) -> PreparedGraph:
    if target:
        adj, feat, labels, mark = load_target_marked(name)
    else:
        adj, feat, labels, mark = load_source_marked(name)
    if hp["base_graph"] == "binary_input":
        adjacency = _binary_input_graph(adj)
    elif hp["base_graph"] == "binary_input_loop_free":
        adjacency = _binary_input_loop_free(adj)
    elif hp["base_graph"] == "binary_undirected_loop_free":
        adjacency = _binary_undirected_loop_free(adj)
    else:
        raise ValueError(f"unsupported SAARCS base_graph={hp['base_graph']!r}")
    features = _aligned_features(name, feat, adjacency, hp)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    mark = np.asarray(mark, dtype=bool).reshape(-1)
    if not (adjacency.shape[0] == features.shape[0] == labels.size == mark.size):
        raise ValueError(f"SAARCS graph arrays disagree for {name}")
    if hp["propagation_self_loops"] == "arc_conditional":
        add_self_loops = name not in NO_SELFLOOP
    elif hp["propagation_self_loops"] == "add":
        add_self_loops = True
    elif hp["propagation_self_loops"] == "none":
        add_self_loops = False
    else:
        raise ValueError(
            "SAARCS propagation_self_loops must be 'arc_conditional', "
            f"'add', or 'none', got {hp['propagation_self_loops']!r}"
        )
    normalized = _symmetric_normalized(adjacency, add_self_loops)
    print(
        f"    [saarcs/graph] {name}: N={adjacency.shape[0]} "
        f"E={adjacency.nnz} F={features.shape[1]}",
        flush=True,
    )
    return PreparedGraph(
        name=name,
        adjacency=adjacency,
        normalized_adjacency=normalized,
        chunks=CSRChunkGraph(adjacency),
        features=features,
        labels=labels,
        mark=mark,
    )


def _propagated_inputs(graph: PreparedGraph, hp: Dict) -> List[np.ndarray]:
    """Compute exact H_0...H_k from Eq. (5) on CPU-resident full graphs."""
    propagated = [np.ascontiguousarray(graph.features, dtype=np.float32)]
    current = propagated[0]
    for hop in range(1, int(hp["num_hops"]) + 1):
        current = np.asarray(graph.normalized_adjacency @ current, dtype=np.float32)
        current = np.ascontiguousarray(current, dtype=np.float32)
        propagated.append(current)
        print(
            f"    [saarcs/propagate] {graph.name}: exact hop {hop}/" f"{int(hp['num_hops'])}",
            flush=True,
        )
    return propagated


class AdaptiveMixHopEncoder(nn.Module):
    """Adaptive node-wise hop attention from Eqs. (5)--(8)."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        embedding_dim: int,
        num_hops: int,
        negative_slope: float,
    ):
        super().__init__()
        self.num_hops = int(num_hops)
        self.attention_dim = int(attention_dim)
        self.embedding_dim = int(embedding_dim)
        if self.attention_dim != self.embedding_dim:
            raise ValueError(
                "SAARCS Eqs. (6)--(8) require W_Z, W_S, and W_T to share " "the same output width"
            )
        self.negative_slope = float(negative_slope)
        self.query = nn.Linear(feature_dim, attention_dim, bias=False)
        self.key = nn.Linear(feature_dim, attention_dim, bias=False)
        self.value = nn.Linear(feature_dim, embedding_dim, bias=False)

    def forward(
        self, base: torch.Tensor, propagated: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(propagated) != self.num_hops:
            raise ValueError(
                f"SAARCS expected {self.num_hops} propagated hops, " f"received {len(propagated)}"
            )
        query = self.query(base)
        logits = []
        for hop_features in propagated:
            residual = hop_features - base
            key = self.key(residual)
            score = (query * key).sum(dim=1) / math.sqrt(self.attention_dim)
            logits.append(F.leaky_relu(score, negative_slope=self.negative_slope))
        weights = torch.softmax(torch.stack(logits, dim=1), dim=1)
        embedding = base.new_zeros((base.shape[0], self.embedding_dim))
        for index, hop_features in enumerate(propagated):
            residual = hop_features - base
            embedding = embedding + weights[:, index : index + 1] * self.value(residual)
        return embedding, weights


class CrossNodeContextReconstructor(nn.Module):
    """Cross-node normal-context reconstruction from Eqs. (9)--(10)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self, query_embedding: torch.Tensor, context_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context_embedding.shape[0] == 0:
            raise ValueError("SAARCS reconstruction requires normal context nodes")
        query = self.query(query_embedding)
        key = self.key(context_embedding)
        attention = torch.softmax(query @ key.T / math.sqrt(self.embedding_dim), dim=1)
        reconstructed = attention @ context_embedding
        return reconstructed, attention


class SAARCS(nn.Module):
    """SAARCS model reconstructed from Eqs. (5)--(12)."""

    def __init__(self, hp: Optional[Dict] = None):
        super().__init__()
        cfg = _resolved_hp(hp)
        self.hp = cfg
        latent_dim = int(cfg["latent_dim"]) if "latent_dim" in cfg else int(cfg["attention_dim"])
        self.encoder = AdaptiveMixHopEncoder(
            int(cfg["feature_dim"]),
            latent_dim,
            latent_dim,
            int(cfg["num_hops"]),
            float(cfg["leaky_relu_slope"]),
        )
        self.reconstructor = CrossNodeContextReconstructor(latent_dim)

    def encode(
        self, base: torch.Tensor, propagated: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(base, propagated)

    def reconstruct(
        self, query_embedding: torch.Tensor, context_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reconstructor(query_embedding, context_embedding)

    def anomaly_scores(
        self, query_embedding: torch.Tensor, context_embedding: torch.Tensor
    ) -> torch.Tensor:
        reconstructed, _ = self.reconstruct(query_embedding, context_embedding)
        return torch.linalg.vector_norm(query_embedding - reconstructed, dim=1)

    def marginal_cosine_loss(
        self,
        query_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        reconstructed, _ = self.reconstruct(query_embedding, context_embedding)
        similarity = F.cosine_similarity(query_embedding, reconstructed, dim=1)
        labels = labels.to(dtype=torch.bool)
        losses = torch.empty_like(similarity)
        losses[~labels] = 1.0 - similarity[~labels]
        losses[labels] = torch.relu(similarity[labels] - float(self.hp["margin"]))
        return losses.mean()


def _encode_indices(
    model: SAARCS,
    propagated: Sequence[np.ndarray],
    indices: np.ndarray,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = np.asarray(indices, dtype=np.int64)
    tensors = [
        torch.from_numpy(np.ascontiguousarray(values[indices], dtype=np.float32)).to(device=device)
        for values in propagated
    ]
    return model.encode(tensors[0], tensors[1:])


def _sample_source_batch(
    graph: PreparedGraph,
    rng: np.random.RandomState,
    num_context: int,
    query_per_class: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    eligible = np.asarray(graph.mark, dtype=bool)
    normal = np.flatnonzero(eligible & (graph.labels == 0))
    anomalous = np.flatnonzero(eligible & (graph.labels == 1))
    if normal.size <= num_context:
        raise ValueError(f"SAARCS source {graph.name} needs more than {num_context} normal nodes")
    if anomalous.size == 0:
        raise ValueError(f"SAARCS source {graph.name} has no labeled anomalies")
    context = rng.choice(normal, size=num_context, replace=False).astype(np.int64)
    available_normal = normal[~np.isin(normal, context)]
    if query_per_class is None:
        count = min(available_normal.size, anomalous.size)
    else:
        count = min(int(query_per_class), available_normal.size, anomalous.size)
    if count <= 0:
        raise ValueError(f"SAARCS source {graph.name} cannot form balanced queries")
    normal_query = rng.choice(available_normal, size=count, replace=False)
    anomaly_query = rng.choice(anomalous, size=count, replace=False)
    query = np.concatenate((normal_query, anomaly_query)).astype(np.int64)
    rng.shuffle(query)
    return context, query


def _resolved_hp(hp: Optional[Dict]) -> Dict:
    cfg = dict(C.SAARCS_HP)
    if hp:
        cfg.update(hp)
    # W_Z, W_S, and W_T share latent_dim. Paired attention_dim/embedding_dim
    # aliases are accepted only when they specify the same width.
    legacy_widths = (
        hp and ("attention_dim" in hp or "embedding_dim" in hp) and "latent_dim" not in hp
    )
    if legacy_widths:
        if "attention_dim" not in hp or "embedding_dim" not in hp:
            raise ValueError(
                "legacy SAARCS width overrides must provide both " "attention_dim and embedding_dim"
            )
        attention_dim = int(hp["attention_dim"])
        embedding_dim = int(hp["embedding_dim"])
        if attention_dim != embedding_dim:
            raise ValueError("legacy SAARCS attention_dim and embedding_dim must be equal")
        cfg["latent_dim"] = attention_dim
    elif "latent_dim" not in cfg:
        attention_dim = int(cfg["attention_dim"])
        embedding_dim = int(cfg["embedding_dim"])
        if attention_dim != embedding_dim:
            raise ValueError("legacy SAARCS attention_dim and embedding_dim must be equal")
        cfg["latent_dim"] = attention_dim
    cfg["attention_dim"] = int(cfg["latent_dim"])
    cfg["embedding_dim"] = int(cfg["latent_dim"])
    for key in (
        "feature_dim",
        "latent_dim",
        "num_hops",
        "num_context",
        "edge_chunk",
        "query_chunk",
    ):
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0:
            raise ValueError(f"SAARCS {key} must be positive")
    if cfg.get("query_per_class") is not None:
        cfg["query_per_class"] = int(cfg["query_per_class"])
    if cfg["feature_dim"] <= 0 or cfg["num_hops"] <= 0:
        raise ValueError("SAARCS feature_dim and num_hops must be positive")
    if cfg["num_context"] <= 0:
        raise ValueError("SAARCS context size must be positive")
    if cfg.get("query_per_class") is not None and cfg["query_per_class"] <= 0:
        raise ValueError("SAARCS query_per_class must be positive or None")
    if cfg.get("hop_attention_mode") != "node_wise":
        raise ValueError(
            "SAARCS hop_attention_mode must be 'node_wise', got "
            f"{cfg.get('hop_attention_mode')!r}"
        )
    return cfg


def _train_seed(
    source_data: Sequence[Tuple[PreparedGraph, Sequence[np.ndarray]]],
    seed: int,
    epochs: int,
    device,
    hp: Dict,
) -> SAARCS:
    set_seed(seed)
    rng = np.random.RandomState(seed)
    model = SAARCS(hp).to(device)
    if epochs <= 0:
        return model.cpu()
    optimizer = Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    report_every = max(1, int(epochs) // 10)
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for graph, propagated in source_data:
            context_idx, query_idx = _sample_source_batch(
                graph,
                rng,
                int(hp["num_context"]),
                hp.get("query_per_class"),
            )
            context_embedding, _ = _encode_indices(model, propagated, context_idx, device)
            query_embedding, _ = _encode_indices(model, propagated, query_idx, device)
            query_labels = torch.from_numpy(
                graph.labels[query_idx].astype(np.int64, copy=False)
            ).to(device=device)
            loss = model.marginal_cosine_loss(query_embedding, context_embedding, query_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"SAARCS non-finite loss: seed={seed} "
                    f"epoch={epoch + 1} graph={graph.name} lr={hp['lr']}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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
                    f"SAARCS non-finite gradient: seed={seed} "
                    f"epoch={epoch + 1} graph={graph.name} "
                    f"parameter={bad_gradient} lr={hp['lr']}"
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch + 1 == int(epochs) or (epoch + 1) % report_every == 0:
            print(
                f"    [saarcs/train] seed={seed} epoch={epoch + 1}/{epochs} "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )
    del optimizer
    model.eval()
    model.cpu()
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def _score_target(
    model: SAARCS,
    graph: PreparedGraph,
    propagated: Sequence[np.ndarray],
    seed: int,
    shot: int,
    device,
    hp: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    chunk = int(hp["query_chunk"])
    if chunk <= 0:
        raise ValueError("SAARCS query_chunk must be positive")
    model = model.to(device)
    model.eval()
    rng = np.random.RandomState(seed)
    normal = np.flatnonzero(graph.mark & (graph.labels == 0))
    if normal.size < int(shot):
        raise ValueError(
            f"SAARCS target {graph.name} has {normal.size} marked normal nodes, "
            f"fewer than shot={shot}"
        )
    context_idx = rng.choice(normal, size=int(shot), replace=False).astype(np.int64)
    context_embedding, _ = _encode_indices(model, propagated, context_idx, device)

    query_mask = np.asarray(graph.mark, dtype=bool).copy()
    query_mask[context_idx] = False
    query_idx = np.flatnonzero(query_mask).astype(np.int64)
    scores = np.empty(query_idx.size, dtype=np.float32)
    for start in range(0, query_idx.size, chunk):
        end = min(start + chunk, query_idx.size)
        query_embedding, _ = _encode_indices(model, propagated, query_idx[start:end], device)
        chunk_scores = model.anomaly_scores(query_embedding, context_embedding).cpu().numpy()
        if not np.isfinite(chunk_scores).all():
            bad = int((~np.isfinite(chunk_scores)).sum())
            raise FloatingPointError(
                f"SAARCS target {graph.name} produced {bad} non-finite " "anomaly scores"
            )
        scores[start:end] = chunk_scores
    return query_idx, scores


def run_saarcs(
    sources,
    targets,
    seeds,
    epochs,
    device,
    shot=10,
    hp=None,
):
    """Train SAARCS on source graphs and evaluate normal-context target queries."""
    cfg = _resolved_hp(hp)
    print(
        f"    [saarcs/version] {cfg['implementation_version']} "
        f"hop_attention={cfg['hop_attention_mode']} hops={cfg['num_hops']}",
        flush=True,
    )
    source_data: List[Tuple[PreparedGraph, Sequence[np.ndarray]]] = []
    if not sources:
        raise ValueError("SAARCS training requires at least one source graph")
    for name in sources:
        graph = _build_graph(name, target=False, hp=cfg)
        source_data.append((graph, _propagated_inputs(graph, cfg)))

    models = []
    for seed in seeds:
        model = _train_seed(source_data, int(seed), int(epochs), device, cfg)
        models.append((int(seed), model))

    source_data.clear()
    gc.collect()
    per_target = {name: [] for name in targets}
    for target in targets:
        print(f"    [saarcs/target] {target}: build", flush=True)
        graph = _build_graph(target, target=True, hp=cfg)
        propagated = _propagated_inputs(graph, cfg)
        for seed, model in models:
            query_idx, scores = _score_target(
                model, graph, propagated, seed, int(shot), device, cfg
            )
            metrics = evaluate(graph.labels[query_idx], scores)
            per_target[target].append(metrics)
            print(
                f"    [saarcs/target] {target} seed={seed} context={int(shot)} "
                f"queries={query_idx.size} {format_metrics(metrics)}",
                flush=True,
            )
            model.cpu()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        del propagated, graph
        gc.collect()
        print(f"    [saarcs/target] {target}: released", flush=True)
    return aggregate(per_target)


__all__ = [
    "AdaptiveMixHopEncoder",
    "CrossNodeContextReconstructor",
    "CSRChunkGraph",
    "PreparedGraph",
    "SAARCS",
    "run_saarcs",
]
