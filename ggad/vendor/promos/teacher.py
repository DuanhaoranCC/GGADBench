"""In-memory GCA teacher used by the ProMoS benchmark runner.

The released ProMoS entry point consumes GCA embeddings generated beforehand.
For the benchmark we train the same kind of teacher on the selected source
graphs at run time, freeze it, and keep its embeddings only in memory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from common.data import EdgeList, exact_edge_spmm


def undirected_binary_adj(adj: sp.spmatrix, self_loops: bool = False) -> sp.csr_matrix:
    """Match PyG ``to_undirected`` after ProMoS discards adjacency weights."""
    support = sp.csr_matrix(adj, dtype=np.float32)
    support = support.maximum(support.T).tocsr()
    support.sum_duplicates()
    support.data[:] = 1.0
    support = support - sp.diags(support.diagonal(), format="csr")
    support.eliminate_zeros()
    if self_loops:
        support = support + sp.eye(support.shape[0], dtype=np.float32, format="csr")
    return support


def pyg_edge_index(adj: sp.spmatrix, device: str) -> torch.Tensor:
    support = undirected_binary_adj(adj, self_loops=False).tocoo()
    index = np.vstack((support.row, support.col)).astype(np.int64, copy=False)
    return torch.from_numpy(index).to(device)


def normalized_gcn_edges(adj: sp.spmatrix) -> EdgeList:
    """CPU edge list for exact chunked inference equivalent to GCNConv."""
    support = undirected_binary_adj(adj, self_loops=True).tocoo()
    degree = (
        np.asarray(
            sp.csr_matrix((support.data, (support.row, support.col)), shape=support.shape).sum(
                axis=1
            )
        )
        .reshape(-1)
        .astype(np.float32)
    )
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    values = support.data * inv_sqrt[support.row] * inv_sqrt[support.col]
    norm = sp.coo_matrix(
        (values.astype(np.float32, copy=False), (support.row, support.col)),
        shape=support.shape,
    )
    return EdgeList(norm)


def normalized_dropped_edges(edge_index: torch.Tensor, n: int) -> EdgeList:
    """Normalize one directed augmented view exactly as default ``GCNConv``."""
    if edge_index.device.type != "cpu":
        raise ValueError("chunked GCA edge augmentation must remain CPU-resident")
    source = edge_index[0].numpy()
    target = edge_index[1].numpy()
    loops = np.arange(n, dtype=np.int64)
    source = np.concatenate((source, loops))
    target = np.concatenate((target, loops))
    degree = np.bincount(target, minlength=n).astype(np.float32, copy=False)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    values = inv_sqrt[source] * inv_sqrt[target]
    # EdgeList uses row as the aggregation target and col as the message source.
    norm = sp.coo_matrix((values, (target, source)), shape=(n, n))
    return EdgeList(norm)


class GCAEncoder(nn.Module):
    """Two-layer GCA encoder from the official PyG-SSL implementation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GCNConv(in_dim, 2 * out_dim),
                GCNConv(2 * out_dim, out_dim),
            ]
        )
        # PyG-SSL passes one PReLU instance to all encoder layers.
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x

    def forward_chunked(self, x: torch.Tensor, edges: EdgeList, edge_chunk: int) -> torch.Tensor:
        """Run the frozen encoder without materializing all target edges on GPU."""
        for conv in self.convs:
            x = F.linear(x, conv.lin.weight)
            x = exact_edge_spmm(edges, x, chunk=edge_chunk)
            if conv.bias is not None:
                x = x + conv.bias
            x = self.activation(x)
        return x


class RuntimeGCA(nn.Module):
    def __init__(self, dim: int, projection_dim: int, tau: float):
        super().__init__()
        self.encoder = GCAEncoder(dim, dim)
        self.fc1 = nn.Linear(dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, dim)
        self.tau = float(tau)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    @staticmethod
    def _similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T

    def _semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        n = z1.shape[0]
        losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            refl = torch.exp(self._similarity(z1[start:end], z1) / self.tau)
            between = torch.exp(self._similarity(z1[start:end], z2) / self.tau)
            local = torch.arange(end - start, device=z1.device)
            positive = between[local, torch.arange(start, end, device=z1.device)]
            self_sim = refl[local, torch.arange(start, end, device=z1.device)]
            losses.append(-torch.log(positive / (refl.sum(1) + between.sum(1) - self_sim)))
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        h1, h2 = self.projection(z1), self.projection(z2)
        return 0.5 * (
            self._semi_loss(h1, h2, batch_size).mean() + self._semi_loss(h2, h1, batch_size).mean()
        )

    def backward_loss_exact(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> float:
        """Backpropagate exact all-node InfoNCE one query block at a time.

        Every query still uses every node in both denominator pools.  Scaling
        each block sum by ``0.5 / n`` gives exactly the same mean objective as
        :meth:`loss`, while releasing each O(batch_size * n) similarity matrix
        before constructing the next one.
        """
        h1, h2 = self.projection(z1), self.projection(z2)
        normalized = (F.normalize(h1, dim=1), F.normalize(h2, dim=1))
        n = int(h1.shape[0])
        if n == 0:
            raise ValueError("GCA contrastive loss requires at least one node")

        detached_sum = 0.0
        scale = 0.5 / n
        for direction, (anchor, other) in enumerate(
            ((normalized[0], normalized[1]), (normalized[1], normalized[0]))
        ):
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                query = anchor[start:end]
                refl = torch.exp((query @ anchor.T) / self.tau)
                between = torch.exp((query @ other.T) / self.tau)
                local = torch.arange(end - start, device=z1.device)
                diagonal = torch.arange(start, end, device=z1.device)
                positive = between[local, diagonal]
                self_sim = refl[local, diagonal]
                chunk_sum = -torch.log(positive / (refl.sum(1) + between.sum(1) - self_sim)).sum()
                detached_sum += float(chunk_sum.detach())
                final_chunk = direction == 1 and end == n
                (chunk_sum * scale).backward(retain_graph=not final_chunk)
                del query, refl, between, positive, self_sim, chunk_sum
        return detached_sum * scale


def _degree_weights(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    degree = torch.bincount(edge_index[1], minlength=n).to(torch.float32)
    edge_log_degree = degree[edge_index[1]].clamp_min(1.0).log()
    denom = edge_log_degree.max() - edge_log_degree.mean()
    if not torch.isfinite(denom) or float(denom.abs()) < 1e-12:
        edge_weights = torch.ones_like(edge_log_degree)
    else:
        edge_weights = (edge_log_degree.max() - edge_log_degree) / denom
    return edge_weights, degree


def _feature_weights_dense(x: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    weighted = (x.abs().T @ degree).clamp_min(1e-12).log()
    denom = weighted.max() - weighted.mean()
    if not torch.isfinite(denom) or float(denom.abs()) < 1e-12:
        return torch.ones_like(weighted)
    return (weighted.max() - weighted) / denom


def _drop_edge_weighted(
    edge_index: torch.Tensor, weights: torch.Tensor, probability: float
) -> torch.Tensor:
    if probability <= 0:
        return edge_index
    scaled = weights / weights.mean().clamp_min(1e-12) * float(probability)
    scaled = scaled.clamp(max=0.7)
    keep = torch.bernoulli(1.0 - scaled).to(torch.bool)
    return edge_index[:, keep]


def _drop_feature_weighted(
    x: torch.Tensor, weights: torch.Tensor, probability: float
) -> torch.Tensor:
    if probability <= 0:
        return x
    scaled = weights / weights.mean().clamp_min(1e-12) * float(probability)
    scaled = scaled.clamp(max=0.7)
    drop = torch.bernoulli(scaled).to(torch.bool)
    out = x.clone()
    out[:, drop] = 0.0
    return out


def _contrastive_nodes(
    z1: torch.Tensor, z2: torch.Tensor, max_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_nodes <= 0 or z1.shape[0] <= max_nodes:
        return z1, z2
    nodes = torch.randperm(z1.shape[0], device=z1.device)[:max_nodes]
    return z1[nodes], z2[nodes]


def train_runtime_gca(graphs: Sequence, hp: dict, device: str, epochs: int) -> RuntimeGCA:
    """Train one source-only GCA teacher. No checkpoint or embedding is saved."""
    dim = int(hp["dim"])
    model = RuntimeGCA(dim, int(hp["gca_projection_dim"]), float(hp["gca_tau"])).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(hp["gca_lr"]),
        weight_decay=float(hp["gca_weight_decay"]),
    )
    batch_size = int(hp["gca_contrast_batch"])
    max_nodes = int(hp.get("gca_loss_max_nodes", 0))

    prepared = []
    for graph in graphs:
        edge_index = graph.teacher_edge_index
        edge_weights, degree = _degree_weights(edge_index, graph.n)
        feature_weights = _feature_weights_dense(graph.x, degree.to(graph.x.device))
        prepared.append((graph, edge_weights, feature_weights))

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        for graph, edge_weights, feature_weights in prepared:
            optimizer.zero_grad(set_to_none=True)
            edge1 = _drop_edge_weighted(
                edge_index=graph.teacher_edge_index,
                weights=edge_weights,
                probability=float(hp["gca_edge_drop_1"]),
            )
            edge2 = _drop_edge_weighted(
                edge_index=graph.teacher_edge_index,
                weights=edge_weights,
                probability=float(hp["gca_edge_drop_2"]),
            )
            x1 = _drop_feature_weighted(graph.x, feature_weights, float(hp["gca_feature_drop_1"]))
            x2 = _drop_feature_weighted(graph.x, feature_weights, float(hp["gca_feature_drop_2"]))
            if graph.teacher_stream:
                norm1 = normalized_dropped_edges(edge1, graph.n)
                norm2 = normalized_dropped_edges(edge2, graph.n)
                z1 = model.encoder.forward_chunked(x1, norm1, int(hp["edge_chunk"]))
                z2 = model.encoder.forward_chunked(x2, norm2, int(hp["edge_chunk"]))
            else:
                z1, z2 = model(x1, edge1), model(x2, edge2)
            z1_loss, z2_loss = _contrastive_nodes(z1, z2, max_nodes)
            loss_value = model.backward_loss_exact(z1_loss, z2_loss, batch_size=batch_size)
            optimizer.step()
            total += loss_value
        if epoch == 0 or epoch + 1 == int(epochs) or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            suffix = "" if max_nodes <= 0 else f" sampled-negatives<={max_nodes}"
            print(
                f"    [promos/gca] epoch={epoch + 1:03d}/{int(epochs)} "
                f"loss={total / max(1, len(prepared)):.5f}{suffix}",
                flush=True,
            )

    model.eval()
    model.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def embed_graph(model: RuntimeGCA, graph, hp: dict, device: str) -> torch.Tensor:
    model.eval()
    threshold = int(hp.get("gca_stream_edge_threshold", 5_000_000))
    if graph.name in hp.get("gca_big_names", ()) or graph.adj.nnz > threshold:
        edges = normalized_gcn_edges(graph.adj)
        return model.encoder.forward_chunked(graph.x, edges, int(hp["edge_chunk"])).detach()
    return model(graph.x, graph.teacher_edge_index).detach()
