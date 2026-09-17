"""Paper-based implementation of MDGPT equations (2)--(7).

The encoder uses a GCN with separate source-domain and target prompts.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from common.data import EdgeList, exact_edge_spmm


def propagate(adjacency, values, edge_chunk=100_000):
    if isinstance(adjacency, EdgeList):
        return exact_edge_spmm(adjacency, values, chunk=int(edge_chunk))
    return torch.sparse.mm(adjacency, values)


class GCN(nn.Module):
    """PReLU(P H W + b), three layers by paper default, without residuals."""

    def __init__(self, feature_dim=8, hidden_dim=256, num_layers=3):
        super().__init__()
        if min(feature_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("GCN dimensions and layer count must be positive")
        dims = [feature_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.activations = nn.ModuleList([nn.PReLU() for _ in range(num_layers)])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, adjacency, edge_chunk=100_000):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(propagate(adjacency, x, edge_chunk)))
        return x


class MDGPT(nn.Module):
    def __init__(self, num_domains, feature_dim=8, hidden_dim=256, num_layers=3):
        super().__init__()
        if num_domains < 1:
            raise ValueError("MDGPT requires at least one source domain")
        self.encoder = GCN(feature_dim, hidden_dim, num_layers)
        self.domain_tokens = nn.Parameter(torch.empty(num_domains, feature_dim))
        nn.init.normal_(self.domain_tokens, mean=1.0, std=0.02)

    def encode_source(self, x, adjacency, domain, edge_chunk=100_000):
        return self.encoder(x * self.domain_tokens[domain], adjacency, edge_chunk)

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        return self


class DualPrompt(nn.Module):
    """Only d+K trainable parameters, with unconstrained mixing coefficients."""

    def __init__(self, domain_tokens):
        super().__init__()
        self.register_buffer("tokens", domain_tokens.detach().clone())
        self.unifying = nn.Parameter(torch.empty_like(self.tokens[0]))
        nn.init.normal_(self.unifying, mean=1.0, std=0.02)
        self.gamma = nn.Parameter(
            self.tokens.new_full((self.tokens.shape[0],), 1.0 / self.tokens.shape[0])
        )

    def mixing(self):
        return self.gamma @ self.tokens

    def forward(self, encoder, x, adjacency, edge_chunk=100_000):
        # Eq.6: adding inputs before GE would change the nonlinear model.
        return encoder(x * self.unifying, adjacency, edge_chunk) + encoder(
            x * self.mixing(), adjacency, edge_chunk
        )


def link_loss(embeddings, anchors, positives, negatives, temperature=0.2):
    """Literal Eq.4: negatives-only denominator; a negative loss is valid."""
    if temperature <= 0 or negatives.ndim != 2 or negatives.shape[1] == 0:
        raise ValueError("temperature and number of negative samples must be positive")
    z = F.normalize(embeddings, p=2, dim=1)
    pos = (z[anchors] * z[positives]).sum(-1) / temperature
    neg = (z[anchors, None, :] * z[negatives]).sum(-1) / temperature
    return (torch.logsumexp(neg, dim=1) - pos).mean()


def prototypes(support_embeddings, support_labels):
    """Unnormalized arithmetic class means, with gradients through both means."""
    means = []
    for cls in (0, 1):
        members = support_embeddings[support_labels == cls]
        if members.shape[0] == 0:
            raise ValueError("Both normal and anomalous support examples are required")
        means.append(members.mean(0))
    return torch.stack(means)


def prototype_logits(embeddings, centers, temperature=0.2):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.normalize(embeddings, dim=1) @ F.normalize(centers, dim=1).T / temperature


def support_loss(embeddings, labels, temperature=0.2):
    return F.cross_entropy(
        prototype_logits(embeddings, prototypes(embeddings, labels), temperature), labels
    )


def sample_triplets(adjacency, count, num_negatives, rng, prepared=False):
    """Uniform positive edges; distinct, verified, within-domain non-neighbors.

    This samples the paper's self-supervised loss instances. Message passing
    still uses the entire source graph. Nodes without a feasible triplet are
    excluded from the loss sampler, never from the graph.
    """
    a = sp.csr_matrix(adjacency, copy=not prepared)
    if not prepared:
        a.setdiag(0)
        a.eliminate_zeros()
        a.sort_indices()
    n = a.shape[0]
    degree = np.diff(a.indptr)
    counts = np.where((degree > 0) & (n - 1 - degree >= num_negatives), degree, 0)
    cumulative = np.cumsum(counts, dtype=np.int64)
    if count <= 0 or num_negatives <= 0 or not len(cumulative) or cumulative[-1] == 0:
        raise ValueError("No feasible positive edges with the requested distinct non-neighbors")
    draws = rng.randint(0, cumulative[-1], size=int(count))
    anchor = np.searchsorted(cumulative, draws, side="right")
    previous = np.where(anchor > 0, cumulative[np.maximum(anchor - 1, 0)], 0)
    positive = a.indices[a.indptr[anchor] + (draws - previous)]
    negative = np.empty((count, num_negatives), dtype=np.int64)
    for column in range(num_negatives):
        candidate = rng.randint(n, size=count)
        for attempt in range(129):
            invalid = (candidate == anchor) | (np.asarray(a[anchor, candidate]).reshape(-1) != 0)
            if column:
                invalid |= (negative[:, :column] == candidate[:, None]).any(1)
            bad = np.flatnonzero(invalid)
            if not bad.size:
                break
            if attempt == 128:
                # Dense graphs: explicit complements guarantee termination.
                for index in bad:
                    forbidden = np.concatenate(
                        (
                            a.indices[a.indptr[anchor[index]] : a.indptr[anchor[index] + 1]],
                            [anchor[index]],
                            negative[index, :column],
                        )
                    )
                    allowed = np.setdiff1d(np.arange(n), forbidden, assume_unique=False)
                    candidate[index] = rng.choice(allowed)
                break
            candidate[bad] = rng.randint(n, size=bad.size)
        negative[:, column] = candidate
    return anchor.astype(np.int64), positive.astype(np.int64), negative


def split_support(labels, mark, shot, seed):
    """k support occurrences/class and every other marked node as query.

    Match the benchmark few-shot fallback for scarce classes (e.g. Disney's
    six anomalies): reserve one query before replacement sampling. Repeated
    occurrences do not increase the number of unique labeled examples.
    """
    labels = np.asarray(labels).reshape(-1)
    mark = np.asarray(mark, dtype=bool).reshape(-1)
    if labels.shape != mark.shape or shot < 1:
        raise ValueError("Invalid labels/mark or non-positive shot")
    if not np.isin(labels[mark], [0, 1]).all():
        raise ValueError("Marked MDGPT labels must be binary")
    rng = np.random.RandomState(int(seed))
    support = []
    for cls in (0, 1):
        pool = np.flatnonzero(mark & (labels == cls))
        if pool.size < 2:
            raise ValueError(
                f"class {cls} needs at least 2 marked nodes for disjoint support/query; got {pool.size}"
            )
        if pool.size <= shot:
            reserved = int(rng.randint(pool.size))
            print(
                f"    [mdgpt/reserve-query] class={cls} available={len(pool)} "
                f"shot={shot}; reserve node={pool[reserved]}, replacement support",
                flush=True,
            )
            pool = np.delete(pool, reserved)
        support.append(rng.choice(pool, size=int(shot), replace=pool.size < shot))
    support = np.concatenate(support).astype(np.int64)
    query_mask = mark.copy()
    query_mask[support] = False
    return support, np.flatnonzero(query_mask)
