"""BRIDGE runner for gfm using SAMGPT's benchmark data protocol.

Reproduced BRIDGE node pipeline:
  * multi-source domain-invariant feature masks;
  * link-prediction pretraining plus variance risk;
  * self-supervised MoE source-prompt assembly and target open prompt;
  * few-shot prototype classification, entropy loss, and spectral regularizer.

Benchmark adaptation is limited to SAMGPT's shared SVD features, marked-node
few-shot split, exact large-graph propagation, and AUROC/AUPRC aggregation.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD

from gfm.runners.ego_neighbor_residual import STREAM_GRAPHS, CompactEdgeList, edge_spmm
from gfm.runners.samgpt import (
    GraphCtx,
    _adj_for,
    _ctx,
    _exact_support_dependency_subgraph,
    _query_nodes,
    _sample_class,
)
from gfm.vendor.bridge import DownPrompt, PrePrompt, spectral_regularization_smooth
from util import aggregate, evaluate, set_seed


@dataclass
class SpectralBasis:
    components: np.ndarray
    values: np.ndarray


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _spmm(edge_chunk: int):
    return lambda adj, x: edge_spmm(adj, x, int(edge_chunk))


def _is_neighbor(sorted_neighbors: np.ndarray, candidate: int) -> bool:
    position = int(np.searchsorted(sorted_neighbors, candidate))
    return position < sorted_neighbors.size and int(sorted_neighbors[position]) == candidate


def _pretrain_tuples(ctxs: Sequence[GraphCtx], negative_count: int, seed: int) -> torch.Tensor:
    """Official one-positive/K-negative tuples without O(N^2) complements.

    Rejection sampling without replacement is uniform over the same block-diagonal
    non-neighbor set produced by the released shuffle(setdiff1d(...)) code.
    """
    rng = np.random.RandomState(seed)
    sizes = [ctx.adj.shape[0] for ctx in ctxs]
    offsets = np.cumsum([0, *sizes[:-1]], dtype=np.int64)
    total_nodes = int(sum(sizes))
    tuples = np.empty((total_nodes, 1 + int(negative_count)), dtype=np.int64)

    for ctx, offset, size in zip(ctxs, offsets, sizes):
        adj = ctx.adj
        if not adj.has_sorted_indices:
            adj = adj.sorted_indices()
        for local_node in range(size):
            global_node = int(offset + local_node)
            row = adj.indices[adj.indptr[local_node] : adj.indptr[local_node + 1]]
            if row.size:
                tuples[global_node, 0] = int(offset + row[rng.randint(row.size)])
            else:
                tuples[global_node, 0] = global_node

            unique_degree = int(np.unique(row).size)
            if total_nodes - unique_degree < negative_count:
                raise ValueError(
                    f"BRIDGE node {global_node} has fewer than {negative_count} non-neighbors"
                )
            chosen = set()
            while len(chosen) < negative_count:
                candidate = int(rng.randint(total_nodes))
                if candidate in chosen:
                    continue
                if int(offset) <= candidate < int(offset + size):
                    if _is_neighbor(row, candidate - int(offset)):
                        continue
                chosen.add(candidate)
            tuples[global_node, 1:] = np.fromiter(chosen, dtype=np.int64, count=negative_count)
    return torch.from_numpy(tuples)


def _spectral_basis(adj: sp.spmatrix, components: int) -> SpectralBasis:
    """Released get_laplacian_evd: TruncatedSVD over the raw Laplacian."""
    adjacency = sp.csr_matrix(adj).astype(np.float32, copy=True)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    # Official setdiag(-degree); negate, expressed without costly CSR mutation.
    laplacian = -adjacency + sp.diags(degree + adjacency.diagonal(), dtype=np.float32, format="csr")
    nodes = int(laplacian.shape[0])
    if nodes <= 1:
        return SpectralBasis(
            components=np.ones((1, nodes), dtype=np.float32),
            values=np.zeros(1, dtype=np.float32),
        )
    rank = min(int(components), nodes - 1)
    svd = TruncatedSVD(n_components=rank, n_iter=20, random_state=42)
    svd.fit(laplacian)
    return SpectralBasis(
        components=np.asarray(svd.components_, dtype=np.float32),
        values=np.sqrt(np.maximum(np.asarray(svd.explained_variance_, dtype=np.float32), 0.0)),
    )


def _pretrain_one_seed(
    ctxs: Sequence[GraphCtx], hp: dict, seed: int, feature_dim: int, device: str
) -> PrePrompt:
    set_seed(seed)
    negative_samples = _pretrain_tuples(ctxs, int(hp["negative_samples"]), seed)
    model = PrePrompt(
        input_dim=feature_dim,
        hidden_dim=int(hp["hid_units"]),
        negative_samples=negative_samples,
        num_layers=int(hp["layers_num"]),
        gcn_dropout=float(hp["gcn_dropout"]),
        combine_type=hp["combinetype"],
        variance_weight=float(hp["variance_weight"]),
        num_sources=len(ctxs),
        n_samples=int(hp["variance_samples"]),
        contrast_chunk=int(hp["contrast_chunk"]),
    ).to(device)
    spmm = _spmm(int(hp["edge_chunk"]))
    prepared = [
        (
            torch.from_numpy(ctx.x).float().to(device),
            _adj_for(ctx, device, big=ctx.name in STREAM_GRAPHS),
        )
        for ctx in ctxs
    ]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["pretrain_lr"]),
        weight_decay=float(hp["pretrain_weight_decay"]),
    )
    epochs = int(hp["pretrain_epochs"])
    patience = int(hp["pretrain_patience"])
    best = float("inf")
    best_state = None
    waiting = 0

    for epoch in range(epochs):
        # Match the released runner, which resets perturbation/dropout RNG each epoch.
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        model.train()
        optimizer.zero_grad()
        loss, link_loss, variance_loss = model(
            [item[0] for item in prepared], [item[1] for item in prepared], spmm
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"BRIDGE pretrain produced non-finite loss at seed={seed}")
        loss.backward()
        optimizer.step()

        value = float(loss.detach().item())
        if value < best:
            best = value
            waiting = 0
            best_state = {
                key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
            }
        else:
            waiting += 1
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            print(
                f"    [bridge-pretrain] seed={seed} epoch={epoch + 1}/{epochs} "
                f"loss={value:.4f} link={float(link_loss):.4f} "
                f"variance={float(variance_loss):.6f}",
                flush=True,
            )
        if waiting >= patience:
            print(f"    [bridge-pretrain] early stop at epoch={epoch + 1}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("BRIDGE pretraining did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    del prepared, negative_samples, optimizer, best_state
    _empty_cuda_cache()
    return model


def _target_training_data(
    ctx: GraphCtx,
    support: np.ndarray,
    layers: int,
    hp: dict,
    device: str,
    full_basis: SpectralBasis | None,
):
    big = ctx.name in STREAM_GRAPHS
    if not big:
        x = torch.from_numpy(ctx.x).float().to(device)
        adj = _adj_for(ctx, device, big=False)
        index = torch.from_numpy(support).long().to(device)
        if full_basis is None:
            raise RuntimeError("BRIDGE full target spectral basis is missing")
        return x, adj, index, full_basis

    nodes, support_relative, x_sub, adj_sub = _exact_support_dependency_subgraph(
        ctx, support, int(layers)
    )
    raw_sub = ctx.adj[nodes][:, nodes].tocsr()
    basis = _spectral_basis(raw_sub, int(hp["spectral_components"]))
    print(
        f"    [bridge-large-support-subgraph] {ctx.name}: support={len(support)} "
        f"dep_nodes={len(nodes)} dep_edges={adj_sub.nnz} "
        f"spectral_rank={basis.components.shape[0]}",
        flush=True,
    )
    return (
        torch.from_numpy(x_sub).float().to(device),
        CompactEdgeList(adj_sub),
        torch.from_numpy(support_relative).long().to(device),
        basis,
    )


def _fit_downstream(
    model: PrePrompt,
    ctx: GraphCtx,
    support: np.ndarray,
    support_labels: torch.Tensor,
    hp: dict,
    device: str,
    full_basis: SpectralBasis | None,
) -> Tuple[DownPrompt, torch.Tensor | None]:
    down = DownPrompt(
        source_masks=model.source_masks(),
        hidden_dim=int(hp["hid_units"]),
        num_classes=2,
        combine_type=hp["combinetype"],
        dropout=float(hp["routing_dropout"]),
    ).to(device)
    x, adj, support_index, basis = _target_training_data(
        ctx, support, int(hp["layers_num"]), hp, device, full_basis
    )
    components = torch.from_numpy(basis.components).float().to(device)
    values = torch.from_numpy(basis.values).float().to(device)
    spmm = _spmm(int(hp["edge_chunk"]))
    steps = (
        int(hp["large_downstream_steps"])
        if ctx.name in STREAM_GRAPHS
        else int(hp["downstream_steps"])
    )
    optimizer = torch.optim.Adam(down.parameters(), lr=float(hp["downstream_lr"]))
    best = float("inf")
    waiting = 0

    model.gcn.eval()
    requires_grad = [parameter.requires_grad for parameter in model.gcn.parameters()]
    for parameter in model.gcn.parameters():
        parameter.requires_grad_(False)
    try:
        for step in range(steps):
            down.train()
            prompted = down.prompted_features(x)
            regularizer = spectral_regularization_smooth(
                prompted, x, components, values, float(hp["reg_thres"])
            )
            embeddings = model.gcn(prompted, adj, spmm, lp=False).squeeze(0)
            probabilities = down.probabilities(
                embeddings, support_index, support_labels, train=True
            )
            classification = F.cross_entropy(probabilities, support_labels)
            entropy = down.entropy(probabilities).mean()
            loss = (
                classification
                + float(hp["lambda_entropy"]) * entropy
                + float(hp["reg_weight"]) * regularizer
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"BRIDGE downstream produced non-finite loss on {ctx.name}"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = float(loss.detach().item())
            if value < best:
                best = value
                waiting = 0
            else:
                waiting += 1
            if (step + 1) % max(steps // 4, 1) == 0 or step == 0:
                print(
                    f"    [bridge-down] {ctx.name} step={step + 1}/{steps} "
                    f"loss={value:.4f} cls={float(classification.detach()):.4f} "
                    f"entropy={float(entropy.detach()):.4f} "
                    f"spectral={float(regularizer.detach()):.4f}",
                    flush=True,
                )
            if waiting >= int(hp["downstream_patience"]):
                print(f"    [bridge-down] early stop at step={step + 1}", flush=True)
                break

        down.eval()
        with torch.no_grad():
            prompted = down.prompted_features(x)
            final_embeddings = model.gcn(prompted, adj, spmm, lp=False).squeeze(0)
            down.probabilities(final_embeddings, support_index, support_labels, train=True)
    finally:
        for parameter, required in zip(model.gcn.parameters(), requires_grad):
            parameter.requires_grad_(required)

    keep_embeddings = final_embeddings if ctx.name not in STREAM_GRAPHS else None
    if keep_embeddings is None:
        del final_embeddings
    del x, adj, support_index, components, values, optimizer
    _empty_cuda_cache()
    return down, keep_embeddings


@torch.no_grad()
def _prompt_features_to_cpu(
    down: DownPrompt, features: np.ndarray, node_chunk: int, device: str
) -> torch.Tensor:
    output = torch.empty(features.shape, dtype=torch.float32, device="cpu")
    block = max(int(node_chunk), 1)
    for start in range(0, features.shape[0], block):
        end = min(start + block, features.shape[0])
        x = torch.from_numpy(features[start:end]).float().to(device)
        output[start:end].copy_(down.prompted_features(x).cpu())
        del x
    return output


@torch.no_grad()
def _gcn_to_cpu_exact(
    gcn, features_cpu: torch.Tensor, adj_norm: sp.spmatrix, hp: dict, device: str
) -> torch.Tensor:
    """Exact row/edge-chunk GCN inference with layer states resident on CPU."""
    csr = sp.csr_matrix(adj_norm).astype(np.float32, copy=False)
    node_chunk = max(int(hp["node_chunk"]), 1)
    edge_chunk = max(int(hp["edge_chunk"]), 1)
    current = features_cpu.contiguous()
    nodes = int(csr.shape[0])

    for layer_index, conv in enumerate(gcn.convs):
        output = torch.empty((nodes, conv.fc.out_features), dtype=torch.float32)
        row_start = 0
        while row_start < nodes:
            max_row_end = min(row_start + node_chunk, nodes)
            edge_budget_end = int(csr.indptr[row_start]) + edge_chunk
            edge_limited_end = int(np.searchsorted(csr.indptr, edge_budget_end, side="right") - 1)
            row_end = max(row_start + 1, min(max_row_end, edge_limited_end))
            row_count = row_end - row_start
            accumulated = torch.zeros(
                (row_count, conv.fc.out_features), dtype=torch.float32, device=device
            )

            first_edge = int(csr.indptr[row_start])
            last_edge = int(csr.indptr[row_end])
            for edge_start in range(first_edge, last_edge, edge_chunk):
                edge_end = min(edge_start + edge_chunk, last_edge)
                positions = np.arange(edge_start, edge_end, dtype=np.int64)
                rows = (
                    np.searchsorted(csr.indptr, positions, side="right") - 1 - row_start
                ).astype(np.int64, copy=False)
                columns = csr.indices[edge_start:edge_end]
                unique_columns, inverse = np.unique(columns, return_inverse=True)

                source_index = torch.from_numpy(unique_columns.astype(np.int64, copy=False))
                source = current.index_select(0, source_index).to(device)
                transformed = conv.fc(source)
                inverse_index = torch.from_numpy(inverse.astype(np.int64, copy=False)).to(device)
                row_index = torch.from_numpy(rows).to(device)
                values = (
                    torch.from_numpy(csr.data[edge_start:edge_end].astype(np.float32, copy=False))
                    .to(device)
                    .unsqueeze(1)
                )
                contribution = transformed.index_select(0, inverse_index)
                contribution.mul_(values)
                accumulated.index_add_(0, row_index, contribution)
                del (source, transformed, inverse_index, row_index, values, contribution)

            if conv.bias is not None:
                accumulated.add_(conv.bias)
            accumulated = conv.act(accumulated)
            if layer_index:
                accumulated.add_(current[row_start:row_end].to(device))
            output[row_start:row_end].copy_(accumulated.cpu())
            del accumulated
            row_start = row_end

        del current
        current = output
        print(
            f"    [bridge-large-layer] layer={layer_index + 1}/{len(gcn.convs)} "
            f"nodes={nodes} hidden={current.shape[1]}",
            flush=True,
        )
        _empty_cuda_cache()
    return current


def _score_from_embeddings(
    down: DownPrompt, embeddings: torch.Tensor, query: np.ndarray, hp: dict, device: str
) -> np.ndarray:
    if down.centers is None:
        raise RuntimeError("BRIDGE target centers were not initialized")
    centers = down.centers.to(device)
    scores = np.empty(query.size, dtype=np.float32)
    chunk = int(hp["eval_query_batch"])
    for start in range(0, query.size, chunk):
        query_chunk = query[start : start + chunk]
        index_cpu = torch.from_numpy(query_chunk).long()
        if embeddings.device.type == "cpu":
            selected = embeddings.index_select(0, index_cpu).to(device)
        else:
            selected = embeddings.index_select(0, index_cpu.to(embeddings.device))
        similarity = F.cosine_similarity(selected.unsqueeze(1), centers.unsqueeze(0), dim=-1)
        probabilities = F.softmax(similarity, dim=1)
        scores[start : start + query_chunk.size] = probabilities[:, 1].detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError("BRIDGE produced non-finite anomaly scores")
    return scores


@torch.no_grad()
def _full_target_embeddings(
    model: PrePrompt, down: DownPrompt, ctx: GraphCtx, hp: dict, device: str
) -> torch.Tensor:
    big = ctx.name in STREAM_GRAPHS
    print(
        f"    [bridge-full-target] {ctx.name}: eval={int(ctx.mark.sum())} "
        f"N={ctx.adj.shape[0]} E={ctx.adj.nnz} edge_chunk={hp['edge_chunk']} "
        f"node_chunk={hp['node_chunk']} query_chunk={hp['eval_query_batch']}",
        flush=True,
    )
    if big:
        if ctx.adj_norm is None:
            raise RuntimeError(f"{ctx.name}: normalized adjacency is missing")
        prompted_cpu = _prompt_features_to_cpu(down, ctx.x, int(hp["node_chunk"]), device)
        return _gcn_to_cpu_exact(model.gcn, prompted_cpu, ctx.adj_norm, hp, device)

    x = torch.from_numpy(ctx.x).float().to(device)
    adj = _adj_for(ctx, device, big=False)
    prompted = down.prompted_features(x)
    del x
    embeddings = model.gcn(prompted, adj, _spmm(int(hp["edge_chunk"])), lp=False).squeeze(0)
    del prompted, adj
    return embeddings


def _score_target(
    model: PrePrompt,
    ctx: GraphCtx,
    hp: dict,
    shot: int,
    seed: int,
    device: str,
    full_basis: SpectralBasis | None,
):
    rng = np.random.RandomState(seed + 40_000)
    support_normal = _sample_class(ctx, 0, shot, rng, reserve_query=True)
    support_anomaly = _sample_class(ctx, 1, shot, rng, reserve_query=True)
    support = np.concatenate([support_normal, support_anomaly])
    query = _query_nodes(ctx, support)
    if query.size == 0:
        raise ValueError(f"{ctx.name}: no query nodes after BRIDGE support selection")
    support_labels = torch.tensor(
        [0] * len(support_normal) + [1] * len(support_anomaly), dtype=torch.long, device=device
    )

    down, embeddings = _fit_downstream(model, ctx, support, support_labels, hp, device, full_basis)
    if embeddings is None:
        embeddings = _full_target_embeddings(model, down, ctx, hp, device)
    scores = _score_from_embeddings(down, embeddings, query, hp, device)
    result = evaluate(ctx.labels[query], scores)
    del embeddings, down
    _empty_cuda_cache()
    return result


def run_bridge(sources, targets, seeds, hp, device, shot=10, feature_dim=8, feature_norm="zscore"):
    src_ctxs = [
        _ctx(name, feature_dim, feature_norm, target=False, build_native_norm=True)
        for name in sources
    ]
    widths = {ctx.x.shape[1] for ctx in src_ctxs}
    if widths != {int(feature_dim)}:
        raise ValueError(f"BRIDGE source feature widths do not match: {sorted(widths)}")

    trained: List[Tuple[int, PrePrompt]] = []
    for seed in seeds:
        model = _pretrain_one_seed(src_ctxs, hp, seed, feature_dim, device)
        model.cpu()
        trained.append((seed, model))
        _empty_cuda_cache()

    per = {name: [] for name in targets}
    for target_index, name in enumerate(targets, start=1):
        print(f"    [bridge-target {target_index}/{len(targets)}] loading {name}", flush=True)
        ctx = _ctx(name, feature_dim, feature_norm, target=True, build_native_norm=True)
        full_basis = None
        if name not in STREAM_GRAPHS:
            full_basis = _spectral_basis(ctx.adj, int(hp["spectral_components"]))
            print(
                f"    [bridge-spectral] {name}: rank={full_basis.components.shape[0]} "
                f"nodes={ctx.adj.shape[0]}",
                flush=True,
            )

        for seed_index, (seed, model) in enumerate(trained, start=1):
            print(
                f"    [bridge-target {target_index}/{len(targets)}] {name} "
                f"seed={seed} ({seed_index}/{len(trained)})",
                flush=True,
            )
            set_seed(seed)
            model.to(device)
            try:
                per[name].append(_score_target(model, ctx, hp, shot, seed, device, full_basis))
            finally:
                model.cpu()
                _empty_cuda_cache()
                gc.collect()
        del ctx, full_basis
        gc.collect()

    return aggregate(per)
