"""Shared source->target runner for supervised PyG GCN/GAT baselines."""

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops, remove_self_loops

from baselines.config import GAT_HP, GAT_QUICK, GCN_HP, GCN_QUICK, MLP_HP, MLP_QUICK
from baselines.gnn import GAT, GCN, MLP
from baselines.preprocess import unify_features
from baselines.protocol import load_dense_source, load_dense_target
from baselines.runtime import (
    empty_device_cache,
    print_progress,
    print_stage,
    seed_for_device,
    select_device,
)
from common.data import aggregate
from util import evaluate


def _to_edge_index(adj, device, add_self_loop=True):
    coo = sp.coo_matrix(adj)
    edge_index = torch.stack(
        [
            torch.as_tensor(coo.row, dtype=torch.int64),
            torch.as_tensor(coo.col, dtype=torch.int64),
        ],
        dim=0,
    )
    if add_self_loop:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=adj.shape[0])
    return edge_index.to(device)


def _as_tensors(feat, label, mark, device):
    x = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(label, dtype=np.int64), device=device)
    idx = torch.as_tensor(
        np.where(np.asarray(mark, dtype=bool))[0], dtype=torch.int64, device=device
    )
    return x, y, idx


def _prepare_cpu_graph(adj, feat, label, mark, cfg, method="gcn"):
    # MLP is the feature-only control: do not even materialize an edge tensor,
    # so graph structure cannot accidentally enter training or inference.
    feature_only = method == "mlp"
    edge_index = (
        torch.empty((2, 0), dtype=torch.int64)
        if feature_only
        else _to_edge_index(adj, "cpu", add_self_loop=cfg["add_self_loop"])
    )
    x, y, idx = _as_tensors(feat, label, mark, "cpu")
    return {
        "edge_index": edge_index,
        "x": x,
        "y": y,
        "idx": idx,
        "data": None if feature_only else Data(x=x, y=y, edge_index=edge_index),
        "num_nodes": adj.shape[0],
        "gcn_degree_inv": None,
        "batched_source": (
            False if feature_only else adj.nnz > cfg.get("source_batch_edges", float("inf"))
        ),
        "batched_target": (
            False if feature_only else adj.nnz > cfg.get("target_sample_edges", float("inf"))
        ),
    }


def _cfg(method, quick, hp=None):
    if method == "gcn":
        cfg = dict(GCN_HP if hp is None else hp)
        if quick:
            cfg.update(GCN_QUICK)
        return cfg
    if method == "gat":
        cfg = dict(GAT_HP if hp is None else hp)
        if quick:
            cfg.update(GAT_QUICK)
        return cfg
    if method == "mlp":
        cfg = dict(MLP_HP if hp is None else hp)
        if quick:
            cfg.update(MLP_QUICK)
        return cfg
    raise ValueError(method)


def _init_model(method, in_dim, cfg):
    if method == "gcn":
        return GCN(
            in_dim=in_dim,
            hid_dim=cfg["hid_dim"],
            out_dim=2,
            num_layers=cfg.get("num_layers", 2),
            activation=cfg.get("activation", "relu"),
            dropout=cfg["dropout"],
        )
    if method == "gat":
        return GAT(
            in_dim=in_dim,
            hid_dim=cfg["hid_dim"],
            out_dim=2,
            num_layers=cfg.get("num_layers", 2),
            activation=cfg.get("activation", "elu"),
            heads=cfg["heads"],
            out_heads=cfg["out_heads"],
            dropout=cfg["dropout"],
        )
    if method == "mlp":
        return MLP(
            in_dim=in_dim,
            hid_dim=cfg["hid_dim"],
            out_dim=2,
            num_layers=cfg["num_layers"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )
    raise ValueError(method)


def _train_one_graph(
    model, optimizer, edge_index, x, y, idx, epochs, progress_label=None, method="gnn"
):
    if idx.numel() == 0:
        return False
    y_train = y[idx]
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    if pos == 0 or neg == 0:
        return False

    weight = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=x.device)
    started = time.perf_counter()
    progress_every = max(1, epochs // 20)
    for epoch in range(epochs):
        model.train()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[idx], y_train, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed = epoch + 1
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == epochs
        ):
            print_progress(method, progress_label, completed, epochs, started)
    return True


def _class_weight(y_train, device):
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)


def _gcn_degree_inv(graph):
    cached = graph["gcn_degree_inv"]
    if cached is not None:
        return cached
    dst = graph["edge_index"][1]
    degree = torch.bincount(dst, minlength=graph["num_nodes"]).to(torch.float32)
    cached = degree.pow(-0.5)
    cached.masked_fill_(torch.isinf(cached), 0.0)
    graph["gcn_degree_inv"] = cached
    return cached


def _gcn_sparse_adj(graph, device):
    edge_index = graph["edge_index"]
    src, dst = edge_index
    degree_inv = _gcn_degree_inv(graph)
    value = degree_inv[src] * degree_inv[dst]
    # sparse.mm expects matrix[row=destination, col=source].
    indices = torch.stack([dst, src], dim=0).to(device)
    return torch.sparse_coo_tensor(
        indices,
        value.to(device),
        (graph["num_nodes"], graph["num_nodes"]),
        device=device,
    ).coalesce()


def _gcn_forward_sparse(model, x, adj_norm):
    h = x
    for index, conv in enumerate(model.convs):
        h = torch.sparse.mm(adj_norm, conv.lin(h))
        if conv.bias is not None:
            h = h + conv.bias
        if index + 1 < len(model.convs):
            h = model.activation(h)
            h = model.dropout(h)
    return h


def _train_gcn_sparse(model, optimizer, graph, device, epochs, progress_label=None):
    idx = graph["idx"].to(device)
    y = graph["y"].to(device)
    if idx.numel() == 0:
        return False
    weight = _class_weight(y[idx], device)
    if weight is None:
        return False
    x = graph["x"].to(device)
    adj_norm = _gcn_sparse_adj(graph, device)
    started = time.perf_counter()
    progress_every = max(1, epochs // 20)
    for epoch in range(epochs):
        model.train()
        logits = _gcn_forward_sparse(model, x, adj_norm)
        loss = F.cross_entropy(logits[idx], y[idx], weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed = epoch + 1
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == epochs
        ):
            print_progress("gcn", progress_label, completed, epochs, started)
    return True


@torch.no_grad()
def _gcn_conv_chunked(
    conv,
    x,
    edge_index_cpu,
    degree_inv,
    device,
    chunk_edges=2_000_000,
    progress_label=None,
    progress_offset=0,
    progress_total=None,
    progress_started=None,
):
    transformed = conv.lin(x)
    out = transformed.new_zeros((transformed.shape[0], transformed.shape[1]))
    inv = degree_inv.to(device)
    src_cpu, dst_cpu = edge_index_cpu
    chunks = max(1, math.ceil(src_cpu.numel() / chunk_edges))
    progress_every = max(1, (progress_total or chunks) // 20)
    for chunk_pos, start in enumerate(range(0, src_cpu.numel(), chunk_edges), 1):
        end = min(start + chunk_edges, src_cpu.numel())
        src = src_cpu[start:end].to(device)
        dst = dst_cpu[start:end].to(device)
        weight = inv[src] * inv[dst]
        out.index_add_(0, dst, transformed[src] * weight.unsqueeze(1))
        completed = progress_offset + chunk_pos
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == progress_total
        ):
            print_progress(
                "gcn",
                progress_label,
                completed,
                progress_total or chunks,
                progress_started or time.perf_counter(),
            )
    if conv.bias is not None:
        out = out + conv.bias
    return out


@torch.no_grad()
def _score_gcn_chunked(model, graph, device, progress_label=None):
    """Exact full-graph GCN inference with global degrees and bounded edges."""
    model.eval()
    x = graph["x"].to(device)
    degree_inv = _gcn_degree_inv(graph)
    chunks = max(1, math.ceil(graph["edge_index"].shape[1] / 2_000_000))
    started = time.perf_counter()
    h = x
    total_chunks = len(model.convs) * chunks
    for index, conv in enumerate(model.convs):
        h = _gcn_conv_chunked(
            conv,
            h,
            graph["edge_index"],
            degree_inv,
            device,
            progress_label=progress_label,
            progress_offset=index * chunks,
            progress_total=total_chunks,
            progress_started=started,
        )
        if index + 1 < len(model.convs):
            h = model.activation(h)
            h = model.dropout(h)
    logits = h
    scores = np.zeros(graph["num_nodes"], dtype=np.float32)
    idx = graph["idx"].numpy()
    # Only marked scores are consumed by evaluation; transfer those rows once.
    scores[idx] = logits[idx].softmax(1)[:, 1].cpu().numpy()
    return scores


def _train_neighbor_prepared(
    model, optimizer, graph, device, cfg, epochs, progress_label=None, method="gat"
):
    train_nodes = graph["idx"]
    if train_nodes.numel() == 0:
        return False
    y_train = graph["y"][train_nodes]
    weight = _class_weight(y_train, device)
    if weight is None:
        return False
    fanouts = list(cfg.get("source_num_neighbors", [-1, -1]))
    layers = max(1, int(cfg.get("num_layers", len(fanouts))))
    fanouts = (fanouts + [fanouts[-1]] * layers)[:layers]
    loader = NeighborLoader(
        graph["data"],
        input_nodes=train_nodes,
        num_neighbors=fanouts,
        batch_size=cfg.get("source_batch_size", 4096),
        shuffle=True,
    )
    started = time.perf_counter()
    progress_every = max(1, epochs // 20)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            seed_n = int(batch.batch_size)
            loss = F.cross_entropy(logits[:seed_n], batch.y[:seed_n], weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        completed = epoch + 1
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == epochs
        ):
            print_progress(method, progress_label, completed, epochs, started)
    return True


@torch.no_grad()
def _score(model, edge_index, x):
    model.eval()
    return model(x, edge_index).softmax(1)[:, 1].detach().cpu().numpy()


@torch.no_grad()
def _score_mlp_prepared(model, graph, device, cfg, progress_label=None):
    """Score every marked node exactly, in feature-only node batches."""
    model.eval()
    eval_nodes = graph["idx"]
    batch_size = max(1, int(cfg.get("target_batch_size", 65_536)))
    total = max(1, math.ceil(eval_nodes.numel() / batch_size))
    progress_every = max(1, total // 20)
    started = time.perf_counter()
    scores = np.zeros(graph["num_nodes"], dtype=np.float32)
    for batch_pos, start in enumerate(range(0, eval_nodes.numel(), batch_size), 1):
        nodes = eval_nodes[start : start + batch_size]
        logits = model(graph["x"][nodes].to(device))
        scores[nodes.numpy()] = logits.softmax(1)[:, 1].cpu().numpy()
        if progress_label is not None and (
            batch_pos == 1 or batch_pos % progress_every == 0 or batch_pos == total
        ):
            print_progress("mlp", progress_label, batch_pos, total, started)
    return scores


@torch.no_grad()
def _score_neighbor_prepared(model, graph, device, cfg, progress_label=None, method="gat"):
    eval_nodes = graph["idx"]
    fanouts = list(cfg.get("target_num_neighbors", [15, 10]))
    layers = max(1, int(cfg.get("num_layers", len(fanouts))))
    fanouts = (fanouts + [fanouts[-1]] * layers)[:layers]
    loader = NeighborLoader(
        graph["data"],
        input_nodes=eval_nodes,
        num_neighbors=fanouts,
        batch_size=cfg.get("target_batch_size", 4096),
        shuffle=False,
    )
    scores = np.zeros(graph["x"].shape[0], dtype=np.float32)
    model.eval()
    total = max(1, len(loader))
    progress_every = max(1, total // 20)
    started = time.perf_counter()
    for batch_pos, batch in enumerate(loader, 1):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        seed_n = int(batch.batch_size)
        node_ids = batch.n_id[:seed_n].detach().cpu().numpy()
        scores[node_ids] = logits[:seed_n].softmax(1)[:, 1].detach().cpu().numpy()
        if progress_label is not None and (
            batch_pos == 1 or batch_pos % progress_every == 0 or batch_pos == total
        ):
            print_progress(method, progress_label, batch_pos, total, started)
    return scores


def run_supervised_gnn(
    method, sources, targets, seeds, device, norm, quick=False, hp=None, progress=None
):
    per = {n: [] for n in targets}
    cfg = _cfg(method, quick, hp=hp)
    select_device(device)

    source_graphs = []
    source_total, seed_total, target_total = len(sources), len(seeds), len(targets)
    for source_pos, sname in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage(method, 1, 4, f"preparing source {source_pos}/{source_total}: {sname}")
        try:
            adj, feat_raw, label, mark = load_dense_source(sname)
            feat = unify_features(feat_raw, norm, sname)
            prepared = _prepare_cpu_graph(adj, feat, label, mark, cfg, method=method)
            source_graphs.append((sname, prepared))
            print_stage(
                method,
                1,
                4,
                f"source ready {source_pos}/{source_total}: {sname} "
                f"N={adj.shape[0]} E={adj.nnz} train={prepared['idx'].numel()} "
                f"path={'feature-only' if method == 'mlp' else ('batched' if prepared['batched_source'] else 'full')} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [{method}] source {sname}: preparation skipped ({str(e)[:100]})")

    trained_models = []
    for seed_pos, seed in enumerate(seeds, 1):
        seed_for_device(seed, device)
        model = optimizer = None
        ok_train = False

        for source_pos, (sname, graph) in enumerate(source_graphs, 1):
            try:
                progress_label = (
                    f"stage 2/4 source={sname} "
                    f"({source_pos}/{len(source_graphs)}) seed={seed} "
                    f"({seed_pos}/{seed_total})"
                )
                if model is None:
                    model = _init_model(method, graph["x"].shape[1], cfg).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                    )
                if method == "gcn" and graph["batched_source"]:
                    print(f"    [gcn] source {sname}: exact full-graph sparse training")
                    trained = _train_gcn_sparse(
                        model,
                        optimizer,
                        graph,
                        device,
                        cfg["num_epoch"],
                        progress_label=progress_label,
                    )
                elif graph["batched_source"]:
                    print(
                        f"    [{method}] source {sname}: NeighborLoader training over all marked nodes"
                    )
                    trained = _train_neighbor_prepared(
                        model,
                        optimizer,
                        graph,
                        device,
                        cfg,
                        cfg["num_epoch"],
                        progress_label=progress_label,
                        method=method,
                    )
                else:
                    edge_index = graph["edge_index"].to(device)
                    x, y, idx = (
                        graph["x"].to(device),
                        graph["y"].to(device),
                        graph["idx"].to(device),
                    )
                    trained = _train_one_graph(
                        model,
                        optimizer,
                        edge_index,
                        x,
                        y,
                        idx,
                        cfg["num_epoch"],
                        progress_label=progress_label,
                        method=method,
                    )
                ok_train = ok_train or trained
            except Exception as e:
                print(f"    [{method}] source {sname} seed={seed} skipped ({str(e)[:100]})")
            finally:
                edge_index = x = y = idx = None
                empty_device_cache(device)

        if not ok_train or model is None:
            print(f"    [{method}] seed={seed}: no source graph trained; skip targets")
            continue
        optimizer = None
        for parameter in model.parameters():
            parameter.grad = None
        trained_models.append((int(seed), model))

    source_graphs.clear()
    graph = None
    adj = feat_raw = feat = label = mark = None
    empty_device_cache(device)

    for target_pos, tname in enumerate(targets, 1):
        started = time.perf_counter()
        print_stage(method, 3, 4, f"preparing target {target_pos}/{target_total}: {tname}")
        try:
            adj, feat_raw, label, mark = load_dense_target(tname)
            feat = unify_features(feat_raw, norm, tname)
            graph = _prepare_cpu_graph(adj, feat, label, mark, cfg, method=method)
            print_stage(
                method,
                3,
                4,
                f"target ready {target_pos}/{target_total}: {tname} "
                f"N={adj.shape[0]} E={adj.nnz} eval={graph['idx'].numel()} "
                f"path={'feature-batch' if method == 'mlp' else ('batched' if graph['batched_target'] else 'full')} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [{method}] target {tname}: preparation skipped ({str(e)[:100]})")
            graph = adj = feat_raw = feat = label = mark = None
            empty_device_cache(device)
            continue

        use_batched_target = graph["batched_target"]
        edge_index = x = None
        if method != "mlp" and not use_batched_target:
            try:
                edge_index = graph["edge_index"].to(device)
                x = graph["x"].to(device)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    print(
                        f"    [{method}] target {tname}: device preparation skipped ({str(e)[:100]})"
                    )
                    graph = None
                    empty_device_cache(device)
                    continue
                print(
                    f"    [{method}] target {tname}: CUDA OOM during preparation; use NeighborLoader"
                )
                edge_index = x = None
                empty_device_cache(device)
                use_batched_target = True

        model_total = len(trained_models)
        score_started = time.perf_counter()
        for seed_pos, (seed, model) in enumerate(trained_models, 1):
            score_label = (
                f"stage 4/4 target={tname} ({target_pos}/{target_total}) "
                f"seed={seed} ({seed_pos}/{model_total})"
            )
            print_stage(method, 4, 4, f"scoring {score_label[10:]}")
            try:
                if method == "mlp":
                    scores = _score_mlp_prepared(
                        model, graph, device, cfg, progress_label=score_label
                    )
                elif method == "gcn" and use_batched_target:
                    print(f"    [gcn] target {tname}: exact global-degree edge-chunk inference")
                    scores = _score_gcn_chunked(model, graph, device, progress_label=score_label)
                elif use_batched_target:
                    print(
                        f"    [{method}] target {tname}: NeighborLoader inference over all marked nodes"
                    )
                    scores = _score_neighbor_prepared(
                        model, graph, device, cfg, progress_label=score_label, method=method
                    )
                else:
                    try:
                        scores = _score(model, edge_index, x)
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        fallback = (
                            "exact edge-chunk inference"
                            if method == "gcn"
                            else "NeighborLoader inference over all marked nodes"
                        )
                        print(f"    [{method}] target {tname}: CUDA OOM, retry {fallback}")
                        edge_index = x = None
                        empty_device_cache(device)
                        use_batched_target = True
                        scores = (
                            _score_gcn_chunked(model, graph, device, progress_label=score_label)
                            if method == "gcn"
                            else _score_neighbor_prepared(
                                model, graph, device, cfg, progress_label=score_label, method=method
                            )
                        )
            except Exception as e:
                print(f"    [{method}] target {tname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)
                continue
            if not np.isfinite(scores).all():
                print(f"    [{method}] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            if not use_batched_target:
                print_progress(
                    method, f"stage 4/4 target={tname} seeds", seed_pos, model_total, score_started
                )
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
            empty_device_cache(device)
        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])
        graph = edge_index = x = None
        adj = feat_raw = feat = label = mark = None
        empty_device_cache(device)

    return aggregate(per)
