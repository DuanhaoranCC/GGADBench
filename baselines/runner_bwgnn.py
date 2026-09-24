"""PyG/edge_index BWGNN under the source->target zero-shot protocol."""

import copy
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, remove_self_loops

from baselines.bwgnn import BWGNN
from baselines.config import BWGNN_HP, BWGNN_QUICK
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


def _cfg_for_sources(source_names, quick=False, hp=None):
    del source_names
    cfg = dict(BWGNN_HP if hp is None else hp)
    if quick:
        cfg.update(BWGNN_QUICK)
    return cfg


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


def _prepare_cpu(adj, feat, label, mark, add_self_loop):
    return (
        _to_edge_index(adj, "cpu", add_self_loop=add_self_loop),
        *_as_tensors(feat, label, mark, "cpu"),
    )


def _train_one_graph(model, optimizer, edge_index, x, y, idx, epochs, progress_label=None):
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
        logits = model.testlarge(edge_index, x)
        loss = F.cross_entropy(logits[idx], y_train, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed = epoch + 1
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == epochs
        ):
            print_progress("bwgnn", progress_label, completed, epochs, started)
    return True


@torch.no_grad()
def _score(model, edge_index, x):
    model.eval()
    return model.testlarge(edge_index, x).softmax(1)[:, 1].detach().cpu().numpy()


@torch.no_grad()
def _score_cpu_fallback(model, edge_index, x):
    model_cpu = copy.deepcopy(model).cpu()
    return _score(model_cpu, edge_index.cpu(), x.cpu())


def run_bwgnn(sources, targets, seeds, device, norm, quick=False, hp=None, progress=None):
    per = {n: [] for n in targets}
    cfg = _cfg_for_sources(sources, quick=quick, hp=hp)
    select_device(device)

    source_graphs = []
    source_total, seed_total, target_total = len(sources), len(seeds), len(targets)
    for source_pos, sname in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage("bwgnn", 1, 4, f"preparing source {source_pos}/{source_total}: {sname}")
        try:
            adj, feat_raw, label, mark = load_dense_source(sname)
            feat = unify_features(feat_raw, norm, sname)
            source_graphs.append(
                (sname, _prepare_cpu(adj, feat, label, mark, cfg["add_self_loop"]))
            )
            print_stage(
                "bwgnn",
                1,
                4,
                f"source ready {source_pos}/{source_total}: {sname} "
                f"N={adj.shape[0]} E={adj.nnz} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [bwgnn] source {sname}: preparation skipped ({str(e)[:100]})")

    trained_models = []
    for seed_pos, seed in enumerate(seeds, 1):
        seed_for_device(seed, device)
        model = optimizer = None
        ok_train = False

        for source_pos, (sname, graph) in enumerate(source_graphs, 1):
            try:
                edge_cpu, x_cpu, y_cpu, idx_cpu = graph
                edge_index = edge_cpu.to(device)
                x, y, idx = x_cpu.to(device), y_cpu.to(device), idx_cpu.to(device)
                if model is None:
                    empty_edge = torch.empty((2, 0), dtype=torch.long, device=device)
                    model = BWGNN(
                        in_feats=x.shape[1],
                        h_feats=cfg["hid_dim"],
                        num_classes=2,
                        # This runner exclusively calls testlarge(edge_index,
                        # x).  Do not retain a source-sized edge tensor inside
                        # every seed model as an unused attribute.
                        edge_index=empty_edge,
                        d=cfg["order"],
                    ).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                    )
                trained = _train_one_graph(
                    model,
                    optimizer,
                    edge_index,
                    x,
                    y,
                    idx,
                    cfg["num_epoch"],
                    progress_label=(
                        f"stage 2/4 source={sname} "
                        f"({source_pos}/{len(source_graphs)}) seed={seed} "
                        f"({seed_pos}/{seed_total})"
                    ),
                )
                ok_train = ok_train or trained
            except Exception as e:
                print(f"    [bwgnn] source {sname} seed={seed} skipped ({str(e)[:100]})")
            finally:
                edge_index = x = y = idx = None
                empty_device_cache(device)

        if not ok_train or model is None:
            print(f"    [bwgnn] seed={seed}: no source graph trained; skip targets")
            continue
        optimizer = None
        for parameter in model.parameters():
            parameter.grad = None
        trained_models.append((int(seed), model))

    # Models contain no graph-sized tensors; release source caches before the
    # first large target is loaded.
    source_graphs.clear()
    graph = edge_cpu = x_cpu = y_cpu = idx_cpu = None
    adj = feat_raw = feat = label = mark = None
    empty_device_cache(device)

    for target_pos, tname in enumerate(targets, 1):
        started = time.perf_counter()
        print_stage("bwgnn", 3, 4, f"preparing target {target_pos}/{target_total}: {tname}")
        try:
            adj, feat_raw, label, mark = load_dense_target(tname)
            feat = unify_features(feat_raw, norm, tname)
            edge_cpu, x_cpu, _y_cpu, _idx_cpu = _prepare_cpu(
                adj, feat, label, mark, cfg["add_self_loop"]
            )
            edge_index = edge_cpu.to(device)
            x = x_cpu.to(device)
            gpu_ready = True
            print_stage(
                "bwgnn",
                3,
                4,
                f"target ready {target_pos}/{target_total}: {tname} "
                f"N={adj.shape[0]} E={adj.nnz} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"    [bwgnn] target {tname}: preparation skipped ({str(e)[:100]})")
                continue
            print(f"    [bwgnn] target {tname}: CUDA OOM during preparation; score on CPU")
            empty_device_cache(device)
            gpu_ready = False
        except Exception as e:
            print(f"    [bwgnn] target {tname}: preparation skipped ({str(e)[:100]})")
            edge_index = x = edge_cpu = x_cpu = None
            adj = feat_raw = feat = label = mark = None
            empty_device_cache(device)
            continue

        trained_total = len(trained_models)
        score_started = time.perf_counter()
        for seed_pos, (seed, model) in enumerate(trained_models, 1):
            print_stage(
                "bwgnn",
                4,
                4,
                f"scoring target={tname} ({target_pos}/{target_total}) "
                f"seed={seed} ({seed_pos}/{trained_total})",
            )
            try:
                if gpu_ready:
                    scores = _score(model, edge_index, x)
                else:
                    scores = _score_cpu_fallback(model, edge_cpu, x_cpu)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or not gpu_ready:
                    print(f"    [bwgnn] target {tname} seed={seed} skipped ({str(e)[:100]})")
                    empty_device_cache(device)
                    continue
                print(f"    [bwgnn] target {tname} seed={seed}: CUDA OOM, retry scoring on CPU")
                empty_device_cache(device)
                try:
                    scores = _score_cpu_fallback(model, edge_cpu, x_cpu)
                except Exception as cpu_error:
                    print(
                        f"    [bwgnn] target {tname} seed={seed}: CPU retry failed "
                        f"({str(cpu_error)[:100]})"
                    )
                    continue
            except Exception as e:
                print(f"    [bwgnn] target {tname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)
                continue
            if not np.isfinite(scores).all():
                print(f"    [bwgnn] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            print_progress(
                "bwgnn", f"stage 4/4 target={tname} seeds", seed_pos, trained_total, score_started
            )
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
            empty_device_cache(device)
        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])
        edge_index = x = edge_cpu = x_cpu = None
        adj = feat_raw = feat = label = mark = None
        empty_device_cache(device)

    return aggregate(per)
