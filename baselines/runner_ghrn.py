"""GHRN under the source->target zero-shot protocol.

Official GHRN is a supervised graph anomaly detector built on BWGNN:

1. train BWGNN and get `pred_y`;
2. prune likely heterophilous edges by high-frequency residuals;
3. train/evaluate BWGNN on the pruned graph.

For this benchmark, target labels are never used for training or pruning.
Target pruning uses predictions from the source-trained stage-1 BWGNN.
"""

import copy
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, remove_self_loops

from baselines.bwgnn import BWGNN
from baselines.config import GHRN_HP, GHRN_QUICK
from baselines.ghrn import prune_edges_by_ghrn
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


def _cfg(quick=False, hp=None):
    cfg = dict(GHRN_HP if hp is None else hp)
    if quick:
        cfg.update(GHRN_QUICK)
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
    edge_index, _ = remove_self_loops(edge_index)
    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=adj.shape[0])
    return edge_index.to(device)


def _as_tensors(feat, label, mark, device):
    x = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(label, dtype=np.int64), device=device)
    idx = torch.as_tensor(
        np.where(np.asarray(mark, dtype=bool))[0], dtype=torch.int64, device=device
    )
    return x, y, idx


def _new_model(in_dim, cfg, device):
    # The runner exclusively uses testlarge(edge_index, x); avoid retaining a
    # graph-sized tensor as an otherwise unused model attribute.
    empty_edge = torch.empty((2, 0), dtype=torch.long, device=device)
    return BWGNN(
        in_feats=in_dim,
        h_feats=cfg["hid_dim"],
        num_classes=2,
        edge_index=empty_edge,
        d=cfg["order"],
    ).to(device)


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
            print_progress("ghrn", progress_label, completed, epochs, started)
    return True


@torch.no_grad()
def _probs(model, edge_index, x):
    model.eval()
    return model.testlarge(edge_index, x).softmax(1)


def _prepare_cpu(name, norm, cfg, target=False):
    if target:
        adj, feat_raw, label, mark = load_dense_target(name)
    else:
        adj, feat_raw, label, mark = load_dense_source(name)
    feat = unify_features(feat_raw, norm, name)
    edge_index = _to_edge_index(adj, "cpu", add_self_loop=cfg["add_self_loop"])
    x, y, idx = _as_tensors(feat, label, mark, "cpu")
    return adj.shape[0], edge_index, x, y, idx, label, mark


def run_ghrn(sources, targets, seeds, device, norm, quick=False, hp=None, progress=None):
    per = {n: [] for n in targets}
    cfg = _cfg(quick=quick, hp=hp)
    select_device(device)

    source_graphs = []
    source_total, seed_total, target_total = len(sources), len(seeds), len(targets)
    for source_pos, sname in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage("ghrn", 1, 5, f"preparing source {source_pos}/{source_total}: {sname}")
        try:
            prepared = _prepare_cpu(sname, norm, cfg, target=False)
            source_graphs.append((sname, prepared))
            n, edge_cpu, _x, _y, idx_cpu, _label, _mark = prepared
            print_stage(
                "ghrn",
                1,
                5,
                f"source ready {source_pos}/{source_total}: {sname} "
                f"N={n} E={edge_cpu.shape[1]} train={idx_cpu.numel()} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [ghrn] source {sname}: preparation skipped ({str(e)[:100]})")

    trained_models = []
    for seed_pos, seed in enumerate(seeds, 1):
        seed_for_device(seed, device)
        stage1 = opt1 = None
        ok_stage1 = False

        # Stage 1: source-trained BWGNN on original graphs, used only to get pred_y.
        for source_pos, (sname, graph) in enumerate(source_graphs, 1):
            try:
                _n, edge_cpu, x_cpu, y_cpu, idx_cpu, _label, _mark = graph
                edge_index = edge_cpu.to(device)
                x, y, idx = x_cpu.to(device), y_cpu.to(device), idx_cpu.to(device)
                if stage1 is None:
                    stage1 = _new_model(x.shape[1], cfg, device)
                    opt1 = torch.optim.Adam(
                        stage1.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
                    )
                ok_stage1 = (
                    _train_one_graph(
                        stage1,
                        opt1,
                        edge_index,
                        x,
                        y,
                        idx,
                        cfg["pretrain_epoch"],
                        progress_label=(
                            f"stage 2/5 pretrain source={sname} "
                            f"({source_pos}/{len(source_graphs)}) seed={seed} "
                            f"({seed_pos}/{seed_total})"
                        ),
                    )
                    or ok_stage1
                )
            except Exception as e:
                print(f"    [ghrn] stage1 source {sname} seed={seed} skipped ({str(e)[:100]})")
            finally:
                edge_index = x = y = idx = None
                empty_device_cache(device)

        if not ok_stage1 or stage1 is None:
            print(f"    [ghrn] seed={seed}: no stage1 source graph trained; skip targets")
            continue

        # Stage 2: prune source graphs by stage1 pred_y, then train final BWGNN.
        final = opt2 = None
        ok_final = False
        for source_pos, (sname, graph) in enumerate(source_graphs, 1):
            try:
                n, edge_cpu, x_cpu, y_cpu, idx_cpu, _label, _mark = graph
                edge_index = edge_cpu.to(device)
                x, y, idx = x_cpu.to(device), y_cpu.to(device), idx_cpu.to(device)
                pred_y = _probs(stage1, edge_index, x)
                pruned = prune_edges_by_ghrn(
                    edge_index,
                    pred_y,
                    n,
                    del_ratio=cfg["del_ratio"],
                    adj_type=cfg["adj_type"],
                    final_self_loop=cfg["add_self_loop"],
                )
                if final is None:
                    final = _new_model(x.shape[1], cfg, device)
                    opt2 = torch.optim.Adam(
                        final.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
                    )
                ok_final = (
                    _train_one_graph(
                        final,
                        opt2,
                        pruned,
                        x,
                        y,
                        idx,
                        cfg["num_epoch"],
                        progress_label=(
                            f"stage 3/5 final source={sname} "
                            f"({source_pos}/{len(source_graphs)}) seed={seed} "
                            f"({seed_pos}/{seed_total})"
                        ),
                    )
                    or ok_final
                )
            except Exception as e:
                print(f"    [ghrn] stage2 source {sname} seed={seed} skipped ({str(e)[:100]})")
            finally:
                edge_index = x = y = idx = pred_y = pruned = None
                empty_device_cache(device)

        if not ok_final or final is None:
            print(f"    [ghrn] seed={seed}: no pruned source graph trained; skip targets")
            continue
        opt1 = opt2 = None
        for trained in (stage1, final):
            for parameter in trained.parameters():
                parameter.grad = None
        trained_models.append((int(seed), stage1, final))

    source_graphs.clear()
    graph = edge_cpu = x_cpu = y_cpu = idx_cpu = None
    empty_device_cache(device)

    for target_pos, tname in enumerate(targets, 1):
        started = time.perf_counter()
        print_stage("ghrn", 4, 5, f"preparing target {target_pos}/{target_total}: {tname}")
        try:
            n, edge_cpu, x_cpu, _y_cpu, _idx_cpu, label, mark = _prepare_cpu(
                tname, norm, cfg, target=True
            )
            print_stage(
                "ghrn",
                4,
                5,
                f"target ready {target_pos}/{target_total}: {tname} "
                f"N={n} E={edge_cpu.shape[1]} eval={int(np.asarray(mark).sum())} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [ghrn] target {tname}: preparation skipped ({str(e)[:100]})")
            empty_device_cache(device)
            continue

        try:
            edge_index = edge_cpu.to(device)
            x = x_cpu.to(device)
            gpu_ready = True
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"    [ghrn] target {tname}: device preparation skipped ({str(e)[:100]})")
                continue
            print(f"    [ghrn] target {tname}: CUDA OOM during preparation; score on CPU")
            edge_index = x = None
            empty_device_cache(device)
            gpu_ready = False

        model_total = len(trained_models)
        score_started = time.perf_counter()
        for seed_pos, (seed, stage1, final) in enumerate(trained_models, 1):
            print_stage(
                "ghrn",
                5,
                5,
                f"pruning/scoring target={tname} ({target_pos}/{target_total}) "
                f"seed={seed} ({seed_pos}/{model_total})",
            )
            try:
                stage1_eval = stage1 if gpu_ready else copy.deepcopy(stage1).cpu()
                final_eval = final if gpu_ready else copy.deepcopy(final).cpu()
                eval_edge = edge_index if gpu_ready else edge_cpu
                eval_x = x if gpu_ready else x_cpu
                pred_y = _probs(stage1_eval, eval_edge, eval_x)
                pruned = prune_edges_by_ghrn(
                    eval_edge,
                    pred_y,
                    n,
                    del_ratio=cfg["del_ratio"],
                    adj_type=cfg["adj_type"],
                    final_self_loop=cfg["add_self_loop"],
                )
                scores = _probs(final_eval, pruned, eval_x)[:, 1].detach().cpu().numpy()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or not gpu_ready:
                    print(f"    [ghrn] target {tname} seed={seed} skipped ({str(e)[:100]})")
                    pred_y = pruned = None
                    empty_device_cache(device)
                    continue
                print(f"    [ghrn] target {tname} seed={seed}: CUDA OOM, retry scoring on CPU")
                pred_y = pruned = edge_index = x = None
                empty_device_cache(device)
                gpu_ready = False
                try:
                    stage1_eval = copy.deepcopy(stage1).cpu()
                    final_eval = copy.deepcopy(final).cpu()
                    pred_y = _probs(stage1_eval, edge_cpu, x_cpu)
                    pruned = prune_edges_by_ghrn(
                        edge_cpu,
                        pred_y,
                        n,
                        del_ratio=cfg["del_ratio"],
                        adj_type=cfg["adj_type"],
                        final_self_loop=cfg["add_self_loop"],
                    )
                    scores = _probs(final_eval, pruned, x_cpu)[:, 1].detach().cpu().numpy()
                except Exception as cpu_error:
                    print(
                        f"    [ghrn] target {tname} seed={seed}: CPU retry failed "
                        f"({str(cpu_error)[:100]})"
                    )
                    pred_y = pruned = stage1_eval = final_eval = None
                    continue
            except Exception as e:
                print(f"    [ghrn] target {tname} seed={seed} skipped ({str(e)[:100]})")
                pred_y = pruned = None
                empty_device_cache(device)
                continue
            if not np.isfinite(scores).all():
                print(f"    [ghrn] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            print_progress(
                "ghrn", f"stage 5/5 target={tname} seeds", seed_pos, model_total, score_started
            )
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
            pred_y = pruned = stage1_eval = final_eval = None
            empty_device_cache(device)
        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])
        edge_index = x = pred_y = pruned = edge_cpu = x_cpu = None
        label = mark = None
        empty_device_cache(device)

    return aggregate(per)
