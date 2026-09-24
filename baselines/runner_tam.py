"""TAM under the source-to-target graph anomaly detection protocol."""

import copy
import time

import numpy as np

from baselines.config import TAM_HP, TAM_LAMBDA_ONE, TAM_QUICK
from baselines.preprocess import unify_features
from baselines.protocol import DENSE_SAFE_E, DENSE_SAFE_N, load_dense_source, load_dense_target
from baselines.runtime import empty_device_cache, print_stage, seed_for_device, select_device
from baselines.tam import (
    init_ensemble,
    prepare_tam_graph,
    score_with_prepared_ensemble,
    train_ensemble_on_graph,
    train_ensemble_on_graph_sparse,
)
from common.data import aggregate
from util import evaluate


def _train_cfg(source_names, quick=False, hp=None):
    del source_names
    cfg = dict(TAM_HP if hp is None else hp)
    if quick:
        cfg.update(TAM_QUICK)
    return cfg


def _lambda_for_source(name, cfg):
    if cfg.get("lamda") is not None:
        return float(cfg["lamda"])
    return 1.0 if name in TAM_LAMBDA_ONE else 0.0


def run_tam(sources, targets, seeds, device, norm, quick=False, hp=None, progress=None):
    per = {n: [] for n in targets}
    cfg = _train_cfg(sources, quick=quick, hp=hp)
    select_device(device)

    source_graphs = []
    source_total, seed_total, target_total = len(sources), len(seeds), len(targets)
    if seeds:
        seed_for_device(seeds[0], device)
    for source_pos, sname in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage("tam", 1, 4, f"preparing source {source_pos}/{source_total}: {sname}")
        try:
            adj, feat_raw, _label, _mark = load_dense_source(sname)
            feat = unify_features(feat_raw, norm, sname)
            sparse = adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E
            source_graphs.append((sname, adj, feat, sparse))
            print_stage(
                "tam",
                1,
                4,
                f"source ready {source_pos}/{source_total}: {sname} "
                f"N={adj.shape[0]} E={adj.nnz} "
                f"path={'sparse' if sparse else 'dense'} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [tam] source {sname}: preparation skipped ({str(e)[:100]})")

    trained_ensembles = []
    for seed_pos, seed in enumerate(seeds, 1):
        seed_for_device(seed, device)
        models = opts = None
        ok_train = False

        for source_pos, (sname, adj, feat, sparse) in enumerate(source_graphs, 1):
            try:
                if models is None:
                    models, opts = init_ensemble(
                        feat.shape[1],
                        device,
                        embedding_dim=cfg["embedding_dim"],
                        cutting=cfg["cutting"],
                        n_tree=cfg["n_tree"],
                        negsamp_ratio=cfg["negsamp_ratio"],
                        readout=cfg["readout"],
                        lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                    )
                train_fn = train_ensemble_on_graph_sparse if sparse else train_ensemble_on_graph
                train_fn(
                    models,
                    opts,
                    adj,
                    feat,
                    device,
                    num_epoch=cfg["num_epoch"],
                    cutting=cfg["cutting"],
                    n_tree=cfg["n_tree"],
                    lamda=_lambda_for_source(sname, cfg),
                    progress_label=(
                        f"stage 2/4 source={sname} "
                        f"({source_pos}/{len(source_graphs)}) seed={seed} "
                        f"({seed_pos}/{seed_total})"
                    ),
                )
                ok_train = True
                empty_device_cache(device)
            except Exception as e:
                print(f"    [tam] source {sname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)

        if not ok_train or models is None:
            print(f"    [tam] seed={seed}: no source graph trained; skip targets")
            continue
        # Adam state is not used for target inference and is substantially
        # larger than the LAMNet ensemble itself.
        opts = None
        for model in models:
            for parameter in model.parameters():
                parameter.grad = None
        trained_ensembles.append((int(seed), models))

    source_graphs.clear()
    adj = feat_raw = feat = _label = _mark = None
    empty_device_cache(device)

    for target_pos, tname in enumerate(targets, 1):
        started = time.perf_counter()
        print_stage("tam", 3, 4, f"preparing target {target_pos}/{target_total}: {tname}")
        try:
            adj, feat_raw, label, mark = load_dense_target(tname)
            feat = unify_features(feat_raw, norm, tname)
            sparse = adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E
            graph = prepare_tam_graph(adj, feat, device, sparse=sparse)
            gpu_ready = True
            print_stage(
                "tam",
                3,
                4,
                f"target ready {target_pos}/{target_total}: {tname} "
                f"N={adj.shape[0]} E={adj.nnz} "
                f"path={'sparse' if sparse else 'dense'} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"    [tam] target {tname}: preparation skipped ({str(e)[:100]})")
                continue
            print(f"    [tam] target {tname}: CUDA OOM during preparation; score on CPU")
            graph = None
            empty_device_cache(device)
            graph = prepare_tam_graph(adj, feat, "cpu", sparse=sparse)
            gpu_ready = False
        except Exception as e:
            print(f"    [tam] target {tname}: preparation skipped ({str(e)[:100]})")
            graph = adj = feat_raw = feat = label = mark = None
            empty_device_cache(device)
            continue

        ensemble_total = len(trained_ensembles)
        for seed_pos, (seed, models) in enumerate(trained_ensembles, 1):
            print_stage(
                "tam",
                4,
                4,
                f"scoring target={tname} ({target_pos}/{target_total}) "
                f"seed={seed} ({seed_pos}/{ensemble_total})",
            )
            try:
                eval_models = (
                    models if gpu_ready else [copy.deepcopy(model).cpu() for model in models]
                )
                scores = score_with_prepared_ensemble(
                    eval_models,
                    graph,
                    cutting=cfg["cutting"],
                    n_tree=cfg["n_tree"],
                    progress_label=(
                        f"stage 4/4 target={tname} "
                        f"({target_pos}/{target_total}) seed={seed} "
                        f"({seed_pos}/{ensemble_total})"
                    ),
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or not gpu_ready:
                    print(f"    [tam] target {tname} seed={seed} skipped ({str(e)[:100]})")
                    empty_device_cache(device)
                    continue
                print(f"    [tam] target {tname} seed={seed}: CUDA OOM, retry scoring on CPU")
                graph = eval_models = None
                empty_device_cache(device)
                gpu_ready = False
                try:
                    graph = prepare_tam_graph(adj, feat, "cpu", sparse=sparse)
                    eval_models = [copy.deepcopy(model).cpu() for model in models]
                    scores = score_with_prepared_ensemble(
                        eval_models,
                        graph,
                        cutting=cfg["cutting"],
                        n_tree=cfg["n_tree"],
                        progress_label=(
                            f"stage 4/4 target={tname} "
                            f"({target_pos}/{target_total}) seed={seed} "
                            f"({seed_pos}/{ensemble_total}) CPU"
                        ),
                    )
                except Exception as cpu_error:
                    print(
                        f"    [tam] target {tname} seed={seed}: CPU retry failed "
                        f"({str(cpu_error)[:100]})"
                    )
                    continue
            except Exception as e:
                print(f"    [tam] target {tname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)
                continue
            if not np.isfinite(scores).all():
                print(f"    [tam] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
            eval_models = None
            empty_device_cache(device)
        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])
        graph = None
        adj = feat_raw = feat = label = mark = None
        empty_device_cache(device)

    return aggregate(per)
