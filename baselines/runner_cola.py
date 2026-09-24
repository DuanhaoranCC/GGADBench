"""CoLA under the source-to-target graph anomaly detection protocol."""

import time

import numpy as np

from baselines.cola import (
    init_model,
    prepare_graph,
    score_model_on_prepared_graph,
    train_model_on_prepared_graph,
)
from baselines.config import COLA_HP, COLA_QUICK
from baselines.preprocess import unify_features
from baselines.protocol import DENSE_SAFE_E, DENSE_SAFE_N, load_dense_source, load_dense_target
from baselines.runtime import empty_device_cache, seed_for_device, select_device
from common.data import aggregate
from util import evaluate


def run_cola(sources, targets, seeds, device, norm, quick=False, hp=None, progress=None):
    hp = dict(COLA_HP if hp is None else hp)
    per = {n: [] for n in targets}
    select_device(device)

    # Keep one tiny model/optimizer per seed, but prepare every source graph only
    # once.  This preserves each seed's source order and optimizer state while
    # avoiding 11 repeated T-Finance loads/normalizations.
    runs = {int(seed): {"model": None, "opt": None, "trained": False} for seed in seeds}

    source_total = len(sources)
    seed_total = len(seeds)
    target_total = len(targets)

    for source_pos, sname in enumerate(sources, 1):
        prep_started = time.perf_counter()
        print(
            f"    [cola-stage 1/4] preparing source {source_pos}/{source_total}: " f"{sname}",
            flush=True,
        )
        try:
            adj, feat_raw, _label, _mark = load_dense_source(sname)
            feat = unify_features(feat_raw, norm, sname)
            sparse_training = adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E
            graph = prepare_graph(adj, feat, device, make_dense=not sparse_training)
            print(
                f"    [cola-stage 1/4] source ready {source_pos}/{source_total}: "
                f"{sname} N={adj.shape[0]} E={adj.nnz} "
                f"path={'sparse' if sparse_training else 'dense'} "
                f"elapsed={(time.perf_counter() - prep_started) / 60:.1f}m",
                flush=True,
            )
        except Exception as e:
            print(f"    [cola] source {sname}: graph preparation skipped " f"({str(e)[:100]})")
            empty_device_cache(device)
            continue

        num_epoch = COLA_QUICK["num_epoch"] if quick else hp["num_epoch"]
        for seed_pos, seed in enumerate(seeds, 1):
            run = runs[int(seed)]
            try:
                if run["model"] is None:
                    # CoLA parameter initialization happens on CPU; seeding the
                    # selected CUDA device as well is harmless and keeps the
                    # helper consistent with the other baselines.
                    seed_for_device(seed, device)
                    run["model"], run["opt"] = init_model(
                        graph["ft"],
                        device,
                        embedding_dim=hp["embedding_dim"],
                        negsamp_ratio=hp["negsamp_ratio"],
                        readout=hp["readout"],
                        lr=hp["lr"],
                        weight_decay=hp["weight_decay"],
                    )
                train_model_on_prepared_graph(
                    run["model"],
                    run["opt"],
                    graph,
                    device,
                    num_epoch=num_epoch,
                    batch_size=hp["batch_size"],
                    subgraph_size=hp["subgraph_size"],
                    negsamp_ratio=hp["negsamp_ratio"],
                    seed=seed,
                    sparse_training=sparse_training,
                    progress_label=(
                        f"stage 2/4 source={sname} "
                        f"({source_pos}/{source_total}) seed={seed} "
                        f"({seed_pos}/{seed_total})"
                    ),
                )
                run["trained"] = True
            except Exception as e:
                print(f"    [cola] source {sname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)

        del graph, adj, feat_raw, feat, _label, _mark
        empty_device_cache(device)

    active_runs = []
    for seed in seeds:
        run = runs[int(seed)]
        if not run["trained"] or run["model"] is None:
            print(f"    [cola] seed={seed}: no source graph trained; skip targets")
            continue
        # Optimizer state is no longer needed after all sources are consumed.
        run["opt"] = None
        active_runs.append((int(seed), run["model"]))

    # Prepare one target at a time and reuse it for every seed.  This changes no
    # sampling or scoring order inside a seed, but removes 11 repeated target
    # loads, SVD normalization passes and full sparse adjacency normalizations.
    for target_pos, tname in enumerate(targets, 1):
        prep_started = time.perf_counter()
        print(
            f"    [cola-stage 3/4] preparing target " f"{target_pos}/{target_total}: {tname}",
            flush=True,
        )
        try:
            adj, feat_raw, label, mark = load_dense_target(tname)
            feat = unify_features(feat_raw, norm, tname)
            sparse_scoring = adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E
            graph = prepare_graph(adj, feat, device, make_dense=not sparse_scoring)
            score_idx = np.where(np.asarray(mark, dtype=bool))[0]
            print(
                f"    [cola-stage 3/4] target ready "
                f"{target_pos}/{target_total}: {tname} N={adj.shape[0]} "
                f"E={adj.nnz} eval={score_idx.size} "
                f"path={'sparse' if sparse_scoring else 'dense'} "
                f"elapsed={(time.perf_counter() - prep_started) / 60:.1f}m",
                flush=True,
            )
        except Exception as e:
            print(f"    [cola] target {tname}: graph preparation skipped " f"({str(e)[:100]})")
            empty_device_cache(device)
            continue

        rounds = COLA_QUICK["auc_test_rounds"] if quick else hp["auc_test_rounds"]
        active_total = len(active_runs)
        for seed_pos, (seed, model) in enumerate(active_runs, 1):
            try:
                scores = score_model_on_prepared_graph(
                    model,
                    graph,
                    device,
                    batch_size=hp["batch_size"],
                    subgraph_size=hp["subgraph_size"],
                    auc_test_rounds=rounds,
                    seed=seed,
                    score_idx=score_idx,
                    sparse_scoring=sparse_scoring,
                    progress_label=(
                        f"stage 4/4 target={tname} "
                        f"({target_pos}/{target_total}) seed={seed} "
                        f"({seed_pos}/{active_total})"
                    ),
                )
            except Exception as e:
                print(f"    [cola] target {tname} seed={seed} skipped ({str(e)[:100]})")
                empty_device_cache(device)
                continue
            if not np.isfinite(scores).all():
                print(f"    [cola] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
            empty_device_cache(device)

        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])

        del graph, adj, feat_raw, feat, label, mark, score_idx
        empty_device_cache(device)

    return aggregate(per)
