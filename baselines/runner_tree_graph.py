"""Source->target runners for RF-Graph and XGB-Graph."""

import time

import numpy as np

from baselines.config import RF_GRAPH_HP, RF_GRAPH_QUICK, XGB_GRAPH_HP, XGB_GRAPH_QUICK
from baselines.preprocess import unify_features
from baselines.protocol import load_dense_source, load_dense_target
from baselines.runtime import print_progress, print_stage, seed_for_device
from baselines.tree_graph import gin_noparam_features, make_rf, make_xgb, predict_proba_chunked
from common.data import aggregate
from util import evaluate


def _cfg(method, quick=False, hp=None):
    if method == "rf_graph":
        cfg = dict(RF_GRAPH_HP if hp is None else hp)
        if quick:
            cfg.update(RF_GRAPH_QUICK)
        return cfg
    if method == "xgb_graph":
        cfg = dict(XGB_GRAPH_HP if hp is None else hp)
        if quick:
            cfg.update(XGB_GRAPH_QUICK)
        return cfg
    raise ValueError(method)


def _make_model(method, cfg, seed):
    if method == "rf_graph":
        return make_rf(cfg, seed)
    if method == "xgb_graph":
        return make_xgb(cfg, seed)
    raise ValueError(method)


def _graph_features(adj, feat_raw, norm, name, cfg):
    feat = unify_features(feat_raw, norm, name)
    return gin_noparam_features(
        adj,
        feat,
        num_layers=cfg["num_layers"],
        agg=cfg["agg"],
    )


def _source_xy(method, sources, norm, cfg):
    xs, ys = [], []
    source_total = len(sources)
    for source_pos, sname in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage(method, 1, 4, f"building source features {source_pos}/{source_total}: {sname}")
        adj, feat_raw, label, mark = load_dense_source(sname)
        x = _graph_features(adj, feat_raw, norm, sname, cfg)
        idx = np.where(np.asarray(mark, dtype=bool))[0]
        xs.append(x[idx])
        ys.append(np.asarray(label, dtype=np.int64)[idx])
        print_stage(
            method,
            1,
            4,
            f"source features ready {source_pos}/{source_total}: {sname} "
            f"N={adj.shape[0]} E={adj.nnz} train={idx.size} "
            f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
        )
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def run_tree_graph(
    method, sources, targets, seeds, device, norm, quick=False, hp=None, progress=None
):
    # device is intentionally unused: RF/XGB run on CPU features in the official
    # implementation after GIN_noparam feature construction.
    del device
    per = {n: [] for n in targets}
    cfg = _cfg(method, quick=quick, hp=hp)
    if method == "xgb_graph":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            print("    [xgb_graph] unavailable: install with python -m pip install xgboost")
            result = aggregate(per)
            if progress is not None:
                for target in targets:
                    progress(target, result[target])
            return result
    if not seeds:
        return aggregate(per)
    # Graph feature propagation is deterministic and independent of the tree
    # seed.  Build it once instead of repeating all sparse multiplications 11x.
    seed_for_device(seeds[0], None)
    try:
        train_x, train_y = _source_xy(method, sources, norm, cfg)
    except Exception as e:
        print(f"    [{method}] source feature build skipped ({str(e)[:100]})")
        return aggregate(per)
    if train_x is None or train_y is None:
        print(f"    [{method}] no source graph trained; skip targets")
        return aggregate(per)
    pos = int((train_y == 1).sum())
    neg = int((train_y == 0).sum())
    if pos == 0 or neg == 0:
        print(f"    [{method}] source labels single-class; skip targets")
        return aggregate(per)
    sample_weight = np.where(train_y == 0, 1.0, neg / max(pos, 1)).astype(np.float32)

    fitted = []
    seed_total, target_total = len(seeds), len(targets)
    fit_started = time.perf_counter()
    for seed_pos, seed in enumerate(seeds, 1):
        seed_for_device(seed, None)
        print_stage(
            method,
            2,
            4,
            f"fitting seed={seed} ({seed_pos}/{seed_total}) "
            f"samples={train_x.shape[0]} features={train_x.shape[1]}",
        )
        try:
            model = _make_model(method, cfg, seed)
        except Exception as e:
            print(f"    [{method}] seed={seed}: model init skipped ({str(e)[:120]})")
            continue
        try:
            model.fit(train_x, train_y, sample_weight=sample_weight)
        except TypeError:
            # Some xgboost versions reject sample_weight for exotic boosters.
            model.fit(train_x, train_y)
        except Exception as e:
            print(f"    [{method}] seed={seed}: fit skipped ({str(e)[:120]})")
            continue
        fitted.append((int(seed), model))
        print_progress(method, "stage 2/4 fitted seeds", seed_pos, seed_total, fit_started)

    # Target graph-aware features are also seed-independent.  Keep the fitted
    # forests/boosters and score each target immediately after its one feature
    # build, so only one large target matrix is resident at a time.
    for target_pos, tname in enumerate(targets, 1):
        started = time.perf_counter()
        print_stage(method, 3, 4, f"building target features {target_pos}/{target_total}: {tname}")
        try:
            adj, feat_raw, label, mark = load_dense_target(tname)
            x = _graph_features(adj, feat_raw, norm, tname, cfg)
            print_stage(
                method,
                3,
                4,
                f"target features ready {target_pos}/{target_total}: {tname} "
                f"N={adj.shape[0]} E={adj.nnz} eval={int(np.asarray(mark).sum())} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as e:
            print(f"    [{method}] target {tname}: feature build skipped ({str(e)[:100]})")
            adj = feat_raw = label = mark = x = None
            continue
        fitted_total = len(fitted)
        score_started = time.perf_counter()
        for seed_pos, (seed, model) in enumerate(fitted, 1):
            print_stage(
                method,
                4,
                4,
                f"scoring target={tname} ({target_pos}/{target_total}) "
                f"seed={seed} ({seed_pos}/{fitted_total})",
            )
            try:
                scores = predict_proba_chunked(model, x, chunk=cfg["predict_chunk"])
            except Exception as e:
                print(f"    [{method}] target {tname} seed={seed} skipped ({str(e)[:100]})")
                continue
            if not np.isfinite(scores).all():
                print(f"    [{method}] target {tname} seed={seed}: NaN/Inf scores; skipped")
                continue
            per[tname].append(evaluate(np.asarray(label)[mark], scores[mark]))
            print_progress(
                method, f"stage 4/4 target={tname} seeds", seed_pos, fitted_total, score_started
            )
            if progress is not None:
                progress(tname, aggregate({tname: per[tname]})[tname])
        if progress is not None and not per[tname]:
            progress(tname, aggregate({tname: per[tname]})[tname])
        adj = feat_raw = label = mark = x = None
    return aggregate(per)
