"""Experiment execution for classical baselines.

All baselines use the source-to-target generalist-GAD protocol:

  raw features -> per-method SVD/normalization
  train on source graph(s) -> score target graph nodes -> evaluate mark nodes

Every method uses its frozen ``*_HP`` dictionary and its
entry in ``FEATURE_CONFIGS`` for both single-source and multi-source runs.
"""

import csv
import os
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from baselines import config as BC
from evaluation import metric_run, metric_path, metric_columns, primary_metric

from common.results import _build_delta, _merge

RESULTS = (
    Path(os.environ.get("GAD_OUTPUT_DIR", Path(__file__).resolve().parents[1] / BC.RESULTS_DIR))
    .expanduser()
    .resolve()
)


def _canonical(name):
    if str(name).lower() in ("dominant", "dominate"):
        return "dominant"
    if name in ("rf-graph", "RF-Graph", "RFGraph"):
        return "rf_graph"
    if name in ("xgb-graph", "XGB-Graph", "XGBGraph"):
        return "xgb_graph"
    return name


def _selected_config(method, mode, src_type=None, tgt_type=None):
    del mode, src_type, tgt_type
    feature = BC.FEATURE_CONFIGS[method]
    return str(feature["norm"]), dict(BC.method_hp(method))


def _run_baseline(name, sources, targets, device, seeds, quick, norm, hp=None, progress=None):
    if name == "cola":
        from baselines.runner_cola import run_cola

        return run_cola(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "tam":
        from baselines.runner_tam import run_tam

        return run_tam(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "bwgnn":
        from baselines.runner_bwgnn import run_bwgnn

        return run_bwgnn(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "ghrn":
        from baselines.runner_ghrn import run_ghrn

        return run_ghrn(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "gcn":
        from baselines.runner_gcn import run_gcn

        return run_gcn(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "gat":
        from baselines.runner_gat import run_gat

        return run_gat(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    if name == "mlp":
        from baselines.runner_supervised_gnn import run_supervised_gnn

        return run_supervised_gnn(
            "mlp", sources, targets, seeds, device, norm, quick, hp=hp, progress=progress
        )
    if name in ("rf_graph", "rf-graph", "RF-Graph", "RFGraph"):
        from baselines.runner_tree_graph import run_tree_graph

        return run_tree_graph(
            "rf_graph", sources, targets, seeds, device, norm, quick, hp=hp, progress=progress
        )
    if name in ("xgb_graph", "xgb-graph", "XGB-Graph", "XGBGraph"):
        from baselines.runner_tree_graph import run_tree_graph

        return run_tree_graph(
            "xgb_graph", sources, targets, seeds, device, norm, quick, hp=hp, progress=progress
        )
    if str(name).lower() in ("dominant", "dominate"):
        from baselines.runner_dominant import run_dominant

        return run_dominant(sources, targets, seeds, device, norm, quick, hp=hp, progress=progress)
    raise ValueError(f"unknown baseline {name}")


def run_baseline(
    name, sources, targets, device, seeds, quick, norm, hp=None, progress=None, feature_dim=None
):
    if feature_dim is None:
        return _run_baseline(
            name,
            sources,
            targets,
            device,
            seeds,
            quick,
            norm,
            hp=hp,
            progress=progress,
        )
    from baselines.preprocess import feature_dimension

    with feature_dimension(feature_dim):
        return _run_baseline(
            name,
            sources,
            targets,
            device,
            seeds,
            quick,
            norm,
            hp=hp,
            progress=progress,
        )


def _write_atomic(path, rows):
    """Write a complete CSV via same-directory atomic replacement."""
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _save_checkpoint(new_rows):
    """Merge results atomically, with an advisory lock where fcntl is available."""
    if not new_rows:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_path = RESULTS / ".source_compare.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            detail_path = metric_path(RESULTS / "source_compare_detail.csv")
            detail = _merge(detail_path, new_rows)
            _write_atomic(detail_path, detail)
            _write_atomic(metric_path(RESULTS / "source_compare_delta.csv"), _build_delta(detail))
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _result_row(method, src_type, tgt_type, mode, target, score):
    return {
        "method": method,
        "protocol": "zero_shot",
        "src_type": src_type,
        "tgt_type": tgt_type,
        "mode": mode,
        "target": target,
        **metric_columns(score),
        "n": score["n"],
    }


@metric_run(config=BC)
def run(args):
    """Execute options parsed and checked by the root run.py."""

    src_type = args.src_type or BC.SRC_TYPE
    tgt_type = args.tgt_type or BC.TGT_TYPE
    source_mode = args.source_mode or BC.SOURCE_MODE
    baselines = (
        [b.strip() for b in args.methods.split(",") if b.strip()]
        if args.methods
        else list(BC.BASELINES)
    )
    quick = args.quick or BC.QUICK
    device = args.device or BC.DEVICE
    modes = ["single", "multi"] if source_mode == "both" else [source_mode]

    all_targets = BC.targets(tgt_type)
    targets = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.targets
        else (all_targets[:2] if quick else all_targets)
    )
    seeds = [0] if quick else BC.SEEDS
    if args.seed is not None:
        seeds = [args.seed]
    direction = f"{src_type}2{tgt_type}"

    print(
        f"baselines [{direction}] source_mode={source_mode} baselines={baselines} "
        f"targets={len(targets)} device={device} quick={quick} "
        "features=per-method"
    )

    new_detail = []
    for name in baselines:
        method = _canonical(name)
        agnostic = method in BC.SOURCE_AGNOSTIC
        # DOMINATE is accepted only as a typo-tolerant CLI alias. Persist the
        # official method name DOMINANT in every result artifact.
        result_name = method if method == "dominant" else name
        method_out = result_name
        computed = {}
        for mode in modes:
            configured_norm, hp = _selected_config(method, mode, src_type, tgt_type)
            feature = BC.FEATURE_CONFIGS.get(
                method, {"dim": BC.BASELINE_SVD_DIM, "norm": configured_norm}
            )
            feature_dim = int(feature["dim"])
            norm = str(feature["norm"])
            srcs = BC.sources(mode, src_type)
            if agnostic and computed:
                res = next(iter(computed.values()))
            else:
                tag = "single-graph train/test; source-agnostic" if agnostic else f"sources={srcs}"
                tag += f" hp={hp}"
                print(
                    f"\n===== baseline {method_out} [{direction}/{mode}] "
                    f"svd{feature_dim}+{norm} ({tag}) ====="
                )

                def checkpoint_target(target, score, *, _method=method_out, _mode=mode):
                    _save_checkpoint(
                        [_result_row(_method, src_type, tgt_type, _mode, target, score)]
                    )

                res = run_baseline(
                    name,
                    srcs,
                    targets,
                    device,
                    seeds,
                    quick,
                    norm,
                    hp=hp,
                    progress=checkpoint_target,
                    feature_dim=feature_dim,
                )
            computed[mode] = res

            for tgt, sc in res.items():
                new_detail.append(_result_row(method_out, src_type, tgt_type, mode, tgt, sc))
            ok = [s[primary_metric() + "_mean"] for s in res.values() if not np.isnan(s[primary_metric() + "_mean"])]
            print(
                f"  [{mode}] mean {primary_metric()}={np.mean(ok):.4f} ({len(ok)}/{len(res)} success)"
                if ok
                else f"  [{mode}] all nan"
            )
            # Save completed results before starting the next method or mode.
            # On POSIX, the advisory lock serializes concurrent result writers.
            _save_checkpoint(new_detail)

    _save_checkpoint(new_detail)
    print(f"\nsaved -> {metric_path(RESULTS / 'source_compare_detail.csv')} "
          f"+ {metric_path(RESULTS / 'source_compare_delta.csv')}")
