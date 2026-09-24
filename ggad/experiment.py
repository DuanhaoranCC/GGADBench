"""Run source-to-target graph anomaly detection across domains and source counts."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import gc
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

import ggad.config as C
from evaluation import metric_run, metric_path, metric_columns, primary_metric

from common.results import _build_delta, _merge, _write

RESULTS = (
    Path(os.environ.get("GAD_OUTPUT_DIR", Path(__file__).resolve().parent / "results"))
    .expanduser()
    .resolve()
)


def _result_paths():
    names = (C.RESULTS_DETAIL_FILE, C.RESULTS_DELTA_FILE)
    for name in names:
        if (
            not isinstance(name, str)
            or not name.strip()
            or any(char in name for char in "/\\:")
            or Path(name).suffix.lower() != ".csv"
        ):
            raise ValueError(
                "RESULTS_DETAIL_FILE / RESULTS_DELTA_FILE must be .csv filenames without directories"
            )
    if names[0].casefold() == names[1].casefold():
        raise ValueError("Detail and delta results must use different filenames")
    return tuple(metric_path(RESULTS / name) for name in names)


# METHOD_PROTOCOL also doubles as the default method selection in config.py.
# CLI overrides must still be able to run any implemented method even when it
# is not part of that current default selection.
SUPPORTED_PROTOCOLS = {
    "unprompt": "zero_shot",
    "anomalygfm_zs": "zero_shot",
    "iaggad_zs": "zero_shot",
    "drggad": "zero_shot",
    "gadmore": "zero_shot",
    "neighbordiv": "zero_shot",
    "promos": "zero_shot",
    "zerogad": "zero_shot",
    "owleye": "zero_shot",
    "tpcagad": "zero_shot",
    "saarcs": "few_shot",
    "tfm4gad": "few_shot",
    "arc": "few_shot",
    "anomalygfm_fs": "few_shot",
    "iaggad_fs": "few_shot",
    "taggad": "few_shot_oracle_count",
    "refigad": "few_shot",
}


def run_method(
    method,
    sources,
    targets,
    device,
    ep,
    seeds,
    shot=None,
    disjoint_query=False,
    target_evaluator=None,
):
    """Dispatch one run; an explicit shot changes target support only.

    Calls without a shot override use the configured support/query protocol. The shot
    sweep uses ``disjoint_query`` to exclude REFIGAD support from its metrics.
    Zero-shot methods can supply their unchanged full target scores to a
    query-selection callback through ``target_evaluator``.
    """
    k = C.SHOT if shot is None else int(shot)
    evaluator_kwargs = (
        {"target_evaluator": target_evaluator} if target_evaluator is not None else {}
    )
    if method == "arc":
        from ggad.runners.arc import run_arc

        return run_arc(sources, targets, seeds, ep["arc"], device, k)
    if method in ("iaggad_zs", "iaggad_fs"):
        from ggad.runners.iaggad import run_iaggad

        st = "normal" if method == "iaggad_fs" else "random"
        return run_iaggad(sources, targets, st, seeds, ep["iaggad"], device, k)
    if method == "taggad":
        from ggad.runners.taggad import run_taggad

        return run_taggad(
            sources,
            targets,
            seeds,
            ep["taggad"],
            device,
            k,
            target_lambdas=C.TAGGAD_HP["target_lambdas"],
            source_shot=C.SHOT if shot is not None else None,
        )
    if method == "unprompt":
        from ggad.runners.unprompt import run_unprompt

        return run_unprompt(
            sources,
            targets,
            seeds,
            device,
            ep["unprompt"],
            ep["grace"],
            **evaluator_kwargs,
        )
    if method in ("anomalygfm_zs", "anomalygfm_fs"):
        from ggad.runners.anomalygfm import run_anomalygfm

        st = "normal" if method == "anomalygfm_fs" else "none"
        emb = 300 if method == "anomalygfm_fs" else 400
        return run_anomalygfm(
            sources,
            targets,
            st,
            seeds,
            device,
            ep[method],
            emb,
            k,
            **(evaluator_kwargs if method == "anomalygfm_zs" else {}),
        )
    if method == "drggad":
        from ggad.runners.drggad import run_drggad

        return run_drggad(sources, targets, seeds, ep["drggad"], device, **evaluator_kwargs)
    if method == "gadmore":
        from ggad.runners.gadmore import run_gadmore

        return run_gadmore(sources, targets, seeds, ep["gadmore"], device, **evaluator_kwargs)
    if method == "neighbordiv":
        from ggad.runners.neighbordiv import run_neighbordiv

        return run_neighbordiv(
            sources, targets, seeds, ep["neighbordiv"], device, **evaluator_kwargs
        )
    if method == "promos":
        from ggad.runners.promos import run_promos

        return run_promos(sources, targets, seeds, ep["promos"], device, **evaluator_kwargs)
    if method == "zerogad":
        from ggad.runners.zerogad import run_zerogad

        return run_zerogad(sources, targets, seeds, ep["zerogad"], device, **evaluator_kwargs)
    if method == "owleye":
        from ggad.runners.owleye import run_owleye

        return run_owleye(sources, targets, seeds, ep["owleye"], device, **evaluator_kwargs)
    if method == "tpcagad":
        from ggad.runners.tpcagad import run_tpcagad

        return run_tpcagad(sources, targets, seeds, ep["tpcagad"], device, **evaluator_kwargs)
    if method == "saarcs":
        from ggad.runners.saarcs import run_saarcs

        return run_saarcs(sources, targets, seeds, ep["saarcs"], device, k)
    if method == "tfm4gad":
        from ggad.runners.tfm4gad import run_tfm4gad

        return run_tfm4gad(sources, targets, seeds, ep["tfm4gad"], device, k)
    if method == "refigad":
        from ggad.runners.refigad import run_refigad

        return run_refigad(
            sources,
            targets,
            seeds,
            device,
            ep["refigad"],
            shot=shot,
            exclude_support=disjoint_query,
        )
    raise ValueError(f"Unknown method: {method}")


@metric_run(config=C)
def run(args):
    """Execute options parsed and checked by the root run.py."""
    detail_path, delta_path = _result_paths()

    src_type = args.src_type or C.SRC_TYPE
    tgt_type = args.tgt_type or C.TGT_TYPE
    source_mode = args.source_mode or C.SOURCE_MODE
    methods = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods
        else list(C.METHODS)
    )
    quick = args.quick or C.QUICK
    device = args.device or C.DEVICE
    os.environ["GAD_BENCHMARK_DEVICE"] = str(device)
    if str(device).startswith("cuda"):
        # All runners are single-device. Selecting it once also ensures the
        # shared set_seed() touches this GPU instead of CUDA's default GPU 0.
        torch.cuda.set_device(torch.device(device))
    modes = ["single", "multi"] if source_mode == "both" else [source_mode]
    ep = C.QUICK_EPOCHS if quick else C.EPOCHS
    all_targets = C.targets(tgt_type)
    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = all_targets[:2] if quick else all_targets
    seeds = [0] if quick else C.SEEDS
    if args.seed is not None:
        seeds = [args.seed]
    direction = f"{src_type}2{tgt_type}"

    print(
        f"Source transfer [{direction}]  source_mode={source_mode}  methods={methods}  device={device}  quick={quick}"
    )
    print(
        f"  sources({src_type}) single={C.sources('single', src_type)}  multi={C.sources('multi', src_type)}"
    )
    print(f"  targets({tgt_type})={len(targets)}  modes={modes}")

    print(f"  seeds={seeds}  repeats={len(seeds)}")
    print(f"  result files: {detail_path}\n                {delta_path}")

    new_detail = []
    for method in methods:
        if method not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unknown method: {method}")
        proto = C.METHOD_PROTOCOL.get(method, SUPPORTED_PROTOCOLS[method])
        source_independent_result = None
        for mode in modes:
            srcs = C.sources(mode, src_type)
            print(f"\n===== {method} [{direction}/{mode}] sources={srcs} =====")
            if method == "tfm4gad" and source_independent_result is not None:
                print("  [tfm4gad] source-independent; reusing the completed target result")
                res = {name: dict(score) for name, score in source_independent_result.items()}
            else:
                res = run_method(method, srcs, targets, device, ep, seeds)
                if method == "tfm4gad":
                    source_independent_result = {name: dict(score) for name, score in res.items()}
            for tgt, sc in res.items():
                new_detail.append(
                    {
                        "method": method,
                        "protocol": proto,
                        "src_type": src_type,
                        "tgt_type": tgt_type,
                        "mode": mode,
                        "target": tgt,
                        **metric_columns(sc),
                        "n": sc["n"],
                    }
                )
            print(f"  [{mode}] mean {primary_metric()}={np.mean([s[primary_metric() + '_mean'] for s in res.values()]):.4f}")
            del res
            # Persist every completed method/mode immediately. A later OOM in
            # another mode or method must not discard hours of finished runs.
            RESULTS.mkdir(parents=True, exist_ok=True)
            completed_detail = _merge(detail_path, new_detail)
            _write(detail_path, completed_detail)
            _write(
                delta_path,
                _build_delta(completed_detail),
            )
            gc.collect()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass

    RESULTS.mkdir(parents=True, exist_ok=True)
    detail = _merge(detail_path, new_detail)
    _write(detail_path, detail)
    delta = _build_delta(detail)
    _write(delta_path, delta)

    def _ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    print(
        f"\n===== [{direction}] Single-to-multi delta (targets successful in both modes only) ====="
    )
    for method in methods:
        rows = [
            r
            for r in delta
            if r["method"] == method and r["src_type"] == src_type and r["tgt_type"] == tgt_type
        ]
        pairs = [
            (r["single_" + primary_metric()], r["multi_" + primary_metric()])
            for r in rows
            if _ok(r["single_" + primary_metric()]) and _ok(r["multi_" + primary_metric()])
        ]
        if pairs:
            ms = np.mean([p[0] for p in pairs])
            mm = np.mean([p[1] for p in pairs])
            drop = len(rows) - len(pairs)
            tag = f"  ({drop} failed or missing targets excluded)" if drop else ""
            print(
                f"  {method:14s} single={ms:.4f}  multi={mm:.4f}  Δ={mm - ms:+.4f}  n={len(pairs)}/{len(rows)}{tag}"
            )
        else:
            print(f"  {method:14s} (no targets successful in both single and multi modes)")
    print(f"\nsaved -> {detail_path}\n         {delta_path}")
