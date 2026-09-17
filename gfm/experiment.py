"""Run source-to-target graph foundation and in-context models.

Invoked by the root run.py with --suite gfm. Settings live in config.py.
"""

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

import gfm.config as C
from common.results import _build_delta, _merge, _write

RESULTS = Path(os.environ.get("GAD_OUTPUT_DIR", Path(__file__).resolve().parent / "results"))


def run_method(method, sources, targets, device, seeds, quick, feature_dim=None, feature_norm=None):
    feature_dim_override = feature_dim
    feature = C.FEATURE_CONFIGS.get(method, {"dim": C.FEATURE_DIM, "norm": C.FEATURE_NORM})
    feature_dim = int(feature["dim"] if feature_dim is None else feature_dim)
    feature_norm = str(feature["norm"] if feature_norm is None else feature_norm)
    if method in ("graphprompt", "graphprompt_enr"):
        from gfm.runners.graphprompt import run_graphprompt

        hp = dict(C.GRAPHPROMPT_HP)
        if quick:
            hp["epochs"] = hp.get("quick_epochs", 5)
            hp["eval_query_batch"] = min(int(hp.get("eval_query_batch", 200_000)), 10_000)
        return run_graphprompt(
            sources,
            targets,
            seeds,
            hp,
            device,
            shot=C.SHOT,
            feature_dim=feature_dim,
            feature_norm=feature_norm,
            encoder_type="ego_neighbor_residual" if method.endswith("_enr") else "native",
            enr_hp=dict(C.ENR_ENCODER_HP),
        )
    if method in ("samgpt", "samgpt_diff", "samgpt_enr"):
        from gfm.runners.samgpt import run_samgpt

        hp = dict(C.SAMGPT_HP)
        if quick:
            hp["pretrain_epochs"] = hp.get("quick_pretrain_epochs", 3)
            hp["downstream_steps"] = hp.get("quick_downstream_steps", 5)
            hp["eval_query_batch"] = min(int(hp.get("eval_query_batch", 200_000)), 10_000)
        return run_samgpt(
            sources,
            targets,
            seeds,
            hp,
            device,
            shot=C.SHOT,
            feature_dim=feature_dim,
            feature_norm=feature_norm,
            encoder_type="ego_neighbor_residual" if method.endswith("_enr") else "native",
            enr_hp=dict(C.ENR_ENCODER_HP),
            downstream_type=("ego_neighbor_difference" if method == "samgpt_diff" else "prototype"),
        )
    if method == "bridge":
        from gfm.runners.bridge import run_bridge

        hp = dict(C.BRIDGE_HP)
        if quick:
            hp["pretrain_epochs"] = hp.get("quick_pretrain_epochs", 3)
            hp["downstream_steps"] = hp.get("quick_downstream_steps", 5)
            hp["large_downstream_steps"] = hp.get("quick_downstream_steps", 5)
            hp["eval_query_batch"] = min(int(hp.get("eval_query_batch", 200_000)), 10_000)
        return run_bridge(
            sources,
            targets,
            seeds,
            hp,
            device,
            shot=C.SHOT,
            feature_dim=feature_dim,
            feature_norm=feature_norm,
        )
    if method == "mdgfm":
        from gfm.runners.mdgfm import run_mdgfm

        hp = dict(C.MDGFM_HP)
        if quick:
            hp["pretrain_epochs"] = hp.get("quick_pretrain_epochs", 3)
            hp["downstream_steps"] = hp.get("quick_downstream_steps", 5)
            hp["eval_query_batch"] = min(int(hp.get("eval_query_batch", 200_000)), 10_000)
        return run_mdgfm(
            sources,
            targets,
            seeds,
            hp,
            device,
            shot=C.SHOT,
            feature_dim=feature_dim,
            feature_norm=feature_norm,
        )
    if method == "mdgpt":
        from gfm.runners.mdgpt import run_mdgpt

        if feature_norm != "none":
            raise ValueError("MDGPT uses its own uncentered SVD without feature normalization")
        hp = dict(C.MDGPT_HP)
        # MDGPT uses its own feature adapter and configured SVD width.
        if feature_dim_override is not None:
            hp["feature_dim"] = int(feature_dim_override)
        epochs = C.MDGPT_QUICK_EPOCHS if quick else C.MDGPT_EPOCHS
        prompt_epochs = C.MDGPT_QUICK_PROMPT_EPOCHS if quick else C.MDGPT_PROMPT_EPOCHS
        return run_mdgpt(
            sources, targets, seeds, epochs, device, shot=C.SHOT, hp=hp, prompt_epochs=prompt_epochs
        )
    raise ValueError(f"Unknown foundation method: {method}")


def run(args):
    """Execute options parsed and checked by the root run.py."""

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
    modes = ["single", "multi"] if source_mode == "both" else [source_mode]
    all_targets = C.targets(tgt_type)
    targets = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.targets
        else (all_targets[:2] if quick else all_targets)
    )
    seeds = [0] if quick else list(C.SEEDS)
    if args.seed is not None:
        seeds = [args.seed]
    direction = f"{src_type}2{tgt_type}"

    print(
        f"Foundation [{direction}] source_mode={source_mode} methods={methods} device={device} quick={quick}"
    )
    print(
        f"  protocol=few_shot  native=per-method features  "
        f"enr_variant={C.ENR_MODE}  shot={C.SHOT}"
    )
    if "samgpt_diff" in methods:
        print(f"  samgpt_diff={C.SAMGPT_DIFF_MODE}")
    print(f"  targets={targets}")
    print(f"  seeds={seeds} repeats={len(seeds)}")

    new_detail = []
    for method in methods:
        proto = C.METHOD_PROTOCOL[method]
        for mode in modes:
            srcs = C.sources(mode, src_type)
            feature = C.FEATURE_CONFIGS.get(method, {"dim": C.FEATURE_DIM, "norm": C.FEATURE_NORM})
            print(
                f"\n===== {method} [{direction}/{mode}] sources={srcs} "
                f"svd{feature['dim']}+{feature['norm']} ====="
            )
            res = run_method(method, srcs, targets, device, seeds, quick)
            torch.cuda.empty_cache()
            for tgt, sc in res.items():
                new_detail.append(
                    {
                        "method": method,
                        "protocol": proto,
                        "src_type": src_type,
                        "tgt_type": tgt_type,
                        "mode": mode,
                        "target": tgt,
                        "AUROC_mean": round(sc["AUROC_mean"], 4),
                        "AUROC_std": round(sc["AUROC_std"], 4),
                        "AUPRC_mean": round(sc["AUPRC_mean"], 4),
                        "AUPRC_std": round(sc["AUPRC_std"], 4),
                        "n": sc["n"],
                    }
                )
            vals = [s["AUROC_mean"] for s in res.values() if not np.isnan(s["AUROC_mean"])]
            print(
                f"  [{mode}] mean AUROC={np.mean(vals):.4f}"
                if vals
                else f"  [{mode}] mean AUROC=nan"
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    detail = _merge(RESULTS / "foundation_compare_detail.csv", new_detail)
    _write(RESULTS / "foundation_compare_detail.csv", detail)
    _write(RESULTS / "foundation_compare_delta.csv", _build_delta(detail))
    print(f"\nsaved -> {RESULTS}\\foundation_compare_detail.csv + _delta.csv")
