"""Target shot sensitivity. Edit SHOT_SWEEP_* in config.py; python run.py --suite ggad --shot-sweep.

Uses the existing runners, full graphs and marked queries. A shot denotes
distinct labeled nodes, with at least one query left in each class. Legacy
Results are written to dedicated shot-sweep files.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]

import numpy as np

import ggad.config as C
from ggad.experiment import SUPPORTED_PROTOCOLS, run_method

RESULTS = (
    Path(os.environ.get("GAD_OUTPUT_DIR", Path(__file__).resolve().parent / "results"))
    .expanduser()
    .resolve()
)
# This describes the existing runners, not a replacement support protocol.
SUPPORT = {
    "arc": "normal_only",
    "iaggad_fs": "normal_only",
    "anomalygfm_fs": "normal_only",
    "saarcs": "normal_only",
    "taggad": "normal_only",
    "refigad": "per_class",
    "tfm4gad": "per_class",
}
HOLDOUT_METHODS = {
    "unprompt",
    "drggad",
    "gadmore",
    "neighbordiv",
    "promos",
    "zerogad",
    "owleye",
    "tpcagad",
    "anomalygfm_zs",
}
RANDOM_METHODS = HOLDOUT_METHODS | {"iaggad_zs"}
SUPPORT.update({method: "random_unlabeled" for method in RANDOM_METHODS})
PROTOCOLS = {
    method: "zero_shot_query_holdout" if method in HOLDOUT_METHODS else SUPPORTED_PROTOCOLS[method]
    for method in SUPPORT
}
FIELDS = [
    "context_id",
    "method",
    "protocol",
    "support_policy",
    "src_type",
    "tgt_type",
    "mode",
    "shot",
    "seed",
    "target",
    "status",
    "reason",
    "support_normal",
    "support_anomaly",
    "query_normal",
    "query_anomaly",
    "AUROC",
    "AUPRC",
]


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _key(row):
    return (row["context_id"], int(row["shot"]), int(row["seed"]), row["target"])


def _write_csv(path, rows, fields):
    """Atomic replacement keeps the previous checkpoint intact on interruption."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _label_counts(name):
    # Same label/mark definitions as load_target_marked, without loading edges
    # or node features just to check support feasibility on million-node graphs.
    from ggad.runners.taggad import _target_label_metadata

    labels, mark = _target_label_metadata(name)
    labels, mark = np.asarray(labels).reshape(-1), np.asarray(mark, dtype=bool).reshape(-1)
    if labels.shape != mark.shape or not np.isin(labels[mark], [0, 1]).all():
        raise ValueError(f"{name}: marked labels must be binary and aligned")
    return (
        int(np.count_nonzero(mark & (labels == 0))),
        int(np.count_nonzero(mark & (labels == 1))),
    )


def _budget(method, shot, counts):
    normal, anomaly = counts
    if method in RANDOM_METHODS:
        reason = "" if normal + anomaly > shot else "shot must leave at least one marked query"
        return {
            "support_normal": "",
            "support_anomaly": "",
            "query_normal": "",
            "query_anomaly": "",
        }, reason
    support_anomaly = shot if SUPPORT[method] == "per_class" else 0
    reason = ""
    if normal <= shot or anomaly <= support_anomaly:
        reason = (
            f"need {shot} distinct normal + {support_anomaly} distinct anomaly "
            f"support and >=1 query/class; available={normal}+{anomaly}"
        )
    return {
        "support_normal": shot,
        "support_anomaly": support_anomaly,
        "query_normal": normal - shot,
        "query_anomaly": anomaly - support_anomaly,
    }, reason


def _random_budgets(name, shots, seeds):
    """Record the actual class counts AFTER label-blind support selection."""
    from ggad.runners.taggad import _target_label_metadata

    labels, mark = _target_label_metadata(name)
    labels, mark = np.asarray(labels).reshape(-1), np.asarray(mark, dtype=bool).reshape(-1)
    normal, anomaly = int(np.count_nonzero(mark & (labels == 0))), int(
        np.count_nonzero(mark & (labels == 1))
    )
    result = {}
    for seed in seeds:
        pool = np.flatnonzero(mark)
        np.random.RandomState(seed).shuffle(pool)
        for shot in shots:
            if shot >= len(pool):
                result[(shot, seed)] = ({}, "shot must be smaller than marked node count")
                continue
            sn = int(np.count_nonzero(labels[pool[:shot]] == 0))
            sa = shot - sn
            budget = dict(
                support_normal=sn,
                support_anomaly=sa,
                query_normal=normal - sn,
                query_anomaly=anomaly - sa,
            )
            reason = (
                "query lacks both classes after random support selection"
                if normal == sn or anomaly == sa
                else ""
            )
            result[(shot, seed)] = (budget, reason)
    return result


def _row_budget(plan, shot, seed, target):
    if plan["method"] in RANDOM_METHODS:
        return plan["random_budgets"][target][(shot, seed)]
    return _budget(plan["method"], shot, plan["counts"][target])


def _settings(args=None):
    methods = args.methods.split(",") if args else list(C.SHOT_SWEEP_METHODS)
    shots = list(C.SHOT_SWEEP_SHOTS)
    src_type = args.src_type if args else C.SHOT_SWEEP_SRC_TYPE
    tgt_type = args.tgt_type if args else C.SHOT_SWEEP_TGT_TYPE
    source_mode = args.source_mode if args else C.SHOT_SWEEP_SOURCE_MODE
    seeds = list(C.SHOT_SWEEP_SEEDS)
    if args and args.seed is not None:
        seeds = [args.seed]
    for name, values in (("METHODS", methods), ("SHOTS", shots), ("SEEDS", seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"SHOT_SWEEP_{name} must be nonempty and contain no duplicates")
    if any(m not in SUPPORT for m in methods):
        raise ValueError(f"SHOT_SWEEP_METHODS supports only: {list(SUPPORT)}")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in shots):
        raise ValueError(
            "SHOT_SWEEP_SHOTS must contain positive integers; 0-shot is a different protocol"
        )
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("SHOT_SWEEP_SEEDS must contain nonnegative integers")
    if src_type not in ("real", "fake") or tgt_type not in ("real", "fake"):
        raise ValueError("SHOT_SWEEP_SRC_TYPE/TGT_TYPE must be real or fake")
    if source_mode not in ("single", "multi", "both"):
        raise ValueError("SHOT_SWEEP_SOURCE_MODE must be single, multi or both")
    stem = C.SHOT_SWEEP_RESULTS_STEM
    if not stem or Path(stem).name != stem or any(c in stem for c in "/\\:"):
        raise ValueError("SHOT_SWEEP_RESULTS_STEM must be a plain file stem")
    if stem in {"source_compare", "source_compare_detail1", "source_compare1"}:
        raise ValueError("Choose a dedicated shot experiment result stem")
    targets = list(C.targets(tgt_type) if C.SHOT_SWEEP_TARGETS is None else C.SHOT_SWEEP_TARGETS)
    if args:
        targets = args.targets.split(",")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("SHOT_SWEEP_TARGETS must be nonempty and contain no duplicates")
    modes = ["single", "multi"] if source_mode == "both" else [source_mode]
    quick = args.quick if args else bool(C.SHOT_SWEEP_QUICK)
    return dict(
        methods=methods,
        shots=sorted(shots),
        seeds=seeds[:1] if quick else seeds,
        modes=modes,
        targets=targets,
        quick=quick,
        stem=stem,
        epochs=dict(C.QUICK_EPOCHS if quick else C.SHOT_SWEEP_EPOCHS),
        device=args.device if args else C.SHOT_SWEEP_DEVICE,
        src_type=src_type,
        tgt_type=tgt_type,
    )


def _code_digest():
    """Conservative resume guard: changed runner/vendor code starts new contexts."""
    paths = {
        Path(__file__).resolve(),
        ROOT / "ggad/shot_query.py",
        ROOT / "ggad/experiment.py",
        ROOT / "util.py",
        ROOT / "common/data.py",
    }
    for directory in ("ggad/runners", "ggad/vendor"):
        paths.update((ROOT / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _plans(settings):
    mode_inputs = {}
    for mode in settings["modes"]:
        sources = list(C.sources(mode, settings["src_type"]))
        targets = [
            t for t in settings["targets"] if not C.SHOT_SWEEP_EXCLUDE_SOURCES or t not in sources
        ]
        if settings["quick"]:
            targets = targets[:2]
        if not sources or not targets:
            raise ValueError(f"{mode}: empty sources or targets after source exclusion")
        mode_inputs[mode] = (sources, targets)
    selected_targets = dict.fromkeys(t for _, targets in mode_inputs.values() for t in targets)
    counts = {name: _label_counts(name) for name in selected_targets}
    random_budgets = (
        {
            name: _random_budgets(name, settings["shots"], settings["seeds"])
            for name in selected_targets
        }
        if any(m in RANDOM_METHODS for m in settings["methods"])
        else {}
    )
    hp = {name: value for name, value in vars(C).items() if name.endswith("_HP")}
    hp.update(SOURCE_SHOT=C.SHOT, REFIGAD_K=C.REFIGAD_K, REFIGAD_K_DEFAULT=C.REFIGAD_K_DEFAULT)
    code = _code_digest()
    plans = []
    for method in settings["methods"]:
        for mode in settings["modes"]:
            sources, targets = mode_inputs[mode]
            context = dict(
                version="target-shot-v1-distinct-disjoint",
                method=method,
                mode=mode,
                src_type=settings["src_type"],
                tgt_type=settings["tgt_type"],
                sources=sources,
                targets=targets,
                epochs=settings["epochs"],
                hp=hp,
                device=str(settings["device"]),
                code_digest=code,
                label_counts={t: counts[t] for t in targets},
            )
            if method == "taggad":
                # Source RNG depends on this list. Include it in the resume
                # context when extending the sweep changes the feasible cohort.
                context["training_targets"] = [
                    t for t in targets if not _budget(method, max(settings["shots"]), counts[t])[1]
                ]
            context_id = hashlib.sha256(_json(context).encode()).hexdigest()[:16]
            plans.append(
                dict(
                    context_id=context_id,
                    context=context,
                    method=method,
                    mode=mode,
                    sources=sources,
                    targets=targets,
                    counts=counts,
                    random_budgets=random_budgets,
                )
            )
    return plans


def _base_row(plan, settings, shot, seed, target):
    budget, reason = _row_budget(plan, shot, seed, target)
    row = {
        "context_id": plan["context_id"],
        "method": plan["method"],
        "protocol": PROTOCOLS[plan["method"]],
        "support_policy": SUPPORT[plan["method"]],
        "src_type": settings["src_type"],
        "tgt_type": settings["tgt_type"],
        "mode": plan["mode"],
        "shot": shot,
        "seed": seed,
        "target": target,
        "status": "skipped" if reason else "pending",
        "reason": reason,
        **budget,
        "AUROC": "",
        "AUPRC": "",
    }
    if reason:
        # No support/query split was made for this infeasible combination.
        for field in budget:
            row[field] = ""
    return row


def _reports(rows, plans, settings):
    """Per-graph stats plus macro curves on a fixed common target cohort.

    Macro standard deviation is over seeds after averaging targets within each
    seed. A failed or infeasible graph is never silently dropped at just one k.
    """
    detail, summary = [], []
    for plan in plans:
        cid, shots, seeds = plan["context_id"], settings["shots"], settings["seeds"]

        def get(shot, seed, target):
            return rows.get((cid, shot, seed, target), {})

        common = [
            t
            for t in plan["targets"]
            if all(get(k, s, t).get("status") == "ok" for k in shots for s in seeds)
        ]
        for shot in shots:
            base = {
                "context_id": cid,
                "method": plan["method"],
                "protocol": PROTOCOLS[plan["method"]],
                "support_policy": SUPPORT[plan["method"]],
                "src_type": settings["src_type"],
                "tgt_type": settings["tgt_type"],
                "mode": plan["mode"],
                "shot": shot,
            }
            for target in plan["targets"]:
                group = [get(shot, s, target) for s in seeds]
                ok = [r for r in group if r.get("status") == "ok"]
                result = dict(
                    base,
                    target=target,
                    n=len(ok),
                    expected_n=len(seeds),
                    complete=len(ok) == len(seeds),
                )
                for metric in ("AUROC", "AUPRC"):
                    values = [float(r[metric]) for r in ok]
                    result[metric + "_mean"] = float(np.mean(values)) if values else ""
                    result[metric + "_std"] = float(np.std(values)) if values else ""
                result["status"] = (
                    "ok"
                    if len(ok) == len(seeds)
                    else (
                        "skipped"
                        if all(r.get("status") == "skipped" for r in group)
                        else "incomplete"
                    )
                )
                detail.append(result)
            result = dict(
                base,
                n_targets=len(common),
                targets_json=_json(common),
                n_seeds=len(seeds) if common else 0,
                seeds_json=_json(seeds),
                shots_json=_json(shots),
            )
            for metric in ("AUROC", "AUPRC"):
                values = (
                    [np.mean([float(get(shot, s, t)[metric]) for t in common]) for s in seeds]
                    if common
                    else []
                )
                result[metric + "_mean"] = float(np.mean(values)) if values else ""
                result[metric + "_std"] = float(np.std(values)) if values else ""
            summary.append(result)
    return detail, summary


def _save(rows, plans, settings, paths):
    _write_csv(paths["seeds"], list(rows.values()), FIELDS)
    detail, summary = _reports(rows, plans, settings)
    _write_csv(paths["detail"], detail, list(detail[0]))
    _write_csv(paths["summary"], summary, list(summary[0]))


def _run(settings, plans, paths):
    import torch

    os.environ["GAD_BENCHMARK_DEVICE"] = str(settings["device"])
    if str(settings["device"]).startswith("cuda"):
        torch.cuda.set_device(torch.device(settings["device"]))
    rows = {}
    if paths["seeds"].exists():
        with paths["seeds"].open(encoding="utf-8-sig", newline="") as handle:
            rows = {_key(row): row for row in csv.DictReader(handle)}
    manifest_path = paths["contexts"]
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.update({p["context_id"]: p["context"] for p in plans})
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(_json(manifest), encoding="utf-8")
    os.replace(str(temporary), str(manifest_path))
    for plan in plans:
        for shot in settings["shots"]:
            # Preserve target-list RNG order on retries: invoke ALL feasible
            # targets even if only one result in this seed group is missing.
            cohort = list(plan["targets"])
            if plan["method"] == "taggad":
                # TA-GGAD samples target masks before training from one Python
                # RNG stream. Keep its target list fixed over k as well.
                cohort = [t for t in cohort if t in plan["context"]["training_targets"]]
            for seed in settings["seeds"]:
                feasible = [t for t in cohort if not _row_budget(plan, shot, seed, t)[1]]
                pending = []
                for target in plan["targets"]:
                    row = _base_row(plan, settings, shot, seed, target)
                    if target not in feasible and row["status"] != "skipped":
                        row.update(
                            status="skipped",
                            reason="TA-GGAD fixed target cohort: insufficient normals at max shot",
                        )
                        for field in (
                            "support_normal",
                            "support_anomaly",
                            "query_normal",
                            "query_anomaly",
                        ):
                            row[field] = ""
                    key = _key(row)
                    if row["status"] == "skipped":
                        rows[key] = row
                        print(
                            f"  [skip] {plan['method']} {target} k={shot} seed={seed}: {row['reason']}"
                        )
                    elif not C.SHOT_SWEEP_RESUME or rows.get(key, {}).get("status") != "ok":
                        pending.append(target)
                        rows[key] = row
                _save(rows, plans, settings, paths)
                if not pending:
                    print(f"  [resume/done] {plan['method']}/{plan['mode']} k={shot} seed={seed}")
                    continue
                print(
                    f"\n===== {plan['method']}/{plan['mode']} shot={shot} seed={seed} "
                    f"sources={plan['sources']} targets={feasible} =====",
                    flush=True,
                )
                try:
                    extra = {}
                    if plan["method"] in HOLDOUT_METHODS:
                        from ggad.shot_query import QueryHoldout

                        extra["target_evaluator"] = QueryHoldout(shot)
                    result = run_method(
                        plan["method"],
                        plan["sources"],
                        feasible,
                        settings["device"],
                        settings["epochs"],
                        [seed],
                        shot=shot,
                        disjoint_query=True,
                        **extra,
                    )
                    for target in feasible:
                        row = _base_row(plan, settings, shot, seed, target)
                        score = result.get(target, {})
                        valid = score.get("n") == 1 and all(
                            math.isfinite(float(score.get(m + "_mean", float("nan"))))
                            for m in ("AUROC", "AUPRC")
                        )
                        row.update(
                            status="ok" if valid else "failed",
                            reason=(
                                ""
                                if valid
                                else "runner returned missing/nonfinite metrics or n != 1"
                            ),
                        )
                        if valid:
                            row.update(
                                AUROC=float(score["AUROC_mean"]), AUPRC=float(score["AUPRC_mean"])
                            )
                        # Recomputed completed targets are needed only to keep
                        # runner RNG order. A failed retry must not erase a
                        # valid checkpoint for a target that was already done.
                        if target in pending or valid:
                            rows[_key(row)] = row
                    _save(rows, plans, settings, paths)
                    if any(
                        rows[(plan["context_id"], shot, seed, t)]["status"] != "ok"
                        for t in feasible
                    ):
                        raise RuntimeError(
                            "Incomplete runner output; saved successful targets. Fix the failure and rerun to resume."
                        )
                except Exception as exc:
                    for target in pending:
                        row = rows[(plan["context_id"], shot, seed, target)]
                        if row["status"] == "pending":
                            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    _save(rows, plans, settings, paths)
                    raise
                finally:
                    gc.collect()
                    if str(settings["device"]).startswith("cuda"):
                        torch.cuda.empty_cache()
    _save(rows, plans, settings, paths)
    return rows


def run(args):
    """Run support-size evaluation from the root CLI."""
    settings = _settings(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = settings["stem"]
    paths = {kind: RESULTS / f"{stem}_{kind}.csv" for kind in ("seeds", "detail", "summary")}
    paths["contexts"] = RESULTS / f"{stem}_contexts.json"
    log_path = RESULTS / f"{stem}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(
        _Tee(sys.stdout, log)
    ), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        try:
            print(f"\n[{datetime.now().isoformat(timespec='seconds')}] target shot sweep")
            print(
                f"methods={settings['methods']} shots={settings['shots']} seeds={settings['seeds']}"
            )
            print(
                f"direction={settings['src_type']}2{settings['tgt_type']} modes={settings['modes']} "
                f"device={settings['device']} quick={settings['quick']}"
            )
            plans = _plans(settings)
            for plan in plans:
                print(
                    f"  {plan['method']}/{plan['mode']} context={plan['context_id']} "
                    f"sources={plan['sources']} targets={plan['targets']}"
                )
                for shot in settings["shots"]:
                    skipped = [
                        t
                        for t in plan["targets"]
                        if any(_row_budget(plan, shot, seed, t)[1] for seed in settings["seeds"])
                    ]
                    if plan["method"] == "taggad":
                        skipped = [
                            t
                            for t in plan["targets"]
                            if t not in plan["context"]["training_targets"]
                        ]
                    print(
                        f"    k={shot}: feasible_for_all_seeds={len(plan['targets']) - len(skipped)} "
                        f"infeasible_for_some_seeds={skipped}"
                    )
            if any(method in HOLDOUT_METHODS for method in settings["methods"]):
                print(
                    "  zero_shot_query_holdout: hold out k random target nodes; training and scores are unchanged."
                )
            if "anomalygfm_fs" in settings["methods"]:
                print(
                    "  AnomalyGFM FS scores do not depend on support; its shot curve reflects query-set changes only."
                )
            _run(settings, plans, paths)
            print(
                "\nComplete. Summary uses targets successful for every shot and seed; detail reports individual graphs."
            )
            for path in [*paths.values(), log_path]:
                print(f"  saved -> {path}")
        except BaseException:
            traceback.print_exc()
            raise
