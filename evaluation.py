"""Evaluation selection and Recall@K with K equal to query anomaly count."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import numpy as np

METRIC_MODES = ("standard", "recall-at-k", "all")
_MODE = ContextVar("evaluation_metrics", default="standard")


def metric_names(mode=None):
    mode = _MODE.get() if mode is None else mode
    if mode not in METRIC_MODES:
        raise ValueError(f"Unknown metrics mode: {mode}")
    if mode == "recall-at-k":
        return ("Rec@K",)
    return ("AUROC", "AUPRC", "Rec@K") if mode == "all" else ("AUROC", "AUPRC")


@contextmanager
def metric_scope(mode):
    metric_names(mode)
    token = _MODE.set(mode)
    try:
        yield
    finally:
        _MODE.reset(token)


def metric_run(function=None, *, config=None):
    if function is None:
        return lambda entry: metric_run(entry, config=config)
    @wraps(function)
    def run(args, *rest, **kwargs):
        mode = getattr(args, "metrics", None) or getattr(config, "EVAL_METRICS", "standard")
        with metric_scope(mode):
            return function(args, *rest, **kwargs)
    return run


def recall_at_k(labels, scores):
    """Rank descending; resolve boundary ties by original evaluation order.

    Inputs contain only evaluated query nodes, with 1 denoting anomalies.
    K=0 is undefined and returns NaN. No score threshold is learned.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Recall@K expects matching one-dimensional labels and scores")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Recall@K requires binary labels on evaluated nodes")
    if not np.isfinite(scores).all():
        raise ValueError("Recall@K requires finite evaluation scores")
    anomalies = labels == 1
    k = int(anomalies.sum())
    if k == 0:
        return float("nan")
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > cutoff
    remaining = k - int(above.sum())
    tied = np.flatnonzero(scores == cutoff)[:remaining]
    return float((anomalies[above].sum() + anomalies[tied].sum()) / k)


def aggregate_metrics(per_target):
    output = {}
    for target, runs in per_target.items():
        row = {}
        for metric in metric_names():
            values = np.asarray([run[metric] for run in runs], dtype=float)
            row[metric + "_mean"] = float(values.mean()) if len(values) else float("nan")
            row[metric + "_std"] = float(values.std()) if len(values) else float("nan")
        row["n"] = len(runs)
        output[target] = row
    return output


def metric_columns(score):
    return {metric + suffix: round(score[metric + suffix], 4)
            for metric in metric_names() for suffix in ("_mean", "_std")}


def format_metrics(score):
    return " ".join(f"{name}={score[name]:.4f}" for name in metric_names() if name in score)


def primary_metric():
    return metric_names()[0]


def metric_path(path):
    suffix = {"standard": "", "recall-at-k": "_recall_at_k", "all": "_all_metrics"}[_MODE.get()]
    return path.with_name(path.stem + suffix + path.suffix)


def build_metric_delta(detail):
    metric = primary_metric()
    grouped = {}
    for row in detail:
        key = (row["method"], row["src_type"], row["tgt_type"], row["target"])
        grouped.setdefault(key, {})[row["mode"]] = float(row[metric + "_mean"])
    output = []
    for (method, source, target_type, target), modes in grouped.items():
        single, multi = modes.get("single"), modes.get("multi")
        output.append({"method": method, "src_type": source, "tgt_type": target_type,
                       "target": target, "single_" + metric: single, "multi_" + metric: multi,
                       "delta": round(multi-single, 4) if single is not None and multi is not None else None})
    return output
