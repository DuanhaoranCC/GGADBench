"""Target holdout-size control for detectors that have no native support input.

Select k marked target nodes without looking at their class labels, then score
every remaining marked node with the original detector scores. There is no new
model, feature processing, score transformation or parameter fitting. This is
an evaluation-subset control, not a claim that the detector learns from k nodes.
"""

from __future__ import annotations

import numpy as np

from util import evaluate


def _vector(value, name):
    result = np.asarray(value)
    if result.ndim > 2 or (result.ndim == 2 and 1 not in result.shape):
        raise ValueError(f"shot query {name} must be a node vector")
    return result.reshape(-1)


def _validate_shot(shot):
    if isinstance(shot, bool) or not isinstance(shot, (int, np.integer)) or shot < 1:
        raise ValueError("shot query shot must be a positive integer")
    return int(shot)


def split_support(labels, mark, shot, seed):
    """Random marked-node prefix; labels do not influence node selection.

    Both classes must remain in the complete marked query set so AUROC and
    AUPRC can be evaluated. A failed split raises; it is never resampled using
    class information and is never repaired by dropping additional nodes.
    """
    shot = _validate_shot(shot)
    labels = _vector(labels, "labels")
    mark = _vector(mark, "mark").astype(bool, copy=False)
    if labels.shape != mark.shape:
        raise ValueError("shot query labels/mark lengths differ")
    if not np.isin(labels[mark], [0, 1]).all():
        raise ValueError("shot query marked labels must be binary")
    pool = np.flatnonzero(mark)
    if shot >= pool.size:
        raise ValueError(
            f"shot query requires shot < marked node count; shot={shot}, marked={pool.size}"
        )
    np.random.RandomState(int(seed)).shuffle(pool)
    support = pool[:shot].copy()
    query_mask = mark.copy()
    query_mask[support] = False
    query = np.flatnonzero(query_mask)
    if np.unique(labels[query]).size != 2:
        raise ValueError("shot query lacks both classes after random holdout")
    return support, query


class QueryHoldout:
    """Callable ``(target, seed, labels, scores, mark) -> query metrics``."""

    def __init__(self, shot):
        self.shot = _validate_shot(shot)

    def __call__(self, target_name, seed, full_labels, full_scores, full_mark):
        labels = _vector(full_labels, "labels")
        scores = _vector(full_scores, "scores")
        mark = _vector(full_mark, "mark").astype(bool, copy=False)
        if labels.shape != scores.shape or labels.shape != mark.shape:
            raise ValueError(f"shot query {target_name}: labels/scores/mark lengths differ")
        support, query = split_support(labels, mark, self.shot, seed)
        # Original scores pass directly to the benchmark metric function.
        result = evaluate(labels[query], scores[query])
        print(
            f"    [shot_query_holdout] {target_name} seed={seed} "
            f"held_out={len(support)} queries={len(query)} "
            f"AUROC={result['AUROC']:.4f} AUPRC={result['AUPRC']:.4f}",
            flush=True,
        )
        return result


__all__ = ["QueryHoldout", "split_support"]
