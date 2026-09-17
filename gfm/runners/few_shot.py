"""Shared support/query splitting helpers for foundation-model runners."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sample_class_support(
    labels,
    mark,
    cls: int,
    k: int,
    rng: np.random.RandomState,
    exclude: Iterable[int] = (),
    *,
    reserve_query: bool = False,
    dataset: str = "graph",
) -> np.ndarray:
    """Sample ``k`` support occurrences while preserving a disjoint query.

    Repeated support occurrences are already the benchmark's fallback when a
    class contains fewer than ``k`` marked nodes.  For target evaluation we
    additionally reserve one unique node before sampling whenever the class
    has at most ``k`` nodes.  This keeps the configured k-shot episode shape
    while guaranteeing that AUROC/AUPRC see an unseen query from both classes.
    """
    excluded = set(int(i) for i in exclude)
    labels = np.asarray(labels)
    mark = np.asarray(mark, dtype=bool)
    pool = np.where((labels == cls) & mark)[0]
    if excluded:
        pool = np.asarray([i for i in pool if int(i) not in excluded], dtype=np.int64)
    if pool.size == 0:
        raise ValueError(f"{dataset}: class {cls} has no marked nodes")

    if reserve_query and pool.size <= int(k):
        if pool.size < 2:
            raise ValueError(
                f"{dataset}: class {cls} has only {pool.size} marked node; "
                "disjoint few-shot support/query evaluation needs at least 2"
            )
        reserve_pos = int(rng.randint(pool.size))
        reserved = int(pool[reserve_pos])
        pool = np.delete(pool, reserve_pos)
        print(
            f"    [few-shot-reserve] {dataset} class={cls}: available="
            f"{pool.size + 1} shot={k}; reserve node={reserved} for query and "
            f"sample support with replacement from {pool.size}",
            flush=True,
        )

    return rng.choice(pool, size=int(k), replace=pool.size < int(k)).astype(np.int64)
