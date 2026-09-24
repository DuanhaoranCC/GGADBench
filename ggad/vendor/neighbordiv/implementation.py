"""NeighborDiv: training-free zero-shot GGAD (arXiv:2605.20879v1).

Paper pipeline:
  truncated SVD (r=8) -> row L1 normalization -> cosine similarity ->
  variance over unordered pairs in each one-hop neighbor set -> absolute
  deviation from the valid-node median -> valid-node z-score.

The main paper uses every unordered pair.  ``_full_diversity`` is mathematically
identical to that enumeration but evaluates its first two moments with sparse
matrix products, avoiding both O(d_i^2) storage and a dense neighbor Gram
matrix.  The paper's uniform k-pair approximation is also available.
"""

from evaluation import format_metrics

import hashlib
import json

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import ggad.config as C
from common.data import (
    CACHE_ROOT,
    _atomic_save_npy,
    _content_fingerprint,
    aggregate,
    load_target_marked,
)
from util import evaluate

CACHE = CACHE_ROOT / "neighbordiv"


def _binary_csr(adj, keep_self_loops=True):
    """Return the binary CSR whose row indices are the paper's N(i)."""
    # The large-graph loaders already return a private float32 CSR.  Reuse it
    # in the default keep-self-loop path so T-Social's 146M-edge index arrays
    # are not duplicated merely to binarize their values.
    reuse = sp.isspmatrix_csr(adj) and adj.dtype == np.float32 and keep_self_loops
    a = sp.csr_matrix(adj, dtype=np.float32, copy=not reuse)
    a.sum_duplicates()
    if not keep_self_loops:
        a.setdiag(0)
        a.eliminate_zeros()
    a.sort_indices()
    a.data.fill(1.0)
    return a


def _project_features(features, hp):
    """Equation (1), followed by the stated L1 and internal L2 normalization."""
    dtype = np.dtype(hp.get("compute_dtype", "float64"))
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("NEIGHBORDIV_HP['compute_dtype'] must be float32 or float64")
    x = (
        features.astype(dtype, copy=False)
        if sp.issparse(features)
        else np.asarray(features, dtype=dtype)
    )
    n, width = x.shape
    rank = min(int(hp["svd_dim"]), n, width)
    if rank < 1:
        raise ValueError(f"NeighborDiv requires a non-empty feature matrix, got {x.shape}")

    algorithm = str(hp.get("svd_algorithm", "arpack"))
    # sklearn's ARPACK path requires rank < min(X.shape).  This fallback only
    # affects feature spaces too small to support the paper's r=8 setting.
    if algorithm == "arpack" and rank >= min(n, width):
        algorithm = "randomized"
    reducer = TruncatedSVD(
        n_components=rank,
        algorithm=algorithm,
        random_state=int(hp.get("svd_seed", 0)),
    )
    projected = np.asarray(reducer.fit_transform(x), dtype=dtype)
    projected = normalize(projected, norm="l1", axis=1, copy=False)
    projected = normalize(projected, norm="l2", axis=1, copy=False)
    return np.ascontiguousarray(projected, dtype=dtype)


def _outer_coordinates(rank):
    """Upper-triangle coordinates weighted so ||q||^2 equals Frobenius norm."""
    out = []
    root_two = float(np.sqrt(2.0))
    for left in range(rank):
        for right in range(left, rank):
            out.append((left, right, 1.0 if left == right else root_two))
    return out


def _outer_block(features, coordinates):
    block = np.empty((features.shape[0], len(coordinates)), dtype=features.dtype)
    for column, (left, right, weight) in enumerate(coordinates):
        block[:, column] = features[:, left] * features[:, right] * weight
    return block


def _full_diversity(adj, features, node_chunk=32_768, outer_memory_mb=256, round_decimals=14):
    """Exact variance of all unordered neighbor-pair cosine similarities.

    For neighbor vectors u_j, the unordered-pair first and second sums are

      1/2 (||sum_j u_j||^2 - sum_j ||u_j||^2)
      1/2 (||sum_j u_j u_j^T||_F^2 - sum_j ||u_j||^4).

    Hence this is exactly the paper's Full variant, without constructing any
    d_i x d_i Gram matrix.  Sparse row chunks only limit temporary memory.
    """
    a = sp.csr_matrix(adj, dtype=np.float32, copy=False)
    x = np.asarray(features)
    n, rank = x.shape
    if a.shape != (n, n):
        raise ValueError(f"adjacency {a.shape} and features {x.shape} do not align")
    if node_chunk < 1 or outer_memory_mb < 1:
        raise ValueError("node_chunk and outer_memory_mb must be positive")

    degree = np.diff(a.indptr).astype(np.int64, copy=False)
    pair_count = degree * (degree - 1) // 2
    valid = pair_count > 0
    diversity = np.zeros(n, dtype=np.float64)
    if not np.any(valid):
        return diversity, valid

    norm2 = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    norm4 = norm2 * norm2
    pair_sum = np.zeros(n, dtype=np.float64)
    diagonal_fourth = np.zeros(n, dtype=np.float64)

    for start in range(0, n, node_chunk):
        end = min(start + node_chunk, n)
        rows = a[start:end]
        summed = np.asarray(rows @ x, dtype=np.float64)
        norm_sums = np.asarray(rows @ np.column_stack((norm2, norm4)), dtype=np.float64)
        pair_sum[start:end] = 0.5 * (np.einsum("ij,ij->i", summed, summed) - norm_sums[:, 0])
        diagonal_fourth[start:end] = norm_sums[:, 1]

    coordinates = _outer_coordinates(rank)
    bytes_per_column = max(n * x.dtype.itemsize, 1)
    columns_per_block = max(
        1, min(len(coordinates), int(outer_memory_mb * 1024**2 // bytes_per_column))
    )
    gram_fourth = np.zeros(n, dtype=np.float64)
    for offset in range(0, len(coordinates), columns_per_block):
        outer = _outer_block(x, coordinates[offset : offset + columns_per_block])
        for start in range(0, n, node_chunk):
            end = min(start + node_chunk, n)
            summed_outer = np.asarray(a[start:end] @ outer, dtype=np.float64)
            gram_fourth[start:end] += np.einsum("ij,ij->i", summed_outer, summed_outer)

    pair_square_sum = 0.5 * (gram_fourth - diagonal_fourth)
    count = pair_count[valid].astype(np.float64)
    mean = pair_sum[valid] / count
    # Roundoff can make a true zero variance slightly negative.
    diversity[valid] = np.maximum(pair_square_sum[valid] / count - mean * mean, 0.0)
    # The moment identity subtracts nearly equal quantities for uniform
    # neighborhoods.  Pair enumeration yields exact ties/zeros there, while
    # BLAS reduction order may leave ~1e-15 residue and spuriously break AUC
    # ties.  Rounding only below float64 numerical accuracy restores the
    # enumeration-equivalent ranking.
    decimals = min(int(round_decimals), 6) if x.dtype == np.float32 else int(round_decimals)
    diversity[valid] = np.round(diversity[valid], decimals=decimals)
    return diversity, valid


def _floyd_sample(total, count, rng):
    """Uniform integer sample without replacement in O(count) memory."""
    selected = set()
    for current in range(int(total) - int(count), int(total)):
        candidate = int(rng.randint(0, current + 1))
        selected.add(current if candidate in selected else candidate)
    return np.fromiter(selected, dtype=np.int64, count=count)


def _unrank_pairs(ranks, degree):
    """Map lexicographic upper-triangle ranks to 0 <= left < right < degree."""
    ranks = np.asarray(ranks, dtype=np.int64)
    width = float(2 * degree - 1)
    left = np.floor((width - np.sqrt(width * width - 8.0 * ranks)) / 2.0).astype(np.int64)
    left = np.clip(left, 0, degree - 2)
    base = left * (2 * degree - left - 1) // 2

    # Correct the occasional one-off produced by the floating-point square root.
    too_high = base > ranks
    while np.any(too_high):
        left[too_high] -= 1
        base = left * (2 * degree - left - 1) // 2
        too_high = base > ranks
    next_base = (left + 1) * (2 * degree - left - 2) // 2
    too_low = ranks >= next_base
    while np.any(too_low):
        left[too_low] += 1
        base = left * (2 * degree - left - 1) // 2
        next_base = (left + 1) * (2 * degree - left - 2) // 2
        too_low = ranks >= next_base
    right = left + 1 + (ranks - base)
    return left, right


def _sample_diversity(adj, features, pair_budget=100, seed=0):
    """Paper Eq. (7)-(9): uniform unordered-pair sampling without replacement."""
    if pair_budget < 1:
        raise ValueError("sample_pairs must be positive")
    a = sp.csr_matrix(adj, copy=False)
    x = np.asarray(features)
    degree = np.diff(a.indptr).astype(np.int64, copy=False)
    pair_count = degree * (degree - 1) // 2
    valid = pair_count > 0
    diversity = np.zeros(a.shape[0], dtype=np.float64)
    rng = np.random.RandomState(int(seed))

    for node in np.flatnonzero(valid):
        begin, end = a.indptr[node], a.indptr[node + 1]
        neighbors = a.indices[begin:end]
        total = int(pair_count[node])
        take = min(int(pair_budget), total)
        if take == total:
            left, right = np.triu_indices(len(neighbors), k=1)
        else:
            left, right = _unrank_pairs(_floyd_sample(total, take, rng), len(neighbors))
        similarities = np.einsum(
            "ij,ij->i", x[neighbors[left]], x[neighbors[right]], dtype=np.float64
        )
        diversity[node] = np.var(similarities, dtype=np.float64)
    return diversity, valid


def _calibrate(diversity, valid):
    """Paper Eq. (10)-(13), including neutral score 0 for degree < 2."""
    score = np.zeros(len(diversity), dtype=np.float64)
    if not np.any(valid):
        return score
    values = np.asarray(diversity, dtype=np.float64)[valid]
    reference = np.median(values)
    deviation = np.abs(values - reference)
    scale = deviation.std()
    if np.isfinite(scale) and scale > np.finfo(np.float64).eps:
        score[valid] = (deviation - deviation.mean()) / scale
    return score


def _resolve_pair_mode(hp, pair_count):
    mode = str(hp.get("pair_mode", "full")).lower()
    if mode not in {"full", "sample", "auto"}:
        raise ValueError("NEIGHBORDIV_HP['pair_mode'] must be full, sample, or auto")
    if mode == "auto":
        total = int(np.sum(pair_count, dtype=np.int64))
        return "full" if total <= int(hp["auto_full_pair_limit"]) else "sample"
    return mode


def _signature(name, input_digest, raw_dim, hp, kind, seed=None):
    payload = {
        "version": hp.get("implementation_version", "unknown"),
        "name": name,
        "input_digest": input_digest,
        "raw_dim": int(raw_dim),
        "svd_dim": int(hp["svd_dim"]),
        "svd_algorithm": str(hp.get("svd_algorithm", "arpack")),
        "svd_seed": int(hp.get("svd_seed", 0)),
        "compute_dtype": str(hp.get("compute_dtype", "float64")),
        "moment_round_decimals": int(hp.get("moment_round_decimals", 14)),
        "keep_input_self_loops": bool(hp.get("keep_input_self_loops", True)),
        "kind": kind,
        "sample_pairs": int(hp.get("sample_pairs", 100)),
        "seed": None if seed is None else int(seed),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE / f"{name}_{kind}_{digest}.npy"


def _load_or_project(name, input_digest, raw_features, hp):
    path = _signature(name, input_digest, raw_features.shape[1], hp, "features")
    if bool(hp.get("cache", True)) and path.exists():
        cached = np.load(path, allow_pickle=False)
        expected_rank = min(int(hp["svd_dim"]), raw_features.shape[0], raw_features.shape[1])
        if cached.shape == (raw_features.shape[0], expected_rank):
            return cached
    projected = _project_features(raw_features, hp)
    if bool(hp.get("cache", True)):
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save_npy(path, projected)
    return projected


def _compute_score(adj, projected, hp, mode, seed):
    if mode == "full":
        diversity, valid = _full_diversity(
            adj,
            projected,
            node_chunk=int(hp["node_chunk"]),
            outer_memory_mb=int(hp["outer_memory_mb"]),
            round_decimals=int(hp.get("moment_round_decimals", 14)),
        )
    else:
        diversity, valid = _sample_diversity(
            adj, projected, pair_budget=int(hp["sample_pairs"]), seed=seed
        )
    return _calibrate(diversity, valid)


def run_neighbordiv(sources, targets, seeds, epochs, device, hp=None, target_evaluator=None):
    """Run the source-independent method in the ggad result format."""
    del sources, epochs, device
    hp = dict(C.NEIGHBORDIV_HP if hp is None else hp)
    seeds = list(seeds) or [0]
    per = {name: [] for name in targets}

    for name in targets:
        raw_adj, raw_features, labels, mark = load_target_marked(name)
        adj = _binary_csr(raw_adj, keep_self_loops=bool(hp.get("keep_input_self_loops", True)))
        del raw_adj
        raw_dim = int(raw_features.shape[1])
        input_digest = _content_fingerprint(
            adj, "neighbordiv-binary-adj-v1"
        ) + _content_fingerprint(raw_features, "neighbordiv-raw-features-v1")
        degree = np.diff(adj.indptr).astype(np.int64, copy=False)
        pair_count = degree * (degree - 1) // 2
        mode = _resolve_pair_mode(hp, pair_count)
        total_pairs = int(np.sum(pair_count, dtype=np.int64))
        print(
            f"    [neighbordiv] {name}: mode={mode} N={adj.shape[0]} E={adj.nnz} "
            f"pairs={total_pairs} r={hp['svd_dim']}"
        )

        # Full is deterministic: calculate it once and repeat the same metric
        # for benchmark seed accounting.  Sampling intentionally follows every
        # supplied seed, matching the paper's five-seed sampling ablation.
        run_seeds = [int(hp.get("svd_seed", 0))] if mode == "full" else seeds
        metrics = []
        projected = None
        for seed in run_seeds:
            path = _signature(
                name,
                input_digest,
                raw_dim,
                hp,
                mode,
                None if mode == "full" else seed,
            )
            cached = bool(hp.get("cache", True)) and path.exists()
            if cached:
                score = np.load(path, allow_pickle=False)
                if score.shape != (adj.shape[0],) or not np.isfinite(score).all():
                    cached = False
            if not cached:
                if projected is None:
                    projected = _load_or_project(name, input_digest, raw_features, hp)
                    # The projected r-dimensional matrix is sufficient from
                    # here onward; release potentially huge raw attributes.
                    raw_features = None
                score = _compute_score(adj, projected, hp, mode, seed)
                if bool(hp.get("cache", True)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_save_npy(path, score)
            # Raw full-mode scores are deterministic, but a target evaluator's
            # held-out query nodes can differ across seeds.
            metric_seeds = seeds if mode == "full" and target_evaluator is not None else [seed]
            for metric_seed in metric_seeds:
                metric = (
                    target_evaluator(
                        name,
                        int(metric_seed),
                        np.asarray(labels),
                        score,
                        np.asarray(mark, dtype=bool),
                    )
                    if target_evaluator is not None
                    else evaluate(np.asarray(labels)[mark], score[np.asarray(mark, dtype=bool)])
                )
                metrics.append(metric)
                print(
                    f"      seed={metric_seed} cached={cached} {format_metrics(metric)}"
                )
        if mode == "full" and target_evaluator is None:
            per[name].extend(dict(metrics[0]) for _ in seeds)
        else:
            per[name].extend(metrics)
    return aggregate(per)


__all__ = [
    "run_neighbordiv",
    "_binary_csr",
    "_project_features",
    "_full_diversity",
    "_sample_diversity",
    "_calibrate",
    "_unrank_pairs",
]
