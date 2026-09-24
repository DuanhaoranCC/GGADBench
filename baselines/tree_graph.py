"""RF-Graph / XGB-Graph baselines ported from GADBench.

Official GADBench implements both methods as:

    graph-aware features = GIN_noparam(raw node features)
    classifier          = RandomForestClassifier / XGBClassifier

`GIN_noparam` has no learnable parameters.  With DGL's GINConv(None,
init_eps=-1), the self term is zero and each layer is just neighbor
aggregation.  This file implements the same feature transform with scipy sparse
matrices, so the benchmark stays DGL-free.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier


def _as_dense_float32(x):
    return np.asarray(x.todense() if sp.issparse(x) else x, dtype=np.float32)


def _row_normalized(adj):
    a = sp.csr_matrix(adj).astype(np.float32)
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    return sp.diags(inv).dot(a).tocsr()


def _aggregate(adj, x, agg):
    # DGL/PyG message passing aggregates messages from source nodes into the
    # destination node.  For a scipy adjacency where A[src, dst] is an edge,
    # destination-wise incoming aggregation is therefore A.T @ X.  Symmetric
    # graphs are unaffected, but directed graphs must use this orientation to
    # match official GADBench's DGL GINConv(None, init_eps=-1).
    a = sp.csr_matrix(adj).astype(np.float32).transpose().tocsr()
    if agg == "sum":
        return a.dot(x).astype(np.float32, copy=False)
    if agg == "mean":
        return _row_normalized(a).dot(x).astype(np.float32, copy=False)
    if agg != "max":
        raise ValueError(f"unknown RF/XGB-Graph agg={agg}")

    # Exact max aggregation without a Python loop over millions of nodes.
    # ``maximum.at`` performs the same destination-wise reduction in C.  Work
    # feature-by-feature so T-Social never materializes an E×F message tensor.
    coo = a.tocoo(copy=False)
    out = np.full(x.shape, -np.inf, dtype=np.float32)
    for feature_col in range(x.shape[1]):
        np.maximum.at(out[:, feature_col], coo.row, x[coo.col, feature_col])
    out[np.isneginf(out)] = 0.0
    return out


def gin_noparam_features(adj, feat, num_layers=2, agg="mean"):
    """Return [X, A_agg X, A_agg^2 X, ...] like official GIN_noparam."""
    h = _as_dense_float32(feat)
    outs = [h]
    for _ in range(num_layers):
        h = _aggregate(adj, h, agg)
        outs.append(h)
    return np.concatenate(outs, axis=1).astype(np.float32, copy=False)


def make_rf(cfg, seed):
    return RandomForestClassifier(
        n_jobs=cfg.get("n_jobs", -1),
        n_estimators=cfg["n_estimators"],
        criterion=cfg["criterion"],
        max_samples=cfg["max_samples"],
        max_features=cfg["max_features"],
        random_state=seed,
    )


def make_xgb(cfg, seed):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "XGB-Graph requires xgboost. Install it with: python -m pip install xgboost"
        ) from e

    # Use hist by default; gpu_hist is version-dependent and fails on many
    # modern xgboost installs.  This is still the official XGBClassifier model.
    return xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        eta=cfg["eta"],
        reg_lambda=cfg["reg_lambda"],
        subsample=cfg["subsample"],
        booster=cfg["booster"],
        tree_method=cfg.get("tree_method", "hist"),
        eval_metric="auc",
        random_state=seed,
        n_jobs=cfg.get("n_jobs", -1),
    )


def predict_proba_chunked(model, x, chunk=200_000):
    scores = np.empty(x.shape[0], dtype=np.float32)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        scores[s:e] = model.predict_proba(x[s:e])[:, 1].astype(np.float32, copy=False)
    return scores
