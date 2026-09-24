"""Dataset similarity values used by AnomalyGFM's scoring rule.

Most fixed values come from the paper's dataset table. The weibo value is an
empirical SVD8 estimate retained from the benchmark configuration, not a paper
value. Datasets without a fixed entry use mean edge cosine similarity.
"""

import numpy as np
import scipy.sparse as sp

_SIM_CACHE = {}

PAPER_SIM = {
    "Facebook": 0.690,
    "Reddit": 0.997,
    "Amazon": 0.645,
    "Disney": 0.804,
    "Amazon-all": 0.645,
    "YelpChi-all": 0.905,
    "tolokers": 0.814,
    "questions": 0.679,
    "t_finance": 0.107,
    "elliptic": 0.356,
    "tsocial": 0.307,
    "weibo": 0.544,
}


def global_avg_sim(name, adj, feat):
    """Cache mean edge cosine similarity, limiting chunk temporaries to 256 MiB."""
    if name in _SIM_CACHE:
        return _SIM_CACHE[name]
    f = np.asarray(feat.todense() if hasattr(feat, "todense") else feat, dtype=np.float64)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    e = sp.coo_matrix(adj)
    row, col, E = e.row, e.col, e.nnz
    d = f.shape[1]

    chunk = max(1, (256 * 1024 * 1024) // (3 * max(d, 1) * 8))
    s = 0.0
    for i in range(0, E, chunk):
        s += float((f[row[i : i + chunk]] * f[col[i : i + chunk]]).sum())
    _SIM_CACHE[name] = s / max(E, 1)
    return _SIM_CACHE[name]
