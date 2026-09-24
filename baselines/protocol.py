"""Load complete source and target graphs for classical baselines.

Evaluation uses the supplied label mask. Runners handle large graphs through
sparse operations or batching without changing the loaded node or edge sets.
"""

import numpy as np

from baselines.preprocess import register_adjacency
from common.data import load_marked, load_target_marked

DENSE_SAFE_N = 12_000
DENSE_SAFE_E = 2_000_000


def _is_large_graph(adj):
    return adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E


def load_dense_source(name):
    """Load a full source graph and register its adjacency for feature alignment."""
    adj, feat, label, mark = load_marked(name)
    register_adjacency(feat, adj)
    if _is_large_graph(adj):
        print(
            f"    [full-dense-source] {name}: train={int(np.asarray(mark, dtype=bool).sum())} "
            f"N={adj.shape[0]} E={adj.nnz}"
        )
    return adj, feat, label, mark


def load_dense_target(name):
    """Load a full target graph and its evaluation mask."""
    result = load_target_marked(name)
    register_adjacency(result[1], result[0])
    return result
