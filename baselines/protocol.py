"""Shared source/target loading for source-aware baseline adaptation.

The semantic protocol must match `common.data` used by the GGAD
methods: target evaluation keeps all marked nodes, includes their k-hop context
when a target graph can be safely reduced, and evaluates only nodes marked by
`mark`.

Classic baselines are less scalable than the GGAD runners, but this block
follows the same data-retention rule as the second block: loaders return the
full source/target graph.  If a method cannot process the returned graph with
its dense official path, its runner must switch to a sparse/chunked or
full-neighbor-batched implementation internally rather than changing the node
set.
"""

import numpy as np
import scipy.sparse as sp

from baselines.preprocess import register_adjacency
from common.data import _khop_nodes, _sample_eval_nodes, load_marked, load_target_marked

DENSE_SAFE_N = 12_000
DENSE_SAFE_E = 2_000_000
DENSE_SOURCE_N = 8_000


def _induced(adj, feat, label, nodes, eval_nodes=None):
    nodes = np.asarray(nodes, dtype=np.int64)
    old_to_new = np.full(adj.shape[0], -1, dtype=np.int64)
    old_to_new[nodes] = np.arange(nodes.size)
    sub_adj = sp.csr_matrix(adj)[nodes][:, nodes].tocsr()
    sub_feat = feat[nodes] if not sp.issparse(feat) else feat[nodes]
    sub_label = np.asarray(label)[nodes]
    if eval_nodes is None:
        sub_mark = np.ones(nodes.size, dtype=bool)
    else:
        eval_new = old_to_new[np.asarray(eval_nodes, dtype=np.int64)]
        eval_new = eval_new[eval_new >= 0]
        sub_mark = np.zeros(nodes.size, dtype=bool)
        sub_mark[eval_new] = True
    return sub_adj, sub_feat, sub_label, sub_mark


def _needs_sampling(adj):
    return adj.shape[0] > DENSE_SAFE_N or adj.nnz > DENSE_SAFE_E


def load_dense_source(name, max_nodes=DENSE_SOURCE_N, hops=2):
    """Return the full source graph.

    `max_nodes` and `hops` are kept for backward-compatible runner signatures,
    but are intentionally ignored.  Earlier versions sampled dense sources such
    as t_finance; the current benchmark instead keeps all source nodes/edges and
    lets each baseline use a bounded sparse/chunked training path when needed.
    """
    del max_nodes, hops
    adj, feat, label, mark = load_marked(name)
    register_adjacency(feat, adj)
    if _needs_sampling(adj):
        print(
            f"    [full-dense-source] {name}: train={int(np.asarray(mark, dtype=bool).sum())} "
            f"N={adj.shape[0]} E={adj.nnz}"
        )
    return adj, feat, label, mark


def load_dense_target(name, hops=2):
    # Target evaluation follows the GGAD runner semantics exactly: no sampling
    # away marked evaluation nodes.  Methods that cannot process the returned
    # graph must use chunk/sparse/neighbor-sampling internally rather than
    # changing the evaluation set.
    result = load_target_marked(name, hops=hops)
    register_adjacency(result[1], result[0])
    return result
