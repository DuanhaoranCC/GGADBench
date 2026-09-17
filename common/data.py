"""Shared dataset loading, feature caches, and full-graph streaming operations."""

import os
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.decomposition import IncrementalPCA


@lru_cache(maxsize=1)
def _iaggad_preprocess_ops():
    """Load feature-alignment helpers only when an aligned runner needs them."""
    from ggad.vendor.iaggad.preprocessing import (
        _to_torch_sparse,
        feat_alignment,
        normalize_adj_sym,
        preprocess_features,
    )

    return _to_torch_sparse, feat_alignment, normalize_adj_sym, preprocess_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GAD_DATA_ROOT") or PROJECT_ROOT / "Dataset").expanduser().resolve()
REAL_DIR = DATA_ROOT / "real"
FAKE_DIR = DATA_ROOT / "fake"
CACHE_ROOT = PROJECT_ROOT / "cache"
CACHE = CACHE_ROOT / "aligned_arc_official_float32_v4"
SVD_CACHE = CACHE_ROOT / "svd_numpy"
SVD_TORCH_CACHE = CACHE_ROOT / "svd_torch_float32"


ROWNORM = ["Amazon", "Amazon-all", "YelpChi", "YelpChi-all", "tolokers", "t_finance"]
NO_SELFLOOP = ["YelpChi", "Facebook"]
DIMS = 64


BIG = {"dgraphfin", "elliptic", "tsocial"}


EDGE_CHUNK = 2_000_000


SAMPLE_EVAL_NODES = 10_000
SAMPLE_NORMAL_RATIO = 10
SAMPLE_FANOUT = (10, 5)
SAMPLE_SEED = 0


def _find(name):
    """Find a MATLAB dataset in the configured real and synthetic directories."""
    for d in (REAL_DIR, FAKE_DIR):
        p = d / f"{name}.mat"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.mat was not found in {REAL_DIR} or {FAKE_DIR}")


def _load_mat4(name):
    """Load adjacency, attributes, labels, and an all-node evaluation mask."""
    m = sio.loadmat(str(_find(name)))
    adj = sp.csr_matrix(m["Network"] if "Network" in m else m["A"])
    feat = m["Attributes"] if "Attributes" in m else m["X"]
    label = np.squeeze(np.asarray(m["Label"] if "Label" in m else m["gnd"]))
    return adj, feat, label, np.ones(adj.shape[0], dtype=bool)


def _load_dgraphfin():
    """Load DGraph-Fin and mark nodes belonging to its released data splits."""
    f = np.load(REAL_DIR / "dgraphfin.npz")
    x = f["x"].astype(np.float64)
    label = (f["y"] == 1).astype(np.int64)
    ei = f["edge_index"]
    n = x.shape[0]
    adj = sp.csr_matrix((np.ones(ei.shape[0]), (ei[:, 0], ei[:, 1])), shape=(n, n))
    mark = np.zeros(n, dtype=bool)
    for k in ("train_mask", "valid_mask", "test_mask"):
        mark[f[k]] = True
    return adj, x, label, mark


def _load_elliptic():
    """Load the Elliptic CSV files and mark nodes with known class labels."""
    import pandas as pd

    folder = REAL_DIR / "elliptic_bitcoin_dataset"
    labels = pd.read_csv(folder / "elliptic_txs_classes.csv").to_numpy()
    feats = pd.read_csv(folder / "elliptic_txs_features.csv", header=None).to_numpy()
    id_to_idx = {labels[i, 0]: i for i in range(labels.shape[0])}
    label = np.zeros(labels.shape[0], dtype=np.int64)
    label[labels[:, 1] == "1"] = 1
    mark = labels[:, 1] != "unknown"
    features = feats[:, 1:].astype(np.float64)
    edges = pd.read_csv(folder / "elliptic_txs_edgelist.csv").to_numpy()
    src = np.array([id_to_idx[e] for e in edges[:, 0]])
    dst = np.array([id_to_idx[e] for e in edges[:, 1]])
    n = labels.shape[0]
    adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return adj, features, label, mark


def _load_tsocial():
    """Load T-Social from NPZ, falling back to the optional DGL graph."""
    npz = REAL_DIR / "tsocial.npz"
    if npz.exists():
        f = np.load(npz)
        x = f["x"].astype(np.float64)
        label = f["y"].astype(np.int64)
        ei = f["edge_index"]  # (E,2) int32
        n = x.shape[0]
        adj = sp.csr_matrix((np.ones(ei.shape[0], np.float32), (ei[:, 0], ei[:, 1])), shape=(n, n))
        return adj, x, label, np.ones(n, dtype=bool)
    import dgl

    g = dgl.load_graphs(str(REAL_DIR / "tsocial"))[0][0]
    feat = g.ndata["feature"].numpy().astype(np.float64)
    label = g.ndata["label"].numpy().astype(np.int64)
    u, v = g.edges()
    n = g.num_nodes()
    adj = sp.csr_matrix((np.ones(u.shape[0], np.float32), (u.numpy(), v.numpy())), shape=(n, n))
    return adj, feat, label, np.ones(n, dtype=bool)


def _load_raw(name):
    """Dispatch dataset loading and return adjacency, features, labels, and mask."""
    if name == "dgraphfin":
        return _load_dgraphfin()
    if name == "elliptic":
        return _load_elliptic()
    if name == "tsocial":
        return _load_tsocial()
    return _load_mat4(name)


def load_marked(name):
    """Load the complete graph and its valid-label evaluation mask."""
    return _load_raw(name)


def _stable_seed(name, seed=SAMPLE_SEED):
    return seed + sum((i + 1) * ord(c) for i, c in enumerate(name))


def _sample_eval_nodes(
    labels, mark, name, max_nodes=SAMPLE_EVAL_NODES, normal_ratio=SAMPLE_NORMAL_RATIO
):
    """Sample a reproducible class-stratified subset of marked nodes."""
    rng = np.random.RandomState(_stable_seed(name))
    pos = np.where((labels == 1) & mark)[0]
    neg = np.where((labels == 0) & mark)[0]
    if len(pos) == 0 or len(neg) == 0:
        pool = np.where(mark)[0]
        return np.sort(rng.choice(pool, size=min(len(pool), max_nodes), replace=False))

    max_pos = max(1, max_nodes // (normal_ratio + 1))
    n_pos = min(len(pos), max_pos)
    n_neg = min(len(neg), max_nodes - n_pos, max(n_pos * normal_ratio, 1))
    pos_sel = rng.choice(pos, size=n_pos, replace=False)
    neg_sel = rng.choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos_sel, neg_sel]))


def _khop_nodes(adj, seeds, hops, name, fanout=SAMPLE_FANOUT):
    """Expand seed nodes through a reproducible bounded-fanout neighborhood."""
    reach = np.zeros(adj.shape[0], dtype=bool)
    frontier = np.asarray(seeds, dtype=np.int64)
    reach[frontier] = True
    rng = np.random.RandomState(_stable_seed(name, seed=17))
    a = (sp.csr_matrix(adj) + sp.csr_matrix(adj).T).tocsr()
    a.data[:] = 1
    for hop in range(hops):
        fan = fanout[min(hop, len(fanout) - 1)]
        sampled = []
        indptr, indices = a.indptr, a.indices
        for node in frontier:
            nbr = indices[indptr[node] : indptr[node + 1]]
            if nbr.size > fan:
                nbr = rng.choice(nbr, size=fan, replace=False)
            sampled.append(nbr)
        neigh = np.concatenate(sampled) if sampled else np.array([], dtype=np.int64)
        if neigh.size == 0:
            break
        new = neigh[~reach[neigh]]
        if new.size == 0:
            break
        reach[new] = True
        frontier = np.unique(new)
    return np.where(reach)[0]


def load_source_marked(name):
    """Load the complete source graph and its valid-label mask."""
    return load_marked(name)


def load_target_marked(name, hops=2):
    """Load the complete target graph without sampling evaluation nodes."""
    del hops
    adj, feat, label, mark = load_marked(name)
    if name in BIG:
        print(
            f"    [full-large-target] {name}: eval={int(np.asarray(mark, dtype=bool).sum())} "
            f"N={adj.shape[0]} E={adj.nnz}"
        )
    return adj, feat, label, mark


def load_real(name):
    adj, feat, label, _ = load_marked(name)
    return adj, feat, label


def _dense(feat):
    return np.asarray(feat.todense() if hasattr(feat, "todense") else feat, dtype=np.float64)


def _dense32(feat):
    return np.asarray(feat.todense() if hasattr(feat, "todense") else feat, dtype=np.float32)


def _iter_aligned_batches(feat, batch_size, proj):
    n = feat.shape[0]
    for start in range(0, n, batch_size):
        x = np.asarray(feat[start : start + batch_size], dtype=np.float32)
        if proj is not None:
            x = x @ proj
        yield start, x.astype(np.float32, copy=False)


def _feat_alignment_large(feat, edge_src, edge_dst, dims, batch_size=100_000):
    """Memory-safe RP/PCA/smoothness alignment for million-node targets."""
    x = _dense32(feat)
    proj = None
    if x.shape[1] < dims:
        rng = np.random.RandomState(0)
        proj = rng.normal(0.0, 1.0 / np.sqrt(256), size=(256, x.shape[1])).T.astype(np.float32)
        pca_dim = 256
    else:
        pca_dim = x.shape[1]

    ipca = IncrementalPCA(n_components=dims, batch_size=batch_size)
    for _, xb in _iter_aligned_batches(x, batch_size, proj):
        ipca.partial_fit(xb)

    xt = np.empty((x.shape[0], dims), dtype=np.float32)
    for start, xb in _iter_aligned_batches(x, batch_size, proj):
        xt[start : start + xb.shape[0]] = ipca.transform(xb).astype(np.float32)

    den = xt.max(0) - xt.min(0)
    smooth = np.zeros(dims, dtype=np.float64)
    edge_chunk = 250_000
    for start in range(0, len(edge_src), edge_chunk):
        es = edge_src[start : start + edge_chunk]
        ed = edge_dst[start : start + edge_chunk]
        diff = (xt[es] - xt[ed]) / den
        smooth += np.sum(diff * diff, axis=0)
    smooth /= max(len(edge_src), 1)
    return xt[:, np.argsort(smooth)]


def _feat_fingerprint(feat, dim):
    """Hash feature values, shape, and projection width for the SVD cache."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(dim).encode())
    if sp.issparse(feat):
        c = sp.csr_matrix(feat)
        h.update(str(c.shape).encode())
        for arr in (c.indptr, c.indices, np.ascontiguousarray(c.data, np.float64)):
            h.update(np.ascontiguousarray(arr).tobytes())
    else:
        a = np.ascontiguousarray(feat, np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _content_fingerprint(value, tag):
    """Hash exact dense/CSR content without densifying or dtype conversion."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(tag).encode("utf-8"))
    if sp.issparse(value):
        matrix = sp.csr_matrix(value, copy=False)
        h.update(str(matrix.shape).encode("utf-8"))
        for array in (matrix.indptr, matrix.indices, matrix.data):
            array = np.ascontiguousarray(array)
            h.update(str(array.dtype).encode("ascii"))
            h.update(memoryview(array).cast("B"))
    else:
        array = np.ascontiguousarray(value)
        h.update(str(array.shape).encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _load_valid_svd_cache(path, expected_shape, expected_dtype=np.float64):
    """Return a complete SVD cache entry, or ``None`` for stale/corrupt data."""
    try:
        cached = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if cached.shape != expected_shape or cached.dtype != expected_dtype:
        return None
    return cached


def _atomic_save_npy(path, value):
    """Publish one ``.npy`` file atomically without sharing temp names."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.npy",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can transiently reject two simultaneous replacements of the
        # same destination even though both source files are complete.  Retry
        # that short sharing violation; this is still lock-free and atomic.
        for attempt in range(20):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def x_svd(feat, dim, cache=True):
    """Project features using NumPy SVD and an optional content-addressed cache."""
    dim = int(dim)
    expected_shape = (
        int(feat.shape[0]),
        min(dim, int(feat.shape[0]), int(feat.shape[1])),
    )
    path = None
    if cache:
        SVD_CACHE.mkdir(parents=True, exist_ok=True)
        path = SVD_CACHE / f"svd{dim}_{_feat_fingerprint(feat, dim)}.npy"
        if path.exists():
            cached = _load_valid_svd_cache(path, expected_shape)
            if cached is not None:
                return cached
    f = _dense(feat)
    U, S, _ = np.linalg.svd(f, full_matrices=False)
    out = U[:, :dim] @ np.diag(S[:dim])
    if path is not None:
        _atomic_save_npy(path, out)
    return out


def x_svd_torch(feat, dim, cache=True):
    """UNPrompt released float32 ``torch.linalg.svd`` with a separate cache."""
    dim = int(dim)
    expected_shape = (
        int(feat.shape[0]),
        min(dim, int(feat.shape[0]), int(feat.shape[1])),
    )
    path = None
    if cache:
        SVD_TORCH_CACHE.mkdir(parents=True, exist_ok=True)
        path = SVD_TORCH_CACHE / (f"svd{dim}_{_feat_fingerprint(feat, dim)}.npy")
        if path.exists():
            cached = _load_valid_svd_cache(path, expected_shape, expected_dtype=np.float32)
            if cached is not None:
                return cached
    values = torch.from_numpy(np.ascontiguousarray(_dense32(feat)))
    U, S, _ = torch.linalg.svd(values, full_matrices=False)
    out = (U[:, :dim] * S[:dim]).numpy()
    if path is not None:
        _atomic_save_npy(path, out)
    return out


def load_aligned(name, target=False, hops=2):
    """Load graph plus ARC/IA-GGAD-aligned features.

    Both source and target loaders keep the complete benchmark graph; target
    evaluation is restricted only by its explicit mark.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    if target:
        adj, feat_raw, label, mark = load_target_marked(name, hops=hops)
    else:
        adj, feat_raw, label, mark = load_source_marked(name)
    coo = adj.tocoo()
    safe_name = name.replace("/", "_").replace("\\", "_")
    feature_digest = _content_fingerprint(
        feat_raw, f"arc-align-float32-d{DIMS}-rownorm{int(name in ROWNORM)}"
    )
    graph_digest = _content_fingerprint(adj, "arc-align-graph-v1")
    cache = CACHE / f"{safe_name}_{feature_digest}_{graph_digest}.npy"
    if cache.exists():
        feat = np.load(cache, allow_pickle=False)
        if feat.shape != (adj.shape[0], DIMS) or feat.dtype != np.float32:
            feat = None
    else:
        feat = None
    if feat is None:
        if name in ROWNORM:
            preprocess_features = _iaggad_preprocess_ops()[3]
            feat = preprocess_features(sp.lil_matrix(feat_raw))
        else:
            feat = np.asarray(sp.lil_matrix(feat_raw).todense(), dtype=np.float32)
        feat = np.ascontiguousarray(feat, dtype=np.float32)
        if name in BIG or adj.shape[0] > 1_000_000:
            feat = _feat_alignment_large(feat, coo.row, coo.col, DIMS)
        else:
            feat_alignment = _iaggad_preprocess_ops()[1]
            feat = feat_alignment(torch.from_numpy(feat), coo.row, coo.col, DIMS)
        feat = np.ascontiguousarray(feat, dtype=np.float32)
        _atomic_save_npy(cache, feat)
    return adj, feat, label, mark


def feat_align_cached(name):
    """Return cached ARC-aligned source features."""
    return load_aligned(name, target=False)[1]


def adj_sym_cond_scipy(name, adj):
    """Normalize adjacency using the dataset-specific self-loop convention."""
    normalize_adj_sym = _iaggad_preprocess_ops()[2]
    return (
        normalize_adj_sym(adj)
        if name in NO_SELFLOOP
        else normalize_adj_sym(adj + sp.eye(adj.shape[0]))
    )


def adj_sym_cond(name, adj, device):
    """Return normalized adjacency as a sparse tensor on the chosen device."""
    _to_torch_sparse = _iaggad_preprocess_ops()[0]
    return _to_torch_sparse(adj_sym_cond_scipy(name, adj)).to(device)


def _cond_selfloop_deg(name, adj):
    """Return adjacency and degrees after dataset-specific self-loop handling."""
    a_sl = (
        sp.csr_matrix(adj)
        if name in NO_SELFLOOP
        else (sp.csr_matrix(adj) + sp.eye(adj.shape[0], format="csr"))
    )
    deg = np.asarray(a_sl.sum(1)).flatten()
    return a_sl, deg


class EdgeList:
    """Store complete graph edges on CPU and stream device-sized blocks."""

    def __init__(self, adj):
        a = sp.coo_matrix(adj).astype(np.float32)
        self.row = torch.from_numpy(a.row.astype(np.int64))  # CPU
        self.col = torch.from_numpy(a.col.astype(np.int64))
        self.val = torch.from_numpy(a.data)
        self.shape = a.shape
        self.nnz = int(a.nnz)

    def chunks(self, device, chunk=EDGE_CHUNK):
        for s in range(0, self.nnz, chunk):
            e = min(s + chunk, self.nnz)
            yield (self.row[s:e].to(device), self.col[s:e].to(device), self.val[s:e].to(device))


class _ExactEdgeSpMM(torch.autograd.Function):
    """Exact differentiable ``A @ Z`` for a CPU-resident full edge list.

    Both passes stream every edge. Only one edge block and the node-level
    output are resident on the accelerator, matching the bounded large-graph
    schedule used by IA-GGAD without sampling nodes or edges.
    """

    @staticmethod
    def forward(ctx, z, row_cpu, col_cpu, val_cpu, n, chunk):
        ctx.row_cpu = row_cpu
        ctx.col_cpu = col_cpu
        ctx.val_cpu = val_cpu
        ctx.chunk = int(chunk)
        ctx.z_shape = tuple(z.shape)
        out = z.new_zeros((int(n), z.shape[1]))
        for start in range(0, val_cpu.numel(), ctx.chunk):
            end = min(start + ctx.chunk, val_cpu.numel())
            row = row_cpu[start:end].to(z.device)
            col = col_cpu[start:end].to(z.device)
            val = val_cpu[start:end].to(device=z.device, dtype=z.dtype).unsqueeze(1)
            out.index_add_(0, row, z[col] * val)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_z = grad_out.new_zeros(ctx.z_shape)
        for start in range(0, ctx.val_cpu.numel(), ctx.chunk):
            end = min(start + ctx.chunk, ctx.val_cpu.numel())
            row = ctx.row_cpu[start:end].to(grad_out.device)
            col = ctx.col_cpu[start:end].to(grad_out.device)
            val = (
                ctx.val_cpu[start:end].to(device=grad_out.device, dtype=grad_out.dtype).unsqueeze(1)
            )
            grad_z.index_add_(0, col, grad_out[row] * val)
        return grad_z, None, None, None, None, None


def exact_edge_spmm(edges, z, val=None, chunk=250_000):
    """Shared IA-GGAD/DRGGAD full-edge differentiable propagation."""
    if not isinstance(edges, EdgeList):
        raise TypeError("exact_edge_spmm expects a CPU-resident EdgeList")
    values = edges.val if val is None else val
    return _ExactEdgeSpMM.apply(z, edges.row, edges.col, values, edges.shape[0], int(chunk))


def exact_edge_spmm_transpose(edges, z, val=None, chunk=250_000):
    """Exact differentiable ``A.T @ Z`` using the same full-edge scheduler."""
    if not isinstance(edges, EdgeList):
        raise TypeError("exact_edge_spmm_transpose expects a CPU-resident EdgeList")
    values = edges.val if val is None else val
    return _ExactEdgeSpMM.apply(z, edges.col, edges.row, values, edges.shape[1], int(chunk))


def column_degree_row_values(edges, cache_attr="_column_degree_row_values"):
    """Return ``A[r,c] / column_degree[r]`` on CPU.

    This intentionally mirrors the official IA-GGAD/DRGGAD max-message code:
    degree is accumulated over edge columns, then indexed by edge rows.
    """
    if not isinstance(edges, EdgeList):
        raise TypeError("column_degree_row_values expects EdgeList")
    if not hasattr(edges, cache_attr):
        degree = torch.zeros(edges.shape[0], dtype=edges.val.dtype)
        degree.index_add_(0, edges.col, edges.val)
        inv = torch.where(degree > 0, degree.reciprocal(), torch.zeros_like(degree))
        setattr(edges, cache_attr, edges.val * inv[edges.row])
    return getattr(edges, cache_attr)


def edge_chunks(adj, device, chunk=EDGE_CHUNK):
    """Iterate over every edge in a CPU edge list or a sparse tensor."""
    if isinstance(adj, EdgeList):
        yield from adj.chunks(device, chunk)
    else:
        a = adj.coalesce()
        idx, val = a.indices(), a.values()
        E = val.numel()
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            yield idx[0, s:e], idx[1, s:e], val[s:e]


def chunk_propagate(name, adj, feat_np, num_hops, device, chunk=EDGE_CHUNK):
    """Compute normalized graph propagation while streaming every edge."""
    n = adj.shape[0]
    a = sp.coo_matrix(adj)
    _, deg = _cond_selfloop_deg(name, adj)
    nz = deg > 0
    dinv = np.zeros(n, dtype=np.float32)
    dinv[nz] = deg[nz] ** -0.5
    dinv_t = torch.from_numpy(dinv).to(device)
    self_t = None
    if name not in NO_SELFLOOP:
        self_np = np.zeros(n, dtype=np.float32)
        self_np[nz] = 1.0 / deg[nz]
        self_t = torch.from_numpy(self_np).to(device)
    row_cpu = torch.from_numpy(a.row.astype(np.int64))
    col_cpu = torch.from_numpy(a.col.astype(np.int64))
    val_cpu = torch.from_numpy(a.data.astype(np.float32))
    E = a.nnz

    x = torch.from_numpy(np.ascontiguousarray(feat_np, np.float32)).to(device)
    x_list = [x]
    for _ in range(num_hops):
        cur = x_list[-1]
        out = torch.zeros_like(cur)
        for s in range(0, E, chunk):
            r = row_cpu[s : s + chunk].to(device)
            c = col_cpu[s : s + chunk].to(device)
            w = (dinv_t[r] * dinv_t[c] * val_cpu[s : s + chunk].to(device)).unsqueeze(
                1
            )  # d_r·d_c·A[r,c]
            out.index_add_(0, c, cur[r] * w)
        if self_t is not None:
            out = out + cur * self_t.unsqueeze(1)
        x_list.append(out)
    return x_list


def aggregate(per):
    """Summarize AUROC and AUPRC across successful seed runs for each target."""
    out = {}
    for name, lst in per.items():
        if not lst:
            out[name] = {
                "AUROC_mean": float("nan"),
                "AUROC_std": float("nan"),
                "AUPRC_mean": float("nan"),
                "AUPRC_std": float("nan"),
                "n": 0,
            }
            continue
        a = np.array([d["AUROC"] for d in lst])
        p = np.array([d["AUPRC"] for d in lst])
        out[name] = {
            "AUROC_mean": float(a.mean()),
            "AUROC_std": float(a.std()),
            "AUPRC_mean": float(p.mean()),
            "AUPRC_std": float(p.std()),
            "n": len(lst),
        }
    return out
