"""Zero-GAD model and source-transfer implementation.

This ports the released Zero-GAD default path without importing its utils.py,
which pulls in DGL/UMAP/GAT modules that are not needed by the paper's default
GCN experiment.  The implemented path follows the official code:

  - every graph feature matrix is projected by truncated SVD to 8 dimensions;
  - small graphs receive the released Laplacian Fourier feature transform;
  - graphs above the configured Fourier size limit are marked unsupported in
    strict mode because dense decomposition is infeasible at benchmark scale;
  - a GCN encoder/decoder is trained by SCE reconstruction plus the released
    middle-feature variance term;
  - target score is 1 - minmax(cosine(reconstruction, input)).
"""

from __future__ import annotations

import gc
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

import ggad.config as C
from common.data import (
    BIG,
    CACHE_ROOT,
    EDGE_CHUNK,
    EdgeList,
    _atomic_save_npy,
    aggregate,
    exact_edge_spmm,
    load_source_marked,
    load_target_marked,
)
from util import evaluate, set_seed, to_torch_sparse

CACHE = CACHE_ROOT / "zerogad_torch_svd"
IMPLEMENTATION_VERSION = "zerogad-source-exp-v5-large-graph-skip"


class UnsupportedFourierGraph(RuntimeError):
    pass


def _is_large_fourier_graph(name: str, num_nodes: int, hp: dict) -> bool:
    max_nodes = int(hp.get("fourier_max_nodes", 0))
    return name in BIG or (max_nodes > 0 and int(num_nodes) > max_nodes)


def _unsupported_message(name: str, num_nodes: int, hp: dict) -> str:
    max_nodes = int(hp.get("fourier_max_nodes", 0))
    return (
        f"{name} has {int(num_nodes)} nodes (Fourier limit={max_nodes}); exact "
        "Zero-GAD preprocessing requires a dense Laplacian eigendecomposition"
    )


def _dense_float64(feat):
    if sp.issparse(feat):
        return np.asarray(feat.toarray(), dtype=np.float64)
    return np.asarray(feat, dtype=np.float64)


def _fingerprint(feat, dim: int, version: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str((dim, version)).encode("utf-8"))
    if sp.issparse(feat):
        c = sp.csr_matrix(feat)
        h.update(str(c.shape).encode("utf-8"))
        for arr in (c.indptr, c.indices, np.ascontiguousarray(c.data, np.float64)):
            h.update(memoryview(np.ascontiguousarray(arr)).cast("B"))
    else:
        a = np.ascontiguousarray(feat, np.float64)
        h.update(str(a.shape).encode("utf-8"))
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _svd_project(feat, dim: int, cache: bool = True) -> np.ndarray:
    """Official float32 torch SVD: U[:, :d] @ diag(S[:d])."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"svd{dim}_{_fingerprint(feat, dim, 'svd-v2-torch-f32')}.npy"
    if cache and path.exists():
        cached = np.load(path, allow_pickle=False)
        expected = (feat.shape[0], min(dim, feat.shape[0], feat.shape[1]))
        if cached.shape == expected and cached.dtype == np.float32:
            return cached

    x = torch.as_tensor(_dense_float64(feat), dtype=torch.float32)
    # full_matrices=False is memory-safe and leaves the first min(N,D)
    # singular triplets used by the official expression unchanged.
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    out = (u[:, :dim] @ torch.diag(s[:dim])).cpu().numpy().astype(np.float32)
    if cache:
        _atomic_save_npy(path, out)
    return out


def _standardize_torch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


def _standardize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (
        (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, ddof=1, keepdims=True) + 1e-8)
    ).astype(np.float32)


def _eigh_laplacian_upper(adj: sp.spmatrix, hp: dict):
    """Dense eigensolver matching torch.linalg.eigh(..., UPLO='U') semantics.

    Random induced subgraphs can contain many isolated nodes/components, so
    their Laplacians often have repeated eigenvalues.  torch's float32 eigh can
    fail to converge there; SciPy double precision is more stable and remains
    mathematically the same dense spectral step.
    """
    a = np.asarray(adj.toarray(), dtype=np.float64)
    deg = a.sum(axis=1)
    lap = np.diag(deg) - a
    upper = np.triu(lap)
    lap = upper + upper.T - np.diag(np.diag(upper))

    def _solve(mat):
        try:
            return sla.eigh(mat, check_finite=False, driver=hp.get("fourier_scipy_driver", "evd"))
        except TypeError:
            return sla.eigh(mat, check_finite=False)

    try:
        return _solve(lap)
    except Exception:
        jitter = float(hp.get("fourier_jitter", 0.0))
        if jitter <= 0.0:
            raise
        scale = max(float(np.abs(np.diag(lap)).mean()), 1.0)
        ramp = np.linspace(0.0, jitter * scale, lap.shape[0], dtype=np.float64)
        return _solve(lap + np.diag(ramp))


def _fourier_transform(adj: sp.spmatrix, feat_np: np.ndarray, hp: dict) -> np.ndarray:
    """Released standard-Laplacian Fourier transform for one manageable graph."""
    a = torch.as_tensor(np.asarray(adj.toarray(), dtype=np.float32))
    lap = torch.diag(a.sum(dim=1)) - a
    try:
        eigvals, eigvecs = torch.linalg.eigh(lap, UPLO="U")
        order = torch.argsort(eigvals)
        u = eigvecs[:, order]
        freq = _standardize_torch(u.T @ torch.as_tensor(feat_np, dtype=torch.float32))
        return (u @ freq).cpu().numpy().astype(np.float32, copy=False)
    except RuntimeError as exc:
        if hp.get("fourier_failure_policy", "raise") == "raise":
            raise RuntimeError(
                "official Zero-GAD torch.linalg.eigh failed; strict input "
                "mode does not substitute another spectral transform"
            ) from exc
        print(
            f"    [zerogad/preprocess] torch eigh failed; exact-shape "
            f"SciPy float64 fallback ({str(exc)[:80]})",
            flush=True,
        )
        # Release the failed dense float32 attempt before allocating the
        # float64 SciPy matrices; official partitions can still approach 10k
        # nodes and otherwise retain several large dense buffers at once.
        del lap, a
        gc.collect()
        eigvals, eigvecs = _eigh_laplacian_upper(adj, hp)
        order = np.argsort(eigvals)
        u = np.asarray(eigvecs[:, order], dtype=np.float64)
        freq = _standardize_np(u.T @ np.asarray(feat_np, dtype=np.float64))
        return np.asarray(u @ freq, dtype=np.float32)


def _fourier_partition_fallback(
    name: str, adj: sp.spmatrix, feat_np: np.ndarray, hp: dict
) -> np.ndarray:
    try:
        return _fourier_transform(adj, feat_np, hp)
    except Exception as exc:
        if hp.get("fourier_failure_policy", "raise") == "raise":
            raise
        min_nodes = int(hp.get("fourier_min_partition_nodes", 512))
        if feat_np.shape[0] <= min_nodes:
            print(
                f"    [zerogad/preprocess] {name}: both eigensolvers failed on "
                f"partition size={feat_np.shape[0]}; standardized SVD fallback "
                f"({str(exc)[:80]})",
                flush=True,
            )
            return _standardize_np(feat_np)
        out = np.empty_like(feat_np, dtype=np.float32)
        for part, nodes in enumerate(np.array_split(np.arange(feat_np.shape[0]), 2)):
            sub_adj = sp.csr_matrix(adj)[nodes][:, nodes]
            out[nodes] = _fourier_partition_fallback(
                f"{name}_retry{part}", sub_adj, feat_np[nodes], hp
            )
        return out


def _partition_fourier(name: str, adj: sp.spmatrix, feat_np: np.ndarray, hp: dict) -> np.ndarray:
    n = feat_np.shape[0]
    parts = int(hp["fourier_partitions"])
    rng = hp.get("_partition_rng")
    if rng is None:
        rng = random.Random(int(hp["feature_seed"]))
    perm = list(range(n))
    rng.shuffle(perm)
    out = np.empty_like(feat_np, dtype=np.float32)
    for i in range(parts):
        nodes = np.asarray(perm[i::parts], dtype=np.int64)
        sub_adj = sp.csr_matrix(adj)[nodes][:, nodes]
        out[nodes] = _fourier_partition_fallback(f"{name}_part{i}", sub_adj, feat_np[nodes], hp)
    return out


def _maybe_fourier(name: str, adj: sp.spmatrix, feat_np: np.ndarray, hp: dict) -> np.ndarray:
    if not hp.get("fourier", True):
        return feat_np.astype(np.float32, copy=False)
    if _is_large_fourier_graph(name, feat_np.shape[0], hp):
        policy = hp.get("big_fourier_policy", "strict_skip")
        if policy == "svd_only_extension":
            print(
                f"    [zerogad/preprocess] {name}: SVD-only large-graph extension; "
                f"not strict Zero-GAD",
                flush=True,
            )
            return feat_np.astype(np.float32, copy=False)
        raise UnsupportedFourierGraph(_unsupported_message(name, feat_np.shape[0], hp))
    partition_names = {
        item.lower().replace("_", "")
        for item in hp.get("fourier_partition_datasets", ["tfinance", "elliptic", "questions"])
    }
    if name.lower().replace("_", "") in partition_names:
        print(
            f"    [zerogad/preprocess] {name}: official 5-part Fourier " f"n={feat_np.shape[0]}",
            flush=True,
        )
        return _partition_fourier(name, adj, feat_np, hp)
    return _fourier_transform(adj, feat_np, hp)


def _row_normalized_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    a = sp.coo_matrix(adj, dtype=np.float32)
    row_sum = np.asarray(sp.csr_matrix(a).sum(1)).ravel().astype(np.float32)
    inv = np.zeros_like(row_sum)
    np.divide(1.0, row_sum + 1e-6, out=inv, where=row_sum > 0)
    return sp.csr_matrix((a.data * inv[a.row], (a.row, a.col)), shape=a.shape, dtype=np.float32)


class RowNormAdj:
    def __init__(self, adj: sp.spmatrix, device: str, hp: dict):
        norm = _row_normalized_adj(adj)
        self.stream = norm.nnz > int(hp["stream_edge_threshold"]) or norm.shape[0] > int(
            hp["stream_node_threshold"]
        )
        self.edge_chunk = int(hp.get("edge_chunk", EDGE_CHUNK))
        if self.stream:
            self.edges = EdgeList(norm)
            self.sparse = None
        else:
            self.edges = None
            self.sparse = to_torch_sparse(norm).to(device)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse is not None:
            return torch.sparse.mm(self.sparse, x)
        return exact_edge_spmm(self.edges, x, chunk=self.edge_chunk)


def _activation(name: str | None):
    if name == "prelu":
        return nn.PReLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name is None:
        return nn.Identity()
    raise NotImplementedError(name)


def _norm(name: str | None):
    if name == "layernorm":
        return nn.LayerNorm
    if name == "batchnorm":
        return nn.BatchNorm1d
    if name is None:
        return None
    raise NotImplementedError(name)


class GraphConv(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, activation=None, residual: bool = False, norm=None
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if residual:
            self.res_fc = (
                nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
            )
        else:
            self.res_fc = None
        self.norm = norm(out_dim) if norm is not None else None
        self.act = activation
        # The released GraphConv explicitly resets its Linear modules after
        # nn.Linear has already initialized them. This second initialization is
        # part of the official RNG sequence and materially changes every later
        # layer, so it must be preserved for reproduction.
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            self.res_fc.reset_parameters()

    def forward(self, x: torch.Tensor, adj: RowNormAdj) -> torch.Tensor:
        out = self.fc(adj.matmul(x))
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        if self.res_fc is not None:
            out = out + self.res_fc(x)
        return out


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        layers: int,
        dropout: float,
        activation: str | None,
        residual: bool,
        norm_name: str | None,
        encoding: bool,
    ):
        super().__init__()
        norm = _norm(norm_name)
        mods = []
        if layers == 1:
            mods.append(
                GraphConv(
                    in_dim,
                    out_dim,
                    activation=_activation(activation) if encoding else None,
                    residual=encoding and residual,
                    norm=norm if encoding else None,
                )
            )
        else:
            mods.append(
                GraphConv(
                    in_dim, hidden, activation=_activation(activation), residual=residual, norm=norm
                )
            )
            for _ in range(1, layers - 1):
                mods.append(
                    GraphConv(
                        hidden,
                        hidden,
                        activation=_activation(activation),
                        residual=residual,
                        norm=norm,
                    )
                )
            mods.append(
                GraphConv(
                    hidden,
                    out_dim,
                    activation=_activation(activation) if encoding else None,
                    residual=encoding and residual,
                    norm=norm if encoding else None,
                )
            )
        self.layers = nn.ModuleList(mods)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: RowNormAdj) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def _sce_loss(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()


class PreModel(nn.Module):
    def __init__(self, hp: dict):
        super().__init__()
        in_dim = int(hp["unifeat"])
        hidden = int(hp["num_hidden"])
        layers = int(hp["num_layers"])
        dropout = float(hp["in_drop"])
        residual = bool(hp["residual"])
        activation = hp["activation"]
        norm = hp.get("norm")
        self.encoder = GCN(
            in_dim, hidden, hidden, layers, dropout, activation, residual, norm, encoding=True
        )
        self.decoder = GCN(
            hidden,
            hidden,
            in_dim,
            int(hp["decoder_layers"]),
            dropout,
            activation,
            residual,
            norm,
            encoding=False,
        )
        # Unused by forward, but present in the released model and therefore
        # part of the official parameter initialization RNG sequence.
        self.hidden_output_layer = nn.Linear(hidden, 1)
        self.encoder_to_decoder = nn.Linear(hidden, hidden, bias=False)
        self.alpha_l = float(hp["alpha_l"])

    @staticmethod
    def variance_loss(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        return torch.mean((x - mean) ** 2)

    def forward(self, x: torch.Tensor, adj: RowNormAdj):
        mid = self.encoder(x, adj)
        mid = self.encoder_to_decoder(mid)
        loss_mid = self.variance_loss(mid)
        recon = self.decoder(mid, adj)
        return recon, loss_mid

    def criterion(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return _sce_loss(x, recon, self.alpha_l)


def _graph(name: str, hp: dict, device: str, target: bool):
    if name in BIG and hp.get("big_fourier_policy", "strict_skip") == "strict_skip":
        raise UnsupportedFourierGraph(_unsupported_message(name, -1, hp))
    adj, feat_raw, labels, mark = load_target_marked(name) if target else load_source_marked(name)
    if (
        _is_large_fourier_graph(name, len(labels), hp)
        and hp.get("big_fourier_policy", "strict_skip") == "strict_skip"
    ):
        raise UnsupportedFourierGraph(_unsupported_message(name, len(labels), hp))
    svd_feat = _svd_project(feat_raw, int(hp["unifeat"]), cache=bool(hp.get("cache", True)))
    feat = _maybe_fourier(name, sp.csr_matrix(adj), svd_feat, hp)
    x = torch.as_tensor(np.ascontiguousarray(feat, dtype=np.float32), device=device)
    return SimpleNamespace(
        name=name,
        x=x,
        adj_raw=sp.csr_matrix(adj),
        adj=RowNormAdj(adj, device, hp),
        labels=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
        n=len(labels),
    )


def _train_one(model: PreModel, graphs, epochs: int, hp: dict):
    opt = Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    beta = float(hp["beta"])
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        for graph in graphs:
            recon, loss_mid = model(graph.x, graph.adj)
            loss = model.criterion(graph.x, recon) + beta * loss_mid
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(hp.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["grad_clip"]))
            opt.step()
            total += float(loss.detach())
            del recon, loss_mid, loss
        if epoch == 0 or epoch + 1 == int(epochs) or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            print(
                f"    [zerogad/train] epoch={epoch + 1:03d}/{int(epochs)} "
                f"loss={total / max(1, len(graphs)):.5f}",
                flush=True,
            )


def _minmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    den = x.max() - x.min()
    if den <= 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / den


@torch.no_grad()
def _score_graph(model: PreModel, graph) -> np.ndarray:
    model.eval()
    recon, _ = model(graph.x, graph.adj)
    sim = (F.normalize(recon, p=2, dim=-1) * F.normalize(graph.x, p=2, dim=-1)).sum(dim=1)
    return (1.0 - _minmax_np(sim.detach().cpu().numpy())).astype(np.float32)


def _release_graph(graph):
    del graph.x, graph.adj


def run_zerogad(sources, targets, seeds, epochs, device, hp=None, target_evaluator=None):
    hp = dict(C.ZEROGAD_HP if hp is None else hp)
    hp["edge_chunk"] = int(hp.get("edge_chunk", EDGE_CHUNK))
    hp["_partition_rng"] = random.Random(int(hp["feature_seed"]))
    print(f"    [zerogad/version] {IMPLEMENTATION_VERSION}", flush=True)
    source_graphs = []
    for source in sources:
        print(f"    [zerogad/build-source] {source}", flush=True)
        try:
            source_graphs.append(_graph(source, hp, device, target=False))
        except UnsupportedFourierGraph as exc:
            print(f"    [zerogad/build-source] {source} skipped: {exc}", flush=True)
    if not source_graphs:
        print("    [zerogad] no Fourier-compatible source graphs; all targets skipped", flush=True)
        return aggregate({name: [] for name in targets})

    trained = []
    for seed in seeds:
        set_seed(seed)
        model = PreModel(hp).to(device)
        _train_one(model, source_graphs, epochs, hp)
        model.eval()
        model.zero_grad(set_to_none=True)
        model.cpu()
        trained.append((seed, model))
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for graph in source_graphs:
        _release_graph(graph)
    source_graphs.clear()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    per = {name: [] for name in targets}
    for name in targets:
        print(f"    [zerogad/target] {name}", flush=True)
        try:
            graph = _graph(name, hp, device, target=True)
        except UnsupportedFourierGraph as exc:
            print(f"    [zerogad/target] {name} unsupported in strict mode: {exc}", flush=True)
            continue
        for seed, model in trained:
            set_seed(seed)
            model.to(device)
            scores = _score_graph(model, graph)
            per[name].append(
                target_evaluator(name, int(seed), graph.labels, scores, graph.mark)
                if target_evaluator is not None
                else evaluate(graph.labels, scores, graph.mark)
            )
            model.cpu()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        _release_graph(graph)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return aggregate(per)
