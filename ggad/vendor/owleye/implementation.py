"""OWLEYE model and source-transfer implementation.

This ports the official demo code in ICLR-2026-OWLEYE-main without depending
on torch_geometric's Data container.  The behavioral path follows the released
zero-shot main.py/train_test.py/model.py:

  - OWLEYE-specific 64-dim PCA feature alignment;
  - raw-adjacency symmetric normalization without adding self-loops;
  - train/test feature scale normalization before final propagation;
  - multi-domain normal pattern dictionaries plus target in-context patterns;
  - cosine/triplet training loss and reconstruction-distance anomaly scores.
"""

from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torch.nn.parameter import Parameter
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

CACHE = CACHE_ROOT / "owleye_pca"
IMPLEMENTATION_VERSION = "owleye-source-exp-v3-official-input"


def _dense32(feat):
    if sp.issparse(feat):
        return np.asarray(feat.toarray(), dtype=np.float32)
    return np.asarray(feat, dtype=np.float32)


def _fingerprint(name: str, feat, dims: int, version: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str((name, dims, version)).encode("utf-8"))
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


def _preprocess_features_sparse(feat):
    x = sp.csr_matrix(feat)
    rowsum = np.asarray(x.sum(1)).ravel()
    inv = np.zeros_like(rowsum)
    np.divide(1.0, rowsum, out=inv, where=rowsum != 0)
    return sp.diags(inv).dot(x).toarray().astype(np.float32, copy=False)


def _feature_batch(feat, start: int, end: int, row_norm: bool) -> np.ndarray:
    block = feat[start:end]
    if sp.issparse(block):
        block = block.toarray()
    block = np.asarray(block)
    if row_norm:
        rowsum = block.sum(axis=1)
        inv = np.zeros_like(rowsum)
        np.divide(1.0, rowsum, out=inv, where=rowsum != 0)
        block = block * inv[:, None]
    return np.asarray(block, dtype=np.float32)


def _batch_bounds(n: int, batch_size: int, min_last: int):
    bounds = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_last:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def _owleye_feat_large(feat_raw, dims: int, hp: dict, row_norm: bool) -> np.ndarray:
    """Official GRP256->PCA64 alignment with bounded host memory.

    PCA is solved exactly from the centered feature covariance.  Raw features
    and the optional 256-d projection are materialized one batch at a time;
    no rows, nodes, or feature coordinates are sampled.
    """
    n, raw_dim = feat_raw.shape
    batch_size = max(int(hp["pca_batch"]), dims)
    bounds = _batch_bounds(n, batch_size, dims)
    projector = None
    if raw_dim < dims:
        projector = GaussianRandomProjection(n_components=256, random_state=0)
        projector.fit(np.zeros((1, raw_dim), dtype=np.float32))

    def batches():
        for start, end in bounds:
            block = _feature_batch(feat_raw, start, end, row_norm)
            if projector is not None:
                block = projector.transform(block).astype(np.float32, copy=False)
            yield start, end, block

    projected_dim = 256 if projector is not None else raw_dim
    feature_sum = np.zeros(projected_dim, dtype=np.float64)
    feature_xtx = np.zeros((projected_dim, projected_dim), dtype=np.float64)
    for _start, _end, block in batches():
        block64 = np.asarray(block, dtype=np.float64)
        feature_sum += block64.sum(axis=0)
        feature_xtx += block64.T @ block64

    mean = feature_sum / max(n, 1)
    centered_xtx = feature_xtx - n * np.outer(mean, mean)
    centered_xtx = (centered_xtx + centered_xtx.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(centered_xtx)
    order = np.argsort(eigvals)[::-1][:dims]
    components = eigvecs[:, order].T
    # sklearn's PCA applies svd_flip with signs selected from Vt rows.
    max_abs_cols = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), max_abs_cols])
    signs[signs == 0] = 1.0
    components *= signs[:, None]

    out = np.empty((n, dims), dtype=np.float32)
    for start, end, block in batches():
        transformed = (np.asarray(block, dtype=np.float64) - mean) @ components.T
        out[start:end] = transformed.astype(np.float32, copy=False)
    return out


def _owleye_feat(name: str, feat_raw, dims: int, hp: dict) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    row_norm = name in set(hp.get("row_norm_datasets", []))
    is_large = feat_raw.shape[0] > int(hp["large_pca_threshold"])
    algorithm = f"covariance-batch{int(hp['pca_batch'])}" if is_large else "sklearn-pca"
    version = f"feat-v4-official-input-{algorithm}-rownorm{int(row_norm)}"
    path = CACHE / f"{name}_{dims}_{_fingerprint(name, feat_raw, dims, version)}.npy"
    if hp.get("cache", True) and path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.shape == (feat_raw.shape[0], dims) and cached.dtype == np.float32:
            return cached

    if is_large:
        out = _owleye_feat_large(feat_raw, dims, hp, row_norm)
        if hp.get("cache", True):
            _atomic_save_npy(path, out)
        return out

    if row_norm:
        x = _preprocess_features_sparse(feat_raw)
    else:
        x = _dense32(feat_raw)

    if x.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        x = transformer.fit_transform(x).astype(np.float32, copy=False)

    pca = PCA(n_components=dims, random_state=0)
    out = pca.fit_transform(x).astype(np.float32, copy=False)

    if hp.get("cache", True):
        _atomic_save_npy(path, out)
    return out


def _sym_normalize_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    a = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.asarray(a.sum(1)).ravel()
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    np.divide(1.0, np.sqrt(rowsum), out=inv_sqrt, where=rowsum > 0)
    values = a.data * inv_sqrt[a.row] * inv_sqrt[a.col]
    # Official normalize_adj: adj.dot(D).transpose().dot(D) = D A^T D.
    return sp.csr_matrix((values, (a.col, a.row)), shape=a.shape, dtype=np.float32)


class SymNormAdj:
    def __init__(self, adj: sp.spmatrix, device: str, hp: dict):
        norm = _sym_normalize_adj(adj)
        self.edge_chunk = int(hp.get("edge_chunk", EDGE_CHUNK))
        self.stream = norm.nnz > int(hp["stream_edge_threshold"]) or norm.shape[0] > int(
            hp["stream_node_threshold"]
        )
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


def _load_graph(name: str, hp: dict, target: bool):
    adj, feat_raw, labels, mark = load_target_marked(name) if target else load_source_marked(name)
    feat = _owleye_feat(name, feat_raw, int(hp["in_feats"]), hp)
    return SimpleNamespace(
        name=name,
        adj_raw=sp.csr_matrix(adj),
        feat_np=np.asarray(feat, dtype=np.float32),
        labels_np=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
        n=int(len(labels)),
    )


def _attach_runtime(graph, device: str, hp: dict):
    graph.feat = torch.as_tensor(
        np.ascontiguousarray(graph.feat_np, dtype=np.float32), device=device
    )
    graph.adj = SymNormAdj(graph.adj_raw, device, hp)
    graph.ano_labels = torch.as_tensor(graph.labels_np, dtype=torch.float32, device=device)
    graph.one_node_features = torch.ones(
        (graph.n, int(hp["st_dim"])), dtype=torch.float32, device=device
    )
    return graph


@torch.no_grad()
def _propagated(graph, device: str, hp: dict):
    if not hasattr(graph, "adj"):
        _attach_runtime(graph, device, hp)
    h = graph.feat
    x_list = []
    for _ in range(int(hp["num_hops"])):
        next_h = graph.adj.matmul(h)
        x_list.append(h.detach().cpu() if graph.name in BIG else h)
        h = next_h
    x_list.append(h.detach().cpu() if graph.name in BIG else h)
    graph.x_list = x_list
    if graph.name in BIG:
        graph.feat = graph.x_list[0]
    return graph


def _release_runtime(graph, keep_feat_np: bool = True):
    for attr in ("feat", "adj", "ano_labels", "one_node_features", "x_list"):
        if hasattr(graph, attr):
            delattr(graph, attr)
    if not keep_feat_np and hasattr(graph, "feat_np"):
        delattr(graph, "feat_np")


def _normalize_feature_scale(graph, hp: dict):
    """Apply the released OWLEYE normalization without its O(N^2) no-op.

    In official utils.normalization(), both pair-distance passes read the same
    stale graph.x_list: feat is rescaled between them, but propagated() is not
    called.  Therefore dist_i == dist_normalized and both medians are equal, so
    the square-root factor is exactly one.  This leaves mean row-L2
    normalization followed by max(1, tau), for every source and target graph.
    """
    feat = np.asarray(graph.feat_np, dtype=np.float32)
    mean_norm = float(np.linalg.norm(feat, axis=1).mean())
    if not np.isfinite(mean_norm) or mean_norm <= 0:
        mean_norm = 1.0
    graph.feat_np = np.ascontiguousarray(
        feat * (max(1.0, float(hp["tau"])) / mean_norm), dtype=np.float32
    )


class BilinearSim(nn.Module):
    def __init__(self, in_channels: int = 512, struct_in_channels: int = 512):
        super().__init__()
        self.sim = nn.Linear(in_channels, in_channels)
        self.struct_sim = nn.Linear(struct_in_channels, struct_in_channels)
        self.weight = Parameter(torch.empty((1, in_channels, in_channels)))
        self.bias = Parameter(torch.empty(1))
        self.temperature = 1.0

    def forward(self, patterns, _emb, struct_patterns, struct_emb):
        item_list = []
        for idx in range(len(patterns)):
            struct_out = self.struct_sim(struct_patterns[idx])
            logits = torch.matmul(struct_out, struct_emb.T)
            logits = F.softmax(logits, dim=0).max(dim=0)[0]
            item_list.append(logits)
        return torch.stack(item_list, dim=0)


class OWLEYE(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_layers=2,
        dropout_rate=0,
        activation="ReLU",
        beta=1,
        num_hops=4,
        st_dim=10,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = int(num_hops)
        self.h_feats = int(h_feats)
        self.st_dim = int(st_dim)
        struct_dim = self.st_dim * self.num_hops
        self.layers.append(nn.Linear(in_feats, h_feats))
        self.struct_layers = nn.ModuleList([nn.Linear(self.st_dim, self.st_dim)])
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))
        for _ in range(1, self.num_hops + 1):
            self.struct_layers.append(nn.Linear(self.st_dim, self.st_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.beta = float(beta)
        self.embedding_dim = int(h_feats) * self.num_hops
        self.Wq = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.Wq_struct = nn.Linear(struct_dim, struct_dim)
        self.Wk_struct = nn.Linear(struct_dim, struct_dim)
        self.domain_sim = BilinearSim(self.embedding_dim, struct_dim)
        self.criteria = nn.TripletMarginLoss(margin=0.2, p=2.0, eps=1e-6, swap=True)
        self.criteria_struct = nn.TripletMarginLoss(margin=0.1, p=2.0, eps=1e-6, swap=True)
        self.mask_ratio = 0.0
        self.temperature = 1.0

    def _attribute_embedding(self, x_list):
        device = next(self.parameters()).device
        x_list = [x.to(device) for x in x_list]
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]
        first = x_list[0]
        return torch.hstack([h_i - first for h_i in x_list[1:]])

    def get_struct_hops(self, graph):
        """Run the structure branch once on the complete graph.

        Every structural hop depends on all neighbors, so this part cannot be
        node-chunked.  SymNormAdj still streams the complete edge set in bounded
        blocks for BIG graphs.
        """
        struct_feat = graph.one_node_features
        struct_feat_list = []
        for i, layer in enumerate(self.struct_layers):
            out = graph.adj.matmul(layer(struct_feat))
            if i != len(self.struct_layers) - 1:
                out = self.act(out)
            struct_feat = out
            struct_feat_list.append(out)
        return struct_feat_list

    @staticmethod
    def _structure_embedding(struct_feat_list):
        first_struct = struct_feat_list[0]
        return torch.hstack([h_i - first_struct for h_i in struct_feat_list[1:]])

    def get_embedding(self, graph):
        attr = self._attribute_embedding(graph.x_list)
        struct = self._structure_embedding(self.get_struct_hops(graph))
        return attr, struct

    def get_embedding_rows(self, graph, rows, struct_hops):
        def take(x):
            index = rows.to(x.device) if torch.is_tensor(rows) else rows
            return x[index]

        attr = self._attribute_embedding([take(x) for x in graph.x_list])
        struct = self._structure_embedding([take(x) for x in struct_hops])
        return attr, struct

    def get_embedding_chunk(self, graph, start: int, end: int, struct_hops):
        return self.get_embedding_rows(graph, slice(start, end), struct_hops)

    def patterns_extraction(self, h_train, struct_emb, _adj_train, y_train, num_prompt=50):
        normal_indices = torch.nonzero((y_train == 0)).squeeze(1).tolist()
        if len(normal_indices) > num_prompt:
            normal_indices = random.sample(normal_indices, num_prompt)
        return h_train[normal_indices], struct_emb[normal_indices]

    def patterns_extraction_for_test_graph(self, h_test, struct_emb, num_prompt=10):
        n = h_test.shape[0]
        selected = random.sample(list(range(n)), min(num_prompt, n))
        return h_test[selected], struct_emb[selected]

    def forward(
        self, h_train, struct_train, wl_pos, y_train, patterns_list, struct_patterns, num_prompt=50
    ):
        del wl_pos
        anomaly_indices = torch.nonzero((y_train == 1)).squeeze(1).tolist()
        normal_pool = torch.nonzero((y_train == 0)).squeeze(1).tolist()
        num_prompt = min(num_prompt, len(anomaly_indices), len(normal_pool))
        if num_prompt <= 0:
            return h_train.sum() * 0.0
        normal_indices = random.sample(normal_pool, num_prompt)
        if len(anomaly_indices) > num_prompt:
            anomaly_indices = random.sample(anomaly_indices, num_prompt)
        anomaly_emb = h_train[anomaly_indices]
        normal_emb = h_train[normal_indices]
        y_positive = torch.ones([len(normal_indices)], device=y_train.device)
        y_negative = -torch.ones([len(anomaly_indices)], device=y_train.device)
        struct_normal = struct_train[normal_indices]
        struct_anomaly = struct_train[anomaly_indices]
        normal_dom = self.domain_sim(patterns_list, normal_emb, struct_patterns, struct_normal)
        anomaly_dom = self.domain_sim(patterns_list, anomaly_emb, struct_patterns, struct_anomaly)
        tilde_normal = self.cross_attention(normal_emb, patterns_list, normal_dom, self.Wq, self.Wk)
        tilde_anomaly = self.cross_attention(
            anomaly_emb, patterns_list, anomaly_dom, self.Wq, self.Wk
        )
        tilde_struct = self.cross_attention(
            struct_normal, struct_patterns, normal_dom, self.Wq_struct, self.Wk_struct
        )
        loss = (
            F.cosine_embedding_loss(normal_emb, tilde_normal, y_positive)
            + F.cosine_embedding_loss(anomaly_emb, tilde_anomaly, y_negative)
            + F.cosine_embedding_loss(normal_emb, tilde_anomaly, y_negative)
        )
        loss = loss + self.criteria(tilde_normal, normal_emb, anomaly_emb)
        loss = loss + self.beta * self.criteria_struct(tilde_struct, struct_normal, struct_anomaly)
        return loss

    def inference(self, patterns, h_test, _adj, struct_patterns, struct_test):
        dom_sim = self.domain_sim(patterns, h_test, struct_patterns, struct_test)
        query = self.cross_attention(h_test, patterns, dom_sim, self.Wq, self.Wk)
        query_struct = self.cross_attention(
            struct_test, struct_patterns, dom_sim, self.Wq_struct, self.Wk_struct
        )
        score = torch.sqrt(torch.sum((query - h_test) ** 2, dim=1)) + self.beta * torch.sqrt(
            torch.sum((query_struct - struct_test) ** 2, dim=1)
        )
        return score, dom_sim

    def cross_attention(self, query_x, support_x, dom_sim, wq, wk):
        dom_sim = dom_sim.T
        q = F.leaky_relu(wq(query_x))
        emb_list = 0
        for idx in range(len(support_x)):
            k_proj = F.leaky_relu(wk(support_x[idx]))
            denom = torch.sqrt(
                torch.tensor(self.embedding_dim, dtype=torch.float32, device=query_x.device)
            )
            attention_scores = torch.matmul(q, k_proj.T) / denom
            k = max(1, int(attention_scores.shape[1] * 0.1))
            _, indices = torch.topk(attention_scores, k, dim=1, largest=False)
            mask = torch.ones_like(attention_scores)
            rows = torch.arange(attention_scores.size(0), device=query_x.device).unsqueeze(1)
            mask[rows, indices] = 0
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
            weights = F.softmax(attention_scores / self.temperature, dim=1)
            weighted = torch.matmul(weights, support_x[idx])
            emb_list += dom_sim[:, idx] * weighted.T
        return (emb_list / len(support_x)).T


def _init_model(hp: dict, device: str) -> OWLEYE:
    model = OWLEYE(
        in_feats=int(hp["in_feats"]),
        h_feats=int(hp["h_feats"]),
        num_layers=int(hp["num_layers"]),
        dropout_rate=0.0,
        activation=hp["activation"],
        beta=float(hp["beta"]),
        num_hops=int(hp["num_hops"]),
        st_dim=int(hp["st_dim"]),
    ).to(device)
    model.mask_ratio = float(hp["mask_ratio"])
    model.temperature = float(hp["temperature"])
    model.domain_sim.temperature = float(hp["temperature"])
    return model


def _extract_patterns(model: OWLEYE, graph, n_support: int):
    emb, struct = model.get_embedding(graph)
    return model.patterns_extraction(emb, struct, graph.adj, graph.ano_labels, num_prompt=n_support)


def _train_model(model: OWLEYE, train_graphs, epochs: int, hp: dict):
    optimizer = Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    patterns = {}
    struct_patterns = {}
    n_support = int(hp["n_support"])
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        for didx, graph in enumerate(train_graphs):
            emb, struct = model.get_embedding(graph)
            pat, spat = model.patterns_extraction(
                emb, struct, graph.adj, graph.ano_labels, num_prompt=n_support
            )
            patterns[didx] = pat.detach()
            struct_patterns[didx] = spat.detach()
            loss = model(
                emb,
                struct,
                graph.adj,
                graph.ano_labels,
                patterns,
                struct_patterns,
                num_prompt=n_support,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(hp.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["grad_clip"]))
            optimizer.step()
            total += float(loss.detach())

            emb, struct = model.get_embedding(graph)
            pat, spat = model.patterns_extraction(
                emb, struct, graph.adj, graph.ano_labels, num_prompt=n_support
            )
            patterns[didx] = pat.detach()
            struct_patterns[didx] = spat.detach()
            del emb, struct, pat, spat, loss
        if epoch == 0 or epoch + 1 == int(epochs) or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            print(
                f"    [owleye/train] epoch={epoch + 1:03d}/{int(epochs)} "
                f"loss={total / max(1, len(train_graphs)):.5f}",
                flush=True,
            )

    model.eval()
    patterns.clear()
    struct_patterns.clear()
    with torch.no_grad():
        for didx, graph in enumerate(train_graphs):
            pat, spat = _extract_patterns(model, graph, n_support)
            patterns[didx] = pat.detach()
            struct_patterns[didx] = spat.detach()
    return patterns, struct_patterns


@torch.no_grad()
def _score_target(model: OWLEYE, graph, patterns, struct_patterns, hp: dict):
    model.eval()
    chunk = int(hp["node_chunk"])
    scores = np.empty(graph.n, dtype=np.float32)
    local_key = len(patterns)
    struct_hops = model.get_struct_hops(graph)

    # Official inference draws target in-context patterns uniformly from the
    # complete target graph.  Compute only those rows after the one full
    # structure propagation, then reuse them for every query chunk.
    idx = random.sample(range(graph.n), min(int(hp["test_patterns"]), graph.n))
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=graph.x_list[0].device)
    pat, spat = model.get_embedding_rows(graph, idx_t, struct_hops)
    p = dict(patterns)
    spats = dict(struct_patterns)
    p[local_key] = pat.detach()
    spats[local_key] = spat.detach()
    for start in range(0, graph.n, chunk):
        end = min(start + chunk, graph.n)
        emb, struct = model.get_embedding_chunk(graph, start, end, struct_hops)
        score, _ = model.inference(p, emb, graph.adj, spats, struct)
        scores[start:end] = score.detach().cpu().numpy()
    del struct_hops
    return scores


def run_owleye(sources, targets, seeds, epochs, device, hp=None, target_evaluator=None):
    hp = dict(C.OWLEYE_HP if hp is None else hp)
    hp["edge_chunk"] = int(hp.get("edge_chunk", EDGE_CHUNK))
    print(f"    [owleye/version] {IMPLEMENTATION_VERSION}", flush=True)

    train_graphs = []
    for source in sources:
        print(f"    [owleye/build-source] {source}", flush=True)
        graph = _load_graph(source, hp, target=False)
        _normalize_feature_scale(graph, hp)
        _attach_runtime(graph, device, hp)
        _propagated(graph, device, hp)
        train_graphs.append(graph)

    per = {name: [] for name in targets}
    trained = []
    for seed in seeds:
        set_seed(seed)
        model = _init_model(hp, device)
        patterns, struct_patterns = _train_model(model, train_graphs, epochs, hp)
        model.cpu()
        patterns = {key: value.detach().cpu() for key, value in patterns.items()}
        struct_patterns = {key: value.detach().cpu() for key, value in struct_patterns.items()}
        trained.append(
            {
                "seed": seed,
                "model": model,
                "patterns": patterns,
                "struct_patterns": struct_patterns,
                "random_state": random.getstate(),
            }
        )
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for graph in train_graphs:
        _release_runtime(graph)
    train_graphs.clear()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # ARC-style target lifetime, specialized for OWLEYE: one complete target is
    # loaded, propagated, scored by every seed model, then released.  No target
    # is sampled and no two BIG targets coexist in accelerator memory.
    for name in targets:
        try:
            print(f"    [owleye/build-target] {name}", flush=True)
            graph = _load_graph(name, hp, target=True)
            _normalize_feature_scale(graph, hp)
            _attach_runtime(graph, device, hp)
            _propagated(graph, device, hp)
        except RuntimeError as exc:
            if name in BIG:
                raise
            print(f"    [owleye/build-target] {name} skipped: {exc}", flush=True)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        for run in trained:
            model = run["model"]
            model.to(device)
            random.setstate(run["random_state"])
            patterns = {key: value.to(device) for key, value in run["patterns"].items()}
            struct_patterns = {
                key: value.to(device) for key, value in run["struct_patterns"].items()
            }
            score = _score_target(model, graph, patterns, struct_patterns, hp)
            run["random_state"] = random.getstate()
            per[graph.name].append(
                target_evaluator(graph.name, int(run["seed"]), graph.labels_np, score, graph.mark)
                if target_evaluator is not None
                else evaluate(graph.labels_np, score, graph.mark)
            )
            model.cpu()
            del patterns, struct_patterns
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        _release_runtime(graph)
        del graph
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    trained.clear()
    return aggregate(per)
