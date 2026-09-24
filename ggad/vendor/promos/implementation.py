"""ProMoS model and source-transfer implementation.

This ports the official ProMoS training and scoring logic into the benchmark:

  - source graphs are exactly the ``sources`` selected by ggad/config.py;
  - node features use the benchmark's official-style 64-dim alignment;
  - the StudentMoE/router/prototype KL+VQ objective follows ProMoS/main_train.py;
  - targets are evaluated on every marked node with no target-node sampling.

A source-only GCA teacher supplies frozen embeddings in memory; the upstream
entry point loads embeddings generated offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

import ggad.config as C
from common.data import BIG, EDGE_CHUNK, NO_SELFLOOP, EdgeList, aggregate, edge_chunks, load_aligned
from ggad.vendor.promos.teacher import (
    embed_graph,
    pyg_edge_index,
    train_runtime_gca,
    undirected_binary_adj,
)
from util import evaluate, set_seed

_KMEANS_BACKEND_REPORTED = False


class MLPExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)
        self.route = nn.Linear(input_dim, self.num_experts)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.route.weight, gain=0.5)
        nn.init.zeros_(self.route.bias)

    def forward(self, x):
        logits = self.route(x)
        k = min(2, self.num_experts)
        top_k_scores, top_k_indices = torch.topk(logits, k, dim=1)
        probabilities = F.softmax(top_k_scores, dim=1)
        return top_k_indices, probabilities


def _similarity(h1, h2):
    h1_norm = (h1**2).sum(dim=1)
    h2_norm = (h2**2).sum(dim=1)
    dist_sq = h1_norm[:, None] + h2_norm[None, :] - 2 * h1 @ h2.T
    return -dist_sq


def sharpen(p, temperature=1):
    del temperature
    return p


def _faiss_rand_perm_prefix(n: int, count: int, seed: int) -> np.ndarray:
    """Return the needed prefix of Faiss ``rand_perm`` exactly.

    Faiss uses ``std::mt19937`` and an in-place Fisher-Yates permutation.  A
    NumPy RandomState exposes the same MT19937 uint32 stream.  Later swaps can
    no longer alter an already-produced prefix, so stopping after ``count``
    swaps is exact and avoids an O(n) Python loop.
    """
    if count < 0 or count > n:
        raise ValueError(f"invalid permutation prefix count={count} for n={n}")
    perm = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    for i in range(count):
        raw = int(rng.randint(0, 1 << 32, dtype=np.uint32))
        j = i + raw % (n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm[:count]


def _faiss_split_empty_clusters(counts: np.ndarray, centroids: np.ndarray) -> None:
    """Port Faiss ``split_clusters`` for unweighted float32 KMeans."""
    k, dim = centroids.shape
    n = int(counts.sum())
    rng = np.random.RandomState(1234)
    eps = np.float32(1.0 / 1024.0)
    plus = np.ones(dim, dtype=np.float32)
    minus = np.ones(dim, dtype=np.float32)
    plus[0::2] += eps
    plus[1::2] -= eps
    minus[0::2] -= eps
    minus[1::2] += eps

    for empty in np.flatnonzero(counts == 0):
        donor = 0
        found = False
        for _ in range(10 * k):
            probability = np.float32(
                (np.float32(counts[donor]) - np.float32(1.0)) / np.float32(n - k)
            )
            raw = int(rng.randint(0, 1 << 32, dtype=np.uint32))
            draw = np.float32(raw) / np.float32((1 << 32) - 1)
            if draw < probability:
                found = True
                break
            donor = (donor + 1) % k
        if not found:
            donor = int(np.argmax(counts))

        original = centroids[donor].copy()
        centroids[empty] = original * plus
        centroids[donor] = original * minus
        counts[empty] = counts[donor] / np.float32(2.0)
        counts[donor] -= counts[empty]


def _faiss_compatible_kmeans(
    x_np: np.ndarray,
    k: int,
    seed: int,
    niter: int,
    max_points_per_centroid: int,
    assignment_chunk: int,
) -> np.ndarray:
    """Dependency-free port of the defaults used by ``faiss.Kmeans``."""
    x_np = np.ascontiguousarray(x_np, dtype=np.float32)
    if not np.isfinite(x_np).all():
        raise ValueError("ProMoS KMeans input contains NaN or Inf")
    if x_np.shape[0] < k:
        raise ValueError(f"KMeans needs at least k={k} rows, got {x_np.shape[0]}")

    sample_count = int(k) * int(max_points_per_centroid)
    if x_np.shape[0] > sample_count:
        # Faiss subsamples with seed, then initializes with seed + 1.
        selected = _faiss_rand_perm_prefix(x_np.shape[0], sample_count, seed)
        x_np = np.ascontiguousarray(x_np[selected])
    if x_np.shape[0] == k:
        return x_np.copy()

    initial = _faiss_rand_perm_prefix(x_np.shape[0], k, seed + 1)
    centroids = x_np[initial].copy()
    previous_objective = None

    for _ in range(int(niter)):
        assignments = np.empty(x_np.shape[0], dtype=np.int64)
        objective = np.float32(0.0)
        for start in range(0, x_np.shape[0], int(assignment_chunk)):
            end = min(start + int(assignment_chunk), x_np.shape[0])
            diff = x_np[start:end, None, :] - centroids[None, :, :]
            distances = np.sum(diff * diff, axis=2, dtype=np.float32)
            local = np.argmin(distances, axis=1)
            assignments[start:end] = local
            objective += np.sum(distances[np.arange(end - start), local], dtype=np.float32)

        counts = np.bincount(assignments, minlength=k).astype(np.float32)
        updated = np.zeros_like(centroids)
        # np.add.at performs unbuffered additions in input-row order, matching
        # Faiss's per-centroid sequential float32 accumulation.
        np.add.at(updated, assignments, x_np)
        occupied = counts > 0
        updated[occupied] *= (np.float32(1.0) / counts[occupied, None]).astype(np.float32)
        _faiss_split_empty_clusters(counts, updated)
        centroids = updated

        # Current Faiss stops only when the float32 objective is unchanged.
        if previous_objective is not None and objective == previous_objective:
            break
        previous_objective = objective
    return centroids


def _kmeans_centroids(
    x_tensor: torch.Tensor, k: int, seed: int, max_points_per_centroid: int, assignment_chunk: int
) -> torch.Tensor:
    global _KMEANS_BACKEND_REPORTED
    x_np = x_tensor.detach().cpu().numpy().astype("float32", copy=False)
    centroids = _faiss_compatible_kmeans(
        x_np,
        int(k),
        int(seed),
        niter=20,
        max_points_per_centroid=int(max_points_per_centroid),
        assignment_chunk=int(assignment_chunk),
    )
    if not _KMEANS_BACKEND_REPORTED:
        cap = int(k) * int(max_points_per_centroid)
        print(
            "    [promos/kmeans] in-repo Faiss-compatible Lloyd; "
            f"20 iterations, seed={seed}, training cap={cap}",
            flush=True,
        )
        _KMEANS_BACKEND_REPORTED = True
    return torch.from_numpy(centroids).to(x_tensor.device)


def _combine_kmeans_and_xavier(
    t_feat: torch.Tensor,
    cluster_nums: List[int],
    dim: int,
    kmeans_seed: int,
    max_points_per_centroid: int,
    assignment_chunk: int,
):
    combined = nn.ParameterList()
    for k in cluster_nums:
        proto = _kmeans_centroids(
            t_feat, int(k), kmeans_seed, max_points_per_centroid, assignment_chunk
        )
        random_proto = F.normalize(torch.randn(int(k), dim, device=t_feat.device), dim=1)
        proto = F.normalize(proto, dim=1) * 0.5 + random_proto * 0.5
        combined.append(nn.Parameter(proto))
    return combined


class StudentMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        t_feat: torch.Tensor | None = None,
        num_experts: int = 20,
        expert_hidden_dim: int = 128,
        expert_output_dim: int = 64,
        prototypes_per_group_generalist: Sequence[int] = (20,),
        prototypes_per_group_specialized: Sequence[int] = (20,),
        dropout: float = 0.1,
        seed: int = 0,
        kmeans_seed: int = 1234,
        kmeans_max_points_per_centroid: int = 256,
        kmeans_assignment_chunk: int = 1024,
    ):
        super().__init__()
        if t_feat is None:
            t_feat = torch.randn(
                max(prototypes_per_group_generalist + prototypes_per_group_specialized),
                expert_output_dim,
            )
        self.output_dim = int(expert_output_dim)
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.num_experts = int(num_experts)
        self.generalist_experts = nn.ModuleList(
            [MLPExpert(input_dim, expert_hidden_dim, self.output_dim) for _ in range(1)]
        )
        self.adapter = nn.Linear(self.output_dim, self.output_dim)
        self.specialized_experts = nn.ModuleList(
            [
                MLPExpert(input_dim, expert_hidden_dim, self.output_dim)
                for _ in range(self.num_experts)
            ]
        )
        self.router_specialized = Router(input_dim, self.num_experts)
        self.prototype_groups_generalist = _combine_kmeans_and_xavier(
            t_feat,
            list(prototypes_per_group_generalist),
            self.output_dim,
            kmeans_seed,
            kmeans_max_points_per_centroid,
            kmeans_assignment_chunk,
        )
        self.prototype_groups_specialized = _combine_kmeans_and_xavier(
            t_feat,
            list(prototypes_per_group_specialized),
            self.output_dim,
            kmeans_seed,
            kmeans_max_points_per_centroid,
            kmeans_assignment_chunk,
        )
        self.dropout = nn.Dropout(float(dropout))

    def adp(self, x):
        return self.adapter(x)

    def forward(self, x):
        expert_outputs_generalist = x.new_zeros((x.size(0), self.output_dim))
        for expert in self.generalist_experts:
            expert_outputs_generalist = expert(self.dropout(x))

        expert_indices, probabilities = self.router_specialized(x)
        expert_outputs_specialized = x.new_zeros((x.size(0), self.output_dim))
        for i, expert in enumerate(self.specialized_experts):
            mask = expert_indices == i
            if mask.any():
                rows, cols = mask.nonzero(as_tuple=True)
                selected = x[rows]
                prob = probabilities[rows, cols]
                expert_outputs_specialized.index_add_(
                    0, rows, expert(self.dropout(selected)) * prob.unsqueeze(1)
                )

        expert_outputs_generalist = expert_outputs_generalist + x
        expert_outputs_specialized = expert_outputs_specialized + x

        dists_generalist = []
        for proto in self.prototype_groups_generalist:
            sim = _similarity(expert_outputs_generalist, proto)
            dists_generalist.append(sharpen(F.softmax(sim / self.temperature, dim=-1)))

        dists_specialized = []
        for proto in self.prototype_groups_specialized:
            sim = _similarity(expert_outputs_specialized, proto)
            dists_specialized.append(sharpen(F.softmax(sim / self.temperature, dim=-1)))

        return (
            dists_generalist,
            dists_specialized,
            expert_outputs_generalist,
            expert_outputs_specialized,
        )


def _compute_loss(student_dists, teacher_dists):
    loss = 0
    for s_dist, t_dist in zip(student_dists, teacher_dists):
        loss = loss + F.kl_div(torch.log(s_dist + 1e-8), t_dist, reduction="batchmean")
    return loss


@torch.no_grad()
def _compute_kl_score(student_dists, teacher_dists):
    score = 0
    for s_dist, t_dist in zip(student_dists, teacher_dists):
        kl = F.kl_div(torch.log(s_dist + 1e-8), t_dist, reduction="none").sum(dim=1)
        score = score + kl
    return score


def _teacher_dists_and_vq(
    model: StudentMoE, t_feat: torch.Tensor, prototype_groups, hp: dict, training: bool
):
    teacher_dists = []
    vq_loss = 0.0
    for proto in prototype_groups:
        sim = F.softmax(_similarity(t_feat, proto) / model.temperature, dim=-1)
        teacher_dists.append(sharpen(sim))
        _, nearest_idx = sim.max(dim=1)
        nearest_proto = proto[nearest_idx]

        if training:
            pro_sim = F.softmax(_similarity(proto, proto) / model.temperature, dim=-1)
            target_struct = pro_sim[nearest_idx]
            kl_res = (
                F.kl_div(torch.log(sim.clamp_min(1e-8)), target_struct, reduction="none")
                .sum(dim=1)
                .detach()
            )
            raw_weight = torch.sigmoid(-float(hp["beta"]) * (kl_res - float(hp["mu"])))
            weights = raw_weight / (raw_weight.sum() + 1e-8)
            mse = F.mse_loss(nearest_proto, t_feat.detach(), reduction="none").sum(dim=1)
            mse1 = F.mse_loss(nearest_proto.detach(), t_feat, reduction="none").sum(dim=1)
            vq_loss = vq_loss + (weights * mse).sum() + (weights * mse1).sum()
        else:
            mse = F.mse_loss(nearest_proto, t_feat.detach(), reduction="none")
            vq_loss = vq_loss + mse.mean(dim=1).detach()
    return teacher_dists, vq_loss


def _training_loss(model: StudentMoE, x_input: torch.Tensor, teacher_feat: torch.Tensor, hp: dict):
    t_feat = model.adp(teacher_feat.detach())
    student_gen, student_spe, _e_g, _e_s = model(x_input)
    teacher_gen, vq_gen = _teacher_dists_and_vq(
        model, t_feat, model.prototype_groups_generalist, hp, training=True
    )
    teacher_spe, vq_spe = _teacher_dists_and_vq(
        model, t_feat, model.prototype_groups_specialized, hp, training=True
    )
    return (
        _compute_loss(student_gen, teacher_gen)
        + _compute_loss(student_spe, teacher_spe)
        + (vq_gen + vq_spe) * float(hp["lam"])
    )


@torch.no_grad()
def _score_chunk(model: StudentMoE, x_input: torch.Tensor, teacher_feat: torch.Tensor, hp: dict):
    t_feat = model.adp(teacher_feat)
    student_gen, student_spe, e_g, e_s = model(x_input)

    teacher_gen, vq_gen = _teacher_dists_and_vq(
        model, t_feat, model.prototype_groups_generalist, hp, training=False
    )
    for idx, proto in enumerate(model.prototype_groups_generalist):
        _, nearest_idx = student_gen[idx].max(dim=1)
        nearest_proto = proto[nearest_idx]
        vq_gen = (
            vq_gen + F.mse_loss(nearest_proto, e_g.detach(), reduction="none").mean(dim=1).detach()
        )
    vq_gen = vq_gen / 2

    teacher_spe, vq_spe = _teacher_dists_and_vq(
        model, t_feat, model.prototype_groups_specialized, hp, training=False
    )
    for idx, proto in enumerate(model.prototype_groups_specialized):
        _, nearest_idx = student_spe[idx].max(dim=1)
        nearest_proto = proto[nearest_idx]
        vq_spe = (
            vq_spe + F.mse_loss(nearest_proto, e_s.detach(), reduction="none").mean(dim=1).detach()
        )
    vq_spe = vq_spe / 2

    return (
        _compute_kl_score(student_gen, teacher_gen)
        + _compute_kl_score(student_spe, teacher_spe)
        + (vq_gen + vq_spe) * float(hp["lam"])
    )


def _residual_plus_x(
    name: str, adj: sp.spmatrix, x_np: np.ndarray, device: str, hp: dict
) -> torch.Tensor:
    x = torch.as_tensor(np.ascontiguousarray(x_np, dtype=np.float32), device=device)
    support = undirected_binary_adj(adj, self_loops=name not in NO_SELFLOOP)
    if name in BIG or adj.nnz > int(hp.get("stream_edge_threshold", 5_000_000)):
        edges = EdgeList(support)
        neigh = torch.zeros_like(x)
        deg = torch.zeros(x.shape[0], dtype=x.dtype, device=device)
        for row, col, _val in edge_chunks(edges, device, int(hp["edge_chunk"])):
            if row.numel() == 0:
                continue
            one = torch.ones(row.numel(), dtype=x.dtype, device=device)
            neigh.index_add_(0, row, x[col])
            deg.index_add_(0, row, one)
        mean = neigh / deg.clamp_min(1.0).unsqueeze(1)
        return x + (x - mean)

    deg = np.asarray(support.sum(1)).reshape(-1, 1).astype(np.float32)
    deg[deg <= 0] = 1.0
    mean = support.dot(np.asarray(x_np, dtype=np.float32)) / deg
    return torch.as_tensor(np.asarray(x_np + (x_np - mean), dtype=np.float32), device=device)


def _graph(name: str, hp: dict, device: str, target: bool = False):
    adj, feat, labels, mark = load_aligned(name, target=target)
    adj = sp.csr_matrix(adj)
    feat = np.asarray(feat, dtype=np.float32)
    x = torch.as_tensor(np.ascontiguousarray(feat), device=device)
    teacher_stream = name in BIG or adj.nnz > int(hp.get("gca_stream_edge_threshold", 5_000_000))
    if teacher_stream:
        teacher_edge_index = None if target else pyg_edge_index(adj, "cpu")
    else:
        teacher_edge_index = pyg_edge_index(adj, device)
    x_input = _residual_plus_x(name, adj, feat, device, hp)
    return SimpleNamespace(
        name=name,
        adj=adj,
        x=x,
        x_input=x_input,
        teacher=None,
        teacher_edge_index=teacher_edge_index,
        teacher_stream=teacher_stream,
        labels=np.asarray(labels, dtype=np.int64),
        mark=np.asarray(mark, dtype=bool),
        n=feat.shape[0],
    )


def _release_graph(graph):
    for field in ("x", "x_input", "teacher", "teacher_edge_index"):
        if hasattr(graph, field):
            delattr(graph, field)


def _release_teacher_inputs(graph):
    del graph.x, graph.teacher_edge_index


def _init_model(hp: dict, teacher_all: torch.Tensor | None, seed: int, device: str) -> StudentMoE:
    if teacher_all is None:
        teacher_all = torch.randn(max(int(hp["num_prototypes"]), 2), int(hp["dim"]), device=device)
    return StudentMoE(
        input_dim=int(hp["dim"]),
        t_feat=teacher_all,
        num_experts=int(hp["num_experts"]),
        expert_hidden_dim=int(hp["hidden_dim"]),
        expert_output_dim=int(hp["dim"]),
        prototypes_per_group_generalist=[int(hp["num_prototypes"])],
        prototypes_per_group_specialized=[int(hp["num_prototypes"])],
        dropout=float(hp["dropout"]),
        seed=seed,
        kmeans_seed=int(hp.get("kmeans_seed", 1234)),
        kmeans_max_points_per_centroid=int(hp.get("kmeans_max_points_per_centroid", 256)),
        kmeans_assignment_chunk=int(hp.get("kmeans_assignment_chunk", 1024)),
    ).to(device)


def _train_one(model: StudentMoE, graphs, epochs: int, hp: dict):
    main_params, proto_params = [], []
    for name, param in model.named_parameters():
        (proto_params if "prototype" in name else main_params).append(param)
    optimizer = Adam(
        [
            {"params": main_params, "lr": float(hp["lr"])},
            {"params": proto_params, "lr": float(hp["proto_lr"])},
        ],
        weight_decay=float(hp["weight_decay"]),
    )

    for epoch in range(int(epochs)):
        total = 0.0
        model.train()
        for graph in graphs:
            loss = _training_loss(model, graph.x_input, graph.teacher, hp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(hp.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp["grad_clip"]))
            optimizer.step()
            total += float(loss.detach())
            del loss
        if epoch == 0 or epoch + 1 == int(epochs) or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            print(
                f"    [promos] epoch={epoch + 1:03d}/{int(epochs)} "
                f"loss={total / max(1, len(graphs)):.5f}",
                flush=True,
            )


@torch.no_grad()
def _score_graph(model: StudentMoE, graph, hp: dict, device: str) -> np.ndarray:
    model.eval()
    scores = np.empty(graph.n, dtype=np.float32)
    chunk = int(hp["node_chunk"])
    for start in range(0, graph.n, chunk):
        end = min(start + chunk, graph.n)
        score = _score_chunk(model, graph.x_input[start:end], graph.teacher[start:end], hp)
        scores[start:end] = score.detach().cpu().numpy()
    return scores


def run_promos(sources, targets, seeds, epochs, device, hp=None, target_evaluator=None):
    """Train ProMoS on source graphs and zero-shot score target graphs."""
    hp = dict(C.PROMOS_HP if hp is None else hp)
    hp["edge_chunk"] = int(hp.get("edge_chunk", EDGE_CHUNK))
    hp["gca_big_names"] = tuple(BIG)
    source_graphs = []
    for source in sources:
        print(f"    [promos/build-source] {source}", flush=True)
        source_graphs.append(_graph(source, hp, device, target=False))

    teacher_seed = int(hp.get("gca_seed", 0))
    set_seed(teacher_seed)
    teacher_epochs = int(hp["gca_epochs"])
    if int(epochs) < int(C.EPOCHS["promos"]):
        teacher_epochs = int(hp.get("gca_quick_epochs", teacher_epochs))
    print(
        f"    [promos/gca] runtime source-only pretraining; "
        f"epochs={teacher_epochs}, save=False",
        flush=True,
    )
    teacher_model = train_runtime_gca(source_graphs, hp, device, epochs=teacher_epochs)
    for graph in source_graphs:
        graph.teacher = embed_graph(teacher_model, graph, hp, device)
        _release_teacher_inputs(graph)

    teacher_model.cpu()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    teacher_all = None
    if source_graphs:
        teacher_all = torch.cat([g.teacher for g in source_graphs], dim=0)

    trained = []
    for seed in seeds:
        set_seed(seed)
        model = _init_model(hp, teacher_all, seed, device)
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
        print(f"    [promos/target] {name}", flush=True)
        graph = _graph(name, hp, device, target=True)
        teacher_model.to(device)
        graph.teacher = embed_graph(teacher_model, graph, hp, device)
        teacher_model.cpu()
        _release_teacher_inputs(graph)
        for seed, model in trained:
            set_seed(seed)
            model.to(device)
            scores = _score_graph(model, graph, hp, device)
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
    del teacher_model
    return aggregate(per)
