"""DOMINANT under the source-to-target generalist-GAD protocol.

The neural architecture, adjacency preparation, Adam objective and anomaly
score follow the official ``DOMINATE`` repository.  SVD8 normalization is the
same cross-dataset feature adapter used by every baseline in this block.

Official DOMINANT reconstructs a dense N-by-N adjacency matrix with the raw
inner product ``Z @ Z.T`` (the released code does not apply the paper's
sigmoid). That exact
path remains available for small graphs. Larger benchmark graphs directly
sample non-negative squared residuals before the row norm; this avoids the old
negative-estimate clamp while avoiding an N-by-N allocation. Million-node GCN
propagation uses an exact differentiable
CPU-edge-list SpMM whose forward and backward stream bounded edge chunks.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from baselines.config import DOMINANT_HP, DOMINANT_QUICK
from baselines.dominant import Dominant
from baselines.preprocess import unify_features
from baselines.protocol import load_dense_source, load_dense_target
from baselines.runtime import (
    empty_device_cache,
    print_progress,
    print_stage,
    seed_for_device,
    select_device,
)
from common.data import EdgeList, aggregate
from util import evaluate


@dataclass
class GraphArrays:
    name: str
    adj_label: sp.csr_matrix
    adj_norm: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    mark: np.ndarray


@dataclass
class TorchGraph:
    name: str
    adj_label: sp.csr_matrix
    adj_norm: object
    features: torch.Tensor


def normalize_adj_official(adj):
    """Exact ``DOMINATE/utils.py::normalize_adj`` sparse equivalent.

    The released expression is ``(A D^-1/2).T D^-1/2``.  It equals the usual
    symmetric normalization for the undirected paper graphs and deliberately
    retains the official transpose behavior for directed inputs.
    """
    matrix = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.asarray(matrix.sum(1)).reshape(-1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    values = (matrix.data * d_inv_sqrt[matrix.row] * d_inv_sqrt[matrix.col]).astype(
        np.float32, copy=False
    )
    return sp.csr_matrix(
        (values, (matrix.col, matrix.row)),
        shape=matrix.shape,
        dtype=np.float32,
    )


def prepare_adjacencies(adj, add_self_loop=True):
    """Return official reconstruction target ``A+I`` and normalized input."""
    matrix = sp.csr_matrix(adj, dtype=np.float32)
    if add_self_loop:
        # The official loader adds I without first deleting existing loops.
        matrix = matrix + sp.eye(matrix.shape[0], dtype=np.float32, format="csr")
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, normalize_adj_official(matrix)


def _to_torch_sparse(matrix, device):
    coo = sp.coo_matrix(matrix, dtype=np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        device=device,
    ).coalesce()


def _materialize(graph, device, cfg):
    dev = select_device(device)
    stream = graph.adj_norm.shape[0] >= int(
        cfg["stream_node_threshold"]
    ) or graph.adj_norm.nnz >= int(cfg["stream_edge_threshold"])
    if stream:
        adjacency = EdgeList(graph.adj_norm)
        adjacency.dominant_edge_chunk = int(cfg["edge_chunk"])
    else:
        adjacency = _to_torch_sparse(graph.adj_norm, dev)
    return TorchGraph(
        name=graph.name,
        adj_label=graph.adj_label,
        adj_norm=adjacency,
        features=torch.as_tensor(graph.features, dtype=torch.float32, device=dev),
    )


def _load_arrays(name, norm, cfg, target):
    if target:
        adj, feat_raw, labels, mark = load_dense_target(name, hops=cfg["target_hops"])
    else:
        adj, feat_raw, labels, mark = load_dense_source(
            name, max_nodes=cfg["source_max_nodes"], hops=cfg["source_hops"]
        )
    features = np.asarray(unify_features(feat_raw, norm, name), dtype=np.float32)
    adj_label, adj_norm = prepare_adjacencies(adj, add_self_loop=bool(cfg["add_self_loop"]))
    return GraphArrays(
        name=name,
        adj_label=adj_label,
        adj_norm=adj_norm,
        features=features,
        labels=np.asarray(labels),
        mark=np.asarray(mark, dtype=bool),
    )


def _sample_columns(num_nodes, sample_size, seed, device):
    size = min(max(int(sample_size), 1), int(num_nodes))
    if size == int(num_nodes):
        return torch.arange(num_nodes, dtype=torch.long, device=device), size
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    # Sampling with replacement avoids a multi-million-element permutation.
    columns = rng.randint(0, num_nodes, size=size).astype(np.int64, copy=False)
    return torch.from_numpy(columns).to(device=device), size


def _uses_exact_structure(graph, cfg):
    return graph.name not in set(cfg.get("approx_structure_datasets", ()))


def _exact_structure_error_block(structure_z, adj_label, start, end):
    target = torch.as_tensor(
        adj_label[start:end].toarray(),
        dtype=structure_z.dtype,
        device=structure_z.device,
    )
    prediction = structure_z[start:end] @ structure_z.T
    return torch.sqrt(torch.sum((prediction - target) ** 2, dim=1))


def _sampled_structure_error_block(
    structure_z,
    adj_label,
    start,
    end,
    sampled_columns,
    sample_count,
):
    num_nodes = structure_z.shape[0]
    sampled_prediction = structure_z[start:end] @ structure_z[sampled_columns].T
    sampled_target = torch.as_tensor(
        adj_label[start:end][:, sampled_columns.detach().cpu().numpy()].toarray(),
        dtype=structure_z.dtype,
        device=structure_z.device,
    )
    row_squared = (float(num_nodes) / float(sample_count)) * torch.sum(
        (sampled_prediction - sampled_target).square(), dim=1
    )
    return torch.sqrt(row_squared)


def structure_error_blocks(
    structure_z,
    adj_label,
    row_batch_size,
    exact,
    sample_size,
    sample_seed,
):
    """Yield official per-node structure errors in bounded row blocks."""
    num_nodes = int(structure_z.shape[0])
    sampled_columns = sample_count = None
    if not exact:
        sampled_columns, sample_count = _sample_columns(
            num_nodes, sample_size, sample_seed, structure_z.device
        )
    for start in range(0, num_nodes, int(row_batch_size)):
        end = min(start + int(row_batch_size), num_nodes)
        if exact:
            errors = _exact_structure_error_block(structure_z, adj_label, start, end)
        else:
            errors = _sampled_structure_error_block(
                structure_z,
                adj_label,
                start,
                end,
                sampled_columns,
                sample_count,
            )
        yield start, end, errors


def train_epoch(model, optimizer, graph, cfg, sample_seed):
    """One official DOMINANT optimization epoch with bounded structure rows."""
    model.train()
    optimizer.zero_grad()
    structure_z, x_hat = model.decoded_embeddings(graph.features, graph.adj_norm)
    attribute_errors = torch.sqrt(torch.sum((x_hat - graph.features) ** 2, dim=1))
    alpha = float(cfg["alpha"])
    exact = _uses_exact_structure(graph, cfg)
    needs_structure = alpha < 1.0
    streaming = isinstance(graph.adj_norm, EdgeList)
    row_batch_size = cfg["large_structure_batch_size"] if streaming else cfg["structure_batch_size"]
    structure_sample_size = (
        cfg["large_structure_sample_size"] if streaming else cfg["structure_sample_size"]
    )

    if alpha > 0.0:
        (alpha * attribute_errors.mean()).backward(retain_graph=needs_structure)

    structure_sum = 0.0
    if needs_structure:
        num_nodes = int(graph.features.shape[0])
        blocks = structure_error_blocks(
            structure_z,
            graph.adj_label,
            row_batch_size=row_batch_size,
            exact=exact,
            sample_size=structure_sample_size,
            sample_seed=sample_seed,
        )
        for _start, end, errors in blocks:
            structure_sum += float(errors.detach().sum().cpu())
            ((1.0 - alpha) * errors.sum() / num_nodes).backward(retain_graph=end < num_nodes)

    optimizer.step()
    attribute_mean = float(attribute_errors.detach().mean().cpu())
    structure_mean = structure_sum / max(int(graph.features.shape[0]), 1)
    return {
        "loss": alpha * attribute_mean + (1.0 - alpha) * structure_mean,
        "attribute": attribute_mean,
        "structure": structure_mean,
        "exact_structure": exact,
    }


@torch.no_grad()
def score_graph(model, graph, cfg, sample_seed):
    """Return the official weighted per-node reconstruction score."""
    model.eval()
    structure_z, x_hat = model.decoded_embeddings(graph.features, graph.adj_norm)
    attribute_errors = torch.sqrt(torch.sum((x_hat - graph.features) ** 2, dim=1)).cpu()
    exact = _uses_exact_structure(graph, cfg)
    streaming = isinstance(graph.adj_norm, EdgeList)
    row_batch_size = cfg["large_structure_batch_size"] if streaming else cfg["structure_batch_size"]
    structure_sample_size = (
        cfg["large_score_structure_sample_size"]
        if streaming
        else cfg["score_structure_sample_size"]
    )
    structure_errors = torch.empty(graph.features.shape[0], dtype=torch.float32)
    for start, end, errors in structure_error_blocks(
        structure_z,
        graph.adj_label,
        row_batch_size=row_batch_size,
        exact=exact,
        sample_size=structure_sample_size,
        sample_seed=sample_seed,
    ):
        structure_errors[start:end] = errors.cpu()
    alpha = float(cfg["alpha"])
    return (alpha * attribute_errors + (1.0 - alpha) * structure_errors).numpy()


def _train_cfg(quick=False, hp=None):
    cfg = dict(DOMINANT_HP)
    if hp is not None:
        cfg.update(hp)
    if quick:
        cfg.update(DOMINANT_QUICK)
    return cfg


def _stable_name_seed(name):
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def _is_oom(error):
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def run_dominant(
    sources,
    targets,
    seeds,
    device,
    norm,
    quick=False,
    hp=None,
    progress=None,
):
    per = {name: [] for name in targets}
    cfg = _train_cfg(quick=quick, hp=hp)
    select_device(device)

    source_graphs = []
    for position, name in enumerate(sources, 1):
        started = time.perf_counter()
        print_stage(
            "dominant",
            1,
            4,
            f"preparing source {position}/{len(sources)}: {name}",
        )
        try:
            arrays = _load_arrays(name, norm, cfg, target=False)
            source_graphs.append(arrays)
            path = "exact" if _uses_exact_structure(arrays, cfg) else "sampled"
            propagation = (
                "edge-chunk"
                if (
                    arrays.adj_norm.shape[0] >= int(cfg["stream_node_threshold"])
                    or arrays.adj_norm.nnz >= int(cfg["stream_edge_threshold"])
                )
                else "torch-sparse"
            )
            print_stage(
                "dominant",
                1,
                4,
                f"source ready {name}: N={arrays.features.shape[0]} "
                f"E={arrays.adj_label.nnz} propagation={propagation} "
                f"structure={path} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
            )
        except Exception as error:
            print(
                f"    [dominant] source {name}: preparation skipped " f"({str(error)[:140]})",
                flush=True,
            )

    trained_models = []
    epochs = int(cfg["num_epoch"])
    for seed_position, seed in enumerate(seeds, 1):
        seed_for_device(seed, device)
        model = Dominant(
            feat_size=source_graphs[0].features.shape[1] if source_graphs else 8,
            hidden_size=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        trained_any = False
        for source_position, arrays in enumerate(source_graphs, 1):
            graph = None
            try:
                graph = _materialize(arrays, device, cfg)
                started = time.perf_counter()
                for epoch in range(epochs):
                    metrics = train_epoch(
                        model,
                        optimizer,
                        graph,
                        cfg,
                        sample_seed=(
                            int(seed) * 1_000_003 + _stable_name_seed(arrays.name) * 997 + epoch
                        ),
                    )
                    if (
                        epoch == 0
                        or (epoch + 1) == epochs
                        or ((epoch + 1) % max(epochs // 20, 1) == 0)
                    ):
                        print_progress(
                            "dominant",
                            f"stage 2/4 source={arrays.name} "
                            f"({source_position}/{len(source_graphs)}) "
                            f"seed={seed} ({seed_position}/{len(seeds)}) "
                            f"loss={metrics['loss']:.4f}",
                            epoch + 1,
                            epochs,
                            started,
                        )
                trained_any = True
            except Exception as error:
                print(
                    f"    [dominant] source {arrays.name} seed={seed} skipped "
                    f"({str(error)[:140]})",
                    flush=True,
                )
                if _is_oom(error):
                    empty_device_cache(device)
            finally:
                graph = None
                empty_device_cache(device)

        if trained_any:
            optimizer = None
            for parameter in model.parameters():
                parameter.grad = None
            trained_models.append((int(seed), model))
        else:
            model = optimizer = None
            empty_device_cache(device)

    source_graphs.clear()
    empty_device_cache(device)

    for target_position, name in enumerate(targets, 1):
        print_stage(
            "dominant",
            3,
            4,
            f"preparing target {target_position}/{len(targets)}: {name}",
        )
        try:
            arrays = _load_arrays(name, norm, cfg, target=True)
        except Exception as error:
            print(
                f"    [dominant] target {name}: preparation skipped " f"({str(error)[:140]})",
                flush=True,
            )
            if progress is not None:
                progress(name, aggregate({name: per[name]})[name])
            continue

        graph = None
        graph_device = device
        try:
            graph = _materialize(arrays, device, cfg)
            print_stage(
                "dominant",
                3,
                4,
                f"target ready {name}: N={arrays.features.shape[0]} "
                f"E={arrays.adj_label.nnz} propagation="
                f"{'edge-chunk' if isinstance(graph.adj_norm, EdgeList) else 'torch-sparse'} "
                f"structure={'exact' if _uses_exact_structure(arrays, cfg) else 'sampled'}",
            )
        except Exception as error:
            if not _is_oom(error):
                print(
                    f"    [dominant] target {name}: materialization skipped "
                    f"({str(error)[:140]})",
                    flush=True,
                )
                continue
            print(f"    [dominant] target {name}: CUDA OOM; retrying on CPU", flush=True)
            graph = None
            empty_device_cache(device)
            graph_device = "cpu"
            graph = _materialize(arrays, "cpu", cfg)

        for model_position, (seed, model) in enumerate(trained_models, 1):
            print_stage(
                "dominant",
                4,
                4,
                f"scoring target={name} ({target_position}/{len(targets)}) "
                f"seed={seed} ({model_position}/{len(trained_models)})",
            )
            eval_model = model
            try:
                if torch.device(graph_device).type == "cpu":
                    eval_model = copy.deepcopy(model).cpu()
                scores = score_graph(
                    eval_model,
                    graph,
                    cfg,
                    sample_seed=(int(seed) * 1_000_003 + _stable_name_seed(name) * 997),
                )
            except Exception as error:
                if not _is_oom(error) or torch.device(graph_device).type == "cpu":
                    print(
                        f"    [dominant] target {name} seed={seed} skipped "
                        f"({str(error)[:140]})",
                        flush=True,
                    )
                    empty_device_cache(device)
                    continue
                print(
                    f"    [dominant] target {name} seed={seed}: CUDA OOM; " "retrying score on CPU",
                    flush=True,
                )
                graph = None
                empty_device_cache(device)
                graph_device = "cpu"
                try:
                    graph = _materialize(arrays, "cpu", cfg)
                    eval_model = copy.deepcopy(model).cpu()
                    scores = score_graph(
                        eval_model,
                        graph,
                        cfg,
                        sample_seed=(int(seed) * 1_000_003 + _stable_name_seed(name) * 997),
                    )
                except Exception as cpu_error:
                    print(
                        f"    [dominant] target {name} seed={seed}: CPU retry "
                        f"failed ({str(cpu_error)[:140]})",
                        flush=True,
                    )
                    continue
            if not np.isfinite(scores).all():
                print(
                    f"    [dominant] target {name} seed={seed}: " "NaN/Inf scores; skipped",
                    flush=True,
                )
                continue
            per[name].append(evaluate(arrays.labels[arrays.mark], scores[arrays.mark]))
            if progress is not None:
                progress(name, aggregate({name: per[name]})[name])
            eval_model = None
            empty_device_cache(device)

        if progress is not None and not per[name]:
            progress(name, aggregate({name: per[name]})[name])
        graph = arrays = None
        empty_device_cache(device)

    return aggregate(per)
