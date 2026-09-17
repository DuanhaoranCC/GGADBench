"""Strict PyG TA-GGAD runner for the source-count transfer benchmark.

The released method is reproduced end to end: official feature alignment and
high-order propagation, cosine-EMA VQ, raw-feature MLP and DGL-equivalent PyG
GCN affinity branches, source KDEs, JS fusion, pseudo-label voting, and the
exhaustive score-weight grid.  The released target query anomaly count is used
unchanged, so this runner is intentionally registered as
``few_shot_oracle_count`` rather than ordinary few-shot evaluation.

Large graphs retain every node and edge.  Edge chunking and node chunking only
bound temporary memory; they do not sample the propagation or evaluation graph.
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.optim import AdamW

import ggad.config as C
from common.data import (
    BIG,
    CACHE_ROOT,
    REAL_DIR,
    _atomic_save_npy,
    _content_fingerprint,
    _feat_alignment_large,
    _find,
    aggregate,
    load_source_marked,
    load_target_marked,
)
from ggad.vendor.taggad import (
    ARC,
    combined_neighbor_scores,
    compute_kde_distribution,
    normalize_score,
    testing_time_adaptive_fusion,
)
from ggad.vendor.taggad.graph import (
    ROW_NORMALIZED_DATASETS,
    build_high_order_adjacency,
    prepare_affinity_edge_index,
    prepare_affinity_score_edge_index,
    prepare_aligned_features,
    propagate_high_order,
)
from util import evaluate, set_seed

ALIGN_CACHE = CACHE_ROOT / "taggad_alignment"
STREAM_GRAPHS = BIG | {"t_finance"}
SPECIAL_CODEBOOK_DATASETS = {"weibo", "BlogCatalog"}


@dataclass
class _TrainedTrial:
    seed: int
    model: ARC
    final_codebook: Tensor
    train_query_kdes: List[tuple]
    train_mlp_kdes: List[tuple]
    train_gcn_kdes: List[tuple]
    target_support_indices: Dict[str, Tensor]
    training_trace: Optional[List[Dict]] = None


def _empty_cuda_cache(device):
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _validate_edge_index(name, edge_name, edge_index, num_nodes):
    """Reject malformed CPU edge lists before any CUDA indexing kernel runs."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"{name} {edge_name} edge_index has invalid shape " f"{tuple(edge_index.shape)}"
        )
    if edge_index.dtype != torch.long:
        raise TypeError(f"{name} {edge_name} edge_index must be torch.long")
    if edge_index.numel():
        minimum = int(edge_index.min())
        maximum = int(edge_index.max())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError(
                f"{name} {edge_name} edge index range [{minimum}, {maximum}] "
                f"is outside [0, {num_nodes})"
            )


def _aligned_cache_path(name, adjacency, raw_features, dims, large=False):
    # Keep TA-GGAD's released float32 result separate from the shared runner's
    # cache so execution order can never select a different implementation.
    safe_name = name.replace("/", "_").replace("\\", "_")
    version = "large_ipca_float32_v3" if large else "official_float32_v2"
    row_norm = int(name in ROW_NORMALIZED_DATASETS)
    feature_digest = _content_fingerprint(
        raw_features, f"taggad-{version}-d{dims}-rownorm{row_norm}"
    )
    graph_digest = _content_fingerprint(adjacency, "taggad-align-graph-v1")
    return ALIGN_CACHE / (f"{version}_{safe_name}_{feature_digest}_{graph_digest}.npy")


def _row_normalize_large(features):
    if sp.issparse(features):
        matrix = sp.csr_matrix(features, dtype=np.float32)
        rowsum = np.asarray(matrix.sum(1), dtype=np.float32).reshape(-1)
        inverse = np.zeros_like(rowsum, dtype=np.float32)
        np.divide(1.0, rowsum, out=inverse, where=rowsum != 0)
        return sp.diags(inverse).dot(matrix)

    values = np.asarray(features, dtype=np.float32)
    rowsum = values.sum(axis=1, keepdims=True)
    inverse = np.zeros_like(rowsum, dtype=np.float32)
    np.divide(1.0, rowsum, out=inverse, where=rowsum != 0)
    values *= inverse
    return values


def _official_aligned_features(name, adjacency, raw_features, dims):
    """Load/cache TA-GGAD feature alignment.

    Million-node targets use the benchmark's exact large-graph alignment path:
    it still keeps all nodes and edges, but avoids materializing the release's
    full GRP/PCA work arrays at once.
    """
    large = name in BIG or adjacency.shape[0] > 1_000_000
    path = _aligned_cache_path(name, adjacency, raw_features, dims, large=large)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.shape == (adjacency.shape[0], dims) and cached.dtype == np.float32:
            return torch.from_numpy(np.ascontiguousarray(cached))

    if large:
        coo = adjacency.tocoo(copy=False)
        features = (
            _row_normalize_large(raw_features) if name in ROW_NORMALIZED_DATASETS else raw_features
        )
        values = _feat_alignment_large(
            features,
            np.asarray(coo.row, dtype=np.int64),
            np.asarray(coo.col, dtype=np.int64),
            dims,
        )
    else:
        aligned = prepare_aligned_features(adjacency, raw_features, name, dims=dims).cpu()
        values = aligned.numpy()
    values = np.ascontiguousarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(path, values)
    return torch.from_numpy(values)


def _target_label_metadata(name):
    """Load only target labels/marks so official mask RNG order is preserved.

    The release samples every target support set before source support sets and
    before model construction.  Reading only label variables avoids loading all
    target graphs concurrently merely to reproduce that random-state ordering.
    """
    if name == "dgraphfin":
        with np.load(REAL_DIR / "dgraphfin.npz") as data:
            labels = (data["y"] == 1).astype(np.int64)
            mark = np.zeros(labels.shape[0], dtype=bool)
            for key in ("train_mask", "valid_mask", "test_mask"):
                mark[data[key]] = True
        return labels, mark
    if name == "elliptic":
        import pandas as pd

        classes = pd.read_csv(
            REAL_DIR / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"
        ).to_numpy()
        labels = np.zeros(classes.shape[0], dtype=np.int64)
        labels[classes[:, 1] == "1"] = 1
        return labels, classes[:, 1] != "unknown"
    if name == "tsocial":
        path = REAL_DIR / "tsocial.npz"
        if not path.exists():
            raise FileNotFoundError(
                "TA-GGAD requires the DGL-free Dataset/real/tsocial.npz conversion"
            )
        with np.load(path) as data:
            labels = np.asarray(data["y"], dtype=np.int64)
        return labels, np.ones(labels.shape[0], dtype=bool)

    values = sio.loadmat(str(_find(name)), variable_names=["Label", "gnd"])
    if "Label" in values:
        labels = np.asarray(values["Label"]).reshape(-1)
    elif "gnd" in values:
        labels = np.asarray(values["gnd"]).reshape(-1)
    else:
        raise KeyError(f"no Label/gnd variable found for {name}")
    return labels, np.ones(labels.shape[0], dtype=bool)


def _sample_normal_indices(labels, mark, shot):
    normal = np.where((np.asarray(labels) == 0) & np.asarray(mark))[0].tolist()
    random.shuffle(normal)
    return torch.as_tensor(normal[:shot], dtype=torch.long)


def _build_graph(name, hp, device, target=False):
    loader = load_target_marked if target else load_source_marked
    adjacency, raw_features, labels, mark = (
        loader(name, hops=hp["num_hops"]) if target else loader(name)
    )
    labels = np.asarray(labels).reshape(-1)
    mark = np.asarray(mark, dtype=bool).reshape(-1)
    if labels.shape[0] != adjacency.shape[0] or mark.shape[0] != adjacency.shape[0]:
        raise ValueError(f"TA-GGAD graph metadata length mismatch for {name}")

    aligned = _official_aligned_features(name, adjacency, raw_features, hp["in_feats"])
    high_order = build_high_order_adjacency(adjacency, name)
    stream = name in STREAM_GRAPHS or adjacency.nnz >= hp["stream_edge_threshold"]
    edge_chunk = hp["edge_chunk"] if stream else None
    graph_device = torch.device(device)
    if target and adjacency.shape[0] >= hp["target_cpu_hops_threshold"]:
        # Exact full-graph fallback for million-node targets.  Keeping all hop
        # tensors, labels and streamed edges on CPU avoids silently sampling or
        # exhausting GPU memory; evaluation moves the frozen model to this same
        # device for mathematically identical inference.
        graph_device = torch.device("cpu")
    x_list = propagate_high_order(
        high_order,
        aligned,
        hp["num_hops"],
        device=graph_device,
        chunk_size=edge_chunk,
    )

    low_edge_index = prepare_affinity_edge_index(adjacency)
    local_edge_index = prepare_affinity_score_edge_index(adjacency)
    for edge_name, edge_index in (
        ("message", low_edge_index),
        ("affinity", local_edge_index),
    ):
        _validate_edge_index(name, edge_name, edge_index, adjacency.shape[0])
    if not stream:
        low_edge_index = low_edge_index.to(graph_device)
        local_edge_index = local_edge_index.to(graph_device)

    return SimpleNamespace(
        name=name,
        x=x_list[0],
        x_list=x_list,
        edge_index=low_edge_index,
        low_edge_index=low_edge_index,
        local_edge_index=local_edge_index,
        edge_chunk_size=edge_chunk,
        labels_np=labels,
        mark=mark,
        ano_labels=torch.as_tensor(labels, dtype=torch.float32, device=graph_device),
        n=adjacency.shape[0],
        stream=stream,
    )


def _sample_normal_mask(graph, shot):
    """Official ``random.shuffle`` normal support sampling."""
    indices = _sample_normal_indices(graph.labels_np, graph.mark, shot).to(graph.x.device)
    mask = torch.zeros(graph.n, dtype=torch.bool, device=graph.x.device)
    mask[indices] = True
    return mask


def _new_model(hp, device):
    args = SimpleNamespace(code_size=hp["code_size"], topk=hp["topk"])
    # The release passes ``drop_rate`` through **kwargs.  ARC's actual
    # ``dropout_rate`` therefore remains its default zero; preserve that bug.
    return ARC(
        args,
        in_feats=hp["in_feats"],
        h_feats=hp["h_feats"],
        num_layers=hp["num_layers"],
        drop_rate=hp["drop_rate"],
        activation=hp["activation"],
        num_hops=hp["num_hops"],
    ).to(device)


def _train_trial(
    seed,
    source_graphs,
    target_metadata,
    epochs,
    hp,
    device,
    capture_training_trace=False,
):
    if epochs < 1:
        raise ValueError("strict TA-GGAD training requires at least one epoch")

    set_seed(seed)
    # Exact release order: all target masks, then all source masks, then model
    # construction.  get_train_loss subsequently consumes this same Python RNG.
    target_support_indices = {
        name: _sample_normal_indices(labels, mark, hp["shot"])
        for name, (labels, mark) in target_metadata.items()
    }
    for graph in source_graphs:
        graph.shot_mask = _sample_normal_mask(graph, hp.get("source_shot", hp["shot"]))

    model = _new_model(hp, device)
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    codebooks = [None] * len(source_graphs)
    train_query_kdes = [None] * len(source_graphs)
    train_mlp_kdes = [None] * len(source_graphs)
    train_gcn_kdes = [None] * len(source_graphs)
    training_trace = [None] * len(source_graphs) if capture_training_trace else None

    for epoch in range(epochs):
        for index, graph in enumerate(source_graphs):
            if epoch == 0:
                mode = "chunked-full-backward" if graph.stream else "full-backward"
                print(
                    f"    [taggad/train] seed={seed:02d} source={graph.name} "
                    f"N={graph.n} stream={graph.stream} mode={mode}",
                    flush=True,
                )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            residual, loss_code, node_gcn, node_mlp, codebook = model(graph, graph)
            prompt_loss = model.get_train_loss(
                residual,
                node_gcn,
                codebook,
                graph.ano_labels,
                hp["num_prompt"],
            )
            score_l2_mlp, score_l1_mlp, score_cos_mlp, loss_cos_mlp = combined_neighbor_scores(
                graph.local_edge_index,
                node_mlp,
                num_nodes=graph.n,
                chunk_size=hp["edge_chunk"],
            )
            score_l2_gcn, score_l1_gcn, score_cos_gcn, loss_cos_gcn = combined_neighbor_scores(
                graph.local_edge_index,
                node_gcn,
                num_nodes=graph.n,
                chunk_size=hp["edge_chunk"],
            )
            query_scores = model.get_test_score(
                residual,
                codebook,
                graph.shot_mask,
                graph.ano_labels,
            )
            mlp_scores = score_l2_mlp + score_l1_mlp + score_cos_mlp
            gcn_scores = score_l2_gcn + score_l1_gcn + score_cos_gcn
            loss = prompt_loss + loss_code.squeeze() + loss_cos_mlp + loss_cos_gcn

            loss.backward()
            optimizer.step()

            if epoch == epochs - 1:
                # The release saves tensors produced immediately before this
                # source's final optimizer update, then reads them during test.
                codebooks[index] = codebook.detach().cpu().clone()
                train_query_kdes[index] = compute_kde_distribution(
                    query_scores.detach(), bins=hp["kde_bins"]
                )
                train_mlp_kdes[index] = compute_kde_distribution(
                    mlp_scores.detach(), bins=hp["kde_bins"]
                )
                train_gcn_kdes[index] = compute_kde_distribution(
                    gcn_scores.detach(), bins=hp["kde_bins"]
                )
                if capture_training_trace:
                    training_trace[index] = {
                        "name": graph.name,
                        "support_indices": torch.nonzero(graph.shot_mask).squeeze(1).detach().cpu(),
                        "query_scores": query_scores.detach().cpu(),
                        "mlp_scores": mlp_scores.detach().cpu(),
                        "gcn_scores": gcn_scores.detach().cpu(),
                        "codebook": codebook.detach().cpu().clone(),
                        "low_edge_index": graph.low_edge_index.detach().cpu(),
                        "local_edge_index": graph.local_edge_index.detach().cpu(),
                        "x_list": [value.detach().cpu() for value in graph.x_list],
                    }

            # Do not keep one source's outputs alive while the next source runs
            # its large VQ allocation.  Backward and all final-epoch snapshots
            # are complete at this point.
            del residual, loss_code, node_gcn, node_mlp, codebook
            del prompt_loss, loss_cos_mlp, loss_cos_gcn, loss
            del score_l2_mlp, score_l1_mlp, score_cos_mlp
            del score_l2_gcn, score_l1_gcn, score_cos_gcn
            del query_scores, mlp_scores, gcn_scores

    final_codebook = torch.cat(codebooks, dim=0)
    model.eval()
    model.zero_grad(set_to_none=True)
    model.cpu()
    del optimizer
    _empty_cuda_cache(device)
    return _TrainedTrial(
        seed=seed,
        model=model,
        final_codebook=final_codebook,
        train_query_kdes=train_query_kdes,
        train_mlp_kdes=train_mlp_kdes,
        train_gcn_kdes=train_gcn_kdes,
        target_support_indices=target_support_indices,
        training_trace=training_trace,
    )


def _residual_subset(model, graph, indices):
    """Exact node-subset residual path used only for large target inference.

    Target VQ is in eval mode: assignments and reconstruction loss do not alter
    the residual or any state.  Skipping those unused tensors avoids the
    otherwise prohibitive ``N x codebook_size`` target distance matrix.
    """
    x_list = [value[indices] for value in graph.x_list]
    for layer_index, layer in enumerate(model.layers):
        if layer_index != 0:
            x_list = [model.dropout(value) for value in x_list]
        x_list = [layer(value) for value in x_list]
        if layer_index != len(model.layers) - 1:
            x_list = [model.act(value) for value in x_list]

    first = x_list[0]
    if graph.name in SPECIAL_CODEBOOK_DATASETS:
        codebook = model.vq._codebook.embed.squeeze(0)
        for index, value in enumerate(x_list[1:]):
            nearest = (value @ codebook.T).topk(k=model.top_k, dim=1).indices
            aggregate_codes = codebook[nearest].mean(dim=1)
            x_list[index + 1] = 0.5 * value + 0.5 * aggregate_codes
    return torch.hstack([value - first for value in x_list[1:]])


def _query_score_large(model, graph, mask, final_codebook, node_chunk):
    support_indices = torch.nonzero(mask & (graph.ano_labels == 0)).squeeze(1)
    support = _residual_subset(model, graph, support_indices)
    nearest_codes = final_codebook[(support @ final_codebook.T).argmax(dim=1)]
    mean_support = support.mean(dim=0, keepdim=True)
    mean_codes = nearest_codes.mean(dim=0, keepdim=True)

    mark = torch.as_tensor(graph.mark, device=mask.device)
    query_indices = torch.nonzero((~mask) & mark).squeeze(1)
    scores = []
    for start in range(0, query_indices.numel(), node_chunk):
        indices = query_indices[start : start + node_chunk]
        query = _residual_subset(model, graph, indices)
        score_support = torch.sqrt(torch.sum((query - mean_support) ** 2, dim=1))
        score_codes = torch.sqrt(torch.sum((query - mean_codes) ** 2, dim=1))
        scores.append((score_support + score_codes) / 2)
    return query_indices, torch.cat(scores)


def _node_mlp_large(model, graph, node_chunk):
    output = graph.x.new_empty((graph.n, graph.x.shape[1]))
    for start in range(0, graph.n, node_chunk):
        end = min(start + node_chunk, graph.n)
        output[start:end] = model.node_mlps(graph.x[start:end])
    return output


def _inverted_affinity(edge_index, node_embedding, graph, hp):
    score_l2, score_l1, score_cos, _ = combined_neighbor_scores(
        edge_index,
        node_embedding,
        num_nodes=graph.n,
        chunk_size=hp["edge_chunk"],
    )
    raw = (score_l2 + score_l1 + score_cos).detach().cpu().numpy()
    inverted = 1 - normalize_score(raw)
    return torch.as_tensor(inverted, dtype=torch.float32, device=node_embedding.device)


def _target_score_components(model, graph, mask, final_codebook, hp):
    mark = torch.as_tensor(graph.mark, device=mask.device)
    if graph.name not in BIG:
        residual, _, node_gcn, node_mlp, _ = model(graph, graph)
        all_query_indices = torch.nonzero(~mask).squeeze(1)
        all_query_scores = model.get_test_score(residual, final_codebook, mask, graph.ano_labels)
        keep = mark[all_query_indices]
        query_indices = all_query_indices[keep]
        query_scores = all_query_scores[keep]
        affinity_mlp = _inverted_affinity(graph.local_edge_index, node_mlp, graph, hp)
        affinity_gcn = _inverted_affinity(graph.local_edge_index, node_gcn, graph, hp)
        return query_indices, query_scores, affinity_mlp, affinity_gcn

    query_indices, query_scores = _query_score_large(
        model, graph, mask, final_codebook, hp["node_chunk"]
    )
    node_mlp = _node_mlp_large(model, graph, hp["node_chunk"])
    affinity_mlp = _inverted_affinity(graph.local_edge_index, node_mlp, graph, hp)
    del node_mlp
    _empty_cuda_cache(graph.x.device)

    node_gcn = model.GCN_model(
        graph.low_edge_index,
        graph.x,
        num_nodes=graph.n,
        chunk_size=graph.edge_chunk_size,
    )
    affinity_gcn = _inverted_affinity(graph.local_edge_index, node_gcn, graph, hp)
    del node_gcn
    _empty_cuda_cache(graph.x.device)
    return query_indices, query_scores, affinity_mlp, affinity_gcn


def _evaluate_trial(trial, graph, lam, hp, device, return_trace=False):
    compute_device = graph.x.device
    model = trial.model.to(compute_device)
    model.eval()
    final_codebook = trial.final_codebook.to(compute_device)
    support_indices = trial.target_support_indices[graph.name].to(compute_device)
    mask = torch.zeros(graph.n, dtype=torch.bool, device=compute_device)
    mask[support_indices] = True

    with torch.no_grad():
        query_indices, query_scores, affinity_mlp, affinity_gcn = _target_score_components(
            model, graph, mask, final_codebook, hp
        )
        query_scores_mlp = (1 - lam) * query_scores + lam * affinity_mlp[query_indices]
        query_scores_gcn = (1 - lam) * query_scores + lam * affinity_gcn[query_indices]
        query_labels = graph.ano_labels[query_indices]
        fusion = testing_time_adaptive_fusion(
            query_scores,
            query_scores_mlp,
            query_scores_gcn,
            query_labels,
            trial.train_query_kdes,
            trial.train_mlp_kdes,
            trial.train_gcn_kdes,
            count_node=hp["count_node"],
            normal_ratio=hp["normal_ratio"],
            bins=hp["kde_bins"],
            alpha=hp["js_alpha"],
        )
        metrics = evaluate(
            graph.labels_np[query_indices.cpu().numpy()],
            fusion["fused_score"].cpu().numpy(),
        )

    trace = None
    if return_trace:
        trace = {
            "seed": trial.seed,
            "lambda": float(lam),
            "support_indices": torch.nonzero(mask).squeeze(1).detach().cpu(),
            "query_indices": query_indices.detach().cpu(),
            "query_labels": query_labels.detach().cpu(),
            "query_scores": query_scores.detach().cpu(),
            "query_scores_mlp": query_scores_mlp.detach().cpu(),
            "query_scores_gcn": query_scores_gcn.detach().cpu(),
            "score_features": fusion["score_features"].detach().cpu(),
            "fused_score": fusion["fused_score"].detach().cpu(),
            "js_values": dict(fusion["js_values"]),
            "js_weights": list(fusion["js_weights"]),
            "best_auc": fusion["best_auc"],
            "best_weights": list(fusion["best_weights"]),
            "best_dims": list(fusion["best_dims"]),
            "num_anomaly_oracle": fusion["num_anomaly_oracle"],
            "pseudo_labels": fusion["pseudo_labels"].detach().cpu(),
            "pseudo_indices": fusion["pseudo_indices"].detach().cpu(),
        }

    print(
        f"    [taggad] {graph.name} seed={trial.seed} "
        f"oracle_M={fusion['num_anomaly_oracle']} "
        f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f}"
    )
    model.cpu()
    del final_codebook, affinity_mlp, affinity_gcn, fusion
    _empty_cuda_cache(device)
    return (metrics, trace) if return_trace else metrics


def run_taggad(
    sources,
    targets,
    seeds,
    epochs,
    device,
    shot=10,
    train=True,
    target_lambdas=None,
    hp_override: Optional[Dict] = None,
    return_traces=False,
    return_training_states=False,
    source_shot=None,
):
    """Train TA-GGAD on selected sources and evaluate oracle-count targets."""
    if return_training_states and not return_traces:
        raise ValueError("return_training_states requires return_traces=True")
    if not train:
        raise ValueError(
            "TA-GGAD's released test adapter requires trained source codebooks "
            "and source KDE distributions; no zero-update protocol is defined"
        )

    hp = dict(C.TAGGAD_HP)
    if hp_override:
        hp.update(hp_override)
    hp["shot"] = int(shot)
    if source_shot is not None:
        hp["source_shot"] = int(source_shot)
    lambdas = dict(hp.get("target_lambdas", {}) if target_lambdas is None else target_lambdas)
    default_lambda = hp.get("target_lambda")
    if default_lambda is not None:
        for name in targets:
            lambdas.setdefault(name, float(default_lambda))
    missing = [name for name in targets if name not in lambdas]
    if missing:
        raise ValueError(
            "TA-GGAD official params/dataset_config.json is absent; provide "
            "explicit target_lambdas for: " + ", ".join(missing)
        )

    target_metadata = {name: _target_label_metadata(name) for name in targets}
    source_graphs = [_build_graph(name, hp, device, target=False) for name in sources]
    trials = [
        _train_trial(
            seed,
            source_graphs,
            target_metadata,
            epochs,
            hp,
            device,
            capture_training_trace=return_training_states,
        )
        for seed in seeds
    ]
    training_states = None
    if return_training_states:
        training_states = [
            {
                "seed": trial.seed,
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in trial.model.state_dict().items()
                },
                "final_codebook": trial.final_codebook.detach().cpu().clone(),
                "query_kdes": [
                    (np.array(x, copy=True), np.array(y, copy=True))
                    for x, y in trial.train_query_kdes
                ],
                "mlp_kdes": [
                    (np.array(x, copy=True), np.array(y, copy=True))
                    for x, y in trial.train_mlp_kdes
                ],
                "gcn_kdes": [
                    (np.array(x, copy=True), np.array(y, copy=True))
                    for x, y in trial.train_gcn_kdes
                ],
                "sources": trial.training_trace,
            }
            for trial in trials
        ]

    for graph in source_graphs:
        del (
            graph.x_list,
            graph.x,
            graph.local_edge_index,
            graph.low_edge_index,
            graph.edge_index,
        )
    source_graphs.clear()
    target_metadata.clear()
    _empty_cuda_cache(device)

    per_target = {name: [] for name in targets}
    traces = {name: [] for name in targets} if return_traces else None
    for name in targets:
        graph = _build_graph(name, hp, device, target=True)
        lam = float(lambdas[name])
        for trial in trials:
            evaluated = _evaluate_trial(
                trial,
                graph,
                lam,
                hp,
                device,
                return_trace=return_traces,
            )
            if return_traces:
                metrics, trace = evaluated
                per_target[name].append(metrics)
                traces[name].append(trace)
            else:
                per_target[name].append(evaluated)
        del graph
        _empty_cuda_cache(device)

    result = aggregate(per_target)
    if return_training_states:
        return result, traces, training_states
    return (result, traces) if return_traces else result
