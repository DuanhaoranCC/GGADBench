"""GAD-MoRE runner using the released model and benchmark-wide data protocol."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint

import ggad.config as C
from common.data import aggregate, load_source_marked, load_target_marked
from ggad.vendor.gadmore.model import GADMoRE
from ggad.vendor.gadmore.preprocess import mcfa, normalize_adjacency, propagate_features
from util import evaluate, set_seed


def _graph(name, device, hp, target=False):
    loader = load_target_marked if target else load_source_marked
    adj, raw_features, labels, mark = loader(name)
    adj = sp.csr_matrix(adj, dtype=np.float32)
    aligned = mcfa(name, adj, raw_features, hp)
    if aligned.shape[1] != int(hp["dim"]):
        raise RuntimeError(
            "{} MCFA width mismatch: got {}, expected {}".format(name, aligned.shape[1], hp["dim"])
        )
    adj_norm = normalize_adjacency(name, adj)
    levels, operator = propagate_features(
        adj_norm,
        aligned,
        hp["num_hops"],
        device,
        hp,
        keep_operator=not target,
    )
    return SimpleNamespace(
        name=name,
        adj_norm=None if target else adj_norm.tocsr(),
        operator=operator,
        x_list=levels,
        labels=np.asarray(labels).astype(np.int64),
        mark=np.asarray(mark, dtype=bool),
        n=adj.shape[0],
    )


def _exact_structure_loss(reconstruction, adj, row_chunk):
    """Exact official full-adjacency BCE with bounded temporary matrices."""
    total = reconstruction.new_zeros(())
    n = reconstruction.shape[0]
    for start in range(0, n, row_chunk):
        end = min(start + row_chunk, n)
        logits = reconstruction[start:end] @ reconstruction.T
        target = torch.from_numpy(adj[start:end].toarray().astype(np.float32, copy=False)).to(
            reconstruction.device
        )
        total = total + F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    return total / (n * n)


def _sampled_structure_loss(reconstruction, adj, sample_count, sample_chunk):
    """Uniform-pair unbiased estimator of the official N x N BCE."""
    n = reconstruction.shape[0]
    count = min(int(sample_count), n * n)
    row = torch.randint(n, (count,), device=reconstruction.device)
    col = torch.randint(n, (count,), device=reconstruction.device)
    row_np = row.cpu().numpy().copy()
    col_np = col.cpu().numpy().copy()
    target_np = np.asarray(adj[row_np, col_np]).reshape(-1).astype(np.float32, copy=False)
    target = torch.from_numpy(target_np)
    total = reconstruction.new_zeros(())
    for start in range(0, count, sample_chunk):
        end = min(start + sample_chunk, count)
        current_row = row[start:end]
        current_col = col[start:end]

        def pair_logits(current, pair_row, pair_col):
            return (current[pair_row] * current[pair_col]).sum(dim=1)

        logits = checkpoint(
            pair_logits,
            reconstruction,
            current_row,
            current_col,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        current_target = target[start:end].to(reconstruction.device)
        total = total + F.binary_cross_entropy_with_logits(logits, current_target, reduction="sum")
    return total / count


def _structure_loss(reconstruction, graph, hp):
    if graph.n <= int(hp["structure_exact_max_nodes"]):
        return _exact_structure_loss(reconstruction, graph.adj_norm, int(hp["structure_row_chunk"]))
    return _sampled_structure_loss(
        reconstruction,
        graph.adj_norm,
        int(hp["structure_pair_samples"]),
        int(hp["structure_sample_chunk"]),
    )


def _positive_matrix(adj, sample_index=None):
    matrix = sp.csr_matrix(adj)
    matrix.data = np.ones_like(matrix.data, dtype=np.float32)
    matrix = matrix.maximum(matrix.T)
    matrix = matrix.maximum(sp.eye(matrix.shape[0], dtype=np.float32, format="csr"))
    matrix.eliminate_zeros()
    if sample_index is not None:
        matrix = matrix[sample_index][:, sample_index].tocsr()
    return matrix


def _contrastive_loss(embedding, graph, hp):
    """Official structure contrastive objective without a dense boolean mask."""
    n = embedding.shape[0]
    sample_index = None
    if n > int(hp["contrastive_sample_threshold"]):
        sample_size = max(int(n * hp["contrastive_sample_ratio"]), 1000)
        sample_size = min(sample_size, int(hp["contrastive_sample_cap"]), n)
        selected = torch.randperm(n, device=embedding.device)[:sample_size]
        embedding = embedding[selected]
        sample_index = selected.cpu().numpy()

    positive = _positive_matrix(graph.adj_norm, sample_index)
    z = F.normalize(embedding, p=2, dim=1)
    temperature = float(hp["contrastive_temperature"])
    row_np, col_np = positive.nonzero()
    row = torch.from_numpy(row_np.astype(np.int64, copy=False)).to(z.device)
    col = torch.from_numpy(col_np.astype(np.int64, copy=False)).to(z.device)
    numerators = z.new_zeros(len(z))
    positive_chunk = int(hp.get("contrastive_positive_chunk", 4096))
    for start in range(0, row.numel(), positive_chunk):
        end = min(start + positive_chunk, row.numel())
        current_row = row[start:end]
        current_col = col[start:end]

        def positive_edge_terms(current, edge_row, edge_col):
            return torch.exp((current[edge_row] * current[edge_col]).sum(dim=1) / temperature)

        terms = checkpoint(
            positive_edge_terms,
            z,
            current_row,
            current_col,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        numerators = numerators.index_add(0, current_row, terms)

    logical_losses = []
    physical = int(hp["contrastive_row_chunk"])
    logical = int(hp.get("contrastive_logical_chunk", 1_000))
    for logical_start in range(0, len(z), logical):
        logical_end = min(logical_start + logical, len(z))
        logical_sum = z.new_zeros(())
        for start in range(logical_start, logical_end, physical):
            end = min(start + physical, logical_end)
            denominator = torch.exp((z[start:end] @ z.T) / temperature).sum(dim=1)
            ratio = numerators[start:end].clamp_min(1e-8)
            ratio = ratio / denominator.clamp_min(1e-8)
            logical_sum = logical_sum - torch.log(ratio).sum()
        logical_losses.append(logical_sum / (logical_end - logical_start))
    return torch.stack(logical_losses).mean()


def _max_message(features, operator):
    normalized = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9)
    row_sum = torch.sparse.sum(operator, dim=1).to_dense().flatten()
    aggregated = torch.sparse.mm(operator, normalized)
    message = torch.sum(normalized * aggregated, dim=1)
    inverse = torch.zeros_like(row_sum)
    nonzero = row_sum != 0
    inverse[nonzero] = row_sum[nonzero].reciprocal()
    message = message * inverse
    return -torch.sum(message)


def _loss(model, graph, hp):
    embedding = model(graph)
    scorer = model.anomaly_scorer
    reconstruction, _, decoded, info = scorer.moe_reconstruction(embedding)

    embed_loss = F.mse_loss(reconstruction, embedding)
    feature_loss = F.mse_loss(decoded, graph.x_list[0])
    structure_loss = _structure_loss(reconstruction, graph, hp)
    contrastive_loss = _contrastive_loss(embedding, graph, hp)
    message_loss = _max_message(embedding, graph.operator)
    entropy = info["entropy"].mean()

    total = (
        hp["w_embed"] * embed_loss
        + hp["w_feature"] * feature_loss
        + hp["w_structure"] * structure_loss
        + hp["contrastive_weight"] * contrastive_loss
        + hp["w_message"] * message_loss
        + hp["w_gate"] * entropy
    )
    scorer.periodic_memory_update(embedding)
    scorer.record_routing_stats(info)
    return total, {
        "embed": embed_loss,
        "feature": feature_loss,
        "structure": structure_loss,
        "contrastive": contrastive_loss,
        "message": message_loss,
        "entropy": entropy,
    }


def _train_one(model, source_graphs, epochs, hp):
    optimizer = Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    for epoch in range(epochs):
        model.anomaly_scorer.router.set_epoch(epoch)
        report = epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 5 == 0
        for graph_id, graph in enumerate(source_graphs):
            model.train()
            loss, parts = _loss(model, graph, hp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if report:
                part_text = " ".join(
                    "{}={:.4g}".format(key, float(value.detach())) for key, value in parts.items()
                )
                stats = model.anomaly_scorer.router.get_memory_stats()
                memory = [round(value, 2) for value in stats["memory_utilization"]]
                print(
                    "    [gadmore] epoch={:03d}/{} graph={} loss={:.5f} "
                    "{} memory={}".format(
                        epoch + 1,
                        epochs,
                        graph.name,
                        float(loss.detach()),
                        part_text,
                        memory,
                    )
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return model


def _release_graph(graph):
    del graph.x_list
    graph.operator = None


def run_gadmore(
    sources, targets, seeds, epochs, device, train=True, hp=None, target_evaluator=None
):
    """Train on benchmark source graphs and score all marked target nodes."""
    hp = dict(C.GADMORE_HP if hp is None else hp)
    source_graphs = [_graph(name, device, hp, target=False) for name in sources] if train else []
    models = []
    for seed in seeds:
        set_seed(seed)
        model = GADMoRE(hp).to(device)
        if train:
            _train_one(model, source_graphs, epochs, hp)
        model.eval().cpu()
        model.move_memory("cpu")
        models.append((seed, model))
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for graph in source_graphs:
        _release_graph(graph)
    source_graphs.clear()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    per_target = {name: [] for name in targets}
    for name in targets:
        print("    [gadmore/target] {}".format(name))
        graph = _graph(name, device, hp, target=True)
        for seed, model in models:
            set_seed(seed)
            model.to(device).eval()
            model.move_memory(device)
            scores = model.anomaly_score(graph.x_list, int(hp["inference_node_chunk"]))
            per_target[name].append(
                target_evaluator(name, int(seed), graph.labels, scores, graph.mark)
                if target_evaluator is not None
                else evaluate(graph.labels[graph.mark], scores[graph.mark])
            )
            model.cpu()
            model.move_memory("cpu")
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        _release_graph(graph)
    return aggregate(per_target)
