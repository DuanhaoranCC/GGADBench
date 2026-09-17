"""Paper-based MDGPT reproduction under the gfm few-shot GAD protocol."""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import scipy.sparse as sp
import torch

from common.data import EdgeList, aggregate, load_source_marked, load_target_marked
from util import evaluate, set_seed, to_torch_sparse

from .model import (
    MDGPT,
    DualPrompt,
    link_loss,
    prototype_logits,
    prototypes,
    sample_triplets,
    split_support,
    support_loss,
)
from .preprocessing import align_features, normalize_graph, support_dependency
from .streaming import frozen_embeddings, frozen_encode

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache"


def _resolved_hp(hp):
    from gfm.config import MDGPT_HP

    cfg = dict(MDGPT_HP)
    if hp:
        unknown = set(hp) - set(cfg) - {"record_run"}
        if unknown:
            raise ValueError(f"Unknown MDGPT hyperparameters: {sorted(unknown)}")
        cfg.update(hp)
    for key in (
        "feature_dim",
        "hidden_dim",
        "num_layers",
        "num_negatives",
        "triplets_per_domain",
        "edge_chunk",
        "node_chunk",
        "stream_node_threshold",
        "stream_edge_threshold",
        "log_every",
    ):
        cfg[key] = int(cfg[key])
        if cfg[key] < 1:
            raise ValueError(f"MDGPT {key} must be positive")
    for key in ("pretrain_lr", "prompt_lr", "temperature"):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"MDGPT {key} must be finite and positive")
    if not np.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("MDGPT weight_decay must be finite and nonnegative")
    return cfg


def _prepare_graph(name, target, hp):
    loader = load_target_marked if target else load_source_marked
    raw, features, labels, mark = loader(name)
    x = align_features(features, hp["feature_dim"], hp["feature_seed"], hp["cache"])
    del features
    raw = sp.csr_matrix(raw, dtype=np.float32)
    norm = normalize_graph(raw)
    # Store the loss-sampling topology separately from A+I, once per source.
    if not target:
        raw = raw.copy()
        raw.setdiag(0)
        raw.eliminate_zeros()
        raw.sort_indices()
    graph = {
        "name": name,
        "x": x,
        "norm": norm,
        "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
        "mark": np.asarray(mark, dtype=bool).reshape(-1),
        "raw": None if target else raw,
        "edges": int(raw.nnz),
    }
    if len(graph["labels"]) != len(x) or len(graph["mark"]) != len(x):
        raise ValueError(f"MDGPT graph {name}: mismatched feature/label/mark length")
    print(
        f"    [mdgpt/graph] {name} N={len(x)} E={raw.nnz} "
        f"marked={int(graph['mark'].sum())} dim={x.shape[1]}",
        flush=True,
    )
    return graph


def _stream(adjacency, hp):
    return (
        adjacency.shape[0] >= hp["stream_node_threshold"]
        or adjacency.nnz >= hp["stream_edge_threshold"]
    )


def _operator(adjacency, device, hp):
    if _stream(adjacency, hp):
        return EdgeList(adjacency)
    return to_torch_sparse(adjacency).to(device)


def _backward_checked(loss, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("MDGPT non-finite training loss")
    loss.backward()
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("MDGPT non-finite parameter gradient")


def _new_model(num_domains, hp, device):
    return MDGPT(num_domains, hp["feature_dim"], hp["hidden_dim"], hp["num_layers"]).to(device)


def _train_seed(graphs, num_domains, seed, epochs, device, hp):
    set_seed(seed)
    model = _new_model(num_domains, hp, device)
    history = []
    if epochs == 0:
        return model.cpu(), history
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["pretrain_lr"], weight_decay=hp["weight_decay"]
    )
    rng = np.random.RandomState(seed)
    # Source graphs in this benchmark have small N; dense-edge graphs stay on
    # the CPU and use exact streamed forward AND backward (no E-by-d tape).
    operators = [_operator(g["norm"], device, hp) for g in graphs]
    features = [torch.from_numpy(g["x"]).to(device) for g in graphs]
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for domain, (graph, x, operator) in enumerate(zip(graphs, features, operators)):
            arrays = sample_triplets(
                graph["raw"], hp["triplets_per_domain"], hp["num_negatives"], rng, prepared=True
            )
            anchor, positive, negative = [torch.from_numpy(a).to(device) for a in arrays]
            embeddings = model.encode_source(x, operator, domain, hp["edge_chunk"])
            loss = link_loss(embeddings, anchor, positive, negative, hp["temperature"]) / len(
                graphs
            )
            _backward_checked(loss, model.parameters())
            total += float(loss.detach())
            del embeddings, loss
        optimizer.step()
        history.append(total)
        if epoch == 0 or (epoch + 1) % hp["log_every"] == 0 or epoch == epochs - 1:
            print(
                f"    [mdgpt/pretrain] seed={seed} epoch={epoch+1}/{epochs} "
                f"negative-only-loss={total:.6f}",
                flush=True,
            )
    del optimizer, operators, features
    return model.cpu().eval(), history


def _support_embedding(model, prompt, graph, support, device, hp, operator=None, x=None):
    if operator is not None:
        return prompt(model.encoder, x, operator, hp["edge_chunk"])[
            torch.as_tensor(support, dtype=torch.long, device=device)
        ]
    options = {
        "output_nodes": support,
        "node_chunk": hp["node_chunk"],
        "work_dir": CACHE_ROOT / "mdgpt_work",
    }
    encoder = model.encoder
    return frozen_encode(
        prompt.unifying, graph["x"], graph["norm"], encoder.layers, encoder.activations, **options
    ) + frozen_encode(
        prompt.mixing(), graph["x"], graph["norm"], encoder.layers, encoder.activations, **options
    )


def _tune_prompt(model, graph, support, seed, epochs, device, hp):
    # Each target starts identically for a given seed, independently of the
    # order of targets and of how much source training consumed the RNG.
    set_seed(seed)
    model.to(device).freeze()
    prompt = DualPrompt(model.domain_tokens).to(device)
    labels = torch.from_numpy(graph["labels"][support]).to(device)
    training_graph, local_support = graph, support
    if _stream(graph["norm"], hp) and epochs:
        norm, x, local_support = support_dependency(
            graph["norm"], graph["x"], support, hp["num_layers"]
        )
        training_graph = {"norm": norm, "x": x}
        print(
            f"    [mdgpt/exact-support] N={len(x)}/{len(graph['x'])} "
            f"E={norm.nnz}/{graph['norm'].nnz}; full degrees retained",
            flush=True,
        )
    operator = x_device = None
    if not _stream(training_graph["norm"], hp):
        operator = _operator(training_graph["norm"], device, hp)
        x_device = torch.from_numpy(training_graph["x"]).to(device)
    history = []
    if epochs:
        optimizer = torch.optim.Adam(
            prompt.parameters(), lr=hp["prompt_lr"], weight_decay=hp["weight_decay"]
        )
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            embedded = _support_embedding(
                model, prompt, training_graph, local_support, device, hp, operator, x_device
            )
            loss = support_loss(embedded, labels, hp["temperature"])
            _backward_checked(loss, prompt.parameters())
            optimizer.step()
            history.append(float(loss.detach()))
            if epoch == 0 or (epoch + 1) % hp["log_every"] == 0 or epoch == epochs - 1:
                print(
                    f"    [mdgpt/prompt] {graph['name']} seed={seed} "
                    f"epoch={epoch+1}/{epochs} support_loss={history[-1]:.6f}",
                    flush=True,
                )
            # Release disk-backed sign caches before the next optimization step.
            del loss, embedded
        del optimizer
    del operator, x_device, training_graph
    return prompt.eval(), history


@torch.no_grad()
def _scores(model, prompt, graph, support, query, device, hp):
    labels = torch.from_numpy(graph["labels"][support]).to(device)
    scores = np.empty(len(query), dtype=np.float32)
    encoder = model.encoder
    if _stream(graph["norm"], hp):
        options = {"node_chunk": hp["node_chunk"], "work_dir": CACHE_ROOT / "mdgpt_work"}
        with frozen_embeddings(
            prompt.unifying,
            graph["x"],
            graph["norm"],
            encoder.layers,
            encoder.activations,
            **options,
        ) as unifying:
            with frozen_embeddings(
                prompt.mixing(),
                graph["x"],
                graph["norm"],
                encoder.layers,
                encoder.activations,
                **options,
            ) as mixing:
                support_z = torch.from_numpy(np.asarray(unifying[support] + mixing[support])).to(
                    device
                )
                centers = prototypes(support_z, labels)
                for start in range(0, len(query), hp["node_chunk"]):
                    ids = query[start : start + hp["node_chunk"]]
                    z = torch.from_numpy(np.asarray(unifying[ids] + mixing[ids])).to(device)
                    logits = prototype_logits(z, centers, hp["temperature"])
                    scores[start : start + len(ids)] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    else:
        operator = _operator(graph["norm"], device, hp)
        x = torch.from_numpy(graph["x"]).to(device)
        z = prompt(encoder, x, operator, hp["edge_chunk"])
        centers = prototypes(z[torch.as_tensor(support, device=device)], labels)
        for start in range(0, len(query), hp["node_chunk"]):
            ids = torch.as_tensor(query[start : start + hp["node_chunk"]], device=device)
            logits = prototype_logits(z[ids], centers, hp["temperature"])
            scores[start : start + len(ids)] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"MDGPT non-finite scores on {graph['name']}")
    return scores


def run_mdgpt(
    sources, targets, seeds, epochs, device, shot=10, train=True, hp=None, prompt_epochs=100
):
    """Source SSL -> frozen encoder -> target support-only prompt tuning.

    train=False disables BOTH optimizers and never loads source data. Final
    evaluation is always all marked query nodes, with support removed. Errors
    propagate; there is no skipped target or approximate-memory fallback.
    """
    hp = _resolved_hp(hp)
    sources, targets, seeds = list(sources), list(targets), [int(s) for s in seeds]
    if not sources or not targets or not seeds:
        raise ValueError("MDGPT sources, targets and seeds must be nonempty")
    if (
        len(set(sources)) != len(sources)
        or len(set(targets)) != len(targets)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("MDGPT source/target names and seeds must not contain duplicates")
    epochs, prompt_epochs = (int(epochs), int(prompt_epochs)) if train else (0, 0)
    if min(epochs, prompt_epochs) < 0:
        raise ValueError("MDGPT epoch budgets must be nonnegative")
    device = str(device)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    metadata = {
        "implementation": hp["implementation_version"],
        "paper": "MDGPT arXiv:2405.13934v4",
        "sources": sources,
        "targets": targets,
        "seeds": seeds,
        "hp": hp,
        "shot_per_class": int(shot),
        "train": bool(train),
        "pretrain_epochs": epochs,
        "prompt_epochs": prompt_epochs,
        "device": device,
        "source_history": {},
        "target_runs": [],
    }
    print(
        f"    [mdgpt/version] {hp['implementation_version']} SVD={hp['feature_dim']} "
        f"layers={hp['num_layers']} hidden={hp['hidden_dim']} epochs={epochs}/{prompt_epochs}",
        flush=True,
    )
    graphs = [_prepare_graph(name, False, hp) for name in sources] if train and epochs else []
    models = []
    for seed in seeds:
        model, history = _train_seed(graphs, len(sources), seed, epochs, device, hp)
        models.append((seed, model))
        metadata["source_history"][str(seed)] = history
    del graphs
    gc.collect()
    per_target = {name: [] for name in targets}
    for name in targets:
        graph = _prepare_graph(name, True, hp)
        for seed, model in models:
            support, query = split_support(graph["labels"], graph["mark"], int(shot), seed)
            prompt, history = _tune_prompt(model, graph, support, seed, prompt_epochs, device, hp)
            scores = _scores(model, prompt, graph, support, query, device, hp)
            metrics = evaluate(graph["labels"][query], scores)
            per_target[name].append(metrics)
            metadata["target_runs"].append(
                {
                    "target": name,
                    "seed": seed,
                    "nodes": len(graph["x"]),
                    "edges": graph["edges"],
                    "marked": int(graph["mark"].sum()),
                    "support": support.tolist(),
                    "unique_support": int(np.unique(support).size),
                    "queries": len(query),
                    "metrics": metrics,
                    "prompt_loss": history,
                    "gamma": prompt.gamma.detach().cpu().tolist(),
                    "trainable_prompt_parameters": sum(p.numel() for p in prompt.parameters()),
                }
            )
            print(
                f"    [mdgpt/target] {name} seed={seed} support={len(support)} "
                f"queries={len(query)} AUROC={metrics['AUROC']:.4f} "
                f"AUPRC={metrics['AUPRC']:.4f}",
                flush=True,
            )
            model.cpu()
            del prompt, scores
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        del graph
        gc.collect()
    result = aggregate(per_target)
    if hp.get("record_run", True):
        destination = (
            Path(os.environ.get("GAD_OUTPUT_DIR", CACHE_ROOT.parent / "results")) / "mdgpt_runs"
        )
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = destination / f"{stamp}_{uuid4().hex[:8]}.json"
        metadata["summary"] = result
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    [mdgpt/results] {path}", flush=True)
    return result
