"""REFIGAD source training and target support/query evaluation."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import ggad.config as C
from common.data import BIG, aggregate
from ggad.vendor.refigad import PromptGADModel, build_graph, construct_features
from util import evaluate, set_seed


def _get_intra_batch(features, labels, k, batch_size, valid=None, return_indices=False):
    """Sample class-specific support and disjoint query nodes for an episode."""
    device = features.device
    lbl0 = (labels == 0) if valid is None else ((labels == 0) & valid)
    lbl1 = (labels == 1) if valid is None else ((labels == 1) & valid)
    norm_idx = lbl0.nonzero().squeeze(1)
    ano_idx = lbl1.nonzero().squeeze(1)
    sup_norm_idx = norm_idx[torch.randperm(len(norm_idx), device=device)[:k]]
    if len(ano_idx) < k:
        sup_ano_idx = ano_idx[torch.randint(0, len(ano_idx), (k,), device=device)]
    else:
        sup_ano_idx = ano_idx[torch.randperm(len(ano_idx), device=device)[:k]]
    support_norm, support_ano = features[sup_norm_idx], features[sup_ano_idx]

    mask = torch.ones(len(features), dtype=torch.bool, device=device)
    mask[sup_norm_idx] = False
    mask[sup_ano_idx] = False
    remain = torch.arange(len(features), device=device)[mask]
    remain_lbl = labels[remain]
    remain_norm = remain[remain_lbl == 0]
    remain_ano = remain[remain_lbl == 1]
    n_q_ano = min(50, len(remain_ano))
    n_q_norm = min(500, len(remain_norm))
    batch_ano = (
        remain_ano[torch.randperm(len(remain_ano), device=device)[:n_q_ano]]
        if n_q_ano > 0
        else torch.tensor([], dtype=torch.long, device=device)
    )
    batch_norm = (
        remain_norm[torch.randperm(len(remain_norm), device=device)[:n_q_norm]]
        if n_q_norm > 0
        else torch.tensor([], dtype=torch.long, device=device)
    )
    query_idx = torch.cat([batch_norm, batch_ano])
    if len(query_idx) == 0:
        result = (support_norm, support_ano, None, None)
    else:
        query_idx = query_idx[torch.randperm(len(query_idx), device=device)]
        result = (
            support_norm,
            support_ano,
            features[query_idx],
            labels[query_idx].float().unsqueeze(1),
        )
    return result + (sup_norm_idx, sup_ano_idx) if return_indices else result


def _k_for(name):
    return C.REFIGAD_K.get(name, C.REFIGAD_K_DEFAULT)


def _prep(name, num_hops, device, target=False):
    """Build REFIGAD graph features, labels, and evaluation mask."""
    g = build_graph(name, num_hops, device, target=target)
    P, labels = construct_features(g)
    return name, P, labels, g.mark


def run_refigad(sources, targets, seeds, device, epochs=None, shot=None, exclude_support=False):
    """Optional target-shot override leaves source training k unchanged.

    ``exclude_support`` evaluates every marked node outside target support.
    The default keeps the original benchmark's evaluation protocol intact.
    """
    hp = C.REFIGAD_HP
    epochs = epochs or hp["epoch"]
    train_k = C.REFIGAD_K_DEFAULT

    src_data = [_prep(n, hp["num_hops"], device, target=False) for n in sources]
    tgt_data = []
    for n in targets:
        try:
            tgt_data.append(_prep(n, hp["num_hops"], device, target=True))
        except RuntimeError as e:
            if n in BIG:
                raise
            print(f"    [refigad] {n} feature construction skipped ({str(e)[:60]})")
            torch.cuda.empty_cache()

    per = {n: [] for n in targets}
    for seed in seeds:
        set_seed(seed)
        model = PromptGADModel(
            input_feature_dim=5,
            d_model=hp["d_model"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            k_shot=train_k,
        ).to(device)
        opt = Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hp["pos_weight"]]).to(device))

        model.train()
        for _ in range(epochs):
            for _name, P, labels, _mark in src_data:
                for _ in range(hp["batches_per_dataset"]):
                    sn, sa, qf, ql = _get_intra_batch(P, labels, train_k, hp["batch_size"])
                    if qf is None or qf.size(0) == 0:
                        continue
                    out = model(sn, sa, qf)
                    loss = criterion(out, ql)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        model.eval()
        with torch.no_grad():
            for name, P, labels, mark in tgt_data:
                set_seed(seed)
                k = _k_for(name) if shot is None else int(shot)
                valid = None if mark.all() else torch.from_numpy(mark).to(device)
                query_mark = mark
                if exclude_support:
                    sn, sa, _, _, normal_indices, anomaly_indices = _get_intra_batch(
                        P, labels, k, 0, valid=valid, return_indices=True
                    )
                    query_mark = np.asarray(mark, dtype=bool).copy()
                    support_indices = torch.cat((normal_indices, anomaly_indices))
                    query_mark[support_indices.cpu().numpy()] = False
                    query_labels = labels.cpu().numpy()[query_mark]
                    if set(np.unique(query_labels).tolist()) != {0, 1}:
                        raise ValueError(
                            f"REFIGAD {name}: query after support exclusion must "
                            "contain both normal and anomaly nodes"
                        )
                else:
                    sn, sa, _, _ = _get_intra_batch(P, labels, k, 0, valid=valid)
                try:
                    probs = []
                    for i in range(0, len(P), hp["batch_size"]):

                        probs.append(model(sn, sa, P[i : i + hp["batch_size"]]).squeeze(-1))
                    probs = torch.cat(probs)[: len(labels)]
                except RuntimeError as e:
                    if name in BIG:
                        raise
                    print(f"    [refigad] {name} scoring skipped ({str(e)[:60]})")
                    torch.cuda.empty_cache()
                    continue
                y, p = labels.cpu().numpy(), probs.cpu().numpy()
                sc = evaluate(y[query_mark], p[query_mark])
                per[name].append(sc)
    return aggregate(per)
