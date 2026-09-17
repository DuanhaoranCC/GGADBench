"""AnomalyGFM source training and zero-shot or few-shot target evaluation."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.optim import Adam

from common.data import BIG, aggregate, load_source_marked, load_target_marked, x_svd
from ggad.vendor.anomalygfm.graph import build_adjs, residual_proto_score, subgraph_proto_score
from ggad.vendor.anomalygfm.model import Model
from ggad.vendor.anomalygfm.similarity import PAPER_SIM, global_avg_sim
from ggad.vendor.iaggad.preprocessing import preprocess_features
from util import evaluate, set_seed

INPUT_DIM = 8
TRAIN_RATE, ALIGN_WEIGHT, NEGSAMP, READOUT = 0.3, 0.1, 1, "avg"
LARGE_N = 1_000_000


def _graph(name, device, target=False):
    if target:
        adj, feat_raw, label, mark = load_target_marked(name, hops=2)
    else:
        adj, feat_raw, label, mark = load_source_marked(name)
    feat = x_svd(feat_raw, INPUT_DIM)
    feat = preprocess_features(sp.lil_matrix(feat))
    feat_t = (
        torch.from_numpy(np.ascontiguousarray(feat, np.float32)).unsqueeze(0).to(device)
    )  # (1,N,8)
    if len(label) > LARGE_N:
        adj_norm, adj_resid = None, None
    else:
        adj_norm, adj_resid = build_adjs(adj, device)
    e = adj.tocoo()
    sim = PAPER_SIM.get(name)
    if sim is None:
        sim = global_avg_sim(name, adj, feat_raw)
    return SimpleNamespace(
        name=name,
        feat=feat_t,
        adj=adj_norm,
        adj_resid=adj_resid,
        mark=mark,
        edge_index=(e.row, e.col),
        labels=torch.from_numpy(label).float().to(device),
        n=len(label),
        sim=sim,
    )


def run_anomalygfm(
    sources,
    targets,
    support_type,
    seeds,
    device,
    epochs,
    emb_dim,
    shot=10,
    train=True,
    target_evaluator=None,
):
    src_g = [_graph(n, device) for n in sources] if train else []
    per = {n: [] for n in targets}
    models = []
    for seed in seeds:
        set_seed(seed)
        model = Model(INPUT_DIM, INPUT_DIM, emb_dim, "prelu", NEGSAMP, READOUT).to(device)
        if train:
            opt = Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
            b_xent = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.tensor([NEGSAMP]).float().to(device)
            )
            idx_trains = []
            for g in src_g:
                idx = np.arange(g.n)
                np.random.shuffle(idx)
                idx_trains.append(torch.as_tensor(idx[: int(g.n * TRAIN_RATE)], device=device))

            model.train()
            for _ in range(epochs):
                for g, it in zip(src_g, idx_trains):
                    normal_raw = torch.randn(emb_dim, device=device)
                    abnormal_raw = torch.randn(emb_dim, device=device)
                    logits, _, _, emb_res, nprompt, aprompt, _ = model(
                        g.feat, g.adj, g.adj_resid, normal_raw, abnormal_raw
                    )
                    lt = g.labels[it]
                    loss_bce = b_xent(torch.squeeze(logits[:, it]), lt).mean()
                    er = emb_res[:, it, :]
                    dif_n = torch.sqrt(torch.sum((nprompt - er[:, lt == 0, :]) ** 2, dim=2))
                    dif_a = torch.sqrt(torch.sum((aprompt - er[:, lt == 1, :]) ** 2, dim=2))
                    loss = loss_bce + dif_a.mean() + ALIGN_WEIGHT * dif_n.mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        model.eval()
        model.zero_grad(set_to_none=True)
        model.cpu()
        models.append(model)
        if train:
            del opt, b_xent, loss
        torch.cuda.empty_cache()
    for g in src_g:
        del g.feat, g.adj, g.adj_resid, g.labels
    src_g.clear()
    torch.cuda.empty_cache()
    for name in targets:
        try:
            g = _graph(name, device, target=True)
        except RuntimeError as e:
            if name in BIG:
                raise
            print(f"    [anomalygfm] {name} graph construction skipped ({str(e)[:60]})")
            torch.cuda.empty_cache()
            continue
        beta = (
            (0.5 if g.sim > 0.5 else 4.0)
            if support_type == "normal"  # fs
            else (0.0 if g.sim > 0.5 else 4.0)
        )  # zs
        y = g.labels.cpu().numpy()
        for seed, model in zip(seeds, models):
            model.to(device)
            set_seed(seed)
            with torch.no_grad():
                try:
                    if g.n > LARGE_N:
                        ano = subgraph_proto_score(
                            model, g.feat, g.edge_index, emb_dim, device, beta, batch=25_000
                        )
                    else:
                        ano = residual_proto_score(
                            model, g.feat, g.adj, g.adj_resid, emb_dim, device, beta
                        )
                    ano = ano.squeeze().cpu().numpy()
                except RuntimeError as e:
                    if name in BIG:
                        raise
                    print(f"    [anomalygfm] {name} skipped ({str(e)[:60]})")
                    model.cpu()
                    torch.cuda.empty_cache()
                    continue
            if target_evaluator is not None:
                per[name].append(target_evaluator(name, int(seed), y, ano, g.mark))
            elif support_type == "normal":
                pool = np.where((y == 0) & g.mark)[0]
                np.random.shuffle(pool)
                m = np.zeros(g.n, dtype=bool)
                m[pool[:shot]] = True
                qidx = np.where(~m & g.mark)[0]
                per[name].append(evaluate(y[qidx], ano[qidx]))
            else:
                per[name].append(evaluate(y[g.mark], ano[g.mark]))
            model.cpu()
            torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()
    return aggregate(per)
