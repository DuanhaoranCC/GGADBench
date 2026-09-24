"""UNPrompt source training and target evaluation."""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from common.data import (
    BIG,
    EdgeList,
    aggregate,
    load_source_marked,
    load_target_marked,
    x_svd_torch,
)
from ggad.vendor.unprompt.grace import traingrace
from ggad.vendor.unprompt.graph import build_adjs, normalize_adj, official_loop_views
from ggad.vendor.unprompt.model import GPFplusAtt, Model, Projection
from ggad.vendor.unprompt.ops import completionloss, completionsim, normalize_score
from util import evaluate, set_seed

UNIFEAT, EMB_DIM, NUMPROMPTS = 8, 128, 10
LR, WD = 1e-3, 0.0
GRACE = {
    "emb_dim": EMB_DIM,
    "tau": 0.5,
    "grace_lr": 1e-3,
    "grace_weight_decay": 1e-5,
    "edge_drop_prob": 0.2,
    "feat_drop_prob": 0.3,
    "grace_epochs": 200,
}


def _graph(name, device, target=False):
    loader = load_target_marked if target else load_source_marked
    adj, feat_raw, label, mark = loader(name)
    feat = torch.from_numpy(x_svd_torch(feat_raw, UNIFEAT))
    feat = nn.BatchNorm1d(feat.shape[1], affine=False)(feat).detach().to(device)  # BatchNorm=zscore
    if target and name in BIG:
        awl, awl_won = None, None
        _, adj_woself = official_loop_views(adj)
        aws = EdgeList(normalize_adj(adj_woself))
    else:
        awl, awl_won, aws = build_adjs(adj)
    awl_dev = awl if target else awl.to(device)
    aws_dev = aws if isinstance(aws, EdgeList) else aws.to(device)
    return SimpleNamespace(
        name=name,
        feat=feat,
        awl=awl_dev,
        awl_won=awl_won,
        aws=aws_dev,
        labels=label,
        mark=mark,
        labels_t=torch.from_numpy(label).float().to(device),
        n=len(label),
    )


@torch.no_grad()
def _completion_score_chunked(model, prompts, proj, g, node_chunk=100_000, edge_chunk=500_000):
    """Exact UNPrompt target scoring with edge/node chunks.

    Equivalent to completionsim(proj(model(mf, None)), proj(model(mf, g.aws))).
    All nodes are scored; only temporary computation is chunked.
    """
    model.eval()
    prompts.eval()
    proj.eval()
    n = g.n
    device = g.feat.device
    hidden = model.gcn1.fc.out_features

    fc_all = torch.empty((n, hidden), dtype=torch.float32, device=device)
    for start in range(0, n, node_chunk):
        end = min(start + node_chunk, n)
        fc_all[start:end] = model.gcn1.fc(prompts.add(g.feat[start:end]))

    adj_out = torch.zeros_like(fc_all)
    if isinstance(g.aws, EdgeList):
        edge_iter = g.aws.chunks(device, edge_chunk)
    else:
        adj = g.aws.coalesce()
        row, col = adj.indices()
        all_val = adj.values()
        edge_iter = (
            (
                row[start : start + edge_chunk],
                col[start : start + edge_chunk],
                all_val[start : start + edge_chunk],
            )
            for start in range(0, row.numel(), edge_chunk)
        )
    for r, c, edge_val in edge_iter:
        adj_out.index_add_(0, r, fc_all[c] * edge_val.unsqueeze(1))

    score = np.empty(n, dtype=np.float32)
    bias = model.gcn1.bias
    for start in range(0, n, node_chunk):
        end = min(start + node_chunk, n)
        z_none = fc_all[start:end]
        z_adj = adj_out[start:end]
        if bias is not None:
            z_none = z_none + bias
            z_adj = z_adj + bias
        h_none = model.gcn1.act(model.gcn1.bn(z_none))
        h_adj = model.gcn1.act(model.gcn1.bn(z_adj))
        p_none = proj(h_none)
        p_adj = proj(h_adj)
        p_none = p_none / (torch.norm(p_none, dim=-1, keepdim=True) + 1e-12)
        p_adj = p_adj / (torch.norm(p_adj, dim=-1, keepdim=True) + 1e-12)
        score[start:end] = torch.sum(p_none * p_adj, dim=1).detach().cpu().numpy()
    return 1 - normalize_score(score)


def run_unprompt(sources, targets, seeds, device, epochs, grace_epochs=None, target_evaluator=None):
    ge = grace_epochs or GRACE["grace_epochs"]
    src = [_graph(n, device) for n in sources]
    per = {n: [] for n in targets}
    trained = []
    for seed in seeds:
        set_seed(seed)
        model = Model(UNIFEAT, EMB_DIM, "prelu").to(device)
        traingrace(
            model,
            [g.feat for g in src],
            [g.awl for g in src],
            [g.awl_won for g in src],
            GRACE,
            device,
            ge,
        )
        model.eval()
        prompts = GPFplusAtt(UNIFEAT, NUMPROMPTS).to(device)
        proj = Projection(EMB_DIM).to(device)
        opt = optim.Adam(
            list(prompts.parameters()) + list(proj.parameters()), lr=LR, weight_decay=WD
        )
        for _ in range(epochs):
            prompts.train()
            proj.train()
            for g in src:
                opt.zero_grad()
                mf = prompts.add(g.feat)
                loss = completionloss(proj(model(mf, g.aws)), proj(model(mf, None)), g.labels_t)
                loss.backward()
                opt.step()
        prompts.eval()
        proj.eval()
        model.zero_grad(set_to_none=True)
        prompts.zero_grad(set_to_none=True)
        proj.zero_grad(set_to_none=True)
        model.cpu()
        prompts.cpu()
        proj.cpu()
        trained.append((seed, model, prompts, proj))
        del opt, loss
        torch.cuda.empty_cache()
    for g in src:
        del g.feat, g.awl, g.aws, g.labels_t
    src.clear()
    torch.cuda.empty_cache()
    for name in targets:
        try:
            g = _graph(name, device, target=True)
        except RuntimeError as e:
            if name in BIG:
                raise
            print(f"    [unprompt] {name} graph construction skipped ({str(e)[:60]})")
            torch.cuda.empty_cache()
            continue
        for seed, model, prompts, proj in trained:
            model.to(device)
            prompts.to(device)
            proj.to(device)
            with torch.no_grad():
                try:
                    if g.n > 1_000_000 or isinstance(g.aws, EdgeList) or g.aws._nnz() > 5_000_000:
                        score = _completion_score_chunked(model, prompts, proj, g)
                    else:
                        mf = prompts.add(g.feat)
                        sim = completionsim(proj(model(mf, None)), proj(model(mf, g.aws)))
                        score = 1 - normalize_score(sim)
                except RuntimeError as e:
                    if name in BIG:
                        raise
                    if "out of memory" not in str(e).lower():
                        print(f"    [unprompt] {name} skipped ({str(e)[:60]})")
                        model.cpu()
                        prompts.cpu()
                        proj.cpu()
                        torch.cuda.empty_cache()
                        continue
                    print(f"    [unprompt] {name}: CUDA OOM, retry exact chunked scoring")
                    torch.cuda.empty_cache()
                    try:
                        score = _completion_score_chunked(model, prompts, proj, g)
                    except RuntimeError as ee:
                        print(f"    [unprompt] {name} chunked skipped ({str(ee)[:60]})")
                        model.cpu()
                        prompts.cpu()
                        proj.cpu()
                        torch.cuda.empty_cache()
                        continue
            metrics = (
                target_evaluator(name, int(seed), g.labels, score, g.mark)
                if target_evaluator is not None
                else evaluate(g.labels[g.mark], score[g.mark])
            )
            per[name].append(metrics)
            model.cpu()
            prompts.cpu()
            proj.cpu()
            torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()
    return aggregate(per)
