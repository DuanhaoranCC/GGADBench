"""ARC source training and normal-support target evaluation."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam

from common.data import BIG, DIMS, adj_sym_cond, aggregate, chunk_propagate, load_aligned
from ggad.vendor.arc.model import ARC
from util import evaluate, set_seed

HP = {
    "h_feats": 1024,
    "num_layers": 4,
    "num_hops": 2,
    "num_prompt": 10,
    "drop_rate": 0,
    "activation": "ELU",
    "lr": 1e-5,
    "weight_decay": 5e-5,
}
CHUNK = 50_000


def _graph(name, num_hops, device, target=False):
    adj, feat, label, mark = load_aligned(name, target=target, hops=num_hops)
    if name in BIG:
        x_list = chunk_propagate(name, adj, feat, num_hops, device)
    else:
        adj_norm = adj_sym_cond(name, adj, device)
        x = torch.from_numpy(np.ascontiguousarray(feat, np.float32)).to(device)
        x_list = [x]
        for _ in range(num_hops):
            x_list.append(torch.sparse.mm(adj_norm, x_list[-1]))
    return SimpleNamespace(
        name=name,
        x_list=x_list,
        labels_np=label,
        mark=mark,
        ano_labels=torch.tensor(label, dtype=torch.float).to(device),
        n=len(label),
    )


def _arc_resid(model, x_list, idx):
    """Compute ARC residuals for selected nodes."""
    return model(SimpleNamespace(x_list=[x[idx] for x in x_list]))


def _arc_score(model, g, mask, chunk=CHUNK):
    """Score query nodes against labeled normal support in bounded chunks."""
    support_idx = (mask & (g.ano_labels == 0)).nonzero().squeeze(1)
    sup = _arc_resid(model, g.x_list, support_idx)
    query_idx = (~mask).nonzero().squeeze(1)
    scores = []
    for s in range(0, query_idx.numel(), chunk):
        qi = query_idx[s : s + chunk]
        q = _arc_resid(model, g.x_list, qi)
        qt = model.cross_attn.cross_attention(q, sup)
        scores.append(torch.sqrt(((q - qt) ** 2).sum(1)))
    return torch.cat(scores)


def run_arc(sources, targets, seeds, epochs, device, shot=10, train=True):
    nh = HP["num_hops"]
    src_g = [_graph(n, nh, device) for n in sources] if train else []
    per = {n: [] for n in targets}
    models = []
    for seed in seeds:
        set_seed(seed)
        model = ARC(
            in_feats=DIMS,
            h_feats=HP["h_feats"],
            num_layers=HP["num_layers"],
            dropout_rate=HP["drop_rate"],
            activation=HP["activation"],
            num_hops=nh,
        ).to(device)
        if train:
            opt = Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
            model.train()
            for _ in range(epochs):
                for g in src_g:
                    loss = model.cross_attn.get_train_loss(model(g), g.ano_labels, HP["num_prompt"])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        model.eval()
        model.zero_grad(set_to_none=True)
        model.cpu()
        models.append((seed, model))
        if train:
            del opt, loss
        torch.cuda.empty_cache()
    for g in src_g:
        del g.x_list, g.ano_labels
    src_g.clear()
    torch.cuda.empty_cache()
    for name in targets:
        try:
            g = _graph(name, nh, device, target=True)
        except RuntimeError as e:
            if name in BIG:
                raise
            print(f"    [arc] {name} graph construction skipped ({str(e)[:60]})")
            torch.cuda.empty_cache()
            continue
        for seed, model in models:
            model.to(device)
            set_seed(seed)
            normal = np.where((g.labels_np == 0) & g.mark)[0]
            np.random.shuffle(normal)
            mask = torch.zeros(g.n, dtype=torch.bool, device=device)
            mask[torch.as_tensor(normal[:shot], device=device)] = True
            with torch.no_grad():
                try:
                    qscore = _arc_score(model, g, mask)
                except RuntimeError as e:
                    if name in BIG:
                        raise
                    print(f"    [arc] {name} skipped ({str(e)[:60]})")
                    model.cpu()
                    torch.cuda.empty_cache()
                    continue
            qidx = (~mask).nonzero().squeeze(1).cpu().numpy()
            keep = g.mark[qidx]
            per[name].append(evaluate(g.labels_np[qidx][keep], qscore.cpu().numpy()[keep]))
            model.cpu()
            torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()
    return aggregate(per)
