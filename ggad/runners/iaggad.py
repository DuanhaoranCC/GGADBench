"""IA-GGAD source training and affinity-enhanced target evaluation."""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from torch.optim import AdamW

from common.data import (
    BIG,
    DIMS,
    EDGE_CHUNK,
    EdgeList,
    adj_sym_cond,
    aggregate,
    chunk_propagate,
    column_degree_row_values,
    edge_chunks,
    exact_edge_spmm,
    load_aligned,
)
from ggad.vendor.iaggad.graph import build_aff, max_message, my_GCN
from ggad.vendor.iaggad.graph import normalize_adj as _iaggad_normalize_adj
from ggad.vendor.iaggad.graph import normalize_score
from ggad.vendor.iaggad.model import GCN
from util import evaluate, remove_self_loop, set_seed

HP = {
    "h_feats": 1024,
    "num_layers": 4,
    "num_hops": 2,
    "num_prompt": 10,
    "drop_rate": 0,
    "activation": "ELU",
    "lr": 1e-5,
    "weight_decay": 5e-5,
    "code_size": 2048,
    "topk": 15,
    "gcn_emb_dim": 128,
}


LAM_DATASET = {
    "ACM": 0.5,
    "citeseer": 0.5,
    "cora": 0.5,
    "weibo": 0.5,
    "Amazon": 0.5,
    "Facebook": 0.5,
    "BlogCatalog": 0.5,
    "Reddit": 0.5,
}
LAM_DEFAULT = 0.5
CHUNK = 50_000


STREAM_GRAPHS = BIG | {"t_finance"}
AFF_TRAIN_EDGE_CHUNK = 250_000


def _build_aff_big(adj, device):
    """Build binary and normalized CPU edge lists for affinity propagation."""
    a = sp.csr_matrix(adj).astype(np.float32)
    a.data[:] = 1.0
    a = remove_self_loop(a)
    n = a.shape[0]
    a_sl = (a + sp.eye(n, dtype=np.float32)).tocsr()
    aff_norm = EdgeList(_iaggad_normalize_adj(a_sl))
    aff_bin = EdgeList(a_sl)
    return aff_norm, aff_bin


def _affinity_embed_train_big(gcn_aff, aff_norm, x):
    """Compute trainable affinity embeddings using full-edge propagation."""
    h = torch.relu(
        exact_edge_spmm(aff_norm, gcn_aff.W1(x), chunk=AFF_TRAIN_EDGE_CHUNK) + gcn_aff.b1
    )
    h = torch.relu(
        exact_edge_spmm(aff_norm, gcn_aff.W2(h), chunk=AFF_TRAIN_EDGE_CHUNK) + gcn_aff.b2
    )
    return h


def _max_message_train_big(feature, aff_bin):
    """Compute the full-edge affinity loss without materializing edge embeddings."""
    feat = feature / torch.norm(feature, dim=-1, keepdim=True)
    loss_val = column_degree_row_values(aff_bin)
    agg = exact_edge_spmm(aff_bin, feat, val=loss_val, chunk=AFF_TRAIN_EDGE_CHUNK)
    return -torch.sum(feat * agg), None


def _max_message_big(feature, aff_bin, device, chunk=EDGE_CHUNK):
    """Compute node affinity messages by streaming the complete edge list."""
    feat = feature / torch.norm(feature, dim=-1, keepdim=True)
    n = feature.size(0)
    message = torch.zeros(n, device=device)
    deg = torch.zeros(n, device=device)
    for r, c, v in edge_chunks(aff_bin, device, chunk):
        sim = torch.nan_to_num((feat[r] * feat[c]).sum(1))
        message.index_add_(0, r, sim * v)
        deg.index_add_(0, c, v)
    r_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    message = message * r_inv
    return -torch.sum(message), message


def _affinity_embed(gcn_aff, aff_norm, x, device, node_chunk=200_000):
    """Compute affinity embeddings with bounded intermediate device memory."""
    if not isinstance(aff_norm, EdgeList):
        return gcn_aff(aff_norm, x)

    @torch.no_grad()
    def _lin_to_cpu(inp, W):
        out = torch.empty((inp.shape[0], W.weight.shape[0]), dtype=torch.float32)
        on_gpu = inp.is_cuda
        for s in range(0, inp.shape[0], node_chunk):
            e = min(s + node_chunk, inp.shape[0])
            blk = inp[s:e] if on_gpu else inp[s:e].to(device)
            out[s:e] = W(blk).cpu()
        return out

    aff_chunk = 250_000

    @torch.no_grad()
    def _spmm(zcpu):
        out = torch.zeros((aff_norm.shape[0], zcpu.shape[1]), device=device)
        for s in range(0, aff_norm.nnz, aff_chunk):
            e = min(s + aff_chunk, aff_norm.nnz)
            r = aff_norm.row[s:e].to(device)
            v = aff_norm.val[s:e].to(device).unsqueeze(1)
            out.index_add_(0, r, zcpu[aff_norm.col[s:e]].to(device) * v)
        return out

    z1 = _lin_to_cpu(x, gcn_aff.W1)  # (N,256) CPU
    p1 = _spmm(z1)
    p1 += gcn_aff.b1
    torch.relu_(p1)
    h1 = p1.cpu()
    del p1
    z2 = _lin_to_cpu(h1, gcn_aff.W2)  # (N,128) CPU
    p2 = _spmm(z2)  # (N,128) GPU
    p2 += gcn_aff.b2
    torch.relu_(p2)
    return p2  # (N,128) GPU = node_emb


def _graph(name, num_hops, device, target=False):
    adj, feat, label, mark = load_aligned(name, target=target)
    if name in STREAM_GRAPHS:
        x_list = chunk_propagate(name, adj, feat, num_hops, device)
        x = x_list[0]
        aff_norm, aff_bin = _build_aff_big(adj, device)
    else:
        adj_norm = adj_sym_cond(name, adj, device)
        x = torch.from_numpy(np.ascontiguousarray(feat, np.float32)).to(device)
        x_list = [x]
        for _ in range(num_hops):
            x_list.append(torch.sparse.mm(adj_norm, x_list[-1]))
        del adj_norm
        torch.cuda.empty_cache()
        aff_norm, aff_bin = build_aff(adj, device)
    return SimpleNamespace(
        name=name,
        x=x,
        x_list=x_list,
        labels_np=label,
        mark=mark,
        ano_labels=torch.tensor(label, dtype=torch.float).to(device),
        aff_norm=aff_norm,
        aff_bin=aff_bin,
        n=len(label),
    )


def _iaggad_resid(model, g, idx):
    """Compute IA-GGAD residuals for selected nodes."""
    mini = SimpleNamespace(x_list=[x[idx] for x in g.x_list], name=g.name)
    return model(mini, mini)[0]


def _iaggad_score(model, gcn_aff, g, mask, final_codebook, lam, chunk=CHUNK):
    """Combine support/codebook distances with graph affinity anomaly scores."""
    support_idx = mask.nonzero().squeeze(1)
    sup = _iaggad_resid(model, g, support_idx)
    code_n = final_codebook[(sup @ final_codebook.T).argmax(1)]
    mean_sup, mean_code = sup.mean(0, keepdim=True), code_n.mean(0, keepdim=True)
    query_idx = (~mask).nonzero().squeeze(1)
    qs = []
    for s in range(0, query_idx.numel(), chunk):
        q = _iaggad_resid(model, g, query_idx[s : s + chunk])
        d = torch.sqrt(((q - mean_sup) ** 2).sum(1))
        dc = torch.sqrt(((q - mean_code) ** 2).sum(1))
        qs.append((d + dc) / 2)
    qs = torch.cat(qs)
    node_emb = _affinity_embed(gcn_aff, g.aff_norm, g.x, g.x.device)
    if isinstance(g.aff_bin, EdgeList):
        _, message = _max_message_big(node_emb, g.aff_bin, g.x.device)
    else:
        _, message = max_message(node_emb, g.aff_bin)
    fm = torch.from_numpy(1 - normalize_score(message.cpu().numpy())).float().to(g.x.device)
    return (1 - lam) * qs + lam * fm[~mask]


def _safe_train_prompt_loss(model, residual, quantized, codebook, labels, requested_prompts):
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    n_prompt = min(requested_prompts, n_neg - n_pos)
    if n_pos <= 0 or n_prompt <= 0:
        return None
    try:
        return model.cross_attn.get_train_loss(residual, quantized, codebook, labels, n_prompt)
    except ValueError:
        return None


def run_iaggad(sources, targets, support_type, seeds, epochs, device, shot=10):
    nh = HP["num_hops"]
    margs = SimpleNamespace(code_size=HP["code_size"], topk=HP["topk"])
    src_g = [_graph(n, nh, device) for n in sources]
    per = {n: [] for n in targets}
    trained = []
    for seed in seeds:
        set_seed(seed)
        model = GCN(
            margs,
            in_feats=DIMS,
            h_feats=HP["h_feats"],
            num_layers=HP["num_layers"],
            dropout_rate=HP["drop_rate"],
            activation=HP["activation"],
            num_hops=nh,
        ).to(device)
        model.concat_datasets = ["weibo", "BlogCatalog"]
        gcn_aff = my_GCN(DIMS, HP["gcn_emb_dim"]).to(device)
        opt = AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        opt_aff = AdamW(gcn_aff.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
        codebooks = {}
        model.train()
        gcn_aff.train()
        for e in range(epochs):
            for i, g in enumerate(src_g):
                residual, loss_code, quantized, codebook = model(g, g)

                prompt_loss = _safe_train_prompt_loss(
                    model, residual, quantized, codebook, g.ano_labels, HP["num_prompt"]
                )
                loss = loss_code.squeeze()
                if prompt_loss is not None:
                    loss = loss + prompt_loss
                if isinstance(g.aff_norm, EdgeList):

                    node_emb = _affinity_embed_train_big(gcn_aff, g.aff_norm, g.x)
                    loss_cut, _ = _max_message_train_big(node_emb, g.aff_bin)
                else:
                    node_emb = gcn_aff(g.aff_norm, g.x)
                    loss_cut, _ = max_message(node_emb, g.aff_bin)
                opt.zero_grad()
                opt_aff.zero_grad()
                loss.backward()
                loss_cut.backward()
                opt.step()
                opt_aff.step()
                if e == epochs - 1:
                    codebooks[i] = codebook.detach()
        final_codebook = torch.cat([codebooks[i] for i in range(len(src_g))], dim=0)
        model.eval()
        gcn_aff.eval()

        model.zero_grad(set_to_none=True)
        gcn_aff.zero_grad(set_to_none=True)
        model.cpu()
        gcn_aff.cpu()
        trained.append((seed, model, gcn_aff, final_codebook.cpu()))
        del opt, opt_aff, codebooks
        del final_codebook
        torch.cuda.empty_cache()
    for g in src_g:
        del g.x_list, g.aff_norm, g.aff_bin, g.x
    src_g.clear()
    torch.cuda.empty_cache()
    for name in targets:
        try:
            g = _graph(name, nh, device, target=True)
        except RuntimeError as e:
            if name in BIG:
                raise
            print(f"    [iaggad] {name} graph construction skipped ({str(e)[:60]})")
            torch.cuda.empty_cache()
            continue
        for seed, model, gcn_aff, final_codebook_cpu in trained:
            model.to(device)
            gcn_aff.to(device)
            final_codebook = final_codebook_cpu.to(device)
            try:
                set_seed(seed)
                pool = (
                    np.where((g.labels_np == 0) & g.mark)[0]
                    if support_type == "normal"
                    else np.where(g.mark)[0]
                )
                np.random.shuffle(pool)
                mask = torch.zeros(g.n, dtype=torch.bool, device=device)
                mask[torch.as_tensor(pool[:shot], device=device)] = True
                lam = LAM_DATASET.get(name, LAM_DEFAULT)
                with torch.no_grad():
                    qs = _iaggad_score(model, gcn_aff, g, mask, final_codebook, lam)
                qidx = (~mask).nonzero().squeeze(1).cpu().numpy()
                keep = g.mark[qidx]
                per[name].append(evaluate(g.labels_np[qidx][keep], qs.cpu().numpy()[keep]))
            except RuntimeError as e:
                if name in BIG:
                    raise
                print(f"    [iaggad] {name} skipped ({str(e)[:60]})")
            finally:
                model.cpu()
                gcn_aff.cpu()
                del final_codebook
                torch.cuda.empty_cache()
        del g
        torch.cuda.empty_cache()
    return aggregate(per)
