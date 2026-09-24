"""CoLA models, random-walk subgraphs, and source-to-target adaptation."""

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

_SPARSE_SCORE_SUPER_BATCHES = 64


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, seq.shape[-1])
        return torch.sum(torch.mul(seq, sim), 1)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    @staticmethod
    def _grouped_negative_index(group_sizes, total, device):
        """Return the official roll index independently inside each logical batch.

        CoLA uses ``cat(c[-2:-1], c[:-1])`` rather than a conventional roll.
        A final singleton batch makes that expression empty, so it reuses its own
        context to keep scoring defined.
        """
        sizes = np.asarray(group_sizes, dtype=np.int64)
        sizes = sizes[sizes > 0]
        if int(sizes.sum()) != total:
            raise ValueError(
                f"CoLA logical batch sizes sum to {int(sizes.sum())}, expected {total}"
            )
        ends = np.cumsum(sizes)
        starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
        neg_idx = np.arange(total, dtype=np.int64) - 1
        neg_idx[starts] = np.where(sizes == 1, starts, ends - 2)
        # Build on CPU once, then transfer once.  Constructing each 300-node
        # group separately on CUDA launches hundreds of tiny kernels per
        # super-batch and defeats the purpose of super-batching.
        return torch.from_numpy(neg_idx).to(device)

    def forward(self, c, h_pl, group_sizes=None, negative_index=None):
        scs = [self.f_k(h_pl, c)]
        c_mi = c
        if negative_index is not None and group_sizes is not None:
            raise ValueError("pass negative_index or group_sizes, not both")
        neg_idx = negative_index
        if neg_idx is None and group_sizes is not None:
            neg_idx = self._grouped_negative_index(group_sizes, c.shape[0], c.device)
        for _ in range(self.negsamp_round):
            if neg_idx is None:
                c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            else:
                c_mi = c_mi[neg_idx]
            scs.append(self.f_k(h_pl, c_mi))
        return torch.cat(tuple(scs))


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super().__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
        self.read = {
            "max": MaxReadout,
            "min": MinReadout,
            "avg": AvgReadout,
            "weighted_sum": WSReadout,
        }[readout]()
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, sparse=False, group_sizes=None, negative_index=None):
        h_1 = self.gcn(seq1, adj, sparse)
        if self.read_mode != "weighted_sum":
            c = self.read(h_1[:, :-1, :])
            h_mv = h_1[:, -1, :]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
        return self.disc(c, h_mv, group_sizes=group_sizes, negative_index=negative_index)


def _normalize_adj(adj):
    """Apply the released symmetric adjacency normalization."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.result_type(rowsum.dtype, np.float32))
    nonzero = rowsum > 0
    d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def _normalized_with_self_loops(adj_csr):
    """Official normalized adjacency plus I, without a float64 intermediate."""
    n = adj_csr.shape[0]
    eye = sp.eye(n, dtype=np.float32, format="csr")
    return (_normalize_adj(adj_csr) + eye).astype(np.float32).tocsr()


def gen_rwr_subgraphs(indptr, indices, n, subgraph_size, rng):
    """Sample fixed-size subgraphs using random walks with restart."""
    reduced = subgraph_size - 1
    deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
    seed = np.arange(n, dtype=np.int64)
    neigh = np.repeat(seed[:, None], reduced, axis=1)
    nz = deg > 0
    if np.any(nz):
        rand = (rng.random((int(nz.sum()), reduced)) * deg[nz, None]).astype(np.int64)
        neigh[nz] = indices[indptr[:-1][nz, None] + rand]
    return np.concatenate([neigh, seed[:, None]], axis=1)


def gen_rwr_subgraphs_for_nodes(indptr, indices, nodes, subgraph_size, rng):
    """RWR-style 1-hop subgraphs for a node batch.

    This is the scalable counterpart of `gen_rwr_subgraphs`: it samples only
    the requested seed nodes, so target inference can cover all eval nodes in
    chunks without materializing an (N, subgraph_size) array.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    reduced = subgraph_size - 1
    deg = (indptr[nodes + 1] - indptr[nodes]).astype(np.int64)
    neigh = np.repeat(nodes[:, None], reduced, axis=1)
    nz = deg > 0
    if np.any(nz):
        rand = (rng.random((int(nz.sum()), reduced)) * deg[nz, None]).astype(np.int64)
        neigh[nz] = indices[indptr[nodes[nz], None] + rand]
    return np.concatenate([neigh, nodes[:, None]], axis=1)


def prepare_graph(adj, feat, device, make_dense=True):
    """Prepare one graph once for any number of CoLA seeds/models."""
    n, ft = feat.shape
    adj_csr = sp.csr_matrix(adj, copy=False)
    adj_norm = _normalized_with_self_loops(adj_csr)
    adj_dense = torch.from_numpy(adj_norm.toarray()).to(device) if make_dense else None
    features = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32)).to(device)
    # Retain CSR indices as views; graph preparation and scoring do not mutate them.
    return {
        "n": n,
        "ft": ft,
        "raw_adj_csr": adj_csr,
        "indptr": adj_csr.indptr,
        "indices": adj_csr.indices,
        "adj_dense": adj_dense,
        "adj_norm_csr": adj_norm,
        "features": features,
        "negative_index_cache": {},
    }


def init_model(
    input_dim,
    device,
    *,
    embedding_dim=64,
    negsamp_ratio=1,
    readout="avg",
    lr=1e-3,
    weight_decay=0.0,
):
    model = Model(input_dim, embedding_dim, "prelu", negsamp_ratio, readout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, opt


def _build_batch(g, idx, subs, subgraph_size, device):
    cb = len(idx)
    sub_b = torch.from_numpy(subs[idx]).to(device)
    ba_core = g["adj_dense"][sub_b.unsqueeze(2), sub_b.unsqueeze(1)]
    ba = torch.zeros(
        (cb, subgraph_size + 1, subgraph_size + 1),
        dtype=ba_core.dtype,
        device=device,
    )
    ba[:, :subgraph_size, :subgraph_size] = ba_core
    ba[:, -1, -1] = 1.0
    picked = g["features"][sub_b]
    bf = torch.zeros(
        (cb, subgraph_size + 1, g["ft"]),
        dtype=picked.dtype,
        device=device,
    )
    bf[:, : subgraph_size - 1, :] = picked[:, :-1, :]
    bf[:, -1:, :] = picked[:, -1:, :]
    return bf, ba


def _build_batch_sparse(g, subs_batch, subgraph_size, device):
    """Build CoLA mini-batch tensors from sparse adjacency for selected seeds."""
    subs_batch = np.asarray(subs_batch, dtype=np.int64)
    cb = subs_batch.shape[0]
    adj = g["adj_norm_csr"]

    # Vectorize over all nodes for each local (row, col) position.  This remains
    # exactly equivalent to adj[nodes][:, nodes], including duplicate sampled
    # nodes, while avoiding B*c*c-sized int64 ``rows`` and ``cols`` temporaries.
    ba_np = np.zeros((cb, subgraph_size + 1, subgraph_size + 1), dtype=np.float32)
    for row_pos in range(subgraph_size):
        rows = subs_batch[:, row_pos]
        for col_pos in range(subgraph_size):
            values = adj[rows, subs_batch[:, col_pos]]
            ba_np[:, row_pos, col_pos] = np.asarray(values).reshape(-1)
    ba_np[:, -1, -1] = 1.0
    ba = torch.from_numpy(ba_np).to(device)
    sub_b = torch.from_numpy(subs_batch).to(device)
    picked = g["features"][sub_b]
    bf = torch.zeros(
        (cb, subgraph_size + 1, g["ft"]),
        dtype=picked.dtype,
        device=device,
    )
    bf[:, : subgraph_size - 1, :] = picked[:, :-1, :]
    bf[:, -1:, :] = picked[:, -1:, :]
    return bf, ba


def _logical_group_sizes(count, batch_size):
    full_groups, remainder = divmod(int(count), int(batch_size))
    groups = [int(batch_size)] * full_groups
    if remainder:
        groups.append(remainder)
    return groups


def _cached_negative_index(g, group_sizes, device):
    """Cache the fixed logical-batch roll index on the prepared graph/device."""
    key = (str(torch.device(device)), tuple(int(v) for v in group_sizes))
    cache = g["negative_index_cache"]
    if key not in cache:
        total = sum(key[1])
        cache[key] = Discriminator._grouped_negative_index(key[1], total, torch.device(device))
    return cache[key]


def _print_progress(kind, label, completed, total, started):
    """Render a dependency-free terminal progress bar for long CoLA stages."""
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    elapsed = time.perf_counter() - started
    eta = elapsed / completed * (total - completed) if completed else 0.0
    width = 20
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if completed == total else "\r"
    print(
        f"    [{kind}] {label} [{bar}] {completed}/{total} "
        f"({100.0 * completed / total:5.1f}%) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
        end=end,
        flush=True,
    )


def train_model_on_prepared_graph(
    model,
    opt,
    g,
    device,
    *,
    num_epoch,
    batch_size=300,
    subgraph_size=4,
    negsamp_ratio=1,
    seed=1,
    sparse_training=False,
    progress_label=None,
):
    """Train one model on an already prepared graph, preserving CoLA updates."""
    rng = np.random.RandomState(seed)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([negsamp_ratio], dtype=torch.float, device=device),
    )
    batch_num = g["n"] // batch_size + 1
    best, best_state = 1e9, None
    model.train()
    started = time.perf_counter()
    progress_every = max(1, num_epoch // 20)
    for epoch in range(num_epoch):
        subs = (
            None
            if sparse_training
            else gen_rwr_subgraphs(g["indptr"], g["indices"], g["n"], subgraph_size, rng)
        )
        all_idx = np.arange(g["n"])
        rng.shuffle(all_idx)
        total_loss, last_loss, last_cb = 0.0, 0.0, 0
        for b in range(batch_num):
            is_final = b == batch_num - 1
            idx = (
                all_idx[b * batch_size :]
                if is_final
                else all_idx[b * batch_size : (b + 1) * batch_size]
            )
            cb = len(idx)
            if cb == 0:
                continue
            lbl = torch.cat(
                [
                    torch.ones(cb, device=device),
                    torch.zeros(cb * negsamp_ratio, device=device),
                ]
            ).unsqueeze(1)
            if sparse_training:
                subs_batch = gen_rwr_subgraphs_for_nodes(
                    g["indptr"], g["indices"], idx, subgraph_size, rng
                )
                bf, ba = _build_batch_sparse(g, subs_batch, subgraph_size, device)
            else:
                bf, ba = _build_batch(g, idx, subs, subgraph_size, device)
            # The original roll is undefined for a one-node remainder.  Reusing
            # its own context is deterministic and affects only that singleton.
            negative_index = _cached_negative_index(g, [1], device) if cb == 1 else None
            logits = model(bf, ba, negative_index=negative_index)
            loss = torch.mean(b_xent(logits, lbl))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss, last_cb = loss.detach().cpu().item(), cb
            if not is_final:
                total_loss += last_loss
        mean_loss = (total_loss * batch_size + last_loss * last_cb) / g["n"]
        if mean_loss < best:
            best = mean_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        completed = epoch + 1
        if progress_label is not None and (
            completed == 1 or completed % progress_every == 0 or completed == num_epoch
        ):
            _print_progress("cola-train", progress_label, completed, num_epoch, started)
    if best_state is not None:
        model.load_state_dict(best_state)


def score_model_on_prepared_graph(
    model,
    g,
    device,
    *,
    batch_size=300,
    subgraph_size=4,
    auc_test_rounds=256,
    seed=1,
    score_idx=None,
    sparse_scoring=False,
    progress_label=None,
):
    """Score one model while reusing a prepared target graph across seeds."""
    rng = np.random.RandomState(seed)
    score_idx = (
        np.arange(g["n"], dtype=np.int64)
        if score_idx is None
        else np.asarray(score_idx, dtype=np.int64)
    )
    if score_idx.size == 0:
        return np.zeros(g["n"], dtype=np.float64)
    model.eval()
    # Accumulate in float64 on the model device; transfer only for progress
    # reports and the final result.
    score_sum = torch.zeros(g["n"], dtype=torch.float64, device=device)
    execution_batch_size = (
        batch_size * _SPARSE_SCORE_SUPER_BATCHES if sparse_scoring else batch_size
    )
    started = time.perf_counter()
    progress_every = max(1, auc_test_rounds // 16)

    with torch.inference_mode():
        for r in range(auc_test_rounds):
            subs = (
                None
                if sparse_scoring
                else gen_rwr_subgraphs(g["indptr"], g["indices"], g["n"], subgraph_size, rng)
            )
            all_idx = score_idx.copy()
            rng.shuffle(all_idx)
            for start in range(0, all_idx.size, execution_batch_size):
                idx = all_idx[start : start + execution_batch_size]
                cb = len(idx)
                if cb == 0:
                    continue
                group_sizes = _logical_group_sizes(cb, batch_size)
                if sparse_scoring:
                    subs_batch = gen_rwr_subgraphs_for_nodes(
                        g["indptr"], g["indices"], idx, subgraph_size, rng
                    )
                    bf, ba = _build_batch_sparse(g, subs_batch, subgraph_size, device)
                    negative_index = _cached_negative_index(g, group_sizes, device)
                else:
                    bf, ba = _build_batch(g, idx, subs, subgraph_size, device)
                    negative_index = _cached_negative_index(g, [1], device) if cb == 1 else None
                logits = torch.sigmoid(torch.squeeze(model(bf, ba, negative_index=negative_index)))
                neg_round = getattr(getattr(model, "disc", None), "negsamp_round", 1)
                pos = logits[:cb]
                neg = logits[cb : cb * (neg_round + 1)].reshape(neg_round, cb).mean(0)
                idx_device = torch.from_numpy(idx).to(device)
                # ``idx`` is a permutation slice, hence unique.  Explicit
                # gather/add/overwrite avoids CUDA atomic index_add and keeps
                # accumulation deterministic while retaining float64 order.
                score_sum[idx_device] = score_sum[idx_device] + (-(pos - neg)).to(torch.float64)

            completed = r + 1
            if progress_label is not None and (
                completed == 1 or completed % progress_every == 0 or completed == auc_test_rounds
            ):
                if g["features"].is_cuda:
                    torch.cuda.synchronize(g["features"].device)
                _print_progress("cola-score", progress_label, completed, auc_test_rounds, started)

    out = score_sum.div(max(auc_test_rounds, 1)).cpu().numpy()
    if score_idx.size != g["n"]:
        mask = np.zeros(g["n"], dtype=bool)
        mask[score_idx] = True
        out[~mask] = 0.0
    return out
