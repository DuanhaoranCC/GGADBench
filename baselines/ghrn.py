"""GHRN graph pruning utilities.

Official GHRN first runs BWGNN to get class probabilities `pred_y`, then
computes the high-frequency residual `pred_y - A_norm pred_y` and removes edges
with the smallest residual inner products.  The official implementation uses
DGL; this file keeps the same math with torch/PyG `edge_index` tensors.
"""

import torch
from torch_geometric.utils import add_self_loops, remove_self_loops


def _edge_norm(edge_index, num_nodes, adj_type="sym"):
    src, dst = edge_index
    dtype = torch.float32
    device = edge_index.device
    out_deg = torch.bincount(src, minlength=num_nodes).to(device=device, dtype=dtype).clamp(min=1)
    in_deg = torch.bincount(dst, minlength=num_nodes).to(device=device, dtype=dtype).clamp(min=1)
    if adj_type == "rw":
        return 1.0 / in_deg[dst]
    if adj_type != "sym":
        raise ValueError(f"unknown GHRN adj_type={adj_type}")
    return torch.rsqrt(out_deg[src] * in_deg[dst])


def _edge_norm_chunk(src, dst, out_deg, in_deg, adj_type):
    if adj_type == "rw":
        return 1.0 / in_deg[dst]
    if adj_type != "sym":
        raise ValueError(f"unknown GHRN adj_type={adj_type}")
    return torch.rsqrt(out_deg[src] * in_deg[dst])


@torch.no_grad()
def prune_edges_by_ghrn(
    edge_index,
    pred_y,
    num_nodes,
    del_ratio=0.015,
    adj_type="sym",
    final_self_loop=True,
    chunk_edges=2_000_000,
):
    """Return a pruned edge_index following official GHRN's random_walk_update.

    Args:
        edge_index: directed edges, shape (2, E), without any required self-loop convention.
        pred_y: node class probabilities, shape (N, C).
        num_nodes: number of nodes.
        del_ratio: fraction of edges to remove, after temporarily adding self-loops.
        adj_type: official option, "sym" or "rw".
        final_self_loop: whether the graph used by BWGNN should contain self-loops.
    """
    if del_ratio <= 0:
        edge_index, _ = remove_self_loops(edge_index)
        if final_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    dtype = torch.float32
    device = edge_index.device
    out_deg = torch.bincount(src, minlength=num_nodes).to(device=device, dtype=dtype).clamp(min=1)
    in_deg = torch.bincount(dst, minlength=num_nodes).to(device=device, dtype=dtype).clamp(min=1)

    # Compute A_norm pred_y by edge chunks.  The original vectorized form
    # materializes |E| x C messages; on T-Social-size graphs this is exactly
    # where GHRN runs out of memory.  Chunking keeps the same math and all
    # nodes/edges, only reducing temporary memory.
    ay = torch.zeros_like(pred_y)
    for start in range(0, src.numel(), chunk_edges):
        end = min(start + chunk_edges, src.numel())
        s = src[start:end]
        d = dst[start:end]
        w = _edge_norm_chunk(s, d, out_deg, in_deg, adj_type).to(pred_y.device)
        ay.index_add_(0, d, pred_y[s] * w.unsqueeze(-1))
    ly = pred_y - ay

    n_drop = int(del_ratio * edge_index.shape[1])
    if n_drop > 0:
        # Keep only the global bottom-k edge residual products.  This replaces
        # torch.argsort(black) over all edges, which can allocate several large
        # edge-length tensors.  The selected drop set is identical to full
        # sorting up to ties.
        best_scores = None
        best_idx = None
        for start in range(0, src.numel(), chunk_edges):
            end = min(start + chunk_edges, src.numel())
            s = src[start:end]
            d = dst[start:end]
            scores = (ly[s] * ly[d]).sum(dim=1)
            idx = torch.arange(start, end, dtype=torch.long, device=edge_index.device)
            if best_scores is None:
                cand_scores, cand_idx = scores, idx
            else:
                cand_scores = torch.cat([best_scores, scores], dim=0)
                cand_idx = torch.cat([best_idx, idx], dim=0)
            if cand_scores.numel() > n_drop:
                best_scores, pos = torch.topk(cand_scores, k=n_drop, largest=False)
                best_idx = cand_idx[pos]
            else:
                best_scores, best_idx = cand_scores, cand_idx

        keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        keep[best_idx] = False
        edge_index = edge_index[:, keep]

    edge_index, _ = remove_self_loops(edge_index)
    if final_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index
