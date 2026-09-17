"""Trainable Ego-Neighbor Residual (ENR) graph encoder.

For X^(0)=X and X^(l)=P X^(l-1), a shared MLP produces Z^(l), then

    H = [(Z^(1)-Z^(0)) || ... || (Z^(L)-Z^(0))].

The discrepancy construction follows the anomaly-aware idea shared by ARC,
UNPrompt, and IA-GGAD. This module uses ARC's conditional-self-loop symmetric
normalization for P, but it is an encoder adapter rather than a reproduction of
ARC's complete model or objective. EdgeList
propagation uses a custom exact backward, so dense/large graphs keep every edge
without retaining E-by-hidden-dimension autograd intermediates.
"""

from __future__ import annotations

import scipy.sparse as sp
import torch
from torch import nn

from common.data import (
    BIG,
    EDGE_CHUNK,
    EdgeList,
    adj_sym_cond_scipy,
    exact_edge_spmm,
    exact_edge_spmm_transpose,
)
from util import to_torch_sparse

STREAM_GRAPHS = BIG | {"t_finance"}


class CompactEdgeList(EdgeList):
    """Foundation-only CPU edge stream with int32 resident indices.

    All benchmark graphs fit signed int32 node ids.  PyTorch aggregation still
    receives int64 indices, but conversion happens for one bounded GPU chunk
    instead of permanently doubling both E-sized CPU index arrays.
    """

    def __init__(self, adj):
        a = sp.coo_matrix(adj, copy=False).astype("float32", copy=False)
        if max(a.shape, default=0) >= 2**31:
            raise ValueError("CompactEdgeList requires int32 node ids")
        self.row = torch.from_numpy(a.row.astype("int32", copy=False))
        self.col = torch.from_numpy(a.col.astype("int32", copy=False))
        self.val = torch.from_numpy(a.data)
        self.shape = a.shape
        self.nnz = int(a.nnz)

    def chunks(self, device, chunk=EDGE_CHUNK):
        for start in range(0, self.nnz, int(chunk)):
            end = min(start + int(chunk), self.nnz)
            yield (
                self.row[start:end].to(device=device, dtype=torch.long),
                self.col[start:end].to(device=device, dtype=torch.long),
                self.val[start:end].to(device),
            )


class _CompactEdgeSpMM(torch.autograd.Function):
    """Exact streamed SpMM whose resident CPU indices remain int32."""

    @staticmethod
    def forward(ctx, z, row_cpu, col_cpu, val_cpu, n, chunk):
        ctx.row_cpu = row_cpu
        ctx.col_cpu = col_cpu
        ctx.val_cpu = val_cpu
        ctx.chunk = int(chunk)
        ctx.z_shape = tuple(z.shape)
        out = z.new_zeros((int(n), z.shape[1]))
        for start in range(0, val_cpu.numel(), ctx.chunk):
            end = min(start + ctx.chunk, val_cpu.numel())
            row = row_cpu[start:end].to(device=z.device, dtype=torch.long)
            col = col_cpu[start:end].to(device=z.device, dtype=torch.long)
            val = val_cpu[start:end].to(device=z.device, dtype=z.dtype).unsqueeze(1)
            out.index_add_(0, row, z[col] * val)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_z = grad_out.new_zeros(ctx.z_shape)
        for start in range(0, ctx.val_cpu.numel(), ctx.chunk):
            end = min(start + ctx.chunk, ctx.val_cpu.numel())
            row = ctx.row_cpu[start:end].to(device=grad_out.device, dtype=torch.long)
            col = ctx.col_cpu[start:end].to(device=grad_out.device, dtype=torch.long)
            val = (
                ctx.val_cpu[start:end].to(device=grad_out.device, dtype=grad_out.dtype).unsqueeze(1)
            )
            grad_z.index_add_(0, col, grad_out[row] * val)
        return grad_z, None, None, None, None, None


def edge_spmm(adj, x: torch.Tensor, edge_chunk: int = EDGE_CHUNK) -> torch.Tensor:
    """Exact A@X for torch sparse tensors or CPU-resident EdgeList."""
    if isinstance(adj, CompactEdgeList):
        return _CompactEdgeSpMM.apply(x, adj.row, adj.col, adj.val, adj.shape[0], int(edge_chunk))
    if isinstance(adj, EdgeList):
        return exact_edge_spmm(adj, x, chunk=edge_chunk)
    return torch.sparse.mm(adj, x)


def edge_spmm_transpose(adj, x: torch.Tensor, edge_chunk: int = EDGE_CHUNK) -> torch.Tensor:
    """Exact A.T@X, including bounded-memory EdgeList backward."""
    if isinstance(adj, CompactEdgeList):
        return _CompactEdgeSpMM.apply(x, adj.col, adj.row, adj.val, adj.shape[1], int(edge_chunk))
    if isinstance(adj, EdgeList):
        return exact_edge_spmm_transpose(adj, x, chunk=edge_chunk)
    return torch.sparse.mm(adj.transpose(0, 1), x)


def arc_operator(name: str, adj: sp.spmatrix, device: str, stream: bool | None = None):
    """Build ARC's P=D^-1/2 A_sl^T D^-1/2 without dropping edges."""
    norm = adj_sym_cond_scipy(name, sp.csr_matrix(adj)).astype("float32")
    use_stream = name in STREAM_GRAPHS if stream is None else bool(stream)
    return EdgeList(norm) if use_stream else to_torch_sparse(norm).to(device)


def arc_operator_from_edge_index(
    edge_index: torch.Tensor, num_nodes: int, self_loop_mask: torch.Tensor | None = None
):
    """ARC P for a PyG mini-batch of disjoint ego subgraphs."""
    src, dst = edge_index[0], edge_index[1]
    if self_loop_mask is None:
        loop = torch.arange(num_nodes, device=edge_index.device)
    else:
        loop = torch.nonzero(self_loop_mask.bool(), as_tuple=False).flatten()
    if loop.numel() > 0:
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
    val = torch.ones(src.numel(), dtype=torch.float32, device=edge_index.device)
    deg = torch.zeros(num_nodes, dtype=val.dtype, device=val.device)
    deg.index_add_(0, src, val)
    inv = torch.zeros_like(deg)
    nz = deg > 0
    inv[nz] = deg[nz].pow(-0.5)
    weight = val * inv[src] * inv[dst]
    # Raw adjacency uses row=source,col=target; ARC's operator is transposed,
    # hence output row=target and input col=source.
    return torch.sparse_coo_tensor(
        torch.stack([dst, src]), weight, (num_nodes, num_nodes)
    ).coalesce()


class EgoNeighborResidualEncoder(nn.Module):
    """Shared-MLP multi-hop ego-neighbor discrepancy encoder."""

    def __init__(
        self,
        in_dim: int,
        hop_dim: int,
        mlp_depth: int,
        num_hops: int,
        dropout: float = 0.0,
        activation: str = "ELU",
    ):
        super().__init__()
        if mlp_depth <= 0 or num_hops <= 0:
            raise ValueError("mlp_depth and num_hops must be positive")
        self.num_hops = int(num_hops)
        self.hop_dim = int(hop_dim)
        self.output_dim = self.hop_dim * self.num_hops
        dims = [int(in_dim)] + [self.hop_dim] * int(mlp_depth)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.act = getattr(nn, activation)()

    def propagate_views(
        self,
        x: torch.Tensor,
        adj,
        edge_chunk: int = EDGE_CHUNK,
        input_prompt: nn.Module | None = None,
    ):
        h = input_prompt(x) if input_prompt is not None else x
        views = [h]
        for _ in range(self.num_hops):
            views.append(edge_spmm(adj, views[-1], edge_chunk))
        return views

    def forward_views(self, views, prompt_layers=None):
        zs = list(views)
        if prompt_layers is not None and len(prompt_layers) != len(self.layers):
            raise ValueError(
                f"prompt layers={len(prompt_layers)} but residual MLP depth={len(self.layers)}"
            )
        for i, layer in enumerate(self.layers):
            if i > 0:
                zs = [self.dropout(z) for z in zs]
            zs = [layer(z) for z in zs]
            if i != len(self.layers) - 1:
                zs = [self.act(z) for z in zs]
            if prompt_layers is not None:
                zs = [prompt_layers[i](z) for z in zs]
        ego = zs[0]
        return torch.cat([z - ego for z in zs[1:]], dim=1)

    def _encode_one_view(self, z: torch.Tensor, prompt_layers=None) -> torch.Tensor:
        """Apply the shared MLP to one propagated view."""
        for i, layer in enumerate(self.layers):
            if i > 0:
                z = self.dropout(z)
            z = layer(z)
            if i != len(self.layers) - 1:
                z = self.act(z)
            if prompt_layers is not None:
                z = prompt_layers[i](z)
        return z

    def forward_inference(
        self,
        x: torch.Tensor,
        adj,
        edge_chunk: int = EDGE_CHUNK,
        prompt_layers=None,
        input_prompt: nn.Module | None = None,
    ):
        """Exact eval-only path with one hidden hop view live at a time.

        The ordinary training path intentionally keeps all views so autograd
        can backpropagate through the shared MLP.  Million-node target
        inference needs no graph, so materializing every N-by-hidden view at
        once only multiplies memory.  Propagated input views are low-dimensional
        and retained exactly; their shared-MLP outputs are written directly to
        the final concatenated tensor.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("forward_inference requires eval mode and no_grad")
        if prompt_layers is not None and len(prompt_layers) != len(self.layers):
            raise ValueError(
                f"prompt layers={len(prompt_layers)} but residual MLP depth={len(self.layers)}"
            )
        views = self.propagate_views(x, adj, edge_chunk, input_prompt=input_prompt)
        ego = self._encode_one_view(views[0], prompt_layers)
        out = ego.new_empty((ego.size(0), self.output_dim))
        for hop, view in enumerate(views[1:]):
            z = self._encode_one_view(view, prompt_layers)
            start = hop * self.hop_dim
            out[:, start : start + self.hop_dim] = z - ego
            del z
        return out

    def forward(
        self,
        x: torch.Tensor,
        adj,
        edge_chunk: int = EDGE_CHUNK,
        prompt_layers=None,
        input_prompt: nn.Module | None = None,
    ):
        views = self.propagate_views(x, adj, edge_chunk, input_prompt=input_prompt)
        return self.forward_views(views, prompt_layers=prompt_layers)
