"""DGL-free graph operators for the official TA-GGAD release.

The low-order branch intentionally reproduces DGL ``GraphConv(norm="both")``
with PyG ``MessagePassing``.  The high-order branch keeps the release's SciPy /
Torch sparse orientation.  Edge chunking only bounds temporary memory; it does
not sample nodes or edges.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from torch import Tensor
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing

ROW_NORMALIZED_DATASETS = {
    "Amazon",
    "Amazon-all",
    "YelpChi",
    "YelpChi-all",
    "tolokers",
    "tfinance",
    "t_finance",
    "DGgraph-fin",
    "dgraphfin",
}
NO_HIGH_ORDER_SELF_LOOP = {"YelpChi", "YelpChi-all", "Facebook"}


def scipy_to_torch_sparse(matrix: sp.spmatrix) -> Tensor:
    """Match the release's float32 SciPy COO -> Torch sparse conversion."""
    coo = sp.coo_matrix(matrix).astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def row_normalize_features(features):
    """Official ``D^-1 X`` feature normalization (including its zero-row rule)."""
    if not sp.issparse(features):
        features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    with np.errstate(divide="ignore"):
        inverse = np.power(rowsum, -1).flatten()
    inverse[np.isinf(inverse)] = 0.0
    features = sp.diags(inverse).dot(features)
    return features.todense()


def raw_edge_index(adjacency: sp.spmatrix) -> Tensor:
    """Return raw SciPy edge orientation used by official feature alignment."""
    matrix = sp.csr_matrix(adjacency)
    row, col = matrix.nonzero()
    return torch.from_numpy(np.vstack((row, col)).astype(np.int64, copy=False))


def feature_alignment(features, edge_index: Tensor, dims: int = 64) -> Tensor:
    """Official GRP256 -> PCA64 -> graph-smoothness component ordering.

    Deliberately keeps the release's epsilon-free min-max scaling because that
    scaling is part of strict parity (constant PCA columns may therefore yield
    NaNs, just as in the release).
    """
    if torch.is_tensor(features):
        # Keep a Torch tensor for PCA: the official code passes FloatTensor
        # directly to sklearn when no GRP step is needed.  Converting it to a
        # NumPy view first measurably changes float32 PCA round-off/order.
        values = features.detach().cpu()
    elif sp.issparse(features):
        values = np.asarray(features.todense())
    else:
        values = np.asarray(features)

    if values.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        projection_input = values.cpu().numpy() if torch.is_tensor(values) else values
        values = transformer.fit_transform(projection_input)

    transformed = PCA(n_components=dims, random_state=0).fit_transform(values)
    transformed = torch.FloatTensor(transformed)
    minimum = torch.min(transformed, dim=0).values
    maximum = torch.max(transformed, dim=0).values
    scaled = (transformed - minimum) / (maximum - minimum)

    src, dst = edge_index.detach().cpu()
    num_edges = int(src.numel())
    smoothness = torch.zeros(transformed.shape[1])
    for column in range(transformed.shape[1]):
        differences = scaled[src, column] - scaled[dst, column]
        smoothness[column] = torch.sum(differences**2) / num_edges
    order = torch.sort(smoothness).indices
    return transformed[:, order]


def prepare_aligned_features(
    adjacency: sp.spmatrix,
    features,
    dataset_name: str,
    dims: int = 64,
) -> Tensor:
    """Apply TA-GGAD's dataset-conditional row normalization and alignment."""
    if dataset_name in ROW_NORMALIZED_DATASETS:
        values = row_normalize_features(features)
    elif sp.issparse(features):
        values = np.asarray(sp.lil_matrix(features).toarray())
    else:
        values = np.asarray(features)
    # Official Dataset always casts the loaded/aligned input to FloatTensor
    # immediately before feat_alignment (including float64 MAT inputs).
    values = torch.FloatTensor(np.asarray(values))
    return feature_alignment(values, raw_edge_index(adjacency), dims=dims)


def normalize_high_order_adjacency(adjacency: sp.spmatrix) -> sp.coo_matrix:
    """Official ``A D^-1/2`` transpose ``D^-1/2`` operator.

    For asymmetric input this is ``D^-1/2 A.T D^-1/2``.  The orientation is
    intentionally not replaced by a conventional directed normalization.
    """
    matrix = sp.coo_matrix(adjacency)
    rowsum = np.asarray(matrix.sum(1))
    with np.errstate(divide="ignore"):
        inverse_sqrt = np.power(rowsum, -0.5).flatten()
    inverse_sqrt[np.isinf(inverse_sqrt)] = 0.0
    degree = sp.diags(inverse_sqrt)
    return matrix.dot(degree).transpose().dot(degree).tocoo()


def build_high_order_adjacency(
    adjacency: sp.spmatrix,
    dataset_name: str,
) -> sp.coo_matrix:
    """Apply the release's conditional high-order self-loop policy."""
    matrix = sp.csr_matrix(adjacency)
    if dataset_name not in NO_HIGH_ORDER_SELF_LOOP:
        matrix = matrix + sp.eye(matrix.shape[0])
    return normalize_high_order_adjacency(matrix)


def prepare_affinity_score_edge_index(
    adjacency: sp.spmatrix,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the edge order seen by official ``adj().coalesce()`` scoring.

    DGL's graph keeps non-loop edges followed by newly appended self-loops for
    GraphConv.  The released TAM code then converts that graph to a Torch sparse
    adjacency and calls ``coalesce()``, which canonicalizes duplicate entries
    and sorts coordinates.  Keep this separate from the GraphConv edge list so
    both official execution paths retain their own exact structure/order.
    """
    matrix = sp.csr_matrix(adjacency, copy=True)
    matrix.sum_duplicates()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.setdiag(1)
    matrix.sum_duplicates()
    matrix.sort_indices()
    coo = matrix.tocoo(copy=False)
    edge_index = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


class _ExactEdgeSpMM(Function):
    """Exact streamed ``A @ X`` with full-edge forward and backward."""

    @staticmethod
    def forward(ctx, x, row, col, value, num_nodes, chunk_size):
        ctx.row = row
        ctx.col = col
        ctx.value = value
        ctx.chunk_size = int(chunk_size)
        ctx.x_shape = tuple(x.shape)
        output = x.new_zeros((int(num_nodes), x.shape[1]))
        for start in range(0, value.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, value.numel())
            r = row[start:end].to(x.device)
            c = col[start:end].to(x.device)
            weight = value[start:end].to(device=x.device, dtype=x.dtype).unsqueeze(1)
            output.index_add_(0, r, x[c] * weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.new_zeros(ctx.x_shape)
        for start in range(0, ctx.value.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, ctx.value.numel())
            r = ctx.row[start:end].to(grad_output.device)
            c = ctx.col[start:end].to(grad_output.device)
            weight = (
                ctx.value[start:end]
                .to(device=grad_output.device, dtype=grad_output.dtype)
                .unsqueeze(1)
            )
            grad_x.index_add_(0, c, grad_output[r] * weight)
        return grad_x, None, None, None, None, None


def exact_edge_spmm(
    row: Tensor,
    col: Tensor,
    value: Tensor,
    x: Tensor,
    num_nodes: int,
    chunk_size: int = 500_000,
) -> Tensor:
    """Apply a CPU- or GPU-resident edge list without dropping any edge."""
    return _ExactEdgeSpMM.apply(x, row, col, value, int(num_nodes), int(chunk_size))


def propagate_high_order(
    adjacency,
    features: Tensor,
    num_hops: int,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
) -> list[Tensor]:
    """Return ``[X, P X, ..., P^k X]`` using full sparse propagation."""
    target_device = torch.device(device) if device is not None else features.device
    current = features.to(target_device)
    outputs = [current]

    if sp.issparse(adjacency):
        operator = scipy_to_torch_sparse(adjacency)
    else:
        operator = adjacency.coalesce()

    if chunk_size is None:
        operator = operator.to(target_device)
        for _ in range(num_hops):
            current = torch.sparse.mm(operator, current)
            outputs.append(current)
        return outputs

    operator = operator.coalesce()
    row, col = operator.indices()
    values = operator.values()
    for _ in range(num_hops):
        current = exact_edge_spmm(row, col, values, current, operator.shape[0], chunk_size)
        outputs.append(current)
    return outputs


def prepare_affinity_edge_index(
    adjacency: sp.spmatrix,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Reproduce ``dgl.from_scipy -> remove_self_loop -> add_self_loop``.

    Directed non-loop edges and their multiplicity are retained.  Existing
    loops are all removed, then exactly one unweighted loop is appended for
    every node.  The returned edge list is deliberately not coalesced.
    """
    matrix = sp.coo_matrix(adjacency, copy=False)
    row = np.asarray(matrix.row, dtype=np.int64)
    col = np.asarray(matrix.col, dtype=np.int64)
    keep = row != col
    nodes = np.arange(matrix.shape[0], dtype=np.int64)
    row = np.concatenate((row[keep], nodes))
    col = np.concatenate((col[keep], nodes))
    edge_index = torch.from_numpy(np.vstack((row, col)))
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def _degree_norms(edge_index: Tensor, num_nodes: int, dtype, device):
    src = edge_index[0]
    dst = edge_index[1]
    out_degree = torch.bincount(src, minlength=num_nodes).to(device=device, dtype=dtype)
    in_degree = torch.bincount(dst, minlength=num_nodes).to(device=device, dtype=dtype)
    src_norm = out_degree.clamp(min=1).pow(-0.5)
    dst_norm = in_degree.clamp(min=1).pow(-0.5)
    return src_norm, dst_norm, in_degree


class DGLBothGraphConv(MessagePassing):
    """PyG implementation of DGL ``GraphConv(norm='both')``.

    The parameter is stored in PyTorch/PyG linear layout ``[out, in]``.  Its
    initializer first creates the official DGL-layout ``[in, out]`` tensor and
    transposes it, which also preserves the release's RNG consumption.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias: bool = True,
        allow_zero_in_degree: bool = False,
    ):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.in_feats = int(in_feats)
        self.out_feats = int(out_feats)
        self.allow_zero_in_degree = allow_zero_in_degree
        self.weight = nn.Parameter(torch.empty(out_feats, in_feats))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feats))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        dgl_layout = self.weight.new_empty((self.in_feats, self.out_feats))
        nn.init.xavier_uniform_(dgl_layout)
        with torch.no_grad():
            self.weight.copy_(dgl_layout.transpose(0, 1))
            if self.bias is not None:
                self.bias.zero_()

    def load_dgl_parameters(self, weight: Tensor, bias: Optional[Tensor] = None):
        """Copy a DGL GraphConv state into this layer."""
        with torch.no_grad():
            self.weight.copy_(weight.transpose(0, 1))
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def _aggregate(
        self,
        x: Tensor,
        edge_index: Tensor,
        chunk_size: Optional[int],
    ) -> Tensor:
        if chunk_size is None:
            return self.propagate(edge_index.to(x.device), x=x, size=(x.shape[0], x.shape[0]))
        src, dst = edge_index[0], edge_index[1]
        values = torch.ones(src.numel(), dtype=x.dtype, device=src.device)
        return exact_edge_spmm(dst, src, values, x, x.shape[0], chunk_size=chunk_size)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        num_nodes = x.shape[0]
        degree_edges = edge_index
        src_norm, dst_norm, in_degree = _degree_norms(degree_edges, num_nodes, x.dtype, x.device)
        if not self.allow_zero_in_degree and torch.any(in_degree == 0):
            raise RuntimeError(
                "DGLBothGraphConv found zero-in-degree nodes; add self-loops "
                "or set allow_zero_in_degree=True"
            )

        h = x * src_norm.unsqueeze(1)
        if self.in_feats > self.out_feats:
            # DGL reduces feature width before message passing in this branch.
            h = F.linear(h, self.weight)
            h = self._aggregate(h, edge_index, chunk_size)
        else:
            h = self._aggregate(h, edge_index, chunk_size)
            h = F.linear(h, self.weight)
        h = h * dst_norm.unsqueeze(1)
        if self.bias is not None:
            h = h + self.bias
        return h


class TAGCN(nn.Module):
    """Official TA-GGAD low-order GCN, with DGL replaced by PyG."""

    def __init__(self, in_feats: int, h_feats: int = 128):
        super().__init__()
        self.conv1 = DGLBothGraphConv(in_feats, 2 * h_feats)
        self.conv2 = DGLBothGraphConv(2 * h_feats, h_feats)
        # Present but unused in the released forward path.
        self.fc1 = nn.Linear(h_feats, h_feats, bias=False)
        self.fc2 = nn.Linear(h_feats, h_feats, bias=False)

    def forward(
        self,
        edge_index: Tensor,
        x: Tensor,
        num_nodes: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        if num_nodes is not None and int(num_nodes) != x.shape[0]:
            raise ValueError(f"num_nodes={num_nodes} does not match x.shape[0]={x.shape[0]}")
        hidden = F.relu(self.conv1(x, edge_index, chunk_size=chunk_size))
        return F.relu(self.conv2(hidden, edge_index, chunk_size=chunk_size))


def _edge_cosine(features: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    return (features[src] * features[dst]).sum(dim=1)


def _minmax_eps(values: Tensor, eps: float = 1e-8) -> Tensor:
    return (values - values.min()) / (values.max() - values.min() + eps)


class _ChunkedNeighborScores(Function):
    """Exact affinity scores with an edge-streamed cosine backward pass.

    L1/L2 are inference-only in the official objective.  Cosine gradients are
    reconstructed from node-level tensors during backward, so autograd never
    retains one graph node per edge chunk.
    """

    @staticmethod
    def forward(ctx, node_features, edge_index, num_nodes, chunk_size):
        count = int(num_nodes)
        step = int(chunk_size)
        device = node_features.device
        dtype = node_features.dtype
        src_all, dst_all = edge_index[0], edge_index[1]

        degree = torch.zeros(count, device=device, dtype=dtype)
        sum_l2 = torch.zeros_like(degree)
        sum_l1 = torch.zeros_like(degree)
        sum_cos = torch.zeros_like(degree)
        norms = torch.norm(node_features, dim=-1, keepdim=True)
        denominators = norms + 1e-8
        normed = node_features / denominators

        for start in range(0, src_all.numel(), step):
            end = min(start + step, src_all.numel())
            src = src_all[start:end].to(device)
            dst = dst_all[start:end].to(device)
            difference = node_features[src] - node_features[dst]
            degree.index_add_(0, src, torch.ones_like(src, dtype=dtype))
            sum_l2.index_add_(0, src, torch.norm(difference, p=2, dim=1))
            sum_l1.index_add_(0, src, torch.norm(difference, p=1, dim=1))
            sum_cos.index_add_(0, src, (normed[src] * normed[dst]).sum(dim=1))

        ctx.edge_index = edge_index
        ctx.chunk_size = step
        ctx.save_for_backward(normed, norms, denominators, degree)
        divisor = degree + 1e-8
        return sum_l2 / divisor, sum_l1 / divisor, sum_cos / divisor

    @staticmethod
    def backward(ctx, grad_l2, grad_l1, grad_cos):
        normed, norms, denominators, degree = ctx.saved_tensors
        if grad_cos is None:
            return torch.zeros_like(normed), None, None, None

        device = normed.device
        src_all, dst_all = ctx.edge_index[0], ctx.edge_index[1]
        grad_normed = torch.zeros_like(normed)
        divisor = degree + 1e-8

        for start in range(0, src_all.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, src_all.numel())
            src = src_all[start:end].to(device)
            dst = dst_all[start:end].to(device)
            weight = (grad_cos[src] / divisor[src]).unsqueeze(1)
            grad_normed.index_add_(0, src, normed[dst] * weight)
            grad_normed.index_add_(0, dst, normed[src] * weight)

        grad_features = grad_normed / denominators
        projection = (grad_normed * normed).sum(dim=1, keepdim=True)
        correction = torch.where(
            norms > 0,
            normed * projection / norms.clamp_min(torch.finfo(norms.dtype).tiny),
            torch.zeros_like(grad_normed),
        )
        grad_features = grad_features - correction
        return grad_features, None, None, None


def combined_neighbor_scores(
    edge_index: Tensor,
    node_features: Tensor,
    num_nodes: Optional[int] = None,
    chunk_size: int = 500_000,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact edge-chunked port of official ``tam_combined_scores``."""
    count = int(node_features.shape[0] if num_nodes is None else num_nodes)
    if count != node_features.shape[0]:
        raise ValueError(f"num_nodes={count} does not match features={node_features.shape[0]}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    if edge_index.dtype != torch.long:
        raise TypeError(f"edge_index must be torch.long, got {edge_index.dtype}")
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    average_l2, average_l1, average_cos = _ChunkedNeighborScores.apply(
        node_features, edge_index, count, int(chunk_size)
    )
    score_l2 = _minmax_eps(average_l2)
    score_l1 = _minmax_eps(average_l1)
    score_cos = _minmax_eps(average_cos)
    loss_cos = -torch.sum(score_cos)
    return score_l2, score_l1, score_cos, loss_cos


def max_message(
    edge_index: Tensor,
    node_features: Tensor,
    chunk_size: int = 500_000,
) -> Tuple[Tensor, Tensor]:
    """Sparse official ``max_message`` including its column-degree quirk."""
    features = node_features / torch.norm(node_features, dim=-1, keepdim=True)
    count = features.shape[0]
    device = features.device
    src_all, dst_all = edge_index[0], edge_index[1]
    message = torch.zeros(count, device=device, dtype=features.dtype)
    column_degree = torch.zeros_like(message)
    grad = torch.is_grad_enabled() and features.requires_grad

    for start in range(0, src_all.numel(), chunk_size):
        end = min(start + chunk_size, src_all.numel())
        src = src_all[start:end].to(device)
        dst = dst_all[start:end].to(device)
        if grad:
            similarity = checkpoint(_edge_cosine, features, src, dst, use_reentrant=False)
        else:
            similarity = _edge_cosine(features, src, dst)
        similarity = torch.nan_to_num(similarity)
        message = message.index_add(0, src, similarity)
        column_degree.index_add_(0, dst, torch.ones_like(similarity))

    inverse = torch.where(
        column_degree > 0,
        column_degree.reciprocal(),
        torch.zeros_like(column_degree),
    )
    message = message * inverse
    return -torch.sum(message), message


# Official names retained for direct parity harnesses.
preprocess_features = row_normalize_features
feat_alignment = feature_alignment
normalize_adj = normalize_high_order_adjacency
tam_combined_scores = combined_neighbor_scores
my_GCN = TAGCN
