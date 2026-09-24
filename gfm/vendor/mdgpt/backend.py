"""Memory-aware, exact sparse propagation for MDGPT."""
import numpy as np
import torch


def csr_tensor(matrix, device):
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64, copy=False)).to(device),
        torch.from_numpy(matrix.indices.astype(np.int64, copy=False)).to(device),
        torch.from_numpy(matrix.data).to(device), size=matrix.shape,
    )


class _FixedSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, matrix, transpose):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, x)

    @staticmethod
    def backward(ctx, gradient):
        return torch.sparse.mm(ctx.transpose, gradient.contiguous()), None, None


class SparseOperator:
    """Fixed adjacency: differentiate node states, never sparse edge weights."""
    def __init__(self, adjacency, device):
        self.matrix = csr_tensor(adjacency, device)
        self.transpose = csr_tensor(adjacency.transpose().tocsr(), device)

    def apply(self, x):
        return _FixedSpMM.apply(x, self.matrix, self.transpose)


def estimated_bytes(adjacency, hp, training=True):
    n, edges = adjacency.shape[0], adjacency.nnz
    # Two CSR operators, dual-branch tapes and workspace; no E-by-hidden tensor.
    sparse = 2 * (edges * 12 + (n + 1) * 8)
    hidden = n * hp["hidden_dim"] * 4
    states = (6 * hp["num_layers"] + 10) if training else 8
    return sparse + hidden * states + n * hp["feature_dim"] * 16 + 256 * 1024**2


def use_full_graph(adjacency, device, hp, training=True):
    requested = hp["target_backend"]
    if requested == "stream":
        return False
    if requested == "full":
        return True
    if str(device).startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        # Inactive allocator blocks can be reused by this process.
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return estimated_bytes(adjacency, hp, training) <= (free + reusable) * hp["gpu_memory_fraction"]
    return (adjacency.shape[0] < hp["stream_node_threshold"]
            and adjacency.nnz < hp["stream_edge_threshold"])


def aggregate_rows(adjacency, values, start, end, device, edge_chunk):
    """Bounded GPU sparse aggregation with host-resident hidden states."""
    if not str(device).startswith("cuda"):
        return np.asarray(adjacency[start:end].dot(values), dtype=np.float32)
    out = torch.zeros((end - start, values.shape[1]), device=device, dtype=torch.float32)
    first, last = int(adjacency.indptr[start]), int(adjacency.indptr[end])
    for offset in range(first, last, edge_chunk):
        stop = min(offset + edge_chunk, last)
        positions = np.arange(offset, stop)
        rows = np.searchsorted(adjacency.indptr, positions, side="right") - 1 - start
        columns, inverse = np.unique(adjacency.indices[offset:stop], return_inverse=True)
        index = torch.from_numpy(np.stack((rows, inverse))).to(device=device, dtype=torch.long)
        weights = torch.from_numpy(adjacency.data[offset:stop]).to(device)
        operator = torch.sparse_coo_tensor(index, weights, (end-start, len(columns)))
        features = torch.from_numpy(np.array(values[columns], copy=True)).to(device)
        out.add_(torch.sparse.mm(operator, features))
    return out
