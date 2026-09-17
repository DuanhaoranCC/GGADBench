"""Exact, disk-backed inference and prompt gradients for frozen MDGPT GCNs.

The graph operator is supplied by the caller, including its normalization and
self loops. Every row and edge participates. Only a bounded block of node
features is transferred to the encoder's device; hidden states live in temporary
NumPy memory maps. This backend implements ``PReLU(Linear(P @ h))`` and a single
multiplicative input prompt, not encoder training or graph sampling.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


class _Workspace:
    """Own memory maps and close Windows file handles before directory removal."""

    def __init__(self, work_dir: Path):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="encode_", dir=str(work_dir))
        self.arrays = {}
        self.closed = False

    def matrix(self, name, shape, dtype=np.float32):
        if name in self.arrays:
            raise RuntimeError("Duplicate temporary matrix: " + name)
        path = Path(self.temp.name) / (name + ".bin")
        value = np.memmap(str(path), mode="w+", dtype=dtype, shape=shape)
        self.arrays[name] = value
        return value

    def remove(self, name):
        value = self.arrays.pop(name, None)
        if value is not None:
            path = Path(value.filename)
            value._mmap.close()
            path.unlink(missing_ok=True)

    def close(self):
        if not self.closed:
            for name in list(self.arrays):
                self.remove(name)
            self.temp.cleanup()
            self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Cleanup during interpreter shutdown must not hide an earlier error.
            pass


def _validate(prompt, x, adjacency, layers, activations, node_chunk):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.dtype != np.float32:
        raise TypeError("x must be a float32 NumPy matrix")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must have positive node and feature dimensions")
    if not sp.isspmatrix_csr(adjacency) or adjacency.shape != (len(x), len(x)):
        raise ValueError("adjacency must be a square CSR matrix matching x")
    if adjacency.dtype != np.float32:
        raise TypeError("adjacency must use float32 normalized weights")
    if prompt.dtype != torch.float32 or tuple(prompt.shape) not in {(x.shape[1],), (1, x.shape[1])}:
        raise ValueError("prompt must be float32 with shape (D,) or (1, D)")
    if int(node_chunk) <= 0:
        raise ValueError("node_chunk must be positive")
    if not layers or len(layers) != len(activations):
        raise ValueError("one PReLU is required for every nonempty GCN layer")
    in_dim = x.shape[1]
    for layer, act in zip(layers, activations):
        if not isinstance(layer, nn.Linear) or not isinstance(act, nn.PReLU):
            raise TypeError("the backend supports nn.Linear and nn.PReLU only")
        if layer.in_features != in_dim or act.num_parameters not in (1, layer.out_features):
            raise ValueError("encoder or PReLU dimensions do not match")
        for parameter in list(layer.parameters()) + list(act.parameters()):
            if parameter.requires_grad:
                raise ValueError("all encoder parameters must be frozen")
            if parameter.device != prompt.device or parameter.dtype != torch.float32:
                raise ValueError("encoder parameters must match prompt device and float32 dtype")
        in_dim = layer.out_features


def _blocks(n, chunk):
    for start in range(0, n, int(chunk)):
        yield start, min(start + int(chunk), n)


def _forward(work, prompt, x, adjacency, layers, activations, node_chunk, keep_signs):
    """Write successive exact full-graph activations, keeping two at most."""
    previous_name = "input"
    previous = work.matrix(previous_name, x.shape)
    prompt_cpu = prompt.detach().cpu().numpy().reshape(1, -1)
    for start, end in _blocks(len(x), node_chunk):
        previous[start:end] = x[start:end] * prompt_cpu
    for index, (layer, act) in enumerate(zip(layers, activations)):
        name = "hidden_" + str(index)
        current = work.matrix(name, (len(x), layer.out_features))
        signs = (
            work.matrix("positive_" + str(index), current.shape, np.bool_) if keep_signs else None
        )
        for start, end in _blocks(len(x), node_chunk):
            # SciPy traverses all nonzeros in these complete CSR rows. It never
            # materializes an N-by-N dense matrix or an E-by-hidden tensor.
            aggregate = np.asarray(adjacency[start:end].dot(previous), dtype=np.float32)
            aggregate_device = torch.from_numpy(aggregate).to(prompt.device)
            preactivation = F.linear(aggregate_device, layer.weight, layer.bias)
            if signs is not None:
                signs[start:end] = (preactivation > 0).cpu().numpy()
            current[start:end] = F.prelu(preactivation, act.weight).cpu().numpy()
        work.remove(previous_name)
        previous_name, previous = name, current
    return previous_name, previous


class _FrozenEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prompt, x, adjacency, layers, activations, output_nodes, node_chunk, work_dir):
        work = _Workspace(work_dir)
        try:
            name, embedding = _forward(
                work, prompt, x, adjacency, layers, activations, node_chunk, keep_signs=True
            )
            # Fancy indexing makes an owning array before the map is closed.
            selected = np.array(embedding[output_nodes], dtype=np.float32, copy=True)
            out = torch.from_numpy(selected).to(prompt.device)
            work.remove(name)
        except BaseException:
            work.close()
            raise
        ctx.work = work
        ctx.x = x
        ctx.adjacency = adjacency
        ctx.nodes = output_nodes
        ctx.node_chunk = int(node_chunk)
        ctx.prompt_shape = tuple(prompt.shape)
        ctx.device = prompt.device
        ctx.save_for_backward(
            *[layer.weight for layer in layers], *[act.weight for act in activations]
        )
        ctx.depth = len(layers)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weights = ctx.saved_tensors[: ctx.depth]
        slopes = ctx.saved_tensors[ctx.depth :]
        work = ctx.work
        n = len(ctx.x)
        grad_name = "backward_initial"
        gradient = work.matrix(grad_name, (n, weights[-1].shape[0]))
        scratch_names = {grad_name}
        try:
            # Duplicate requested nodes contribute repeatedly to the same node.
            # New files are zero-filled by the OS; explicit block initialization
            # also documents this requirement independently of file semantics.
            for start, end in _blocks(n, ctx.node_chunk):
                gradient[start:end] = 0
            np.add.at(gradient, ctx.nodes, grad_out.detach().cpu().numpy())
            transpose = ctx.adjacency.transpose().tocsr()
            prompt_grad = torch.zeros(
                ctx.prompt_shape, device=ctx.device, dtype=torch.float32
            ).reshape(-1)
            for index in reversed(range(ctx.depth)):
                pre_name = "backward_linear_" + str(index)
                scratch_names.add(pre_name)
                pre = work.matrix(pre_name, (n, weights[index].shape[1]))
                positive = work.arrays["positive_" + str(index)]
                for start, end in _blocks(n, ctx.node_chunk):
                    # Copy protects the mapping from CPU torch views surviving
                    # its lifetime and avoids modifying the upstream gradient.
                    block = torch.from_numpy(np.array(gradient[start:end], copy=True)).to(
                        ctx.device
                    )
                    mask = torch.from_numpy(np.array(positive[start:end], copy=True)).to(ctx.device)
                    derivative = torch.where(mask, torch.ones((), device=ctx.device), slopes[index])
                    pre[start:end] = ((block * derivative) @ weights[index]).cpu().numpy()
                work.remove(grad_name)
                scratch_names.discard(grad_name)
                if index:
                    next_name = "backward_hidden_" + str(index)
                    scratch_names.add(next_name)
                    next_gradient = work.matrix(next_name, pre.shape)
                for start, end in _blocks(n, ctx.node_chunk):
                    block = np.asarray(transpose[start:end].dot(pre), dtype=np.float32)
                    if index:
                        next_gradient[start:end] = block
                    else:
                        product = torch.from_numpy(block).to(ctx.device)
                        feature = torch.from_numpy(np.array(ctx.x[start:end], copy=True)).to(
                            ctx.device
                        )
                        prompt_grad.add_((product * feature).sum(dim=0))
                work.remove(pre_name)
                scratch_names.discard(pre_name)
                if index:
                    grad_name, gradient = next_name, next_gradient
            return prompt_grad.reshape(ctx.prompt_shape), None, None, None, None, None, None, None
        finally:
            for name in scratch_names:
                work.remove(name)
        # Sign maps are owned by the autograd context. Keeping them until that
        # context is released supports retain_graph=True without leaking files
        # between optimizer iterations once the caller releases its loss/output.


def frozen_encode(
    prompt: torch.Tensor,
    x: np.ndarray,
    adjacency: sp.csr_matrix,
    layers: Sequence[nn.Linear],
    activations: Sequence[nn.PReLU],
    *,
    output_nodes: np.ndarray,
    node_chunk: int = 4096,
    work_dir: Path,
) -> torch.Tensor:
    """Encode selected nodes with exact gradients only for the input prompt.

    ``output_nodes`` is required: full-graph device output defeats the memory
    bound. Its order and repeated indices are preserved. The entire normalized
    graph still participates in every message-passing layer. The caller must
    not mutate x, adjacency, or the frozen encoder before backward completes.
    Release the returned output/loss after each optimizer step so the autograd
    context can close its disk-backed sign cache.
    """
    _validate(prompt, x, adjacency, layers, activations, node_chunk)
    nodes = np.asarray(output_nodes)
    if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("output_nodes must be a one-dimensional integer array")
    nodes = np.array(nodes, dtype=np.int64, copy=True)
    if len(nodes) and (nodes.min() < 0 or nodes.max() >= len(x)):
        raise IndexError("output_nodes contains an out-of-range node")
    return _FrozenEncode.apply(
        prompt, x, adjacency, layers, activations, nodes, int(node_chunk), Path(work_dir)
    )


@contextmanager
def frozen_embeddings(
    prompt: torch.Tensor,
    x: np.ndarray,
    adjacency: sp.csr_matrix,
    layers: Sequence[nn.Linear],
    activations: Sequence[nn.PReLU],
    *,
    node_chunk: int = 4096,
    work_dir: Path,
) -> Iterator[np.memmap]:
    """Yield full-graph float32 host embeddings for chunked inference.

    The yielded memory map is valid only inside the ``with`` block. This path
    never builds an autograd graph; to optimize prompts use ``frozen_encode``.
    All maps and temporary files are closed and deleted on normal/error exit.
    """
    _validate(prompt, x, adjacency, layers, activations, node_chunk)
    work = _Workspace(Path(work_dir))
    try:
        with torch.no_grad():
            _, embedding = _forward(
                work, prompt, x, adjacency, layers, activations, node_chunk, keep_signs=False
            )
        yield embedding
    finally:
        work.close()
