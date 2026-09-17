"""BRIDGE GCN layer adapted from the official node implementation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class GCN(nn.Module):
    """Linear(no bias) -> normalized adjacency propagation -> bias -> PReLU."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor, adj, spmm: Callable) -> torch.Tensor:
        transformed = self.fc(x)
        out = spmm(adj, transformed)
        del transformed
        if self.bias is not None:
            out.add_(self.bias)
        return self.act(out)
