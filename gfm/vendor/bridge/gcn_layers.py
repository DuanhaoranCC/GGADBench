"""Residual BRIDGE GCN stack adapted from the official node implementation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .gcn import GCN


class GcnLayers(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_layers = int(num_layers)
        self.convs = nn.ModuleList(
            [
                GCN(input_dim if layer == 0 else hidden_dim, hidden_dim)
                for layer in range(self.num_layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, seq: torch.Tensor, adj, spmm: Callable, lp: bool = False) -> torch.Tensor:
        graph_output = seq.squeeze(0) if seq.dim() == 3 else seq
        for layer, conv in enumerate(self.convs):
            previous = graph_output
            graph_output = conv(previous, adj, spmm)
            if layer:
                graph_output.add_(previous)
            if lp:
                graph_output = self.bns[layer](graph_output)
                graph_output = self.dropout(graph_output)
        return graph_output.unsqueeze(0)
