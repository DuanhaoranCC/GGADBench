"""BRIDGE link-prediction prompt wrapper."""

from __future__ import annotations

import torch
from torch import nn


class Lp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Retained for fidelity with the released implementation.
        self.prompt = nn.Parameter(torch.empty(1, hidden_dim))
        self.activation = nn.ELU()
        nn.init.xavier_uniform_(self.prompt)

    def forward(self, gcn, seq: torch.Tensor, adj, spmm) -> torch.Tensor:
        return self.activation(gcn(seq, adj, spmm, lp=True).squeeze(0))
