"""BRIDGE graph readout adapted from the official implementation."""

from __future__ import annotations

import torch
from torch import nn


class AvgReadout(nn.Module):
    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(seq, dim=1)
        weight = mask.unsqueeze(-1)
        return torch.sum(seq * weight, dim=1) / torch.sum(weight).clamp_min(1e-12)
