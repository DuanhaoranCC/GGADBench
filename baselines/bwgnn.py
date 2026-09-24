"""BWGNN (ICML'22) PyG/edge_index port for gadBench baselines.

Adapted from the official ``BWGNN.py`` implementation,
homogeneous version only.  The learnable layers and Beta-wavelet coefficients
are unchanged; DGL message passing is replaced with equivalent edge_index
aggregation because this benchmark block must run without DGL.
"""

import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, edge_index, feat):
        def unn_laplacian(feat, d_invsqrt, edge_index):
            src, dst = edge_index
            out = torch.zeros_like(feat)
            # Large target graphs such as t_finance have tens of millions of
            # edges.  The mathematically direct implementation
            #     msg = feat[src] * d_invsqrt[src]
            # materializes an |E| x hidden tensor and can OOM even though the
            # operation is just a sparse aggregation.  Accumulating by edge
            # chunks is exactly the same message passing, only with bounded
            # temporary memory.
            chunk_edges = 2_000_000
            for start in range(0, src.numel(), chunk_edges):
                end = min(start + chunk_edges, src.numel())
                s = src[start:end]
                d = dst[start:end]
                msg = feat[s] * d_invsqrt[s]
                out.index_add_(0, d, msg)
            return feat - out * d_invsqrt

        _src, dst = edge_index
        deg = torch.bincount(dst, minlength=feat.shape[0]).to(feat.device, feat.dtype)
        d_invsqrt = torch.pow(deg.clamp(min=1), -0.5).unsqueeze(-1)
        h = self._theta[0] * feat
        for k in range(1, self._k):
            feat = unn_laplacian(feat, d_invsqrt, edge_index)
            h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / scipy.special.beta(i + 1, d + 1 - i))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, edge_index, d=2, batch=False):
        super().__init__()
        self.edge_index = edge_index
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    @staticmethod
    def _laplacian_powers(edge_index, feat, order, chunk_edges=2_000_000):
        """Compute [H, LH, ..., L^dH] once for every Beta wavelet.

        All ``d+1`` PolyConv filters share the basis, requiring ``d`` sparse
        Laplacian passes in total.
        """
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=feat.shape[0]).to(feat.device, feat.dtype)
        d_invsqrt = torch.pow(deg.clamp(min=1), -0.5).unsqueeze(-1)
        powers = [feat]
        current = feat
        for _ in range(order):
            aggregated = torch.zeros_like(current)
            for start in range(0, src.numel(), chunk_edges):
                end = min(start + chunk_edges, src.numel())
                s = src[start:end]
                dst_chunk = dst[start:end]
                msg = current[s] * d_invsqrt[s]
                aggregated.index_add_(0, dst_chunk, msg)
            current = current - aggregated * d_invsqrt
            powers.append(current)
        return powers

    def _wavelet_outputs(self, edge_index, h):
        powers = self._laplacian_powers(edge_index, h, self.d)
        outputs = []
        for conv in self.conv:
            out = conv._theta[0] * powers[0]
            for coefficient, power in zip(conv._theta[1:], powers[1:]):
                out = out + coefficient * power
            outputs.append(out)
        return outputs

    def _project_wavelets(self, edge_index, h, chunk_edges=2_000_000):
        """Fuse the shared polynomial basis directly into ``linear3``.

        For each power k, linearity gives
        ``sum_i theta[i,k] * (L^k H) @ W_i.T``.  Computing that expression
        power-by-power retains the exact filter and projection while keeping
        only one N×hidden power in memory, which is essential on T-Social.
        """
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=h.shape[0]).to(h.device, h.dtype)
        d_invsqrt = torch.pow(deg.clamp(min=1), -0.5).unsqueeze(-1)
        if self.linear3.bias is None:
            projected = h.new_zeros((h.shape[0], self.linear3.out_features))
        else:
            projected = self.linear3.bias.unsqueeze(0).expand(h.shape[0], -1).clone()

        block_width = h.shape[1]
        current = h
        for power_index in range(self.d + 1):
            effective_weight = None
            for wavelet_index, conv in enumerate(self.conv):
                block = self.linear3.weight[
                    :, wavelet_index * block_width : (wavelet_index + 1) * block_width
                ]
                weighted = conv._theta[power_index] * block
                effective_weight = (
                    weighted if effective_weight is None else effective_weight + weighted
                )
            projected = projected + current.matmul(effective_weight.t())
            if power_index == self.d:
                break
            aggregated = torch.zeros_like(current)
            for start in range(0, src.numel(), chunk_edges):
                end = min(start + chunk_edges, src.numel())
                s = src[start:end]
                dst_chunk = dst[start:end]
                aggregated.index_add_(0, dst_chunk, current[s] * d_invsqrt[s])
            current = current - aggregated * d_invsqrt
        return projected

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self._project_wavelets(self.edge_index, h)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, edge_index, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self._project_wavelets(edge_index, h)
        h = self.act(h)
        h = self.linear4(h)
        return h
