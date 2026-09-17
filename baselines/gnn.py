"""Standard supervised MLP/GCN/GAT baselines for ggad.

These are plain DGL node-classification backbones used as classic supervised
baselines.  The experiment protocol is implemented in
`baselines/runner_supervised_gnn.py`.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


def _activation_module(name):
    name = str(name).lower()
    choices = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }
    if name not in choices:
        raise ValueError(f"unsupported activation: {name!r}")
    return choices[name]()


class MLP(nn.Module):
    """Feature-only node classifier with no access to graph structure."""

    def __init__(self, in_dim, hid_dim=64, out_dim=2, num_layers=2, activation="relu", dropout=0.0):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("MLP num_layers must be at least 1")
        blocks = []
        width = int(in_dim)
        for _ in range(int(num_layers)):
            blocks.append(nn.Linear(width, int(hid_dim)))
            blocks.append(_activation_module(activation))
            blocks.append(nn.Dropout(float(dropout)))
            width = int(hid_dim)
        self.hidden = nn.Sequential(*blocks)
        self.classifier = nn.Linear(width, int(out_dim))

    def forward(self, x, edge_index=None):
        del edge_index
        return self.classifier(self.hidden(x))


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=2, num_layers=2, activation="relu", dropout=0.0):
        super().__init__()
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("GCN num_layers must be at least 1")
        widths = (
            [int(in_dim), int(out_dim)]
            if num_layers == 1
            else [int(in_dim)] + [int(hid_dim)] * (num_layers - 1) + [int(out_dim)]
        )
        self.convs = nn.ModuleList(
            [GCNConv(widths[index], widths[index + 1]) for index in range(num_layers)]
        )
        self.activation = _activation_module(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for index, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if index + 1 < len(self.convs):
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim=64,
        out_dim=2,
        num_layers=2,
        activation="elu",
        heads=4,
        out_heads=1,
        dropout=0.0,
    ):
        super().__init__()
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("GAT num_layers must be at least 1")
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GATConv(
                    int(in_dim),
                    int(out_dim),
                    heads=int(out_heads),
                    concat=False,
                    dropout=float(dropout),
                )
            )
        else:
            width = int(in_dim)
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(
                        width,
                        int(hid_dim),
                        heads=int(heads),
                        concat=True,
                        dropout=float(dropout),
                    )
                )
                width = int(hid_dim) * int(heads)
            self.convs.append(
                GATConv(
                    width,
                    int(out_dim),
                    heads=int(out_heads),
                    concat=False,
                    dropout=float(dropout),
                )
            )
        self.activation = _activation_module(activation)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x, edge_index):
        h = x
        for index, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if index + 1 < len(self.convs):
                h = self.activation(h)
                h = self.dropout(h)
        return h
