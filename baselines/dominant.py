"""Official DOMINANT graph autoencoder, ported to sparse PyTorch tensors.

The module follows the official DOMINANT ``layers.py`` and ``model.py``. The
only model-level extension is exposing the structure-decoder embedding so the
runner can evaluate ``sigmoid(Z @ Z.T)`` in row blocks instead of allocating a
full dense N-by-N matrix.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import EdgeList, exact_edge_spmm


class GraphConvolution(nn.Module):
    """The official DOMINANT GCN layer with sparse-adjacency support."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Exact initialization used by DOMINATE/layers.py.
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.mm(inputs, self.weight)
        if isinstance(adj, EdgeList):
            output = exact_edge_spmm(
                adj,
                support,
                chunk=int(getattr(adj, "dominant_edge_chunk", 500_000)),
            )
        elif adj.layout == torch.strided:
            output = torch.mm(adj, support)
        else:
            output = torch.sparse.mm(adj, support)
        return output if self.bias is None else output + self.bias


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.relu(self.gc2(x, adj))


class AttributeDecoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.relu(self.gc2(x, adj))


class StructureDecoder(nn.Module):
    def __init__(self, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def embedding(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return F.dropout(x, self.dropout, training=self.training)

    def forward(self, x, adj):
        z = self.embedding(x, adj)
        return z @ z.T


class Dominant(nn.Module):
    """DOMINANT as released in the official repository."""

    def __init__(self, feat_size, hidden_size, dropout):
        super().__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)

    def decoded_embeddings(self, x, adj):
        encoded = self.shared_encoder(x, adj)
        # Keep the official decoder call order because each decoder consumes a
        # dropout mask during training.
        x_hat = self.attr_decoder(encoded, adj)
        structure_z = self.struct_decoder.embedding(encoded, adj)
        return structure_z, x_hat

    def forward(self, x, adj):
        structure_z, x_hat = self.decoded_embeddings(x, adj)
        return structure_z @ structure_z.T, x_hat
