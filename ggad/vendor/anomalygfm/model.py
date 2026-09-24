"""AnomalyGFM graph convolution and residual prototype model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import glorot


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == "prelu" else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.sparse.mm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class AvgReadout(nn.Module):
    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        return torch.sum(torch.mul(seq, sim), 1)


class SimplePrompt(nn.Module):
    def __init__(self, input_size):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
        self.a = nn.Linear(input_size, input_size)
        self.reset_parameters()
        self.act = nn.ReLU()

    def reset_parameters(self):
        glorot(self.global_emb)
        self.a.reset_parameters()

    def forward(self, x):
        return x + self.a(x) + self.global_emb


class Model(nn.Module):

    def __init__(self, n_in_1, n_in_2, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.fc_map = nn.Linear(n_in_1, n_in_2, bias=False)
        self.gcn1 = GCN(n_in_2, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, 1, bias=False)
        self.fc2 = nn.Linear(n_h, 1, bias=False)
        self.fc_normal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.fc_abnormal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.prompt = SimplePrompt(300)
        self.act = nn.ReLU()
        self.read = {
            "max": MaxReadout,
            "min": MinReadout,
            "avg": AvgReadout,
            "weighted_sum": WSReadout,
        }[readout]()

    def forward(self, seq1, adj, adj_resid, normal_prompt, abnormal_prompt):

        h_1 = self.gcn1(seq1, adj, sparse=True)
        emb = self.gcn2(h_1, adj, sparse=True)

        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))

        # residual feature: emb_neighbors = (D^-1 A_noloop) @ emb
        emb_neighbors = torch.sparse.mm(adj_resid, torch.squeeze(emb, 0)).unsqueeze(0)
        emb_residual = emb - emb_neighbors

        logit = self.fc1(emb)
        logit_residual = self.fc2(emb_residual)
        return (
            logit,
            logit_residual,
            emb,
            emb_residual,
            normal_prompt,
            abnormal_prompt,
            emb_neighbors,
        )
