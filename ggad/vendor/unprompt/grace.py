"""GRACE contrastive pretraining used by UNPrompt."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ggad.vendor.unprompt.graph import sparse_mx_to_torch_sparse_tensor


def drop_feature(x, drop_prob):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(adj_withloop_won, drop_prob):
    adj = sp.coo_matrix(adj_withloop_won)
    num_edges = adj.nnz
    size = adj.shape
    edge_delete = np.random.choice(num_edges, int(drop_prob * num_edges), replace=False)
    not_equal = adj.row[edge_delete] != adj.col[edge_delete]
    edge_delete = edge_delete[not_equal]
    keep = np.ones(num_edges, dtype=bool)
    keep[edge_delete] = False
    masked = sp.coo_matrix((adj.data[keep], (adj.row[keep], adj.col[keep])), shape=size)
    return sparse_mx_to_torch_sparse_tensor(masked)


class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super().__init__()
        self.model = model
        self.tau = tau
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, features, adj):
        z = self.model(features, adj)
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        return torch.mm(F.normalize(z1), F.normalize(z2).t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl = f(self.sim(z1, z1))
        betw = f(self.sim(z1, z2))
        return -torch.log(betw.diag() / (refl.sum(1) + betw.sum(1) - refl.diag()))

    def loss(self, h1, h2):

        if h1.size(0) > 5000:
            idx = torch.randperm(h1.size(0), device=h1.device)[:5000]
            h1, h2 = h1[idx], h2[idx]
        return ((self.semi_loss(h1, h2) + self.semi_loss(h2, h1)) * 0.5).mean()


def traingrace(model, feats, adj_clean, raw_withloop, hp, device, epochs):
    grace = ModelGrace(model, hp["emb_dim"], 2 * hp["emb_dim"], tau=hp["tau"]).to(device)
    opt = torch.optim.Adam(
        grace.parameters(), lr=hp["grace_lr"], weight_decay=hp["grace_weight_decay"]
    )
    edge_drop, feat_drop = hp["edge_drop_prob"], hp["feat_drop_prob"]
    for _ in range(epochs):
        for i in range(len(feats)):
            grace.train()
            opt.zero_grad()
            f, a = feats[i], adj_clean[i]
            f_aug = drop_feature(f, feat_drop)
            a_aug = mask_edge(raw_withloop[i], edge_drop).to(device) if edge_drop > 0 else a
            z1 = grace(f, a)
            z2 = grace(f_aug, a_aug)
            loss = grace.loss(z1, z2)
            loss.backward()
            opt.step()
