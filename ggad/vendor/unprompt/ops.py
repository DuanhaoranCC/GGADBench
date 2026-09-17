"""Completion losses and anomaly scoring for UNPrompt."""

import numpy as np
import torch


def completionloss(feature1, feature2, ano_label):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    diff = -torch.sum(feature1 * feature2, dim=1)
    modified = torch.where(ano_label == 0, diff, -diff)
    loss = torch.mean(modified)
    return loss


def completionsim(feature1, feature2):
    feature1 = feature1 / torch.norm(feature1, dim=-1, keepdim=True)
    feature2 = feature2 / torch.norm(feature2, dim=-1, keepdim=True)
    dist = torch.sum(feature1 * feature2, dim=1)
    dist = dist.detach().cpu().numpy()
    return dist


def normalize_score(ano_score):
    return (ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score))
