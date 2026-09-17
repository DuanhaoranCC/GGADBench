"""Official BRIDGE node model rewritten as a flat, device-agnostic package."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .gcn_layers import GcnLayers
from .lp import Lp
from .readout import AvgReadout


def _class_centers(
    labels: torch.Tensor, embeddings: torch.Tensor, num_classes: int
) -> torch.Tensor:
    centers = []
    for cls in range(int(num_classes)):
        selected = embeddings[labels == cls]
        if selected.numel() == 0:
            centers.append(embeddings.new_zeros(embeddings.size(1)))
        else:
            centers.append(selected.mean(dim=0))
    return torch.stack(centers, dim=0)


def _compare_loss_chunk(
    features: torch.Tensor,
    anchor_indices: torch.Tensor,
    tuples: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Compute one tuple-loss chunk; split out for gradient checkpointing."""
    anchor = features[anchor_indices]
    candidates = features[tuples]
    similarity = F.cosine_similarity(anchor.unsqueeze(1), candidates, dim=2)
    exp_similarity = torch.exp(similarity / temperature)
    numerator = exp_similarity[:, 0]
    denominator = exp_similarity[:, 1:].sum(dim=1).clamp_min(1e-12)
    return -torch.log(numerator / denominator).sum()


def compare_loss(
    features: torch.Tensor,
    tuples_cpu: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Released BRIDGE tuple objective with memory-bounded autograd chunks.

    Merely slicing the forward pass is insufficient: autograd otherwise retains
    every chunk's gathered candidate tensor until ``backward()``.  Checkpointing
    recomputes one chunk at a time during backward while preserving the same
    objective and gradients.
    """
    total = features.new_zeros(())
    count = int(tuples_cpu.size(0))
    step = max(int(chunk_size), 1)
    temperature_tensor = features.new_tensor(float(temperature))
    for start in range(0, count, step):
        tuples = tuples_cpu[start : start + step].to(
            features.device, dtype=torch.long, non_blocking=True
        )
        anchor_indices = torch.arange(start, start + tuples.size(0), device=features.device)
        total = total + checkpoint(
            _compare_loss_chunk,
            features,
            anchor_indices,
            tuples,
            temperature_tensor,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return total / max(count, 1)


def spectral_regularization_smooth(
    prompted: torch.Tensor,
    original: torch.Tensor,
    components: torch.Tensor,
    values: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Released BRIDGE spectral regularizer (paper Eq. 20)."""
    prompted_spectral = components @ prompted
    original_spectral = components @ original
    delta = (prompted_spectral[:-1].T * values[:-1] - prompted_spectral[1:].T * values[1:]).T.abs()
    delta0 = (original_spectral[:-1].T * values[:-1] - original_spectral[1:].T * values[1:]).T.abs()
    valid = (values[:-1] - values[1:]) > 1e-2
    if not torch.any(valid):
        return prompted.new_zeros(())
    return F.relu(delta - float(threshold) * delta0)[valid].mean()


class RoutingNetwork(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(float(dropout))
        self.layer2 = nn.Linear(64, 1)

    def forward(self, source_experts: torch.Tensor) -> torch.Tensor:
        score = self.layer2(self.dropout(F.relu(self.layer1(source_experts))))
        return F.softmax(score, dim=0)


class WeightedPrompt(nn.Module):
    def __init__(self, num_sources: int, input_dim: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_sources))
        self.routing = RoutingNetwork(input_dim, dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, source_masks: torch.Tensor) -> torch.Tensor:
        experts = self.weight.T * source_masks
        assignment = self.routing(experts)
        return assignment.T @ experts


class ComposedToken(nn.Module):
    def __init__(self, source_masks: torch.Tensor, combine_type: str, dropout: float):
        super().__init__()
        self.register_buffer("source_masks", source_masks.detach().clone())
        self.prompt = WeightedPrompt(source_masks.size(0), source_masks.size(1), dropout)
        self.combine_type = combine_type

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        token = self.prompt(self.source_masks)
        if self.combine_type == "add":
            return features + token
        if self.combine_type == "mul":
            return features * token
        raise ValueError(f"unknown BRIDGE combine type: {self.combine_type}")


class OpenFeaturePrompt(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.weight * features


class CombinePrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, 2))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, source_prompted: torch.Tensor, target_prompted: torch.Tensor) -> torch.Tensor:
        mixed = self.weight[0, 0] * source_prompted + self.weight[0, 1] * target_prompted
        return F.elu(mixed)


class PreFeaturePrompt(nn.Module):
    def __init__(self, source_masks: torch.Tensor, combine_type: str, dropout: float):
        super().__init__()
        input_dim = int(source_masks.size(1))
        self.source_prompt = ComposedToken(source_masks, combine_type, dropout)
        self.target_prompt = OpenFeaturePrompt(input_dim)
        self.combine = CombinePrompt()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.combine(self.source_prompt(features), self.target_prompt(features))


class DownPrompt(nn.Module):
    """BRIDGE MoE/open-prompt downstream classifier."""

    def __init__(
        self,
        source_masks: torch.Tensor,
        hidden_dim: int,
        num_classes: int,
        combine_type: str,
        dropout: float,
    ):
        super().__init__()
        self.feature_prompt = PreFeaturePrompt(source_masks, combine_type, dropout)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.centers: torch.Tensor | None = None

    def prompted_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.feature_prompt(features)

    def encode(self, features: torch.Tensor, adj, gcn: GcnLayers, spmm: Callable) -> torch.Tensor:
        prompted = self.prompted_features(features)
        return gcn(prompted, adj, spmm, lp=False).squeeze(0)

    def probabilities(
        self,
        embeddings: torch.Tensor,
        index: torch.Tensor,
        labels: torch.Tensor | None = None,
        train: bool = False,
    ) -> torch.Tensor:
        selected = embeddings[index]
        if train:
            if labels is None:
                raise ValueError("BRIDGE downstream training requires support labels")
            centers = _class_centers(labels, selected, self.num_classes)
            self.centers = centers.detach()
        else:
            if self.centers is None:
                raise RuntimeError("BRIDGE downstream class centers are not initialized")
            centers = self.centers.to(selected.device)
        similarity = F.cosine_similarity(selected.unsqueeze(1), centers.unsqueeze(0), dim=-1)
        return F.softmax(similarity, dim=1)

    @staticmethod
    def entropy(probabilities: torch.Tensor) -> torch.Tensor:
        return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)


class PrePrompt(nn.Module):
    """BRIDGE source pretraining model with dynamic source count/input width."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        negative_samples: torch.Tensor,
        num_layers: int,
        gcn_dropout: float,
        combine_type: str,
        variance_weight: float,
        num_sources: int,
        n_samples: int,
        contrast_chunk: int = 4096,
    ):
        super().__init__()
        self.lp = Lp(input_dim, hidden_dim)
        self.gcn = GcnLayers(input_dim, hidden_dim, num_layers, gcn_dropout)
        self.read = AvgReadout()
        self.combine_type = combine_type
        # Keep tuple indices on CPU and transfer only one loss chunk at a time.
        self.negative_samples = negative_samples.to(device="cpu", dtype=torch.long)
        self.masks_logits = nn.Parameter(torch.randn(num_sources, input_dim))
        self.n_samples = max(int(n_samples), 1)
        self.variance_weight = float(variance_weight)
        self.contrast_chunk = int(contrast_chunk)

    def forward(
        self, feature_list: Sequence[torch.Tensor], adj_list: Sequence, spmm: Callable
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(feature_list) != self.masks_logits.size(0):
            raise ValueError("BRIDGE source count does not match feature masks")
        mask_probability = torch.sigmoid(self.masks_logits)
        masked = [
            features * mask_probability[source].unsqueeze(0)
            for source, features in enumerate(feature_list)
        ]
        logits = torch.cat(
            [self.lp(self.gcn, features, adj, spmm) for features, adj in zip(masked, adj_list)],
            dim=0,
        )
        link_loss = compare_loss(logits, self.negative_samples, chunk_size=self.contrast_chunk)

        perturbed_losses = []
        for sample in range(self.n_samples):
            # The released code indexes one source mask per perturbation. Modulo
            # keeps the same rule valid for benchmark single-source runs.
            reference = mask_probability[sample % mask_probability.size(0)]
            noise = torch.randn(1, mask_probability.size(1), device=feature_list[0].device)
            noise = noise * (1.0 - reference).unsqueeze(0)
            noisy = [
                base + noise.expand_as(base) * (1.0 - mask_probability[source]).unsqueeze(0)
                for source, base in enumerate(masked)
            ]
            noisy_logits = torch.cat(
                [self.lp(self.gcn, features, adj, spmm) for features, adj in zip(noisy, adj_list)],
                dim=0,
            )
            perturbed_losses.append(
                compare_loss(noisy_logits, self.negative_samples, chunk_size=self.contrast_chunk)
            )
        variance_loss = torch.var(torch.stack(perturbed_losses), unbiased=len(perturbed_losses) > 1)
        total = link_loss + self.variance_weight * variance_loss
        return total, link_loss.detach(), variance_loss.detach()

    def source_masks(self) -> torch.Tensor:
        return torch.sigmoid(self.masks_logits).detach()

    def embed(self, features: torch.Tensor, adj, spmm: Callable) -> torch.Tensor:
        return self.gcn(features, adj, spmm, lp=False).squeeze(0)
