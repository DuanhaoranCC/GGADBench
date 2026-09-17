"""Official GAD-MoRE model port for the source-domain benchmark.

The implementation follows ``GAD-MoRE-main/model.py``. The only additions are
Python 3.8-compatible annotations, explicit memory-device migration, and a
chunked inference helper that avoids materializing all node embeddings at once.
"""

from typing import Dict, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class NodeMemoryEntry:
    def __init__(self, node_embedding: torch.Tensor, performance_score: float):
        self.node_embedding = node_embedding.clone().detach()
        self.performance_score = performance_score
        self.access_count = 1
        self.last_access_epoch = 0

    def compute_similarity(self, query_embedding: torch.Tensor) -> float:
        return F.cosine_similarity(
            self.node_embedding.unsqueeze(0), query_embedding.unsqueeze(0)
        ).item()


class ExpertMemoryBank:
    def __init__(self, memory_size: int, embedding_dim: int, coldstart_epochs: int = 5):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.coldstart_epochs = coldstart_epochs
        self.quality_warmup_epochs = 10
        self.current_epoch = 0
        self.memories: List[NodeMemoryEntry] = []
        self.behavior_vector = None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def add_memory(self, node_embedding: torch.Tensor, performance_score: float):
        if self.current_epoch < self.coldstart_epochs:
            return
        if self.current_epoch < self.coldstart_epochs + self.quality_warmup_epochs:
            progress = (self.current_epoch - self.coldstart_epochs) / self.quality_warmup_epochs
            threshold = 0.3 + 0.4 * max(0.0, min(1.0, progress))
        else:
            threshold = 0.7
        if performance_score < threshold:
            return

        memory = NodeMemoryEntry(node_embedding, performance_score)
        if len(self.memories) < self.memory_size:
            self.memories.append(memory)
        else:
            worst_idx = min(
                range(len(self.memories)),
                key=lambda i: self.memories[i].performance_score,
            )
            if performance_score - self.memories[worst_idx].performance_score > 0.1:
                self.memories[worst_idx] = memory
        self._update_behavior_profile()

    def _update_behavior_profile(self):
        if not self.memories:
            self.behavior_vector = torch.zeros(self.embedding_dim)
            return
        weights = F.softmax(torch.tensor([m.performance_score for m in self.memories]), dim=0)
        embeddings = torch.stack([m.node_embedding for m in self.memories])
        weights = weights.to(embeddings.device)
        self.behavior_vector = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)

    def is_empty(self) -> bool:
        return not self.memories

    def move(self, device):
        for memory in self.memories:
            memory.node_embedding = memory.node_embedding.to(device)
        if self.behavior_vector is not None:
            self.behavior_vector = self.behavior_vector.to(device)

    def get_stats(self) -> dict:
        if not self.memories:
            return {
                "memory_count": 0,
                "avg_score": 0.0,
                "coldstart_remaining": max(0, self.coldstart_epochs - self.current_epoch),
            }
        return {
            "memory_count": len(self.memories),
            "avg_score": np.mean([m.performance_score for m in self.memories]),
            "coldstart_remaining": max(0, self.coldstart_epochs - self.current_epoch),
        }


class ExpertMemoryRouter(nn.Module):
    def __init__(
        self, embedding_dim: int, num_experts: int, experts: nn.ModuleList, memory_size: int = 64
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.memory_size = memory_size
        self.experts = experts
        self.warmup_epochs = 15
        self.current_epoch = 0
        self.exploration_ratio = 1.0
        self.expert_memories = [
            ExpertMemoryBank(memory_size, embedding_dim, self.warmup_epochs)
            for _ in range(num_experts)
        ]
        self.output_proj = nn.Linear(num_experts, num_experts)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if epoch < 5:
            self.exploration_ratio = 1.0
        elif epoch < self.warmup_epochs:
            progress = (epoch - 5) / (self.warmup_epochs - 5)
            self.exploration_ratio = 0.8 * (1.0 - progress) + 0.2
        elif epoch < self.warmup_epochs + 10:
            progress = (epoch - self.warmup_epochs) / 10
            self.exploration_ratio = 0.2 * (1.0 - progress) + 0.05
        else:
            self.exploration_ratio = 0.05
        for memory_bank in self.expert_memories:
            memory_bank.set_epoch(epoch)

    def get_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        fullness = sum(len(mem.memories) for mem in self.expert_memories)
        fullness /= self.num_experts * self.memory_size
        exploration = min(self.exploration_ratio + (1.0 - fullness) * 0.3, 0.9)
        if self.training and torch.rand(1).item() < exploration:
            logits = torch.ones(node_embeds.size(0), self.num_experts, device=node_embeds.device)
            return logits + torch.randn_like(logits) * 0.1
        return self._get_memory_based_logits(node_embeds)

    def _get_memory_based_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(node_embeds.size(0), self.num_experts, device=node_embeds.device)
        for expert_id in range(self.num_experts):
            bank = self.expert_memories[expert_id]
            if bank.is_empty():
                continue
            manifold = self.experts[expert_id].manifold
            with torch.no_grad():
                memory = torch.stack([m.node_embedding for m in bank.memories]).to(
                    node_embeds.device
                )
                memory = manifold.expmap0(memory)
                blocks = []
                for start in range(0, node_embeds.size(0), 512):
                    query = manifold.expmap0(node_embeds[start : start + 512].detach())
                    distances = manifold.dist(query.unsqueeze(1), memory.unsqueeze(0))
                    blocks.append(distances.min(dim=1).values)
                logits[:, expert_id] = -torch.cat(blocks)
        return self.output_proj(logits)

    def update_memory(
        self,
        node_embeds: torch.Tensor,
        expert_errors: torch.Tensor,
        expert_assignments: torch.Tensor,
    ):
        if not self.training or node_embeds.size(0) < 10:
            return
        with torch.no_grad():
            for expert_id in range(self.num_experts):
                mask = expert_assignments[:, expert_id] > 0.1
                if mask.sum() == 0:
                    continue
                embeddings = node_embeds[mask]
                errors = expert_errors[mask, expert_id]
                scores = 1.0 - errors / (errors.max() + 1e-6)
                quality_mask = scores >= 0.7
                if quality_mask.sum() == 0:
                    quality_mask = scores >= torch.quantile(scores, 0.7)
                if quality_mask.sum() == 0:
                    continue
                embeddings = embeddings[quality_mask]
                scores = scores[quality_mask]
                order = torch.argsort(scores, descending=True)
                count = min(3, len(embeddings), self.memory_size // 4)
                for i in range(count):
                    idx = order[i]
                    self.expert_memories[expert_id].add_memory(embeddings[idx], scores[idx].item())

    def move_memory(self, device):
        for bank in self.expert_memories:
            bank.move(device)

    def get_memory_stats(self) -> dict:
        return {
            "memory_utilization": [
                len(mem.memories) / self.memory_size for mem in self.expert_memories
            ],
            "avg_scores": [mem.get_stats()["avg_score"] for mem in self.expert_memories],
            "coldstart_remaining": [
                mem.get_stats()["coldstart_remaining"] for mem in self.expert_memories
            ],
            "exploration_ratio": self.exploration_ratio,
            "current_epoch": self.current_epoch,
        }


class ExpertMemoryRouterWrapper(nn.Module):
    def __init__(
        self, embedding_dim: int, num_experts: int, experts: nn.ModuleList, memory_size: int = 32
    ):
        super().__init__()
        self.expert_router = ExpertMemoryRouter(embedding_dim, num_experts, experts, memory_size)

    def get_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        return self.expert_router.get_logits(node_embeds)

    def update_memory(
        self,
        node_embeds: torch.Tensor,
        expert_errors: torch.Tensor,
        expert_assignments: Optional[torch.Tensor] = None,
    ):
        if expert_assignments is None:
            with torch.no_grad():
                expert_assignments = F.softmax(1.0 / (expert_errors + 1e-6), dim=-1)
        self.expert_router.update_memory(node_embeds, expert_errors, expert_assignments)

    def set_epoch(self, epoch: int):
        self.expert_router.set_epoch(epoch)

    def move_memory(self, device):
        self.expert_router.move_memory(device)

    def get_memory_stats(self) -> dict:
        return self.expert_router.get_memory_stats()


class KappaLinear(nn.Module):
    def __init__(
        self, manifold, in_dim: int, out_dim: int, dropout: float = 0.0, use_bias: bool = True
    ):
        super().__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        result = self.manifold.mobius_matvec(weight, x)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            result = self.manifold.mobius_add(result, self.manifold.expmap0(bias))
        return result


class RiemannianExpert(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        initial_curvature: float = 0.0,
    ):
        super().__init__()
        self.manifold = geoopt.Stereographic(k=initial_curvature, learnable=True)
        self.linear1 = KappaLinear(self.manifold, in_dim, hidden_dim, dropout)
        self.linear2 = KappaLinear(self.manifold, hidden_dim, out_dim, dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x)
        hidden = self.linear1(x)
        hidden = self.act(self.manifold.logmap0(hidden))
        return self.linear2(self.manifold.expmap0(hidden))


class AnomalyScorerMoE(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        original_feature_dim: int,
        num_experts: int = 5,
        expert_hidden_dim: int = 128,
        top_k: int = 2,
        init_curvs=None,
        gate_temperature: float = 1.0,
        gate_noise_type: str = "none",
        gate_noise_std: float = 0.0,
        memory_size: int = 32,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_temperature = gate_temperature
        self.gate_noise_type = gate_noise_type
        self.gate_noise_std = gate_noise_std
        if init_curvs is None:
            init_curvs = (
                [0.0]
                + [-0.5 * (i + 1) for i in range(num_experts // 2)]
                + [0.5 * (i + 1) for i in range(num_experts - num_experts // 2 - 1)]
            )
        self.experts = nn.ModuleList(
            [
                RiemannianExpert(
                    embedding_dim,
                    expert_hidden_dim,
                    embedding_dim,
                    dropout=0.1,
                    initial_curvature=curvature,
                )
                for curvature in init_curvs
            ]
        )
        self.router = ExpertMemoryRouterWrapper(
            embedding_dim, num_experts, self.experts, memory_size
        )
        self.feature_decoder = nn.Linear(embedding_dim, original_feature_dim)

    def _apply_noise(self, logits):
        if not self.training:
            return logits
        if self.gate_noise_type == "gaussian" and self.gate_noise_std > 0:
            return logits + self.gate_noise_std * torch.randn_like(logits)
        if self.gate_noise_type == "gumbel":
            uniform = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            return logits - torch.log(-torch.log(uniform))
        return logits

    def moe_reconstruction(self, node_embeds: torch.Tensor, update_memory: bool = True):
        gate_logits = self.router.get_logits(node_embeds)
        noisy_logits = self._apply_noise(gate_logits)
        gate_weights = F.softmax(noisy_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(noisy_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals / max(self.gate_temperature, 1e-6), dim=-1)

        unique_experts = torch.unique(topk_idx)
        # The released implementation evaluates every active expert on every
        # node, then gathers only the top-k outputs. That retains an
        # N x active_experts x embedding_dim autograd graph and exhausts GPU
        # memory on source graphs such as t_finance. Expert MLPs are node-wise,
        # so evaluating only rows actually routed to each expert is exactly the
        # same computation. Keep the original expert/global-chunk call order so
        # weight-dropout RNG consumption is preserved as well.
        routed_outputs = node_embeds.new_zeros(
            (node_embeds.size(0), self.top_k, self.embedding_dim)
        )
        chunk_size = 256 if node_embeds.size(0) > 512 else node_embeds.size(0)
        for expert_id in unique_experts.tolist():
            expert = self.experts[expert_id]
            for start in range(0, node_embeds.size(0), chunk_size):
                end = min(start + chunk_size, node_embeds.size(0))
                local_rows, slots = torch.nonzero(topk_idx[start:end] == expert_id, as_tuple=True)
                selected = node_embeds[start:end][local_rows]
                if selected.numel() == 0:
                    # Geoopt cannot project a 0 x D tensor. The released dense
                    # call would only consume the two weight-dropout masks;
                    # reproduce those RNG draws without retaining a useless
                    # autograd graph or performing manifold operations.
                    if self.training:
                        with torch.no_grad():
                            F.dropout(
                                expert.linear1.weight,
                                expert.linear1.dropout,
                                training=True,
                            )
                            F.dropout(
                                expert.linear2.weight,
                                expert.linear2.dropout,
                                training=True,
                            )
                    continue

                def expert_to_tangent(current, current_expert=expert):
                    return current_expert.manifold.logmap0(current_expert(current))

                if self.training:
                    out = checkpoint(expert_to_tangent, selected, use_reentrant=True)
                else:
                    out = expert_to_tangent(selected)
                routed_outputs[start + local_rows, slots] = out

        reconstructed = (routed_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        if self.training and update_memory:
            batch_size = min(64, len(node_embeds))
            expert_errors = torch.zeros(
                len(node_embeds), self.num_experts, device=node_embeds.device
            )
            for start in range(0, len(node_embeds), batch_size):
                current = node_embeds[start : start + batch_size]
                with torch.no_grad():
                    for expert_id, expert in enumerate(self.experts):
                        out = expert.manifold.logmap0(expert(current))
                        expert_errors[start : start + len(current), expert_id] = F.mse_loss(
                            out, current, reduction="none"
                        ).mean(dim=1)
            assignments = torch.zeros_like(expert_errors)
            assignments.scatter_(1, topk_idx, topk_weights)
            self.router.update_memory(node_embeds, expert_errors, assignments)

        entropy = -(gate_weights * torch.log(gate_weights + 1e-9)).sum(dim=-1)
        info = {
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "gate_logits": gate_logits,
            "noisy_gate_logits": noisy_logits,
            "entropy": entropy,
        }
        return reconstructed, gate_weights, self.feature_decoder(reconstructed), info

    def periodic_memory_update(self, node_embeddings: torch.Tensor):
        if hasattr(self, "_update_step_counter"):
            self._update_step_counter += 1
        else:
            self._update_step_counter = 0
        if self._update_step_counter % 20 != 0:
            return
        size = min(100, node_embeddings.size(0))
        indices = torch.randperm(node_embeddings.size(0))[:size]
        samples = node_embeddings[indices]
        with torch.no_grad():
            errors = []
            for expert in self.experts:
                output = expert.manifold.logmap0(expert(samples))
                errors.append(F.mse_loss(output, samples, reduction="none").mean(dim=1))
        self.router.update_memory(samples, torch.stack(errors, dim=1))

    def move_memory(self, device):
        self.router.move_memory(device)

    def get_latest_routing_stats(self) -> Optional[Dict]:
        return getattr(self, "latest_routing_stats", None)

    def record_routing_stats(self, info):
        with torch.no_grad():
            topk_idx = info["topk_idx"]
            counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float()
            distribution = counts / counts.sum().clamp_min(1.0)
            self.latest_routing_stats = {
                "entropy_mean": float(info["entropy"].mean()),
                "expert_usage_dist": distribution.cpu().tolist(),
                "load_balance_cv": float(distribution.std() / distribution.mean().clamp_min(1e-9)),
                "avg_topk_weight": info["topk_weights"].mean().item(),
            }


class GADMoRE(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = dict(hp)
        in_feats = int(hp["dim"])
        h_feats = int(hp["hidden_dim"])
        num_layers = int(hp["num_layers"])
        self.num_hops = int(hp["num_hops"])
        self.act = getattr(nn, hp["activation"])()
        self.dropout = nn.Dropout(hp["dropout_rate"]) if hp["dropout_rate"] > 0 else nn.Identity()
        self.layers = nn.ModuleList()
        if num_layers > 0:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for _ in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))
        embedding_dim = h_feats * self.num_hops
        self.anomaly_scorer = AnomalyScorerMoE(
            embedding_dim=embedding_dim,
            original_feature_dim=in_feats,
            num_experts=hp["num_experts"],
            expert_hidden_dim=hp["expert_hidden_dim"],
            top_k=hp["top_k"],
            init_curvs=hp.get("init_curvs"),
            gate_temperature=hp["gate_temperature"],
            gate_noise_type=hp["gate_noise_type"],
            gate_noise_std=hp["gate_noise_std"],
            memory_size=hp["memory_size"],
        )

    def encode(self, x_list):
        current = x_list
        for layer_id, layer in enumerate(self.layers):
            if layer_id != 0:
                current = [self.dropout(x) for x in current]
            current = [layer(x) for x in current]
            if layer_id != len(self.layers) - 1:
                current = [self.act(x) for x in current]
        return torch.hstack([current[i] - current[0] for i in range(1, len(current))])

    def forward(self, graph):
        return self.encode(graph.x_list)

    def move_memory(self, device):
        self.anomaly_scorer.move_memory(device)

    @torch.no_grad()
    def anomaly_score(self, x_list, node_chunk: int):
        scores = []
        for start in range(0, x_list[0].shape[0], node_chunk):
            embedding = self.encode([x[start : start + node_chunk] for x in x_list])
            reconstructed, _, _, _ = self.anomaly_scorer.moe_reconstruction(
                embedding, update_memory=False
            )
            scores.append(torch.sqrt(torch.sum((embedding - reconstructed) ** 2, dim=1)).cpu())
        return torch.cat(scores).numpy()
