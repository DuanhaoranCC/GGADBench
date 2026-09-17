"""Regression tests for TAGGAD's exact memory-bounded affinity backward."""

import unittest
from types import SimpleNamespace

import torch

from ggad.runners.taggad import _validate_edge_index
from ggad.vendor.taggad import ARC
from ggad.vendor.taggad.graph import combined_neighbor_scores


def _reference_scores(edge_index, features):
    src, dst = edge_index
    count = features.shape[0]
    degree = features.new_zeros(count)
    sum_l2 = features.new_zeros(count)
    sum_l1 = features.new_zeros(count)
    sum_cos = features.new_zeros(count)
    degree.index_add_(0, src, torch.ones_like(src, dtype=features.dtype))
    difference = features[src] - features[dst]
    sum_l2.index_add_(0, src, torch.norm(difference, p=2, dim=1))
    sum_l1.index_add_(0, src, torch.norm(difference, p=1, dim=1))
    normed = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
    sum_cos.index_add_(0, src, (normed[src] * normed[dst]).sum(dim=1))

    def minmax(values):
        return (values - values.min()) / (values.max() - values.min() + 1e-8)

    divisor = degree + 1e-8
    score_l2 = minmax(sum_l2 / divisor)
    score_l1 = minmax(sum_l1 / divisor)
    score_cos = minmax(sum_cos / divisor)
    return score_l2, score_l1, score_cos, -score_cos.sum()


class TAGGADChunkedScoreTests(unittest.TestCase):
    def _assert_matches_reference(self, device, chunk_size):
        edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 0],
                [0, 1, 0, 2, 1, 2, 3, 4, 5, 4, 1],
            ],
            dtype=torch.long,
            device=device,
        )
        base = torch.tensor(
            [
                [0.2, -0.4, 1.1],
                [1.0, 0.3, -0.2],
                [-0.7, 0.8, 0.4],
                [0.5, -1.2, 0.6],
                [1.4, 0.1, -0.9],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        expected_features = base.clone().requires_grad_(True)
        actual_features = base.clone().requires_grad_(True)
        expected = _reference_scores(edge_index, expected_features)
        actual = combined_neighbor_scores(
            edge_index,
            actual_features,
            num_nodes=base.shape[0],
            chunk_size=chunk_size,
        )
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(actual_value, expected_value)

        expected[-1].backward()
        actual[-1].backward()
        torch.testing.assert_close(
            actual_features.grad,
            expected_features.grad,
            rtol=1e-8,
            atol=1e-10,
        )

    def test_cpu_matches_full_graph_across_chunk_boundaries(self):
        for chunk_size in (1, 4, 100):
            with self.subTest(chunk_size=chunk_size):
                self._assert_matches_reference("cpu", chunk_size)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_full_graph_across_chunk_boundaries(self):
        for chunk_size in (1, 4, 100):
            with self.subTest(chunk_size=chunk_size):
                self._assert_matches_reference("cuda", chunk_size)
        torch.cuda.synchronize()

    def test_bad_edge_is_rejected_before_cuda(self):
        bad = torch.tensor([[0, 1, 5], [1, 2, 0]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "outside"):
            _validate_edge_index("toy", "affinity", bad, num_nodes=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_affinity_loss_reaches_mlp_and_gcn_parameters(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        count = 24
        features = torch.randn(count, 8, device=device)
        src = torch.arange(96, dtype=torch.long) % count
        dst = (src * 7 + torch.arange(96, dtype=torch.long) // count + 1) % count
        edge_index = torch.stack((src, dst))
        graph = SimpleNamespace(
            name="toy",
            x=features,
            x_list=[features.clone() for _ in range(3)],
            edge_index=edge_index,
            low_edge_index=edge_index,
            edge_chunk_size=7,
        )
        model = ARC(
            SimpleNamespace(code_size=8, topk=3),
            in_feats=8,
            h_feats=16,
            num_layers=2,
            activation="ELU",
            num_hops=2,
        ).to(device)
        _, code_loss, node_gcn, node_mlp, _ = model(graph, graph)
        mlp_loss = combined_neighbor_scores(edge_index, node_mlp, count, chunk_size=7)[-1]
        gcn_loss = combined_neighbor_scores(edge_index, node_gcn, count, chunk_size=7)[-1]
        (code_loss.squeeze() + mlp_loss + gcn_loss).backward()
        torch.cuda.synchronize()

        for branch in (model.node_mlps, model.GCN_model):
            gradients = [
                parameter.grad for parameter in branch.parameters() if parameter.requires_grad
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
            )


if __name__ == "__main__":
    unittest.main()
