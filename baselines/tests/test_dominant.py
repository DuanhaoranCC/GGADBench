"""Official-code equivalence and integration tests for DOMINANT."""

import copy
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from baselines import config
from baselines.dominant import Dominant
from baselines.experiment import _canonical
from baselines.runner_dominant import (
    GraphArrays,
    TorchGraph,
    _materialize,
    _to_torch_sparse,
    normalize_adj_official,
    prepare_adjacencies,
    score_graph,
    structure_error_blocks,
    train_epoch,
)
from common.data import EdgeList


class DominantOfficialEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.adj = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        self.adj_label, self.adj_norm = prepare_adjacencies(self.adj)
        self.features = torch.tensor(
            [
                [0.2, -0.1, 0.4],
                [0.0, 0.3, -0.2],
                [0.5, 0.1, 0.0],
                [-0.2, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )

    def test_sparse_normalization_matches_released_utils_expression(self):
        with_eye = self.adj + sp.eye(4, dtype=np.float32, format="csr")
        rowsum = np.asarray(with_eye.sum(1))
        inv = np.power(rowsum, -0.5).reshape(-1)
        diagonal = sp.diags(inv)
        expected = with_eye.dot(diagonal).transpose().dot(diagonal).toarray()
        np.testing.assert_allclose(
            normalize_adj_official(with_eye).toarray(),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_sparse_model_forward_matches_dense_official_math(self):
        torch.default_generator.manual_seed(7)
        model = Dominant(feat_size=3, hidden_size=5, dropout=0.0).eval()
        dense_adj = torch.from_numpy(self.adj_norm.toarray())
        sparse_adj = _to_torch_sparse(self.adj_norm, "cpu")
        with torch.no_grad():
            dense_a_hat, dense_x_hat = model(self.features, dense_adj)
            sparse_a_hat, sparse_x_hat = model(self.features, sparse_adj)
        torch.testing.assert_close(sparse_a_hat, dense_a_hat)
        torch.testing.assert_close(sparse_x_hat, dense_x_hat)

    def test_edge_chunk_model_forward_and_backward_match_dense_math(self):
        torch.default_generator.manual_seed(9)
        dense_model = Dominant(feat_size=3, hidden_size=5, dropout=0.0)
        chunk_model = copy.deepcopy(dense_model)
        dense_adj = torch.from_numpy(self.adj_norm.toarray())
        chunk_adj = EdgeList(self.adj_norm)
        chunk_adj.dominant_edge_chunk = 2

        dense_a_hat, dense_x_hat = dense_model(self.features, dense_adj)
        chunk_a_hat, chunk_x_hat = chunk_model(self.features, chunk_adj)
        torch.testing.assert_close(chunk_a_hat, dense_a_hat)
        torch.testing.assert_close(chunk_x_hat, dense_x_hat)

        (dense_a_hat.square().mean() + dense_x_hat.square().mean()).backward()
        (chunk_a_hat.square().mean() + chunk_x_hat.square().mean()).backward()
        for dense_parameter, chunk_parameter in zip(
            dense_model.parameters(), chunk_model.parameters()
        ):
            torch.testing.assert_close(
                chunk_parameter.grad,
                dense_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_materialize_selects_cpu_edge_list_at_stream_threshold(self):
        arrays = GraphArrays(
            name="toy",
            adj_label=self.adj_label,
            adj_norm=self.adj_norm,
            features=self.features.numpy(),
            labels=np.array([0, 0, 1, 1]),
            mark=np.ones(4, dtype=bool),
        )
        graph = _materialize(
            arrays,
            "cpu",
            {
                "stream_node_threshold": 1,
                "stream_edge_threshold": 1,
                "edge_chunk": 2,
            },
        )
        self.assertIsInstance(graph.adj_norm, EdgeList)
        self.assertEqual(graph.adj_norm.dominant_edge_chunk, 2)

    def test_chunked_score_matches_released_loss_func(self):
        torch.default_generator.manual_seed(11)
        model = Dominant(feat_size=3, hidden_size=4, dropout=0.0).eval()
        graph = TorchGraph(
            "toy",
            self.adj_label,
            _to_torch_sparse(self.adj_norm, "cpu"),
            self.features,
        )
        cfg = {
            "alpha": 0.8,
            "exact_structure_max_nodes": 10,
            "structure_batch_size": 2,
            "score_structure_sample_size": 4,
        }
        with torch.no_grad():
            a_hat, x_hat = model(graph.features, graph.adj_norm)
            attr = torch.sqrt(torch.sum((x_hat - graph.features) ** 2, dim=1))
            target = torch.from_numpy(self.adj_label.toarray())
            struct = torch.sqrt(torch.sum((a_hat - target) ** 2, dim=1))
            expected = (0.8 * attr + 0.2 * struct).numpy()
        actual = score_graph(model, graph, cfg, sample_seed=0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_sampled_formula_is_exact_when_all_columns_are_used(self):
        structure_z = torch.tensor(
            [[0.2, 0.4], [0.1, -0.3], [0.5, 0.2], [-0.2, 0.1]], requires_grad=True
        )
        exact = torch.cat(
            [
                errors
                for _start, _end, errors in structure_error_blocks(
                    structure_z,
                    self.adj_label,
                    row_batch_size=2,
                    exact=True,
                    sample_size=4,
                    sample_seed=3,
                )
            ]
        )
        sampled = torch.cat(
            [
                errors
                for _start, _end, errors in structure_error_blocks(
                    structure_z,
                    self.adj_label,
                    row_batch_size=2,
                    exact=False,
                    sample_size=4,
                    sample_seed=3,
                )
            ]
        )
        torch.testing.assert_close(sampled, exact, rtol=1e-5, atol=1e-6)

    def test_sampled_residual_never_clips_a_negative_algebraic_estimate(self):
        structure_z = torch.tensor([[0.1], [1.0]], dtype=torch.float64)
        target = sp.eye(2, dtype=np.float32, format="csr")
        sampled = torch.cat(
            [
                errors
                for _start, _end, errors in structure_error_blocks(
                    structure_z,
                    target,
                    row_batch_size=2,
                    exact=False,
                    sample_size=1,
                    sample_seed=0,
                )
            ]
        )
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertGreater(float(sampled[1]), 0.0)

    def test_exact_training_epoch_updates_official_model(self):
        torch.default_generator.manual_seed(13)
        model = Dominant(feat_size=3, hidden_size=4, dropout=0.0)
        graph = TorchGraph(
            "toy",
            self.adj_label,
            _to_torch_sparse(self.adj_norm, "cpu"),
            self.features,
        )
        cfg = {
            "alpha": 0.8,
            "exact_structure_max_nodes": 10,
            "structure_batch_size": 2,
            "structure_sample_size": 4,
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = train_epoch(model, optimizer, graph, cfg, sample_seed=0)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(metrics["exact_structure"])
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        )

    def test_blockwise_optimizer_step_matches_dense_released_loss(self):
        torch.default_generator.manual_seed(17)
        dense_model = Dominant(feat_size=3, hidden_size=4, dropout=0.0)
        block_model = copy.deepcopy(dense_model)
        dense_optimizer = torch.optim.Adam(dense_model.parameters(), lr=5e-3)
        block_optimizer = torch.optim.Adam(block_model.parameters(), lr=5e-3)
        dense_adj = torch.from_numpy(self.adj_norm.toarray())
        target = torch.from_numpy(self.adj_label.toarray())

        dense_optimizer.zero_grad()
        a_hat, x_hat = dense_model(self.features, dense_adj)
        attr = torch.sqrt(torch.sum((x_hat - self.features) ** 2, dim=1))
        struct = torch.sqrt(torch.sum((a_hat - target) ** 2, dim=1))
        (0.8 * attr + 0.2 * struct).mean().backward()
        dense_optimizer.step()

        graph = TorchGraph(
            "toy",
            self.adj_label,
            _to_torch_sparse(self.adj_norm, "cpu"),
            self.features,
        )
        cfg = {
            "alpha": 0.8,
            "exact_structure_max_nodes": 10,
            "structure_batch_size": 2,
            "structure_sample_size": 4,
        }
        train_epoch(block_model, block_optimizer, graph, cfg, sample_seed=0)
        for dense_parameter, block_parameter in zip(
            dense_model.parameters(), block_model.parameters()
        ):
            torch.testing.assert_close(block_parameter, dense_parameter, rtol=2e-5, atol=2e-6)


class DominantIntegrationTests(unittest.TestCase):
    def test_official_name_is_canonical_and_typo_is_only_an_alias(self):
        self.assertEqual(_canonical("DOMINANT"), "dominant")
        self.assertEqual(_canonical("DOMINATE"), "dominant")
        self.assertIs(config.method_hp("dominant"), config.DOMINANT_HP)
        self.assertIn("dominant", config.BASELINES)


if __name__ == "__main__":
    unittest.main()
