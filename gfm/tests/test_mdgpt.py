"""Paper-equation and protocol tests for the MDGPT reconstruction."""

import copy
import math
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from common.data import EdgeList
from gfm.vendor.mdgpt.model import (
    GCN,
    MDGPT,
    DualPrompt,
    link_loss,
    prototype_logits,
    prototypes,
    sample_triplets,
    split_support,
    support_loss,
)
from gfm.vendor.mdgpt.preprocessing import align_features, normalize_graph, support_dependency


def _directed_graph():
    # Includes asymmetric edge weights, a pre-existing loop and an isolated node.
    return sp.csr_matrix(
        np.array(
            [
                [2.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )


def _dense_reference(encoder, features, adjacency):
    # Independent full-matrix expression, including bias AFTER propagation.
    state = features
    for layer, activation in zip(encoder.layers, encoder.activations):
        state = F.prelu(adjacency @ state @ layer.weight.T + layer.bias, activation.weight)
    return state


class MDGPTEquationTests(unittest.TestCase):
    def test_equation_four_has_negative_only_denominator(self):
        embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        actual = link_loss(
            embeddings,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[2, 3]]),
            temperature=1.0,
        )
        expected = math.log(math.exp(-1.0) + 1.0) - 1.0
        self.assertAlmostEqual(float(actual), expected, places=6)
        self.assertLess(float(actual), 0.0)
        actual.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_equation_four_rejects_empty_negatives_and_bad_temperature(self):
        embeddings = torch.eye(3)
        for negative, tau in (
            (torch.empty(1, 0, dtype=torch.long), 0.2),
            (torch.tensor([[2]]), 0.0),
        ):
            with self.assertRaises(ValueError):
                link_loss(
                    embeddings, torch.tensor([0]), torch.tensor([1]), negative, temperature=tau
                )

    def test_dual_prompt_encodes_twice_and_keeps_signed_mixing_coefficients(self):
        class NonlinearEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x, adjacency, edge_chunk):
                self.calls += 1
                return x.relu()

        prompt = DualPrompt(torch.eye(2))
        with torch.no_grad():
            prompt.unifying.copy_(torch.tensor([-1.0, 2.0]))
            prompt.gamma.copy_(torch.tensor([2.0, -1.0]))
        encoder = NonlinearEncoder()
        x = torch.tensor([[1.0, -1.0]])
        result = prompt(encoder, x, None)
        torch.testing.assert_close(prompt.mixing(), torch.tensor([2.0, -1.0]))
        torch.testing.assert_close(result, torch.tensor([[2.0, 1.0]]))
        self.assertEqual(encoder.calls, 2)
        self.assertFalse(torch.equal(result, (x * (prompt.unifying + prompt.mixing())).relu()))
        self.assertEqual(sum(p.numel() for p in prompt.parameters()), 4)

    def test_prototype_is_mean_before_cosine_not_mean_of_normalized_embeddings(self):
        support = torch.tensor(
            [[4.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -6.0]], requires_grad=True
        )
        labels = torch.tensor([0, 0, 1, 1])
        centers = prototypes(support, labels)
        torch.testing.assert_close(centers, torch.tensor([[2.0, 1.0], [-1.0, -3.0]]))
        loss = support_loss(support, labels, temperature=0.7)
        direct = F.cross_entropy(prototype_logits(support, centers, 0.7), labels)
        torch.testing.assert_close(loss, direct)
        actual_grad = torch.autograd.grad(loss, support, retain_graph=True)[0]
        detached = F.cross_entropy(prototype_logits(support, centers.detach(), 0.7), labels)
        detached_grad = torch.autograd.grad(detached, support)[0]
        self.assertGreater(float((actual_grad - detached_grad).abs().max()), 1e-5)

    def test_anomaly_log_odds_preserves_softmax_ranking(self):
        embeddings = torch.tensor([[1.0, 0.1], [0.6, 0.8], [-0.2, 1.0], [-0.8, -0.6]])
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = prototype_logits(embeddings, centers, 0.3)
        odds = logits[:, 1] - logits[:, 0]
        probability = logits.softmax(1)[:, 1]
        self.assertTrue(torch.equal(torch.argsort(odds), torch.argsort(probability)))
        torch.testing.assert_close(odds, torch.logit(probability), atol=2e-6, rtol=2e-6)

    def test_prompt_optimization_preserves_frozen_encoder_and_domain_tokens(self):
        torch.manual_seed(19)
        model = MDGPT(2, feature_dim=3, hidden_dim=5, num_layers=3).freeze()
        prompt = DualPrompt(model.domain_tokens)
        features = torch.randn(8, 3)
        graph = torch.eye(8).to_sparse()
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        model_before = {key: value.clone() for key, value in model.state_dict().items()}
        prompt_before = {key: value.clone() for key, value in prompt.named_parameters()}
        optimizer = torch.optim.Adam(prompt.parameters(), lr=0.01)
        loss = support_loss(prompt(model.encoder, features, graph), labels)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNotNone(prompt.unifying.grad)
        self.assertIsNotNone(prompt.gamma.grad)
        optimizer.step()
        for key, actual in model.state_dict().items():
            self.assertTrue(torch.equal(actual, model_before[key]), key)
        for key, actual in prompt.named_parameters():
            self.assertFalse(torch.equal(actual, prompt_before[key]), key)
        self.assertFalse(prompt.tokens.requires_grad)


class MDGPTGraphAndAlignmentTests(unittest.TestCase):
    def test_normalization_preserves_direction_weights_and_adds_identity(self):
        adjacency = _directed_graph()
        before = adjacency.toarray().copy()
        augmented = before + np.eye(5)
        degree = augmented.sum(1)
        expected = augmented / np.sqrt(degree[:, None] * degree[None, :])
        actual = normalize_graph(adjacency).toarray()
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_array_equal(adjacency.toarray(), before)
        self.assertEqual(actual[1, 0], 0.0)
        self.assertEqual(actual[4, 4], 1.0)
        self.assertAlmostEqual(float(actual[0, 0]), 0.5)

    def test_chunked_three_layer_gcn_matches_dense_forward_and_all_gradients(self):
        torch.manual_seed(7)
        normalized = normalize_graph(_directed_graph())
        sparse_model = GCN(feature_dim=3, hidden_dim=4, num_layers=3)
        # Nonzero bias catches accidentally applying adjacency after bias.
        with torch.no_grad():
            for layer in sparse_model.layers:
                layer.bias.normal_(0.0, 0.2)
        dense_model = copy.deepcopy(sparse_model)
        x_sparse = torch.randn(5, 3, requires_grad=True)
        x_dense = x_sparse.detach().clone().requires_grad_(True)
        sparse = sparse_model(x_sparse, EdgeList(normalized), edge_chunk=2)
        dense = _dense_reference(dense_model, x_dense, torch.from_numpy(normalized.toarray()))
        torch.testing.assert_close(sparse, dense, atol=2e-6, rtol=2e-6)
        weights = torch.linspace(-1.0, 2.0, sparse.numel()).reshape(sparse.shape)
        (sparse * weights).sum().backward()
        (dense * weights).sum().backward()
        torch.testing.assert_close(x_sparse.grad, x_dense.grad, atol=3e-6, rtol=3e-6)
        for (name, actual), (_, expected) in zip(
            sparse_model.named_parameters(), dense_model.named_parameters()
        ):
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-6, msg=name)

    def test_svd_dense_and_sparse_match_uncentered_us_with_fixed_right_sign(self):
        rng = np.random.RandomState(8)
        features = rng.normal(size=(14, 6)) + 2.0
        _, _, vt = np.linalg.svd(features, full_matrices=False)
        right = vt[:3].T.copy()
        pivots = np.argmax(np.abs(right), axis=0)
        right *= np.sign(right[pivots, np.arange(3)])
        expected = features @ right
        for x in (features, sp.csr_matrix(features)):
            actual = align_features(x, dim=3, cache=False)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
            self.assertEqual(actual.dtype, np.float32)

    def test_support_dependency_matches_full_graph_outputs_and_gradients(self):
        torch.manual_seed(31)
        n = 12
        row = np.arange(n - 2)
        raw = sp.csr_matrix(
            (np.linspace(1.0, 4.0, len(row)), (row, row + 1)), shape=(n, n), dtype=np.float32
        )
        normalized = normalize_graph(raw)
        features = np.random.RandomState(1).normal(size=(n, 3)).astype(np.float32)
        # Repeated support indices additionally check remapping order.
        support = np.array([4, 0, 4], dtype=np.int64)
        reduced, local_x, local_support = support_dependency(
            normalized, features, support, num_layers=3
        )
        self.assertEqual(reduced.shape, (8, 8))
        np.testing.assert_array_equal(local_support, [4, 0, 4])
        # The boundary's omitted outgoing edge must NOT trigger renormalization.
        np.testing.assert_array_equal(reduced.toarray(), normalized[:8, :8].toarray())
        full_model = GCN(feature_dim=3, hidden_dim=4, num_layers=3)
        local_model = copy.deepcopy(full_model)
        prompt_full = torch.randn(3, requires_grad=True)
        prompt_local = prompt_full.detach().clone().requires_grad_(True)
        full_z = _dense_reference(
            full_model,
            torch.from_numpy(features) * prompt_full,
            torch.from_numpy(normalized.toarray()),
        )[support]
        local_z = _dense_reference(
            local_model,
            torch.from_numpy(local_x) * prompt_local,
            torch.from_numpy(reduced.toarray()),
        )[local_support]
        torch.testing.assert_close(local_z, full_z, atol=2e-6, rtol=2e-6)
        weights = torch.linspace(-2.0, 1.0, full_z.numel()).reshape(full_z.shape)
        (full_z * weights).sum().backward()
        (local_z * weights).sum().backward()
        torch.testing.assert_close(prompt_local.grad, prompt_full.grad, atol=3e-6, rtol=3e-6)
        for actual, expected in zip(local_model.parameters(), full_model.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-6)

    def test_svd_wide_arpack_path_has_correct_truncated_gram(self):
        rng = np.random.RandomState(13)
        features = sp.random(12, 2051, density=0.03, random_state=rng, format="csr")
        actual = align_features(features, dim=3, cache=False)
        u, s, _ = np.linalg.svd(features.toarray(), full_matrices=False)
        expected = (u[:, :3] * s[:3]) @ (u[:, :3] * s[:3]).T
        # The reconstruction Gram is independent of singular-vector signs.
        np.testing.assert_allclose(actual @ actual.T, expected, atol=1e-5, rtol=1e-5)

    def test_svd_refuses_padding_and_nonfinite_features(self):
        for x, dimension in (
            (np.ones((8, 2)), 3),
            (np.ones((2, 8)), 3),
            (np.array([[np.nan, 1.0], [1.0, 2.0]]), 1),
        ):
            with self.assertRaises(ValueError):
                align_features(x, dim=dimension, cache=False)


class MDGPTSamplingAndProtocolTests(unittest.TestCase):
    def test_triplets_are_real_edges_and_distinct_non_neighbors(self):
        n = 9
        adjacency = sp.csr_matrix(
            (np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n)
        )
        adjacency.setdiag(3.0)
        anchor, positive, negative = sample_triplets(
            adjacency, count=100, num_negatives=4, rng=np.random.RandomState(4)
        )
        self.assertTrue((anchor != positive).all())
        self.assertTrue((np.asarray(adjacency[anchor, positive]).ravel() != 0).all())
        for a, neg in zip(anchor, negative):
            self.assertEqual(len(set(neg.tolist())), 4)
            self.assertNotIn(a, neg)
            self.assertEqual(adjacency[a, neg].nnz, 0)
        repeated = sample_triplets(adjacency, 100, 4, np.random.RandomState(4))
        for actual, expected in zip((anchor, positive, negative), repeated):
            np.testing.assert_array_equal(actual, expected)

    def test_complete_graph_cannot_silently_emit_false_negative_edges(self):
        with self.assertRaises(ValueError):
            sample_triplets(sp.csr_matrix(np.ones((5, 5))), 10, 1, np.random.RandomState(0))

    def test_support_excludes_unknown_and_queries_are_complete_and_disjoint(self):
        labels = np.array([0] * 7 + [1] * 7 + [2, 3, -1])
        mark = np.array([True] * 14 + [False] * 3)
        support, query = split_support(labels, mark, shot=3, seed=42)
        self.assertEqual(len(support), 6)
        self.assertEqual(len(np.unique(support)), 6)
        self.assertEqual(np.count_nonzero(labels[support] == 0), 3)
        self.assertEqual(np.count_nonzero(labels[support] == 1), 3)
        self.assertEqual(np.intersect1d(support, query).size, 0)
        np.testing.assert_array_equal(np.sort(np.r_[support, query]), np.flatnonzero(mark))
        other_support, other_query = split_support(labels, mark, shot=3, seed=42)
        np.testing.assert_array_equal(support, other_support)
        np.testing.assert_array_equal(query, other_query)

    def test_support_refuses_insufficient_class_or_marked_unknown_label(self):
        for labels, mark, shot in (
            ([0, 0, 1], [1, 1, 1], 2),
            ([0, 0, 1, 1, 2], [1, 1, 1, 1, 1], 1),
        ):
            with self.assertRaises(ValueError):
                split_support(np.array(labels), np.array(mark), shot=shot, seed=0)

    def test_scarce_class_reserves_query_before_sampling_support_with_replacement(self):
        labels = np.array([0, 0, 1, 1, -1])
        mark = np.array([1, 1, 1, 1, 0], dtype=bool)
        support, query = split_support(labels, mark, shot=3, seed=2)
        self.assertEqual(len(support), 6)
        self.assertEqual(len(np.unique(support)), 2)
        self.assertEqual(np.count_nonzero(labels[support] == 0), 3)
        self.assertEqual(np.count_nonzero(labels[support] == 1), 3)
        self.assertEqual(len(query), 2)
        self.assertEqual(set(labels[query]), {0, 1})
        self.assertEqual(np.intersect1d(support, query).size, 0)
        np.testing.assert_array_equal(np.union1d(support, query), np.flatnonzero(mark))

    def test_query_labels_cannot_change_prompt_optimization_or_scores(self):
        from gfm.vendor.mdgpt.implementation import _resolved_hp, _scores, _tune_prompt

        torch.manual_seed(16)
        hp = _resolved_hp(
            {
                "feature_dim": 3,
                "hidden_dim": 4,
                "num_layers": 3,
                "cache": False,
                "record_run": False,
            }
        )
        model = MDGPT(2, feature_dim=3, hidden_dim=4, num_layers=3)
        graph = {
            "name": "toy-label-invariance",
            "x": np.random.RandomState(7).normal(size=(5, 3)).astype(np.float32),
            "norm": normalize_graph(_directed_graph()),
            "labels": np.array([0, 1, 0, 1, -1], dtype=np.int64),
            "mark": np.array([1, 1, 1, 1, 0], dtype=bool),
        }
        support = np.array([0, 1], dtype=np.int64)
        query = np.array([2, 3], dtype=np.int64)
        changed = dict(graph, labels=np.array([0, 1, 1, 0, 99], dtype=np.int64))
        prompt_a, loss_a = _tune_prompt(model, graph, support, 9, 2, "cpu", hp)
        prompt_b, loss_b = _tune_prompt(model, changed, support, 9, 2, "cpu", hp)
        np.testing.assert_array_equal(loss_a, loss_b)
        for a, b in zip(prompt_a.parameters(), prompt_b.parameters()):
            self.assertTrue(torch.equal(a, b))
        score_a = _scores(model, prompt_a, graph, support, query, "cpu", hp)
        score_b = _scores(model, prompt_b, changed, support, query, "cpu", hp)
        np.testing.assert_array_equal(score_a, score_b)


if __name__ == "__main__":
    unittest.main()
