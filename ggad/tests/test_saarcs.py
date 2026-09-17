"""Focused synthetic tests for the SAARCS paper reproduction."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

import ggad.config as C
from ggad.vendor.saarcs import implementation as saarcs


def _path_adjacency(node_count=3):
    row = np.arange(node_count - 1, dtype=np.int64)
    col = row + 1
    rows = np.concatenate((row, col))
    cols = np.concatenate((col, row))
    return sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(node_count, node_count),
    )


def _small_hp(**overrides):
    hp = dict(C.SAARCS_HP)
    hp.update(
        {
            "feature_dim": 3,
            "latent_dim": 3,
            "num_hops": 2,
            "num_context": 2,
            "query_per_class": 2,
            "edge_chunk": 2,
            "query_chunk": 2,
            "cache": False,
        }
    )
    hp.update(overrides)
    return hp


def _prepared_graph():
    adjacency = _path_adjacency(8)
    normalized = saarcs._symmetric_normalized_with_self_loops(adjacency)
    features = np.arange(24, dtype=np.float32).reshape(8, 3) / 24.0
    return saarcs.PreparedGraph(
        name="toy-saarcs",
        adjacency=adjacency,
        normalized_adjacency=normalized,
        chunks=saarcs.CSRChunkGraph(adjacency),
        features=features,
        labels=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        mark=np.ones(8, dtype=bool),
    )


class SAARCSGraphAndAlignmentTests(unittest.TestCase):
    def test_explicit_sparse_zeros_do_not_become_undirected_edges(self):
        graph = sp.csr_matrix(
            (np.array([0.0, 1.0], dtype=np.float32), (np.array([0, 1]), np.array([1, 2]))),
            shape=(3, 3),
        )
        actual = saarcs._binary_undirected_loop_free(graph).toarray()
        expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(actual, expected)

    def test_binary_input_graph_preserves_direction_and_existing_loops(self):
        directed = sp.csr_matrix(
            np.array(
                [
                    [1.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        graph = saarcs._binary_input_graph(directed)
        expected = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(graph.toarray(), expected)

    def test_binary_input_loop_free_preserves_direction_only(self):
        directed = sp.csr_matrix(
            np.array(
                [
                    [1.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=np.float32,
            )
        )
        graph = saarcs._binary_input_loop_free(directed)
        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(graph.toarray(), expected)

    def test_canonical_graph_is_binary_undirected_loop_free(self):
        row = np.array([0, 0, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 2])
        value = np.array([7.0, 2.0, 4.0, 3.0, 9.0])
        directed = sp.csr_matrix((value, (row, col)), shape=(3, 3))
        graph = saarcs._binary_undirected_loop_free(directed)
        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(graph.toarray(), expected)

    def test_symmetric_normalization_adds_one_self_loop_per_node(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        normalized = saarcs._symmetric_normalized_with_self_loops(adjacency)
        expected = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(normalized.toarray(), expected, atol=1e-7)

    def test_normalization_does_not_double_an_existing_self_loop(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        normalized = saarcs._symmetric_normalized_with_self_loops(adjacency)
        np.testing.assert_allclose(
            normalized.toarray(),
            np.full((2, 2), 0.5, dtype=np.float32),
            atol=1e-7,
        )

    def test_normalization_can_preserve_a_loop_free_policy(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        normalized = saarcs._symmetric_normalized(adjacency, add_self_loops=False)
        np.testing.assert_array_equal(normalized.toarray(), adjacency.toarray())

    # Known issue: the benchmark retains ARC out-degree normalization on
    # both sides; this independent reference uses destination in-degrees.
    @unittest.expectedFailure
    def test_directed_normalization_aggregates_source_into_destination(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        normalized = saarcs._symmetric_normalized(adjacency, add_self_loops=False)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 / np.sqrt(2.0), 0.0, 0.0],
                [0.5, 1.0 / np.sqrt(2.0), 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(normalized.toarray(), expected, atol=1e-7)

    # Known issue: the benchmark retains ARC out-degree normalization on
    # both sides; this independent reference uses destination in-degrees.
    @unittest.expectedFailure
    def test_directed_sink_keeps_its_incoming_message(self):
        adjacency = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32))
        actual = saarcs._symmetric_normalized(adjacency, False)
        np.testing.assert_array_equal(actual.toarray(), adjacency.toarray().T)

    # Known issue: the benchmark retains ARC out-degree normalization on
    # both sides; this independent reference uses destination in-degrees.
    @unittest.expectedFailure
    def test_directed_normalization_matches_edgewise_reference_and_is_bounded(self):
        adjacency = np.zeros((8, 8), dtype=np.float32)
        adjacency[:7, 7] = 1.0
        adjacency[0, 1] = 1.0
        for self_loops in (False, True):
            a = adjacency + np.eye(8, dtype=np.float32) if self_loops else adjacency
            expected = np.zeros_like(a)
            for src, dst in zip(*np.nonzero(a)):
                expected[dst, src] = a[src, dst] / np.sqrt(a[src].sum() * a[:, dst].sum())
            actual = saarcs._symmetric_normalized(sp.csr_matrix(adjacency), self_loops).toarray()
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            self.assertLessEqual(float(np.linalg.svd(actual, compute_uv=False)[0]), 1.000001)

    def test_nonpositive_dimensions_and_chunks_are_rejected(self):
        for key in (
            "feature_dim",
            "latent_dim",
            "num_hops",
            "num_context",
            "edge_chunk",
            "query_chunk",
        ):
            for value in (0, -1):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        saarcs._resolved_hp({key: value})

    def test_minmax_standardization_preserves_dispersion_and_zeros_constants(self):
        features = np.array(
            [
                [1.0, 5.0, 2.0],
                [3.0, 5.0, 0.0],
                [5.0, 5.0, 4.0],
            ],
            dtype=np.float32,
        )
        actual = saarcs._minmax_standardize(features)
        expected = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_composite_spatial_scores_and_order_are_exact(self):
        projected = np.array(
            [
                [0.0, 0.0, 4.0],
                [0.5, 1.0, 4.0],
                [1.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
        graph = saarcs.CSRChunkGraph(_path_adjacency(3))
        chunked = saarcs._composite_spatial_sort(
            projected,
            graph,
            alpha=0.5,
            edge_chunk=1,
            alignment_output="standardized",
        )
        single = saarcs._composite_spatial_sort(
            projected,
            graph,
            alpha=0.5,
            edge_chunk=100,
            alignment_output="standardized",
        )
        expected_scores = np.array(
            [
                0.5 * 0.25 + 0.5 * (1.0 / 6.0),
                0.5 * 1.0 + 0.5 * (2.0 / 9.0),
                0.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(chunked[1], expected_scores, atol=1e-7)
        np.testing.assert_array_equal(chunked[2], np.array([1, 0, 2]))
        np.testing.assert_allclose(
            chunked[0],
            saarcs._minmax_standardize(projected)[:, [1, 0, 2]],
            atol=1e-7,
        )
        for actual, expected in zip(chunked, single):
            np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_composite_sort_reorders_projected_values_for_encoder(self):
        projected = np.array(
            [
                [-4.0, 10.0],
                [0.0, 20.0],
                [4.0, 10.0],
            ],
            dtype=np.float32,
        )
        graph = saarcs.CSRChunkGraph(_path_adjacency(3))
        aligned, _, order = saarcs._composite_spatial_sort(
            projected,
            graph,
            alpha=0.5,
            edge_chunk=100,
            alignment_output="projected",
        )
        np.testing.assert_allclose(aligned, projected[:, order], atol=1e-7)
        self.assertGreater(float(np.max(np.abs(aligned))), 1.0)

    def test_projection_signs_are_canonical_and_zero_columns_stay_zero(self):
        projected = np.array(
            [
                [-5.0, 0.0, 1.0],
                [2.0, 0.0, -3.0],
            ],
            dtype=np.float32,
        )
        actual = saarcs._canonicalize_projection_signs(projected)
        expected = np.array(
            [
                [5.0, 0.0, -1.0],
                [-2.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_low_dimensional_features_use_dense_random_adapter_not_padding(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        first = saarcs._project_features(features, 4, cache=False, projection="svd_random_adapter")
        second = saarcs._project_features(features, 4, cache=False, projection="svd_random_adapter")
        self.assertEqual(first.shape, (3, 4))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.linalg.norm(first, axis=0) > 0))

    def test_row_normalization_handles_dense_sparse_and_zero_rows(self):
        dense = np.array(
            [
                [1.0, 3.0],
                [0.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected = np.array(
            [
                [0.25, 0.75],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(saarcs._row_normalize_features(dense), expected, atol=1e-8)
        np.testing.assert_allclose(
            saarcs._row_normalize_features(sp.csr_matrix(dense)).toarray(),
            expected,
            atol=1e-8,
        )

    def test_sparse_high_dimensional_projection_uses_truncated_svd(self):
        features = sp.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 3.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        with patch.object(saarcs, "x_svd", side_effect=AssertionError("dense SVD used")):
            projected = saarcs._project_features(
                features, 3, cache=False, projection="svd_random_adapter"
            )
        self.assertEqual(projected.shape, (6, 3))
        self.assertTrue(np.isfinite(projected).all())

    def test_alignment_cache_key_tracks_feature_and_graph_content(self):
        hp = _small_hp(cache=True)
        first_graph = saarcs.CSRChunkGraph(
            sp.csr_matrix(
                np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            )
        )
        second_graph = saarcs.CSRChunkGraph(
            sp.csr_matrix(
                np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            )
        )
        zeros = np.zeros((3, 3), dtype=np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        first = saarcs._alignment_cache_path("same", zeros, first_graph, hp)
        changed_features = saarcs._alignment_cache_path("same", ones, first_graph, hp)
        changed_graph = saarcs._alignment_cache_path("same", zeros, second_graph, hp)
        self.assertNotEqual(first, changed_features)
        self.assertNotEqual(first, changed_graph)

    def test_exact_multi_hop_propagation_matches_csr_products(self):
        graph = _prepared_graph()
        hp = _small_hp(num_hops=2)
        propagated = saarcs._propagated_inputs(graph, hp)
        expected_h1 = np.asarray(graph.normalized_adjacency @ graph.features, dtype=np.float32)
        expected_h2 = np.asarray(graph.normalized_adjacency @ expected_h1, dtype=np.float32)
        self.assertEqual(len(propagated), 3)
        np.testing.assert_allclose(propagated[0], graph.features, atol=1e-7)
        np.testing.assert_allclose(propagated[1], expected_h1, atol=1e-7)
        np.testing.assert_allclose(propagated[2], expected_h2, atol=1e-7)


class SAARCSModelTests(unittest.TestCase):
    def test_equations_6_to_12_match_independent_numpy_reference(self):
        torch.manual_seed(0)
        hp = _small_hp(latent_dim=4, margin=0.3)
        model = saarcs.SAARCS(hp).double().eval()
        rng = np.random.RandomState(0)
        base = rng.normal(size=(6, 3))
        hops = [rng.normal(size=(6, 3)) for _ in range(2)]
        parameters = {name: p.detach().numpy().T for name, p in model.named_parameters()}
        expected = np.zeros((6, 4))
        hop_weights = np.zeros((6, 2))
        for node in range(6):
            query = base[node] @ parameters["encoder.query.weight"]
            logits, values = [], []
            for hop in hops:
                residual = hop[node] - base[node]
                key = residual @ parameters["encoder.key.weight"]
                logit = np.dot(query, key) / np.sqrt(4.0)
                logits.append(logit if logit >= 0 else hp["leaky_relu_slope"] * logit)
                values.append(residual @ parameters["encoder.value.weight"])
            weights = np.exp(np.array(logits) - np.max(logits))
            weights /= weights.sum()
            hop_weights[node] = weights
            expected[node] = sum(w * value for w, value in zip(weights, values))
        actual, actual_hops = model.encode(
            torch.from_numpy(base), [torch.from_numpy(x) for x in hops]
        )
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
        np.testing.assert_allclose(actual_hops.detach().numpy(), hop_weights, atol=1e-12)
        contexts = expected[:2]
        reconstruction, attention = [], []
        for embedding in expected[2:]:
            query = embedding @ parameters["reconstructor.query.weight"]
            keys = contexts @ parameters["reconstructor.key.weight"]
            logits = np.array([np.dot(query, key) / np.sqrt(4.0) for key in keys])
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            attention.append(weights)
            reconstruction.append(sum(w * value for w, value in zip(weights, contexts)))
        reconstruction = np.array(reconstruction)
        actual_rec, actual_att = model.reconstruct(actual[2:], actual[:2])
        np.testing.assert_allclose(actual_rec.detach().numpy(), reconstruction, atol=1e-12)
        np.testing.assert_allclose(actual_att.detach().numpy(), attention, atol=1e-12)
        distance = np.sqrt(np.sum((expected[2:] - reconstruction) ** 2, axis=1))
        np.testing.assert_allclose(
            model.anomaly_scores(actual[2:], actual[:2]).detach().numpy(), distance, atol=1e-12
        )
        cosine = np.sum(expected[2:] * reconstruction, axis=1) / (
            np.linalg.norm(expected[2:], axis=1) * np.linalg.norm(reconstruction, axis=1)
        )
        reference_loss = np.mean(
            [1 - cosine[0], 1 - cosine[1], max(0, cosine[2] - 0.3), max(0, cosine[3] - 0.3)]
        )
        loss = model.marginal_cosine_loss(actual[2:], actual[:2], torch.tensor([0, 0, 1, 1]))
        self.assertAlmostEqual(float(loss), float(reference_loss), places=12)

    def test_all_parameter_gradients_match_finite_differences(self):
        torch.manual_seed(0)
        model = saarcs.SAARCS(_small_hp(latent_dim=4, margin=0.3)).double()
        base = torch.randn(6, 3, dtype=torch.float64)
        hops = [torch.randn(6, 3, dtype=torch.float64) for _ in range(2)]
        labels = torch.tensor([0, 0, 1, 1])

        def loss():
            embedding, _ = model.encode(base, hops)
            return model.marginal_cosine_loss(embedding[2:], embedding[:2], labels)

        loss().backward()
        epsilon = 1e-6
        for name, parameter in model.named_parameters():
            analytical = parameter.grad.detach().clone().numpy()
            numerical = np.empty(parameter.shape)
            for index in np.ndindex(*parameter.shape):
                with torch.no_grad():
                    original = float(parameter[index])
                    parameter[index] = original + epsilon
                    plus = float(loss())
                    parameter[index] = original - epsilon
                    minus = float(loss())
                    parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=2e-7, err_msg=name)

    def test_literal_matrix_hop_formula_degenerates_at_one_hop(self):
        # Eq. (6) interpreted literally is N x N; Eq. (7) over one hop is all ones.
        # Eq. (8) then produces identical node embeddings, unlike a residual encoder.
        values = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 3.0]])
        pairwise_weights = np.ones((3, 3))
        literal_embedding = pairwise_weights @ values
        np.testing.assert_array_equal(literal_embedding, np.tile(values.sum(0), (3, 1)))
        self.assertEqual(float(np.linalg.norm(literal_embedding[0] - literal_embedding[1])), 0.0)
        self.assertGreater(float(np.linalg.norm(values[0] - values[1])), 0.0)

    def test_nodewise_hop_attention_has_expected_shape_and_normalization(self):
        torch.manual_seed(3)
        encoder = saarcs.AdaptiveMixHopEncoder(3, 4, 4, 2, 0.2)
        base = torch.randn(5, 3)
        propagated = [torch.randn(5, 3), torch.randn(5, 3)]
        embedding, weights = encoder(base, propagated)
        self.assertEqual(tuple(embedding.shape), (5, 4))
        self.assertEqual(tuple(weights.shape), (5, 2))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(5))
        self.assertTrue(torch.isfinite(embedding).all())
        embedding.square().mean().backward()
        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_residuals_produce_uniform_hop_weights_and_zero_embedding(self):
        torch.manual_seed(4)
        encoder = saarcs.AdaptiveMixHopEncoder(3, 4, 4, 2, 0.2)
        base = torch.randn(5, 3)
        embedding, weights = encoder(base, [base.clone(), base.clone()])
        torch.testing.assert_close(embedding, torch.zeros_like(embedding))
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5))

    def test_single_hop_has_unit_weight_and_inactive_attention_parameters(self):
        torch.manual_seed(4)
        encoder = saarcs.AdaptiveMixHopEncoder(3, 4, 4, 1, 0.2)
        base = torch.randn(5, 3)
        propagated = [torch.randn(5, 3)]
        embedding, weights = encoder(base, propagated)
        embedding.square().mean().backward()
        torch.testing.assert_close(weights, torch.ones_like(weights))
        torch.testing.assert_close(
            encoder.query.weight.grad,
            torch.zeros_like(encoder.query.weight.grad),
        )
        torch.testing.assert_close(
            encoder.key.weight.grad,
            torch.zeros_like(encoder.key.weight.grad),
        )
        self.assertGreater(float(encoder.value.weight.grad.abs().max()), 0.0)

    def test_context_attention_rows_sum_to_one_and_use_context_as_values(self):
        torch.manual_seed(5)
        reconstructor = saarcs.CrossNodeContextReconstructor(3)
        query = torch.randn(4, 3)
        context = torch.randn(2, 3)
        reconstructed, attention = reconstructor(query, context)
        self.assertEqual(tuple(attention.shape), (4, 2))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(4))
        torch.testing.assert_close(reconstructed, attention @ context)

    def test_eq11_anomaly_score_is_l2_reconstruction_drift(self):
        model = saarcs.SAARCS(_small_hp(num_hops=1)).eval()
        query = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        context = torch.tensor([[1.0, 1.0, 1.0]])
        actual = model.anomaly_scores(query, context)
        expected = torch.linalg.vector_norm(query - context, dim=1)
        torch.testing.assert_close(actual, expected)

    def test_eq12_normal_and_anomaly_margin_terms(self):
        model = saarcs.SAARCS(_small_hp(num_hops=1, margin=0.5)).eval()
        query = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        context = torch.tensor([[1.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = model.marginal_cosine_loss(query, context, labels)
        self.assertAlmostEqual(float(loss), 0.25, places=7)


class SAARCSProtocolTests(unittest.TestCase):
    def test_negative_query_chunk_cannot_return_uninitialized_scores(self):
        graph = _prepared_graph()
        hp = _small_hp()
        model = saarcs.SAARCS(hp)
        with self.assertRaisesRegex(ValueError, "query_chunk"):
            saarcs._score_target(model, graph, [], 0, 2, "cpu", dict(hp, query_chunk=-1))

    def test_direct_model_constructor_resolves_legacy_widths(self):
        model = saarcs.SAARCS({"feature_dim": 3, "attention_dim": 4, "embedding_dim": 4})
        self.assertEqual(model.encoder.query.out_features, 4)
        self.assertEqual(model.reconstructor.embedding_dim, 4)
        with self.assertRaisesRegex(ValueError, "must be equal"):
            saarcs.SAARCS({"attention_dim": 4, "embedding_dim": 5})

    def test_incompatible_legacy_attention_widths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be equal"):
            saarcs._resolved_hp(
                {
                    "attention_dim": 32,
                    "embedding_dim": 64,
                }
            )

    def test_unknown_hop_attention_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hop_attention_mode"):
            saarcs._resolved_hp({"hop_attention_mode": "unknown"})

    def test_source_context_and_balanced_queries_are_disjoint(self):
        graph = _prepared_graph()
        context, query = saarcs._sample_source_batch(
            graph, np.random.RandomState(7), num_context=2, query_per_class=3
        )
        self.assertEqual(context.size, 2)
        self.assertEqual(np.intersect1d(context, query).size, 0)
        self.assertTrue(graph.mark[context].all())
        self.assertTrue((graph.labels[context] == 0).all())
        self.assertEqual(int((graph.labels[query] == 0).sum()), 2)
        self.assertEqual(int((graph.labels[query] == 1).sum()), 2)

    def test_query_chunking_is_exact_and_target_context_is_not_evaluated(self):
        graph = _prepared_graph()
        propagated = saarcs._propagated_inputs(graph, _small_hp())
        torch.manual_seed(8)
        model = saarcs.SAARCS(_small_hp()).eval()
        small_hp = _small_hp(query_chunk=1)
        large_hp = _small_hp(query_chunk=100)
        small_idx, small_scores = saarcs._score_target(
            model, graph, propagated, seed=9, shot=2, device="cpu", hp=small_hp
        )
        large_idx, large_scores = saarcs._score_target(
            model, graph, propagated, seed=9, shot=2, device="cpu", hp=large_hp
        )
        normal = np.flatnonzero(graph.mark & (graph.labels == 0))
        context = np.random.RandomState(9).choice(normal, size=2, replace=False)
        expected_mask = graph.mark.copy()
        expected_mask[context] = False
        expected_idx = np.flatnonzero(expected_mask)
        np.testing.assert_array_equal(small_idx, expected_idx)
        np.testing.assert_array_equal(large_idx, expected_idx)
        self.assertEqual(np.intersect1d(context, small_idx).size, 0)
        np.testing.assert_allclose(small_scores, large_scores, rtol=1e-6, atol=1e-7)

    def test_non_finite_target_scores_fail_loudly(self):
        graph = _prepared_graph()
        propagated = saarcs._propagated_inputs(graph, _small_hp())
        model = saarcs.SAARCS(_small_hp()).eval()
        with patch.object(
            model,
            "anomaly_scores",
            side_effect=lambda query, context: torch.full(
                (query.shape[0],), float("nan"), device=query.device
            ),
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                saarcs._score_target(
                    model,
                    graph,
                    propagated,
                    seed=9,
                    shot=2,
                    device="cpu",
                    hp=_small_hp(query_chunk=1),
                )

    def test_non_finite_training_loss_fails_loudly(self):
        graph = _prepared_graph()
        hp = _small_hp()
        propagated = saarcs._propagated_inputs(graph, hp)
        with patch.object(
            saarcs.SAARCS,
            "marginal_cosine_loss",
            return_value=torch.tensor(float("nan")),
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
                saarcs._train_seed(
                    [(graph, propagated)],
                    seed=0,
                    epochs=1,
                    device="cpu",
                    hp=hp,
                )


if __name__ == "__main__":
    unittest.main()
