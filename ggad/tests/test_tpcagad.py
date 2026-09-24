"""Focused synthetic tests for the TPCA-GAD paper reproduction."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

import ggad.config as C
from ggad.vendor.tpcagad import implementation as tpca


def _toy_adjacency():
    # Triangle 0-1-2, tail 2-3, isolated node 4.
    row = np.array([0, 1, 1, 2, 2, 0, 2, 3])
    col = np.array([1, 0, 2, 1, 0, 2, 3, 2])
    return sp.csr_matrix((np.ones(row.size), (row, col)), shape=(5, 5))


def _small_hp(**overrides):
    hp = dict(C.TPCAGAD_HP)
    hp.update(
        {
            "attribute_dim": 4,
            "hidden_dim": 8,
            "audit_dim": 8,
            "num_layers": 2,
            "edge_chunk": 2,
            "node_chunk": 2,
            "stream_node_threshold": 1,
            "stream_edge_threshold": 1,
            "cache": False,
        }
    )
    hp.update(overrides)
    return hp


def _dense_paper_aggregate(
    graph, message, audit, previous_audit, tau, edge_chunk, checkpoint_chunks
):
    """Independent Eq. (5)/(6) reference, including C_ii=1 only in aggregation."""
    adjacency = sp.csr_matrix((np.ones(graph.nnz), graph.indices, graph.indptr), shape=graph.shape)
    mask = torch.as_tensor(adjacency.toarray(), dtype=message.dtype, device=message.device)
    compatibility = torch.exp(-((audit[:, None] - audit[None, :]) ** 2).sum(2) / tau)
    weights = compatibility * (
        mask + torch.eye(graph.n, dtype=message.dtype, device=message.device)
    )
    output = (weights @ message) / weights.sum(1, keepdim=True)
    consistency = message.new_zeros(())
    if previous_audit is not None:
        previous = torch.exp(
            -((previous_audit[:, None] - previous_audit[None, :]) ** 2).sum(2) / tau
        )
        consistency = (((compatibility - previous) ** 2) * mask).sum()
        consistency = consistency / max(graph.nnz, 1)
    return output, consistency


class TPCAGADStructuralTests(unittest.TestCase):
    def test_explicit_sparse_zeros_are_not_edges(self):
        adjacency = sp.coo_matrix(([0.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))
        graph = tpca._binary_undirected_loop_free(adjacency)
        self.assertEqual(graph.nnz, 2)
        self.assertEqual(graph[0, 1], 0)

    def test_model_constructor_rejects_invalid_settings(self):
        for hp in (
            {"attribute_dim": 0},
            {"num_layers": 1.5},
            {"tau": 0},
            {"compatibility_mode": "typo"},
            {"source_split": (1.2, -0.1, -0.1)},
            {"mlp_depth": 3},
            {"num_layers": True},
            {"hidden_dim": np.nan},
            {"source_split": (0.3, np.nan, 0.7)},
        ):
            with self.subTest(hp=hp), self.assertRaises(ValueError):
                tpca.TPCAGAD(hp)

    def test_corrupt_structural_cache_is_rebuilt(self):
        hp = _small_hp(cache=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tpca, "CACHE", Path(directory)
        ):
            graph, local, pooled = tpca._load_or_build_structural_inputs(
                "toy", _toy_adjacency(), hp
            )
            path = Path(directory) / (tpca._graph_digest(graph, hp) + ".npz")
            path.write_bytes(b"interrupted cache")
            _, actual_local, actual_pooled = tpca._load_or_build_structural_inputs("toy", graph, hp)
            np.testing.assert_array_equal(actual_local, local)
            np.testing.assert_array_equal(actual_pooled, pooled)
            with np.load(path) as cached:
                np.testing.assert_array_equal(cached["local"], local)

    def test_partial_or_invalid_structural_cache_is_rebuilt(self):
        hp = _small_hp(cache=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tpca, "CACHE", Path(directory)
        ):
            graph, local, pooled = tpca._load_or_build_structural_inputs(
                "toy", _toy_adjacency(), hp
            )
            path = Path(directory) / (tpca._graph_digest(graph, hp) + ".npz")
            invalid_caches = (
                {"local": local},
                {"local": np.full_like(local, np.nan), "pooled": pooled},
                {"local": local[:-1], "pooled": pooled},
                {"local": local.astype(np.float64), "pooled": pooled},
            )
            for content in invalid_caches:
                with self.subTest(keys=list(content), shape=content["local"].shape):
                    np.savez(path, **content)
                    _, actual_local, actual_pooled = tpca._load_or_build_structural_inputs(
                        "toy", graph, hp
                    )
                    np.testing.assert_array_equal(actual_local, local)
                    np.testing.assert_array_equal(actual_pooled, pooled)

    def test_failed_cache_publication_leaves_no_partial_archive(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tpca, "CACHE", Path(directory)
        ), patch.object(
            tpca.os, "replace", side_effect=OSError("publication failed")
        ), self.assertRaisesRegex(
            OSError, "publication failed"
        ):
            try:
                tpca._load_or_build_structural_inputs(
                    "toy", _toy_adjacency(), _small_hp(cache=True)
                )
            finally:
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_core_and_clustering_match_bruteforce_random_graphs(self):
        rng = np.random.RandomState(71)
        for n in range(2, 11):
            for probability in (0.0, 0.3, 0.8, 1.0):
                upper = np.triu(rng.rand(n, n) < probability, 1)
                a = (upper | upper.T).astype(np.int64)
                graph = sp.csr_matrix(a)
                degree = a.sum(1)
                denominator = degree * (degree - 1)
                expected_cc = np.zeros(n)
                valid = denominator > 0
                expected_cc[valid] = np.diag(a @ a @ a)[valid] / denominator[valid]
                core = np.zeros(n, dtype=np.int64)
                for k in range(1, n):
                    active = np.ones(n, dtype=bool)
                    while True:
                        remove = active & ((a[:, active].sum(1)) < k)
                        if not remove.any():
                            break
                        active[remove] = False
                    core[active] = k
                np.testing.assert_array_equal(tpca._exact_core_numbers(graph), core)
                np.testing.assert_allclose(
                    tpca._exact_clustering_coefficients(graph), expected_cc, atol=1e-7
                )

    def test_unsupported_paper_choices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fingerprint_hops=1"):
            tpca._resolved_hp({"fingerprint_hops": 2})
        with self.assertRaisesRegex(ValueError, "compatibility_mode"):
            tpca._resolved_hp({"compatibility_mode": "silent-fallback"})

    def test_large_exact_structure_requires_numba_or_cache(self):
        hp = _small_hp(python_structure_max_edges=1)
        with patch.object(tpca, "njit", None), self.assertRaisesRegex(
            RuntimeError, "requires Numba"
        ):
            tpca._load_or_build_structural_inputs("large-without-numba", _toy_adjacency(), hp)

    def test_canonical_graph_is_binary_undirected_loop_free(self):
        row = np.array([0, 0, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 2])
        val = np.array([7.0, 2.0, 4.0, 3.0, 9.0])
        directed = sp.csr_matrix((val, (row, col)), shape=(3, 3))
        graph = tpca._binary_undirected_loop_free(directed)
        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(graph.toarray(), expected)

    def test_exact_fingerprint_statistics_and_pool(self):
        graph, local, pooled = tpca._load_or_build_structural_inputs(
            "toy", _toy_adjacency(), _small_hp()
        )
        self.assertEqual(graph.nnz, 8)
        expected_degree = np.array([2, 2, 3, 1, 0], dtype=np.float32) / 3.0
        expected_clustering = np.array([1, 1, 1 / 3, 0, 0], dtype=np.float32)
        expected_core = np.array([1, 1, 1, 0.5, 0], dtype=np.float32)
        np.testing.assert_allclose(local[:, 0], expected_degree, atol=1e-7)
        np.testing.assert_allclose(local[:, 1], expected_clustering, atol=1e-7)
        np.testing.assert_allclose(local[:, 2], expected_core, atol=1e-7)
        np.testing.assert_allclose(pooled[0], (local[1] + local[2]) / 2, atol=1e-7)
        np.testing.assert_array_equal(pooled[4], np.zeros(3, dtype=np.float32))

    def test_source_split_is_reproducible_disjoint_and_stratified(self):
        labels = np.array([0] * 70 + [1] * 30)
        first = tpca._stratified_source_split(labels, 7)
        second = tpca._stratified_source_split(labels, 7)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        train, val, test = first
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        self.assertEqual(train.size + val.size + test.size, labels.size)
        self.assertAlmostEqual(train.size / labels.size, 0.3, places=2)
        self.assertAlmostEqual(val.size / labels.size, 0.1, places=2)
        for split in first:
            self.assertAlmostEqual(labels[split].mean(), 0.3, delta=0.08)


class TPCAGADChunkTests(unittest.TestCase):
    def _run_aggregate(self, edge_chunk, checkpoint_chunks):
        torch.manual_seed(3)
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        message = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        audit = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        previous = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        output, consistency = tpca._exact_tpca_aggregate(
            graph,
            message,
            audit,
            previous,
            0.5,
            edge_chunk,
            checkpoint_chunks=checkpoint_chunks,
        )
        loss = output.square().sum() + consistency
        gradients = torch.autograd.grad(loss, (message, audit, previous))
        return output.detach(), consistency.detach(), [g.detach() for g in gradients]

    def test_chunked_forward_and_backward_match_single_chunk(self):
        reference = self._run_aggregate(100, False)
        chunked = self._run_aggregate(2, True)
        torch.testing.assert_close(chunked[0], reference[0], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(chunked[1], reference[1], rtol=1e-9, atol=1e-10)
        for actual, expected in zip(chunked[2], reference[2]):
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)

    def test_normalized_aggregation_preserves_constant_messages(self):
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        message = torch.ones(5, 3)
        audit = torch.randn(5, 2)
        output, _ = tpca._exact_tpca_aggregate(graph, message, audit, None, 0.5, 2, False)
        torch.testing.assert_close(output, message)

    # Known v3 limitation: aggregation excludes self; the paper reference includes self.
    @unittest.expectedFailure
    def test_eq6_includes_the_center_node(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
        graph = tpca.CSRChunkGraph(adjacency)
        message = torch.tensor([[1.0], [3.0], [7.0]])
        audit = torch.zeros(3, 2)
        output, _ = tpca._exact_tpca_aggregate(graph, message, audit, None, 0.5, 2, False)
        expected = torch.tensor([[2.0], [2.0], [7.0]])
        torch.testing.assert_close(output, expected)

    # Known v3 limitation: aggregation excludes self; the paper reference includes self.
    @unittest.expectedFailure
    def test_extreme_audit_distance_rejects_neighbor_and_preserves_self(self):
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            )
        )
        graph = tpca.CSRChunkGraph(adjacency)
        message = torch.tensor([[1.0], [3.0]], requires_grad=True)
        audit = torch.tensor([[0.0], [100.0]], requires_grad=True)
        output, _ = tpca._exact_tpca_aggregate(graph, message, audit, None, 0.25, 2, False)
        torch.testing.assert_close(output, message)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(message.grad).all())
        self.assertTrue(torch.isfinite(audit.grad).all())

    # Known v3 limitation: aggregation excludes self; the paper reference includes self.
    @unittest.expectedFailure
    def test_eq6_and_gradients_match_independent_dense_paper_reference(self):
        torch.manual_seed(23)
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        inputs = [torch.randn(5, d, dtype=torch.float64, requires_grad=True) for d in (4, 3, 3)]
        expected, expected_cons = _dense_paper_aggregate(graph, *inputs, 0.5, 2, False)
        actual, actual_cons = tpca._exact_tpca_aggregate(graph, *inputs, 0.5, 2, True)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(actual_cons, expected_cons, atol=1e-12, rtol=1e-10)
        expected_grads = torch.autograd.grad(
            expected.square().sum() + expected_cons, inputs, retain_graph=True
        )
        actual_grads = torch.autograd.grad(actual.square().sum() + actual_cons, inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-10)

    def test_shared_static_consistency_is_zero(self):
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        message = torch.randn(5, 3)
        audit = torch.randn(5, 2)
        _, consistency = tpca._exact_tpca_aggregate(graph, message, audit, audit, 0.5, 2, False)
        self.assertEqual(float(consistency), 0.0)


class TPCAGADModelTests(unittest.TestCase):
    # Known v3 limitation: 1 - mean(exp(-distance)) loses tiny float32 deviations.
    @unittest.expectedFailure
    def test_tiny_topology_deviations_are_preserved(self):
        graph = tpca.CSRChunkGraph(
            sp.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        )
        audit = torch.tensor([[0.0], [1e-5], [3e-5]])
        _, actual = tpca._score_components_from_embeddings(graph, torch.zeros(3, 1), audit, 0.5, 1)
        edges = torch.tensor([2e-10, 8e-10], dtype=torch.float64)
        deviations = -torch.expm1(-edges)
        expected = torch.stack((deviations[0], deviations.mean(), deviations[1]))
        self.assertTrue((actual > 0).all())
        torch.testing.assert_close(actual.double(), expected, atol=1e-16, rtol=1e-6)
        self.assertTrue(torch.equal(1 - torch.exp(-edges.float()), torch.zeros(2)))

    def test_invalid_source_validation_fails_instead_of_returning_random_model(self):
        hp = _small_hp()
        graph = self._prepared_graph()
        graph.labels[:] = 0
        with self.assertRaisesRegex(ValueError, "requires both classes"):
            tpca._train_seed([graph], 0, 1, "cpu", hp)

    def test_nonfinite_target_scores_are_not_silently_sanitized(self):
        graph = self._prepared_graph()
        with patch.object(tpca, "_build_graph", return_value=graph), patch.object(
            tpca, "_train_seed", return_value=tpca.TPCAGAD(_small_hp())
        ), patch.object(
            tpca, "_score_target", return_value=np.full(graph.labels.size, np.nan)
        ), self.assertRaisesRegex(
            ValueError, "non-finite evaluation scores"
        ):
            tpca.run_tpcagad(["source"], ["toy-model"], [0], 1, "cpu", hp=_small_hp())

    def test_valid_source_training_returns_finite_parameters(self):
        hp = _small_hp()
        base = self._prepared_graph()
        adjacency = sp.block_diag([base.adjacency] * 4, format="csr")
        graph = tpca.PreparedGraph(
            name="trainable-toy",
            adjacency=adjacency,
            chunks=tpca.CSRChunkGraph(adjacency),
            local=np.tile(base.local, (4, 1)),
            pooled=np.tile(base.pooled, (4, 1)),
            attributes=np.tile(base.attributes, (4, 1)),
            labels=np.tile(base.labels, 4),
            mark=np.ones(20, dtype=bool),
        )
        model = tpca._train_seed([graph], 0, 1, "cpu", hp)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_empty_source_validation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires finite AUROC"):
            tpca._validation_metrics(tpca.TPCAGAD(_small_hp()), [], [])

    def test_selection_uses_classifier_and_only_validation_indices(self):
        graph = self._prepared_graph()
        graph.labels = np.array([0, 1, 0, 1, 0])
        hp = _small_hp(selection_metric="source_val_macro_auroc")
        model = tpca.TPCAGAD(hp)
        tensors = [
            (graph, *tpca._device_node_tensors(graph, "cpu"), torch.from_numpy(graph.labels))
        ]
        splits = [(np.array([2, 3]), np.array([0, 1]), np.array([4]))]
        with patch.object(
            model, "forward", return_value=(torch.zeros(5, 8), [], torch.tensor(0.0))
        ), patch.object(
            model.classifier,
            "forward",
            return_value=torch.tensor([[0.0], [1.0], [100.0], [-100.0], [0.0]]),
        ), patch.object(
            model,
            "anomaly_scores",
            side_effect=AssertionError("classifier selection must not use target scores"),
        ):
            auc, _ = tpca._validation_metrics(model, tensors, splits)
        self.assertEqual(auc, 1.0)

    def _prepared_graph(self):
        adjacency = _toy_adjacency()
        graph, local, pooled = tpca._load_or_build_structural_inputs(
            "toy-model", adjacency, _small_hp()
        )
        return tpca.PreparedGraph(
            name="toy-model",
            adjacency=graph,
            chunks=tpca.CSRChunkGraph(graph),
            local=local,
            pooled=pooled,
            attributes=np.arange(20, dtype=np.float32).reshape(5, 4) / 20,
            labels=np.array([0, 0, 1, 1, 0], dtype=np.int64),
            mark=np.ones(5, dtype=bool),
        )

    def test_layer_specific_model_has_finite_nonzero_consistency_gradients(self):
        hp = _small_hp(
            compatibility_mode="layer_specific",
            stream_node_threshold=100,
            stream_edge_threshold=100,
        )
        model = tpca.TPCAGAD(hp)
        graph = self._prepared_graph()
        local = torch.from_numpy(graph.local)
        pooled = torch.from_numpy(graph.pooled)
        attrs = torch.from_numpy(graph.attributes)
        h, audits, consistency = model(local, pooled, attrs, graph.chunks, checkpoint_chunks=True)
        self.assertEqual(tuple(h.shape), (5, 8))
        self.assertEqual(len(audits), 2)
        self.assertTrue(torch.isfinite(consistency))
        self.assertGreater(float(consistency.detach()), 0.0)
        loss = model.classifier(h).square().mean() + consistency
        loss.backward()
        named = dict(model.named_parameters())
        mlp_weight = "net.weight" if hp["mlp_depth"] == 1 else "net.0.weight"
        for name in (
            f"context_mlp.{mlp_weight}",
            "attr_projection.net.weight",
            f"input_mlp.{mlp_weight}",
            f"structural_projection.{mlp_weight}",
            f"audit_mlps.0.{mlp_weight}",
            f"audit_mlps.1.{mlp_weight}",
            "blocks.0.message.weight",
            "blocks.0.gate.weight",
            "classifier.weight",
        ):
            grad = named[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)

    # Known v3 limitation: aggregation excludes self; the paper reference includes self.
    @unittest.expectedFailure
    def test_complete_component_with_identical_fingerprints_has_uniform_eq6_output(self):
        adjacency = sp.csr_matrix(np.ones((4, 4)) - np.eye(4))
        hp = _small_hp(stream_node_threshold=100, stream_edge_threshold=100)
        _, local, pooled = tpca._load_or_build_structural_inputs("complete", adjacency, hp)
        model = tpca.TPCAGAD(hp).eval()
        attributes = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        with torch.no_grad():
            h, _, _ = model(
                torch.from_numpy(local),
                torch.from_numpy(pooled),
                attributes,
                tpca.CSRChunkGraph(adjacency),
            )
        # Eq. (6) averages the same complete set including self at every node.
        # Identical fingerprints also give identical gates and residuals.
        torch.testing.assert_close(h, h[0:1].expand_as(h), atol=1e-6, rtol=1e-5)

    def test_eq9_neighbor_context_still_excludes_self(self):
        adjacency = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        graph = tpca.CSRChunkGraph(adjacency)
        representation, topology = tpca._score_components_from_embeddings(
            graph, torch.tensor([[1.0], [3.0]]), torch.zeros(2, 1), 0.5, 1
        )
        torch.testing.assert_close(representation, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(topology, torch.zeros(2))

    # Known v3 limitation: aggregation excludes self; the paper reference includes self.
    @unittest.expectedFailure
    def test_streaming_matches_independent_paper_aggregation(self):
        graph = self._prepared_graph()
        for mode in ("shared_static", "layer_specific"):
            for audit_scale in (1e-4, 1.0, 1000.0):
                with self.subTest(mode=mode, audit_scale=audit_scale):
                    torch.manual_seed(17)
                    hp = _small_hp(compatibility_mode=mode)
                    model = tpca.TPCAGAD(hp).eval()
                    audit_embedding = model.audit_embedding
                    with torch.no_grad(), patch.object(
                        model,
                        "audit_embedding",
                        side_effect=lambda phi, layer: audit_scale * audit_embedding(phi, layer),
                    ):
                        with patch.object(tpca, "_exact_tpca_aggregate", _dense_paper_aggregate):
                            h, audits, _ = model(
                                torch.from_numpy(graph.local),
                                torch.from_numpy(graph.pooled),
                                torch.from_numpy(graph.attributes),
                                graph.chunks,
                            )
                        expected = model.anomaly_score_components(h, audits[-1], graph.chunks)
                        actual = tpca._score_target_components_streaming(model, graph, "cpu", hp)
                    for a, e in zip(actual, expected):
                        np.testing.assert_allclose(a, e.numpy(), rtol=1e-5, atol=1e-6)

    def test_isolated_node_score_is_zero(self):
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        h = torch.randn(5, 4)
        audit = torch.randn(5, 3)
        scores = tpca._scores_from_embeddings(graph, h, audit, 0.5, 0.5, 2)
        self.assertEqual(float(scores[4]), 0.0)
        self.assertTrue(torch.isfinite(scores).all())

    def test_eq9_components_recombine_exactly_for_all_alpha_endpoints(self):
        graph = tpca.CSRChunkGraph(_toy_adjacency())
        torch.manual_seed(11)
        h = torch.randn(5, 4)
        audit = torch.randn(5, 3)
        representation, topology = tpca._score_components_from_embeddings(graph, h, audit, 0.5, 2)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            expected = tpca._scores_from_embeddings(graph, h, audit, 0.5, alpha, 2)
            actual = tpca._combine_score_components(representation, topology, alpha)
            torch.testing.assert_close(actual, expected)
        self.assertEqual(float(representation[4]), 0.0)
        self.assertEqual(float(topology[4]), 0.0)

    def test_streaming_and_in_memory_scores_match(self):
        hp = _small_hp()
        model = tpca.TPCAGAD(hp).eval()
        graph = self._prepared_graph()
        local = torch.from_numpy(graph.local)
        pooled = torch.from_numpy(graph.pooled)
        attrs = torch.from_numpy(graph.attributes)
        with torch.no_grad():
            h, audits, _ = model(local, pooled, attrs, graph.chunks, False)
            expected_components = model.anomaly_score_components(h, audits[-1], graph.chunks)
            expected = model.anomaly_scores(h, audits[-1], graph.chunks).numpy()
            actual_components = tpca._score_target_components_streaming(model, graph, "cpu", hp)
            actual = tpca._score_target_streaming(model, graph, "cpu", hp)
        for streamed, in_memory in zip(actual_components, expected_components):
            np.testing.assert_allclose(streamed, in_memory.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
