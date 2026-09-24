"""Regression checks for official GGAD input construction and cache identity."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from common.data import _content_fingerprint
from ggad.runners import taggad, unprompt
from ggad.vendor.anomalygfm.graph import build_adjs as anomalygfm_adjs
from ggad.vendor.iaggad.graph import build_aff
from ggad.vendor.unprompt.graph import build_adjs as unprompt_adjs


class OfficialInputTests(unittest.TestCase):
    def test_target_runners_use_the_current_full_graph_loader_signature(self):
        adjacency = sp.eye(12, format="csr", dtype=np.float32)
        features = np.random.RandomState(0).normal(size=(12, 8)).astype(np.float32)
        labels = np.array([0, 1] * 6)
        mark = np.ones(12, dtype=bool)

        def load(name):
            self.assertEqual(name, "toy-target")
            return adjacency, features, labels, mark

        with patch.object(unprompt, "load_target_marked", side_effect=load) as loader, patch.object(
            unprompt, "x_svd_torch", return_value=features
        ):
            graph = unprompt._graph("toy-target", "cpu", target=True)
            self.assertEqual(graph.n, 12)
            np.testing.assert_array_equal(graph.mark, mark)
            loader.assert_called_once_with("toy-target")

        hp = dict(taggad.C.TAGGAD_HP, in_feats=8)
        with patch.object(taggad, "load_target_marked", side_effect=load) as loader, patch.object(
            taggad, "_official_aligned_features", return_value=torch.from_numpy(features)
        ):
            graph = taggad._build_graph("toy-target", hp, "cpu", target=True)
            self.assertEqual(graph.n, 12)
            np.testing.assert_array_equal(graph.mark, mark)
            loader.assert_called_once_with("toy-target")

    def test_anomalygfm_keeps_raw_loops_only_in_gcn_branch(self):
        raw = sp.csr_matrix(np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
        gcn, residual = anomalygfm_adjs(raw, "cpu")
        rowsum = np.asarray(raw.sum(1)).ravel()
        inv_sqrt = np.zeros_like(rowsum)
        np.power(rowsum, -0.5, out=inv_sqrt, where=rowsum > 0)
        expected_gcn = raw.dot(sp.diags(inv_sqrt)).T.dot(sp.diags(inv_sqrt))
        expected_gcn = expected_gcn + sp.eye(3)
        no_loop = raw.copy()
        no_loop.setdiag(0)
        no_loop.eliminate_zeros()
        rowsum = np.asarray(no_loop.sum(1)).ravel()
        inv = np.zeros_like(rowsum)
        np.divide(1.0, rowsum, out=inv, where=rowsum > 0)
        expected_residual = sp.diags(inv).dot(no_loop)
        np.testing.assert_allclose(gcn.to_dense().numpy(), expected_gcn.toarray())
        np.testing.assert_allclose(residual.to_dense().numpy(), expected_residual.toarray())

    def test_unprompt_uses_released_conditional_loop_branch(self):
        raw = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int64))
        with_loop, raw_with_loop, without_loop = unprompt_adjs(raw)

        def row_normalize(matrix):
            rowsum = np.asarray(matrix.sum(1)).ravel()
            inv = np.zeros_like(rowsum, dtype=np.float64)
            np.divide(1.0, rowsum, out=inv, where=rowsum > 0)
            return sp.diags(inv).dot(matrix).toarray()

        expected_with = raw + sp.eye(3)
        np.testing.assert_array_equal(raw_with_loop.toarray(), expected_with.toarray())
        np.testing.assert_allclose(with_loop.to_dense().numpy(), row_normalize(expected_with))
        np.testing.assert_allclose(without_loop.to_dense().numpy(), row_normalize(raw))

    def test_iaggad_directed_norm_matches_graphconv_both(self):
        raw = sp.csr_matrix(
            np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], np.float32)
        )
        actual, _ = build_aff(raw, "cpu")
        graph = raw.copy()
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph = graph + sp.eye(4, dtype=np.float32)
        coo = graph.tocoo()
        out_degree = np.asarray(graph.sum(1)).ravel()
        in_degree = np.asarray(graph.sum(0)).ravel()
        values = coo.data / np.sqrt(out_degree[coo.row] * in_degree[coo.col])
        expected = sp.coo_matrix((values, (coo.col, coo.row)), shape=graph.shape)
        np.testing.assert_allclose(actual.to_dense().numpy(), expected.toarray())

    def test_content_fingerprint_changes_when_values_or_edges_change(self):
        first_features = np.ones((4, 3), dtype=np.float32)
        second_features = first_features.copy()
        second_features[0, 0] = 2
        self.assertNotEqual(
            _content_fingerprint(first_features, "features"),
            _content_fingerprint(second_features, "features"),
        )
        first_graph = sp.csr_matrix(([1.0, 1.0], ([0, 1], [1, 2])), shape=(4, 4))
        second_graph = sp.csr_matrix(([1.0, 1.0], ([0, 2], [1, 3])), shape=(4, 4))
        self.assertEqual(first_graph.nnz, second_graph.nnz)
        self.assertNotEqual(
            _content_fingerprint(first_graph, "graph"),
            _content_fingerprint(second_graph, "graph"),
        )


if __name__ == "__main__":
    unittest.main()
