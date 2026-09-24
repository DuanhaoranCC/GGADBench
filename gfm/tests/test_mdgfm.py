"""Focused tests for the sparse MDGFM model implementation."""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import gfm.config as config
import gfm.experiment as foundation_run
import gfm.runners.mdgfm as mdgfm_runner
from gfm.vendor.mdgfm import implementation as mdgfm


def _small_hp(**overrides):
    hp = {
        "hid_units": 8,
        "layers_num": 3,
        "pretrain_dropout": 0.0,
        "combinetype": "mul",
        "pretrain_lr": 1e-3,
        "pretrain_weight_decay": 1e-4,
        "pretrain_epochs": 1,
        "pretrain_patience": 5,
        "edge_chunk": 3,
        "knn_k": 2,
        "homophilic_knn_k": 2,
        "homophilic_datasets": (),
        "knn_query_chunk": 4,
        "exact_knn_max_nodes": 100,
        "lsh_tables": 2,
        "lsh_window": 3,
        "gsl_search_seed": 0,
        "gsl_dropout": 0.0,
        "alignment_temperature": 0.2,
        "alignment_batch_size": 4,
        "contrastive_key_chunk": 5,
        "downstream_lr": 1e-3,
        "downstream_steps": 1,
        "adjacency_mix": 0.5,
        "prototype_temperature": 1.0,
        "target_knn_refresh_interval": 1,
        "target_knn_refresh_max_nodes": 100,
        "full_target_tune_max_nodes": 100,
        "eval_query_batch": 4,
        "cache_features": False,
        "clip_grad": 8.0,
    }
    hp.update(overrides)
    return hp


def _dense_knn_operator(h: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(h, dim=1, eps=1e-12)
    rows = torch.arange(h.shape[0])[:, None].expand_as(cols)
    raw = F.relu((normalized[:, None, :] * normalized[cols]).sum(2))
    sparse = torch.zeros(h.shape[0], h.shape[0], dtype=h.dtype)
    sparse[rows.reshape(-1), cols.reshape(-1)] += raw.reshape(-1)
    sym = 0.5 * (sparse + sparse.T)
    with_loops = sym + torch.eye(h.shape[0], dtype=h.dtype)
    inv = with_loops.sum(1).clamp_min(1e-10).rsqrt()
    return inv[:, None] * with_loops * inv[None, :]


def _dense_directional_alignment(query_all, key_all, positive, temperature):
    query_all = F.normalize(query_all, dim=1, eps=1e-12)
    key_all = F.normalize(key_all, dim=1, eps=1e-12)
    logits = query_all @ key_all.T / temperature
    denominator = torch.logsumexp(logits, dim=1)
    identity = (denominator - logits.diagonal()).mean()
    numerators = []
    for row in range(positive.shape[0]):
        valid = positive[row] > 0
        numerators.append(
            torch.logsumexp(logits[row, valid] + torch.log(positive[row, valid]), dim=0)
        )
    graph = (denominator - torch.stack(numerators)).mean()
    return identity + graph


def test_original_adjacency_normalization_matches_dense_reference():
    adjacency = sp.coo_matrix(
        (
            np.ones(7, dtype=np.float32),
            (
                np.array([0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 1, 0, 2, 1, 2, 2]),
            ),
        ),
        shape=(3, 3),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = mdgfm._normalized_original_adjacency(adjacency)
    assert not any(issubclass(item.category, sp.SparseEfficiencyWarning) for item in caught)

    binary = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    inv = np.power(binary.sum(1), -0.5)
    expected = inv[:, None] * binary * inv[None, :]
    np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-6, atol=1e-6)


def test_fixed_csr_spmm_forward_and_backward_match_dense():
    matrix = sp.csr_matrix(
        np.array(
            [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]],
            dtype=np.float32,
        )
    )
    graph = mdgfm.FixedCSRGraph(matrix)
    base = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-1.0, 2.0]], dtype=torch.float32)
    custom_x = base.clone().requires_grad_(True)
    dense_x = base.clone().requires_grad_(True)
    custom = graph.spmm(custom_x, edge_chunk=2)
    dense = torch.from_numpy(matrix.toarray()) @ dense_x
    torch.testing.assert_close(custom, dense)

    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]])
    (custom * weight).sum().backward()
    (dense * weight).sum().backward()
    torch.testing.assert_close(custom_x.grad, dense_x.grad)


def test_differentiable_knn_spmm_matches_dense_and_has_live_gradients():
    cols = torch.tensor([[1, 2], [0, 3], [3, 1], [2, 0]], dtype=torch.long)
    base_values = torch.tensor(
        [[0.2, 0.1], [0.3, 0.15], [0.25, 0.05], [0.4, 0.12]],
        dtype=torch.float32,
    )
    base_diagonal = torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float32)
    base_x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7.0

    custom_x = base_x.clone().requires_grad_(True)
    custom_values = base_values.clone().requires_grad_(True)
    custom_diagonal = base_diagonal.clone().requires_grad_(True)
    graph = mdgfm.DifferentiableKNNGraph(cols, custom_values, custom_diagonal)
    custom = graph.spmm(custom_x, edge_chunk=3)

    dense_x = base_x.clone().requires_grad_(True)
    dense_values = base_values.clone().requires_grad_(True)
    dense_diagonal = base_diagonal.clone().requires_grad_(True)
    rows = torch.arange(4).repeat_interleave(2)
    destinations = cols.reshape(-1)
    indices = torch.cat(
        (
            torch.stack((rows, destinations)),
            torch.stack((destinations, rows)),
            torch.stack((torch.arange(4), torch.arange(4))),
        ),
        dim=1,
    )
    edge_values = torch.cat((dense_values.reshape(-1), dense_values.reshape(-1), dense_diagonal))
    dense_matrix = torch.sparse_coo_tensor(indices, edge_values, size=(4, 4)).to_dense()
    dense = dense_matrix @ dense_x
    torch.testing.assert_close(custom, dense)

    custom.square().sum().backward()
    dense.square().sum().backward()
    torch.testing.assert_close(custom_x.grad, dense_x.grad)
    torch.testing.assert_close(custom_values.grad, dense_values.grad)
    torch.testing.assert_close(custom_diagonal.grad, dense_diagonal.grad)
    assert torch.isfinite(custom_values.grad).all()


def test_exact_knn_matches_dense_topk_and_excludes_self():
    rng = np.random.RandomState(4)
    features = rng.normal(size=(11, 5)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    actual = mdgfm._exact_knn_cols_numpy(features, k=3, query_chunk=4)

    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    expected = np.argsort(-similarity, axis=1)[:, :3]
    np.testing.assert_array_equal(actual, expected)
    assert not np.any(actual == np.arange(len(features))[:, None])


def test_sparse_eq10_eq11_matches_tiny_dense_reference():
    h = torch.tensor(
        [[1.0, 0.2, -0.1], [0.9, 0.1, 0.3], [-0.2, 1.0, 0.4], [0.1, 0.8, 0.7], [0.5, -0.4, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    cols = torch.tensor([[1, 4], [0, 4], [3, 1], [2, 4], [1, 3]], dtype=torch.long)
    graph = mdgfm._refined_from_cols(
        h, cols, {"knn_query_chunk": 2, "gsl_dropout": 0.0}, training=False
    )
    dense = _dense_knn_operator(h, cols)
    signal = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 3.0
    torch.testing.assert_close(
        graph.spmm(signal, edge_chunk=3),
        dense @ signal,
        rtol=1e-6,
        atol=1e-6,
    )


def test_alignment_loss_matches_dense_reference_and_detaches_positive_weights():
    refined_embedding = torch.tensor(
        [[1.0, 0.1, 0.3], [0.2, 1.0, 0.4], [0.6, 0.5, 1.0]],
        requires_grad=True,
    )
    original_embedding = torch.tensor(
        [[0.8, 0.2, 0.4], [0.1, 0.9, 0.5], [0.7, 0.3, 0.8]],
        requires_grad=True,
    )
    cols = torch.tensor([[1], [2], [0]], dtype=torch.long)
    values = torch.tensor([[0.2], [0.3], [0.4]], requires_grad=True)
    diagonal = torch.tensor([0.5, 0.6, 0.7], requires_grad=True)
    graph = mdgfm.DifferentiableKNNGraph(cols, values, diagonal)
    temperature = 0.2

    actual = mdgfm.alignment_loss(
        refined_embedding,
        original_embedding,
        graph,
        temperature=temperature,
        anchor_batch_size=0,
        key_chunk=2,
    )
    positive = torch.zeros(3, 3)
    rows = torch.arange(3)
    positive[rows, cols[:, 0]] += values.detach()[:, 0]
    positive[cols[:, 0], rows] += values.detach()[:, 0]
    positive[rows, rows] += diagonal.detach()
    expected = 0.5 * (
        _dense_directional_alignment(refined_embedding, original_embedding, positive, temperature)
        + _dense_directional_alignment(original_embedding, refined_embedding, positive, temperature)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.backward()
    assert values.grad is None
    assert diagonal.grad is None
    assert torch.isfinite(refined_embedding.grad).all()
    assert torch.isfinite(original_embedding.grad).all()


def test_source_and_target_prompt_equations_and_balance_transfer():
    pretrain = mdgfm.MDGFMPretrain(
        feature_dim=2,
        hidden_dim=4,
        num_sources=2,
        num_layers=1,
        dropout=0.0,
        prompt_type="mul",
    )
    with torch.no_grad():
        pretrain.domain_tokens[0].weight.copy_(torch.tensor([[2.0, 3.0]]))
        pretrain.shared_token.weight.copy_(torch.tensor([[5.0, 7.0]]))
        pretrain.balance_tokens[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        pretrain.balance_tokens[1].weight.copy_(torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
    x = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
    expected_source = F.relu(x * torch.tensor([[2.0, 3.0]]))
    expected_source *= torch.tensor([[5.0, 7.0]])
    torch.testing.assert_close(pretrain.source_features(0, x), expected_source)

    source_tokens, shared, source_balances = pretrain.transfer_tokens()
    adapter = mdgfm.MDGFMDownstreamPrompt(
        source_tokens=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        shared_token=torch.tensor([[7.0, 11.0]]),
        source_balance_tokens=source_balances,
        feature_dim=2,
        prompt_type="add",
    )
    torch.testing.assert_close(adapter.balance_token.weight, source_balances.mean(0, keepdim=True))
    with torch.no_grad():
        adapter.meta_weights.copy_(torch.tensor([[0.25, 0.75]]))
        adapter.specific_token.copy_(torch.tensor([[2.0, 3.0]]))
        adapter.fusion_weights.copy_(torch.tensor([[0.4, 0.6]]))
    target_x = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    composed = torch.tensor([[3.5, 4.5]])
    meta = F.relu(target_x + composed) * torch.tensor([[7.0, 11.0]])
    specific = target_x * torch.tensor([[2.0, 3.0]])
    expected_target = F.elu(0.4 * meta + 0.6 * specific)
    torch.testing.assert_close(adapter.prompt_features(target_x), expected_target)
    torch.testing.assert_close(shared, torch.tensor([[5.0, 7.0]]))


def test_no_grad_gcn_path_matches_regular_forward():
    adjacency = mdgfm._normalized_original_adjacency(
        sp.csr_matrix(
            np.array(
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                dtype=np.float32,
            )
        )
    )
    graph = mdgfm.FixedCSRGraph(adjacency)
    model = mdgfm.MDGFMGCN(3, 5, num_layers=3, dropout=0.0).eval()
    x = torch.tensor(
        [[1.0, 0.0, 2.0], [0.5, -1.0, 0.2], [1.2, 0.4, -0.3], [0.1, 0.8, 1.5]],
        dtype=torch.float32,
    )
    regular = model(x, graph, edge_chunk=3, pretrain_view=False)
    with torch.no_grad():
        optimized = model(x, graph, edge_chunk=3, pretrain_view=False)
    torch.testing.assert_close(optimized, regular, rtol=1e-6, atol=1e-6)


def test_sparse_graph_objects_never_store_dense_n_by_n_arrays():
    n, k = 257, 3
    cols = np.stack([(np.arange(n) + offset) % n for offset in range(1, k + 1)], axis=1).astype(
        np.int32
    )
    values = np.full((n, k), 0.1, dtype=np.float32)
    diagonal = np.full(n, 0.7, dtype=np.float32)
    fixed = mdgfm.FixedKNNGraph(cols, values, diagonal)
    csr = fixed.to_csr()
    assert sp.issparse(csr)
    assert fixed.cols.numel() == n * k
    assert fixed.values.numel() == n * k

    differentiable = mdgfm.DifferentiableKNNGraph(
        torch.from_numpy(cols.astype(np.int64)),
        torch.from_numpy(values),
        torch.from_numpy(diagonal),
    )
    for graph in (fixed, differentiable):
        for value in vars(graph).values():
            if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim == 2:
                assert tuple(value.shape) != (n, n)


def test_score_target_uses_disjoint_complete_query_and_restores_grad_mode():
    n = 8
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    mark = np.array([True, True, True, True, True, True, True, False])
    adjacency = mdgfm._normalized_original_adjacency(sp.eye(n, format="csr"))
    context = mdgfm.GraphContext(
        "target",
        np.ones((n, 2), dtype=np.float32),
        adjacency,
        mdgfm.FixedCSRGraph(adjacency),
        labels,
        mark,
    )
    model = mdgfm.MDGFMPretrain(
        feature_dim=2,
        hidden_dim=4,
        num_sources=1,
        num_layers=1,
        dropout=0.0,
        prompt_type="mul",
    )

    def support_sampler(labels_, mark_, cls, shot, rng, **kwargs):
        return np.array([0 if cls == 0 else 1], dtype=np.int64)

    captured = {}

    def evaluate(labels_, scores):
        captured["labels"] = labels_.copy()
        captured["scores"] = scores.copy()
        return {"ok": True}

    torch.set_grad_enabled(True)
    before = [parameter.requires_grad for parameter in model.gcn.parameters()]
    with mock.patch.object(mdgfm, "sample_class_support", support_sampler), mock.patch.object(
        mdgfm, "_train_target_full", lambda *args, **kwargs: None
    ), mock.patch.object(
        mdgfm,
        "_final_target_embedding",
        lambda *args, **kwargs: np.arange(n * 3, dtype=np.float32).reshape(n, 3),
    ), mock.patch.object(
        mdgfm, "evaluate", evaluate
    ):
        result = mdgfm._score_target(
            model,
            context,
            _small_hp(),
            shot=1,
            seed=0,
            feature_dim=2,
            device="cpu",
        )
    assert result == {"ok": True}
    assert torch.is_grad_enabled()
    assert [parameter.requires_grad for parameter in model.gcn.parameters()] == before
    np.testing.assert_array_equal(captured["labels"], labels[[2, 3, 4, 5, 6]])
    assert captured["scores"].shape == (5,)


def test_synthetic_end_to_end_run():
    rng = np.random.RandomState(0)
    n = 12
    row = np.arange(n - 1)
    raw = sp.csr_matrix(
        (
            np.ones(2 * (n - 1), dtype=np.float32),
            (np.r_[row, row + 1], np.r_[row + 1, row]),
        ),
        shape=(n, n),
    )
    adjacency = mdgfm._normalized_original_adjacency(raw)
    labels = np.array([0, 1] * (n // 2), dtype=np.int64)
    mark = np.ones(n, dtype=bool)
    source = mdgfm.GraphContext(
        "source",
        rng.normal(size=(n, 4)).astype(np.float32),
        adjacency,
        mdgfm.FixedCSRGraph(adjacency),
        labels,
        mark,
    )
    target = mdgfm.GraphContext(
        "target",
        rng.normal(size=(n, 4)).astype(np.float32),
        adjacency,
        mdgfm.FixedCSRGraph(adjacency),
        labels,
        mark,
    )

    def context(name, target, feature_dim, feature_norm, hp):
        return locals_target if target else locals_source

    locals_source = source
    locals_target = target
    with mock.patch.object(mdgfm, "_context", context):
        result = mdgfm.run_mdgfm(
            ["source"],
            ["target"],
            [0],
            _small_hp(),
            "cpu",
            shot=1,
            feature_dim=4,
            feature_norm="none",
        )
    assert set(result) == {"target"}
    assert result["target"]["n"] == 1
    assert np.isfinite(result["target"]["AUROC_mean"])
    assert np.isfinite(result["target"]["AUPRC_mean"])


def test_config_and_dispatch():
    assert config.METHOD_PROTOCOL["mdgfm"] == "few_shot"
    assert config.FOUNDATION_HP["mdgfm"] is config.MDGFM_HP

    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"target": {"AUROC_mean": 0.5}}

    with mock.patch.object(mdgfm_runner, "run_mdgfm", fake_run):
        result = foundation_run.run_method("mdgfm", ["source"], ["target"], "cpu", [0], quick=True)
    hp = captured["args"][3]
    assert hp["pretrain_epochs"] == config.MDGFM_HP["quick_pretrain_epochs"]
    assert hp["downstream_steps"] == config.MDGFM_HP["quick_downstream_steps"]
    assert captured["kwargs"]["feature_dim"] == config.FEATURE_DIM
    assert result["target"]["AUROC_mean"] == 0.5


def load_tests(loader, tests, pattern):
    """Expose the plain focused checks to dependency-free unittest discovery."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
