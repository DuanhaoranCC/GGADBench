"""Exact GPU propagation, backend selection and bounded support adaptation."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from gfm.vendor.mdgpt import implementation as runner
from gfm.vendor.mdgpt.backend import SparseOperator, use_full_graph
from gfm.vendor.mdgpt.model import DualPrompt, MDGPT, support_loss
from common.data import EdgeList
from gfm.vendor.mdgpt.streaming import frozen_encode


class BackendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.x = np.random.RandomState(3).normal(size=(12, 4)).astype(np.float32)
        a = np.random.RandomState(4).uniform(size=(12, 12)).astype(np.float32)
        a[a < .5] = 0
        a /= a.sum(axis=1, keepdims=True)
        self.a = sp.csr_matrix(a)
        self.hp = runner._resolved_hp({"feature_dim": 4, "hidden_dim": 7,
                                      "num_layers": 3, "cache": False})

    def test_sparse_input_gradient_matches_dense_asymmetric_graph(self):
        x = torch.tensor(self.x, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        actual = SparseOperator(self.a, "cpu").apply(x)
        expected = torch.tensor(self.a.toarray()) @ reference
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, reference.grad)

    def test_gpu_selection_uses_memory_not_edge_threshold(self):
        hp = dict(self.hp, stream_edge_threshold=1)
        with patch.object(torch.cuda, "mem_get_info", return_value=(2**30, 2**30)), \
             patch.object(torch.cuda, "memory_reserved", return_value=0), \
             patch.object(torch.cuda, "memory_allocated", return_value=0):
            self.assertTrue(use_full_graph(self.a, "cuda:0", hp))
        with patch.object(torch.cuda, "mem_get_info", return_value=(1, 2**30)), \
             patch.object(torch.cuda, "memory_reserved", return_value=0), \
             patch.object(torch.cuda, "memory_allocated", return_value=0):
            self.assertFalse(use_full_graph(self.a, "cuda:0", hp))

    def test_early_stop_restores_evaluated_best_and_caps_requested_steps(self):
        model = MDGPT(2, 4, 7, 3)
        graph = {"name": "toy", "x": self.x, "norm": self.a,
                 "labels": np.array([0, 1] * 6, dtype=np.int64)}
        hp = dict(self.hp, prompt_max_epochs=5, prompt_min_epochs=2,
                  prompt_patience=2, prompt_min_delta=10.)
        with contextlib.redirect_stdout(io.StringIO()):
            prompt, history = runner._tune_prompt(model, graph, np.arange(4), 0, 800, "cpu", hp)
        self.assertEqual(len(history), 3)
        z = prompt(model.encoder, torch.tensor(self.x), SparseOperator(self.a, "cpu"))[:4]
        value = float(support_loss(z, torch.tensor([0, 1, 0, 1]), hp["temperature"]))
        self.assertLessEqual(value, min(history) + 1e-6)
        with contextlib.redirect_stdout(io.StringIO()):
            _, history = runner._tune_prompt(model, graph, np.arange(4), 0, 800, "cpu",
                                            dict(hp, prompt_patience=100))
        self.assertEqual(len(history), 5)

    def test_oom_retries_only_auto_and_restarts_same_seed(self):
        graph = {"name": "toy", "x": self.x, "norm": self.a,
                 "labels": np.array([0, 1] * 6, dtype=np.int64)}
        sentinel = (object(), [])
        with patch.object(runner, "use_full_graph", return_value=True), \
             patch.object(runner, "_fit_prompt", side_effect=[torch.cuda.OutOfMemoryError(), sentinel]) as fit, \
             patch.object(torch.cuda, "empty_cache"), contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(runner._tune_prompt(None, graph, np.arange(4), 9, 5, "cuda:0", self.hp), sentinel)
        self.assertTrue(fit.call_args_list[0].args[-1])
        self.assertFalse(fit.call_args_list[1].args[-1])
        self.assertEqual(fit.call_args_list[0].args[4:7], fit.call_args_list[1].args[4:7])

    @unittest.skipUnless(os.environ.get("MDGPT_TEST_CUDA") == "1" and torch.cuda.is_available(), "CUDA opt-in")
    def test_cuda_pretraining_encoder_and_domain_token_gradients(self):
        full = MDGPT(2, 4, 7, 3).cuda()
        streamed = MDGPT(2, 4, 7, 3).cuda()
        streamed.load_state_dict(full.state_dict())
        x = torch.tensor(self.x, device="cuda")
        expected = full.encode_source(x, SparseOperator(self.a, "cuda"), 1)
        actual = streamed.encode_source(x, EdgeList(self.a), 1, 5)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)
        for left, right in zip(full.parameters(), streamed.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=5e-5)

    @unittest.skipUnless(os.environ.get("MDGPT_TEST_CUDA") == "1" and torch.cuda.is_available(), "CUDA opt-in")
    def test_cuda_full_and_stream_match_dual_prompt_gradients(self):
        model = MDGPT(2, 4, 7, 3).cuda().freeze()
        one = DualPrompt(model.domain_tokens).cuda()
        two = DualPrompt(model.domain_tokens).cuda()
        two.load_state_dict(one.state_dict())
        nodes = np.array([1, 4, 1, 8])
        expected = one(model.encoder, torch.tensor(self.x, device="cuda"), SparseOperator(self.a, "cuda"))[nodes]
        expected.square().sum().backward()
        with tempfile.TemporaryDirectory() as directory:
            options = dict(output_nodes=nodes, node_chunk=3, edge_chunk=5, work_dir=Path(directory))
            actual = (frozen_encode(two.unifying, self.x, self.a, model.encoder.layers, model.encoder.activations, **options)
                      + frozen_encode(two.mixing(), self.x, self.a, model.encoder.layers, model.encoder.activations, **options))
            actual.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(two.unifying.grad, one.unifying.grad, rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(two.gamma.grad, one.gamma.grad, rtol=5e-5, atol=3e-6)
            del actual
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
