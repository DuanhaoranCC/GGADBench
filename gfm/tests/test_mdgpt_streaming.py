"""Dense-reference checks for MDGPT's exact disk-backed frozen GCN backend.

CPU checks run by default. Set MDGPT_TEST_CUDA=1 to repeat the numerical check
on an available CUDA device; this opt-in avoids consuming a benchmark's GPU.
"""

import gc
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from gfm.vendor.mdgpt import streaming


class MDGPTStreamingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mdgpt-stream-test-")
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        rng = np.random.RandomState(2)
        self.x = rng.normal(size=(19, 5)).astype(np.float32)
        adjacency = rng.uniform(size=(19, 19)).astype(np.float32)
        adjacency[adjacency < 0.8] = 0
        adjacency += np.eye(19, dtype=np.float32)
        adjacency /= adjacency.sum(axis=1, keepdims=True)
        # Deliberately asymmetric: backward must use P.T, even though the
        # paper's undirected graph operator is normally symmetric.
        self.adjacency = sp.csr_matrix(adjacency)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.layers = nn.ModuleList([nn.Linear(5, 7), nn.Linear(7, 4), nn.Linear(4, 6)])
            self.activations = nn.ModuleList([nn.PReLU(7), nn.PReLU(), nn.PReLU(6)])
        with torch.no_grad():
            self.activations[0].weight.copy_(torch.linspace(-0.3, 0.4, 7))
            self.activations[1].weight.zero_()
            self.activations[2].weight.copy_(torch.linspace(-0.2, 0.5, 6))
        self.layers.requires_grad_(False)
        self.activations.requires_grad_(False)
        self.nodes = np.array([3, 12, 3, 18, 0], dtype=np.int64)
        self.prompt = torch.tensor([0.6, -1.3, 0.1, 0.8, 1.5], requires_grad=True)

    def _dense(self, prompt):
        h = torch.tensor(self.x, device=prompt.device) * prompt
        adjacency = torch.tensor(self.adjacency.toarray(), device=prompt.device)
        for layer, activation in zip(self.layers, self.activations):
            h = activation(layer(adjacency @ h))
        return h

    def _encode(self, prompt, **overrides):
        options = dict(output_nodes=self.nodes, node_chunk=3, work_dir=self.work)
        options.update(overrides)
        return streaming.frozen_encode(
            prompt, self.x, self.adjacency, self.layers, self.activations, **options
        )

    def _check_selected_output_and_gradient(self, device):
        self.layers.to(device)
        self.activations.to(device)
        dense_prompt = self.prompt.detach().to(device).requires_grad_()
        streamed_prompt = dense_prompt.detach().clone().requires_grad_()
        expected = self._dense(dense_prompt)[self.nodes]
        expected.square().sum().backward()
        actual = self._encode(streamed_prompt)
        self.assertTrue(actual.requires_grad)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        loss = actual.square().sum()
        loss.backward(retain_graph=True)
        first_gradient = streamed_prompt.grad.clone()
        torch.testing.assert_close(first_gradient, dense_prompt.grad, rtol=3e-5, atol=2e-6)
        # Context-owned sign maps must survive a retained autograd graph.
        streamed_prompt.grad = None
        loss.backward()
        torch.testing.assert_close(streamed_prompt.grad, first_gradient)
        self.assertTrue(all(p.grad is None for p in self.layers.parameters()))
        self.assertTrue(all(p.grad is None for p in self.activations.parameters()))
        del actual, loss
        gc.collect()
        self.assertEqual(list(self.work.iterdir()), [])

    def test_selected_outputs_and_prompt_gradient_match_dense(self):
        # Covers repeated/unsorted output nodes, multiple hidden dimensions,
        # scalar/channel PReLU and negative/zero slopes in the same network.
        self._check_selected_output_and_gradient("cpu")

    @unittest.skipUnless(
        os.environ.get("MDGPT_TEST_CUDA") == "1" and torch.cuda.is_available(),
        "set MDGPT_TEST_CUDA=1 with CUDA available",
    )
    def test_cuda_selected_outputs_and_prompt_gradient_match_dense(self):
        self._check_selected_output_and_gradient("cuda:0")

    def test_row_prompt_shape_preserved_in_backward(self):
        prompt = self.prompt.detach().reshape(1, -1).requires_grad_()
        reference_prompt = prompt.detach().clone().requires_grad_()
        expected = self._dense(reference_prompt)[self.nodes]
        expected.sum().backward()
        actual = self._encode(prompt, node_chunk=1)
        actual.sum().backward()
        self.assertEqual(tuple(prompt.grad.shape), (1, 5))
        torch.testing.assert_close(prompt.grad, reference_prompt.grad, rtol=3e-5, atol=2e-6)
        del actual
        gc.collect()
        self.assertEqual(list(self.work.iterdir()), [])

    def test_full_inference_matches_dense_and_closes_maps(self):
        with torch.no_grad():
            expected = self._dense(self.prompt).numpy()
        with streaming.frozen_embeddings(
            self.prompt,
            self.x,
            self.adjacency,
            self.layers,
            self.activations,
            node_chunk=4,
            work_dir=self.work,
        ) as output:
            self.assertIsInstance(output, np.memmap)
            self.assertEqual(output.shape, (19, 6))
            np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
            self.assertTrue(list(self.work.iterdir()))
        self.assertTrue(output._mmap.closed)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_inference_context_cleans_up_after_consumer_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with streaming.frozen_embeddings(
                self.prompt,
                self.x,
                self.adjacency,
                self.layers,
                self.activations,
                node_chunk=3,
                work_dir=self.work,
            ):
                raise RuntimeError("consumer failed")
        self.assertEqual(list(self.work.iterdir()), [])

    def test_encoder_exception_cleans_partial_forward_files(self):
        with patch.object(streaming.F, "linear", side_effect=RuntimeError("layer failed")):
            with self.assertRaisesRegex(RuntimeError, "layer failed"):
                self._encode(self.prompt)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_unfrozen_linear_or_activation_is_rejected(self):
        for parameter in (self.layers[0].weight, self.activations[0].weight):
            with self.subTest(shape=tuple(parameter.shape)):
                parameter.requires_grad_(True)
                with self.assertRaisesRegex(ValueError, "frozen"):
                    self._encode(self.prompt)
                parameter.requires_grad_(False)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_invalid_output_nodes_are_rejected_before_allocating_files(self):
        for nodes in (np.array([[0]]), np.array([0.5]), np.array([True])):
            with self.subTest(nodes=nodes):
                with self.assertRaisesRegex(ValueError, "integer array"):
                    self._encode(self.prompt, output_nodes=nodes)
        for nodes in (np.array([-1]), np.array([len(self.x)])):
            with self.subTest(nodes=nodes):
                with self.assertRaises(IndexError):
                    self._encode(self.prompt, output_nodes=nodes)
        self.assertEqual(list(self.work.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
