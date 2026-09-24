"""End-to-end MDGPT runner protocol tests on full, tiny mocked graphs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from gfm.vendor.mdgpt import implementation as mdgpt


def _raw_graph(seed):
    rng = np.random.RandomState(seed)
    n = 14
    rows = np.repeat(np.arange(n), 2)
    columns = np.column_stack(((np.arange(n) + 1) % n, (np.arange(n) + 4) % n)).reshape(-1)
    adjacency = sp.csr_matrix(
        (rng.uniform(0.2, 1.5, len(rows)).astype(np.float32), (rows, columns)),
        shape=(n, n),
    )
    features = rng.normal(size=(n, 4)).astype(np.float32)
    labels = np.array([0] * 6 + [1] * 6 + [2, -1], dtype=np.int64)
    mark = np.array([True] * 12 + [False, False])
    return adjacency, features, labels, mark


def _small_hp(stream=False):
    return {
        "feature_dim": 3,
        "hidden_dim": 5,
        "num_layers": 3,
        "num_negatives": 2,
        "triplets_per_domain": 12,
        "edge_chunk": 3,
        "node_chunk": 4,
        "stream_node_threshold": 1 if stream else 1_000_000,
        "stream_edge_threshold": 1 if stream else 1_000_000,
        "pretrain_lr": 1e-3,
        "prompt_lr": 1e-2,
        "cache": False,
        "record_run": False,
        "log_every": 100,
    }


class MDGPTRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.graphs = {
            "source-a": _raw_graph(1),
            "source-b": _raw_graph(2),
            "target-a": _raw_graph(3),
            "target-b": _raw_graph(4),
        }

    def _load(self, name):
        return tuple(value.copy() for value in self.graphs[name])

    def _run(self, targets=("target-a",), stream=False):
        captured = {"source": [], "prompt": [], "scores": {}}
        original_train = mdgpt._train_seed
        original_prompt = mdgpt._tune_prompt
        original_scores = mdgpt._scores

        def train_and_record(*args, **kwargs):
            model, history = original_train(*args, **kwargs)
            captured["source"].append(
                (
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    np.asarray(history),
                )
            )
            return model, history

        def prompt_and_record(*args, **kwargs):
            prompt, history = original_prompt(*args, **kwargs)
            captured["prompt"].append(
                (
                    args[1]["name"],
                    {
                        key: value.detach().cpu().clone()
                        for key, value in prompt.state_dict().items()
                    },
                    np.asarray(history),
                )
            )
            return prompt, history

        def score_and_record(model, prompt, graph, support, query, *args, **kwargs):
            scores = original_scores(model, prompt, graph, support, query, *args, **kwargs)
            captured["scores"][graph["name"]] = (
                support.copy(),
                query.copy(),
                scores.copy(),
            )
            return scores

        with tempfile.TemporaryDirectory() as directory, patch.object(
            mdgpt, "CACHE_ROOT", Path(directory)
        ), patch.object(
            mdgpt, "load_source_marked", side_effect=self._load
        ) as source_loader, patch.object(
            mdgpt, "load_target_marked", side_effect=self._load
        ) as target_loader, patch.object(
            mdgpt, "_train_seed", side_effect=train_and_record
        ), patch.object(
            mdgpt, "_tune_prompt", side_effect=prompt_and_record
        ), patch.object(
            mdgpt, "_scores", side_effect=score_and_record
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            result = mdgpt.run_mdgpt(
                ["source-a", "source-b"],
                list(targets),
                [7],
                2,
                "cpu",
                shot=2,
                hp=_small_hp(stream),
                prompt_epochs=2,
            )
            captured["source_loads"] = source_loader.call_count
            captured["target_loads"] = target_loader.call_count
            # No disk-backed activations may remain after target/model release.
            self.assertFalse(list(Path(directory).rglob("*.bin")))
        return result, captured

    def test_target_order_does_not_change_source_model_prompt_or_scores(self):
        forward, first = self._run(targets=("target-a", "target-b"))
        reverse, second = self._run(targets=("target-b", "target-a"))
        self.assertEqual(forward, reverse)
        for key, value in first["source"][0][0].items():
            torch.testing.assert_close(value, second["source"][0][0][key], atol=0, rtol=0)
        second_prompts = {name: (state, history) for name, state, history in second["prompt"]}
        for name, state, history in first["prompt"]:
            other_state, other_history = second_prompts[name]
            np.testing.assert_array_equal(history, other_history)
            for key, value in state.items():
                torch.testing.assert_close(value, other_state[key], atol=0, rtol=0)
            for left, right in zip(first["scores"][name], second["scores"][name]):
                np.testing.assert_array_equal(left, right)

    def test_forced_stream_matches_regular_training_adaptation_and_full_queries(self):
        ordinary, first = self._run(stream=False)
        streaming, second = self._run(stream=True)
        np.testing.assert_allclose(
            first["source"][0][1], second["source"][0][1], atol=3e-6, rtol=3e-5
        )
        for key, value in first["source"][0][0].items():
            torch.testing.assert_close(value, second["source"][0][0][key], atol=3e-6, rtol=3e-5)
        for (_, state, history), (_, other_state, other_history) in zip(
            first["prompt"], second["prompt"]
        ):
            np.testing.assert_allclose(history, other_history, atol=3e-6, rtol=3e-5)
            for key, value in state.items():
                torch.testing.assert_close(value, other_state[key], atol=3e-6, rtol=3e-5)
        for key in first["scores"]:
            support_a, query_a, score_a = first["scores"][key]
            support_b, query_b, score_b = second["scores"][key]
            np.testing.assert_array_equal(support_a, support_b)
            np.testing.assert_array_equal(query_a, query_b)
            np.testing.assert_allclose(score_a, score_b, atol=3e-6, rtol=3e-5)
        self.assertEqual(ordinary, streaming)

    def test_unknown_nodes_are_context_only_and_every_marked_query_is_scored(self):
        _, captured = self._run(stream=True)
        support, query, scores = captured["scores"]["target-a"]
        labels, mark = self.graphs["target-a"][2:]
        self.assertTrue(mark[support].all())
        self.assertTrue(mark[query].all())
        self.assertEqual(np.count_nonzero(labels[support] == 0), 2)
        self.assertEqual(np.count_nonzero(labels[support] == 1), 2)
        self.assertFalse(np.intersect1d(support, query).size)
        np.testing.assert_array_equal(np.union1d(support, query), np.flatnonzero(mark))
        self.assertEqual(len(query), int(mark.sum()) - len(np.unique(support)))
        self.assertEqual(len(scores), len(query))
        self.assertEqual(captured["source_loads"], 2)


if __name__ == "__main__":
    unittest.main()
