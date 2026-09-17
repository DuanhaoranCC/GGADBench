"""Target-shot routing and support/query isolation on tiny in-memory graphs."""

import contextlib
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from ggad import experiment as run
from ggad.runners import iaggad, refigad, taggad


class ShotDispatchTests(unittest.TestCase):
    def test_explicit_shot_reaches_every_few_shot_runner_without_global_mutation(self):
        routes = {
            "arc": ("arc", "run_arc", 5),
            "iaggad_fs": ("iaggad", "run_iaggad", 6),
            "anomalygfm_fs": ("anomalygfm", "run_anomalygfm", 7),
            "saarcs": ("saarcs", "run_saarcs", 5),
            "tfm4gad": ("tfm4gad", "run_tfm4gad", 5),
            "taggad": ("taggad", "run_taggad", 5),
            "refigad": ("refigad", "run_refigad", None),
        }
        original_shot = run.C.SHOT
        for method, (module_name, function_name, position) in routes.items():
            with self.subTest(method=method):
                module = types.ModuleType("ggad.runners." + module_name)
                function = Mock(return_value={"target": "result"})
                setattr(module, function_name, function)
                with patch.dict(sys.modules, {module.__name__: module}):
                    result = run.run_method(
                        method,
                        ["source"],
                        ["target"],
                        "cpu",
                        run.C.EPOCHS,
                        [7],
                        shot=3,
                        disjoint_query=True,
                    )
                self.assertEqual(result, {"target": "result"})
                args, kwargs = function.call_args
                if position is None:
                    self.assertEqual(kwargs["shot"], 3)
                    self.assertTrue(kwargs["exclude_support"])
                else:
                    self.assertEqual(args[position], 3)
                if method == "taggad":
                    self.assertEqual(kwargs["source_shot"], original_shot)
                self.assertEqual(run.C.SHOT, original_shot)

    def test_legacy_refigad_call_retains_dataset_k_and_original_query_protocol(self):
        with patch.object(refigad, "run_refigad", return_value={}) as runner:
            run.run_method("refigad", [], [], "cpu", run.C.EPOCHS, [0])
        self.assertIsNone(runner.call_args.kwargs["shot"])
        self.assertFalse(runner.call_args.kwargs["exclude_support"])

    def test_iaggad_zero_shot_uses_native_random_context_at_requested_shot(self):
        with patch.object(iaggad, "run_iaggad", return_value={}) as runner:
            run.run_method("iaggad_zs", [], [], "cpu", run.C.EPOCHS, [7], shot=3)
        self.assertEqual(runner.call_args.args[2], "random")
        self.assertEqual(runner.call_args.args[6], 3)
        self.assertNotIn("target_evaluator", runner.call_args.kwargs)


class _ToyIAGGAD(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vq = SimpleNamespace(codebook=torch.ones(2, 2))


class IAGGADRandomSupportTests(unittest.TestCase):
    def test_native_random_support_matches_independent_seed_shuffle(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 0])
        mark = np.array([True] * 6 + [False, False])
        graph = SimpleNamespace(n=len(labels), labels_np=labels, mark=mark)
        seeds = [0, 7]
        for shot in (1, 3):
            masks = []

            def score(model, affinity, target, mask, codebook, lam):
                masks.append(mask.cpu().numpy().copy())
                return torch.arange(target.n, dtype=torch.float32)[~mask]

            with self.subTest(shot=shot), patch.object(iaggad, "GCN", _ToyIAGGAD), patch.object(
                iaggad, "my_GCN", _ToyIAGGAD
            ), patch.object(iaggad, "_graph", return_value=graph), patch.object(
                iaggad, "_iaggad_score", side_effect=score
            ), patch.object(
                iaggad, "evaluate", return_value={"AUROC": 0.5, "AUPRC": 0.5}
            ) as metric:
                result = iaggad.run_iaggad(
                    [], ["toy"], "random", seeds, 1, "cpu", shot=shot, train=False
                )
            self.assertEqual(result["toy"]["n"], 2)
            for seed, mask, call in zip(seeds, masks, metric.call_args_list):
                pool = np.flatnonzero(mark)
                np.random.RandomState(seed).shuffle(pool)
                expected = np.zeros(len(labels), dtype=bool)
                expected[pool[:shot]] = True
                np.testing.assert_array_equal(mask, expected)
                np.testing.assert_array_equal(call.args[0], labels[mark & ~expected])
                np.testing.assert_array_equal(call.args[1], np.flatnonzero(mark & ~expected))


class _ToyREFIGAD(torch.nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self.k_shot = kwargs["k_shot"]
        self.calls = []
        self.instances.append(self)

    def forward(self, normal, anomaly, query):
        self.calls.append((self.training, normal.clone(), anomaly.clone(), query.clone()))
        return query[:, :1] + self.offset


class REFIGADShotTests(unittest.TestCase):
    def setUp(self):
        _ToyREFIGAD.instances.clear()
        self.features = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 5)
        self.labels = torch.tensor([0] * 4 + [1] * 4 + [2, 0])
        self.mark = np.array([True] * 8 + [False, False])
        self.hp = dict(refigad.C.REFIGAD_HP, batch_size=3, batches_per_dataset=1)

    def test_sampler_exposes_exact_indices_without_changing_sampling(self):
        valid = torch.as_tensor(self.mark)
        torch.manual_seed(7)
        original = refigad._get_intra_batch(self.features, self.labels, 2, 0, valid=valid)
        torch.manual_seed(7)
        indexed = refigad._get_intra_batch(
            self.features, self.labels, 2, 0, valid=valid, return_indices=True
        )
        self.assertEqual(len(original), 4)
        self.assertEqual(len(indexed), 6)
        for first, second in zip(original, indexed[:4]):
            torch.testing.assert_close(first, second)
        torch.testing.assert_close(indexed[0], self.features[indexed[4]])
        torch.testing.assert_close(indexed[1], self.features[indexed[5]])
        self.assertTrue(valid[indexed[4]].all())
        self.assertTrue(valid[indexed[5]].all())

    def _run(self, shot, exclude_support, train=False):
        def prep(name, *args, **kwargs):
            return name, self.features, self.labels, self.mark

        with patch.object(refigad, "_prep", side_effect=prep), patch.object(
            refigad, "PromptGADModel", _ToyREFIGAD
        ), patch.object(refigad.C, "REFIGAD_HP", self.hp), patch.object(
            refigad.C, "REFIGAD_K_DEFAULT", 2
        ), patch.object(
            refigad, "evaluate", return_value={"AUROC": 0.5, "AUPRC": 0.5}
        ) as metric:
            result = refigad.run_refigad(
                ["source"],
                ["target"],
                [7],
                "cpu",
                epochs=1,
                train=train,
                shot=shot,
                exclude_support=exclude_support,
            )
        return result, metric

    def test_disjoint_query_excludes_support_and_preserves_full_graph_scoring(self):
        result, metric = self._run(shot=2, exclude_support=True)
        model = _ToyREFIGAD.instances[-1]
        calls = model.calls
        support_ids = torch.cat((calls[0][1][:, 0], calls[0][2][:, 0])).long().numpy()
        scored_ids = torch.cat([entry[3][:, 0] for entry in calls]).long().numpy()
        expected_query = np.flatnonzero(self.mark & ~np.isin(np.arange(10), support_ids))
        np.testing.assert_array_equal(scored_ids, np.arange(10))
        np.testing.assert_array_equal(metric.call_args.args[0], self.labels.numpy()[expected_query])
        np.testing.assert_array_equal(metric.call_args.args[1], expected_query.astype(float))
        np.testing.assert_array_equal(self.mark, [True] * 8 + [False, False])
        self.assertEqual(result["target"]["n"], 1)

    def test_default_query_protocol_still_includes_all_marked_nodes(self):
        _, metric = self._run(shot=2, exclude_support=False)
        np.testing.assert_array_equal(metric.call_args.args[1], np.arange(8, dtype=float))

    def test_target_shot_does_not_change_source_training_support(self):
        self._run(shot=1, exclude_support=True, train=True)
        model = _ToyREFIGAD.instances[-1]
        self.assertEqual(model.k_shot, 2)
        self.assertEqual(len(model.calls[0][1]), 2)
        self.assertTrue(model.calls[0][0])
        for training, normal, anomaly, _ in model.calls[1:]:
            self.assertFalse(training)
            self.assertEqual(len(normal), 1)
            self.assertEqual(len(anomaly), 1)

    def test_query_without_both_classes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both normal and anomaly"):
            self._run(shot=4, exclude_support=True)


class _ToyTAGGAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))

    def forward(self, graph, other):
        embedding = graph.x * self.weight
        return embedding, self.weight.square().view(1), embedding, embedding, embedding[:2]

    def get_train_loss(self, *args):
        return self.weight.square()

    def get_test_score(self, residual, codebook, mask, labels):
        return residual[~mask, 0]


class TAGGADSourceShotTests(unittest.TestCase):
    def _trial(self, target_shot, source_shot=None):
        labels = np.array([0] * 8 + [1] * 4)
        features = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        graph = SimpleNamespace(
            name="source",
            n=12,
            labels_np=labels,
            mark=np.ones(12, dtype=bool),
            x=features,
            stream=False,
            ano_labels=torch.as_tensor(labels),
            local_edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        hp = dict(taggad.C.TAGGAD_HP, shot=target_shot)
        if source_shot is not None:
            hp["source_shot"] = source_shot

        def neighbor_scores(edges, values, **kwargs):
            scores = values[:, 0]
            return scores, scores, scores, values.square().mean()

        with patch.object(
            taggad, "_new_model", side_effect=lambda *args: _ToyTAGGAD()
        ), patch.object(
            taggad, "combined_neighbor_scores", side_effect=neighbor_scores
        ), patch.object(
            taggad, "compute_kde_distribution", return_value=(np.arange(2), np.arange(2))
        ), patch.object(
            taggad, "_empty_cuda_cache"
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            trial = taggad._train_trial(
                7,
                [graph],
                {"target": (labels, np.ones(12, dtype=bool))},
                1,
                hp,
                "cpu",
            )
        return trial, graph.shot_mask

    def test_target_shot_changes_support_while_source_mask_and_model_stay_fixed(self):
        first, first_mask = self._trial(1, source_shot=3)
        second, second_mask = self._trial(5, source_shot=3)
        self.assertEqual(len(first.target_support_indices["target"]), 1)
        self.assertEqual(len(second.target_support_indices["target"]), 5)
        self.assertEqual(int(first_mask.sum()), 3)
        torch.testing.assert_close(first_mask, second_mask, atol=0, rtol=0)
        torch.testing.assert_close(first.model.weight, second.model.weight, atol=0, rtol=0)
        torch.testing.assert_close(first.final_codebook, second.final_codebook, atol=0, rtol=0)

    def test_default_source_support_retains_target_shot(self):
        _, mask = self._trial(4)
        self.assertEqual(int(mask.sum()), 4)


if __name__ == "__main__":
    unittest.main()
