"""Target-evaluation hooks preserve raw scores, node order and seed identity."""

import contextlib
import importlib
import io
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp
import torch

from ggad import experiment as run

METHOD_MODULES = {
    "unprompt": "ggad.runners.unprompt",
    "anomalygfm_zs": "ggad.runners.anomalygfm",
    "drggad": "ggad.vendor.drggad.implementation",
    "gadmore": "ggad.runners.gadmore",
    "neighbordiv": "ggad.vendor.neighbordiv.implementation",
    "promos": "ggad.vendor.promos.implementation",
    "zerogad": "ggad.vendor.zerogad.implementation",
    "owleye": "ggad.vendor.owleye.implementation",
    "tpcagad": "ggad.vendor.tpcagad.implementation",
}


class _DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cluster_centers_dict = {}

    def forward(self, *args, **kwargs):
        return torch.ones(6, 2)

    def add(self, features):
        return features

    def move_memory(self, device):
        return self

    def anomaly_score(self, *args):
        return np.arange(6, dtype=np.float32) / 5

    def get_anomaly_score(self, graph):
        return torch.arange(6, dtype=torch.float32) / 5


class ZeroShotCallbackTests(unittest.TestCase):
    def test_dispatch_passes_callback_to_all_zero_shot_score_providers(self):
        callback = Mock()
        for method in METHOD_MODULES:
            runner_name = "anomalygfm" if method == "anomalygfm_zs" else method
            module = types.ModuleType("ggad.runners." + runner_name)
            function = Mock(return_value={})
            setattr(module, "run_" + runner_name, function)
            with self.subTest(method=method), patch.dict(sys.modules, {module.__name__: module}):
                run.run_method(method, [], [], "cpu", run.C.EPOCHS, [7], target_evaluator=callback)
            self.assertIs(function.call_args.kwargs["target_evaluator"], callback)

    def test_all_hooks_pass_full_arrays_per_seed_and_use_callback_metrics(self):
        for method, module_name in METHOD_MODULES.items():
            with self.subTest(method=method):
                module = importlib.import_module(module_name)
                self._check_hook(method, module)

    def _check_hook(self, method, module):
        labels = np.array([0, 0, 1, 1, 2, 0])
        mark = np.array([True, True, True, True, False, False])
        scores = np.arange(6, dtype=np.float32) / 5
        graph = SimpleNamespace(
            name="toy",
            n=6,
            labels=labels,
            labels_np=labels,
            mark=mark,
            feat=torch.ones(6, 2),
            x_list=[torch.ones(6, 2)],
            aws=torch.eye(6).to_sparse(),
            sim=0.2,
            adj=torch.eye(6).to_sparse(),
            adj_resid=torch.eye(6).to_sparse(),
        )
        replacements = {}
        runner_name = method
        args = ([], ["toy"], [3, 9], 1, "cpu")
        kwargs = {"train": False}
        if method == "unprompt":
            args = ([], ["toy"], [3, 9], "cpu", 1)
            replacements = {
                "Model": _DummyModel,
                "GPFplusAtt": _DummyModel,
                "Projection": _DummyModel,
                "_graph": Mock(return_value=graph),
                "completionsim": Mock(return_value=1 - scores),
                "normalize_score": lambda value: value,
            }
        elif method == "anomalygfm_zs":
            runner_name = "anomalygfm"
            graph.labels = torch.as_tensor(labels)
            args = ([], ["toy"], "none", [3, 9], "cpu", 1, 4)
            replacements = {
                "Model": _DummyModel,
                "_graph": Mock(return_value=graph),
                "residual_proto_score": Mock(return_value=torch.as_tensor(scores)),
            }
        elif method == "drggad":
            replacements = {
                "DR": _DummyModel,
                "_graph": Mock(return_value=graph),
                "_initialize_random_prototype": Mock(),
            }
        elif method == "gadmore":
            replacements = {
                "GADMoRE": _DummyModel,
                "_graph": Mock(return_value=graph),
                "_release_graph": Mock(),
            }
        elif method == "promos":
            replacements = {
                "RuntimeGCA": _DummyModel,
                "_init_model": _DummyModel,
                "_graph": Mock(return_value=graph),
                "embed_graph": Mock(return_value=torch.ones(6, 2)),
                "_release_teacher_inputs": Mock(),
                "_release_graph": Mock(),
                "_score_graph": Mock(return_value=scores),
            }
        elif method == "zerogad":
            replacements = {
                "PreModel": _DummyModel,
                "_graph": Mock(return_value=graph),
                "_release_graph": Mock(),
                "_score_graph": Mock(return_value=scores),
            }
        elif method == "owleye":
            replacements = {
                "_init_model": _DummyModel,
                "_load_graph": Mock(return_value=graph),
                "_normalize_feature_scale": Mock(),
                "_attach_runtime": Mock(),
                "_propagated": Mock(),
                "_release_runtime": Mock(),
                "_score_target": Mock(return_value=scores),
            }
        elif method == "tpcagad":
            replacements = {
                "_prepare_seed_models": Mock(return_value=[_DummyModel(), _DummyModel()]),
                "_build_graph": Mock(return_value=graph),
                "_score_target": Mock(return_value=scores),
            }
        elif method == "neighbordiv":
            hp = dict(module.C.NEIGHBORDIV_HP, cache=False)
            kwargs["hp"] = hp
            replacements = {
                "load_target_marked": Mock(return_value=(sp.eye(6), np.ones((6, 3)), labels, mark)),
                "_resolve_pair_mode": Mock(return_value="full"),
                "_load_or_project": Mock(return_value=np.ones((6, 3))),
                "_compute_score": Mock(return_value=scores),
            }

        seen = []

        def evaluator(name, seed, full_labels, full_scores, full_mark):
            self.assertEqual(name, "toy")
            self.assertIsInstance(seed, int)
            for value in (full_labels, full_scores, full_mark):
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.shape, (6,))
            np.testing.assert_array_equal(full_labels, labels)
            np.testing.assert_allclose(full_scores, scores, atol=1e-7)
            np.testing.assert_array_equal(full_mark, mark)
            seen.append(seed)
            return {"AUROC": seed / 10, "AUPRC": seed / 20}

        with contextlib.ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(
                patch.object(
                    module,
                    "evaluate",
                    side_effect=AssertionError("original metrics bypass required"),
                )
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = getattr(module, "run_" + runner_name)(
                *args, target_evaluator=evaluator, **kwargs
            )
        self.assertEqual(seen, [3, 9])
        self.assertEqual(result["toy"]["n"], 2)
        self.assertAlmostEqual(result["toy"]["AUROC_mean"], 0.6)
        self.assertAlmostEqual(result["toy"]["AUPRC_mean"], 0.3)
        if method == "neighbordiv":
            replacements["_compute_score"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
