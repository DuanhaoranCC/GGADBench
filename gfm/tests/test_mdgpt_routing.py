"""Foundation integration and dispatch checks without model training."""

import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import gfm.config as C
import gfm.experiment as entry
from gfm.vendor.mdgpt import implementation, preprocessing


class MDGPTFoundationRoutingTests(unittest.TestCase):
    def test_registered_foundation_model_and_local_implementation_paths(self):
        root = Path(entry.__file__).resolve().parents[1]
        self.assertEqual(C.METHOD_PROTOCOL["mdgpt"], "few_shot")
        self.assertEqual(
            C.FEATURE_CONFIGS["mdgpt"], {"dim": C.MDGPT_HP["feature_dim"], "norm": "none"}
        )
        self.assertIs(C.FOUNDATION_HP["mdgpt"], C.MDGPT_HP)
        self.assertEqual(implementation.CACHE_ROOT, root / "gfm/cache")
        self.assertEqual(preprocessing.CACHE_ROOT, root / "gfm/cache")
        for name in (
            "gfm.runners.mdgpt",
            "gfm.vendor.mdgpt.model",
            "gfm.vendor.mdgpt.streaming",
        ):
            module = importlib.import_module(name)
            self.assertIn(root / "gfm", Path(module.__file__).resolve().parents)

    def test_regular_and_quick_dispatch_native_training_budgets_and_shot(self):
        sentinel = {"target": {"AUROC_mean": 0.7}}
        for quick, epochs, prompt_epochs in ((False, 200, 100), (True, 2, 2)):
            with self.subTest(quick=quick), patch.object(C, "SHOT", 3), patch.object(
                C, "MDGPT_EPOCHS", 200
            ), patch.object(C, "MDGPT_PROMPT_EPOCHS", 100), patch.object(
                C, "MDGPT_QUICK_EPOCHS", 2
            ), patch.object(
                C, "MDGPT_QUICK_PROMPT_EPOCHS", 2
            ), patch(
                "gfm.runners.mdgpt.run_mdgpt", return_value=sentinel
            ) as runner:
                result = entry.run_method("mdgpt", ["source"], ["target"], "cpu", [7], quick)
            self.assertIs(result, sentinel)
            self.assertEqual(runner.call_args.args, (["source"], ["target"], [7], epochs, "cpu"))
            self.assertEqual(runner.call_args.kwargs["shot"], 3)
            self.assertEqual(runner.call_args.kwargs["prompt_epochs"], prompt_epochs)
            self.assertEqual(
                runner.call_args.kwargs["hp"]["feature_dim"], C.MDGPT_HP["feature_dim"]
            )
            self.assertNotIn("feature_norm", runner.call_args.kwargs)

    def test_native_width_is_configurable_without_extra_foundation_normalization(self):
        with patch.object(C, "MDGPT_HP", dict(C.MDGPT_HP, feature_dim=4)), patch(
            "gfm.runners.mdgpt.run_mdgpt", return_value={}
        ) as runner:
            entry.run_method("mdgpt", ["source"], ["target"], "cpu", [0], False)
            self.assertEqual(runner.call_args.kwargs["hp"]["feature_dim"], 4)
            entry.run_method("mdgpt", ["source"], ["target"], "cpu", [0], False, feature_dim=8)
            self.assertEqual(runner.call_args.kwargs["hp"]["feature_dim"], 8)
            self.assertEqual(C.MDGPT_HP["feature_dim"], 4)
            with self.assertRaisesRegex(ValueError, "without feature normalization"):
                entry.run_method(
                    "mdgpt", ["source"], ["target"], "cpu", [0], False, feature_norm="zscore"
                )


if __name__ == "__main__":
    unittest.main()
