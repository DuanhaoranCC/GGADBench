"""Evaluation selection across all public benchmark suites."""
import csv
import importlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import util
from evaluation import metric_scope, metric_names, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_recall_only_skips_existing_metrics_and_applies_query_mask(self):
        with metric_scope("recall-at-k"), \
             patch.object(util, "roc_auc_score", side_effect=AssertionError("AUROC")), \
             patch.object(util, "average_precision_score", side_effect=AssertionError("AUPRC")):
            result = util.evaluate([0, 1, 1, -1], [.8, .7, 100., 200.],
                                   np.array([True, True, False, False]))
            self.assertEqual(result, {"Rec@K": 0.})
            self.assertEqual(util.aggregate({"target": [result]})["target"]["n"], 1)
        self.assertEqual(metric_names(), ("AUROC", "AUPRC"))

    def test_stable_ties_and_empty_anomaly_set(self):
        self.assertEqual(recall_at_k([0, 1, 1, 0], [1., 1., 1., 1.]), .5)
        self.assertTrue(np.isnan(recall_at_k([0, 0], [1., 2.])))

    def test_every_suite_writes_only_selected_metric_without_touching_old_csv(self):
        for suite, method, stem in (("ggad", "arc", "detail"),
                                    ("gfm", "mdgpt", "foundation_compare_detail"),
                                    ("baselines", "rf_graph", "source_compare_detail")):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                entry = importlib.import_module(suite + ".experiment")
                folder = Path(directory)
                old = folder / (stem + ".csv")
                old.write_text("existing results", encoding="utf-8")
                args = SimpleNamespace(src_type="real", tgt_type="real", source_mode="single",
                                       methods=method, targets="Disney", device="cpu", quick=True,
                                       seed=0)
                def fake_run(*args, **kwargs):
                    self.assertEqual(metric_names(), ("Rec@K",))
                    return util.aggregate({"Disney": [util.evaluate([0, 1], [.1, .9])]})
                if suite == "ggad":
                    config_patch = patch.multiple(entry.C, RESULTS_DETAIL_FILE="detail.csv", RESULTS_DELTA_FILE="delta.csv")
                else:
                    config_patch = patch.dict({}, {})
                config = entry.BC if suite == "baselines" else entry.C
                with patch.object(entry, "RESULTS", folder), config_patch, \
                     patch.object(config, "EVAL_METRICS", "recall-at-k"), \
                     patch.object(entry, "run_baseline" if suite == "baselines" else "run_method", side_effect=fake_run), \
                     redirect_stdout(io.StringIO()):
                    entry.run(args)
                with (folder / (stem + "_recall_at_k.csv")).open(encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(float(row["Rec@K_mean"]), 1.)
                self.assertNotIn("AUROC_mean", row)
                self.assertEqual(old.read_text(encoding="utf-8"), "existing results")


if __name__ == "__main__":
    unittest.main()
