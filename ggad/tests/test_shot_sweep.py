"""Shot budget, comparable reporting and resumable driver; no model training."""

import csv
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ggad.config as C
import ggad.shot_sweep as sweep

METHODS = [
    "arc",
    "iaggad_fs",
    "anomalygfm_fs",
    "refigad",
    "saarcs",
    "taggad",
    "tfm4gad",
    "unprompt",
    "drggad",
    "gadmore",
    "neighbordiv",
    "promos",
    "zerogad",
    "owleye",
    "tpcagad",
    "iaggad_zs",
    "anomalygfm_zs",
]


def _settings(**overrides):
    settings = dict(
        methods=["refigad"],
        shots=[1, 3],
        seeds=[2, 7],
        modes=["multi"],
        targets=["a", "b"],
        quick=False,
        stem="isolated_shot",
        epochs={"refigad": 2},
        device="cpu",
        src_type="real",
        tgt_type="real",
    )
    settings.update(overrides)
    return settings


def _plan(
    method="refigad",
    context_id="context-a",
    targets=None,
    counts=None,
    training_targets=None,
    random_budgets=None,
):
    targets = ["a", "b"] if targets is None else targets
    counts = counts or {target: (100, 50) for target in targets}
    if random_budgets is None and method in sweep.RANDOM_METHODS:
        random_budgets = {
            target: {
                (shot, seed): (
                    dict(
                        support_normal=shot,
                        support_anomaly=0,
                        query_normal=counts[target][0] - shot,
                        query_anomaly=counts[target][1],
                    ),
                    "",
                )
                for shot in (1, 3)
                for seed in (2, 7)
            }
            for target in targets
        }
    return dict(
        context_id=context_id,
        context={"training_targets": (targets if training_targets is None else training_targets)},
        method=method,
        mode="multi",
        sources=["source"],
        targets=targets,
        counts=counts,
        random_budgets=random_budgets or {},
    )


def _paths(directory):
    root = Path(directory)
    paths = {name: root / ("isolated_" + name + ".csv") for name in ("seeds", "detail", "summary")}
    paths["contexts"] = root / "isolated_contexts.json"
    return paths


def _metrics(auc=0.8, ap=0.6, n=1):
    return {"AUROC_mean": auc, "AUPRC_mean": ap, "n": n}


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ShotConfigurationTests(unittest.TestCase):
    def test_public_cli_overrides_keep_the_full_training_schedule(self):
        args = Namespace(
            methods="arc,refigad",
            src_type="real",
            tgt_type="fake",
            source_mode="single",
            targets="cora,citeseer",
            quick=False,
            device="cpu",
            seed=0,
        )
        settings = sweep._settings(args)
        self.assertEqual(settings["methods"], ["arc", "refigad"])
        self.assertEqual(settings["targets"], ["cora", "citeseer"])
        self.assertEqual(settings["modes"], ["single"])
        self.assertEqual(settings["seeds"], [0])
        self.assertEqual(settings["shots"], sorted(C.SHOT_SWEEP_SHOTS))
        self.assertEqual(settings["epochs"], C.SHOT_SWEEP_EPOCHS)

    def test_default_seventeen_ggad_protocol_paths_include_existing_zero_shot_aliases(self):
        self.assertEqual(set(C.SHOT_SWEEP_METHODS), set(METHODS))
        self.assertEqual(len(C.SHOT_SWEEP_METHODS), 17)
        self.assertIn("drggad", METHODS)
        self.assertIn("tfm4gad", METHODS)
        for method in ("iaggad_zs", "anomalygfm_zs"):
            with self.subTest(method=method), patch.object(C, "SHOT_SWEEP_METHODS", [method]):
                self.assertEqual(sweep._settings()["methods"], [method])
        for method in ("mdgpt", "mdgfm"):
            with self.subTest(method=method), patch.object(C, "SHOT_SWEEP_METHODS", [method]):
                with self.assertRaises(ValueError):
                    sweep._settings()

    def test_invalid_budgets_duplicate_seeds_and_paths_fail_before_training(self):
        for name, value in [
            ("SHOT_SWEEP_SHOTS", []),
            ("SHOT_SWEEP_SHOTS", [1, 1]),
            ("SHOT_SWEEP_SHOTS", [0]),
            ("SHOT_SWEEP_SHOTS", [-1]),
            ("SHOT_SWEEP_SHOTS", [True]),
            ("SHOT_SWEEP_SHOTS", [1.5]),
            ("SHOT_SWEEP_SEEDS", [3, 3]),
            ("SHOT_SWEEP_SEEDS", [-1]),
            ("SHOT_SWEEP_RESULTS_STEM", "../source_compare_detail1"),
            ("SHOT_SWEEP_RESULTS_STEM", "source_compare"),
            ("SHOT_SWEEP_SOURCE_MODE", "invalid"),
            ("SHOT_SWEEP_TARGETS", ["a", "a"]),
        ]:
            with self.subTest(name=name, value=value), patch.object(C, name, value):
                with self.assertRaises(ValueError):
                    sweep._settings()

    def test_quick_uses_configured_first_seed_and_keeps_requested_shots(self):
        with patch.multiple(
            C,
            SHOT_SWEEP_METHODS=["arc"],
            SHOT_SWEEP_SHOTS=[10, 1, 5],
            SHOT_SWEEP_SEEDS=[7, 2],
            SHOT_SWEEP_QUICK=True,
            SHOT_SWEEP_SOURCE_MODE="both",
        ):
            settings = sweep._settings()
        self.assertEqual(settings["seeds"], [7])
        self.assertEqual(settings["shots"], [1, 5, 10])
        self.assertEqual(settings["modes"], ["single", "multi"])
        self.assertEqual(settings["epochs"], C.QUICK_EPOCHS)

    def test_distinct_support_budget_never_exhausts_either_query_class(self):
        for method in set(METHODS) - sweep.RANDOM_METHODS:
            with self.subTest(method=method):
                budget, reason = sweep._budget(method, 5, (10, 6))
                self.assertFalse(reason)
                self.assertEqual(budget["support_normal"], 5)
                self.assertEqual(budget["query_normal"], 5)
                self.assertEqual(
                    budget["support_anomaly"], 5 if sweep.SUPPORT[method] == "per_class" else 0
                )
                self.assertEqual(
                    budget["query_anomaly"], 1 if sweep.SUPPORT[method] == "per_class" else 6
                )
                self.assertTrue(sweep._budget(method, 10, (10, 6))[1])
                self.assertTrue(sweep._budget(method, 1, (10, 0))[1])
        for method in ("refigad", "tfm4gad"):
            # Disney-like anomaly scarcity must skip 10-shot, not fabricate
            # ten distinct labels by sampling the same six nodes repeatedly.
            self.assertTrue(sweep._budget(method, 10, (100, 6))[1])
            row = sweep._base_row(_plan(method, counts={"a": (100, 6)}), _settings(), 10, 2, "a")
            self.assertEqual(row["status"], "skipped")
            for field in ("support_normal", "support_anomaly", "query_normal", "query_anomaly"):
                self.assertEqual(row[field], "")

    def test_random_holdout_does_not_require_k_nodes_per_class(self):
        for method in sweep.RANDOM_METHODS:
            with self.subTest(method=method):
                budget, reason = sweep._budget(method, 10, (118, 6))
                self.assertFalse(reason)
                self.assertTrue(all(value == "" for value in budget.values()))
                self.assertTrue(sweep._budget(method, 124, (118, 6))[1])

    def test_random_budgets_record_actual_seed_specific_counts_and_missing_query_class(self):
        labels = np.array([0, 0, 1, -1, 1])
        mark = np.array([True, True, True, False, False])
        with patch("ggad.runners.taggad._target_label_metadata", return_value=(labels, mark)):
            actual = sweep._random_budgets("toy", [1, 2, 3], [0, 1])
        # Seed 0 holds out node 2, the only marked anomaly. Seed 1 holds
        # out node 0, leaving both classes. Unknown/unmarked nodes never enter.
        self.assertEqual(
            actual[(1, 0)][0],
            dict(support_normal=0, support_anomaly=1, query_normal=2, query_anomaly=0),
        )
        self.assertIn("query lacks both classes", actual[(1, 0)][1])
        self.assertEqual(
            actual[(1, 1)][0],
            dict(support_normal=1, support_anomaly=0, query_normal=1, query_anomaly=1),
        )
        self.assertEqual(actual[(1, 1)][1], "")
        self.assertTrue(actual[(3, 0)][1])
        plan = _plan(
            method="drggad", targets=["toy"], counts={"toy": (2, 1)}, random_budgets={"toy": actual}
        )
        skipped = sweep._base_row(plan, _settings(), 1, 0, "toy")
        pending = sweep._base_row(plan, _settings(), 1, 1, "toy")
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(pending["support_normal"], 1)
        self.assertEqual(pending["support_anomaly"], 0)
        self.assertEqual(pending["protocol"], "zero_shot_query_holdout")

    def test_label_count_uses_only_marked_binary_nodes(self):
        labels = np.array([0, 1, -1, 0, 1, 99])
        mark = np.array([True, True, False, True, False, False])
        with patch("ggad.runners.taggad._target_label_metadata", return_value=(labels, mark)):
            self.assertEqual(sweep._label_counts("mock"), (2, 1))
        mark[-1] = True
        with patch("ggad.runners.taggad._target_label_metadata", return_value=(labels, mark)):
            with self.assertRaisesRegex(ValueError, "marked labels"):
                sweep._label_counts("mock")

    def test_resume_context_changes_for_model_settings_not_added_shots(self):
        settings = _settings()
        with patch.object(C, "sources", return_value=["source"]), patch.object(
            sweep, "_label_counts", return_value=(100, 50)
        ), patch.object(sweep, "_code_digest", return_value="fixed-code"):
            baseline = sweep._plans(settings)[0]
            expanded = sweep._plans(_settings(shots=[1, 3, 10]))[0]
            changed = sweep._plans(_settings(epochs={"refigad": 3}))[0]
            with patch.object(C, "REFIGAD_HP", dict(C.REFIGAD_HP, d_model=999)):
                changed_hp = sweep._plans(settings)[0]
        self.assertEqual(baseline["context_id"], expanded["context_id"])
        self.assertNotEqual(baseline["context_id"], changed["context_id"])
        self.assertNotEqual(baseline["context_id"], changed_hp["context_id"])

    def test_quick_preflight_only_reads_selected_targets(self):
        settings = _settings(
            methods=["drggad"], targets=["source", "a", "b", "missing_large"], quick=True
        )
        with patch.object(C, "sources", return_value=["source"]), patch.object(
            C, "SHOT_SWEEP_EXCLUDE_SOURCES", True
        ), patch.object(sweep, "_label_counts", return_value=(100, 50)) as count, patch.object(
            sweep, "_random_budgets", return_value={}
        ) as budget, patch.object(
            sweep, "_code_digest", return_value="fixed-code"
        ):
            plans = sweep._plans(settings)
        self.assertEqual(plans[0]["targets"], ["a", "b"])
        self.assertEqual([call.args[0] for call in count.call_args_list], ["a", "b"])
        self.assertEqual([call.args[0] for call in budget.call_args_list], ["a", "b"])

    def test_tag_expanded_shots_change_resume_context_if_training_cohort_changes(self):
        counts = {"a": (100, 50), "b": (4, 2)}
        with patch.object(C, "sources", return_value=["source"]), patch.object(
            sweep, "_label_counts", side_effect=counts.__getitem__
        ), patch.object(sweep, "_code_digest", return_value="fixed-code"):
            baseline = sweep._plans(_settings(methods=["taggad"], shots=[1, 3]))[0]
            expanded = sweep._plans(_settings(methods=["taggad"], shots=[1, 3, 4]))[0]
        self.assertEqual(baseline["context"]["training_targets"], ["a", "b"])
        self.assertEqual(expanded["context"]["training_targets"], ["a"])
        self.assertNotEqual(baseline["context_id"], expanded["context_id"])


class ShotReportTests(unittest.TestCase):
    def _rows(self, plan, settings):
        rows = {}
        for shot in settings["shots"]:
            for seed in settings["seeds"]:
                for target in plan["targets"]:
                    row = sweep._base_row(plan, settings, shot, seed, target)
                    row.update(status="ok", AUROC=0.8, AUPRC=0.6)
                    rows[sweep._key(row)] = row
        return rows

    def test_macro_uses_same_complete_target_cohort_for_all_shots_and_seeds(self):
        settings, plan = _settings(), _plan(targets=["a", "b", "c"])
        rows = self._rows(plan, settings)
        rows[(plan["context_id"], 3, 7, "b")].update(status="failed", AUROC="", AUPRC="")
        del rows[(plan["context_id"], 1, 2, "c")]
        for row in rows.values():
            if row["target"] == "a":
                row.update(AUROC=0.2, AUPRC=0.1)
        detail, summary = sweep._reports(rows, [plan], settings)
        for row in summary:
            self.assertEqual(json.loads(row["targets_json"]), ["a"])
            self.assertEqual(row["n_targets"], 1)
            self.assertAlmostEqual(row["AUROC_mean"], 0.2)
        incomplete = next(r for r in detail if r["shot"] == 3 and r["target"] == "b")
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(incomplete["n"], 1)
        self.assertEqual(incomplete["expected_n"], 2)

    def test_macro_std_is_over_seed_means_not_all_target_measurements(self):
        settings, plan = _settings(), _plan()
        rows = self._rows(plan, settings)
        for shot in settings["shots"]:
            for seed in settings["seeds"]:
                for target in plan["targets"]:
                    auc = {(2, "a"): 0.1, (2, "b"): 0.9, (7, "a"): 0.3, (7, "b"): 0.7}[
                        (seed, target)
                    ]
                    rows[(plan["context_id"], shot, seed, target)].update(AUROC=auc, AUPRC=auc)
        _, summary = sweep._reports(rows, [plan], settings)
        for row in summary:
            self.assertAlmostEqual(row["AUROC_mean"], 0.5)
            self.assertAlmostEqual(row["AUROC_std"], 0.0)

    def test_no_common_targets_does_not_report_a_misleading_macro_score(self):
        settings, plan = _settings(), _plan()
        _, summary = sweep._reports({}, [plan], settings)
        for row in summary:
            self.assertEqual(row["n_targets"], 0)
            self.assertEqual(row["n_seeds"], 0)
            self.assertEqual(row["AUROC_mean"], "")


class ShotDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = _paths(self.temporary.name)
        self.stack_patches = [
            patch.dict(os.environ),
            patch.object(C, "SHOT_SWEEP_RESUME", True),
            redirect_stdout(io.StringIO()),
        ]
        for manager in self.stack_patches:
            manager.__enter__()
            self.addCleanup(manager.__exit__, None, None, None)

    def test_every_method_receives_explicit_shot_one_seed_and_disjoint_query(self):
        settings = _settings(methods=METHODS)
        plans = [_plan(method=m, context_id=m) for m in METHODS]
        with patch.object(
            sweep, "run_method", return_value={"a": _metrics(), "b": _metrics()}
        ) as run:
            rows = sweep._run(settings, plans, self.paths)
        self.assertEqual(run.call_count, len(METHODS) * 2 * 2)
        calls = {
            (call.args[0], call.kwargs["shot"], tuple(call.args[5])) for call in run.call_args_list
        }
        self.assertEqual(
            calls,
            {
                (m, k, (seed,))
                for m in METHODS
                for k in settings["shots"]
                for seed in settings["seeds"]
            },
        )
        for call in run.call_args_list:
            self.assertEqual(call.args[1:3], (["source"], ["a", "b"]))
            self.assertEqual(call.args[4], settings["epochs"])
            self.assertIs(call.kwargs["disjoint_query"], True)
            if call.args[0] in sweep.HOLDOUT_METHODS:
                from ggad.shot_query import QueryHoldout

                self.assertIsInstance(call.kwargs["target_evaluator"], QueryHoldout)
                self.assertEqual(call.kwargs["target_evaluator"].shot, call.kwargs["shot"])
            else:
                self.assertNotIn("target_evaluator", call.kwargs)
        self.assertTrue(all(row["status"] == "ok" for row in rows.values()))
        self.assertTrue(all(path.exists() for path in self.paths.values()))

    def test_resume_skips_complete_groups_but_never_reuses_changed_context(self):
        settings, plan = _settings(shots=[1], seeds=[2]), _plan()
        with patch.object(
            sweep, "run_method", return_value={"a": _metrics(), "b": _metrics()}
        ) as run:
            sweep._run(settings, [plan], self.paths)
            sweep._run(settings, [plan], self.paths)
            self.assertEqual(run.call_count, 1)
            changed = _plan(context_id="changed-hyperparameters")
            rows = sweep._run(settings, [changed], self.paths)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["context_id"] for row in _read(self.paths["seeds"])},
            {plan["context_id"], changed["context_id"]},
        )

    def test_partial_retry_calls_full_target_list_and_retains_prior_success(self):
        settings, plan = _settings(shots=[1], seeds=[2]), _plan(method="taggad")
        with patch.object(sweep, "run_method", return_value={"a": _metrics(0.9, 0.8)}) as run:
            with self.assertRaisesRegex(RuntimeError, "Incomplete runner output"):
                sweep._run(settings, [plan], self.paths)
        saved = {row["target"]: row for row in _read(self.paths["seeds"])}
        self.assertEqual(saved["a"]["status"], "ok")
        self.assertEqual(saved["b"]["status"], "failed")
        # Target a is intentionally omitted on retry. Its successful result
        # must survive while the original complete target RNG order is used.
        with patch.object(sweep, "run_method", return_value={"b": _metrics()}) as run:
            rows = sweep._run(settings, [plan], self.paths)
        self.assertEqual(run.call_args.args[2], ["a", "b"])
        self.assertEqual(rows[(plan["context_id"], 1, 2, "a")]["AUROC"], "0.9")
        self.assertTrue(all(row["status"] == "ok" for row in rows.values()))

    def test_keyboard_interrupt_persists_checkpoint_and_resumes_missing_seed(self):
        settings, plan = _settings(shots=[1]), _plan()
        with patch.object(
            sweep,
            "run_method",
            side_effect=[{"a": _metrics(), "b": _metrics()}, KeyboardInterrupt()],
        ):
            with self.assertRaises(KeyboardInterrupt):
                sweep._run(settings, [plan], self.paths)
        saved = _read(self.paths["seeds"])
        self.assertTrue(all(row["status"] == "ok" for row in saved if row["seed"] == "2"))
        self.assertTrue(all(row["status"] == "pending" for row in saved if row["seed"] == "7"))
        with patch.object(
            sweep, "run_method", return_value={"a": _metrics(), "b": _metrics()}
        ) as run:
            sweep._run(settings, [plan], self.paths)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[5], [7])

    def test_tag_fixed_cohort_keeps_target_order_identical_across_shots(self):
        settings = _settings(shots=[1, 3], seeds=[2])
        plan = _plan(method="taggad", counts={"a": (100, 5), "b": (3, 5)}, training_targets=["a"])
        with patch.object(sweep, "run_method", return_value={"a": _metrics()}) as run:
            rows = sweep._run(settings, [plan], self.paths)
        self.assertEqual([call.args[2] for call in run.call_args_list], [["a"], ["a"]])
        for shot in settings["shots"]:
            self.assertEqual(rows[(plan["context_id"], shot, 2, "b")]["status"], "skipped")

    def test_random_query_failure_skips_only_affected_seed_target(self):
        settings = _settings(methods=["drggad"], shots=[1], seeds=[0, 1], targets=["toy"])
        budgets = {
            (1, 0): (
                dict(support_normal=0, support_anomaly=1, query_normal=2, query_anomaly=0),
                "query lacks both classes",
            ),
            (1, 1): (
                dict(support_normal=1, support_anomaly=0, query_normal=1, query_anomaly=1),
                "",
            ),
        }
        plan = _plan(
            method="drggad",
            targets=["toy"],
            counts={"toy": (2, 1)},
            random_budgets={"toy": budgets},
        )
        with patch.object(sweep, "run_method", return_value={"toy": _metrics()}) as run:
            rows = sweep._run(settings, [plan], self.paths)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[5], [1])
        self.assertEqual(rows[(plan["context_id"], 1, 0, "toy")]["status"], "skipped")
        self.assertEqual(rows[(plan["context_id"], 1, 1, "toy")]["status"], "ok")

    def test_nonfinite_and_partial_seed_metrics_fail_and_are_retryable(self):
        settings, plan = _settings(shots=[1], seeds=[2]), _plan()
        with patch.object(
            sweep, "run_method", return_value={"a": _metrics(auc=float("nan")), "b": _metrics(n=2)}
        ):
            with self.assertRaisesRegex(RuntimeError, "Incomplete runner output"):
                sweep._run(settings, [plan], self.paths)
        self.assertTrue(all(row["status"] == "failed" for row in _read(self.paths["seeds"])))
        with patch.object(
            sweep, "run_method", return_value={"a": _metrics(), "b": _metrics()}
        ) as run:
            rows = sweep._run(settings, [plan], self.paths)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(all(row["status"] == "ok" for row in rows.values()))


if __name__ == "__main__":
    unittest.main()
