"""Dependency-free checks for the public launcher, without loading graph data.

Run from the repository root with ``python -m unittest discover -s tests``.
Temporary nonempty files model dataset presence only; no model receives them.
"""

import builtins
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_launcher", ROOT / "run.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
MODEL_MODULES = {
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "torch_geometric",
    "einops",
    "geoopt",
    "pandas",
    "sympy",
    "xgboost",
    "tabpfn",
    "dgl",
    "numba",
}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gad-cli-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.data = self.directory / "Dataset"
        (self.data / "fake").mkdir(parents=True)
        (self.data / "real").mkdir()
        for name in ("cora", "citeseer", "ACM", "BlogCatalog"):
            (self.data / "fake" / f"{name}.mat").write_bytes(b"path-only fixture")
        self.output = self.directory / "new results"
        windows_environment = {
            key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ
        }
        self.environment = mock.patch.dict(os.environ, windows_environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        original_import = builtins.__import__

        def forbid_model_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in MODEL_MODULES:
                raise AssertionError(f"Launcher imported a model dependency: {name}")
            return original_import(name, *args, **kwargs)

        import_guard = mock.patch("builtins.__import__", side_effect=forbid_model_import)
        import_guard.start()
        self.addCleanup(import_guard.stop)

    def invoke(self, arguments, available=False):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with mock.patch.object(launcher, "_has_module", return_value=available):
                try:
                    code = launcher.main(arguments)
                except SystemExit as error:
                    code = error.code
        return code, stdout.getvalue() + stderr.getvalue()

    def arguments(self, suite="ggad", method="neighbordiv"):
        return [
            "--suite",
            suite,
            "--methods",
            method,
            "--src-type",
            "fake",
            "--tgt-type",
            "fake",
            "--source-mode",
            "single",
            "--device",
            "cpu",
            "--targets",
            "cora",
            "--data-root",
            str(self.data),
            "--output-dir",
            str(self.output),
        ]

    def test_help_and_method_discovery_work_without_site_packages(self):
        for suite, expected in (
            ("ggad", "iaggad_fs"),
            ("baselines", "rf_graph"),
            ("gfm", "graphprompt_enr"),
        ):
            with self.subTest(suite=suite):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        "-B",
                        str(ROOT / "run.py"),
                        "--suite",
                        suite,
                        "--list-methods",
                    ],
                    cwd=self.directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(expected, completed.stdout)
        code, output = self.invoke(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("--dry-run", output)

    def test_listing_does_not_require_data_or_check_dependencies(self):
        with mock.patch.object(launcher, "_check_datasets") as datasets:
            with mock.patch.object(launcher, "_dependency_report") as dependencies:
                code, output = self.invoke(["--list-methods", "--data-root", "missing"])
        self.assertEqual(code, 0, output)
        datasets.assert_not_called()
        dependencies.assert_not_called()

    def test_invalid_options_fail_before_importing_an_experiment(self):
        invalid = (
            ["--methods", "unknown_method"],
            ["--methods", "arc,,gadmore"],
            ["--methods", "arc,arc"],
            ["--device", "cuda:-1"],
            ["--device", "gpu"],
            ["--targets", "../outside"],
            ["--targets", ""],
            ["--source-mode", "invalid"],
            ["--seed", "-1"],
            ["--suite", "gfm", "--shot-sweep"],
        )
        with mock.patch.object(launcher.importlib, "import_module") as child:
            for extra in invalid:
                with self.subTest(options=extra):
                    code, output = self.invoke(self.arguments() + extra + ["--dry-run"])
                    self.assertEqual(code, 2, output)
            child.assert_not_called()

    def test_missing_and_empty_datasets_report_preparation_instructions(self):
        for filename in ("missing", "empty"):
            with self.subTest(dataset=filename):
                if filename == "empty":
                    (self.data / "fake" / "empty.mat").touch()
                code, output = self.invoke(self.arguments() + ["--targets", filename, "--dry-run"])
                self.assertEqual(code, 2)
                self.assertIn("missing or empty", output)
                self.assertIn("README.md", output)
                self.assertIn(str(self.data / "fake" / f"{filename}.mat"), output)

    def test_dry_run_reports_only_selected_dependencies_and_writes_nothing(self):
        with mock.patch.object(launcher.importlib, "import_module") as child:
            code, output = self.invoke(self.arguments() + ["--dry-run"])
        self.assertEqual(code, 0, output)
        self.assertIn("Missing dependencies", output)
        self.assertIn("pip install", output)
        self.assertNotIn("tabpfn", output.lower())
        self.assertIn(str(self.output / "source_domains_detail.csv"), output)
        self.assertFalse(self.output.exists())
        child.assert_not_called()

    def test_execution_blocks_missing_dependencies(self):
        with mock.patch.object(launcher.importlib, "import_module") as child:
            code, output = self.invoke(self.arguments())
        self.assertEqual(code, 2)
        self.assertIn("Missing dependencies", output)
        child.assert_not_called()

    def test_suite_routing_environment_and_seed_override(self):
        selections = (
            ("ggad", "neighbordiv", "ggad.experiment", "source_domains_detail.csv"),
            ("baselines", "rf_graph", "baselines.experiment", "source_compare_detail.csv"),
            ("gfm", "mdgpt", "gfm.experiment", "foundation_compare_detail.csv"),
        )
        for suite, method, module, output_name in selections:
            with self.subTest(suite=suite):
                experiment = mock.Mock()
                experiment.run.return_value = 17
                with mock.patch.object(
                    launcher.importlib, "import_module", return_value=experiment
                ) as load:
                    code, output = self.invoke(
                        self.arguments(suite, method) + ["--seed", "7"], available=True
                    )
                self.assertEqual(code, 17, output)
                load.assert_called_once_with(module)
                args = experiment.run.call_args.args[0]
                self.assertEqual(args.methods, method)
                self.assertEqual(args.targets, "cora")
                self.assertEqual(args.seed, 7)
                self.assertFalse(args.quick)
                self.assertEqual(os.environ["GAD_DATA_ROOT"], str(self.data))
                self.assertEqual(os.environ["GAD_OUTPUT_DIR"], str(self.output))
                self.assertIn(str(self.output / output_name), output)
                self.assertFalse(self.output.exists())

    def test_shot_sweep_uses_the_same_public_entry_and_path_only_dry_run(self):
        experiment = mock.Mock()
        experiment.run.return_value = None
        arguments = self.arguments(method="arc") + ["--shot-sweep", "--seed", "3"]
        with mock.patch.object(
            launcher.importlib, "import_module", return_value=experiment
        ) as load:
            code, output = self.invoke(arguments + ["--dry-run"])
            self.assertEqual(code, 0, output)
            load.assert_not_called()
            self.assertFalse(self.output.exists())
            code, output = self.invoke(arguments, available=True)
        self.assertEqual(code, 0, output)
        load.assert_called_once_with("ggad.shot_sweep")
        self.assertEqual(experiment.run.call_args.args[0].seed, 3)

    def test_environment_paths_are_used_and_cli_paths_take_precedence(self):
        environment_output = self.directory / "environment-output"
        with mock.patch.dict(
            os.environ,
            {
                "GAD_DATA_ROOT": str(self.data),
                "GAD_OUTPUT_DIR": str(environment_output),
            },
        ):
            code, output = self.invoke(
                ["--methods", "neighbordiv", "--targets", "cora", "--dry-run"]
            )
            self.assertEqual(code, 0, output)
            self.assertIn(str(environment_output / "source_domains_detail.csv"), output)
            code, output = self.invoke(self.arguments() + ["--dry-run"])
            self.assertEqual(code, 0, output)
            self.assertIn(str(self.output / "source_domains_detail.csv"), output)
            self.assertNotIn(str(environment_output), output)

    def test_quick_targets_and_explicit_override_match_suite_semantics(self):
        base = [
            "--methods",
            "neighbordiv",
            "--tgt-type",
            "fake",
            "--quick",
            "--data-root",
            str(self.data),
            "--output-dir",
            str(self.output),
            "--dry-run",
        ]
        code, output = self.invoke(base)
        self.assertEqual(code, 0, output)
        self.assertIn("Targets: cora, citeseer", output)
        code, output = self.invoke(base + ["--targets", "cora,citeseer,ACM"])
        self.assertEqual(code, 0, output)
        self.assertIn("Targets: cora, citeseer, ACM", output)
        # Quick mode must not silently remove configured source graphs.
        (self.data / "fake" / "BlogCatalog.mat").unlink()
        code, output = self.invoke(
            self.arguments(method="gadmore") + ["--source-mode", "multi", "--quick", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("BlogCatalog.mat", output)

    def test_output_file_collision_is_rejected(self):
        self.output.write_text("existing file", encoding="utf-8")
        code, output = self.invoke(self.arguments() + ["--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("blocked by an existing file", output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing file")

    def test_special_dataset_layouts_and_conditional_dgl_dependency(self):
        real = self.data / "real"
        (real / "dgraphfin.npz").write_bytes(b"path-only fixture")
        (real / "tsocial").write_bytes(b"path-only fixture")
        elliptic = real / "elliptic_bitcoin_dataset"
        elliptic.mkdir()
        for suffix in ("classes", "features", "edgelist"):
            (elliptic / f"elliptic_txs_{suffix}.csv").write_text("fixture", encoding="utf-8")
        selected = launcher._check_datasets(["dgraphfin", "elliptic", "tsocial"], self.data)
        required = launcher._required_modules("ggad", ["neighbordiv"], selected)
        self.assertIn("pandas", required)
        self.assertIn("dgl", required)
        (real / "tsocial.npz").write_bytes(b"path-only fixture")
        selected = launcher._check_datasets(["tsocial"], self.data)
        required = launcher._required_modules("ggad", ["neighbordiv"], selected)
        self.assertNotIn("dgl", required)

    def test_tpcagad_numba_note_does_not_block_small_graph_dry_run(self):
        code, output = self.invoke(self.arguments(method="tpcagad") + ["--dry-run"])
        self.assertEqual(code, 0, output)
        self.assertIn("optional numba", output)
        self.assertIn("2,000,000", output)
        self.assertIn("Small graphs do not require it", output)


if __name__ == "__main__":
    unittest.main()
