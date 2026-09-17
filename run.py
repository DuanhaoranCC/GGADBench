"""Launch a benchmark suite with readable preflight checks.

Help, method discovery, and dry runs use only Python's standard library. A dry
run checks options and required file paths, reports missing dependencies, and
prints the exact command without importing model libraries or loading graphs.
Actual runs call the selected suite directly through this single entry point. This launcher never installs packages or downloads datasets.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = {
    "ggad": ("ggad/config.py", "ggad/experiment.py"),
    "baselines": ("baselines/config.py", "baselines/experiment.py"),
    "gfm": ("gfm/config.py", "gfm/experiment.py"),
}
SOURCE_INDEPENDENT = {"neighbordiv", "tfm4gad"}
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "einops": "einops",
    "geoopt": "geoopt",
    "pandas": "pandas",
    "sympy": "sympy",
    "torch_geometric": "torch-geometric",
    "xgboost": "xgboost",
    "tabpfn": "tabpfn",
    "dgl": "dgl",
}


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing {name} declaration in {path}")


def _configuration(suite):
    config_path, entry_path = (ROOT / item for item in SUITES[suite])
    # These run-control files contain standard-library imports and constants.
    config = runpy.run_path(str(config_path))
    if suite == "ggad":
        methods = _literal_assignment(entry_path, "SUPPORTED_PROTOCOLS")
    elif suite == "baselines":
        methods = {name: "baseline" for name in config["FEATURE_CONFIGS"]}
    else:
        methods = config["METHOD_PROTOCOL"]
    return config, methods


def _comma_list(value, label):
    items = [item.strip() for item in value.split(",")]
    if not all(items):
        raise ValueError(f"{label} must be a nonempty comma-separated list")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} contains duplicate names")
    return items


def _display_command(command):
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _dataset_paths(name, data_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid dataset name {name!r}; use a dataset name, not a path")
    real = data_root / "real"
    if name == "elliptic":
        folder = real / "elliptic_bitcoin_dataset"
        return [
            [folder / f"elliptic_txs_{part}.csv"] for part in ("classes", "features", "edgelist")
        ]
    if name == "dgraphfin":
        return [[real / "dgraphfin.npz"]]
    if name == "tsocial":
        return [[real / "tsocial.npz", real / "tsocial"]]
    # Match the shared loader's real-first search order.
    return [[data_root / domain / f"{name}.mat" for domain in ("real", "fake")]]


def _check_datasets(names, data_root):
    missing, selected = [], {}
    for name in dict.fromkeys(names):
        selected[name] = []
        for alternatives in _dataset_paths(name, data_root):
            found = next(
                (path for path in alternatives if path.is_file() and path.stat().st_size > 0), None
            )
            if found is None:
                missing.append(" or ".join(str(path) for path in alternatives))
            else:
                selected[name].append(found)
    if missing:
        raise ValueError(
            "Required dataset files are missing or empty:\n  "
            + "\n  ".join(missing)
            + "\nUse --data-root PATH for a Dataset directory containing real/ and fake/."
            + "\nSee README.md for data preparation; no datasets were downloaded."
        )
    return selected


def _required_modules(suite, methods, selected_datasets):
    required = {"numpy", "scipy", "sklearn", "torch"}
    chosen = set(methods)
    if suite == "ggad":
        if "gadmore" in chosen:
            required.add("geoopt")
        if chosen & {"iaggad_zs", "iaggad_fs", "taggad"}:
            required.add("einops")
        if chosen & {"taggad", "promos"}:
            required.add("torch_geometric")
        if "tfm4gad" in chosen:
            required.add("tabpfn")
    elif suite == "baselines":
        if chosen & {"gcn", "gat", "mlp", "bwgnn", "ghrn"}:
            required.add("torch_geometric")
        if chosen & {"bwgnn", "ghrn"}:
            required.add("sympy")
        if "xgb_graph" in chosen:
            required.add("xgboost")
    if "elliptic" in selected_datasets:
        required.add("pandas")
    if "tsocial" in selected_datasets and any(
        path.name == "tsocial" for path in selected_datasets["tsocial"]
    ):
        required.add("dgl")
    return sorted(required)


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _dependency_report(required, dry_run):
    missing = [PACKAGES[name] for name in required if not _has_module(name)]
    if not missing:
        print(
            "Dependencies: required packages were found (binary compatibility is checked at runtime)."
        )
        return
    message = "Missing dependencies for this selection: " + ", ".join(missing)
    core = [name for name in missing if name not in {"tabpfn", "dgl"}]
    if core:
        message += "\nInstall the reference environment: " + _display_command(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
        )
    if "tabpfn" in missing:
        message += "\nInstall TFM4GAD's optional package: " + _display_command(
            [sys.executable, "-m", "pip", "install", "tabpfn"]
        )
    if "dgl" in missing:
        message += (
            "\nUse the portable tsocial.npz format, or install DGL matching "
            "your PyTorch/CUDA build to read the raw tsocial graph."
        )
    if dry_run:
        print(message)
        print("Dry run continues without these packages; execution would require them.")
    else:
        raise ValueError(message)


def _validate_device(device):
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("--device must be cpu, cuda, or cuda:N with a non-negative GPU index")


def _check_device_runtime(device):
    if device == "cpu":
        return
    try:
        import torch
    except (ImportError, OSError) as error:
        raise ValueError(f"PyTorch could not be loaded: {error}") from error
    if not torch.cuda.is_available():
        raise ValueError(
            f"Requested {device}, but CUDA is unavailable. Select --device cpu explicitly."
        )
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )


def _output_directory(suite, config, override):
    if override:
        path = Path(override).expanduser().resolve()
    elif suite == "baselines":
        path = (ROOT / config["RESULTS_DIR"]).resolve()
    else:
        path = ROOT / suite / "results"
    for parent in (path, *path.parents):
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Output directory is blocked by an existing file: {parent}")
            if not os.access(parent, os.W_OK):
                raise ValueError(f"Output directory is not writable: {parent}")
            break
    return path


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite", choices=tuple(SUITES), default="ggad", help="benchmark suite (default: ggad)"
    )
    parser.add_argument(
        "--methods", help="comma-separated method names; otherwise use the suite config"
    )
    parser.add_argument(
        "--seed", type=int, help="run a single seed instead of the configured seed list"
    )
    parser.add_argument(
        "--shot-sweep",
        action="store_true",
        help="GGAD support-size evaluation using SHOT_SWEEP_* settings",
    )
    parser.add_argument("--src-type", choices=("real", "fake"))
    parser.add_argument("--tgt-type", choices=("real", "fake"))
    parser.add_argument("--source-mode", choices=("single", "multi", "both"))
    parser.add_argument(
        "--targets", help="comma-separated target names; overrides quick target selection"
    )
    parser.add_argument("--device", help="cpu, cuda, or cuda:N; otherwise use the suite config")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use seed 0, fewer training steps, and two default targets",
    )
    parser.add_argument(
        "--data-root",
        help="directory containing real/ and fake/ (default: GAD_DATA_ROOT or <project root>/Dataset/)",
    )
    parser.add_argument(
        "--output-dir", help="write this suite's CSV outputs directly here (or use GAD_OUTPUT_DIR)"
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="list every supported method in the selected suite and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check options and dataset paths without training or importing model libraries",
    )
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.shot_sweep and args.suite != "ggad":
            raise ValueError("--shot-sweep is supported only with --suite ggad")
        if args.seed is not None and args.seed < 0:
            raise ValueError("--seed must be a non-negative integer")
        config, supported = _configuration(args.suite)
        if args.list_methods:
            print(f"Available {args.suite} methods:")
            for name, protocol in supported.items():
                print(f"  {name:20s} {protocol}")
            return 0
        prefix = "SHOT_SWEEP_" if args.shot_sweep else ""
        method_key = "BASELINES" if args.suite == "baselines" else prefix + "METHODS"
        methods = (
            _comma_list(args.methods, "--methods")
            if args.methods is not None
            else list(config[method_key])
        )
        unknown = [name for name in methods if name not in supported]
        if not methods or unknown:
            raise ValueError(
                f"Unknown or empty method selection: {unknown or methods}. Use --suite {args.suite} --list-methods."
            )
        src_type = args.src_type or config[prefix + "SRC_TYPE"]
        tgt_type = args.tgt_type or config[prefix + "TGT_TYPE"]
        source_mode = args.source_mode or config[prefix + "SOURCE_MODE"]
        device = args.device or config[prefix + "DEVICE"]
        quick = args.quick or config[prefix + "QUICK"]
        _validate_device(device)
        modes = ("single", "multi") if source_mode == "both" else (source_mode,)
        sources = {mode: list(config[f"{src_type.upper()}_{mode.upper()}"]) for mode in modes}
        default_targets = list(config[f"{tgt_type.upper()}_TARGETS"])
        if args.shot_sweep:
            if config["SHOT_SWEEP_TARGETS"] is not None:
                default_targets = list(config["SHOT_SWEEP_TARGETS"])
        elif quick:
            default_targets = default_targets[:2]
        targets = (
            _comma_list(args.targets, "--targets") if args.targets is not None else default_targets
        )
        if not targets:
            raise ValueError("At least one target dataset is required")
        data_root = (
            Path(args.data_root or os.environ.get("GAD_DATA_ROOT") or ROOT / "Dataset")
            .expanduser()
            .resolve()
        )
        output_dir = _output_directory(
            args.suite, config, args.output_dir or os.environ.get("GAD_OUTPUT_DIR")
        )
        use_sources = not (args.suite == "ggad" and set(methods) <= SOURCE_INDEPENDENT)
        names = targets + (
            [name for group in sources.values() for name in group] if use_sources else []
        )
        selected = _check_datasets(names, data_root)
        command = [
            sys.executable,
            str(ROOT / "run.py"),
            "--suite",
            args.suite,
            "--src-type",
            src_type,
            "--tgt-type",
            tgt_type,
            "--source-mode",
            source_mode,
            "--device",
            device,
            "--methods",
            ",".join(methods),
            "--targets",
            ",".join(targets),
        ]
        command.extend(["--data-root", str(data_root), "--output-dir", str(output_dir)])
        if quick:
            command.append("--quick")
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        if args.shot_sweep:
            command.append("--shot-sweep")
        print(
            f"Suite: {args.suite}; methods: {', '.join(methods)}; device: {device}; quick: {quick}"
        )
        print(
            f"Sources: {sources}"
            if use_sources
            else "Sources: unused by the selected source-independent methods"
        )
        print(
            f"Targets: {', '.join(targets)}\nData directory: {data_root}\nOutput directory: {output_dir}"
        )
        if args.shot_sweep:
            stem = config["SHOT_SWEEP_RESULTS_STEM"]
            filenames = tuple(f"{stem}_{kind}.csv" for kind in ("seeds", "detail", "summary"))
            print(f"Support sizes: {config['SHOT_SWEEP_SHOTS']}")
        elif args.suite == "ggad":
            filenames = (config["RESULTS_DETAIL_FILE"], config["RESULTS_DELTA_FILE"])
        elif args.suite == "baselines":
            filenames = ("source_compare_detail.csv", "source_compare_delta.csv")
        else:
            filenames = ("foundation_compare_detail.csv", "foundation_compare_delta.csv")
        for filename in filenames:
            print(f"  CSV: {output_dir / filename}")
        print("Command: " + _display_command(command))
        _dependency_report(_required_modules(args.suite, methods, selected), args.dry_run)
        if (
            args.suite == "baselines"
            and set(methods) & {"gcn", "gat"}
            and set(selected) & {"dgraphfin", "elliptic", "tsocial"}
        ):
            if not (_has_module("pyg_lib") or _has_module("torch_sparse")):
                print(
                    "Large-graph neighbor batching additionally needs pyg-lib or torch-sparse; see README.md."
                )
        if "tfm4gad" in methods:
            print(
                "TFM4GAD may download its TabPFN checkpoint on execution if absent; see README.md for checkpoint setup."
            )
        if "tpcagad" in methods and not _has_module("numba"):
            print(
                "TPCA-GAD needs optional numba for uncached structural preprocessing above "
                "2,000,000 adjacency entries under the default configuration. Small graphs "
                "do not require it. Install when needed: "
                + _display_command([sys.executable, "-m", "pip", "install", "numba"])
            )
        if args.dry_run:
            print(
                "Dry run complete: no data was loaded, no model libraries were imported, and no files were created."
            )
            return 0
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _check_device_runtime(device)
        # Set data/output locations before importing any model or shared loader.
        os.environ["GAD_DATA_ROOT"] = str(data_root)
        os.environ["GAD_OUTPUT_DIR"] = str(output_dir)
        args.src_type, args.tgt_type = src_type, tgt_type
        args.source_mode, args.device, args.quick = source_mode, device, quick
        args.methods, args.targets = ",".join(methods), ",".join(targets)
        sys.stdout.flush()
        module = "shot_sweep" if args.shot_sweep else "experiment"
        experiment = importlib.import_module(f"{args.suite}.{module}")
        return experiment.run(args) or 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
