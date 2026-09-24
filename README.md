# GAD Benchmark

A benchmark for source-to-target graph anomaly detection, classical baselines,
and graph foundation models, with zero-shot and few-shot evaluation.

## Install

Use Python 3.10 in a virtual environment. Install the PyTorch 2.2.2 build matching
your CPU/CUDA setup, then install the reference dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional dependencies depend on the selected method:

- **TFM4GAD:** `python -m pip install tabpfn`. Use a version compatible with your
  PyTorch/scikit-learn environment. Its TabPFN v2 checkpoint downloads on first
  use; set `TFM4GAD_MODEL_CKPT` to use an existing checkpoint instead.
- **Large-graph PyG neighbor batching:** `pyg-lib` or `torch-sparse`. Follow the
  [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)
  for wheels matching your PyTorch/CUDA versions.
- **TPCA-GAD:** `python -m pip install numba` for uncached structural preprocessing
  above 2 million edges.
- **T-Social DGL input:** `dgl`, unnecessary when using `tsocial.npz`.

## Prepare data

The default data directory is `Dataset/` in the project root, alongside `run.py`.
Follow [Dataset/README.md](Dataset/README.md) for download links, exact filenames,
and placement instructions for all 22 configured datasets. Only the selected
source and target datasets are required. The included `real/` and `fake/` folders
contain only empty `.gitkeep` files.

No `--data-root` argument is needed with this layout. To use another location,
pass `--data-root /path/to/Dataset` or set `GAD_DATA_ROOT`.

## Run

Run every benchmark through the root `run.py`. Select `ggad` (generalist graph
anomaly detection), `gfm` (graph foundation models), or `baselines`. These examples use Facebook as the source and Disney as the target:

```bash
python run.py --suite ggad --methods arc --src-type real --tgt-type real --source-mode single --targets Disney
python run.py --suite baselines --methods mlp --src-type real --tgt-type real --source-mode single --targets Disney
python run.py --suite gfm --methods graphprompt --src-type real --tgt-type real --source-mode single --targets Disney
```

To run a suite exactly as configured, using one seed and the full training schedule:

```bash
python run.py --suite ggad --seed 0
python run.py --suite gfm --seed 0
python run.py --suite baselines --seed 0
```

Set `EVAL_METRICS` in the selected suite's `config.py` to `"standard"`
(AUROC/AUPRC, the default), `"recall-at-k"` (Rec@K only), or `"all"`.
Training and query selection are unchanged. Rec@K uses K equal to the number
of actual anomalies in the evaluated nodes, following the same evaluation
mask and support/query protocol as AUROC and AUPRC.
Boundary score ties follow query order; no-anomaly queries return NaN.
Nondefault modes write separate `_recall_at_k.csv` or `_all_metrics.csv` files.

| Option | Purpose |
| --- | --- |
| `--seed 0` | Run one seed with the configured training schedule |
| `--shot-sweep` | Run GGAD support-size evaluation from `SHOT_SWEEP_*` settings |
| `--list-methods` | List supported methods in the selected suite |
| `--dry-run` | Check method names, data paths, and dependencies without training |
| `--quick` | Fewer epochs, one seed, and two default targets for a smoke run |
| `--device cuda:0` | Select a GPU; GPU 0 is the default |
| `--device cpu` | Explicitly run on CPU when needed |
| `--source-mode single`, `multi`, or `both` | Select the source protocol |
| `--targets Disney,Reddit` | Override the target list |
| `--data-root PATH` | Select the dataset directory |
| `--output-dir PATH` | Select the result directory |

Set source datasets, seeds, support sizes, and fixed hyperparameters in the
selected suite's `config.py`. Results contain per-target AUROC/AUPRC and
single-source versus multi-source differences, saved in `ggad/results/`, `baselines/results/`, or `gfm/results/` by default. Use distinct output directories when
changing experiment settings. Caches and results are ignored by Git.

For support-size experiments, configure `SHOT_SWEEP_*` in `ggad/config.py`,
then run `python run.py --suite ggad --shot-sweep`. The same CLI options apply.
Support feasibility is checked during execution; `--dry-run` only checks file
paths and reports settings without loading labels or creating files.
TA-GGAD retains the `few_shot_oracle_count` label because it uses the target
anomaly count. Method-specific preprocessing and prompt adaptation remain part
of normal model execution.

## Try without downloading a dataset

```bash
python examples/create_demo_data.py
python run.py --suite ggad --methods neighbordiv --src-type real --tgt-type real --source-mode single --targets Disney --quick --device cpu --data-root examples/demo_data --output-dir results/demo
```

This creates synthetic graphs, using Facebook/Disney filenames only for
compatibility. Their scores are not real benchmark results. The generator
refuses to overwrite existing files.

## Code structure

```text
run.py                        Only benchmark CLI; validates and dispatches runs
Dataset/real/, Dataset/fake/   Default locations for user-provided datasets
ggad/                         Generalist graph anomaly detection methods
  config.py                   Experiment settings and fixed hyperparameters
  experiment.py, runners/     Internal training and evaluation
  vendor/                     Model implementations
  shot_sweep.py, shot_query.py Support-size evaluation
gfm/                          Graph foundation models, config, and execution
baselines/                    Classical models, config, and execution
common/                       Shared data loading, caches, and CSV results
util.py                       Metrics, sparse operators, and random seeds
examples/                     Synthetic data generator and T-Social converter
tests/                        Shared/CLI tests; each suite has its own tests/
licenses/                     Upstream license texts and attribution
```

## Checks

```bash
python -m unittest discover -s tests
python -m unittest discover -s ggad/tests
python -m unittest discover -s gfm/tests
python -m unittest discover -s baselines/tests
```

Nine numerical tests are marked as expected failures: three reflect SAARCS's
retained directed normalization convention; six reflect TPCA-GAD v3's center-node
aggregation difference and small-deviation precision limitation. CUDA, full
datasets, and optional TabPFN execution require separate environment validation.

Optional formatting tools: install with
`python -m pip install black==24.4.2 isort==5.13.2 flake8==7.0.0`, then run
`python -m isort .`, `python -m black .`, and `python -m flake8 .`.

Upstream license texts and attribution are in `licenses/`. No project-wide
license has been selected; confirm licensing before publishing.
