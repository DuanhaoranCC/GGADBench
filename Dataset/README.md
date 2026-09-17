# Datasets

Download the datasets needed by your selected source and target configuration.
Paths below are relative to the project root, alongside `run.py`. Keep filenames
and capitalization exactly as shown. Dataset files are not included in this
repository; `real/` and `fake/` initially contain only `.gitkeep` placeholders.

## ARC datasets

Download the following 15 files from the
[ARC dataset directory](https://github.com/yixinliu233/ARC/tree/main/dataset).
Open a file and choose **Download raw file**, or download the ARC repository ZIP
and copy the files from its `dataset/` directory.

| Destination directory | Files to copy |
| --- | --- |
| `Dataset/real/` | `Facebook.mat`, `Amazon.mat`, `questions.mat`, `Reddit.mat`, `tolokers.mat`, `weibo.mat`, `YelpChi.mat` |
| `Dataset/fake/` | `cora.mat`, `citeseer.mat`, `ACM.mat`, `BlogCatalog.mat`, `Flickr.mat`, `pubmed.mat`, `cs.mat`, `photo.mat` |

Use the supplied anomaly-detection versions, including their existing labels.
The `fake/` files already contain injected anomalies; no anomaly injection step
is needed.

## Amazon-all, YelpChi-all, and T-Finance

The [TAM authors' dataset collection](https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z)
contains ready-to-use MATLAB files. This download is linked in the
[official TAM README](https://github.com/mala-lab/TAM-master/#datasets).
Download these three files and place them directly in `Dataset/real/`:

| File | Nodes | Features |
| --- | ---: | ---: |
| `Amazon-all.mat` | 11,944 | 25 |
| `YelpChi-all.mat` | 45,941 | 32 |
| `t_finance.mat` | 39,357 | 10 |

Keep `Amazon.mat` and `YelpChi.mat` from ARC as separate datasets. The `-all`
files above use different graph versions. The original
[CARE-GNN downloads](https://github.com/YingtongDou/CARE-GNN/tree/master/data)
have different fields and node sets, so simply renaming them does not reproduce
these prepared files. Use TAM's `t_finance.mat` here; the original BWGNN
`tfinance` file is a DGL graph, not a MATLAB file.

## Disney

Download [Disney.mat from the UNPrompt authors' repository](https://github.com/mala-lab/UNPrompt/blob/main/Datasets/Disney.mat)
using **Download raw file**, then save it as `Dataset/real/Disney.mat`.
This prepared version has 124 nodes and 28 features and matches the Disney file
used by this benchmark. No conversion is needed.

## DGraph-Fin

Download the original **DGraph-Fin** dataset (`DGraphFin.zip`) from the
[official DGraph website](https://dgraph.xinye.com/), following the site's
download instructions. Select DGraph-Fin, rather than DGraph-Fin2 or DGraph-Nearby.
The [authors' baseline repository](https://github.com/DGraphXinye/DGraphFin_baseline)
also points to this download.

Extract `dgraphfin.npz` and place it at `Dataset/real/dgraphfin.npz`.
Use the original NPZ containing `x`, `y`, `edge_index` with shape `E x 2`,
`train_mask`, `valid_mask`, and `test_mask`. A PyG-processed `.pt` file cannot be
used in its place. The configuration/CLI dataset name is `dgraphfin`.

## Elliptic

Download **Elliptic Data Set** from
[Elliptic's Kaggle page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
using **Download**; sign in if Kaggle requests it. Extract the original CSV files
and arrange them as follows, without an extra nested directory:

```text
Dataset/real/elliptic_bitcoin_dataset/
  elliptic_txs_classes.csv
  elliptic_txs_features.csv
  elliptic_txs_edgelist.csv
```

Keep the original CSV contents and row order. The loader treats class `1` as
anomalous and excludes `unknown` labels from evaluation. The configuration/CLI
dataset name is `elliptic`.

## T-Social

Download `tsocial.zip` from the
[BWGNN authors' dataset folder](https://drive.google.com/drive/folders/1DKO8edxNJ3UUWq-w2f6mLlKecr1itEUr),
linked through the [official BWGNN repository](https://github.com/squareRoot3/Rethinking-Anomaly-Detection#how-to-run).
Extract the DGL graph file named `tsocial` (no extension) and place it at
`Dataset/real/tsocial`.

The benchmark can read this file with DGL installed. To avoid requiring DGL for
subsequent benchmark runs, install a DGL build compatible with your PyTorch
environment and run the existing converter once from the project root:

```bash
python examples/convert_tsocial.py
```

This writes `Dataset/real/tsocial.npz` with `x`, binary `y`, and `edge_index`
(`E x 2`). The loader prefers this NPZ when present. The configuration/CLI
dataset name is `tsocial`.

## Check the paths

After preparing the datasets for a run, add `--dry-run` to its command to check
file paths and dependencies without training. For example, after downloading
Facebook and Disney:

```bash
python run.py --suite baselines --methods mlp --src-type real --tgt-type real --source-mode single --targets Disney --dry-run
```

This checks file presence, not file contents. MATLAB inputs must contain
adjacency `Network` or `A` (`N x N`), features `Attributes` or `X` (`N x D`), and
labels `Label` or `gnd` (`N` values: 0 normal, 1 anomalous).
To store datasets elsewhere, pass `--data-root /path/to/Dataset` to `run.py`,
or set `GAD_DATA_ROOT` (also used by the T-Social converter).
