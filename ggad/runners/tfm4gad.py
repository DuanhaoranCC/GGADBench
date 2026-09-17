"""TFM4GAD runner using benchmark-owned data, seeds, few-shot and metrics."""

from __future__ import annotations

import gc

import numpy as np

import ggad.config as C
from common.data import CACHE_ROOT, aggregate, load_target_marked
from ggad.vendor.tfm4gad import (
    augment_node_features,
    create_tabpfn_classifier,
    predict_positive_batched,
    require_tabpfn,
)
from util import evaluate, set_seed


def _sample_support(labels, mark, shot, seed):
    """Benchmark few-shot protocol: k labeled normals and k labeled anomalies."""
    set_seed(seed)
    normal = np.where((labels == 0) & mark)[0]
    anomaly = np.where((labels == 1) & mark)[0]
    np.random.shuffle(normal)
    np.random.shuffle(anomaly)
    if normal.size < shot or anomaly.size < shot:
        raise ValueError(
            f"TFM4GAD needs {shot} marked nodes from each class, got "
            f"normal={normal.size}, anomaly={anomaly.size}"
        )
    return np.sort(np.concatenate([normal[:shot], anomaly[:shot]]))


def run_tfm4gad(sources, targets, seeds, epochs, device, shot=10, train=True):
    """Run target-graph in-context classification; source graphs are unused."""
    del sources, epochs, train
    hp = dict(C.TFM4GAD_HP)
    # Fail before expensive graph augmentation when the optional model package
    # is unavailable.  The returned object is discarded; each split still gets
    # a fresh classifier exactly as in the release.
    require_tabpfn()

    per_target = {name: [] for name in targets}
    cache_dir = CACHE_ROOT / "tfm4gad"
    for name in targets:
        print(f"  [tfm4gad/target] {name}")
        adjacency, raw_features, labels, mark = load_target_marked(name)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        mark = np.asarray(mark, dtype=bool).reshape(-1)
        features = augment_node_features(name, adjacency, raw_features, hp, cache_dir=cache_dir)
        print(f"    graph-table shape={features.shape} support={shot}+{shot}")

        for seed in seeds:
            support = _sample_support(labels, mark, int(shot), int(seed))
            query_mask = mark.copy()
            query_mask[support] = False
            query = np.where(query_mask)[0]
            if np.unique(labels[query]).size != 2:
                raise ValueError(f"TFM4GAD query for {name} does not contain both classes")

            classifier = create_tabpfn_classifier(device, hp)
            classifier.fit(features[support], labels[support])
            scores = predict_positive_batched(
                classifier, features, query, int(hp["inference_batch_size"])
            )
            result = evaluate(labels[query], scores)
            per_target[name].append(result)
            print(
                f"    seed={seed:02d} AUROC={result['AUROC']:.4f} " f"AUPRC={result['AUPRC']:.4f}"
            )
            del classifier
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        del adjacency, raw_features, features
        gc.collect()
    return aggregate(per_target)


__all__ = ["run_tfm4gad", "_sample_support"]
