"""MDGFM runner for source-to-target graph anomaly detection.

The implementation keeps the paper's source-domain pretraining and target
few-shot prompt/prototype stages while using the benchmark's shared SVD feature
adapter. Support nodes are removed from the target query set and every remaining
marked node is scored; graph operators stay sparse throughout.
"""

from gfm.vendor.mdgfm import run_mdgfm as _run_mdgfm


def run_mdgfm(sources, targets, seeds, hp, device, shot=10, feature_dim=8, feature_norm="zscore"):
    return _run_mdgfm(
        sources=sources,
        targets=targets,
        seeds=seeds,
        hp=hp,
        device=device,
        shot=shot,
        feature_dim=feature_dim,
        feature_norm=feature_norm,
    )


__all__ = ["run_mdgfm"]
