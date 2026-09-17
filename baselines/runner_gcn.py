"""GCN baseline dispatch."""

from baselines.runner_supervised_gnn import run_supervised_gnn


def run_gcn(sources, targets, seeds, device, norm, quick=False, hp=None, progress=None):
    return run_supervised_gnn(
        "gcn", sources, targets, seeds, device, norm, quick, hp=hp, progress=progress
    )
