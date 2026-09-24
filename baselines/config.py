"""Run controls and frozen hyperparameters for classic baselines."""

# ========================= run controls =========================
DEVICE = "cuda:0"
EVAL_METRICS = "standard"  # "standard" | "recall-at-k" | "all"

QUICK = False
SEEDS = list(range(11))  # strict seeds 0..10

SRC_TYPE = "fake"  # "real" | "fake"
TGT_TYPE = "fake"  # "real" | "fake"
SOURCE_MODE = "both"  # "single" | "multi" | "both"

RESULTS_DIR = "baselines/results"

# ========================= datasets =========================
REAL_SINGLE = ["Facebook"]
REAL_MULTI = ["Facebook", "t_finance", "Amazon"]
REAL_TARGETS = [
    "Facebook",
    "Amazon",
    "Amazon-all",
    "Disney",
    "questions",
    "Reddit",
    "tolokers",
    "t_finance",
    "weibo",
    "YelpChi",
    "YelpChi-all",
    "dgraphfin",
    "elliptic",
    "tsocial",
]

FAKE_SINGLE = ["cora"]
FAKE_MULTI = ["cora", "citeseer", "ACM", "BlogCatalog"]
FAKE_TARGETS = ["cora", "citeseer", "ACM", "BlogCatalog", "Flickr", "pubmed", "cs", "photo"]


def sources(mode=None, src_type=None):
    mode = mode or SOURCE_MODE
    src_type = src_type or SRC_TYPE
    single, multi = (REAL_SINGLE, REAL_MULTI) if src_type == "real" else (FAKE_SINGLE, FAKE_MULTI)
    return single if mode == "single" else multi


def targets(tgt_type=None):
    tgt_type = tgt_type or TGT_TYPE
    return REAL_TARGETS if tgt_type == "real" else FAKE_TARGETS


# ========================= methods / feature preprocessing =========================
BASELINES = [
    "gat",
    "mlp",
    "rf_graph",
    "xgb_graph",
    "dominant",
]
SOURCE_AGNOSTIC = set()

BASELINE_SVD_DIM = 8

FEATURE_CONFIGS = {
    "cola": {"dim": 8, "norm": "zscore"},
    "tam": {"dim": 64, "norm": "arc"},
    "bwgnn": {"dim": 8, "norm": "center"},
    "ghrn": {"dim": 8, "norm": "zscore"},
    "gcn": {"dim": 8, "norm": "none"},
    "gat": {"dim": 8, "norm": "zscore"},
    "mlp": {"dim": 8, "norm": "zscore"},
    "rf_graph": {"dim": 8, "norm": "zscore"},
    "xgb_graph": {"dim": 8, "norm": "center"},
    "dominant": {"dim": 64, "norm": "arc"},
}

# ========================= DOMINANT hyperparameters =========================
DOMINANT_HP = {
    "hidden_dim": 128,
    "num_epoch": 100,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "alpha": 0.8,
    "add_self_loop": True,  # official target/input adjacency is A+I
    # Ordinary datasets use the released all-pairs formula in row blocks.
    # Only the million-scale extension datasets retain an explicit fallback.
    "approx_structure_datasets": ("dgraphfin", "elliptic", "tsocial"),
    "structure_batch_size": 256,
    "structure_sample_size": 256,
    "score_structure_sample_size": 512,
    # Large-graph GCN adjacency remains on CPU as an edge list. Forward and
    # backward stream exact edge blocks; structure rows use a larger node block
    # and fewer sampled columns so million-node inference is memory-bounded.
    "stream_node_threshold": 200_000,
    "stream_edge_threshold": 5_000_000,
    "edge_chunk": 500_000,
    "large_structure_batch_size": 8192,
    "large_structure_sample_size": 128,
    "large_score_structure_sample_size": 64,
}
DOMINANT_QUICK = {
    "num_epoch": 2,
    "structure_sample_size": 64,
    "score_structure_sample_size": 128,
}

# Plain feature-only supervised baseline. It deliberately receives no
# adjacency or graph-derived features, making it the control for whether a
# graph model improves over the shared SVD8 node attributes alone.
MLP_HP = {
    "hid_dim": 32,
    "num_layers": 2,  # number of hidden Linear blocks
    "activation": "leaky_relu",
    "dropout": 0.0,
    "num_epoch": 200,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "target_batch_size": 65_536,  # exact node-wise inference chunk
}
MLP_QUICK = {
    "num_epoch": 5,
}


# ========================= CoLA hyperparameters =========================
COLA_HP = {
    "embedding_dim": 32,
    "batch_size": 128,
    "subgraph_size": 8,
    "readout": "weighted_sum",
    "auc_test_rounds": 256,
    "negsamp_ratio": 2,
    "weight_decay": 1e-4,
    "lr": 1e-6,
    "num_epoch": 100,
}
COLA_QUICK = {
    "num_epoch": 5,
    "auc_test_rounds": 8,
}
# ========================= TAM hyperparameters =========================
# Official defaults: two-layer GCN LAMNet, Adam(lr=1e-5), 500 epochs, T=3, K=4.
# README uses larger truncation K for larger graphs.
TAM_HP = {
    "lr": 5e-3,
    "weight_decay": 1e-6,
    "embedding_dim": 64,
    "num_epoch": 300,
    "cutting": 1,
    "n_tree": 1,
    "lamda": 1.0,  # None selects the official source-specific rule.
    "readout": "avg",
    "negsamp_ratio": 2,
}
TAM_QUICK = {
    "num_epoch": 3,
    "cutting": 1,
    "n_tree": 1,
}
TAM_LAMBDA_ONE = {"BlogCatalog", "ACM", "Facebook"}
# ========================= BWGNN hyperparameters =========================
# Official supervised defaults from BWGNN/main.py and README:
# lr=0.01 in train(), hidden=64, order=2, epoch=100, homo graph.
BWGNN_HP = {
    "hid_dim": 32,
    "order": 5,
    "num_epoch": 100,
    "lr": 1e-6,
    "weight_decay": 1e-3,
    "add_self_loop": True,
}
BWGNN_QUICK = {
    "num_epoch": 5,
}
# ========================= GHRN hyperparameters =========================
# Official GHRN uses BWGNN probabilities to prune low high-frequency-score edges,
# then trains BWGNN on the pruned graph.  The DGL pruning is ported to edge_index.
GHRN_HP = {
    "hid_dim": 128,
    "order": 2,
    "pretrain_epoch": 50,
    "num_epoch": 100,
    "lr": 1e-6,
    "weight_decay": 1e-5,
    "del_ratio": 0.015,
    "adj_type": "sym",
    "add_self_loop": True,
}
GHRN_QUICK = {
    "pretrain_epoch": 5,
    "num_epoch": 5,
}
# ========================= GCN/GAT hyperparameters =========================
# Plain supervised node-classification baselines, trained on source labels and
# evaluated zero-shot on target graphs.
GCN_HP = {
    "hid_dim": 64,
    "num_layers": 2,
    "activation": "relu",
    "num_epoch": 200,
    "lr": 1e-5,
    "weight_decay": 0.0,
    "dropout": 0.5,
    "add_self_loop": True,
    # Exact full-graph inference is tried below this edge budget.  Above it,
    # or after a CUDA OOM, all marked target nodes are still evaluated as
    # seeds.  NeighborLoader uses full neighbors (`-1`) per layer, so this is
    # neighborhood batching rather than neighbor sampling.
    "target_sample_edges": 80_000_000,
    "target_batch_size": 4096,
    "target_num_neighbors": [-1, -1],
    "source_batch_size": 4096,
    "source_num_neighbors": [-1, -1],
    "source_batch_edges": 2_000_000,
}
GCN_QUICK = {
    "num_epoch": 5,
}
GAT_HP = {
    "hid_dim": 8,
    "num_layers": 2,
    "activation": "relu",
    "heads": 8,
    "out_heads": 1,
    "num_epoch": 300,
    "lr": 5e-5,
    "weight_decay": 1e-6,
    "dropout": 0.0,
    "add_self_loop": True,
    # GAT stores attention per edge, so very large targets use NeighborLoader
    # inference.  This does not subsample eval nodes or neighbors: every
    # mark=True node is an input seed and full neighbors (`-1`) are expanded
    # per layer.
    "target_sample_edges": 50_000_000,
    "target_batch_size": 4096,
    "target_num_neighbors": [-1, -1],
    "source_batch_size": 4096,
    "source_num_neighbors": [-1, -1],
    "source_batch_edges": 2_000_000,
}
GAT_QUICK = {
    "num_epoch": 5,
}
# ========================= RF/XGB-Graph hyperparameters =========================
# Official GADBench RFGraph/XGBGraph first applies GIN_noparam:
#   h_final = concat(X, GIN(X), ..., GIN^L(X))
# and then fits a tree classifier. GIN_noparam uses scipy sparse matrices.
RF_GRAPH_HP = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_samples": 1.0,
    "max_features": "sqrt",
    "num_layers": 3,
    "agg": "mean",
    "predict_chunk": 200_000,
    "n_jobs": -1,
}
RF_GRAPH_QUICK = {
    "n_estimators": 20,
}
XGB_GRAPH_HP = {
    "n_estimators": 50,
    "eta": 0.05,
    "reg_lambda": 10.0,
    "subsample": 0.5,
    "booster": "dart",
    "tree_method": "hist",
    "num_layers": 4,
    "agg": "mean",
    "predict_chunk": 200_000,
    "n_jobs": -1,
}
XGB_GRAPH_QUICK = {
    "n_estimators": 20,
}


def method_hp(method):
    """Return the single frozen hyperparameter dictionary for a baseline."""
    name = f"{str(method).upper()}_HP"
    try:
        return globals()[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline method {method!r}") from exc
