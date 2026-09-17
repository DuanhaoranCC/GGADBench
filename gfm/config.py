"""Fixed configurations for source-to-target graph foundation experiments."""

# ---------------- data protocol ----------------
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

METHOD_PROTOCOL = {
    "graphprompt": "few_shot",
    "graphprompt_enr": "few_shot",
    "samgpt": "few_shot",
    "samgpt_diff": "few_shot",
    "samgpt_enr": "few_shot",
    "bridge": "few_shot",
    "mdgfm": "few_shot",
    "mdgpt": "few_shot",
}

METHODS = ["mdgpt"]


DEVICE = "cuda:0"
SRC_TYPE = "fake"  # "real" | "fake"
TGT_TYPE = "fake"  # "real" | "fake"
SOURCE_MODE = "both"  # "single" | "multi" | "both"
QUICK = False

SEEDS = list(range(11))
SHOT = 10  # Per-class support size for few-shot episodes.

# Unified tabular feature adapter for graph foundation models.

FEATURE_DIM = 8
FEATURE_NORM = "zscore"
FEATURE_CONFIGS = {
    "graphprompt": {"dim": 8, "norm": "zscore"},
    "samgpt": {"dim": 8, "norm": "zscore"},
    "bridge": {"dim": 64, "norm": "zscore"},
    "mdgfm": {"dim": 8, "norm": "zscore"},
    "mdgpt": {"dim": 10, "norm": "none"},
}
# *_enr variants explicitly retain ego and multi-hop neighbor branches:
# [X,PX,...,P^L X] -> shared MLP -> [Z^1-Z^0 || ... || Z^L-Z^0].
# Downstream modules remain method-specific.
ENR_ENCODER_HP = {
    "hidden_dim": 128,
    "num_layers": 3,
    "num_hops": 2,
    "dropout": 0.0,
    "activation": "ELU",
}
ENR_MODE = "multi_hop_ego_neighbor_residual"

# `samgpt_diff` is a downstream anomaly adaptation, not an encoder replacement:
# native SAMGPT multi-domain pretraining and frozen source-domain tokens remain;
# target prompt tuning learns normal ego-neighbor agreement and anomalous
# disagreement using the few-shot support labels.
SAMGPT_DIFF_MODE = "native_pretrain+domain_tokens+ego_neighbor_prompt_tuning"


GRAPHPROMPT_HP = {
    # GIN encoder with a feature-weighted prompt and prototype classifier.
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "scalar": 1e3,
    "nhop_neighbour": 0,
    "lr": 1e-2,
    "weight_decay": 1e-3,
    "epochs": 50,
    "quick_epochs": 5,
    "batch_query_per_class": 128,
    "eval_query_batch": 200_000,
    # Exact edge-chunk message passing for large/dense graphs.
    # Match the bounded dense-edge schedule used by DRGGAD; changing this only
    # changes execution order, never the node/edge set or model output.
    "edge_chunk": 500_000,
    "node_chunk": 100_000,
}


SAMGPT_HP = {
    # GCN backbone with source-domain prompts and target adaptation.
    # Inputs use the shared SVD feature adapter above.
    "hid_units": 128,
    "layers_num": 3,
    "combinetype": "add",
    "alpha": 2.0,
    "beta": 0.5,
    "pretrain_lr": 1e-6,
    "pretrain_weight_decay": 1e-4,
    "pretrain_epochs": 100,
    "quick_pretrain_epochs": 3,
    "drop_percent": 0.2,
    # Do not materialize multiple augmented copies for very dense source graphs.
    # Dense sources reuse the normalized full graph for the two GraphCL views.
    "aug_max_edges": 5_000_000,
    "downstream_lr": 1e-1,
    "downstream_steps": 400,
    "quick_downstream_steps": 5,
    # Large targets still train SAMGPT's target downstream prompt.  The runner
    # computes the exact L-layer dependency subgraph of the few-shot support
    # nodes for this training loss (no fanout/random sampling), then evaluates
    # all marked target nodes on the full graph with chunked sparse propagation.
    "large_downstream_steps": 400,
    "eval_query_batch": 200_000,
    "edge_chunk": 500_000,
}

BRIDGE_HP = {
    # Fixed BRIDGE configuration for source pretraining and target adaptation.
    "hid_units": 256,
    "layers_num": 2,
    "gcn_dropout": 0.0,
    "combinetype": "mul",
    "pretrain_lr": 5e-3,
    "pretrain_weight_decay": 1e-4,
    "pretrain_epochs": 100,
    "quick_pretrain_epochs": 3,
    "pretrain_patience": 20,
    "negative_samples": 50,
    "variance_samples": 2,
    "variance_weight": 1e6,
    "contrast_chunk": 4096,
    "routing_dropout": 0.0,
    "downstream_lr": 1e-2,
    "downstream_steps": 200,
    "quick_downstream_steps": 5,
    "large_downstream_steps": 200,
    "downstream_patience": 20,
    "lambda_entropy": 0.05,
    "reg_weight": 1.0,
    "reg_thres": 0.4,
    "spectral_components": 100,
    # Exact execution controls; they change only batching/order, not nodes/edges.
    "edge_chunk": 500_000,
    "node_chunk": 100_000,
    "eval_query_batch": 200_000,
}

MDGFM_HP = {
    # The paper's PCA50 projection is replaced by the shared SVD8 adapter.
    "hid_units": 64,
    "layers_num": 3,
    "pretrain_dropout": 0.1,
    "combinetype": "mul",
    "pretrain_lr": 5e-3,
    "pretrain_weight_decay": 5e-3,
    "pretrain_epochs": 30,
    "quick_pretrain_epochs": 3,
    "pretrain_patience": 20,
    # Eq. (10)-(11): exact block-to-all kNN for moderate graphs and bounded
    # random-projection candidate search followed by exact cosine reranking for
    # large graphs. Both paths retain all nodes and O(Nk) sparse graph storage.
    "knn_k": 5,
    "homophilic_knn_k": 30,
    "homophilic_datasets": tuple(FAKE_TARGETS),
    "knn_k_by_dataset": {},
    "exact_knn_max_nodes": 12_000,
    "knn_query_chunk": 1_024,
    "lsh_tables": 2,
    "lsh_window": 64,
    "gsl_search_seed": 0,
    "gsl_dropout": 0.5,
    # Eq. (13)-(15): official temperature and batch size. Every sampled anchor
    # still contrasts against all graph nodes; keys are merely streamed.
    "alignment_temperature": 0.1,
    "alignment_batch_size": 64,
    "contrastive_key_chunk": 100_000,
    # Downstream prompt budget and original/refined adjacency blend.
    "downstream_lr": 1e-3,
    "downstream_steps": 100,
    "quick_downstream_steps": 5,
    "adjacency_mix": 0.5,
    "prototype_temperature": 0.5,
    "target_knn_refresh_interval": 1,
    "target_knn_refresh_max_nodes": 5_000,
    "full_target_tune_max_nodes": 5_000,
    # Execution-only bounds; these change neither graph membership nor the
    # evaluated query set. All non-support marked target nodes are scored.
    "edge_chunk": 500_000,
    "eval_query_batch": 200_000,
    "cache_features": True,
    "clip_grad": 1.0,
}

# MDGPT paper-based implementation with a fixed benchmark configuration.
# feature_dim=50 requires every selected graph to have at least 50 features;
# insufficient width raises an error rather than padding features.
MDGPT_HP = {
    "implementation_version": "paper-v1",
    "feature_dim": 10,
    "hidden_dim": 256,
    "num_layers": 6,
    "pretrain_lr": 5e-5,
    "prompt_lr": 1e-2,
    "weight_decay": 0.0,
    "temperature": 0.05,
    "num_negatives": 1,
    "triplets_per_domain": 4096,
    "edge_chunk": 100_000,
    "node_chunk": 4096,
    "stream_node_threshold": 200_000,
    "stream_edge_threshold": 5_000_000,
    "feature_seed": 42,
    "cache": True,
    "log_every": 20,
}

MDGPT_EPOCHS = 25
MDGPT_PROMPT_EPOCHS = 800
MDGPT_QUICK_EPOCHS = 2
MDGPT_QUICK_PROMPT_EPOCHS = 2

FOUNDATION_HP = {
    "graphprompt": GRAPHPROMPT_HP,
    "graphprompt_enr": GRAPHPROMPT_HP,
    "samgpt": SAMGPT_HP,
    "samgpt_diff": SAMGPT_HP,
    "samgpt_enr": SAMGPT_HP,
    "bridge": BRIDGE_HP,
    "mdgfm": MDGFM_HP,
    "mdgpt": MDGPT_HP,
}

RESULTS_DIR = "gfm/results"


def sources(mode=None, src_type=None):
    mode = mode or SOURCE_MODE
    src_type = src_type or SRC_TYPE
    single, multi = (REAL_SINGLE, REAL_MULTI) if src_type == "real" else (FAKE_SINGLE, FAKE_MULTI)
    return single if mode == "single" else multi


def targets(tgt_type=None):
    tgt_type = tgt_type or TGT_TYPE
    return REAL_TARGETS if tgt_type == "real" else FAKE_TARGETS
