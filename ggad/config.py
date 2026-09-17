"""Run controls and frozen hyperparameters for source-to-target benchmarks."""

from pathlib import Path

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

# Default selection; ggad.experiment.SUPPORTED_PROTOCOLS lists every method.
METHOD_PROTOCOL = {"gadmore": "zero_shot"}
METHODS = list(METHOD_PROTOCOL)


DEVICE = "cuda:0"
SRC_TYPE = "fake"
TGT_TYPE = "real"
SOURCE_MODE = "multi"
QUICK = False


RESULTS_DETAIL_FILE = "source_domains_detail.csv"
RESULTS_DELTA_FILE = "source_domains_delta.csv"


EPOCHS = {
    "arc": 40,
    "iaggad": 40,
    "taggad": 40,
    "unprompt": 900,
    "grace": 200,
    "anomalygfm_zs": 301,
    "anomalygfm_fs": 300,
    "refigad": 100,
    "drggad": 60,
    "gadmore": 40,
    "neighbordiv": 0,
    "promos": 10,
    "zerogad": 200,
    "owleye": 100,
    "tpcagad": 200,
    "saarcs": 100,
    "tfm4gad": 0,
}
QUICK_EPOCHS = {
    "arc": 5,
    "iaggad": 3,
    "taggad": 1,
    "unprompt": 30,
    "grace": 20,
    "anomalygfm_zs": 10,
    "anomalygfm_fs": 10,
    "refigad": 5,
    "drggad": 5,
    "gadmore": 2,
    "neighbordiv": 0,
    "promos": 2,
    "zerogad": 2,
    "owleye": 2,
    "tpcagad": 2,
    "saarcs": 2,
    "tfm4gad": 0,
}


SEEDS = list(range(11))  # 0-10, inclusive
SHOT = 10


SHOT_SWEEP_METHODS = [
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
SHOT_SWEEP_SHOTS = [1, 3, 5, 10, 20, 50]
SHOT_SWEEP_SRC_TYPE = SRC_TYPE  # "real" | "fake"
SHOT_SWEEP_TGT_TYPE = TGT_TYPE  # "real" | "fake"
SHOT_SWEEP_SOURCE_MODE = SOURCE_MODE  # "single" | "multi" | "both"
SHOT_SWEEP_TARGETS = None
SHOT_SWEEP_EXCLUDE_SOURCES = False
SHOT_SWEEP_SEEDS = list(SEEDS)
SHOT_SWEEP_DEVICE = DEVICE
SHOT_SWEEP_QUICK = False
SHOT_SWEEP_EPOCHS = dict(EPOCHS)
SHOT_SWEEP_RESUME = True
SHOT_SWEEP_RESULTS_STEM = "source_shot"  # results/source_shot_{seeds,detail,summary}.csv + .log


REFIGAD_K = {"Facebook": 10}
REFIGAD_K_DEFAULT = 10

REFIGAD_HP = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 4,
    "dim_feedforward": 128,
    "epoch": 100,
    "lr": 1e-4,
    "weight_decay": 5e-5,
    "batch_size": 512,
    "batches_per_dataset": 20,
    "pos_weight": 4.0,
    "num_hops": 2,
}

# TA-GGAD official release configuration.  The release calls ``num_layers=4``
# but constructs three Linear layers, and passes ``drop_rate`` rather than the
# constructor's ``dropout_rate``; the effective official dropout is therefore
# zero.  ``taggad`` preserves the released query anomaly-count oracle and is
# therefore labelled ``few_shot_oracle_count`` in ggad/experiment.py.
#
# The downloaded release does not contain params/dataset_config.json even
# though its active Dataset path expects that file.  The paper says
# hyperparameters are fixed across all targets, and the released CLI exposes
# ``--lamda=0.1`` as its only available value.  Use that uniform value by
# default; ``target_lambdas`` remains an explicit reproducibility override.
TAGGAD_HP = {
    "implementation_version": "official-release-pyg-port-v1",
    "in_feats": 64,
    "h_feats": 1024,
    "num_layers": 4,
    "num_hops": 2,
    "num_prompt": 10,
    # Official config key; ARC accepts it through **kwargs, so effective dropout
    # remains the constructor default 0 rather than the apparent 0.3.
    "drop_rate": 0.3,
    "activation": "ELU",
    "lr": 1e-5,
    "weight_decay": 5e-5,
    "code_size": 2048,
    "topk": 15,
    "gcn_emb_dim": 128,
    "count_node": 1,
    "normal_ratio": 10,
    "kde_bins": 500,
    "js_alpha": 3.0,
    "target_lambda": 0.1,
    "edge_chunk": 250_000,
    "node_chunk": 32_768,
    "stream_edge_threshold": 5_000_000,
    "target_cpu_hops_threshold": 1_000_000,
    "target_lambdas": {},
}

# DR-GGAD official main.py hyperparameters.  It follows ARC-style feature
# alignment and trains a source-supervised generalist detector, then freezes
# parameters for target inference.
DRGGAD_HP = {
    "h_feats": 1024,
    "num_layers": 4,
    "num_hops": 2,
    "drop_rate": 0.2,
    "activation": "ELU",
    "lr": 1e-5,
    "weight_decay": 5e-5,
    "alpha": 0.1,
    "k": 1,
    "epoch": 60,
    # Memory-safe exact computation for dense-edge / million-node graphs.
    # These do not subsample evaluation nodes or edges; they only bound
    # temporary tensors.
    "edge_chunk": 500_000,
    "node_chunk": 32_768,
}

# TPCA-GAD (ESWA 2026) paper-only reproduction. No official code is released.
# alpha, consistency lambda, MLP details, pooling, cross-graph feature-width
# alignment, split RNG, and graph conventions are unreported; every such
# reproduction choice is explicit in the frozen configuration below.
TPCAGAD_HP = {
    "implementation_version": "paper-reproduction-v3",
    "cache_version": "exact-structural-inputs-v1",
    "attribute_dim": 8,  # target-label-free shared SVD width
    "hidden_dim": 128,
    "audit_dim": 128,
    "num_layers": 1,
    "fingerprint_hops": 1,
    "pool": "mean_exclude_center",
    "degree_normalization": "graph_max",
    "core_normalization": "graph_max",
    # Layer-specific audit heads make Eq. 12 executable. "shared_static"
    # preserves literal Eq. 5 / Algorithm 1 and yields zero consistency loss.
    "compatibility_mode": "shared_static",
    # The paper writes a sum in Eq. (12) but gives no reduction. Averaging over
    # edges and layers keeps lambda independent of graph size.
    "consistency_reduction": "mean_edges_layers",
    # The paper is single-source. The benchmark's requested multi-source
    # extension optimizes the equal-weight mean source loss once per epoch.
    "multi_source_reduction": "macro_mean",
    "tau": 0.25,
    "alpha": 1.0,
    "consistency_lambda": 1.0,
    "mlp_depth": 2,
    "activation": "PReLU",
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "source_split": (0.3, 0.1, 0.6),
    "selection_metric": "source_val_macro_auroc",
    # Full validation requires another complete pass over every source.
    # Validate at epoch 1, this interval, and the final epoch.
    "validation_interval": 20,
    "score_mode": "raw_eq9",
    "base_graph": "binary_undirected_loop_free",
    "isolated_score": 0.0,
    # Exact runtime controls: all nodes/edges are retained. Training recomputes
    # edge gathers in backward; large-target inference keeps node tensors on CPU.
    "edge_chunk": 250_000,
    "node_chunk": 32_768,
    "stream_node_threshold": 200_000,
    "stream_edge_threshold": 5_000_000,
    # Exact degree/core/clustering preprocessing uses Numba above this size.
    # A cache hit does not require Numba on the destination machine.
    "python_structure_max_edges": 2_000_000,
    "cache": True,
}

# SAARCS (AAAI 2026) paper-only reproduction. No official code is released.
# Paper protocol: multi-source supervised training followed by target inference
# with n_k=10 known-normal context nodes; context nodes are removed from query
# evaluation, so this method is registered as normal few-shot rather than
# zero-shot. Eq. (1)--(4) use target-label-free truncated SVD at high raw width
# and a fixed random adapter at low raw width, a per-column min-max copy only
# for composite-score calculation, and reorder the original projected columns.
# This follows the paper's wording and the released ARC predecessor. The
# supplied binary graph is preserved instead of silently adding reverse edges.
# Eq. (5)--(8) use exact full-graph CSR propagation and node-wise hop attention;
# the printed quadratic Eq. (6) is not materialized because it conflicts with
# the per-node hop-weight text and is infeasible on the paper's own graphs.
# W_Z/W_S/W_T share the single d_a width required by Eqs. (6)--(8).
SAARCS_HP = {
    "implementation_version": "paper-reproduction-v5",
    "cache_version": "arc-input-svd-random64-composite-v5",
    "feature_dim": 128,
    # Latent width is unreported by SAARCS; ARC uses width 1024.
    "latent_dim": 1024,
    "num_hops": 5,
    "num_context": 10,
    # None matches ARC's released training protocol: use every labeled anomaly
    # and sample an equally sized, disjoint normal query set on each source.
    "query_per_class": 64,
    "spatial_alpha": 0.25,
    "standardization": "minmax",
    "projection": "svd_random_adapter",
    "projection_sign": "max_abs_positive",
    "alignment_output": "projected",
    "raw_feature_norm": "arc_conditional",
    "base_graph": "binary_input_loop_free",
    "propagation_self_loops": "arc_conditional",
    "hop_attention_mode": "node_wise",
    "leaky_relu_slope": 0.2,
    "margin": 0.7,
    "lr": 1e-5,
    "weight_decay": 1e-2,
    # Exact runtime controls: propagation and feature sorting retain every node
    # and edge; only temporary edge/query tensors are chunked.
    "edge_chunk": 250_000,
    "query_chunk": 32_768,
    "cache": True,
}


# GAD-MoRE official release configuration.
# Model/training values below come from GAD-MoRE-main/main.py. The released
# code calls num_layers=4 but constructs three Linear layers; the port keeps
# that exact behavior in ggad/vendor/gadmore/model.py.
GADMORE_OFFICIAL_HP = {
    "implementation_version": "official-release-port-v1",
    "dim": 32,
    "hidden_dim": 1024,
    "num_layers": 4,
    "dropout_rate": 0.0,
    "activation": "ELU",
    "num_hops": 2,
    "num_experts": 5,
    "expert_hidden_dim": 256,
    "top_k": 2,
    "init_curvs": None,  # official expansion: [0,-0.5,-1,0.5,1]
    "gate_temperature": 0.7,
    "gate_noise_type": "gumbel",
    "gate_noise_std": 0.0,
    "memory_size": 32,
    "epochs": 40,
    "lr": 5e-5,
    "weight_decay": 5e-5,
    "contrastive_weight": 0.1,
    "contrastive_temperature": 0.1,
    "w_embed": 1.0,
    "w_feature": 0.5,
    "w_structure": 0.1,
    "w_gate": 0.01,
    "w_message": 1.0,
}

GADMORE_RUNTIME_HP = {
    # Scalability extensions. They preserve all nodes/edges in MCFA,
    # propagation and inference. Incremental PCA is used only when exact
    # sklearn PCA would require holding an extreme node matrix in RAM.
    "large_pca_node_threshold": 200_000,
    "large_pca_batch": 50_000,
    "preprocess_node_chunk": 50_000,
    "stream_edge_threshold": 5_000_000,
    "edge_chunk": 500_000,
    "inference_node_chunk": 8_192,
    # The official structure term allocates N x N. Up to the threshold we
    # evaluate exactly in row chunks. Above it, uniform pair sampling is an
    # unbiased estimate of the same fractional-adjacency BCE.
    "structure_exact_max_nodes": 5_000,
    "structure_row_chunk": 256,
    "structure_pair_samples": 100_000,
    "structure_sample_chunk": 16_384,  # checkpointed exact gradient block
    # Official code samples 30% of nodes once N>8000, but that can still form
    # an impractical quadratic matrix on million-node graphs. The cap is the
    # only additional approximation; the induced graph objective is otherwise
    # evaluated exactly using row chunks and sparse positive pairs.
    "contrastive_sample_threshold": 8_000,
    "contrastive_sample_ratio": 0.3,
    "contrastive_sample_cap": 4_096,
    "contrastive_positive_chunk": 16_384,  # checkpointed; all positive edges retained
    "contrastive_row_chunk": 512,  # physical block only
    "contrastive_logical_chunk": 1_000,  # released loss weighting
}

# The runner consumes one mapping, while the official model/training values
# remain separately auditable from execution-only memory controls.
GADMORE_HP = {**GADMORE_OFFICIAL_HP, **GADMORE_RUNTIME_HP}

# NeighborDiv (arXiv:2605.20879v1) configuration.  This is the training-free
# method proposed in the GGAD paper; it never reads source graphs or labels.
NEIGHBORDIV_HP = {
    "implementation_version": "v2-exact-gpu-moment",
    # Paper defaults (Sec. 3 and App. B.4).
    "svd_dim": 8,
    "threshold_lambda": 1.0,  # only for optional binary predictions; AUC/AP do not use it
    "pair_mode": "auto",  # "full" (paper main table) | "sample" (App. E.3) | "auto"
    "sample_pairs": 100,  # paper's scalable sampling ablation
    # The paper specifies truncated SVD but not its numerical solver/seed.
    # ARPACK gives a stable top-r subspace; these are explicit reproduction choices.
    "svd_algorithm": "arpack",
    "svd_seed": 0,
    "compute_dtype": "float64",  # stable variance/ranking; set float32 only for extreme RAM pressure
    # Million-node targets project/store features in float32.  The exact Full
    # pair statistic is unchanged; only floating-point precision differs.
    "large_graph_node_threshold": 1_000_000,
    "large_graph_compute_dtype": "float32",
    "moment_round_decimals": 14,  # removes sub-ULP cancellation artifacts in exact moment evaluation
    "keep_input_self_loops": True,  # use A exactly as stored; never add a self-loop
    # After trying the exact CUDA path, auto uses the paper's k=100 approximation
    # beyond this total number of unordered pairs; smaller graphs use CPU Full.
    "auto_full_pair_limit": 100_000_000,
    # Exact Full backend.  auto uses one CUDA CSR-SpMM for large graphs when
    # the conservative memory estimate fits, otherwise retains CPU Full/sample.
    "full_backend": "auto",  # "auto" | "cpu" | "torch"
    "gpu_min_edges": 5_000_000,
    "gpu_compute_dtype": "float32",
    "gpu_memory_fraction": 0.80,
    "gpu_output_chunk": 262_144,
    # Memory controls.  Full is computed by an exact moment identity rather
    # than materializing d_i x d_i Gram matrices; row and outer-product blocks
    # only bound temporary memory and do not approximate nodes, edges or pairs.
    "node_chunk": 32_768,
    "outer_memory_mb": 256,
    "cache": True,
}

# ProMoS (Generalist Graph Anomaly Detection via Prototype-Based Distillation)
# configuration. Published defaults from the released main_train.py are kept:
# dim=64, epochs=10, lr/proto_lr=5e-3, 20 experts, 20 prototypes, beta=1.0,
# mu=0.6, lambda=0.5. The released entry point consumes offline GCA embeddings;
# our unified runner instead pretrains one GCA on the selected source graphs at
# run time, freezes it, and keeps all representations in memory. Nothing under
# emb/ or checkpoints/ is read or written. The GCA architecture/augmentation
# follows the cited PyG-SSL implementation. Its YAML defaults provide the
# otherwise-unreported teacher optimizer, 50 epochs, projection width, and tau.
PROMOS_HP = {
    "dim": 64,
    "hidden_dim": 128,
    "dropout": 0.1,
    "lr": 5e-3,
    "proto_lr": 5e-3,
    "weight_decay": 5e-5,
    "num_prototypes": 20,
    "num_experts": 20,
    "beta": 1.0,
    "mu": 0.6,
    "lam": 0.5,
    "gca_seed": 0,  # one frozen teacher, shared by all student trials as in official code
    "gca_epochs": 50,
    "gca_quick_epochs": 2,
    "gca_projection_dim": 32,
    "gca_tau": 0.4,
    "gca_lr": 1e-4,
    "gca_weight_decay": 5e-4,
    "gca_edge_drop_1": 0.3,  # actual GCATrainer values (the YAML values are unused there)
    "gca_edge_drop_2": 0.4,
    "gca_feature_drop_1": 0.1,
    "gca_feature_drop_2": 0.0,
    "gca_contrast_batch": 1024,
    "gca_loss_max_nodes": 0,  # 0 = exact all-node negative pool; positive value is an explicit approximation
    "gca_stream_edge_threshold": 5_000_000,
    # Dependency-free port of the released faiss.Kmeans defaults.
    "kmeans_seed": 1234,
    "kmeans_max_points_per_centroid": 256,  # official Faiss default (20 * 256 = 5120 rows)
    "kmeans_assignment_chunk": 1024,  # exact memory chunk; does not change assignments
    "stream_edge_threshold": 5_000_000,
    "edge_chunk": 500_000,
    "node_chunk": 200_000,
    "grad_clip": 0.0,  # released ProMoS applies no gradient clipping
}

# Zero-GAD (MM 2025) configuration.
# This block follows the released code. Section 5.4 reports alpha=2, beta=0.5,
# whereas released train.py defaults to alpha_l=3, beta=0.5. The paper also
# trains on Facebook/Flickr/BlogCatalog/ACM and reports five runs, while the
# released entry point hard-codes Flickr/tolokers/YelpChi/ACM and one seed=42.
# Our source sets and repeated seeds remain benchmark-owned protocol choices.
# Official default path in the released train.py/model.py uses GCN encoder and
# GCN decoder, SVD feature dimension 8, hidden 256, 3 encoder layers, 2 decoder
# layers, SCE(alpha=3), beta=0.5, lr=1e-3, weight_decay=5e-4, in_drop=0.2.
# The released preprocessing applies SVD to every graph, then a dense
# Laplacian-Fourier transform. Dense eigendecomposition is cubic in node count
# and requires several dense N x N buffers. In strict mode, graphs above 15k
# nodes (including pubmed/questions/t_finance/YelpChi* and benchmark BIG) are
# marked unsupported (nan) before SVD/Fourier allocation. Set
# big_fourier_policy="svd_only_extension" only for a separately named,
# non-paper large-graph engineering ablation.
ZEROGAD_HP = {
    "unifeat": 8,
    "num_hidden": 256,
    "num_layers": 3,
    "decoder_layers": 2,
    "in_drop": 0.2,
    "attn_drop": 0.1,  # official arg retained for provenance; GCN path ignores it
    "residual": False,
    "norm": None,
    "activation": "prelu",
    "loss_fn": "sce",
    "alpha_l": 3,
    "beta": 0.5,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "fourier": True,
    "fourier_partition_datasets": ["tfinance", "elliptic", "questions"],
    "fourier_partitions": 5,
    "fourier_min_partition_nodes": 512,
    "fourier_scipy_driver": "evd",
    "fourier_jitter": 1e-8,
    "fourier_failure_policy": "raise",
    "fourier_max_nodes": 15_000,
    "big_fourier_policy": "strict_skip",  # "strict_skip" | "svd_only_extension"
    "feature_seed": 42,
    "stream_node_threshold": 200_000,
    "stream_edge_threshold": 5_000_000,
    "edge_chunk": 500_000,
    "grad_clip": 0.0,
    "cache": True,
}

# OWLEYE (arXiv:2601.19102 / ICLR-2026 demo) configuration.
# Official saved config in ICLR-2026-OWLEYE-main/best_param_pt/OWLEYE.json:
# lr=3.129648e-5, drop_rate=0.2, h_feats=512, num_hops=4,
# weight_decay=2.848035868e-4, num_layers=3, activation=ELU,
# n_support=2000, mask_ratio=0.7, temperature=0.001, tau=1.
# The runner keeps benchmark-owned data splits/evaluation and ports the model
# logic only. BIG targets follow the common full-graph protocol: exact full-edge
# propagation, exact node-chunk inference, all marked nodes evaluated, and a
# hard failure instead of a silent skip.
OWLEYE_HP = {
    "in_feats": 64,
    "h_feats": 512,
    "num_layers": 3,
    "activation": "ELU",
    "drop_rate": 0.0,  # released JSON's 0.2 is ignored by the released constructor
    "num_hops": 4,
    "beta": 0.01,
    "lr": 3.12964801067075e-05,
    "weight_decay": 0.0002848035868435802,
    "n_support": 2000,
    "mask_ratio": 0.7,
    "temperature": 0.001,
    "tau": 1.0,
    "st_dim": 10,
    "test_patterns": 10,
    "row_norm_datasets": ["Amazon", "YelpChi"],
    "large_pca_threshold": 200_000,
    "pca_batch": 50_000,
    "stream_node_threshold": 200_000,
    "stream_edge_threshold": 5_000_000,
    "edge_chunk": 500_000,
    # Query x 2000-support attention is the inference memory bottleneck.
    "node_chunk": 4_096,
    "grad_clip": 0.0,
    "cache": True,
}

# TFM4GAD (ICLR 2026) official graph-to-table augmentation with TabPFN-v2.
# The model itself has no source pretraining: it fits an in-context classifier
# on the target support set.  For benchmark comparability, run.py passes the
# global SHOT as k normal + k anomaly (the release/paper instead uses a fixed
# 80 normal + 20 anomaly context).  Graph preprocessing and all augmentation
# hyperparameters below reproduce TabPFN/tfm4gad/icl.conf.yaml.  The checkpoint
# is stored under assets/checkpoints/ within this release.
_TFM4GAD_DIR = Path(__file__).resolve().parents[1] / "assets" / "checkpoints"
TFM4GAD_HP = {
    "implementation_version": "official-tabpfn-v2-scipy-port-v1",
    "model_version": "v2",
    "model_cache_dir": str(_TFM4GAD_DIR),
    "model_ckpt": str(_TFM4GAD_DIR / "tabpfn-v2-classifier.ckpt"),
    "model_repo": "Prior-Labs/TabPFN-v2-clf",
    "model_filename": "tabpfn-v2-classifier.ckpt",
    "hf_endpoint": "https://huggingface.co",
    # Official paper datasets stay within the v2 pretraining feature regime.
    # Our high-dimensional injected citation graphs exceed 500 augmented
    # columns; this flag retains TabPFN's own feature subsampling instead of
    # introducing a non-paper PCA/SVD projection.
    "ignore_pretraining_limits": True,
    "inference_batch_size": 2048,
    "neighbor_type": "bwgnn",
    "neighbor_num_layers": 2,
    "sage_agg": "mean",
    "sage_standard_scale": False,
    "dataset_neighbor_overrides": {},
    "degree_log": True,
    "pagerank_alpha": 0.85,
    "pagerank_max_iterations": 100,
    "pagerank_tol": 1e-6,
    "pagerank_log": True,
    "lap_pe_k": 16,
    "lap_pe_approx_threshold": 100_000,
    "lap_pe_approx_method": "randomized_svd",
    "lap_pe_seed": 0,
}

RESULTS_DIR = "ggad/results"


def sources(mode=None, src_type=None):
    """Return the configured source graphs for the selected domain and mode."""
    mode = mode or SOURCE_MODE
    single, multi = (
        (REAL_SINGLE, REAL_MULTI) if (src_type or SRC_TYPE) == "real" else (FAKE_SINGLE, FAKE_MULTI)
    )
    return single if mode == "single" else multi


def targets(tgt_type=None):
    """Return the configured target graphs for a domain."""
    return REAL_TARGETS if (tgt_type or TGT_TYPE) == "real" else FAKE_TARGETS
