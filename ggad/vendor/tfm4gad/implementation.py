"""TFM4GAD graph-to-table augmentation and TabPFN inference.

This is a SciPy/NumPy port of the released ``TabPFN/tfm4gad`` code.  It keeps
the released mathematical operations while avoiding DGL and dense adjacency
matrices.  Sparse operations always retain every node and edge.
"""

from __future__ import annotations

import math
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from sklearn.utils.extmath import randomized_svd

_DATASET_OVERRIDES = {
    "amazon": {"type": "bwgnn", "num_layers": 1},
    "amazon-all": {"type": "bwgnn", "num_layers": 1},
    "yelpchi": {"type": "bwgnn", "num_layers": 2},
    "yelpchi-all": {"type": "bwgnn", "num_layers": 2},
    "t_finance": {"type": "sage", "num_layers": 2},
    "tfinance": {"type": "sage", "num_layers": 2},
    "tsocial": {"type": "bwgnn", "num_layers": 3},
    "tolokers": {"type": "bwgnn", "num_layers": 1},
    "elliptic": {"type": "bwgnn", "num_layers": 1},
    "dgraphfin": {"type": "bwgnn", "num_layers": 1},
    "reddit": {"type": "bwgnn", "num_layers": 1},
    "questions": {"type": "bwgnn", "num_layers": 1},
}


def resolve_augmentation_config(name: str, hp: Dict) -> Dict:
    """Merge the official YAML defaults with its dataset overrides."""
    neighbor = {
        "type": hp.get("neighbor_type", "bwgnn"),
        "num_layers": int(hp.get("neighbor_num_layers", 2)),
        "agg": hp.get("sage_agg", "mean"),
        "sc": bool(hp.get("sage_standard_scale", False)),
    }
    neighbor.update(_DATASET_OVERRIDES.get(name.lower(), {}))
    neighbor.update(hp.get("dataset_neighbor_overrides", {}).get(name, {}))
    return {
        "neighbor": neighbor,
        "degree_log": bool(hp.get("degree_log", True)),
        "pagerank_alpha": float(hp.get("pagerank_alpha", 0.85)),
        "pagerank_max_iterations": int(hp.get("pagerank_max_iterations", 100)),
        "pagerank_tol": float(hp.get("pagerank_tol", 1e-6)),
        "pagerank_log": bool(hp.get("pagerank_log", True)),
        "lap_pe_k": int(hp.get("lap_pe_k", 16)),
        "lap_pe_approx_threshold": int(hp.get("lap_pe_approx_threshold", 100_000)),
        "lap_pe_approx_method": hp.get("lap_pe_approx_method", "randomized_svd"),
        "lap_pe_seed": int(hp.get("lap_pe_seed", 0)),
    }


def prepare_bidirected_graph(adjacency) -> sp.csr_matrix:
    """Match ``dgl.to_bidirected`` then remove/add one self-loop per node."""
    adjacency = sp.csr_matrix(adjacency, dtype=np.float32)
    graph = adjacency.maximum(adjacency.T).tocsr()
    graph = (graph + sp.eye(graph.shape[0], dtype=np.float32, format="csr")).tocsr()
    if graph.nnz:
        graph.data[:] = 1.0
    graph.sort_indices()
    return graph


def _dense32(features) -> np.ndarray:
    if sp.issparse(features):
        features = features.toarray()
    return np.ascontiguousarray(features, dtype=np.float32)


def _standard_scale(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + eps
    return ((values - mean) / std).astype(np.float32, copy=False)


def _incoming_sum(graph: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    # DGL update_all aggregates source rows into destinations.  The graph is
    # bidirected, but retaining .T documents and preserves that orientation.
    return np.asarray(graph.T @ values, dtype=np.float32)


def _sage_features(graph: sp.csr_matrix, features: np.ndarray, conf: Dict) -> np.ndarray:
    aggregate = conf.get("agg", "mean")
    if aggregate not in {"sum", "mean"}:
        raise ValueError("TFM4GAD SciPy port supports the released sum/mean SAGE aggregators")
    scale = bool(conf.get("sc", False))
    current = features
    blocks = [_standard_scale(current) if scale else current]
    degree = np.asarray(graph.sum(axis=0)).reshape(-1).astype(np.float32)
    degree[degree < 1] = 1
    for _ in range(int(conf["num_layers"])):
        current = _incoming_sum(graph, current)
        if aggregate == "mean":
            current /= degree[:, None]
        if scale:
            current = _standard_scale(current)
        blocks.append(current)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _calculate_theta(degree: int) -> list[list[float]]:
    """Closed-form equivalent of the release's SymPy beta-polynomial code."""
    filters = []
    for i in range(degree + 1):
        coefficients = [0.0] * (degree + 1)
        leading = (degree + 1) * math.comb(degree, i)
        for j in range(degree - i + 1):
            power = i + j
            coefficients[power] = leading * math.comb(degree - i, j) * ((-1.0) ** j) / (2.0**power)
        filters.append(coefficients)
    return filters


def _normalized_neighbor(
    graph: sp.csr_matrix, values: np.ndarray, degree_inv_sqrt: np.ndarray
) -> np.ndarray:
    scaled = values * degree_inv_sqrt[:, None]
    return _incoming_sum(graph, scaled) * degree_inv_sqrt[:, None]


def _bwgnn_features(graph: sp.csr_matrix, features: np.ndarray, degree: int) -> np.ndarray:
    graph_degree = np.asarray(graph.sum(axis=0)).reshape(-1).astype(np.float32)
    degree_inv_sqrt = np.power(np.maximum(graph_degree, 1.0), -0.5).astype(np.float32)
    powers = [features]
    for _ in range(degree):
        previous = powers[-1]
        powers.append(previous - _normalized_neighbor(graph, previous, degree_inv_sqrt))

    outputs = []
    for theta in _calculate_theta(degree):
        current = np.zeros_like(features)
        for coefficient, basis in zip(theta, powers):
            if coefficient:
                current += np.float32(coefficient) * basis
        outputs.append(current)
    return np.concatenate(outputs, axis=1).astype(np.float32, copy=False)


def _cache_path(
    cache_dir: Optional[Path], name: str, graph: sp.csr_matrix, suffix: str
) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}_{graph.shape[0]}_{graph.nnz}_{suffix}.npy"


def _pagerank(graph: sp.csr_matrix, name: str, conf: Dict, cache_dir: Optional[Path]) -> np.ndarray:
    alpha = conf["pagerank_alpha"]
    tolerance = conf["pagerank_tol"]
    path = _cache_path(cache_dir, name, graph, f"pagerank_a{alpha:g}")
    if path is not None and path.exists():
        values = np.load(path)
    else:
        nodes = graph.shape[0]
        values = np.full(nodes, 1.0 / nodes, dtype=np.float32)
        degree = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32)
        degree[degree < 1] = 1
        reset = np.float32((1.0 - alpha) / nodes)
        for _ in range(conf["pagerank_max_iterations"]):
            previous = values.copy()
            values = (
                np.float32(alpha) * np.asarray(graph.T @ (values / degree), dtype=np.float32)
                + reset
            )
            if np.abs(values - previous).sum() < tolerance:
                break
        if path is not None:
            np.save(path, values)
    if conf["pagerank_log"]:
        values = np.log(np.float32(tolerance) + values)
    return values.astype(np.float32, copy=False)[:, None]


def _normalized_laplacian(graph: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(graph.sum(axis=0)).reshape(-1)
    inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    normalized = graph.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :])
    return (sp.eye(graph.shape[0], dtype=np.float64, format="csr") - normalized).tocsr()


def _laplacian_pe(
    graph: sp.csr_matrix, name: str, conf: Dict, cache_dir: Optional[Path]
) -> np.ndarray:
    requested = conf["lap_pe_k"]
    method = conf["lap_pe_approx_method"]
    path = _cache_path(cache_dir, name, graph, f"lappe_k{requested}_{method}")
    if path is not None and path.exists():
        return np.asarray(np.load(path), dtype=np.float32)

    nodes = graph.shape[0]
    count = min(requested, max(nodes - 1, 0))
    if count == 0:
        values = np.empty((nodes, 0), dtype=np.float32)
    else:
        laplacian = _normalized_laplacian(graph)
        if nodes > conf["lap_pe_approx_threshold"]:
            if method != "randomized_svd":
                raise ValueError(f"unsupported official LapPE approximation: {method}")
            vectors, _, _ = randomized_svd(laplacian, n_components=count + 1, random_state=0)
            values = vectors[:, 1 : count + 1]
        elif count + 1 < nodes - 1:
            eigenvalues, eigenvectors = eigs(
                laplacian,
                k=count + 1,
                which="SR",
                ncv=min(nodes, max(4 * requested, count + 2)),
                tol=1e-2,
            )
            order = eigenvalues.argsort()[1 : count + 1]
            values = eigenvectors[:, order].real
        else:
            eigenvalues, eigenvectors = np.linalg.eig(laplacian.toarray())
            order = eigenvalues.argsort()[1 : count + 1]
            values = eigenvectors[:, order].real

        rng = np.random.RandomState(conf["lap_pe_seed"])
        signs = np.where(rng.rand(values.shape[1]) > 0.5, 1.0, -1.0)
        values = np.asarray(values * signs, dtype=np.float32)
    if path is not None:
        np.save(path, values)
    return values


def augment_node_features(
    name: str, adjacency, base_features, hp: Dict, cache_dir: Optional[Path] = None
) -> np.ndarray:
    """Apply the complete released TFM4GAD graph-to-table transformation."""
    graph = prepare_bidirected_graph(adjacency)
    features = _dense32(base_features)
    conf = resolve_augmentation_config(name, hp)
    neighbor = conf["neighbor"]
    if neighbor["type"] == "sage":
        neighborhood = _sage_features(graph, features, neighbor)
    elif neighbor["type"] == "bwgnn":
        neighborhood = _bwgnn_features(graph, features, int(neighbor["num_layers"]))
    else:
        raise ValueError(f"unknown TFM4GAD neighbor augmentation {neighbor['type']!r}")

    degree = np.asarray(graph.sum(axis=0)).reshape(-1).astype(np.float32)
    degree_feature = np.log1p(degree) if conf["degree_log"] else degree
    page_rank = _pagerank(graph, name, conf, cache_dir)
    lap_pe = _laplacian_pe(graph, name, conf, cache_dir)
    return np.concatenate(
        [neighborhood, degree_feature[:, None], page_rank, lap_pe], axis=1
    ).astype(np.float32, copy=False)


def require_tabpfn():
    """Import TabPFN lazily so every other benchmark method remains usable."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as error:
        raise ImportError(
            "TFM4GAD requires TabPFN. Install it with: python -m pip install tabpfn"
        ) from error
    return TabPFNClassifier


def _download_checkpoint(checkpoint: Path, hp: Dict) -> None:
    """Download the public v2 checkpoint to the configured local asset path."""
    if checkpoint.exists():
        return
    endpoint = str(hp.get("hf_endpoint", "https://huggingface.co")).rstrip("/")
    repository = hp.get("model_repo", "Prior-Labs/TabPFN-v2-clf")
    filename = hp.get("model_filename", "tabpfn-v2-classifier.ckpt")
    url = f"{endpoint}/{repository}/resolve/main/{filename}"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
    print(f"  [tfm4gad] downloading TabPFN-v2 -> {checkpoint}")
    try:
        with urllib.request.urlopen(url) as response, open(partial, "wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        os.replace(partial, checkpoint)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def create_tabpfn_classifier(device: str, hp: Dict):
    """Create the released classifier, downloading its v2 weights if absent."""
    endpoint = hp.get("hf_endpoint")
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
    model_cache_dir = hp.get("model_cache_dir")
    if model_cache_dir:
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(Path(model_cache_dir).resolve())
    TabPFNClassifier = require_tabpfn()

    checkpoint = os.environ.get("TFM4GAD_MODEL_CKPT") or hp.get("model_ckpt")
    classifier_overrides = {
        "device": device,
        "ignore_pretraining_limits": bool(hp.get("ignore_pretraining_limits", False)),
    }
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        _download_checkpoint(checkpoint_path, hp)
        checkpoint = str(checkpoint_path)
        try:
            from tabpfn.constants import ModelVersion

            version = ModelVersion.V2_5 if "v2.5" in checkpoint else ModelVersion.V2
            return TabPFNClassifier.create_default_for_version(
                version, model_path=checkpoint, **classifier_overrides
            )
        except (ImportError, AttributeError):
            return TabPFNClassifier(model_path=checkpoint, **classifier_overrides)

    # Kept for custom configs; the benchmark config always supplies an explicit
    # project-local v2 checkpoint path.
    return TabPFNClassifier(**classifier_overrides)


def predict_positive_batched(
    classifier, features: np.ndarray, indices: Iterable[int], batch_size: int
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.empty(indices.size, dtype=np.float64)
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        probabilities = np.asarray(classifier.predict_proba(features[batch_indices]))
        scores[start : start + batch_indices.size] = (
            probabilities[:, 1]
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2
            else probabilities.reshape(-1)
        )
    return scores
