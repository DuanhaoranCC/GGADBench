"""Official MCFA and adjacency propagation with bounded large-graph paths."""

import numpy as np
import scipy.sparse as sp
import torch
from geoopt import Stereographic
from sklearn.decomposition import PCA, IncrementalPCA

ROW_NORMALIZED = {"Amazon", "YelpChi", "tolokers", "tfinance", "t_finance"}
NO_SELF_LOOP = {"YelpChi", "Facebook"}


def _row_normalize(features):
    matrix = sp.csr_matrix(features)
    rowsum = np.asarray(matrix.sum(1)).ravel()
    inv = np.zeros_like(rowsum)
    nonzero = rowsum != 0
    inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(inv).dot(matrix)


def _dense_float32(features):
    if sp.issparse(features):
        return features.toarray().astype(np.float32, copy=False)
    return np.asarray(features, dtype=np.float32)


def _undirected_adjacency(adj):
    source = sp.coo_matrix(adj)
    matrix = sp.coo_matrix(
        (np.ones(source.nnz), (source.row, source.col)),
        shape=source.shape,
    )
    matrix = matrix + matrix.T.multiply(matrix.T > matrix)
    matrix = matrix - matrix.multiply(matrix.T > matrix)
    return matrix.tocsr()


def _normalized_laplacian(adj):
    matrix = _undirected_adjacency(adj)
    degree = np.asarray(matrix.sum(1)).ravel()
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = degree[nonzero] ** -0.5
    return sp.eye(matrix.shape[0]) - (sp.diags(inv).dot(matrix).dot(sp.diags(inv)))


def _laplacian_scores(features, laplacian):
    scores = []
    for column in range(features.shape[1]):
        values = features[:, column]
        values = (values - values.mean()) / (values.std() + 1e-9)
        scores.append(float(values @ laplacian.dot(values)))
    return np.asarray(scores)


def _select_branch(candidates, laplacian, output_dim):
    """Select low-Laplacian-score columns and preserve the MCFA branch width.

    The release assumes every dataset has enough input dimensions. Benchmark
    graphs such as tolokers violate that assumption (10 raw dimensions versus
    a 16-dimensional Euclidean branch). Zero padding is the only extension
    that keeps the shared branch layout without inventing or duplicating
    information.
    """
    indices = np.argsort(_laplacian_scores(candidates, laplacian))[:output_dim]
    selected = np.asarray(candidates[:, indices], dtype=np.float32)
    missing = int(output_dim) - selected.shape[1]
    if missing > 0:
        selected = np.pad(selected, ((0, 0), (0, missing)), mode="constant")
    return np.ascontiguousarray(selected, dtype=np.float32)


def _pca(features, components, hp):
    if features.shape[1] <= components:
        return features
    if features.shape[0] <= hp["large_pca_node_threshold"]:
        return (
            PCA(n_components=components, random_state=0)
            .fit_transform(features)
            .astype(np.float32, copy=False)
        )

    batch = max(int(hp["large_pca_batch"]), components)
    reducer = IncrementalPCA(n_components=components, batch_size=batch)
    starts = list(range(0, features.shape[0], batch))
    if len(starts) > 1 and features.shape[0] - starts[-1] < components:
        starts.pop()
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else features.shape[0]
        reducer.partial_fit(features[start:end])
    output = np.empty((features.shape[0], components), dtype=np.float32)
    for start in range(0, features.shape[0], batch):
        end = min(start + batch, features.shape[0])
        output[start:end] = reducer.transform(features[start:end]).astype(np.float32, copy=False)
    return output


def _manifold_roundtrip(features, curvature, chunk):
    manifold = Stereographic(k=curvature)
    output = np.empty_like(features, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk):
            current = torch.from_numpy(np.ascontiguousarray(features[start : start + chunk]))
            projected = manifold.proju(manifold.origin(current.shape), current)
            output[start : start + len(current)] = manifold.logmap0(
                manifold.expmap0(projected)
            ).numpy()
    return output


def mcfa(name, adj, features, hp):
    """Match the official feat_alignment; no labels or persistent caches."""
    if name in ROW_NORMALIZED and sp.issparse(features):
        features = _row_normalize(features)
    features = _dense_float32(features)
    laplacian = _normalized_laplacian(adj)

    dims = int(hp["dim"])
    euclidean_dims = (dims * 4) // 8
    curved_dims = dims // 8
    euclidean_dims += dims - euclidean_dims - 4 * curved_dims
    branches = []

    candidates = _pca(features, euclidean_dims * 4, hp)
    branches.append(_select_branch(candidates, laplacian, euclidean_dims))

    for curvature in (-1.0, 1.0, -0.5, 0.5):
        tangent = _manifold_roundtrip(features, curvature, int(hp["preprocess_node_chunk"]))
        candidates = _pca(tangent, curved_dims * 4, hp)
        branches.append(_select_branch(candidates, laplacian, curved_dims))

    aligned = np.ascontiguousarray(np.concatenate(branches, axis=1), dtype=np.float32)
    if aligned.shape[1] != dims:
        raise RuntimeError(
            f"MCFA produced {aligned.shape[1]} dimensions for {name}; " f"expected {dims}"
        )
    return aligned


def normalize_adjacency(name, adj):
    """Official conditional self-loop and normalized transposed adjacency."""
    matrix = sp.csr_matrix(adj)
    if name not in NO_SELF_LOOP:
        matrix = matrix + sp.eye(matrix.shape[0], format="csr")
    rowsum = np.asarray(matrix.sum(1)).ravel()
    inv = np.zeros_like(rowsum)
    nonzero = rowsum > 0
    inv[nonzero] = rowsum[nonzero] ** -0.5
    matrix = matrix.tocoo()
    values = matrix.data.astype(np.float32, copy=False) * inv[matrix.row] * inv[matrix.col]
    # normalize_adj in the release returns (A D)^T D.
    return sp.coo_matrix((values, (matrix.col, matrix.row)), shape=matrix.shape)


def to_torch_sparse(adj, device=None):
    matrix = sp.coo_matrix(adj, dtype=np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    output = torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()
    return output if device is None else output.to(device)


def _stream_spmm(row, col, val, features, device, edge_chunk):
    output = torch.zeros_like(features)
    for start in range(0, len(val), edge_chunk):
        end = min(start + edge_chunk, len(val))
        current_row = torch.from_numpy(np.asarray(row[start:end], dtype=np.int64)).to(device)
        current_col = torch.from_numpy(np.asarray(col[start:end], dtype=np.int64)).to(device)
        current_val = (
            torch.from_numpy(np.asarray(val[start:end], dtype=np.float32)).to(device).unsqueeze(1)
        )
        output.index_add_(0, current_row, features[current_col] * current_val)
    return output


def propagate_features(adj_norm, features, hops, device, hp, keep_operator=False):
    """Full-node, full-edge propagation; edge chunking is mathematically exact."""
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    stream = adj_norm.nnz >= int(hp["stream_edge_threshold"])
    operator = None if stream else to_torch_sparse(adj_norm, device)
    stream_arrays = None
    if stream:
        coo = sp.coo_matrix(adj_norm, dtype=np.float32)
        stream_arrays = (coo.row, coo.col, coo.data)
    levels = [x]
    for _ in range(hops):
        if operator is None:
            levels.append(_stream_spmm(*stream_arrays, levels[-1], device, int(hp["edge_chunk"])))
        else:
            levels.append(torch.sparse.mm(operator, levels[-1]))
    if keep_operator and operator is None:
        operator = to_torch_sparse(adj_norm, device)
    return levels, operator
