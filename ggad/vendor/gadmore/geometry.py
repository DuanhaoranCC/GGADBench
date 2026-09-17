"""Differentiable constant-curvature operations for GAD-MoRE.

The implementation follows the kappa-stereographic model used by the official
GraphMoRE implementation (the closest released predecessor cited by GAD-MoRE).
It supports negative, zero and positive curvature with a single coordinate
system and does not require geoopt at benchmark runtime.
"""

import math

import torch

EPS = 1e-7


def _norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(EPS)


def _atanh(x):
    x = x.clamp(min=-1.0 + 4 * EPS, max=1.0 - 4 * EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project(x, curvature, kind):
    """Project stereographic coordinates into the numerically valid region."""
    if kind != "hyperbolic":
        return x
    c = curvature.abs().clamp_min(EPS)
    max_norm = (1.0 - 4 * EPS) / torch.sqrt(c)
    norm = _norm(x)
    scale = torch.minimum(torch.ones_like(norm), max_norm / norm)
    return x * scale


def expmap0(v, curvature, kind):
    """Exponential map at the origin in kappa-stereographic coordinates."""
    if kind == "euclidean":
        return v
    c = curvature.abs().clamp_min(EPS)
    sqrt_c = torch.sqrt(c)
    norm = _norm(v)
    # In the kappa-stereographic model the conformal factor at the origin is 2,
    # so exp_0 uses sqrt(|k|) * ||v|| / 2.  Missing this factor pushes spherical
    # branches to the chart pole for ordinary embeddings and destabilizes
    # curvature learning.
    arg = sqrt_c * norm / 2.0
    if kind == "hyperbolic":
        factor = torch.tanh(arg) / (sqrt_c * norm)
    else:
        # The stereographic chart has a pole at pi/2.  Clipping only prevents
        # a numerical chart singularity; it does not change ordinary inputs.
        arg = arg.clamp(max=math.pi / 2 - 1e-4)
        factor = torch.tan(arg) / (sqrt_c * norm)
    return project(v * factor, curvature, kind)


def logmap0(x, curvature, kind):
    """Logarithmic map at the origin, inverse of :func:`expmap0`."""
    if kind == "euclidean":
        return x
    x = project(x, curvature, kind)
    c = curvature.abs().clamp_min(EPS)
    sqrt_c = torch.sqrt(c)
    norm = _norm(x)
    arg = sqrt_c * norm
    if kind == "hyperbolic":
        factor = 2.0 * _atanh(arg) / arg
    else:
        factor = 2.0 * torch.atan(arg) / arg
    return x * factor


def mobius_add(x, y, curvature, kind):
    """Generalized Mobius addition for signed constant curvature."""
    if kind == "euclidean":
        return x + y
    k = -curvature.abs() if kind == "hyperbolic" else curvature.abs()
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    den = (1 - 2 * k * xy + k.square() * x2 * y2).clamp_min(EPS)
    return project(num / den, curvature, kind)


def mobius_matvec(weight, x, curvature, kind):
    """Matrix-vector multiplication performed through the origin tangent."""
    tangent = logmap0(x, curvature, kind)
    mapped = torch.nn.functional.linear(tangent, weight)
    return expmap0(mapped, curvature, kind)


def distance(x, y, curvature, kind):
    """Geodesic distance between broadcast-compatible manifold points."""
    if kind == "euclidean":
        return torch.linalg.vector_norm(x - y, dim=-1)
    c = curvature.abs().clamp_min(EPS)
    delta = mobius_add(-x, y, curvature, kind)
    z = torch.sqrt(c) * _norm(delta).squeeze(-1)
    if kind == "hyperbolic":
        return 2.0 * _atanh(z) / torch.sqrt(c)
    return 2.0 * torch.atan(z) / torch.sqrt(c)


def tangent_distance(x, y, curvature, kind):
    """Distance after mapping tangent vectors to an expert manifold."""
    return distance(expmap0(x, curvature, kind), expmap0(y, curvature, kind), curvature, kind)


def pairwise_min_distance(query, memory, curvature, kind, query_chunk=4096, memory_chunk=512):
    """Exact nearest-memory geodesic distance with bounded temporaries."""
    if memory.numel() == 0:
        raise ValueError("memory must be non-empty")
    outputs = []
    memory_points = expmap0(memory, curvature, kind)
    for qs in range(0, query.shape[0], query_chunk):
        q = expmap0(query[qs : qs + query_chunk], curvature, kind)
        best = None
        for ms in range(0, memory.shape[0], memory_chunk):
            m = memory_points[ms : ms + memory_chunk]
            distances = distance(q[:, None, :], m[None, :, :], curvature, kind)
            block = distances.min(dim=1).values
            best = block if best is None else torch.minimum(best, block)
        outputs.append(best)
    return torch.cat(outputs, dim=0)


def pairwise_min_distance_sq(query, memory, curvature, kind, query_chunk=4096, memory_chunk=512):
    """Backward-compatible squared nearest-memory distance."""
    return pairwise_min_distance(query, memory, curvature, kind, query_chunk, memory_chunk).square()
