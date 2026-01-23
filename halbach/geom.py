from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

__all__ = [
    "ParamMap",
    "build_param_map",
    "build_symmetry_indices",
    "center_layer_indices",
    "build_roi_points",
    "build_r_bases_from_vars",
    "pack_grad",
    "pack_x",
    "unpack_x",
    "sample_sphere_surface_fibonacci",
    "sample_sphere_surface_random",
]


def center_layer_indices(K: int, n_fix: int) -> NDArray[np.int_]:
    """Return the center layer indices to fix."""
    if n_fix < 0 or n_fix > K:
        raise ValueError(f"n_fix must be between 0 and K, got {n_fix}")
    if n_fix % 2 != 0:
        raise ValueError(f"n_fix must be even, got {n_fix}")
    if n_fix == 0:
        return np.array([], dtype=np.int_)
    if K % 2 != 0:
        raise ValueError("K must be even when fixing center layers")
    half = K // 2
    start = half - n_fix // 2
    end = half + n_fix // 2
    return np.arange(start, end, dtype=np.int_)


def build_symmetry_indices(
    K: int,
    *,
    n_fix: int = 4,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Build z-symmetry indices with fixed center layers."""
    fixed_center = center_layer_indices(K, n_fix)
    half = K // 2
    lower_end = half - n_fix // 2
    if lower_end < 0:
        raise ValueError(f"Invalid n_fix={n_fix} for K={K}")
    lower_var = np.arange(0, lower_end)
    upper_var = (K - 1) - lower_var
    return lower_var, upper_var, fixed_center


@dataclass(frozen=True)
class ParamMap:
    free_alpha_idx: NDArray[np.int_]
    free_r_idx: NDArray[np.int_]
    free_alpha_mask: NDArray[np.bool_]
    free_r_mask: NDArray[np.bool_]
    lower_var: NDArray[np.int_]
    upper_var: NDArray[np.int_]
    lower_to_upper: NDArray[np.int_]
    fixed_k_radius: NDArray[np.int_]


def build_param_map(R: int, K: int, *, n_fix_radius: int) -> ParamMap:
    lower_var, upper_var, fixed_k = build_symmetry_indices(K, n_fix=n_fix_radius)
    free_alpha_mask = np.ones(R * K, dtype=bool)
    free_alpha_idx = np.arange(R * K, dtype=np.int_)

    free_r_mask = np.ones(K, dtype=bool)
    if fixed_k.size:
        free_r_mask[fixed_k] = False
    free_r_idx = lower_var[free_r_mask[lower_var]]

    lower_to_upper = np.full(K, -1, dtype=np.int_)
    if lower_var.size:
        lower_to_upper[lower_var] = upper_var

    return ParamMap(
        free_alpha_idx=free_alpha_idx,
        free_r_idx=free_r_idx,
        free_alpha_mask=free_alpha_mask,
        free_r_mask=free_r_mask,
        lower_var=lower_var,
        upper_var=upper_var,
        lower_to_upper=lower_to_upper,
        fixed_k_radius=fixed_k,
    )


def sample_sphere_surface_fibonacci(
    n: int,
    radius: float,
    *,
    seed: int | None = None,
    half_x: bool = False,
) -> FloatArray:
    if n <= 0:
        raise ValueError("n must be >= 1")
    offset = 0.0
    if seed is not None:
        rng = np.random.default_rng(seed)
        offset = float(rng.uniform(0.0, 2.0 * math.pi))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden * i + offset
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    if half_x:
        x = np.abs(x)
    pts = np.column_stack([x, y, z]) * float(radius)
    return np.ascontiguousarray(pts, dtype=np.float64)


def sample_sphere_surface_random(
    n: int,
    radius: float,
    *,
    seed: int = 0,
    half_x: bool = False,
) -> FloatArray:
    if n <= 0:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(size=(n, 3))
    norms = np.linalg.norm(vec, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    vec = vec / norms[:, None]
    if half_x:
        vec[:, 0] = np.abs(vec[:, 0])
    pts = vec * float(radius)
    return np.ascontiguousarray(pts, dtype=np.float64)


RoiMode = Literal["volume-grid", "volume-subsample", "surface-fibonacci", "surface-random"]


def build_roi_points(
    roi_r: float,
    roi_step: float,
    *,
    mode: RoiMode = "volume-grid",
    n_samples: int = 300,
    seed: int = 0,
    half_x: bool = False,
) -> FloatArray:
    """Generate ROI points for optimization."""
    if mode == "surface-fibonacci":
        return sample_sphere_surface_fibonacci(n_samples, roi_r, seed=seed, half_x=half_x)
    if mode == "surface-random":
        return sample_sphere_surface_random(n_samples, roi_r, seed=seed, half_x=half_x)

    xs = np.linspace(-roi_r, roi_r, 2 * int(np.ceil(roi_r / roi_step)) + 1)
    ys = xs
    zs = xs
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    mask = (X**2 + Y**2 + Z**2) <= roi_r**2 + 1e-15
    P = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float64)
    P = np.ascontiguousarray(P)
    if mode == "volume-grid":
        return P
    if mode == "volume-subsample":
        npts = int(P.shape[0])
        if n_samples <= 0 or n_samples >= npts:
            return P
        rng = np.random.default_rng(seed)
        idx = rng.choice(npts, size=n_samples, replace=False)
        return np.ascontiguousarray(P[idx], dtype=np.float64)
    raise ValueError(f"Unsupported ROI mode: {mode}")


def build_r_bases_from_vars(
    r_vars: NDArray[np.float64],
    K: int,
    r0: float,
    lower_var: NDArray[np.int_],
    upper_var: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Build z-symmetric r_bases from lower-half variables."""
    r = np.full(K, r0, dtype=np.float64)
    for j, k_low in enumerate(lower_var):
        k_up = upper_var[j]
        r[k_low] = r_vars[j]
        r[k_up] = r_vars[j]
    return r


def pack_x(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    param_map: ParamMap,
) -> NDArray[np.float64]:
    """Concatenate x from all alphas and symmetric r_bases variables."""
    alpha_free = alphas.ravel()[param_map.free_alpha_idx]
    r_free = r_bases[param_map.free_r_idx]
    return np.concatenate([alpha_free, r_free])


def unpack_x(
    x: NDArray[np.float64],
    alphas0: NDArray[np.float64],
    r_bases0: NDArray[np.float64],
    param_map: ParamMap,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct full arrays while keeping fixed layers at baseline values."""
    alphas = np.array(alphas0, dtype=np.float64, copy=True)
    r_bases = np.array(r_bases0, dtype=np.float64, copy=True)
    n_alpha = int(param_map.free_alpha_idx.size)
    al_flat = alphas.ravel()
    al_flat[param_map.free_alpha_idx] = x[:n_alpha]
    r_vars = x[n_alpha:]
    if r_vars.size != param_map.free_r_idx.size:
        raise ValueError("x size does not match parameter map")
    for idx, k_low in enumerate(param_map.free_r_idx):
        r_bases[k_low] = r_vars[idx]
        k_up = param_map.lower_to_upper[k_low]
        if k_up >= 0:
            r_bases[k_up] = r_vars[idx]
    return alphas, r_bases


def pack_grad(
    grad_alphas: NDArray[np.float64],
    grad_r_bases: NDArray[np.float64],
    param_map: ParamMap,
) -> NDArray[np.float64]:
    """Pack full gradients into reduced x-space."""
    g_alpha = grad_alphas.ravel()[param_map.free_alpha_idx]
    g_r = np.zeros(param_map.free_r_idx.size, dtype=np.float64)
    for idx, k_low in enumerate(param_map.free_r_idx):
        k_up = param_map.lower_to_upper[k_low]
        if k_up >= 0:
            g_r[idx] = grad_r_bases[k_low] + grad_r_bases[k_up]
        else:
            g_r[idx] = grad_r_bases[k_low]
    return np.concatenate([g_alpha, g_r])
