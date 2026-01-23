from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

__all__ = [
    "build_symmetry_indices",
    "build_roi_points",
    "build_r_bases_from_vars",
    "pack_x",
    "unpack_x",
    "sample_sphere_surface_fibonacci",
    "sample_sphere_surface_random",
]


def build_symmetry_indices(
    K: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Build z-symmetry indices (central 4 layers fixed)."""
    fixed_center = np.arange(K // 2 - 2, K // 2 + 2)
    lower_var = np.arange(0, K // 2 - 2)
    upper_var = (K - 1) - lower_var
    return lower_var, upper_var, fixed_center


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


def pack_x(alphas: NDArray[np.float64], r_vars: NDArray[np.float64]) -> NDArray[np.float64]:
    """Concatenate x = [alphas(:); r_vars]."""
    return np.concatenate([alphas.ravel(), r_vars])


def unpack_x(
    x: NDArray[np.float64],
    R: int,
    K: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split x into alphas (R, K) and r_vars."""
    P = R * K
    al = x[:P].reshape(R, K)
    rv = x[P:]
    return al, rv
