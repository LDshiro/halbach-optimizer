from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "build_symmetry_indices",
    "build_roi_points",
    "build_r_bases_from_vars",
    "pack_x",
    "unpack_x",
]


def build_symmetry_indices(
    K: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Build z-symmetry indices (central 4 layers fixed)."""
    fixed_center = np.arange(K // 2 - 2, K // 2 + 2)
    lower_var = np.arange(0, K // 2 - 2)
    upper_var = (K - 1) - lower_var
    return lower_var, upper_var, fixed_center


def build_roi_points(roi_r: float, roi_step: float) -> NDArray[np.float64]:
    """Generate ROI grid points inside a sphere."""
    xs = np.linspace(-roi_r, roi_r, 2 * int(np.ceil(roi_r / roi_step)) + 1)
    ys = xs
    zs = xs
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    mask = (X**2 + Y**2 + Z**2) <= roi_r**2 + 1e-15
    P = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float64)
    return P


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
