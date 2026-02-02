from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize_radii(radii_m: float | NDArray[np.float64], R: int, K: int) -> NDArray[np.float64]:
    if np.ndim(radii_m) == 0:
        return np.full((R, K), float(radii_m), dtype=np.float64)
    radii_arr = np.asarray(radii_m, dtype=np.float64)
    if radii_arr.shape == (K,):
        return np.repeat(radii_arr[None, :], R, axis=0)
    if radii_arr.shape == (R, K):
        return radii_arr
    raise ValueError("radii_m must be scalar, shape (K,), or shape (R,K)")


def make_z_positions_uniform(K: int, length_m: float) -> NDArray[np.float64]:
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return np.array([0.0], dtype=np.float64)
    half = 0.5 * float(length_m)
    return np.linspace(-half, half, K, dtype=np.float64)


def build_ring_stack_positions(
    *,
    N: int,
    K: int,
    R: int,
    radii_m: float | NDArray[np.float64],
    z_positions_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    if N < 1 or K < 1 or R < 1:
        raise ValueError("N,K,R must be >= 1")
    z_positions = np.asarray(z_positions_m, dtype=np.float64)
    if z_positions.shape != (K,):
        raise ValueError("z_positions_m must have shape (K,)")
    radii = _normalize_radii(radii_m, R, K)

    theta = 2.0 * math.pi * np.arange(N, dtype=np.float64) / float(N)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    r0_rkn = np.zeros((R, K, N, 3), dtype=np.float64)
    for r in range(R):
        for k in range(K):
            radius = float(radii[r, k])
            x = radius * cos_t
            y = radius * sin_t
            z = float(z_positions[k])
            r0_rkn[r, k, :, 0] = x
            r0_rkn[r, k, :, 1] = y
            r0_rkn[r, k, :, 2] = z

    info: dict[str, Any] = {"R": int(R), "K": int(K), "N": int(N)}
    return r0_rkn, info


__all__ = ["build_ring_stack_positions", "make_z_positions_uniform"]
