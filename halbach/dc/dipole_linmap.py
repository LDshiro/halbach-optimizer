from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

EPS = 1e-30


def build_B_mxy_matrices(
    pts: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    *,
    factor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    pts_arr = np.asarray(pts, dtype=np.float64)
    r0_arr = np.asarray(r0_flat, dtype=np.float64)
    P = int(pts_arr.shape[0])
    M = int(r0_arr.shape[0])
    D = 2 * M

    Ax = np.zeros((P, D), dtype=np.float64)
    Ay = np.zeros((P, D), dtype=np.float64)
    Az = np.zeros((P, D), dtype=np.float64)

    scale = float(factor)
    for p in range(P):
        px, py, pz = pts_arr[p]
        for i in range(M):
            rx = px - r0_arr[i, 0]
            ry = py - r0_arr[i, 1]
            rz = pz - r0_arr[i, 2]
            r2 = rx * rx + ry * ry + rz * rz
            r2 = max(r2, EPS)
            rmag = math.sqrt(r2)
            inv_r3 = 1.0 / (r2 * rmag)
            inv_r5 = inv_r3 / r2

            Gxx = 3.0 * rx * rx * inv_r5 - inv_r3
            Gxy = 3.0 * rx * ry * inv_r5
            Gyx = 3.0 * ry * rx * inv_r5
            Gyy = 3.0 * ry * ry * inv_r5 - inv_r3
            Gzx = 3.0 * rz * rx * inv_r5
            Gzy = 3.0 * rz * ry * inv_r5

            col_mx = 2 * i
            col_my = 2 * i + 1

            Ax[p, col_mx] += scale * Gxx
            Ax[p, col_my] += scale * Gxy

            Ay[p, col_mx] += scale * Gyx
            Ay[p, col_my] += scale * Gyy

            Az[p, col_mx] += scale * Gzx
            Az[p, col_my] += scale * Gzy

    return Ax, Ay, Az


def build_Ay_diff(Ay: NDArray[np.float64], center_idx: int) -> NDArray[np.float64]:
    return Ay - Ay[int(center_idx) : int(center_idx) + 1, :]


__all__ = ["build_B_mxy_matrices", "build_Ay_diff"]
