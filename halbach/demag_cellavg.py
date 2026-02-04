from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

EPS = 1e-30
FOUR_PI = 4.0 * math.pi


def _safe_log(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.log(np.maximum(x, EPS)), dtype=np.float64)


def _atan2_ratio(num: NDArray[np.float64], den: NDArray[np.float64]) -> NDArray[np.float64]:
    angle = np.arctan2(num, den)
    angle = np.where(num == 0.0, 0.0, angle)
    adjust = np.where(den < 0.0, np.sign(num) * math.pi, 0.0)
    return np.asarray(angle - adjust, dtype=np.float64)


def _f(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    R = np.sqrt(x * x + y * y + z * z) + EPS
    term1 = (1.0 / 6.0) * (2.0 * x * x - y * y - z * z) * R
    term2 = 0.5 * y * (z * z - x * x) * _safe_log(y + R)
    term3 = 0.5 * z * (y * y - x * x) * _safe_log(z + R)
    term4 = -x * y * z * _atan2_ratio(y * z, x * R)
    return np.asarray(term1 + term2 + term3 + term4, dtype=np.float64)


def _g(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    R = np.sqrt(x * x + y * y + z * z) + EPS
    term1 = -(1.0 / 3.0) * x * y * R
    term2 = x * y * z * _safe_log(z + R)
    term3 = (1.0 / 6.0) * y * (3.0 * z * z - y * y) * _safe_log(x + R)
    term4 = (1.0 / 6.0) * x * (3.0 * z * z - x * x) * _safe_log(y + R)
    term5 = -(1.0 / 6.0) * z * z * z * _atan2_ratio(x * y, z * R)
    term6 = -(1.0 / 2.0) * y * y * z * _atan2_ratio(x * z, y * R)
    term7 = -(1.0 / 2.0) * x * x * z * _atan2_ratio(y * z, x * R)
    return np.asarray(term1 + term2 + term3 + term4 + term5 + term6 + term7, dtype=np.float64)


def _gamma(e1: int, e2: int, e3: int) -> float:
    power = abs(e1) + abs(e2) + abs(e3)
    return 8.0 / ((-2.0) ** power)


def _l_operator(
    phi: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    hx: float,
    hy: float,
    hz: float,
) -> NDArray[np.float64]:
    acc = np.zeros_like(x, dtype=np.float64)
    for e1 in (-1, 0, 1):
        for e2 in (-1, 0, 1):
            for e3 in (-1, 0, 1):
                w = _gamma(e1, e2, e3)
                acc += w * phi(x + e1 * hx, y + e2 * hy, z + e3 * hz)
    return acc / (FOUR_PI * hx * hy * hz)


def demag_N_cellavg(s: NDArray[np.float64], h: Sequence[float]) -> NDArray[np.float64]:
    s_arr = np.asarray(s, dtype=np.float64)
    if s_arr.shape[-1] != 3:
        raise ValueError("s must have shape (..., 3)")
    hx, hy, hz = (float(h[0]), float(h[1]), float(h[2]))

    x = np.asarray(s_arr[..., 0], dtype=np.float64)
    y = np.asarray(s_arr[..., 1], dtype=np.float64)
    z = np.asarray(s_arr[..., 2], dtype=np.float64)

    Nxx = _l_operator(_f, x, y, z, hx, hy, hz)
    Nxy = _l_operator(_g, x, y, z, hx, hy, hz)
    Nyy = _l_operator(_f, y, x, z, hy, hx, hz)
    Nzz = _l_operator(_f, z, y, x, hz, hy, hx)
    Nxz = _l_operator(_g, x, z, y, hx, hz, hy)
    Nyz = _l_operator(_g, y, z, x, hy, hz, hx)

    out_shape = s_arr.shape[:-1] + (3, 3)
    N = np.empty(out_shape, dtype=np.float64)
    N[..., 0, 0] = Nxx
    N[..., 0, 1] = Nxy
    N[..., 0, 2] = Nxz
    N[..., 1, 0] = Nxy
    N[..., 1, 1] = Nyy
    N[..., 1, 2] = Nyz
    N[..., 2, 0] = Nxz
    N[..., 2, 1] = Nyz
    N[..., 2, 2] = Nzz
    return N


def coupling_scalar_easy_axis(
    ui: NDArray[np.float64],
    uj: NDArray[np.float64],
    N: NDArray[np.float64],
) -> NDArray[np.float64]:
    ui_f = np.asarray(ui, dtype=np.float64)
    uj_f = np.asarray(uj, dtype=np.float64)
    N_f = np.asarray(N, dtype=np.float64)
    return np.asarray(-np.einsum("...i,...ij,...j->...", ui_f, N_f, uj_f), dtype=np.float64)


__all__ = ["demag_N_cellavg", "coupling_scalar_easy_axis"]
