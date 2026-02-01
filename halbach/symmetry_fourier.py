from __future__ import annotations

import numpy as np

from halbach.types import FloatArray

__all__ = ["build_fourier_x0_features", "delta_full_from_fourier"]


def build_fourier_x0_features(theta: FloatArray, H: int) -> tuple[FloatArray, FloatArray]:
    if H < 0:
        raise ValueError("H must be >= 0")
    N = int(theta.shape[0])
    if N % 2 != 0:
        raise ValueError("theta length must be even for x=0 mirror symmetry")
    if H == 0:
        empty = np.zeros((0, N), dtype=np.float64)
        return empty, empty

    h = np.arange(H, dtype=np.float64)
    odd = 2.0 * h + 1.0
    even = 2.0 * (h + 1.0)
    theta_row = np.asarray(theta, dtype=np.float64)[None, :]
    cos_odd = np.cos(odd[:, None] * theta_row)
    sin_even = np.sin(even[:, None] * theta_row)
    return cos_odd.astype(np.float64), sin_even.astype(np.float64)


def delta_full_from_fourier(
    coeffs: FloatArray, cos_odd: FloatArray, sin_even: FloatArray
) -> FloatArray:
    if cos_odd.shape != sin_even.shape:
        raise ValueError("cos_odd and sin_even must have the same shape")
    H = int(cos_odd.shape[0])
    if coeffs.shape[1] != 2 * H:
        raise ValueError("coeffs second dimension must be 2*H")
    a = coeffs[:, :H]
    b = coeffs[:, H:]
    return np.asarray(a @ cos_odd + b @ sin_even, dtype=np.float64)
