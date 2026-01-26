import numpy as np

from halbach.symmetry import build_mirror_x0
from halbach.symmetry_fourier import build_fourier_x0_features, delta_full_from_fourier


def test_fourier_x0_basis_mirror_and_fixed_points() -> None:
    N = 12
    H = 3
    K = 4
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float64)
    mirror = build_mirror_x0(N)
    cos_odd, sin_even = build_fourier_x0_features(theta, H)

    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(size=(K, 2 * H))
    delta_full = delta_full_from_fourier(coeffs, cos_odd, sin_even)

    np.testing.assert_allclose(delta_full[:, mirror.mirror_idx], -delta_full, atol=1e-12)
    np.testing.assert_allclose(delta_full[:, mirror.fixed_idx], 0.0, atol=1e-12)


def test_fourier_x0_basis_zero_H() -> None:
    N = 8
    H = 0
    K = 3
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float64)
    cos_odd, sin_even = build_fourier_x0_features(theta, H)
    coeffs = np.zeros((K, 0), dtype=np.float64)
    delta_full = delta_full_from_fourier(coeffs, cos_odd, sin_even)
    np.testing.assert_allclose(delta_full, 0.0, atol=1e-12)
