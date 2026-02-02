import math

import numpy as np
from numpy.typing import NDArray

from halbach.demag_cellavg import demag_N_cellavg


def _assert_finite(N: NDArray[np.float64]) -> None:
    assert np.isfinite(N).all()


def test_cellavg_symmetry() -> None:
    s = np.array([0.37, 0.91, 1.23], dtype=np.float64)
    N = demag_N_cellavg(s, (1.0, 1.0, 1.0))
    _assert_finite(N)
    assert abs(float(N[0, 1] - N[1, 0])) < 1e-12
    assert abs(float(N[0, 2] - N[2, 0])) < 1e-12
    assert abs(float(N[1, 2] - N[2, 1])) < 1e-12


def test_cellavg_evenness() -> None:
    s = np.array([0.37, 0.91, 1.23], dtype=np.float64)
    N = demag_N_cellavg(s, (1.0, 1.0, 1.0))
    N2 = demag_N_cellavg(-s, (1.0, 1.0, 1.0))
    _assert_finite(N)
    _assert_finite(N2)
    diff = np.max(np.abs(N - N2))
    assert float(diff) < 1e-12


def test_cellavg_farfield_matches_dipole_tensor() -> None:
    s = np.array([40.0, 30.0, 20.0], dtype=np.float64)
    N = demag_N_cellavg(s, (1.0, 1.0, 1.0))
    _assert_finite(N)

    r2 = float(np.dot(s, s))
    rmag = math.sqrt(r2)
    eye = np.eye(3, dtype=np.float64)
    dip = 3.0 * np.outer(s, s) / (rmag**5) - eye / (rmag**3)
    N_dip = -(1.0 / (4.0 * math.pi)) * dip

    norm_ref = float(np.linalg.norm(N_dip))
    denom = max(norm_ref, 1e-30)
    rel_err = float(np.linalg.norm(N - N_dip)) / denom
    assert float(rel_err) < 2e-3


def _subdip_reference_N(s: NDArray[np.float64], a: float, n: int) -> NDArray[np.float64]:
    coords = ((np.arange(n, dtype=np.float64) + 0.5) / n - 0.5) * a
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    pos = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    src = pos
    tgt = pos
    V = a * a * a
    Vsub = V / float(n**3)

    r = tgt[:, None, :] + s[None, None, :] - src[None, :, :]
    r2 = np.sum(r * r, axis=-1)
    rmag = np.sqrt(r2)
    invr3 = 1.0 / (rmag * r2)
    invr5 = invr3 / r2

    N_ref = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        rdotm = r[..., k] * Vsub
        term1 = 3.0 * r * rdotm[..., None] * invr5[..., None]
        m_vec = np.zeros(3, dtype=np.float64)
        m_vec[k] = Vsub
        term2 = m_vec * invr3[..., None]
        H = (1.0 / (4.0 * math.pi)) * (term1 - term2)
        H_sum = np.sum(H, axis=1)
        H_avg = np.mean(H_sum, axis=0)
        N_ref[:, k] = -H_avg
    return N_ref


def test_cellavg_nearfield_matches_subdip_reference() -> None:
    a = 1.0
    s = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    N = demag_N_cellavg(s, (a, a, a))
    _assert_finite(N)

    N_ref = _subdip_reference_N(s, a, n=14)
    err = np.max(np.abs(N - N_ref))
    assert float(err) < 2e-3


def test_cellavg_axis_cases_are_finite() -> None:
    for s in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        N = demag_N_cellavg(s.astype(np.float64), (1.0, 1.0, 1.0))
        _assert_finite(N)
