import numpy as np

from halbach.physics import compute_B_all_from_m_flat

EPS = 1e-30


def _reference_B(
    pts: np.ndarray,
    r0_flat: np.ndarray,
    m_flat: np.ndarray,
    factor: float,
) -> np.ndarray:
    r = pts[None, :, :] - r0_flat[:, None, :]
    r2 = np.sum(r * r, axis=2)
    rmag = np.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    rhat = r / rmag[:, :, None]
    mdotr = np.sum(m_flat[:, None, :] * rhat, axis=2)
    term = (3.0 * mdotr[:, :, None] * rhat - m_flat[:, None, :]) * invr3[:, :, None]
    return np.asarray(float(factor) * np.sum(term, axis=0), dtype=np.float64)


def test_compute_B_all_from_m_flat_matches_reference() -> None:
    rng = np.random.default_rng(0)
    M = 5
    P = 3
    r0_flat = rng.normal(size=(M, 3)).astype(np.float64)
    m_flat = rng.normal(size=(M, 3)).astype(np.float64)
    pts = rng.normal(size=(P, 3)).astype(np.float64)
    factor = 1.234

    ref = _reference_B(pts, r0_flat, m_flat, factor)
    out = compute_B_all_from_m_flat(pts, r0_flat, m_flat, float(factor))

    assert np.allclose(out, ref, rtol=1e-10, atol=1e-12)
