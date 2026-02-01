import numpy as np

from halbach.constants import FACTOR, m0, phi0
from halbach.geom import build_roi_points
from halbach.physics import objective_only


def test_objective_only_smoke_runs_fast() -> None:
    """
    超小規模モデルで objective_only が例外なく計算できることを確認する。
    - Numba JIT の基本動作
    - 形状整合
    - NaN/inf を出さない
    """
    N = 12
    K = 6
    R = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)

    z_layers = np.linspace(-0.05, 0.05, K)
    ring_offsets = np.array([0.0], dtype=float)

    rng = np.random.default_rng(0)
    alphas = 1e-3 * rng.standard_normal((R, K))
    r0 = 0.2
    r_bases = r0 + 1e-4 * rng.standard_normal(K)

    pts = build_roi_points(roi_r=0.03, roi_step=0.03)

    J = objective_only(
        alphas,
        r_bases,
        theta,
        sin2,
        cth,
        sth,
        z_layers,
        ring_offsets,
        pts,
        FACTOR,
        phi0,
        m0,
    )

    assert np.isfinite(J)
    assert J >= 0.0
