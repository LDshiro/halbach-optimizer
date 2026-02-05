import numpy as np
import pytest

from halbach.geom import build_roi_points
from halbach.objective import objective_with_grads_fixed
from halbach.types import Geometry


def _make_geom() -> Geometry:
    N = 6
    K = 3
    R = 1
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    ring_offsets = np.array([0.0], dtype=np.float64)
    dz = float(z_layers[1] - z_layers[0]) if K > 1 else 0.0
    Lz = float(z_layers[-1] - z_layers[0]) if K > 1 else 0.0
    return Geometry(
        theta=theta.astype(np.float64),
        sin2=sin2.astype(np.float64),
        cth=cth.astype(np.float64),
        sth=sth.astype(np.float64),
        z_layers=z_layers.astype(np.float64),
        ring_offsets=ring_offsets,
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )


def test_jax_matches_analytic() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax

    geom = _make_geom()
    rng = np.random.default_rng(0)
    alphas = (1e-3 * rng.standard_normal(size=(geom.R, geom.K))).astype(np.float64)
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    pts = build_roi_points(roi_r=0.02, roi_step=0.02)

    J_ana, gA_ana, gR_ana, B0n_ana = objective_with_grads_fixed(alphas, r_bases, geom, pts)
    J_jax, gA_jax, gR_jax, B0n_jax = objective_with_grads_fixed_jax(alphas, r_bases, geom, pts)

    np.testing.assert_allclose(J_jax, J_ana, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(B0n_jax, B0n_ana, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(gA_jax, gA_ana, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gR_jax, gR_ana, rtol=1e-6, atol=1e-6)
