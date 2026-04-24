import numpy as np
import pytest

from halbach.geom import build_roi_points
from halbach.types import Geometry


def _make_geom() -> Geometry:
    N = 8
    K = 4
    R = 2
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.03, 0.03, K)
    ring_offsets = np.linspace(0.0, 0.005, R, dtype=np.float64)
    dz = float(z_layers[1] - z_layers[0]) if K > 1 else 0.0
    Lz = float(z_layers[-1] - z_layers[0]) if K > 1 else 0.0
    return Geometry(
        theta=theta.astype(np.float64),
        sin2=sin2.astype(np.float64),
        cth=cth.astype(np.float64),
        sth=sth.astype(np.float64),
        z_layers=z_layers.astype(np.float64),
        ring_offsets=ring_offsets.astype(np.float64),
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )


def test_jax_legacy_beta_zero_matches_existing() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax
    from halbach.autodiff.jax_objective_legacy_beta_tilt import (
        objective_with_grads_fixed_beta_tilt_jax,
    )

    geom = _make_geom()
    rng = np.random.default_rng(7)
    alphas = (1e-3 * rng.standard_normal(size=(geom.R, geom.K))).astype(np.float64)
    beta = np.zeros((geom.R, geom.K), dtype=np.float64)
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)

    J_old, gA_old, gR_old, B0_old = objective_with_grads_fixed_jax(alphas, r_bases, geom, pts)
    J_new, gA_new, gB_new, gR_new, B0_new = objective_with_grads_fixed_beta_tilt_jax(
        alphas, beta, r_bases, geom, pts
    )

    np.testing.assert_allclose(J_new, J_old, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(B0_new, B0_old, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(gA_new, gA_old, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gR_new, gR_old, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gB_new, 0.0, rtol=0.0, atol=1e-7)
