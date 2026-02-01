import numpy as np
import pytest

from halbach.geom import build_roi_points
from halbach.symmetry import build_mirror_x0
from halbach.types import Geometry


def _make_geom() -> Geometry:
    N = 8
    K = 5
    R = 2
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    ring_offsets = np.array([0.0, 0.01], dtype=np.float64)
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


def test_jax_delta_phi_zero_matches_legacy() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax
    from halbach.autodiff.jax_objective_delta_phi import objective_with_grads_delta_phi_x0_jax

    geom = _make_geom()
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)

    mirror = build_mirror_x0(geom.N)
    delta_rep = np.zeros((geom.K, mirror.rep_idx.size), dtype=np.float64)
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    alphas = np.zeros((geom.R, geom.K), dtype=np.float64)

    J_legacy, _gA, _gR, B0n_legacy = objective_with_grads_fixed_jax(alphas, r_bases, geom, pts)
    J_delta, _gD, _gR2, B0n_delta = objective_with_grads_delta_phi_x0_jax(
        delta_rep, r_bases, geom, pts
    )

    np.testing.assert_allclose(J_delta, J_legacy, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(B0n_delta, B0n_legacy, rtol=1e-6, atol=1e-8)
