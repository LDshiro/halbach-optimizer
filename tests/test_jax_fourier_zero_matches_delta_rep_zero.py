import numpy as np
import pytest

from halbach.geom import build_roi_points
from halbach.symmetry import build_mirror_x0
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


def test_jax_fourier_zero_matches_delta_rep_zero() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_delta_phi import objective_with_grads_delta_phi_x0_jax
    from halbach.autodiff.jax_objective_delta_phi_fourier import (
        objective_with_grads_delta_phi_fourier_x0_jax,
    )

    geom = _make_geom()
    pts = build_roi_points(roi_r=0.02, roi_step=0.02)
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)

    mirror = build_mirror_x0(geom.N)
    delta_rep = np.zeros((geom.K, mirror.rep_idx.size), dtype=np.float64)
    H = 2
    coeffs = np.zeros((geom.K, 2 * H), dtype=np.float64)

    J_rep, _, _, _ = objective_with_grads_delta_phi_x0_jax(delta_rep, r_bases, geom, pts)
    J_fourier, _, _, _ = objective_with_grads_delta_phi_fourier_x0_jax(
        coeffs, r_bases, geom, pts, H=H
    )

    np.testing.assert_allclose(J_fourier, J_rep, rtol=1e-6, atol=1e-8)
