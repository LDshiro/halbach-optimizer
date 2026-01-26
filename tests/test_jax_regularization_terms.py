import numpy as np
import pytest
from numpy.typing import NDArray

from halbach.geom import build_roi_points
from halbach.symmetry import build_mirror_x0
from halbach.symmetry_fourier import build_fourier_x0_features, delta_full_from_fourier
from halbach.types import Geometry


def _make_geom() -> Geometry:
    N = 8
    K = 4
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


def _regularization_terms(delta_full: NDArray[np.float64]) -> tuple[float, float, float]:
    R0 = float(np.mean(delta_full * delta_full))
    dtheta = np.roll(delta_full, -1, axis=1) - delta_full
    Rtheta = float(np.mean(dtheta * dtheta))
    if delta_full.shape[0] > 1:
        dz = delta_full[1:, :] - delta_full[:-1, :]
        Rz = float(np.mean(dz * dz))
    else:
        Rz = 0.0
    return R0, Rtheta, Rz


def test_regularization_terms_delta_rep() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_delta_phi import objective_with_grads_delta_phi_x0_jax

    geom = _make_geom()
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)
    mirror = build_mirror_x0(geom.N)

    rng = np.random.default_rng(0)
    delta_rep = rng.standard_normal(size=(geom.K, mirror.rep_idx.size))
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    delta_full = delta_rep @ mirror.basis
    R0, Rtheta, Rz = _regularization_terms(delta_full)

    J_base, _, _, _ = objective_with_grads_delta_phi_x0_jax(delta_rep, r_bases, geom, pts)

    lam0 = 0.7
    J0, _, _, _ = objective_with_grads_delta_phi_x0_jax(delta_rep, r_bases, geom, pts, lambda0=lam0)
    np.testing.assert_allclose(J0 - J_base, 0.5 * lam0 * R0, rtol=1e-6, atol=1e-8)

    lam_theta = 0.4
    Jt, _, _, _ = objective_with_grads_delta_phi_x0_jax(
        delta_rep, r_bases, geom, pts, lambda_theta=lam_theta
    )
    np.testing.assert_allclose(Jt - J_base, 0.5 * lam_theta * Rtheta, rtol=1e-6, atol=1e-8)

    lam_z = 0.9
    Jz, _, _, _ = objective_with_grads_delta_phi_x0_jax(
        delta_rep, r_bases, geom, pts, lambda_z=lam_z
    )
    np.testing.assert_allclose(Jz - J_base, 0.5 * lam_z * Rz, rtol=1e-6, atol=1e-8)


def test_regularization_terms_fourier() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_delta_phi_fourier import (
        objective_with_grads_delta_phi_fourier_x0_jax,
    )

    geom = _make_geom()
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)
    H = 2
    cos_odd, sin_even = build_fourier_x0_features(geom.theta, H)

    rng = np.random.default_rng(1)
    coeffs = rng.standard_normal(size=(geom.K, 2 * H))
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    delta_full = delta_full_from_fourier(coeffs, cos_odd, sin_even)
    R0, Rtheta, Rz = _regularization_terms(delta_full)

    J_base, _, _, _ = objective_with_grads_delta_phi_fourier_x0_jax(coeffs, r_bases, geom, pts, H=H)

    lam0 = 0.6
    J0, _, _, _ = objective_with_grads_delta_phi_fourier_x0_jax(
        coeffs, r_bases, geom, pts, H=H, lambda0=lam0
    )
    np.testing.assert_allclose(J0 - J_base, 0.5 * lam0 * R0, rtol=1e-6, atol=1e-8)

    lam_theta = 0.5
    Jt, _, _, _ = objective_with_grads_delta_phi_fourier_x0_jax(
        coeffs, r_bases, geom, pts, H=H, lambda_theta=lam_theta
    )
    np.testing.assert_allclose(Jt - J_base, 0.5 * lam_theta * Rtheta, rtol=1e-6, atol=1e-8)

    lam_z = 0.8
    Jz, _, _, _ = objective_with_grads_delta_phi_fourier_x0_jax(
        coeffs, r_bases, geom, pts, H=H, lambda_z=lam_z
    )
    np.testing.assert_allclose(Jz - J_base, 0.5 * lam_z * Rz, rtol=1e-6, atol=1e-8)
