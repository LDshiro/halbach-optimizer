import numpy as np
import pytest

from halbach.constants import FACTOR, m0, phi0
from halbach.geom import build_roi_points
from halbach.near import NearWindow, build_near_graph
from halbach.symmetry import build_mirror_x0
from halbach.types import Geometry


def _build_geom() -> tuple[Geometry, np.ndarray]:
    N = 6
    K = 3
    R = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.03, 0.03, K)
    ring_offsets = np.array([0.0], dtype=float)

    dzs = np.diff(z_layers)
    dz = float(np.median(np.abs(dzs))) if dzs.size > 0 else 0.01
    Lz = dz * K
    geom = Geometry(
        theta=theta,
        sin2=sin2,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )
    pts = build_roi_points(roi_r=0.02, roi_step=0.02)
    return geom, pts


def test_sc_delta_chi0_matches_fixed() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_delta_phi import objective_with_grads_delta_phi_x0_jax
    from halbach.autodiff.jax_objective_self_consistent_delta_phi import (
        objective_with_grads_self_consistent_delta_phi_x0_jax,
    )

    geom, pts = _build_geom()
    mirror = build_mirror_x0(int(geom.N))
    n_rep = int(mirror.rep_idx.size)

    rng = np.random.default_rng(0)
    delta_rep = 1e-3 * rng.standard_normal((geom.K, n_rep))
    r_bases = 0.2 + 1e-4 * rng.standard_normal(geom.K)

    near = build_near_graph(geom.R, geom.K, geom.N, NearWindow(wr=0, wz=1, wphi=1))

    J_fix, gD_fix, gR_fix, B0_fix = objective_with_grads_delta_phi_x0_jax(
        delta_rep,
        r_bases,
        geom,
        pts,
        lambda0=0.0,
        lambda_theta=0.0,
        lambda_z=0.0,
        factor=FACTOR,
        phi0=phi0,
        m0=m0,
    )
    J_sc, gD_sc, gR_sc, B0_sc, _sc = objective_with_grads_self_consistent_delta_phi_x0_jax(
        delta_rep,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        chi=0.0,
        Nd=1.0 / 3.0,
        p0=m0,
        volume_m3=1e-6,
        iters=3,
        omega=0.6,
        near_kernel="dipole",
        subdip_n=1,
        lambda0=0.0,
        lambda_theta=0.0,
        lambda_z=0.0,
        factor=FACTOR,
        phi0=phi0,
        use_jit=False,
    )

    np.testing.assert_allclose(J_sc, J_fix, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(B0_sc, B0_fix, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(gD_sc, gD_fix, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(gR_sc, gR_fix, rtol=1e-6, atol=1e-9)
