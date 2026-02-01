import numpy as np
import pytest

from halbach.constants import FACTOR, m0, phi0
from halbach.geom import build_roi_points
from halbach.near import NearWindow, build_near_graph
from halbach.symmetry import build_mirror_x0
from halbach.types import Geometry


def _build_geom() -> tuple[Geometry, np.ndarray]:
    N = 10
    K = 4
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
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)
    return geom, pts


def _grad_norm(gD: np.ndarray, gR: np.ndarray) -> float:
    return float(np.linalg.norm(np.concatenate([gD.ravel(), gR])))


@pytest.mark.parametrize("near_kernel", ["dipole", "multi-dipole"])  # type: ignore[misc]
def test_sc_delta_field_scale_invariance(near_kernel: str) -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_self_consistent_delta_phi import (
        objective_with_grads_self_consistent_delta_phi_x0_jax,
    )

    geom, pts = _build_geom()
    mirror = build_mirror_x0(int(geom.N))
    n_rep = int(mirror.rep_idx.size)

    rng = np.random.default_rng(2)
    delta_rep = 1e-3 * rng.standard_normal((geom.K, n_rep))
    r_bases = 0.2 + 1e-4 * rng.standard_normal(geom.K)

    near = build_near_graph(geom.R, geom.K, geom.N, NearWindow(wr=0, wz=1, wphi=1))
    scale = 10.0

    J1, gD1, gR1, _B01, _sc1 = objective_with_grads_self_consistent_delta_phi_x0_jax(
        delta_rep,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        chi=0.05,
        Nd=1.0 / 3.0,
        p0=m0,
        volume_m3=1e-6,
        iters=5,
        omega=0.6,
        near_kernel=near_kernel,
        subdip_n=2,
        lambda0=0.0,
        lambda_theta=0.0,
        lambda_z=0.0,
        factor=FACTOR,
        phi0=phi0,
    )
    J2, gD2, gR2, _B02, _sc2 = objective_with_grads_self_consistent_delta_phi_x0_jax(
        delta_rep,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        chi=0.05,
        Nd=1.0 / 3.0,
        p0=m0,
        volume_m3=1e-6,
        iters=5,
        omega=0.6,
        near_kernel=near_kernel,
        subdip_n=2,
        lambda0=0.0,
        lambda_theta=0.0,
        lambda_z=0.0,
        factor=FACTOR * scale,
        phi0=phi0,
    )

    ratio_J = J2 / J1
    ratio_g = _grad_norm(gD2, gR2) / _grad_norm(gD1, gR1)

    np.testing.assert_allclose(ratio_J, scale * scale, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(ratio_g, scale * scale, rtol=1e-6, atol=1e-9)
