import numpy as np
import pytest

from halbach.geom import build_roi_points
from halbach.near import NearWindow, build_near_graph
from halbach.types import Geometry


def _make_geom() -> Geometry:
    N = 8
    K = 4
    R = 1
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.03, 0.03, K)
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


@pytest.mark.parametrize("near_kernel", ["dipole", "multi-dipole"])
def test_jax_sc_legacy_beta_zero_matches_existing(near_kernel: str) -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_self_consistent_legacy import (
        objective_with_grads_self_consistent_legacy_jax,
    )
    from halbach.autodiff.jax_objective_self_consistent_legacy_beta_tilt import (
        objective_with_grads_self_consistent_legacy_beta_tilt_jax,
    )

    geom = _make_geom()
    rng = np.random.default_rng(11)
    alphas = (1e-3 * rng.standard_normal(size=(geom.R, geom.K))).astype(np.float64)
    beta = np.zeros((geom.R, geom.K), dtype=np.float64)
    r_bases = (0.2 + 1e-3 * np.arange(geom.K, dtype=np.float64)).astype(np.float64)
    pts = build_roi_points(roi_r=0.03, roi_step=0.03)
    near = build_near_graph(geom.R, geom.K, geom.N, NearWindow(wr=0, wz=1, wphi=1))

    kw = dict(
        chi=0.05,
        Nd=1.0 / 3.0,
        p0=1.0,
        volume_m3=1e-6,
        near_kernel=near_kernel,
        subdip_n=2,
        iters=8,
        omega=0.6,
    )
    J_old, gA_old, gR_old, B0_old, _extras_old = objective_with_grads_self_consistent_legacy_jax(
        alphas,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        **kw,
    )
    J_new, gA_new, gB_new, gR_new, B0_new, _extras_new = (
        objective_with_grads_self_consistent_legacy_beta_tilt_jax(
            alphas,
            beta,
            r_bases,
            geom,
            pts,
            near.nbr_idx,
            near.nbr_mask,
            **kw,
        )
    )

    np.testing.assert_allclose(J_new, J_old, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(B0_new, B0_old, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(gA_new, gA_old, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gR_new, gR_old, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gB_new, 0.0, rtol=0.0, atol=1e-6)
