import numpy as np
import pytest

from halbach.constants import FACTOR
from halbach.geom import build_roi_points
from halbach.near import NearWindow, build_near_graph
from halbach.types import Geometry


def _build_geom() -> tuple[Geometry, np.ndarray]:
    N = 6
    K = 2
    R = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
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


def _grad_norm(gA: np.ndarray, gR: np.ndarray) -> float:
    return float(np.linalg.norm(np.concatenate([gA.ravel(), gR])))


def test_implicit_grad_matches_unrolled() -> None:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_objective_self_consistent_legacy import (
        objective_with_grads_self_consistent_legacy_jax,
    )

    geom, pts = _build_geom()
    rng = np.random.default_rng(0)
    alphas = 1e-3 * rng.standard_normal((geom.R, geom.K))
    r_bases = 0.2 + 1e-4 * rng.standard_normal(geom.K)

    near = build_near_graph(geom.R, geom.K, geom.N, NearWindow(wr=0, wz=1, wphi=1))

    kwargs = dict(
        chi=0.05,
        Nd=1.0 / 3.0,
        p0=1.0,
        volume_m3=1e-6,
        near_kernel="dipole",
        subdip_n=2,
        iters=40,
        omega=0.6,
        factor=FACTOR,
        use_jit=False,
    )

    J_exp, gA_exp, gR_exp, _B0_exp, _sc_exp = objective_with_grads_self_consistent_legacy_jax(
        alphas,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        implicit_diff=False,
        **kwargs,
    )
    J_imp, gA_imp, gR_imp, _B0_imp, _sc_imp = objective_with_grads_self_consistent_legacy_jax(
        alphas,
        r_bases,
        geom,
        pts,
        near.nbr_idx,
        near.nbr_mask,
        implicit_diff=True,
        **kwargs,
    )

    np.testing.assert_allclose(J_imp, J_exp, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        _grad_norm(gA_imp, gR_imp),
        _grad_norm(gA_exp, gR_exp),
        rtol=1e-6,
        atol=1e-9,
    )
