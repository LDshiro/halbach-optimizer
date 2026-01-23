import numpy as np

from halbach.constants import FACTOR
from halbach.objective import objective_with_grads_fixed
from halbach.types import Geometry


def test_objective_scales_with_factor() -> None:
    R = 1
    K = 2
    N = 6
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.array([-0.01, 0.01], dtype=np.float64)
    ring_offsets = np.array([0.0], dtype=np.float64)
    dz = float(np.median(np.diff(z_layers)))
    Lz = float(z_layers.max() - z_layers.min())
    geom = Geometry(
        theta=theta,
        sin2=sin2,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=N,
        R=R,
        K=K,
        dz=dz,
        Lz=Lz,
    )

    alphas = np.array([[0.01, -0.02]], dtype=np.float64)
    r_bases = np.array([0.12, 0.13], dtype=np.float64)
    pts = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]], dtype=np.float64)

    J1, gA1, gR1, _ = objective_with_grads_fixed(alphas, r_bases, geom, pts, factor=FACTOR)
    scale = 10.0
    J2, gA2, gR2, _ = objective_with_grads_fixed(alphas, r_bases, geom, pts, factor=FACTOR * scale)

    g1 = np.concatenate([gA1.ravel(), gR1])
    g2 = np.concatenate([gA2.ravel(), gR2])
    ratio_J = J2 / J1
    ratio_g = np.linalg.norm(g2) / np.linalg.norm(g1)

    np.testing.assert_allclose(ratio_J, scale**2, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(ratio_g, scale**2, rtol=1e-6, atol=0.0)
