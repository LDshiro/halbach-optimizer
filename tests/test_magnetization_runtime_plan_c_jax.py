import numpy as np
import pytest

pytest.importorskip("jax")

from halbach.magnetization_runtime import (  # noqa: E402
    compute_p_flat_self_consistent_from_u_jax,
    compute_p_flat_self_consistent_jax,
)
from halbach.types import Geometry  # noqa: E402


def _geom() -> Geometry:
    N = 4
    K = 2
    R = 1
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float64)
    return Geometry(
        theta=theta,
        sin2=np.sin(2.0 * theta),
        cth=np.cos(theta),
        sth=np.sin(theta),
        z_layers=np.array([-0.012, 0.012], dtype=np.float64),
        ring_offsets=np.array([0.0], dtype=np.float64),
        N=N,
        K=K,
        R=R,
        dz=0.024,
        Lz=0.024,
    )


def _state() -> tuple[Geometry, np.ndarray, np.ndarray, np.ndarray]:
    geom = _geom()
    phi_rkn = np.broadcast_to(
        geom.theta[None, None, :] + np.pi / 2.0,
        (geom.R, geom.K, geom.N),
    ).astype(np.float64)
    rho = np.array([0.05, 0.052], dtype=np.float64)[None, :, None]
    x = rho * geom.cth[None, None, :]
    y = rho * geom.sth[None, None, :]
    z = np.broadcast_to(geom.z_layers[None, :, None], phi_rkn.shape)
    r0_rkn = np.stack([x, y, z], axis=-1).astype(np.float64)
    u_flat = np.stack(
        [
            np.cos(phi_rkn).reshape(-1),
            np.sin(phi_rkn).reshape(-1),
            np.zeros(phi_rkn.size, dtype=np.float64),
        ],
        axis=1,
    )
    return geom, phi_rkn, r0_rkn, u_flat


def _sc_cfg(kernel: str) -> dict[str, object]:
    cfg: dict[str, object] = {
        "chi": 0.01,
        "Nd": 0.22,
        "p0": 1.3,
        "volume_mm3": 250.0,
        "iters": 2,
        "omega": 0.5,
        "near_window": {"wr": 0, "wz": 1, "wphi": 1},
        "near_kernel": kernel,
    }
    if kernel == "multi-dipole":
        cfg["subdip_n"] = 1
    if kernel == "gl-double-mixed":
        cfg["gl_order"] = 2
    return cfg


@pytest.mark.parametrize("kernel", ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"])
def test_plan_c_u_flat_helper_matches_nominal_phi_solver(kernel: str) -> None:
    geom, phi_rkn, r0_rkn, u_flat = _state()
    cfg = _sc_cfg(kernel)

    expected = compute_p_flat_self_consistent_jax(phi_rkn, r0_rkn, geom, cfg)
    actual = compute_p_flat_self_consistent_from_u_jax(
        u_flat,
        r0_rkn.reshape(-1, 3),
        np.full(phi_rkn.size, 1.3, dtype=np.float64),
        geom,
        cfg,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_plan_c_u_flat_helper_preserves_p0_flat_when_chi_zero() -> None:
    geom, _phi_rkn, r0_rkn, u_flat = _state()
    p0_flat = np.linspace(1.0, 1.4, u_flat.shape[0], dtype=np.float64)
    cfg = _sc_cfg("dipole")
    cfg["chi"] = 0.0

    actual = compute_p_flat_self_consistent_from_u_jax(
        u_flat,
        r0_rkn.reshape(-1, 3),
        p0_flat,
        geom,
        cfg,
    )

    np.testing.assert_allclose(actual, p0_flat)
