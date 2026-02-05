import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph
from halbach.sc_linear_system import solve_p_easy_axis_linear_system


def _build_ring_geom() -> tuple[np.ndarray, np.ndarray]:
    R, K, N = 1, 2, 6
    radius = 0.10
    z_pitch = 0.02
    r0_list: list[list[float]] = []
    phi_list: list[float] = []
    for _r in range(R):
        for k in range(K):
            z = (k - 0.5) * z_pitch
            for n in range(N):
                theta = 2.0 * np.pi * n / N
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                r0_list.append([x, y, z])
                phi_list.append(2.0 * theta)
    return np.asarray(phi_list, dtype=np.float64), np.asarray(r0_list, dtype=np.float64)


def test_linear_system_matches_jax_solver() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from halbach.autodiff.jax_self_consistent import solve_p_easy_axis_near

    phi_flat, r0_flat = _build_ring_geom()
    R, K, N = 1, 2, 6
    near = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))

    p0 = 1.0
    chi = 0.05
    Nd = 1.0 / 3.0
    volume_m3 = 1e-6
    iters = 80
    omega = 0.6

    p_fp = solve_p_easy_axis_near(
        jnp.asarray(phi_flat),
        jnp.asarray(r0_flat),
        jnp.asarray(near.nbr_idx, dtype=jnp.int32),
        jnp.asarray(near.nbr_mask, dtype=bool),
        p0=p0,
        chi=chi,
        Nd=Nd,
        volume_m3=volume_m3,
        iters=iters,
        omega=omega,
    )

    p_ls, stats = solve_p_easy_axis_linear_system(
        phi_flat,
        r0_flat,
        near.nbr_idx,
        near.nbr_mask,
        near_kernel="dipole",
        volume_m3=volume_m3,
        p0=p0,
        chi=chi,
        Nd=Nd,
        subdip_n=2,
    )

    p_fp_np = np.asarray(p_fp, dtype=np.float64)
    denom = max(float(np.linalg.norm(p_fp_np)), 1e-30)
    rel = float(np.linalg.norm(p_ls - p_fp_np)) / denom
    assert float(rel) < 1e-9
    assert float(stats["residual_norm"]) < 1e-9
    assert np.isfinite(p_ls).all()


def test_linear_system_chi_zero() -> None:
    phi_flat, r0_flat = _build_ring_geom()
    R, K, N = 1, 2, 6
    near = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))

    p0 = 1.0
    p_ls, stats = solve_p_easy_axis_linear_system(
        phi_flat,
        r0_flat,
        near.nbr_idx,
        near.nbr_mask,
        near_kernel="dipole",
        volume_m3=1e-6,
        p0=p0,
        chi=0.0,
        Nd=1.0 / 3.0,
        subdip_n=2,
    )
    np.testing.assert_allclose(p_ls, p0, rtol=0.0, atol=1e-12)
    assert float(stats["residual_norm"]) < 1e-12
