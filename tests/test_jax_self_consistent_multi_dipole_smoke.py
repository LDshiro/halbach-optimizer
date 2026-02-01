import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph


def _build_positions(R: int, K: int, N: int, r_bases: np.ndarray) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    z_layers = np.linspace(-0.01, 0.01, K)
    r0 = np.zeros((R * K * N, 3), dtype=np.float64)
    idx = 0
    for _r in range(R):
        for k in range(K):
            for n in range(N):
                rho = r_bases[k]
                r0[idx, 0] = rho * np.cos(theta[n])
                r0[idx, 1] = rho * np.sin(theta[n])
                r0[idx, 2] = z_layers[k]
                idx += 1
    return r0


def test_multi_dipole_smoke() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from halbach.autodiff.jax_self_consistent import solve_p_easy_axis_near_multi_dipole

    R, K, N = 1, 2, 6
    r_bases = np.linspace(0.10, 0.12, K)
    r0 = _build_positions(R, K, N, r_bases)
    phi = np.linspace(0.0, 2.0 * np.pi, R * K * N)
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))

    p0 = 0.9
    p = solve_p_easy_axis_near_multi_dipole(
        jnp.asarray(phi),
        jnp.asarray(r0),
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.0,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        subdip_n=2,
        iters=5,
        omega=0.6,
    )
    np.testing.assert_allclose(np.asarray(p), p0, rtol=0.0, atol=1e-12)

    p2 = solve_p_easy_axis_near_multi_dipole(
        jnp.asarray(phi),
        jnp.asarray(r0),
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        subdip_n=2,
        iters=10,
        omega=0.6,
    )
    p3 = solve_p_easy_axis_near_multi_dipole(
        jnp.asarray(phi),
        jnp.asarray(r0),
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        subdip_n=3,
        iters=10,
        omega=0.6,
    )
    assert np.isfinite(np.asarray(p2)).all()
    assert np.isfinite(np.asarray(p3)).all()
    assert float(np.linalg.norm(np.asarray(p2 - p3))) > 0.0
