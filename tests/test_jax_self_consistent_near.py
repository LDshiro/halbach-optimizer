from typing import Any

import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph


def _jnp() -> Any:
    _ = pytest.importorskip("jax")
    import jax.numpy as jnp

    return jnp


def _solve_p_easy_axis_near() -> Any:
    pytest.importorskip("jax")
    from halbach.autodiff.jax_self_consistent import solve_p_easy_axis_near

    return solve_p_easy_axis_near


def _build_positions(R: int, K: int, N: int) -> Any:
    jnp = _jnp()
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    r_bases = np.linspace(0.10, 0.12, K)
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
    return jnp.asarray(r0)


def test_chi_zero_returns_p0() -> None:
    jnp = _jnp()
    solve_p_easy_axis_near = _solve_p_easy_axis_near()
    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)
    p0 = 0.7
    p = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.0,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=5,
        omega=0.6,
    )
    np.testing.assert_allclose(np.asarray(p), p0, rtol=0.0, atol=1e-12)


def test_mask_padding_no_nan() -> None:
    jnp = _jnp()
    solve_p_easy_axis_near = _solve_p_easy_axis_near()
    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)
    p = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=0.9,
        chi=0.1,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=3,
        omega=0.6,
    )
    assert np.isfinite(np.asarray(p)).all()


def test_convergence_sanity() -> None:
    jnp = _jnp()
    solve_p_easy_axis_near = _solve_p_easy_axis_near()
    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)
    p0 = 0.8
    p5 = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=5,
        omega=0.6,
    )
    p20 = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=20,
        omega=0.6,
    )
    delta5 = np.linalg.norm(np.asarray(p5) - p0)
    delta20 = np.linalg.norm(np.asarray(p20) - np.asarray(p5))
    assert delta20 <= delta5 + 1e-12
