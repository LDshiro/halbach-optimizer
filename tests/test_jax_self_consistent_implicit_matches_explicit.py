import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph


def _jnp() -> object:
    _ = pytest.importorskip("jax")
    import jax.numpy as jnp

    return jnp


def _build_positions(R: int, K: int, N: int) -> object:
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


def test_implicit_matches_explicit() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import solve_p_easy_axis_near

    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)

    kwargs = dict(p0=1.0, chi=0.05, Nd=1.0 / 3.0, volume_m3=1e-6, iters=80, omega=0.6)

    p_exp = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        implicit_diff=False,
        **kwargs,
    )
    p_imp = solve_p_easy_axis_near(
        phi,
        r0,
        jnp.asarray(graph.nbr_idx),
        jnp.asarray(graph.nbr_mask),
        implicit_diff=True,
        **kwargs,
    )

    np.testing.assert_allclose(np.asarray(p_imp), np.asarray(p_exp), rtol=0.0, atol=1e-10)
