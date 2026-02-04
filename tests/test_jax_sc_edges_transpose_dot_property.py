import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph, edges_from_near


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


def test_edges_transpose_dot_property() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        _apply_k_edges,
        _apply_kt_edges,
        _build_k_edge_dipole,
    )

    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    i_edge_np, j_edge_np = edges_from_near(graph.nbr_idx, graph.nbr_mask)

    rng = np.random.default_rng(2)
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, R * K * N), dtype=jnp.float64)
    r0 = _build_positions(R, K, N)
    p = jnp.asarray(rng.uniform(0.5, 1.5, R * K * N), dtype=jnp.float64)
    v = jnp.asarray(rng.uniform(-1.0, 1.0, R * K * N), dtype=jnp.float64)

    i_edge = jnp.asarray(i_edge_np, dtype=jnp.int32)
    j_edge = jnp.asarray(j_edge_np, dtype=jnp.int32)
    k_edge = _build_k_edge_dipole(phi, r0, i_edge, j_edge)

    Kp = _apply_k_edges(p, i_edge, j_edge, k_edge, int(phi.shape[0]))
    KT_v = _apply_kt_edges(v, i_edge, j_edge, k_edge, int(phi.shape[0]))
    left = jnp.dot(v, Kp)
    right = jnp.dot(KT_v, p)

    np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-12, atol=1e-12)
