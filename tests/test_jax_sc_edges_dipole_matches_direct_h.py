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


def test_edges_dipole_matches_direct_h() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        _apply_k_edges,
        _build_k_edge_dipole,
        _compute_h_ext_u_near,
    )

    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    i_edge_np, j_edge_np = edges_from_near(graph.nbr_idx, graph.nbr_mask)

    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, R * K * N), dtype=jnp.float64)
    r0 = _build_positions(R, K, N)
    p = jnp.asarray(rng.uniform(0.5, 1.5, R * K * N), dtype=jnp.float64)

    nbr_idx = jnp.asarray(graph.nbr_idx)
    nbr_mask = jnp.asarray(graph.nbr_mask)
    i_edge = jnp.asarray(i_edge_np, dtype=jnp.int32)
    j_edge = jnp.asarray(j_edge_np, dtype=jnp.int32)

    h_ref = _compute_h_ext_u_near(phi, r0, p, nbr_idx, nbr_mask)
    k_edge = _build_k_edge_dipole(phi, r0, i_edge, j_edge)
    h_edge = _apply_k_edges(p, i_edge, j_edge, k_edge, int(phi.shape[0]))

    np.testing.assert_allclose(np.asarray(h_edge), np.asarray(h_ref), rtol=1e-12, atol=1e-12)
