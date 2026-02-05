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


def _edge_partition_face_to_face(
    i_edge: np.ndarray,
    j_edge: np.ndarray,
    *,
    R: int,
    K: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_i = (i_edge % N).astype(np.int32)
    n_j = (j_edge % N).astype(np.int32)
    dn_raw = (n_i - n_j) % N
    dn = np.where(dn_raw > (N // 2), dn_raw - N, dn_raw).astype(np.int32)

    k_i = ((i_edge // N) % K).astype(np.int32)
    k_j = ((j_edge // N) % K).astype(np.int32)
    dk = (k_i - k_j).astype(np.int32)

    r_i = (i_edge // (K * N)).astype(np.int32)
    r_j = (j_edge // (K * N)).astype(np.int32)

    hi_mask = (r_i == r_j) & (np.abs(dk) == 1) & (dn == 0)
    idx_hi = np.nonzero(hi_mask)[0].astype(np.int32)
    idx_lo = np.nonzero(~hi_mask)[0].astype(np.int32)
    return idx_lo, idx_hi


def test_gl_double_mixed_matches_uniform_when_same_order() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        _gl_double_delta_table_n2,
        _k_edge_gl_double_from_table,
        _k_edge_gl_double_mixed,
    )

    R, K, N = 1, 2, 4
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    i_edge_np, j_edge_np = edges_from_near(graph.nbr_idx, graph.nbr_mask)
    idx_lo_np, idx_hi_np = _edge_partition_face_to_face(i_edge_np, j_edge_np, R=R, K=K, N=N)

    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, R * K * N), dtype=jnp.float64)
    r0 = _build_positions(R, K, N)

    i_edge = jnp.asarray(i_edge_np, dtype=jnp.int32)
    j_edge = jnp.asarray(j_edge_np, dtype=jnp.int32)
    idx_lo = jnp.asarray(idx_lo_np, dtype=jnp.int32)
    idx_hi = jnp.asarray(idx_hi_np, dtype=jnp.int32)

    delta_offsets, delta_w = _gl_double_delta_table_n2(0.01)
    k_mixed = _k_edge_gl_double_mixed(
        phi,
        r0,
        i_edge,
        j_edge,
        idx_lo,
        idx_hi,
        delta_offsets,
        delta_w,
        delta_offsets,
        delta_w,
    )
    k_uniform = _k_edge_gl_double_from_table(phi, r0, i_edge, j_edge, delta_offsets, delta_w)

    np.testing.assert_allclose(np.asarray(k_mixed), np.asarray(k_uniform), rtol=1e-12, atol=1e-12)
