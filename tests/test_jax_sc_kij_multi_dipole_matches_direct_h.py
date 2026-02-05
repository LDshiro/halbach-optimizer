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


@pytest.mark.parametrize("subdip_n", [1, 2])
def test_kij_multi_dipole_matches_direct_h(subdip_n: int) -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        _build_kij_multi_dipole,
        _compute_h_ext_u_near_multi_dipole,
        _h_from_kij,
        make_subdip_offsets_grid,
    )

    R, K, N = 1, 2, 4
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, R * K * N), dtype=jnp.float64)
    r0 = _build_positions(R, K, N)
    p = jnp.asarray(rng.uniform(0.5, 1.5, R * K * N), dtype=jnp.float64)

    nbr_idx = jnp.asarray(graph.nbr_idx)
    nbr_mask = jnp.asarray(graph.nbr_mask)

    cube_edge = 1e-6 ** (1.0 / 3.0)
    offsets = make_subdip_offsets_grid(subdip_n, cube_edge)

    h_ref = _compute_h_ext_u_near_multi_dipole(phi, r0, p, nbr_idx, nbr_mask, offsets)
    kij = _build_kij_multi_dipole(phi, r0, nbr_idx, nbr_mask, offsets)
    h_kij = _h_from_kij(p, nbr_idx, nbr_mask, kij)

    np.testing.assert_allclose(np.asarray(h_kij), np.asarray(h_ref), rtol=1e-12, atol=1e-12)
