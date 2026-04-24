from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from halbach.near import NearWindow, build_near_graph


def _jnp() -> Any:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    return jnp


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


def test_p0_vector_chi_zero_identity() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        solve_p_easy_axis_near_multi_dipole_with_p0_flat,
        solve_p_easy_axis_near_with_p0_flat,
    )

    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)
    p0_vec = jnp.asarray(np.linspace(0.7, 1.1, R * K * N), dtype=jnp.float64)
    nbr_idx = jnp.asarray(graph.nbr_idx)
    nbr_mask = jnp.asarray(graph.nbr_mask)

    p_dip = solve_p_easy_axis_near_with_p0_flat(
        phi,
        r0,
        nbr_idx,
        nbr_mask,
        p0_flat=p0_vec,
        chi=0.0,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=5,
        omega=0.6,
    )
    p_multi = solve_p_easy_axis_near_multi_dipole_with_p0_flat(
        phi,
        r0,
        nbr_idx,
        nbr_mask,
        p0_flat=p0_vec,
        chi=0.0,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        subdip_n=2,
        iters=5,
        omega=0.6,
    )

    np.testing.assert_allclose(np.asarray(p_dip), np.asarray(p0_vec), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(p_multi), np.asarray(p0_vec), rtol=0.0, atol=1e-12)


def test_p0_vector_outputs_are_finite() -> None:
    jnp = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        solve_p_easy_axis_near_multi_dipole_with_p0_flat,
        solve_p_easy_axis_near_with_p0_flat,
    )

    R, K, N = 1, 2, 6
    graph = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    phi = jnp.linspace(0.0, 2.0 * np.pi, R * K * N)
    r0 = _build_positions(R, K, N)
    p0_vec = jnp.asarray(np.linspace(0.9, 1.05, R * K * N), dtype=jnp.float64)
    nbr_idx = jnp.asarray(graph.nbr_idx)
    nbr_mask = jnp.asarray(graph.nbr_mask)

    p_dip = solve_p_easy_axis_near_with_p0_flat(
        phi,
        r0,
        nbr_idx,
        nbr_mask,
        p0_flat=p0_vec,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        iters=8,
        omega=0.6,
    )
    p_multi = solve_p_easy_axis_near_multi_dipole_with_p0_flat(
        phi,
        r0,
        nbr_idx,
        nbr_mask,
        p0_flat=p0_vec,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=1e-6,
        subdip_n=2,
        iters=8,
        omega=0.6,
    )

    assert np.isfinite(np.asarray(p_dip)).all()
    assert np.isfinite(np.asarray(p_multi)).all()
