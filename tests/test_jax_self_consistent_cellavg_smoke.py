import numpy as np
import pytest

from halbach.autodiff.jax_self_consistent import solve_p_easy_axis_near_cellavg
from halbach.near import NearWindow, build_near_graph


def test_cellavg_smoke() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    R, K, N = 1, 2, 6
    M = R * K * N
    r0_flat = rng.standard_normal((M, 3)).astype(np.float64) * 0.1
    phi_flat = (rng.standard_normal(M) * 0.1).astype(np.float64)

    near = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    nbr_idx = jnp.asarray(near.nbr_idx, dtype=jnp.int32)
    nbr_mask = jnp.asarray(near.nbr_mask, dtype=bool)

    p0 = 1.0
    volume_m3 = 1e-6

    p = solve_p_easy_axis_near_cellavg(
        jnp.asarray(phi_flat),
        jnp.asarray(r0_flat),
        nbr_idx,
        nbr_mask,
        p0=p0,
        chi=0.0,
        Nd=1.0 / 3.0,
        volume_m3=volume_m3,
        iters=5,
        omega=0.6,
    )
    np.testing.assert_allclose(np.asarray(p), p0, rtol=0.0, atol=1e-12)

    p2 = solve_p_easy_axis_near_cellavg(
        jnp.asarray(phi_flat),
        jnp.asarray(r0_flat),
        nbr_idx,
        nbr_mask,
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=volume_m3,
        iters=10,
        omega=0.6,
    )
    p2_np = np.asarray(p2)
    assert np.isfinite(p2_np).all()
    assert float(p2_np.min()) > 0.0
    assert float(p2_np.max()) < 2.0 * p0

    phi_flat2 = phi_flat + 1e-3
    p3 = solve_p_easy_axis_near_cellavg(
        jnp.asarray(phi_flat2),
        jnp.asarray(r0_flat),
        nbr_idx,
        nbr_mask,
        p0=p0,
        chi=0.05,
        Nd=1.0 / 3.0,
        volume_m3=volume_m3,
        iters=10,
        omega=0.6,
    )
    delta = float(np.mean(np.asarray(p3)) - np.mean(p2_np))
    assert abs(delta) > 0.0
