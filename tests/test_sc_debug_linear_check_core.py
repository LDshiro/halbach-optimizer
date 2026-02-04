import numpy as np

from halbach.sc_linear_system import (
    build_A_sparse,
    build_b,
    build_C_edges_from_phi,
    build_C_sparse,
    build_T2_edges,
    edges_from_near,
    solve_p_linear,
)


def test_linear_check_core_dipole() -> None:
    rng = np.random.default_rng(0)
    M = 12
    phi_flat = rng.standard_normal(M).astype(np.float64)
    r0_flat = rng.standard_normal((M, 3)).astype(np.float64) * 0.1

    deg = 2
    nbr_idx = np.zeros((M, deg), dtype=np.int32)
    nbr_mask = np.ones((M, deg), dtype=bool)
    for i in range(M):
        nbr_idx[i, 0] = (i + 1) % M
        nbr_idx[i, 1] = (i + 2) % M

    i_edge, j_edge = edges_from_near(nbr_idx, nbr_mask)
    T2 = build_T2_edges(
        r0_flat,
        i_edge,
        j_edge,
        near_kernel="dipole",
        volume_m3=1e-6,
        subdip_n=2,
    )
    C_edge = build_C_edges_from_phi(phi_flat, i_edge, j_edge, T2)
    C = build_C_sparse(M, i_edge, j_edge, C_edge)
    A = build_A_sparse(C, 0.05, 1.0 / 3.0)
    b = build_b(1.0, M)
    p_ls = solve_p_linear(A, b)
    assert np.isfinite(A.data).all()
    assert np.isfinite(b).all()
    assert np.isfinite(p_ls).all()
    assert float(np.max(np.abs(C_edge))) >= 0.0
