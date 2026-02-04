from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from halbach.autodiff.jax_demag_cellavg import demag_N_cellavg_jax
from halbach.constants import FACTOR, mu0

cast(Any, jax.config).update("jax_enable_x64", True)

H_FACTOR = float(FACTOR / mu0)  # 1 / (4*pi)
EPS = 1e-30


def _dipole_h_from_r_m(r_ij: jnp.ndarray, m_j: jnp.ndarray) -> jnp.ndarray:
    """
    Compute H field from dipoles at offsets r_ij with moments m_j.

    TODO: Replace with cube-average (tensor) kernel for near-field.
    """
    r2 = jnp.sum(r_ij * r_ij, axis=-1)
    rmag = jnp.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    invr5 = invr3 / r2
    mdotr = jnp.sum(m_j * r_ij, axis=-1)
    term = 3.0 * r_ij * mdotr[..., None] * invr5[..., None] - m_j * invr3[..., None]
    return H_FACTOR * term


def make_subdip_offsets_grid(subdip_n: int, cube_edge_m: float) -> jnp.ndarray:
    """
    Returns offsets (S,3) for an n×n×n grid of cell centers within [-a/2, a/2]^3.
    """
    if subdip_n < 1:
        raise ValueError("subdip_n must be >= 1")
    coords = (np.arange(subdip_n, dtype=np.float64) + 0.5) / subdip_n - 0.5
    coords = coords * float(cube_edge_m)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    offsets = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return jnp.asarray(offsets, dtype=jnp.float64)


def _compute_h_ext_u_near(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    p_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
) -> jnp.ndarray:
    u_i = jnp.stack([jnp.cos(phi_flat), jnp.sin(phi_flat), jnp.zeros_like(phi_flat)], axis=1)

    r_j = r0_flat[nbr_idx]
    p_j = p_flat[nbr_idx]
    phi_j = phi_flat[nbr_idx]
    u_j = jnp.stack([jnp.cos(phi_j), jnp.sin(phi_j), jnp.zeros_like(phi_j)], axis=-1)
    m_j = p_j[..., None] * u_j

    r_i = r0_flat[:, None, :]
    r_ij = r_i - r_j

    mask_f = nbr_mask.astype(r0_flat.dtype)
    r_ij_safe = r_ij * mask_f[..., None] + (1.0 - mask_f[..., None]) * jnp.array(
        [1.0, 0.0, 0.0], dtype=r0_flat.dtype
    )

    H_ij = _dipole_h_from_r_m(r_ij_safe, m_j)
    H_ij = H_ij * mask_f[..., None]

    H_sum = jnp.sum(H_ij, axis=1)
    h_ext = jnp.sum(H_sum * u_i, axis=1)
    return h_ext


def _compute_h_ext_u_near_multi_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    p_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    u_i = jnp.stack([jnp.cos(phi_flat), jnp.sin(phi_flat), jnp.zeros_like(phi_flat)], axis=1)

    r_j = r0_flat[nbr_idx]
    p_j = p_flat[nbr_idx]
    phi_j = phi_flat[nbr_idx]
    u_j = jnp.stack([jnp.cos(phi_j), jnp.sin(phi_j), jnp.zeros_like(phi_j)], axis=-1)
    m_j = p_j[..., None] * u_j

    r_i = r0_flat[:, None, :]
    r_ij = r_i - r_j

    mask_f = nbr_mask.astype(r0_flat.dtype)
    m_j_safe = m_j * mask_f[..., None]
    r_ij_safe = r_ij * mask_f[..., None] + (1.0 - mask_f[..., None]) * jnp.array(
        [1.0, 0.0, 0.0], dtype=r0_flat.dtype
    )

    S = offsets.shape[0]
    r_sub = r_ij_safe[..., None, :] - offsets[None, None, :, :]
    m_sub = m_j_safe[..., None, :] / float(S)

    r2 = jnp.sum(r_sub * r_sub, axis=-1)
    rmag = jnp.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    invr5 = invr3 / r2
    mdotr = jnp.sum(m_sub * r_sub, axis=-1)
    term = 3.0 * r_sub * mdotr[..., None] * invr5[..., None] - m_sub * invr3[..., None]
    H_sub = H_FACTOR * term
    H_sub = H_sub * mask_f[..., None, None]

    H_sum = jnp.sum(H_sub, axis=(1, 2))
    h_ext = jnp.sum(H_sum * u_i, axis=1)
    return h_ext


def _segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    out = jnp.zeros((num_segments,), dtype=data.dtype)
    return out.at[segment_ids].add(data)


def _apply_k_edges(
    p: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    k_edge: jnp.ndarray,
    M: int,
) -> jnp.ndarray:
    return _segment_sum(k_edge * p[j_edge], i_edge, M)


def _apply_kt_edges(
    v: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    k_edge: jnp.ndarray,
    M: int,
) -> jnp.ndarray:
    return _segment_sum(k_edge * v[i_edge], j_edge, M)


def _gl_double_delta_table_n2(cube_edge_m: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    a = float(cube_edge_m)
    dx = jnp.array([-a / jnp.sqrt(3.0), 0.0, +a / jnp.sqrt(3.0)], dtype=jnp.float64)
    wx = jnp.array([0.25, 0.5, 0.25], dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(dx, dx, dx, indexing="ij")
    WX, WY, WZ = jnp.meshgrid(wx, wx, wx, indexing="ij")
    delta_offsets = jnp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    delta_w = (WX * WY * WZ).reshape(-1)
    return delta_offsets, delta_w


def _gl_double_delta_table_n3(cube_edge_m: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    a = float(cube_edge_m)
    x = 0.5 * a * jnp.sqrt(3.0 / 5.0)
    dx = jnp.array([-2.0 * x, -1.0 * x, 0.0, +1.0 * x, +2.0 * x], dtype=jnp.float64)
    wx = jnp.array(
        [25.0 / 324.0, 20.0 / 81.0, 19.0 / 54.0, 20.0 / 81.0, 25.0 / 324.0],
        dtype=jnp.float64,
    )
    X, Y, Z = jnp.meshgrid(dx, dx, dx, indexing="ij")
    WX, WY, WZ = jnp.meshgrid(wx, wx, wx, indexing="ij")
    delta_offsets = jnp.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    delta_w = (WX * WY * WZ).reshape(-1)
    return delta_offsets, delta_w


def _k_edge_gl_double_from_table(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    delta_offsets: jnp.ndarray,
    delta_w: jnp.ndarray,
) -> jnp.ndarray:
    c = jnp.cos(phi_flat).astype(jnp.float64)
    s = jnp.sin(phi_flat).astype(jnp.float64)
    ui_c = c[i_edge]
    ui_s = s[i_edge]
    uj_c = c[j_edge]
    uj_s = s[j_edge]
    ui_dot_uj = ui_c * uj_c + ui_s * uj_s

    base = (r0_flat[i_edge] - r0_flat[j_edge]).astype(jnp.float64)
    rx0 = base[:, 0]
    ry0 = base[:, 1]
    rz0 = base[:, 2]
    acc = jnp.zeros((base.shape[0],), dtype=jnp.float64)
    D = int(delta_offsets.shape[0])

    def body(d: int, acc_d: jnp.ndarray) -> jnp.ndarray:
        ox = delta_offsets[d, 0]
        oy = delta_offsets[d, 1]
        oz = delta_offsets[d, 2]
        rx = rx0 + ox
        ry = ry0 + oy
        rz = rz0 + oz
        r2 = jnp.maximum(rx * rx + ry * ry + rz * rz, EPS)
        rmag = jnp.sqrt(r2) + EPS
        invr3 = 1.0 / (rmag * r2 + EPS)
        invr5 = invr3 / r2
        ui_dot_r = ui_c * rx + ui_s * ry
        uj_dot_r = uj_c * rx + uj_s * ry
        term = 3.0 * ui_dot_r * uj_dot_r * invr5 - ui_dot_uj * invr3
        return acc_d + delta_w[d] * term

    acc = cast(jnp.ndarray, jax.lax.fori_loop(0, D, body, acc))
    return (H_FACTOR * acc).astype(jnp.float64)


def _k_edge_gl_double_mixed(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    idx_lo: jnp.ndarray,
    idx_hi: jnp.ndarray,
    delta_lo_offsets: jnp.ndarray,
    delta_lo_w: jnp.ndarray,
    delta_hi_offsets: jnp.ndarray,
    delta_hi_w: jnp.ndarray,
) -> jnp.ndarray:
    E = int(i_edge.shape[0])
    k = jnp.zeros((E,), dtype=jnp.float64)

    i_lo = i_edge[idx_lo]
    j_lo = j_edge[idx_lo]
    k_lo = _k_edge_gl_double_from_table(phi_flat, r0_flat, i_lo, j_lo, delta_lo_offsets, delta_lo_w)
    k = k.at[idx_lo].set(k_lo)

    i_hi = i_edge[idx_hi]
    j_hi = j_edge[idx_hi]
    k_hi = _k_edge_gl_double_from_table(phi_flat, r0_flat, i_hi, j_hi, delta_hi_offsets, delta_hi_w)
    k = k.at[idx_hi].set(k_hi)
    return k


def _h_from_kij(
    p_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    kij: jnp.ndarray,
) -> jnp.ndarray:
    p_j = p_flat[nbr_idx]
    p_j = jnp.where(nbr_mask, p_j, 0.0)
    return jnp.sum(kij * p_j, axis=1)


def _build_kij_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
) -> jnp.ndarray:
    c = jnp.cos(phi_flat)
    s = jnp.sin(phi_flat)

    r_i = r0_flat[:, None, :]
    r_j = r0_flat[nbr_idx]
    r = r_i - r_j
    rx = r[..., 0]
    ry = r[..., 1]
    rz = r[..., 2]
    r2 = rx * rx + ry * ry + rz * rz
    r2_safe = jnp.where(nbr_mask, r2, 1.0)
    rmag = jnp.sqrt(r2_safe) + EPS
    invr3 = 1.0 / (rmag * r2_safe + EPS)
    invr5 = invr3 / r2_safe

    ui_c = c[:, None]
    ui_s = s[:, None]
    uj_c = c[nbr_idx]
    uj_s = s[nbr_idx]

    ui_dot_r = ui_c * rx + ui_s * ry
    uj_dot_r = uj_c * rx + uj_s * ry
    ui_dot_uj = ui_c * uj_c + ui_s * uj_s

    kij = H_FACTOR * (3.0 * ui_dot_r * uj_dot_r * invr5 - ui_dot_uj * invr3)
    kij = jnp.where(nbr_mask, kij, 0.0)
    return kij


def _build_kij_multi_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    c = jnp.cos(phi_flat)
    s = jnp.sin(phi_flat)

    r_i = r0_flat[:, None, :]
    r_j = r0_flat[nbr_idx]
    s_ij = r_i - r_j
    rx0 = s_ij[..., 0]
    ry0 = s_ij[..., 1]
    rz0 = s_ij[..., 2]

    ui_c = c[:, None]
    ui_s = s[:, None]
    uj_c = c[nbr_idx]
    uj_s = s[nbr_idx]
    ui_dot_uj = ui_c * uj_c + ui_s * uj_s

    acc1 = jnp.zeros_like(rx0)
    acc2 = jnp.zeros_like(rx0)

    for q in range(int(offsets.shape[0])):
        rx = rx0 - offsets[q, 0]
        ry = ry0 - offsets[q, 1]
        rz = rz0 - offsets[q, 2]
        r2 = rx * rx + ry * ry + rz * rz
        r2_safe = jnp.where(nbr_mask, r2, 1.0)
        rmag = jnp.sqrt(r2_safe) + EPS
        invr3 = 1.0 / (rmag * r2_safe + EPS)
        invr5 = invr3 / r2_safe
        ui_dot_r = ui_c * rx + ui_s * ry
        uj_dot_r = uj_c * rx + uj_s * ry
        acc1 = acc1 + 3.0 * ui_dot_r * uj_dot_r * invr5
        acc2 = acc2 + invr3

    S = float(offsets.shape[0])
    mean1 = acc1 / S
    mean2 = acc2 / S
    kij = H_FACTOR * (mean1 - ui_dot_uj * mean2)
    kij = jnp.where(nbr_mask, kij, 0.0)
    return kij


def _build_k_edge_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
) -> jnp.ndarray:
    c = jnp.cos(phi_flat)
    s = jnp.sin(phi_flat)
    ui_c = c[i_edge]
    ui_s = s[i_edge]
    uj_c = c[j_edge]
    uj_s = s[j_edge]

    r = r0_flat[i_edge] - r0_flat[j_edge]
    rx = r[:, 0]
    ry = r[:, 1]
    rz = r[:, 2]
    r2 = jnp.maximum(rx * rx + ry * ry + rz * rz, EPS)
    rmag = jnp.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    invr5 = invr3 / r2
    ui_dot_r = ui_c * rx + ui_s * ry
    uj_dot_r = uj_c * rx + uj_s * ry
    ui_dot_uj = ui_c * uj_c + ui_s * uj_s
    return H_FACTOR * (3.0 * ui_dot_r * uj_dot_r * invr5 - ui_dot_uj * invr3)


def _build_k_edge_multi_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    c = jnp.cos(phi_flat)
    s = jnp.sin(phi_flat)
    ui_c = c[i_edge]
    ui_s = s[i_edge]
    uj_c = c[j_edge]
    uj_s = s[j_edge]
    ui_dot_uj = ui_c * uj_c + ui_s * uj_s

    base = r0_flat[i_edge] - r0_flat[j_edge]
    rx0 = base[:, 0]
    ry0 = base[:, 1]
    rz0 = base[:, 2]
    acc1 = jnp.zeros((base.shape[0],), dtype=jnp.float64)
    acc2 = jnp.zeros((base.shape[0],), dtype=jnp.float64)

    def body(q: int, state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        acc1_q, acc2_q = state
        ox = offsets[q, 0]
        oy = offsets[q, 1]
        oz = offsets[q, 2]
        rx = rx0 - ox
        ry = ry0 - oy
        rz = rz0 - oz
        r2 = jnp.maximum(rx * rx + ry * ry + rz * rz, EPS)
        rmag = jnp.sqrt(r2) + EPS
        invr3 = 1.0 / (rmag * r2 + EPS)
        invr5 = invr3 / r2
        ui_dot_r = ui_c * rx + ui_s * ry
        uj_dot_r = uj_c * rx + uj_s * ry
        acc1_q = acc1_q + 3.0 * ui_dot_r * uj_dot_r * invr5
        acc2_q = acc2_q + invr3
        return acc1_q, acc2_q

    acc1, acc2 = cast(
        tuple[jnp.ndarray, jnp.ndarray],
        jax.lax.fori_loop(0, int(offsets.shape[0]), body, (acc1, acc2)),
    )
    mean1 = acc1 / float(offsets.shape[0])
    mean2 = acc2 / float(offsets.shape[0])
    return H_FACTOR * (mean1 - ui_dot_uj * mean2)


def _build_k_edge_cellavg(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    volume_m3: float,
) -> jnp.ndarray:
    a_edge = float(volume_m3) ** (1.0 / 3.0)
    h = jnp.array([a_edge, a_edge, a_edge], dtype=jnp.float64)
    s = r0_flat[i_edge] - r0_flat[j_edge]
    N = demag_N_cellavg_jax(s, h)
    N2 = N[:, 0:2, 0:2]
    c = jnp.cos(phi_flat)
    sphi = jnp.sin(phi_flat)
    ui = jnp.stack([c[i_edge], sphi[i_edge]], axis=1)
    uj = jnp.stack([c[j_edge], sphi[j_edge]], axis=1)
    tmp = jnp.einsum("eab,eb->ea", N2, uj)
    uNu = jnp.einsum("ea,ea->e", ui, tmp)
    return -(uNu) / float(volume_m3)


def _solve_linear_system_custom_linear_solve(
    h_op: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    denom: float,
    chi: float,
    volume_m3: float,
    iters: int,
    omega: float,
    dtype: Any,
    h_op_T: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    def matvec(p: jnp.ndarray) -> jnp.ndarray:
        return denom * p - chi * volume_m3 * h_op(p)

    def solve(_matvec_unused: Any, rhs: jnp.ndarray) -> jnp.ndarray:
        p = rhs / denom

        def body(_t: int, p_now: jnp.ndarray) -> jnp.ndarray:
            hp = h_op(p_now)
            p_fp = (rhs + chi * volume_m3 * hp) / denom
            return (1.0 - omega) * p_now + omega * p_fp

        return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), body, p))

    def transpose_solve(_matvec_unused: Any, rhs: jnp.ndarray) -> jnp.ndarray:
        if h_op_T is None:
            M = rhs.shape[0]
            seed = jnp.zeros((M,), dtype=dtype)
            hT_fun = jax.linear_transpose(h_op, seed)

            def h_op_t(v: jnp.ndarray) -> jnp.ndarray:
                return cast(jnp.ndarray, hT_fun(v)[0])

            h_op_t_use: Callable[[jnp.ndarray], jnp.ndarray] = h_op_t
        else:
            h_op_t_use = h_op_T

        x = rhs / denom

        def body(_t: int, x_now: jnp.ndarray) -> jnp.ndarray:
            hx = h_op_t_use(x_now)
            x_fp = (rhs + chi * volume_m3 * hx) / denom
            return (1.0 - omega) * x_now + omega * x_fp

        return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), body, x))

    return cast(
        jnp.ndarray,
        jax.lax.custom_linear_solve(
            matvec, b, solve=solve, transpose_solve=transpose_solve, symmetric=False
        ),
    )


def solve_p_easy_axis_near(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    *,
    p0: float,
    chi: float,
    Nd: float,
    volume_m3: float,
    iters: int = 30,
    omega: float = 0.6,
    implicit_diff: bool = False,
    i_edge: jnp.ndarray | None = None,
    j_edge: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Fixed-iteration damped solver for p (easy-axis, near-only).
    Must NOT apply field_scale.
    """
    denom = 1.0 + chi * Nd
    if chi == 0.0:
        return jnp.full((phi_flat.shape[0],), float(p0), dtype=r0_flat.dtype)

    h_op_t: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if i_edge is not None and j_edge is not None:
        M = int(phi_flat.shape[0])
        k_edge = _build_k_edge_dipole(phi_flat, r0_flat, i_edge, j_edge)

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _apply_k_edges(p, i_edge, j_edge, k_edge, M)

        def h_op_t(v: jnp.ndarray) -> jnp.ndarray:
            return _apply_kt_edges(v, i_edge, j_edge, k_edge, M)

    else:
        kij = _build_kij_dipole(phi_flat, r0_flat, nbr_idx, nbr_mask)

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _h_from_kij(p, nbr_idx, nbr_mask, kij)

    if implicit_diff:
        b = jnp.full((phi_flat.shape[0],), float(p0), dtype=r0_flat.dtype)
        return _solve_linear_system_custom_linear_solve(
            h_op,
            b,
            float(denom),
            float(chi),
            float(volume_m3),
            int(iters),
            float(omega),
            r0_flat.dtype,
            h_op_T=h_op_t,
        )

    p_init = jnp.full((phi_flat.shape[0],), p0, dtype=r0_flat.dtype)

    def _body(i: int, p: jnp.ndarray) -> jnp.ndarray:
        h_ext = h_op(p)
        p_new = (p0 + chi * volume_m3 * h_ext) / denom
        return (1.0 - omega) * p + omega * p_new

    return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), _body, p_init))


def solve_p_easy_axis_near_multi_dipole(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    *,
    p0: float,
    chi: float,
    Nd: float,
    volume_m3: float,
    subdip_n: int = 2,
    iters: int = 30,
    omega: float = 0.6,
    implicit_diff: bool = False,
    i_edge: jnp.ndarray | None = None,
    j_edge: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Fixed-iteration damped solver for p (easy-axis, near-only) using multi-dipole source split.

    TODO: Replace near kernel with cube-average tensor model.
    """
    if subdip_n < 1:
        raise ValueError("subdip_n must be >= 1")
    denom = 1.0 + chi * Nd
    if chi == 0.0:
        return jnp.full((phi_flat.shape[0],), float(p0), dtype=r0_flat.dtype)

    cube_edge = float(volume_m3) ** (1.0 / 3.0)
    offsets = make_subdip_offsets_grid(subdip_n, cube_edge)
    h_op_t: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if i_edge is not None and j_edge is not None:
        M = int(phi_flat.shape[0])
        k_edge = _build_k_edge_multi_dipole(phi_flat, r0_flat, i_edge, j_edge, offsets)

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _apply_k_edges(p, i_edge, j_edge, k_edge, M)

        def h_op_t(v: jnp.ndarray) -> jnp.ndarray:
            return _apply_kt_edges(v, i_edge, j_edge, k_edge, M)

    else:
        kij = _build_kij_multi_dipole(phi_flat, r0_flat, nbr_idx, nbr_mask, offsets)

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _h_from_kij(p, nbr_idx, nbr_mask, kij)

    if implicit_diff:
        b = jnp.full((phi_flat.shape[0],), float(p0), dtype=r0_flat.dtype)
        return _solve_linear_system_custom_linear_solve(
            h_op,
            b,
            float(denom),
            float(chi),
            float(volume_m3),
            int(iters),
            float(omega),
            r0_flat.dtype,
            h_op_T=h_op_t,
        )

    p_init = jnp.full((phi_flat.shape[0],), p0, dtype=r0_flat.dtype)

    def _body(i: int, p: jnp.ndarray) -> jnp.ndarray:
        h_ext = h_op(p)
        p_new = (p0 + chi * volume_m3 * h_ext) / denom
        return (1.0 - omega) * p + omega * p_new

    return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), _body, p_init))


def solve_p_easy_axis_near_cellavg(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    *,
    p0: float,
    chi: float,
    Nd: float,
    volume_m3: float,
    iters: int = 30,
    omega: float = 0.6,
    implicit_diff: bool = False,
    i_edge: jnp.ndarray | None = None,
    j_edge: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Fixed-iteration damped solver for p (easy-axis, near-only) using cell-averaged demag tensor.
    Must NOT apply field_scale.
    """
    denom = 1.0 + chi * Nd
    M = int(phi_flat.shape[0])
    a = float(volume_m3) ** (1.0 / 3.0)
    h = jnp.array([a, a, a], dtype=jnp.float64)

    u_flat = jnp.stack([jnp.cos(phi_flat), jnp.sin(phi_flat), jnp.zeros_like(phi_flat)], axis=1)

    r_i = r0_flat[:, None, :]
    r_j = r0_flat[nbr_idx]
    s_ij = r_i - r_j
    u_i = u_flat[:, None, :]
    u_j = u_flat[nbr_idx]

    s_ij = jnp.where(nbr_mask[..., None], s_ij, 0.0)
    u_j = jnp.where(nbr_mask[..., None], u_j, 0.0)

    h_op_t: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    if i_edge is not None and j_edge is not None:
        k_edge = _build_k_edge_cellavg(phi_flat, r0_flat, i_edge, j_edge, float(volume_m3))

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _apply_k_edges(p, i_edge, j_edge, k_edge, M)

        def h_op_t(v: jnp.ndarray) -> jnp.ndarray:
            return _apply_kt_edges(v, i_edge, j_edge, k_edge, M)

    else:
        N_ij = demag_N_cellavg_jax(s_ij, h)
        Nu_j = jnp.einsum("...ab,...b->...a", N_ij, u_j)
        c_ij = -jnp.einsum("...a,...a->...", u_i, Nu_j)
        c_ij = jnp.where(nbr_mask, c_ij, 0.0)
        kij = c_ij / float(volume_m3)

        def h_op(p: jnp.ndarray) -> jnp.ndarray:
            return _h_from_kij(p, nbr_idx, nbr_mask, kij)

    if chi == 0.0:
        return jnp.full((M,), float(p0), dtype=jnp.float64)

    if implicit_diff:
        b = jnp.full((M,), float(p0), dtype=jnp.float64)

        return _solve_linear_system_custom_linear_solve(
            h_op,
            b,
            float(denom),
            float(chi),
            float(volume_m3),
            int(iters),
            float(omega),
            jnp.float64,
            h_op_T=h_op_t,
        )

    p_init = jnp.full((M,), float(p0), dtype=jnp.float64)

    def _body(i: int, p: jnp.ndarray) -> jnp.ndarray:
        h_ext = h_op(p)
        p_new = (float(p0) + float(chi) * float(volume_m3) * h_ext) / denom
        return (1.0 - float(omega)) * p + float(omega) * p_new

    return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), _body, p_init))


def solve_p_easy_axis_near_gl_double_mixed(
    phi_flat: jnp.ndarray,
    r0_flat: jnp.ndarray,
    *,
    p0: float | jnp.ndarray,
    chi: float,
    Nd: float,
    volume_m3: float,
    iters: int,
    omega: float,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    idx_lo: jnp.ndarray,
    idx_hi: jnp.ndarray,
    delta_lo_offsets: jnp.ndarray,
    delta_lo_w: jnp.ndarray,
    delta_hi_offsets: jnp.ndarray,
    delta_hi_w: jnp.ndarray,
    implicit_diff: bool = True,
) -> jnp.ndarray:
    M = int(phi_flat.shape[0])
    if chi == 0.0:
        if jnp.ndim(p0) == 0:
            return jnp.full((M,), float(p0), dtype=jnp.float64)
        return cast(jnp.ndarray, jnp.asarray(p0, dtype=jnp.float64))

    denom = 1.0 + chi * Nd
    if jnp.ndim(p0) == 0:
        p0_vec = jnp.full((M,), float(p0), dtype=jnp.float64)
    else:
        p0_vec = jnp.asarray(p0, dtype=jnp.float64)

    k_edge = _k_edge_gl_double_mixed(
        phi_flat.astype(jnp.float64),
        r0_flat.astype(jnp.float64),
        i_edge,
        j_edge,
        idx_lo,
        idx_hi,
        delta_lo_offsets,
        delta_lo_w,
        delta_hi_offsets,
        delta_hi_w,
    )

    def h_op(p: jnp.ndarray) -> jnp.ndarray:
        return _apply_k_edges(p, i_edge, j_edge, k_edge, M)

    def h_op_t(v: jnp.ndarray) -> jnp.ndarray:
        return _apply_kt_edges(v, i_edge, j_edge, k_edge, M)

    if implicit_diff:
        return _solve_linear_system_custom_linear_solve(
            h_op,
            p0_vec,
            float(denom),
            float(chi),
            float(volume_m3),
            int(iters),
            float(omega),
            jnp.float64,
            h_op_T=h_op_t,
        )

    p = p0_vec

    def body(_t: int, p_now: jnp.ndarray) -> jnp.ndarray:
        h = h_op(p_now)
        p_fp = (p0_vec + chi * volume_m3 * h) / denom
        return (1.0 - omega) * p_now + omega * p_fp

    return cast(jnp.ndarray, jax.lax.fori_loop(0, int(iters), body, p))


__all__ = [
    "_compute_h_ext_u_near",
    "_compute_h_ext_u_near_multi_dipole",
    "_segment_sum",
    "_apply_k_edges",
    "_apply_kt_edges",
    "_h_from_kij",
    "_build_kij_dipole",
    "_build_kij_multi_dipole",
    "_build_k_edge_dipole",
    "_build_k_edge_multi_dipole",
    "_build_k_edge_cellavg",
    "_gl_double_delta_table_n2",
    "_gl_double_delta_table_n3",
    "_k_edge_gl_double_from_table",
    "_k_edge_gl_double_mixed",
    "make_subdip_offsets_grid",
    "solve_p_easy_axis_near",
    "solve_p_easy_axis_near_multi_dipole",
    "solve_p_easy_axis_near_cellavg",
    "solve_p_easy_axis_near_gl_double_mixed",
]
