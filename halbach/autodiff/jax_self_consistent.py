from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp

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
) -> jnp.ndarray:
    """
    Fixed-iteration damped solver for p (easy-axis, near-only).
    Must NOT apply field_scale.
    """
    denom = 1.0 + chi * Nd
    p_init = jnp.full((phi_flat.shape[0],), p0, dtype=r0_flat.dtype)

    def _body(i: int, p: jnp.ndarray) -> jnp.ndarray:
        h_ext = _compute_h_ext_u_near(phi_flat, r0_flat, p, nbr_idx, nbr_mask)
        p_new = (p0 + chi * volume_m3 * h_ext) / denom
        return (1.0 - omega) * p + omega * p_new

    return jax.lax.fori_loop(0, int(iters), _body, p_init)


__all__ = ["solve_p_easy_axis_near"]
