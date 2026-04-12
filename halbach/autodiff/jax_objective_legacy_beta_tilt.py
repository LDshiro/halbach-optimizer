from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, phi0
from halbach.types import Geometry

cast(Any, jax.config).update("jax_enable_x64", True)

EPS = 1e-30


def _compute_b_all_from_m(
    m_flat: Any,
    r0_flat: Any,
    pts: Any,
    factor: float,
) -> Any:
    origin = jnp.zeros((1, 3), dtype=pts.dtype)
    pts_all = jnp.concatenate([pts, origin], axis=0)

    r = pts_all[None, :, :] - r0_flat[:, None, :]
    r2 = jnp.sum(r * r, axis=2)
    rmag = jnp.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    rhat = r / rmag[:, :, None]
    mdotr = jnp.sum(m_flat[:, None, :] * rhat, axis=2)
    term = (3.0 * mdotr[:, :, None] * rhat - m_flat[:, None, :]) * invr3[:, :, None]
    return factor * jnp.sum(term, axis=0)


def objective_with_grads_fixed_beta_tilt_jax(
    alphas: NDArray[np.float64],
    beta_tilt_x: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    factor: float = FACTOR,
    ring_active_mask: NDArray[np.bool_] | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """JAX objective and gradients (y-space) for legacy-alpha + beta_tilt_x."""
    alphas_j = jnp.asarray(alphas, dtype=jnp.float64)
    beta_j = jnp.asarray(beta_tilt_x, dtype=jnp.float64)
    r_bases_j = jnp.asarray(r_bases, dtype=jnp.float64)
    theta = jnp.asarray(geom.theta, dtype=jnp.float64)
    sin2 = jnp.asarray(geom.sin2, dtype=jnp.float64)
    cth = jnp.asarray(geom.cth, dtype=jnp.float64)
    sth = jnp.asarray(geom.sth, dtype=jnp.float64)
    z_layers = jnp.asarray(geom.z_layers, dtype=jnp.float64)
    ring_offsets = jnp.asarray(geom.ring_offsets, dtype=jnp.float64)
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    active_rk_j = None
    if ring_active_mask is not None:
        mask = np.asarray(ring_active_mask, dtype=bool)
        if mask.shape != alphas.shape:
            raise ValueError(
                f"ring_active_mask shape {mask.shape} does not match alphas shape {alphas.shape}"
            )
        active_rk_j = jnp.asarray(mask, dtype=jnp.float64)
    factor_val = float(factor)

    def _objective_only(a: Any, b: Any, r: Any) -> tuple[Any, Any]:
        R = a.shape[0]
        K = a.shape[1]
        N = theta.shape[0]

        rho = r[None, :] + ring_offsets[:, None]
        px = rho[:, :, None] * cth[None, None, :]
        py = rho[:, :, None] * sth[None, None, :]
        pz = jnp.broadcast_to(z_layers[None, :, None], (R, K, N))
        r0 = jnp.stack([px, py, pz], axis=-1)

        phi_rkn = 2.0 * theta[None, None, :] + phi0 + a[:, :, None] * sin2[None, None, :]
        beta_rkn = jnp.broadcast_to(b[:, :, None], phi_rkn.shape)
        cos_beta = jnp.cos(beta_rkn)
        ux = cos_beta * jnp.cos(phi_rkn)
        uy = cos_beta * jnp.sin(phi_rkn)
        uz = jnp.sin(beta_rkn)
        u = jnp.stack([ux, uy, uz], axis=-1)
        m_rkn = m0 * u
        if active_rk_j is not None:
            m_rkn = m_rkn * active_rk_j[:, :, None, None]

        r0_flat = r0.reshape(-1, 3)
        m_flat = m_rkn.reshape(-1, 3)
        B_all = _compute_b_all_from_m(m_flat, r0_flat, pts_j, factor_val)
        B = B_all[:-1]
        B0 = B_all[-1]
        diff = B - B0
        J = jnp.mean(jnp.sum(diff * diff, axis=1))
        B0n = jnp.sqrt(jnp.sum(B0 * B0))
        return J, B0n

    (J, B0n), grads = jax.value_and_grad(_objective_only, argnums=(0, 1, 2), has_aux=True)(
        alphas_j, beta_j, r_bases_j
    )
    g_alpha, g_beta, g_rbase = grads

    return (
        float(J),
        np.asarray(g_alpha, dtype=np.float64),
        np.asarray(g_beta, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
    )


__all__ = ["objective_with_grads_fixed_beta_tilt_jax"]
