from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, phi0
from halbach.types import Geometry

jax.config.update("jax_enable_x64", True)

EPS = 1e-30


def _compute_b_all(
    alphas: Any,
    r_bases: Any,
    theta: Any,
    sin2: Any,
    cth: Any,
    sth: Any,
    z_layers: Any,
    ring_offsets: Any,
    pts: Any,
    factor: float,
    phi0_val: float,
    m0_val: float,
) -> Any:
    R = alphas.shape[0]
    K = alphas.shape[1]
    N = theta.shape[0]

    rho = r_bases[None, :] + ring_offsets[:, None]
    px = rho[:, :, None] * cth[None, None, :]
    py = rho[:, :, None] * sth[None, None, :]
    pz = jnp.broadcast_to(z_layers[None, :, None], (R, K, N))
    r0 = jnp.stack([px, py, pz], axis=-1)

    phi = 2.0 * theta[None, None, :] + phi0_val + alphas[:, :, None] * sin2[None, None, :]
    mx = m0_val * jnp.cos(phi)
    my = m0_val * jnp.sin(phi)
    mz = jnp.zeros_like(mx)
    m = jnp.stack([mx, my, mz], axis=-1)

    r0_flat = r0.reshape(-1, 3)
    m_flat = m.reshape(-1, 3)

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


def _objective_and_b0(
    alphas: Any,
    r_bases: Any,
    theta: Any,
    sin2: Any,
    cth: Any,
    sth: Any,
    z_layers: Any,
    ring_offsets: Any,
    pts: Any,
    factor: float,
    phi0_val: float,
    m0_val: float,
) -> tuple[Any, Any]:
    B_all = _compute_b_all(
        alphas,
        r_bases,
        theta,
        sin2,
        cth,
        sth,
        z_layers,
        ring_offsets,
        pts,
        factor,
        phi0_val,
        m0_val,
    )
    B = B_all[:-1]
    B0 = B_all[-1]
    diff = B - B0
    J = jnp.mean(jnp.sum(diff * diff, axis=1))
    B0n = jnp.sqrt(jnp.sum(B0 * B0))
    return J, B0n


def objective_with_grads_fixed_jax(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    factor: float = FACTOR,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
    """
    JAX objective and gradients (y-space) with |B0|.
    """
    alphas_j = jnp.asarray(alphas, dtype=jnp.float64)
    r_bases_j = jnp.asarray(r_bases, dtype=jnp.float64)
    theta = jnp.asarray(geom.theta, dtype=jnp.float64)
    sin2 = jnp.asarray(geom.sin2, dtype=jnp.float64)
    cth = jnp.asarray(geom.cth, dtype=jnp.float64)
    sth = jnp.asarray(geom.sth, dtype=jnp.float64)
    z_layers = jnp.asarray(geom.z_layers, dtype=jnp.float64)
    ring_offsets = jnp.asarray(geom.ring_offsets, dtype=jnp.float64)
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    factor_val = float(factor)

    def _objective_only(a: Any, r: Any) -> tuple[Any, Any]:
        return _objective_and_b0(
            a,
            r,
            theta,
            sin2,
            cth,
            sth,
            z_layers,
            ring_offsets,
            pts_j,
            factor_val,
            phi0,
            m0,
        )

    (J, B0n), grads = jax.value_and_grad(_objective_only, argnums=(0, 1), has_aux=True)(
        alphas_j, r_bases_j
    )
    g_alpha, g_rbase = grads

    return (
        float(J),
        np.asarray(g_alpha, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
    )


__all__ = ["objective_with_grads_fixed_jax"]
