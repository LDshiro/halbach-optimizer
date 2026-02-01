from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, phi0
from halbach.symmetry import build_mirror_x0
from halbach.types import Geometry

cast(Any, jax.config).update("jax_enable_x64", True)

EPS = 1e-30


def _compute_b_all_from_delta(
    delta_full: Any,
    r_bases: Any,
    theta: Any,
    cth: Any,
    sth: Any,
    z_layers: Any,
    ring_offsets: Any,
    pts: Any,
    factor: float,
    phi0_val: float,
    m0_val: float,
) -> Any:
    R = ring_offsets.shape[0]

    phi = 2.0 * theta[None, :] + phi0_val + delta_full
    mx = m0_val * jnp.cos(phi)
    my = m0_val * jnp.sin(phi)
    mz = jnp.zeros_like(mx)
    m = jnp.stack([mx, my, mz], axis=-1)
    m = jnp.broadcast_to(m[None, :, :, :], (R, m.shape[0], m.shape[1], 3))

    rho = r_bases[None, :] + ring_offsets[:, None]
    px = rho[:, :, None] * cth[None, None, :]
    py = rho[:, :, None] * sth[None, None, :]
    pz = jnp.broadcast_to(z_layers[None, :, None], px.shape)
    r0 = jnp.stack([px, py, pz], axis=-1)

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


def _objective_and_b0_delta(
    delta_full: Any,
    r_bases: Any,
    theta: Any,
    cth: Any,
    sth: Any,
    z_layers: Any,
    ring_offsets: Any,
    pts: Any,
    factor: float,
    phi0_val: float,
    m0_val: float,
) -> tuple[Any, Any]:
    B_all = _compute_b_all_from_delta(
        delta_full,
        r_bases,
        theta,
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


def objective_with_grads_delta_phi_x0_jax(
    delta_rep: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    *,
    lambda0: float = 0.0,
    lambda_theta: float = 0.0,
    lambda_z: float = 0.0,
    factor: float = FACTOR,
    phi0: float = phi0,
    m0: float = m0,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
    """
    JAX objective and gradients for x=0 mirror delta-phi model.
    """
    mirror = build_mirror_x0(int(geom.theta.shape[0]))

    delta_rep_j = jnp.asarray(delta_rep, dtype=jnp.float64)
    r_bases_j = jnp.asarray(r_bases, dtype=jnp.float64)
    theta = jnp.asarray(geom.theta, dtype=jnp.float64)
    cth = jnp.asarray(geom.cth, dtype=jnp.float64)
    sth = jnp.asarray(geom.sth, dtype=jnp.float64)
    z_layers = jnp.asarray(geom.z_layers, dtype=jnp.float64)
    ring_offsets = jnp.asarray(geom.ring_offsets, dtype=jnp.float64)
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    basis = jnp.asarray(mirror.basis, dtype=jnp.float64)
    factor_val = float(factor)
    lambda0_val = float(lambda0)
    lambda_theta_val = float(lambda_theta)
    lambda_z_val = float(lambda_z)

    def _objective_only(d_rep: Any, r_b: Any) -> tuple[Any, Any]:
        delta_full = d_rep @ basis
        J_data, B0n = _objective_and_b0_delta(
            delta_full,
            r_b,
            theta,
            cth,
            sth,
            z_layers,
            ring_offsets,
            pts_j,
            factor_val,
            float(phi0),
            float(m0),
        )
        R0 = jnp.mean(delta_full * delta_full)
        dtheta = jnp.roll(delta_full, -1, axis=1) - delta_full
        Rtheta = jnp.mean(dtheta * dtheta)
        if delta_full.shape[0] > 1:
            dz = delta_full[1:, :] - delta_full[:-1, :]
            Rz = jnp.mean(dz * dz)
        else:
            Rz = jnp.array(0.0, dtype=delta_full.dtype)
        J_total = (
            J_data
            + 0.5 * lambda0_val * R0
            + 0.5 * lambda_theta_val * Rtheta
            + 0.5 * lambda_z_val * Rz
        )
        return J_total, B0n

    (J, B0n), grads = jax.value_and_grad(_objective_only, argnums=(0, 1), has_aux=True)(
        delta_rep_j, r_bases_j
    )
    g_delta, g_rbase = grads

    return (
        float(J),
        np.asarray(g_delta, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
    )


__all__ = ["objective_with_grads_delta_phi_x0_jax"]
