from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.autodiff.jax_self_consistent import (
    solve_p_easy_axis_near,
    solve_p_easy_axis_near_multi_dipole,
)
from halbach.constants import FACTOR, phi0
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


def objective_with_grads_self_consistent_legacy_jax(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    *,
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    near_kernel: str = "dipole",
    subdip_n: int = 2,
    iters: int = 30,
    omega: float = 0.6,
    factor: float = FACTOR,
    phi0_val: float = phi0,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float, dict[str, float | int | str]]:
    """
    JAX objective and gradients (y-space) for self-consistent easy-axis model.

    self-consistent uses H kernel factor = FACTOR/mu0 internally (no field_scale).
    TODO: Replace near kernel with multi-dipole / cube-average tensor model.
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
    nbr_idx_j = jnp.asarray(nbr_idx, dtype=jnp.int32)
    nbr_mask_j = jnp.asarray(nbr_mask, dtype=bool)

    factor_val = float(factor)
    chi_val = float(chi)
    p0_val = float(p0)
    Nd_val = float(Nd)
    volume_val = float(volume_m3)
    iters_val = int(iters)
    omega_val = float(omega)

    def _objective_only(a: Any, r: Any) -> tuple[Any, Any]:
        R = a.shape[0]
        K = a.shape[1]
        N = theta.shape[0]
        rho = r[None, :] + ring_offsets[:, None]
        px = rho[:, :, None] * cth[None, None, :]
        py = rho[:, :, None] * sth[None, None, :]
        pz = jnp.broadcast_to(z_layers[None, :, None], (R, K, N))
        r0 = jnp.stack([px, py, pz], axis=-1)

        phi = 2.0 * theta[None, None, :] + phi0_val + a[:, :, None] * sin2[None, None, :]

        r0_flat = r0.reshape(-1, 3)
        phi_flat = phi.reshape(-1)

        if chi_val == 0.0:
            p_flat = jnp.full((phi_flat.shape[0],), p0_val, dtype=r0_flat.dtype)
        else:
            if near_kernel == "dipole":
                p_flat = solve_p_easy_axis_near(
                    phi_flat,
                    r0_flat,
                    nbr_idx_j,
                    nbr_mask_j,
                    p0=p0_val,
                    chi=chi_val,
                    Nd=Nd_val,
                    volume_m3=volume_val,
                    iters=iters_val,
                    omega=omega_val,
                )
            elif near_kernel == "multi-dipole":
                p_flat = solve_p_easy_axis_near_multi_dipole(
                    phi_flat,
                    r0_flat,
                    nbr_idx_j,
                    nbr_mask_j,
                    p0=p0_val,
                    chi=chi_val,
                    Nd=Nd_val,
                    volume_m3=volume_val,
                    subdip_n=int(subdip_n),
                    iters=iters_val,
                    omega=omega_val,
                )
            else:
                raise ValueError(f"Unsupported near_kernel: {near_kernel}")

        p_min = jnp.min(p_flat)
        p_max = jnp.max(p_flat)
        p_mean = jnp.mean(p_flat)
        p_std = jnp.std(p_flat)
        p_rel_std = p_std / (jnp.abs(p_mean) + EPS)

        u = jnp.stack([jnp.cos(phi_flat), jnp.sin(phi_flat), jnp.zeros_like(phi_flat)], axis=1)
        m_flat = p_flat[:, None] * u

        B_all = _compute_b_all_from_m(m_flat, r0_flat, pts_j, factor_val)
        B = B_all[:-1]
        B0 = B_all[-1]
        diff = B - B0
        J = jnp.mean(jnp.sum(diff * diff, axis=1))
        B0n = jnp.sqrt(jnp.sum(B0 * B0))
        return J, (B0n, p_min, p_max, p_mean, p_std, p_rel_std)

    (J, aux), grads = jax.value_and_grad(_objective_only, argnums=(0, 1), has_aux=True)(
        alphas_j, r_bases_j
    )
    B0n, p_min, p_max, p_mean, p_std, p_rel_std = aux
    g_alpha, g_rbase = grads

    sc_extras: dict[str, float | int | str] = {
        "sc_p_min": float(p_min),
        "sc_p_max": float(p_max),
        "sc_p_mean": float(p_mean),
        "sc_p_std": float(p_std),
        "sc_p_rel_std": float(p_rel_std),
        "sc_near_kernel": str(near_kernel),
        "sc_subdip_n": int(subdip_n),
        "sc_near_deg_max": int(nbr_idx.shape[1]),
    }

    return (
        float(J),
        np.asarray(g_alpha, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
        sc_extras,
    )


__all__ = ["objective_with_grads_self_consistent_legacy_jax"]
