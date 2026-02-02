from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.autodiff.jax_self_consistent import (
    solve_p_easy_axis_near,
    solve_p_easy_axis_near_cellavg,
    solve_p_easy_axis_near_multi_dipole,
)
from halbach.constants import FACTOR, phi0
from halbach.symmetry_fourier import build_fourier_x0_features
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


def objective_with_grads_self_consistent_delta_phi_fourier_x0_jax(
    coeffs: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    *,
    H: int,
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    iters: int = 30,
    omega: float = 0.6,
    near_kernel: str = "dipole",
    subdip_n: int = 2,
    lambda0: float = 0.0,
    lambda_theta: float = 0.0,
    lambda_z: float = 0.0,
    factor: float = FACTOR,
    phi0: float = phi0,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float, dict[str, float | int | str]]:
    """
    JAX objective and gradients for x=0 mirror delta-phi Fourier model (self-consistent).

    self-consistent uses H kernel factor = FACTOR/mu0 internally (no field_scale).
    TODO: Replace near kernel with multi-dipole / cube-average tensor model.
    """
    if H < 0:
        raise ValueError("H must be >= 0")
    if coeffs.shape[1] != 2 * H:
        raise ValueError("coeffs second dimension must be 2*H")

    near_kernel_norm = "cellavg" if near_kernel == "cube-average" else str(near_kernel)
    cos_odd, sin_even = build_fourier_x0_features(geom.theta, H)

    coeffs_j = jnp.asarray(coeffs, dtype=jnp.float64)
    r_bases_j = jnp.asarray(r_bases, dtype=jnp.float64)
    theta = jnp.asarray(geom.theta, dtype=jnp.float64)
    cth = jnp.asarray(geom.cth, dtype=jnp.float64)
    sth = jnp.asarray(geom.sth, dtype=jnp.float64)
    z_layers = jnp.asarray(geom.z_layers, dtype=jnp.float64)
    ring_offsets = jnp.asarray(geom.ring_offsets, dtype=jnp.float64)
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    cos_odd_j = jnp.asarray(cos_odd, dtype=jnp.float64)
    sin_even_j = jnp.asarray(sin_even, dtype=jnp.float64)
    nbr_idx_j = jnp.asarray(nbr_idx, dtype=jnp.int32)
    nbr_mask_j = jnp.asarray(nbr_mask, dtype=bool)

    factor_val = float(factor)
    chi_val = float(chi)
    p0_val = float(p0)
    Nd_val = float(Nd)
    volume_val = float(volume_m3)
    iters_val = int(iters)
    omega_val = float(omega)
    lambda0_val = float(lambda0)
    lambda_theta_val = float(lambda_theta)
    lambda_z_val = float(lambda_z)
    phi0_val = float(phi0)

    def _objective_only(c: Any, r_b: Any) -> tuple[Any, Any]:
        a = c[:, :H]
        b = c[:, H:]
        delta_full = a @ cos_odd_j + b @ sin_even_j
        phi_kn = 2.0 * theta[None, :] + phi0_val + delta_full
        R = ring_offsets.shape[0]
        K = r_b.shape[0]
        N = theta.shape[0]
        phi = jnp.broadcast_to(phi_kn[None, :, :], (R, K, N))

        rho = r_b[None, :] + ring_offsets[:, None]
        px = rho[:, :, None] * cth[None, None, :]
        py = rho[:, :, None] * sth[None, None, :]
        pz = jnp.broadcast_to(z_layers[None, :, None], (R, K, N))
        r0 = jnp.stack([px, py, pz], axis=-1)

        r0_flat = r0.reshape(-1, 3)
        phi_flat = phi.reshape(-1)

        if chi_val == 0.0:
            p_flat = jnp.full((phi_flat.shape[0],), p0_val, dtype=r0_flat.dtype)
        else:
            if near_kernel_norm == "dipole":
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
            elif near_kernel_norm == "multi-dipole":
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
            elif near_kernel_norm == "cellavg":
                p_flat = solve_p_easy_axis_near_cellavg(
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
        J_data = jnp.mean(jnp.sum(diff * diff, axis=1))
        B0n = jnp.sqrt(jnp.sum(B0 * B0))

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
        return J_total, (B0n, p_min, p_max, p_mean, p_std, p_rel_std)

    (J, aux), grads = jax.value_and_grad(_objective_only, argnums=(0, 1), has_aux=True)(
        coeffs_j, r_bases_j
    )
    B0n, p_min, p_max, p_mean, p_std, p_rel_std = aux
    g_coeffs, g_rbase = grads

    sc_extras: dict[str, float | int | str] = {
        "sc_p_min": float(p_min),
        "sc_p_max": float(p_max),
        "sc_p_mean": float(p_mean),
        "sc_p_std": float(p_std),
        "sc_p_rel_std": float(p_rel_std),
        "sc_near_kernel": str(near_kernel_norm),
        "sc_subdip_n": int(subdip_n),
        "sc_near_deg_max": int(nbr_idx.shape[1]),
    }

    return (
        float(J),
        np.asarray(g_coeffs, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
        sc_extras,
    )


__all__ = ["objective_with_grads_self_consistent_delta_phi_fourier_x0_jax"]
