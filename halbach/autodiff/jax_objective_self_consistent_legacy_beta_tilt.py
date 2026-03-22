from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from halbach.autodiff.jax_self_consistent import (
    _build_kij_dipole_from_u,
    _build_kij_multi_dipole_from_u,
    _h_from_kij,
    solve_p_easy_axis_near_from_u,
    solve_p_easy_axis_near_multi_dipole_from_u,
)
from halbach.constants import FACTOR, phi0
from halbach.near import edges_from_near
from halbach.types import Geometry

cast(Any, jax.config).update("jax_enable_x64", True)

EPS = 1e-30
JIT_CACHE_MAX = 16
_JIT_CACHE: dict[tuple[Any, ...], object] = {}
_T = TypeVar("_T")


def _cache_get_or_build(key: tuple[Any, ...], build_fn: Callable[[], _T]) -> _T:
    if key in _JIT_CACHE:
        return cast(_T, _JIT_CACHE[key])
    if len(_JIT_CACHE) >= JIT_CACHE_MAX:
        _JIT_CACHE.clear()
    fn = build_fn()
    _JIT_CACHE[key] = fn
    return fn


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


def _build_objective_only(
    *,
    theta: jnp.ndarray,
    sin2: jnp.ndarray,
    cth: jnp.ndarray,
    sth: jnp.ndarray,
    z_layers: jnp.ndarray,
    ring_offsets: jnp.ndarray,
    pts: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    i_edge: jnp.ndarray,
    j_edge: jnp.ndarray,
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    iters: int,
    omega: float,
    near_kernel_norm: str,
    subdip_n: int,
    factor_val: float,
    phi0_val: float,
    implicit_diff: bool,
) -> Callable[[Any, Any, Any], tuple[Any, Any]]:
    denom = 1.0 + chi * Nd

    def _objective_only(a: Any, b: Any, r: Any) -> tuple[Any, Any]:
        R = a.shape[0]
        K = a.shape[1]
        N = theta.shape[0]

        rho = r[None, :] + ring_offsets[:, None]
        px = rho[:, :, None] * cth[None, None, :]
        py = rho[:, :, None] * sth[None, None, :]
        pz = jnp.broadcast_to(z_layers[None, :, None], (R, K, N))
        r0 = jnp.stack([px, py, pz], axis=-1)

        phi_rkn = 2.0 * theta[None, None, :] + phi0_val + a[:, :, None] * sin2[None, None, :]
        beta_rkn = jnp.broadcast_to(b[:, :, None], phi_rkn.shape)
        cos_beta = jnp.cos(beta_rkn)
        ux = cos_beta * jnp.cos(phi_rkn)
        uy = cos_beta * jnp.sin(phi_rkn)
        uz = jnp.sin(beta_rkn)
        u_rkn = jnp.stack([ux, uy, uz], axis=-1)

        r0_flat = r0.reshape(-1, 3)
        u_flat = u_rkn.reshape(-1, 3)

        if chi == 0.0:
            p_flat = jnp.full((u_flat.shape[0],), p0, dtype=r0_flat.dtype)
            h_ext = jnp.zeros_like(p_flat)
        else:
            if near_kernel_norm == "dipole":
                p_flat = solve_p_easy_axis_near_from_u(
                    u_flat,
                    r0_flat,
                    nbr_idx,
                    nbr_mask,
                    p0=p0,
                    chi=chi,
                    Nd=Nd,
                    volume_m3=volume_m3,
                    iters=iters,
                    omega=omega,
                    implicit_diff=implicit_diff,
                    i_edge=i_edge,
                    j_edge=j_edge,
                )
                kij = _build_kij_dipole_from_u(u_flat, r0_flat, nbr_idx, nbr_mask)
                h_ext = _h_from_kij(p_flat, nbr_idx, nbr_mask, kij)
            elif near_kernel_norm == "multi-dipole":
                p_flat = solve_p_easy_axis_near_multi_dipole_from_u(
                    u_flat,
                    r0_flat,
                    nbr_idx,
                    nbr_mask,
                    p0=p0,
                    chi=chi,
                    Nd=Nd,
                    volume_m3=volume_m3,
                    subdip_n=int(subdip_n),
                    iters=iters,
                    omega=omega,
                    implicit_diff=implicit_diff,
                    i_edge=i_edge,
                    j_edge=j_edge,
                )
                cube_edge = float(volume_m3) ** (1.0 / 3.0)
                offsets = (np.arange(int(subdip_n), dtype=np.float64) + 0.5) / float(subdip_n) - 0.5
                offsets = offsets * cube_edge
                xx, yy, zz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
                offsets_j = jnp.asarray(
                    np.stack([xx, yy, zz], axis=-1).reshape(-1, 3), dtype=jnp.float64
                )
                kij = _build_kij_multi_dipole_from_u(u_flat, r0_flat, nbr_idx, nbr_mask, offsets_j)
                h_ext = _h_from_kij(p_flat, nbr_idx, nbr_mask, kij)
            else:
                raise ValueError(
                    "beta_tilt_x with self-consistent supports near_kernel only in {'dipole','multi-dipole'}"
                )

        p_min = jnp.min(p_flat)
        p_max = jnp.max(p_flat)
        p_mean = jnp.mean(p_flat)
        p_std = jnp.std(p_flat)
        p_rel_std = p_std / (jnp.abs(p_mean) + EPS)

        res_vec = denom * p_flat - (p0 + chi * volume_m3 * h_ext)
        res_rel = jnp.linalg.norm(res_vec) / jnp.maximum(jnp.linalg.norm(p_flat), EPS)

        m_flat = p_flat[:, None] * u_flat
        B_all = _compute_b_all_from_m(m_flat, r0_flat, pts, factor_val)
        B = B_all[:-1]
        B0 = B_all[-1]
        diff = B - B0
        J = jnp.mean(jnp.sum(diff * diff, axis=1))
        B0n = jnp.sqrt(jnp.sum(B0 * B0))
        return J, (B0n, p_min, p_max, p_mean, p_std, p_rel_std, res_rel)

    return _objective_only


def _get_compiled_vg(
    *,
    theta: NDArray[np.float64],
    sin2: NDArray[np.float64],
    cth: NDArray[np.float64],
    sth: NDArray[np.float64],
    z_layers: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    pts: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    iters: int,
    omega: float,
    near_kernel_norm: str,
    subdip_n: int,
    factor_val: float,
    phi0_val: float,
    implicit_diff: bool,
) -> Callable[..., Any]:
    key = (
        id(theta),
        id(sin2),
        id(cth),
        id(sth),
        id(z_layers),
        id(ring_offsets),
        id(pts),
        pts.shape,
        id(nbr_idx),
        id(nbr_mask),
        nbr_idx.shape,
        int(np.count_nonzero(nbr_mask)),
        float(chi),
        float(Nd),
        float(p0),
        float(volume_m3),
        int(iters),
        float(omega),
        str(near_kernel_norm),
        int(subdip_n),
        float(factor_val),
        float(phi0_val),
        bool(implicit_diff),
    )

    def build_fn() -> Callable[..., Any]:
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        sin2_j = jnp.asarray(sin2, dtype=jnp.float64)
        cth_j = jnp.asarray(cth, dtype=jnp.float64)
        sth_j = jnp.asarray(sth, dtype=jnp.float64)
        z_layers_j = jnp.asarray(z_layers, dtype=jnp.float64)
        ring_offsets_j = jnp.asarray(ring_offsets, dtype=jnp.float64)
        pts_j = jnp.asarray(pts, dtype=jnp.float64)
        nbr_idx_j = jnp.asarray(nbr_idx, dtype=jnp.int32)
        nbr_mask_j = jnp.asarray(nbr_mask, dtype=bool)
        i_edge_np, j_edge_np = edges_from_near(
            np.asarray(nbr_idx, dtype=np.int32), np.asarray(nbr_mask, dtype=bool)
        )
        i_edge_j = jnp.asarray(i_edge_np, dtype=jnp.int32)
        j_edge_j = jnp.asarray(j_edge_np, dtype=jnp.int32)
        objective_only = _build_objective_only(
            theta=theta_j,
            sin2=sin2_j,
            cth=cth_j,
            sth=sth_j,
            z_layers=z_layers_j,
            ring_offsets=ring_offsets_j,
            pts=pts_j,
            nbr_idx=nbr_idx_j,
            nbr_mask=nbr_mask_j,
            i_edge=i_edge_j,
            j_edge=j_edge_j,
            chi=float(chi),
            Nd=float(Nd),
            p0=float(p0),
            volume_m3=float(volume_m3),
            iters=int(iters),
            omega=float(omega),
            near_kernel_norm=str(near_kernel_norm),
            subdip_n=int(subdip_n),
            factor_val=float(factor_val),
            phi0_val=float(phi0_val),
            implicit_diff=bool(implicit_diff),
        )
        return cast(
            Callable[..., Any],
            jax.jit(jax.value_and_grad(objective_only, argnums=(0, 1, 2), has_aux=True)),
        )

    return _cache_get_or_build(key, build_fn)


def objective_with_grads_self_consistent_legacy_beta_tilt_jax(
    alphas: NDArray[np.float64],
    beta_tilt_x: NDArray[np.float64],
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
    use_jit: bool = True,
    implicit_diff: bool = True,
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    dict[str, float | int | str],
]:
    """JAX objective and gradients for legacy-alpha + beta_tilt_x (self-consistent)."""
    near_kernel_norm = "cellavg" if near_kernel == "cube-average" else str(near_kernel)
    if near_kernel_norm not in ("dipole", "multi-dipole"):
        raise ValueError(
            "beta_tilt_x with self-consistent supports near_kernel only in {'dipole','multi-dipole'}"
        )

    alphas_j = jnp.asarray(alphas, dtype=jnp.float64)
    beta_j = jnp.asarray(beta_tilt_x, dtype=jnp.float64)
    r_bases_j = jnp.asarray(r_bases, dtype=jnp.float64)

    factor_val = float(factor)
    chi_val = float(chi)
    p0_val = float(p0)
    Nd_val = float(Nd)
    volume_val = float(volume_m3)
    iters_val = int(iters)
    omega_val = float(omega)

    if use_jit:
        vg = _get_compiled_vg(
            theta=np.asarray(geom.theta, dtype=np.float64),
            sin2=np.asarray(geom.sin2, dtype=np.float64),
            cth=np.asarray(geom.cth, dtype=np.float64),
            sth=np.asarray(geom.sth, dtype=np.float64),
            z_layers=np.asarray(geom.z_layers, dtype=np.float64),
            ring_offsets=np.asarray(geom.ring_offsets, dtype=np.float64),
            pts=np.asarray(pts, dtype=np.float64),
            nbr_idx=np.asarray(nbr_idx, dtype=np.int32),
            nbr_mask=np.asarray(nbr_mask, dtype=bool),
            chi=float(chi_val),
            Nd=float(Nd_val),
            p0=float(p0_val),
            volume_m3=float(volume_val),
            iters=int(iters_val),
            omega=float(omega_val),
            near_kernel_norm=str(near_kernel_norm),
            subdip_n=int(subdip_n),
            factor_val=float(factor_val),
            phi0_val=float(phi0_val),
            implicit_diff=bool(implicit_diff),
        )
        (J, aux), grads = vg(alphas_j, beta_j, r_bases_j)
    else:
        theta_j = jnp.asarray(geom.theta, dtype=jnp.float64)
        sin2_j = jnp.asarray(geom.sin2, dtype=jnp.float64)
        cth_j = jnp.asarray(geom.cth, dtype=jnp.float64)
        sth_j = jnp.asarray(geom.sth, dtype=jnp.float64)
        z_layers_j = jnp.asarray(geom.z_layers, dtype=jnp.float64)
        ring_offsets_j = jnp.asarray(geom.ring_offsets, dtype=jnp.float64)
        pts_j = jnp.asarray(pts, dtype=jnp.float64)
        nbr_idx_j = jnp.asarray(nbr_idx, dtype=jnp.int32)
        nbr_mask_j = jnp.asarray(nbr_mask, dtype=bool)
        i_edge_np, j_edge_np = edges_from_near(
            np.asarray(nbr_idx, dtype=np.int32), np.asarray(nbr_mask, dtype=bool)
        )
        i_edge_j = jnp.asarray(i_edge_np, dtype=jnp.int32)
        j_edge_j = jnp.asarray(j_edge_np, dtype=jnp.int32)
        objective_only = _build_objective_only(
            theta=theta_j,
            sin2=sin2_j,
            cth=cth_j,
            sth=sth_j,
            z_layers=z_layers_j,
            ring_offsets=ring_offsets_j,
            pts=pts_j,
            nbr_idx=nbr_idx_j,
            nbr_mask=nbr_mask_j,
            i_edge=i_edge_j,
            j_edge=j_edge_j,
            chi=float(chi_val),
            Nd=float(Nd_val),
            p0=float(p0_val),
            volume_m3=float(volume_val),
            iters=int(iters_val),
            omega=float(omega_val),
            near_kernel_norm=str(near_kernel_norm),
            subdip_n=int(subdip_n),
            factor_val=float(factor_val),
            phi0_val=float(phi0_val),
            implicit_diff=bool(implicit_diff),
        )
        (J, aux), grads = jax.value_and_grad(objective_only, argnums=(0, 1, 2), has_aux=True)(
            alphas_j, beta_j, r_bases_j
        )

    B0n, p_min, p_max, p_mean, p_std, p_rel_std, res_rel = aux
    g_alpha, g_beta, g_rbase = grads

    sc_extras: dict[str, float | int | str] = {
        "sc_p_min": float(p_min),
        "sc_p_max": float(p_max),
        "sc_p_mean": float(p_mean),
        "sc_p_std": float(p_std),
        "sc_p_rel_std": float(p_rel_std),
        "sc_residual_rel": float(res_rel),
        "sc_near_kernel": str(near_kernel_norm),
        "sc_subdip_n": int(subdip_n),
        "sc_near_deg_max": int(nbr_idx.shape[1]),
    }

    return (
        float(J),
        np.asarray(g_alpha, dtype=np.float64),
        np.asarray(g_beta, dtype=np.float64),
        np.asarray(g_rbase, dtype=np.float64),
        float(B0n),
        sc_extras,
    )


__all__ = ["objective_with_grads_self_consistent_legacy_beta_tilt_jax"]
