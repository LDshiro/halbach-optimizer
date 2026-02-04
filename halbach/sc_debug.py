from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, mu0, phi0
from halbach.magnetization_runtime import get_magnetization_config_from_meta, sc_cfg_fingerprint
from halbach.near import NearWindow, build_near_graph, flatten_index, unflatten_index
from halbach.physics import compute_B_and_B0_from_m_flat
from halbach.sc_linear_system import (
    build_A_sparse,
    build_b,
    build_C_edges_from_phi,
    build_C_sparse,
    build_T2_edges,
    edges_from_near,
    residual_norm,
    solve_p_linear,
)
from halbach.types import Geometry

EPS = 1e-30
H_FACTOR = float(FACTOR / mu0)


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"meta.json in {run_dir} must be a JSON object")
    return data


def _parse_sc_cfg(sc_cfg: dict[str, Any]) -> dict[str, Any]:
    near_window = sc_cfg.get("near_window", {})
    if not isinstance(near_window, dict):
        raise ValueError("self_consistent.near_window must be a dict")
    near_kernel = str(sc_cfg.get("near_kernel", "dipole"))
    if near_kernel == "cube-average":
        near_kernel = "cellavg"
    return dict(
        chi=float(sc_cfg.get("chi", 0.0)),
        Nd=float(sc_cfg.get("Nd", 1.0 / 3.0)),
        p0=float(sc_cfg.get("p0", float(m0))),
        volume_mm3=float(sc_cfg.get("volume_mm3", 1000.0)),
        iters=int(sc_cfg.get("iters", 30)),
        omega=float(sc_cfg.get("omega", 0.6)),
        near_window=dict(
            wr=int(near_window.get("wr", 0)),
            wz=int(near_window.get("wz", 1)),
            wphi=int(near_window.get("wphi", 2)),
        ),
        near_kernel=near_kernel,
        subdip_n=int(sc_cfg.get("subdip_n", 2)),
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _neighbor_hist(counts: NDArray[np.int_]) -> dict[str, int]:
    unique, freqs = np.unique(counts, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, freqs, strict=True)}


def _load_saved_p(
    run_dir: Path, fp_expected: str, expected_size: int
) -> NDArray[np.float64] | None:
    results_path = run_dir / "results.npz"
    if not results_path.is_file():
        return None
    try:
        with np.load(results_path, allow_pickle=False) as data:
            if "sc_p_flat" not in data or "sc_cfg_fingerprint" not in data:
                return None
            p_candidate = np.asarray(data["sc_p_flat"], dtype=np.float64)
            fp_raw = np.asarray(data["sc_cfg_fingerprint"])
            fp_saved = str(fp_raw.reshape(-1)[0])
            if fp_saved != fp_expected:
                return None
            if p_candidate.shape != (expected_size,):
                return None
            return p_candidate
    except Exception:
        return None


def _compute_h_ext_u_numpy(
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    u_i = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    r_j = r0_flat[nbr_idx]
    p_j = p_flat[nbr_idx]
    phi_j = phi_flat[nbr_idx]
    u_j = np.stack([np.cos(phi_j), np.sin(phi_j), np.zeros_like(phi_j)], axis=-1)
    m_j = p_j[..., None] * u_j

    r_i = r0_flat[:, None, :]
    r_ij = r_i - r_j
    mask_f = nbr_mask.astype(np.float64)
    r_ij_safe = r_ij * mask_f[..., None] + (1.0 - mask_f[..., None]) * np.array(
        [1.0, 0.0, 0.0], dtype=r0_flat.dtype
    )

    r2 = np.sum(r_ij_safe * r_ij_safe, axis=-1)
    rmag = np.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    invr5 = invr3 / r2
    mdotr = np.sum(m_j * r_ij_safe, axis=-1)
    term = 3.0 * r_ij_safe * mdotr[..., None] * invr5[..., None] - m_j * invr3[..., None]
    H_ij = H_FACTOR * term
    H_ij = H_ij * mask_f[..., None]
    H_sum = np.sum(H_ij, axis=1)
    return np.asarray(np.sum(H_sum * u_i, axis=1), dtype=np.float64)


def _multi_dipole_offsets(subdip_n: int, cube_edge: float) -> NDArray[np.float64]:
    coords = (np.arange(subdip_n, dtype=np.float64) + 0.5) / subdip_n - 0.5
    coords = coords * float(cube_edge)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return cast(NDArray[np.float64], np.stack([xx, yy, zz], axis=-1).reshape(-1, 3))


def _compute_h_ext_u_numpy_multi(
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    u_i = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    r_j = r0_flat[nbr_idx]
    p_j = p_flat[nbr_idx]
    phi_j = phi_flat[nbr_idx]
    u_j = np.stack([np.cos(phi_j), np.sin(phi_j), np.zeros_like(phi_j)], axis=-1)
    m_j = p_j[..., None] * u_j

    r_i = r0_flat[:, None, :]
    r_ij = r_i - r_j
    mask_f = nbr_mask.astype(np.float64)
    m_j_safe = m_j * mask_f[..., None]
    r_ij_safe = r_ij * mask_f[..., None] + (1.0 - mask_f[..., None]) * np.array(
        [1.0, 0.0, 0.0], dtype=r0_flat.dtype
    )

    S = int(offsets.shape[0])
    r_sub = r_ij_safe[..., None, :] - offsets[None, None, :, :]
    m_sub = m_j_safe[..., None, :] / float(S)

    r2 = np.sum(r_sub * r_sub, axis=-1)
    rmag = np.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    invr5 = invr3 / r2
    mdotr = np.sum(m_sub * r_sub, axis=-1)
    term = 3.0 * r_sub * mdotr[..., None] * invr5[..., None] - m_sub * invr3[..., None]
    H_sub = H_FACTOR * term
    H_sub = H_sub * mask_f[..., None, None]
    H_sum = np.sum(H_sub, axis=(1, 2))
    return np.asarray(np.sum(H_sum * u_i, axis=1), dtype=np.float64)


def _compute_h_ext_u_numpy_cellavg(
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    volume_m3: float,
) -> NDArray[np.float64]:
    from halbach.demag_cellavg import demag_N_cellavg

    u_flat = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    r_i = r0_flat[:, None, :]
    r_j = r0_flat[nbr_idx]
    s_ij = r_i - r_j
    u_i = u_flat[:, None, :]
    u_j = u_flat[nbr_idx]
    s_ij = np.where(nbr_mask[..., None], s_ij, 0.0)
    u_j = np.where(nbr_mask[..., None], u_j, 0.0)
    a = float(volume_m3) ** (1.0 / 3.0)
    N_ij = demag_N_cellavg(s_ij.reshape(-1, 3), (a, a, a)).reshape(
        s_ij.shape[0], s_ij.shape[1], 3, 3
    )
    Nu_j = np.einsum("...ab,...b->...a", N_ij, u_j)
    c_ij = -np.einsum("...a,...a->...", u_i, Nu_j)
    c_ij = np.where(nbr_mask, c_ij, 0.0)
    p_j = p_flat[nbr_idx]
    p_j = np.where(nbr_mask, p_j, 0.0)
    h_ext = np.sum(c_ij * (p_j / float(volume_m3)), axis=1)
    return np.asarray(h_ext, dtype=np.float64)


def _compute_j_from_m_flat(
    m_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
) -> float:
    origin = np.zeros(3, dtype=np.float64)
    B, B0 = compute_B_and_B0_from_m_flat(pts, r0_flat, m_flat, float(factor), origin)
    diff = B - B0
    return float(np.mean(np.sum(diff * diff, axis=1)))


def _write_solver_trace_csv(
    path: Path,
    trace_rows: list[dict[str, float]],
) -> None:
    fieldnames = ["iter", "rel_change", "p_mean", "p_std", "p_min", "p_max", "p_rel_std"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow({k: row.get(k, 0.0) for k in fieldnames})


def _compute_p_trace_jax(
    *,
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    iters: int,
    omega: float,
    near_kernel: str,
    subdip_n: int,
) -> tuple[NDArray[np.float64], list[dict[str, float]]]:
    phi = np.asarray(phi_flat, dtype=np.float64)
    r0 = np.asarray(r0_flat, dtype=np.float64)
    nbr_idx_n = np.asarray(nbr_idx, dtype=np.int32)
    nbr_mask_n = np.asarray(nbr_mask, dtype=bool)

    offsets: NDArray[np.float64] | None = None
    c_ij: NDArray[np.float64] | None = None
    if near_kernel == "multi-dipole":
        cube_edge = float(volume_m3) ** (1.0 / 3.0)
        offsets = _multi_dipole_offsets(subdip_n, cube_edge)
    elif near_kernel == "cellavg":
        from halbach.demag_cellavg import demag_N_cellavg

        a = float(volume_m3) ** (1.0 / 3.0)
        h = (a, a, a)
        u_flat = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
        r_i = r0[:, None, :]
        r_j = r0[nbr_idx_n]
        s_ij = r_i - r_j
        u_i = u_flat[:, None, :]
        u_j = u_flat[nbr_idx_n]
        s_ij = np.where(nbr_mask_n[..., None], s_ij, 0.0)
        u_j = np.where(nbr_mask_n[..., None], u_j, 0.0)
        N_ij = demag_N_cellavg(s_ij.reshape(-1, 3), h).reshape(s_ij.shape[0], s_ij.shape[1], 3, 3)
        Nu_j = np.einsum("...ab,...b->...a", N_ij, u_j)
        c_ij = -np.einsum("...a,...a->...", u_i, Nu_j)
        c_ij = np.where(nbr_mask_n, c_ij, 0.0)

    denom = 1.0 + chi * Nd
    p = np.full((phi.shape[0],), float(p0), dtype=np.float64)
    trace: list[dict[str, float]] = []

    for i in range(int(iters)):
        if chi == 0.0:
            p_new = p
        else:
            if near_kernel == "multi-dipole":
                if offsets is None:
                    raise ValueError("offsets missing for multi-dipole kernel")
                h_ext = _compute_h_ext_u_numpy_multi(
                    phi,
                    r0,
                    p,
                    nbr_idx_n,
                    nbr_mask_n,
                    offsets,
                )
            elif near_kernel == "cellavg":
                if c_ij is None:
                    raise ValueError("c_ij missing for cellavg kernel")
                p_j = p[nbr_idx_n]
                p_j = np.where(nbr_mask_n, p_j, 0.0)
                h_ext = np.sum(c_ij * (p_j / float(volume_m3)), axis=1)
            else:
                h_ext = _compute_h_ext_u_numpy(
                    phi,
                    r0,
                    p,
                    nbr_idx_n,
                    nbr_mask_n,
                )
            p_new = (float(p0) + float(chi) * float(volume_m3) * h_ext) / denom
        p_next = (1.0 - float(omega)) * p + float(omega) * p_new
        rel_change = float(np.linalg.norm(p_next - p) / (np.linalg.norm(p) + EPS))
        p_min = float(np.min(p_next))
        p_max = float(np.max(p_next))
        p_mean = float(np.mean(p_next))
        p_std = float(np.std(p_next))
        p_rel_std = p_std / (abs(p_mean) + EPS)
        trace.append(
            dict(
                iter=float(i),
                rel_change=rel_change,
                p_mean=p_mean,
                p_std=p_std,
                p_min=p_min,
                p_max=p_max,
                p_rel_std=p_rel_std,
            )
        )
        p = np.asarray(p_next, dtype=np.float64)

    return np.asarray(p, dtype=np.float64), trace


def _build_linear_system(
    *,
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    near_kernel: str,
    subdip_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    M = int(phi_flat.shape[0])
    deg = int(nbr_idx.shape[1])

    i_all = np.repeat(np.arange(M, dtype=np.int64), deg)
    j_all = nbr_idx.reshape(-1)
    m_all = nbr_mask.reshape(-1)
    i = i_all[m_all]
    j = j_all[m_all]
    E = int(i.shape[0])

    s = r0_flat[i] - r0_flat[j]
    ui = np.stack([np.cos(phi_flat[i]), np.sin(phi_flat[i]), np.zeros_like(phi_flat[i])], axis=1)
    uj = np.stack([np.cos(phi_flat[j]), np.sin(phi_flat[j]), np.zeros_like(phi_flat[j])], axis=1)

    V = float(volume_m3)
    if near_kernel == "multi-dipole":
        a = V ** (1.0 / 3.0)
        offsets = _multi_dipole_offsets(subdip_n, a)
        r_sub = s[:, None, :] - offsets[None, :, :]
        r2 = np.sum(r_sub * r_sub, axis=-1)
        rmag = np.sqrt(r2) + EPS
        invr3 = 1.0 / (rmag * r2 + EPS)
        invr5 = invr3 / r2
        mdotr = np.sum(uj[:, None, :] * r_sub, axis=2)
        term = 3.0 * r_sub * mdotr[..., None] * invr5[..., None] - uj[:, None, :] * invr3[..., None]
        H_sub = (1.0 / (4.0 * np.pi)) * term
        H_sum = np.mean(H_sub, axis=1)
        scalar = np.sum(ui * H_sum, axis=1)
        C_ij = V * scalar
    elif near_kernel == "cellavg":
        from halbach.demag_cellavg import demag_N_cellavg

        a = V ** (1.0 / 3.0)
        N = demag_N_cellavg(s, (a, a, a))
        C_ij = -np.einsum("ei,eij,ej->e", ui, N, uj)
    else:
        r2 = np.sum(s * s, axis=1)
        rmag = np.sqrt(r2) + EPS
        invr3 = 1.0 / (rmag * r2 + EPS)
        invr5 = invr3 / r2
        mdotr = np.sum(uj * s, axis=1)
        term = 3.0 * s * mdotr[:, None] * invr5[:, None] - uj * invr3[:, None]
        H = (1.0 / (4.0 * np.pi)) * term
        scalar = np.sum(ui * H, axis=1)
        C_ij = V * scalar

    C = np.zeros((M, M), dtype=np.float64)
    if E > 0:
        np.add.at(C, (i, j), C_ij)
    C_max_abs = float(np.max(np.abs(C_ij))) if E > 0 else 0.0
    C_row_abs_sum_max = float(np.max(np.sum(np.abs(C), axis=1))) if M > 0 else 0.0

    denom = 1.0 + chi * Nd
    alpha = chi / denom if denom != 0.0 else 0.0
    A = np.eye(M, dtype=np.float64) - alpha * C
    b = (p0 / denom) * np.ones(M, dtype=np.float64)
    return A, b, dict(E=float(E), C_max_abs=C_max_abs, C_row_abs_sum_max=C_row_abs_sum_max)


def _field_scale_check(
    *,
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    scale_factor: float,
) -> dict[str, Any]:
    u = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    m_flat = np.asarray(p_flat[:, None] * u, dtype=np.float64)
    J0 = _compute_j_from_m_flat(m_flat, r0_flat, pts, float(factor))
    J1 = _compute_j_from_m_flat(m_flat, r0_flat, pts, float(factor) * float(scale_factor))
    ratio = J1 / J0 if J0 != 0.0 else float("nan")
    expected = float(scale_factor) ** 2
    rel_err = abs(ratio - expected) / (abs(expected) + EPS)
    return dict(
        ratio_J=ratio,
        expected=expected,
        rel_error=rel_err,
        pass_ratio=bool(rel_err < 1e-3),
    )


def _objective_p_invariance_check(
    *,
    run_dir: Path,
    geom: Geometry,
    pts: NDArray[np.float64],
    near_kernel: str,
    subdip_n: int,
    chi: float,
    Nd: float,
    p0: float,
    volume_m3: float,
    iters: int,
    omega: float,
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    scale_factor: float,
) -> dict[str, Any]:
    meta = _load_meta(run_dir)
    angle_model = str(meta.get("angle_model", "legacy-alpha"))
    regularization = meta.get("regularization", {})
    lambda0 = float(regularization.get("lambda0", 0.0)) if isinstance(regularization, dict) else 0.0
    lambda_theta = (
        float(regularization.get("lambda_theta", 0.0)) if isinstance(regularization, dict) else 0.0
    )
    lambda_z = (
        float(regularization.get("lambda_z", 0.0)) if isinstance(regularization, dict) else 0.0
    )

    results_path = run_dir / "results.npz"
    if not results_path.is_file():
        return dict(checked=False, reason="results.npz not found")

    with np.load(results_path, allow_pickle=False) as data:
        if angle_model == "legacy-alpha":
            if "alphas_opt" not in data or "r_bases_opt" not in data:
                return dict(checked=False, reason="alphas_opt/r_bases_opt missing")
            alphas = np.asarray(data["alphas_opt"], dtype=np.float64)
            r_bases = np.asarray(data["r_bases_opt"], dtype=np.float64)
            from halbach.autodiff.jax_objective_self_consistent_legacy import (
                objective_with_grads_self_consistent_legacy_jax,
            )

            J0, _gA, _gR, _B0, sc0 = objective_with_grads_self_consistent_legacy_jax(
                alphas,
                r_bases,
                geom,
                pts,
                nbr_idx,
                nbr_mask,
                chi=chi,
                Nd=Nd,
                p0=p0,
                volume_m3=volume_m3,
                near_kernel=near_kernel,
                subdip_n=subdip_n,
                iters=iters,
                omega=omega,
                factor=FACTOR,
                phi0_val=phi0,
            )
            J1, _gA1, _gR1, _B01, sc1 = objective_with_grads_self_consistent_legacy_jax(
                alphas,
                r_bases,
                geom,
                pts,
                nbr_idx,
                nbr_mask,
                chi=chi,
                Nd=Nd,
                p0=p0,
                volume_m3=volume_m3,
                near_kernel=near_kernel,
                subdip_n=subdip_n,
                iters=iters,
                omega=omega,
                factor=FACTOR * float(scale_factor),
                phi0_val=phi0,
            )
        elif angle_model == "delta-rep-x0":
            if "delta_rep_opt" not in data or "r_bases_opt" not in data:
                return dict(checked=False, reason="delta_rep_opt/r_bases_opt missing")
            delta_rep = np.asarray(data["delta_rep_opt"], dtype=np.float64)
            r_bases = np.asarray(data["r_bases_opt"], dtype=np.float64)
            from halbach.autodiff.jax_objective_self_consistent_delta_phi import (
                objective_with_grads_self_consistent_delta_phi_x0_jax,
            )

            J0, _gD, _gR, _B0, sc0 = objective_with_grads_self_consistent_delta_phi_x0_jax(
                delta_rep,
                r_bases,
                geom,
                pts,
                nbr_idx,
                nbr_mask,
                chi=chi,
                Nd=Nd,
                p0=p0,
                volume_m3=volume_m3,
                iters=iters,
                omega=omega,
                near_kernel=near_kernel,
                subdip_n=subdip_n,
                lambda0=lambda0,
                lambda_theta=lambda_theta,
                lambda_z=lambda_z,
                factor=FACTOR,
                phi0=phi0,
            )
            J1, _gD1, _gR1, _B01, sc1 = objective_with_grads_self_consistent_delta_phi_x0_jax(
                delta_rep,
                r_bases,
                geom,
                pts,
                nbr_idx,
                nbr_mask,
                chi=chi,
                Nd=Nd,
                p0=p0,
                volume_m3=volume_m3,
                iters=iters,
                omega=omega,
                near_kernel=near_kernel,
                subdip_n=subdip_n,
                lambda0=lambda0,
                lambda_theta=lambda_theta,
                lambda_z=lambda_z,
                factor=FACTOR * float(scale_factor),
                phi0=phi0,
            )
        elif angle_model == "fourier-x0":
            if "fourier_coeffs_opt" not in data or "r_bases_opt" not in data:
                return dict(checked=False, reason="fourier_coeffs_opt/r_bases_opt missing")
            coeffs = np.asarray(data["fourier_coeffs_opt"], dtype=np.float64)
            r_bases = np.asarray(data["r_bases_opt"], dtype=np.float64)
            meta_H = meta.get("fourier_H")
            if meta_H is None:
                H = int(coeffs.shape[1] // 2)
            else:
                H = int(meta_H)
            from halbach.autodiff.jax_objective_self_consistent_delta_phi_fourier import (
                objective_with_grads_self_consistent_delta_phi_fourier_x0_jax,
            )

            J0, _gC, _gR, _B0, sc0 = objective_with_grads_self_consistent_delta_phi_fourier_x0_jax(
                coeffs,
                r_bases,
                geom,
                pts,
                nbr_idx,
                nbr_mask,
                H=H,
                chi=chi,
                Nd=Nd,
                p0=p0,
                volume_m3=volume_m3,
                iters=iters,
                omega=omega,
                near_kernel=near_kernel,
                subdip_n=subdip_n,
                lambda0=lambda0,
                lambda_theta=lambda_theta,
                lambda_z=lambda_z,
                factor=FACTOR,
                phi0=phi0,
            )
            J1, _gC1, _gR1, _B01, sc1 = (
                objective_with_grads_self_consistent_delta_phi_fourier_x0_jax(
                    coeffs,
                    r_bases,
                    geom,
                    pts,
                    nbr_idx,
                    nbr_mask,
                    H=H,
                    chi=chi,
                    Nd=Nd,
                    p0=p0,
                    volume_m3=volume_m3,
                    iters=iters,
                    omega=omega,
                    near_kernel=near_kernel,
                    subdip_n=subdip_n,
                    lambda0=lambda0,
                    lambda_theta=lambda_theta,
                    lambda_z=lambda_z,
                    factor=FACTOR * float(scale_factor),
                    phi0=phi0,
                )
            )
        else:
            return dict(checked=False, reason=f"unsupported angle_model: {angle_model}")

    p_mean0 = float(sc0.get("sc_p_mean", float("nan")))
    p_mean1 = float(sc1.get("sc_p_mean", float("nan")))
    p_std0 = float(sc0.get("sc_p_std", float("nan")))
    p_std1 = float(sc1.get("sc_p_std", float("nan")))
    delta_mean = abs(p_mean1 - p_mean0)
    delta_std = abs(p_std1 - p_std0)
    return dict(
        checked=True,
        angle_model=angle_model,
        J0=float(J0),
        J1=float(J1),
        p_mean0=p_mean0,
        p_mean1=p_mean1,
        p_std0=p_std0,
        p_std1=p_std1,
        p_mean_delta=delta_mean,
        p_std_delta=delta_std,
        pass_p_invariance=bool(delta_mean < 1e-9 and delta_std < 1e-9),
    )


def make_sc_debug_bundle(
    *,
    run_dir: Path,
    out_dir: Path,
    geom: Geometry,
    phi_rkn: NDArray[np.float64],
    r0_rkn: NDArray[np.float64],
    pts: NDArray[np.float64] | None,
    factor: float | None,
    field_scale_check: bool = True,
    scale_factor: float = 10.0,
    sample_magnets: int = 16,
    seed: int = 1234,
    force_compute_p: bool = False,
    sc_cfg_override: dict[str, Any] | None = None,
    linear_check: bool = False,
    linear_check_max_M: int = 256,
) -> Path:
    debug_dir = _ensure_dir(out_dir / "sc_debug")
    meta = _load_meta(run_dir)
    model_effective, sc_cfg_raw = get_magnetization_config_from_meta(meta)

    summary: dict[str, Any] = dict(model_effective=model_effective)
    check_report: dict[str, Any] = {"pass": True, "failures": []}

    if model_effective != "self-consistent-easy-axis":
        _write_json(debug_dir / "summary.json", summary)
        _write_json(debug_dir / "check_report.json", check_report)
        return debug_dir

    if sc_cfg_override is not None:
        sc_cfg_raw = dict(sc_cfg_raw)
        sc_cfg_raw.update(sc_cfg_override)
    sc_cfg = _parse_sc_cfg(sc_cfg_raw)
    fp_expected = sc_cfg_fingerprint(sc_cfg_raw)
    summary["sc_cfg_fingerprint"] = fp_expected
    summary["sc_cfg"] = dict(sc_cfg)

    window = sc_cfg["near_window"]
    near = build_near_graph(
        int(geom.R),
        int(geom.K),
        int(geom.N),
        NearWindow(wr=int(window["wr"]), wz=int(window["wz"]), wphi=int(window["wphi"])),
    )
    nbr_counts = np.sum(near.nbr_mask, axis=1).astype(int)
    sample_size = min(int(sample_magnets), int(near.nbr_idx.shape[0]))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(near.nbr_idx.shape[0], size=sample_size, replace=False)
    sample_neighbors: list[dict[str, Any]] = []
    for idx in sample_idx:
        r, k, n = unflatten_index(int(idx), int(geom.R), int(geom.K), int(geom.N))
        neighbors = near.nbr_idx[idx][near.nbr_mask[idx]].tolist()
        sample_neighbors.append(
            dict(idx=int(idx), r=int(r), k=int(k), n=int(n), neighbors=neighbors)
        )

    wrap_idx = flatten_index(0, 0, 0, int(geom.R), int(geom.K), int(geom.N))
    wrap_target = flatten_index(0, 0, int(geom.N) - 1, int(geom.R), int(geom.K), int(geom.N))
    wrap_ok = bool(wrap_target in near.nbr_idx[wrap_idx][near.nbr_mask[wrap_idx]])

    _write_json(
        debug_dir / "near_graph.json",
        dict(
            M=int(near.nbr_idx.shape[0]),
            deg_max=int(near.deg_max),
            window=dict(wr=int(window["wr"]), wz=int(window["wz"]), wphi=int(window["wphi"])),
            neighbor_count_hist=_neighbor_hist(nbr_counts),
            sample_neighbors=sample_neighbors,
            wrap_n0_has_nlast=wrap_ok,
        ),
    )

    phi_flat = np.asarray(phi_rkn, dtype=np.float64).reshape(-1)
    r0_flat = np.asarray(r0_rkn, dtype=np.float64).reshape(-1, 3)
    if str(sc_cfg["near_kernel"]) == "cellavg":
        from halbach.demag_cellavg import demag_N_cellavg

        edge_idx = np.argwhere(near.nbr_mask)
        sample_edges = min(128, int(edge_idx.shape[0]))
        if sample_edges > 0:
            edge_sel = rng.choice(edge_idx.shape[0], size=sample_edges, replace=False)
            edges = edge_idx[edge_sel]
            src_idx = edges[:, 0]
            nbr_pos = edges[:, 1]
            dst_idx = near.nbr_idx[src_idx, nbr_pos]
            s_edges = r0_flat[src_idx] - r0_flat[dst_idx]
            a = float(sc_cfg["volume_mm3"]) * 1e-9
            a = a ** (1.0 / 3.0)
            N_edges = demag_N_cellavg(s_edges, (a, a, a))
            finite = np.isfinite(N_edges)
            trace_vals = np.trace(N_edges, axis1=1, axis2=2)
            payload = dict(
                sample_edges=int(sample_edges),
                nan_count=int(np.size(N_edges) - np.count_nonzero(finite)),
                trace_min=float(np.min(trace_vals)),
                trace_max=float(np.max(trace_vals)),
                trace_mean=float(np.mean(trace_vals)),
                max_abs=float(np.max(np.abs(N_edges))),
            )
        else:
            payload = dict(sample_edges=0, nan_count=0, trace_min=0.0, trace_max=0.0)
        _write_json(debug_dir / "cellavg_tensor_stats.json", payload)
    p_flat = None if force_compute_p else _load_saved_p(run_dir, fp_expected, phi_flat.size)
    trace_rows: list[dict[str, float]] = []
    if p_flat is not None:
        sc_p_source = "saved"
    elif force_compute_p:
        sc_p_source = "computed(forced)"
    else:
        sc_p_source = "computed"
    summary["sc_p_source"] = sc_p_source

    if p_flat is None:
        import importlib.util

        if importlib.util.find_spec("jax") is None:
            raise RuntimeError(
                "self-consistent debug requires JAX unless sc_p_flat is saved in results.npz"
            )
        p_flat, trace_rows = _compute_p_trace_jax(
            phi_flat=phi_flat,
            r0_flat=r0_flat,
            nbr_idx=near.nbr_idx,
            nbr_mask=near.nbr_mask,
            chi=float(sc_cfg["chi"]),
            Nd=float(sc_cfg["Nd"]),
            p0=float(sc_cfg["p0"]),
            volume_m3=float(sc_cfg["volume_mm3"]) * 1e-9,
            iters=int(sc_cfg["iters"]),
            omega=float(sc_cfg["omega"]),
            near_kernel=str(sc_cfg["near_kernel"]),
            subdip_n=int(sc_cfg["subdip_n"]),
        )

    p_min = float(np.min(p_flat))
    p_max = float(np.max(p_flat))
    p_mean = float(np.mean(p_flat))
    p_std = float(np.std(p_flat))
    p_rel_std = p_std / (abs(p_mean) + EPS)
    summary.update(
        dict(
            sc_p_min=p_min,
            sc_p_max=p_max,
            sc_p_mean=p_mean,
            sc_p_std=p_std,
            sc_p_rel_std=p_rel_std,
        )
    )

    if trace_rows:
        _write_solver_trace_csv(debug_dir / "solver_trace.csv", trace_rows)
        summary["sc_last_rel_change"] = float(trace_rows[-1]["rel_change"])
    else:
        _write_solver_trace_csv(debug_dir / "solver_trace.csv", trace_rows)
        summary["sc_last_rel_change"] = None

    if str(sc_cfg["near_kernel"]) == "multi-dipole":
        cube_edge = float(sc_cfg["volume_mm3"]) * 1e-9
        cube_edge = cube_edge ** (1.0 / 3.0)
        offsets = _multi_dipole_offsets(int(sc_cfg["subdip_n"]), cube_edge)
        _write_json(
            debug_dir / "multi_dipole_offsets.json",
            dict(
                cube_edge=float(cube_edge),
                mean=np.mean(offsets, axis=0).tolist(),
                min=np.min(offsets, axis=0).tolist(),
                max=np.max(offsets, axis=0).tolist(),
            ),
        )

    if linear_check:
        linear_payload: dict[str, Any]
        M = int(phi_flat.size)
        if M > int(linear_check_max_M):
            linear_payload = dict(
                skipped=True,
                reason=f"M={M} > linear_check_max_M={linear_check_max_M}",
                M=int(M),
                near_kernel=str(sc_cfg["near_kernel"]),
            )
        else:
            i_edge, j_edge = edges_from_near(near.nbr_idx, near.nbr_mask)
            T2 = build_T2_edges(
                r0_flat,
                i_edge,
                j_edge,
                near_kernel=str(sc_cfg["near_kernel"]),
                volume_m3=float(sc_cfg["volume_mm3"]) * 1e-9,
                subdip_n=int(sc_cfg["subdip_n"]),
            )
            C_edge = build_C_edges_from_phi(phi_flat, i_edge, j_edge, T2)
            C = build_C_sparse(int(M), i_edge, j_edge, C_edge)
            A = build_A_sparse(C, float(sc_cfg["chi"]), float(sc_cfg["Nd"]))
            b = build_b(float(sc_cfg["p0"]), int(M))
            p_ls = solve_p_linear(A, b)
            denom = max(float(np.linalg.norm(p_ls)), 1e-30)
            rel_diff = float(np.linalg.norm(p_flat - p_ls)) / denom
            res_dbg = residual_norm(A, p_flat, b)
            res_ls = residual_norm(A, p_ls, b)
            pass_check = bool(rel_diff < 1e-8 and res_dbg < 1e-8)
            C_abs = C.copy()
            C_abs.data = np.abs(C_abs.data)
            row_sum = np.asarray(C_abs.sum(axis=1)).ravel()
            C_row_abs_sum_max = float(np.max(row_sum)) if row_sum.size > 0 else 0.0
            C_max_abs = float(np.max(np.abs(C_edge))) if C_edge.size > 0 else 0.0
            linear_payload = dict(
                M=int(M),
                E=int(C_edge.size),
                near_kernel=str(sc_cfg["near_kernel"]),
                chi=float(sc_cfg["chi"]),
                Nd=float(sc_cfg["Nd"]),
                p0=float(sc_cfg["p0"]),
                volume_m3=float(sc_cfg["volume_mm3"]) * 1e-9,
                rel_diff=rel_diff,
                res_dbg=res_dbg,
                res_ls=res_ls,
                C_max_abs=C_max_abs,
                C_row_abs_sum_max=C_row_abs_sum_max,
            )
            linear_payload["pass"] = pass_check
            if not pass_check:
                check_report["pass"] = False
                failures_list = check_report.get("failures", [])
                if isinstance(failures_list, list):
                    failures_list.append("linear_check_failed")
                    check_report["failures"] = failures_list
        _write_json(debug_dir / "linear_system_check.json", linear_payload)

    if str(sc_cfg["near_kernel"]) == "multi-dipole":
        cube_edge = (float(sc_cfg["volume_mm3"]) * 1e-9) ** (1.0 / 3.0)
        offsets = _multi_dipole_offsets(int(sc_cfg["subdip_n"]), cube_edge)
        h_ext = _compute_h_ext_u_numpy_multi(
            phi_flat,
            r0_flat,
            p_flat,
            near.nbr_idx,
            near.nbr_mask,
            offsets,
        )
    elif str(sc_cfg["near_kernel"]) == "cellavg":
        h_ext = _compute_h_ext_u_numpy_cellavg(
            phi_flat,
            r0_flat,
            p_flat,
            near.nbr_idx,
            near.nbr_mask,
            float(sc_cfg["volume_mm3"]) * 1e-9,
        )
    else:
        h_ext = _compute_h_ext_u_numpy(
            phi_flat,
            r0_flat,
            p_flat,
            near.nbr_idx,
            near.nbr_mask,
        )

    denom = 1.0 + float(sc_cfg["chi"]) * float(sc_cfg["Nd"])
    res = denom * p_flat - (
        float(sc_cfg["p0"]) + float(sc_cfg["chi"]) * float(sc_cfg["volume_mm3"]) * 1e-9 * h_ext
    )
    res_norm = float(np.linalg.norm(res) / (np.linalg.norm(p_flat) + EPS))
    summary["sc_residual_norm"] = res_norm

    sample_idx = np.asarray(sample_idx, dtype=int)
    sample_rkn = np.array(
        [unflatten_index(int(idx), int(geom.R), int(geom.K), int(geom.N)) for idx in sample_idx],
        dtype=int,
    )
    neighbor_counts = nbr_counts[sample_idx]
    samples_payload = dict(
        idx=sample_idx,
        r=sample_rkn[:, 0],
        k=sample_rkn[:, 1],
        n=sample_rkn[:, 2],
        phi=phi_flat[sample_idx],
        p=p_flat[sample_idx],
        neighbor_count=neighbor_counts,
        h_ext=h_ext[sample_idx],
        residual=res[sample_idx],
    )
    np.savez_compressed(debug_dir / "samples.npz", **samples_payload)

    if field_scale_check and pts is not None and factor is not None:
        field_check = _field_scale_check(
            phi_flat=phi_flat,
            r0_flat=r0_flat,
            p_flat=p_flat,
            pts=np.asarray(pts, dtype=np.float64),
            factor=float(factor),
            scale_factor=float(scale_factor),
        )
        try:
            inv = _objective_p_invariance_check(
                run_dir=run_dir,
                geom=geom,
                pts=np.asarray(pts, dtype=np.float64),
                near_kernel=str(sc_cfg["near_kernel"]),
                subdip_n=int(sc_cfg["subdip_n"]),
                chi=float(sc_cfg["chi"]),
                Nd=float(sc_cfg["Nd"]),
                p0=float(sc_cfg["p0"]),
                volume_m3=float(sc_cfg["volume_mm3"]) * 1e-9,
                iters=int(sc_cfg["iters"]),
                omega=float(sc_cfg["omega"]),
                nbr_idx=near.nbr_idx,
                nbr_mask=near.nbr_mask,
                scale_factor=float(scale_factor),
            )
            field_check["p_invariance"] = inv
        except Exception as exc:
            field_check["p_invariance"] = dict(checked=False, reason=str(exc))
        _write_json(debug_dir / "field_scale_check.json", field_check)
    else:
        _write_json(
            debug_dir / "field_scale_check.json",
            dict(skipped=True, reason="field_scale_check disabled or missing pts/factor"),
        )

    failures: list[str] = []
    if not wrap_ok:
        failures.append("near_wrap_failed")
    if res_norm > 1e-3:
        failures.append("residual_too_large")
    if field_scale_check and pts is not None and factor is not None:
        field_data = json.loads((debug_dir / "field_scale_check.json").read_text(encoding="utf-8"))
        if isinstance(field_data, dict) and not field_data.get("pass_ratio", True):
            failures.append("field_scale_ratio_mismatch")
        inv = field_data.get("p_invariance")
        if isinstance(inv, dict) and inv.get("checked") and not inv.get("pass_p_invariance", True):
            failures.append("field_scale_leak_suspected")
    if failures:
        check_report["pass"] = False
        check_report["failures"] = failures
    _write_json(debug_dir / "summary.json", summary)
    _write_json(debug_dir / "check_report.json", check_report)
    return debug_dir


__all__ = ["make_sc_debug_bundle"]
