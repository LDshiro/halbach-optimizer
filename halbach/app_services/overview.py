from __future__ import annotations

from typing import Any

import numpy as np

from halbach.angles_runtime import phi_rkn_from_run
from halbach.constants import FACTOR, m0, phi0
from halbach.magnetization_runtime import (
    compute_b_and_b0_from_m_flat,
    compute_m_flat_from_run,
    get_magnetization_config_from_meta,
)
from halbach.physics import compute_B_and_B0, compute_B_and_B0_phi_rkn
from halbach.run_types import RunBundle

from .runs import path_to_display
from .types import RunSummary


def _build_r0_rkn(run: RunBundle) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
    rho = r_bases[None, :] + np.asarray(run.geometry.ring_offsets, dtype=np.float64)[:, None]
    px = rho[:, :, None] * np.asarray(run.geometry.cth, dtype=np.float64)[None, None, :]
    py = rho[:, :, None] * np.asarray(run.geometry.sth, dtype=np.float64)[None, None, :]
    pz = np.broadcast_to(
        np.asarray(run.geometry.z_layers, dtype=np.float64)[None, :, None], px.shape
    )
    return np.stack([px, py, pz], axis=-1)


def compute_center_b0_t(run: RunBundle) -> float:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    model_effective, _sc_cfg = get_magnetization_config_from_meta(run.meta)
    if model_effective == "self-consistent-easy-axis":
        phi_rkn = phi_rkn_from_run(run, phi0=phi0)
        r0_rkn = _build_r0_rkn(run)
        m_flat, _debug = compute_m_flat_from_run(run.run_dir, run.geometry, phi_rkn, r0_rkn)
        r0_flat = r0_rkn.reshape(-1, 3)
        _, _, _, B0x, B0y, B0z = compute_b_and_b0_from_m_flat(m_flat, r0_flat, pts, factor=FACTOR)
    elif run.meta.get("angle_model", "legacy-alpha") == "legacy-alpha":
        _, _, _, B0x, B0y, B0z = compute_B_and_B0(
            run.results.alphas,
            run.results.r_bases,
            run.geometry.theta,
            run.geometry.sin2,
            run.geometry.cth,
            run.geometry.sth,
            run.geometry.z_layers,
            run.geometry.ring_offsets,
            pts,
            FACTOR,
            phi0,
            m0,
        )
    else:
        phi_rkn = phi_rkn_from_run(run, phi0=phi0)
        _, _, _, B0x, B0y, B0z = compute_B_and_B0_phi_rkn(
            phi_rkn,
            run.results.r_bases,
            run.geometry.cth,
            run.geometry.sth,
            run.geometry.z_layers,
            run.geometry.ring_offsets,
            pts,
            FACTOR,
            m0,
        )
    return float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))


def build_run_summary(run: RunBundle) -> RunSummary:
    model_effective_meta, sc_cfg_meta = get_magnetization_config_from_meta(run.meta)
    geometry_summary: dict[str, object] = {
        "N": int(run.geometry.N),
        "K": int(run.geometry.K),
        "R": int(run.geometry.R),
        "dz": float(run.geometry.dz),
        "Lz": float(run.geometry.Lz),
        "radial_profile_mode": (
            str(run.meta.get("radial_profile", {}).get("mode", "uniform"))
            if isinstance(run.meta.get("radial_profile"), dict)
            else "uniform"
        ),
    }
    key_stats: dict[str, object] = {}
    if run.meta.get("framework") == "dc":
        extras = run.results.extras
        if "p_opt" in extras:
            p_opt = np.asarray(extras["p_opt"], dtype=np.float64).reshape(-1)
            key_stats["p_opt_min"] = float(np.min(p_opt))
            key_stats["p_opt_max"] = float(np.max(p_opt))
            key_stats["p_opt_mean"] = float(np.mean(p_opt))
        if "p_sc_post" in extras:
            p_sc_post = np.asarray(extras["p_sc_post"], dtype=np.float64).reshape(-1)
            key_stats["p_sc_post_min"] = float(np.min(p_sc_post))
            key_stats["p_sc_post_max"] = float(np.max(p_sc_post))
            key_stats["p_sc_post_mean"] = float(np.mean(p_sc_post))
        if "z_norm" in extras:
            z_norm = np.asarray(extras["z_norm"], dtype=np.float64).reshape(-1)
            key_stats["z_norm_min"] = float(np.min(z_norm))
            key_stats["z_norm_max"] = float(np.max(z_norm))
            key_stats["z_norm_mean"] = float(np.mean(z_norm))
    else:
        key_stats["B0_T"] = compute_center_b0_t(run)
        key_stats["r_bases_min"] = float(np.min(run.results.r_bases))
        key_stats["r_bases_max"] = float(np.max(run.results.r_bases))
        key_stats["alphas_min"] = float(np.min(run.results.alphas))
        key_stats["alphas_max"] = float(np.max(run.results.alphas))
        key_stats["alphas_shape"] = [int(value) for value in run.results.alphas.shape]

    magnetization_debug: dict[str, object] = {
        "model_effective_meta": model_effective_meta,
        "self_consistent": cast_jsonable_dict(sc_cfg_meta),
    }
    return RunSummary(
        name=run.name,
        run_path=path_to_display(run.run_dir),
        results_path=str(run.results_path),
        meta_path=None if run.meta_path is None else str(run.meta_path),
        trace_path=None if run.trace_path is None else str(run.trace_path),
        framework=str(run.meta.get("framework", "legacy")),
        geometry_summary=geometry_summary,
        key_stats=key_stats,
        magnetization_debug=magnetization_debug,
    )


def build_run_delta_summary(initial: RunBundle, optimized: RunBundle) -> dict[str, object]:
    if initial.meta.get("framework") == "dc" or optimized.meta.get("framework") == "dc":
        return {"available": False, "reason": "dc-framework"}
    if initial.results.alphas.shape != optimized.results.alphas.shape:
        return {"available": False, "reason": "shape-mismatch"}

    dalphas = optimized.results.alphas - initial.results.alphas
    dr_bases = optimized.results.r_bases - initial.results.r_bases
    return {
        "available": True,
        "alphas_delta_min": float(np.min(dalphas)),
        "alphas_delta_max": float(np.max(dalphas)),
        "r_bases_delta_min": float(np.min(dr_bases)),
        "r_bases_delta_max": float(np.max(dr_bases)),
    }


def cast_jsonable_dict(raw: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            result[str(key)] = cast_jsonable_dict(value)
        elif isinstance(value, list):
            result[str(key)] = [cast_jsonable_item(item) for item in value]
        else:
            result[str(key)] = cast_jsonable_item(value)
    return result


def cast_jsonable_item(value: Any) -> object:
    if isinstance(value, bool | str | int | float):
        return value
    if value is None:
        return None
    return str(value)


__all__ = ["build_run_delta_summary", "build_run_summary", "compute_center_b0_t"]
