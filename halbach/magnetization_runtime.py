from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0
from halbach.near import NearWindow, build_near_graph, edges_from_near
from halbach.physics import compute_B_and_B0_from_m_flat
from halbach.types import Geometry

EPS = 1e-30


def get_magnetization_config_from_meta(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    magnetization = meta.get("magnetization", {})
    if not isinstance(magnetization, dict):
        return "fixed", {}
    model_effective = magnetization.get("model_effective", "fixed")
    if not isinstance(model_effective, str):
        return "fixed", {}
    if model_effective != "self-consistent-easy-axis":
        return model_effective, {}
    sc_cfg_raw = magnetization.get("self_consistent", {})
    if not isinstance(sc_cfg_raw, dict):
        raise ValueError("meta['magnetization']['self_consistent'] must be a dict")
    return model_effective, sc_cfg_raw


def _require_jax() -> None:
    if importlib.util.find_spec("jax") is None:
        raise RuntimeError(
            "JAX is required for self-consistent visualization. "
            "Install `jax` and `jaxlib` to render self-consistent runs."
        )


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"meta.json in {run_dir} must be a JSON object")
    return cast(dict[str, Any], data)


def _parse_sc_cfg(sc_cfg: dict[str, Any]) -> dict[str, Any]:
    near_window = sc_cfg.get("near_window", {})
    if not isinstance(near_window, dict):
        raise ValueError("self_consistent.near_window must be a dict")
    near_kernel = str(sc_cfg.get("near_kernel", "dipole"))
    if near_kernel == "cube-average":
        near_kernel = "cellavg"
    gl_order_raw = sc_cfg.get("gl_order")
    gl_order: int | None = None
    if gl_order_raw is not None:
        gl_order = int(gl_order_raw)
        if gl_order not in (2, 3):
            raise ValueError("self_consistent.gl_order must be 2 or 3 when provided")
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
        gl_order=gl_order,
    )


def _edge_partition_face_to_face(
    i_edge: NDArray[np.int32],
    j_edge: NDArray[np.int32],
    *,
    R: int,
    K: int,
    N: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    n_i = (i_edge % N).astype(np.int32)
    n_j = (j_edge % N).astype(np.int32)
    dn_raw = (n_i - n_j) % N
    dn = np.where(dn_raw > (N // 2), dn_raw - N, dn_raw).astype(np.int32)

    k_i = ((i_edge // N) % K).astype(np.int32)
    k_j = ((j_edge // N) % K).astype(np.int32)
    dk = (k_i - k_j).astype(np.int32)

    r_i = (i_edge // (K * N)).astype(np.int32)
    r_j = (j_edge // (K * N)).astype(np.int32)

    hi_mask = (r_i == r_j) & (np.abs(dk) == 1) & (dn == 0)
    idx_hi = np.nonzero(hi_mask)[0].astype(np.int32)
    idx_lo = np.nonzero(~hi_mask)[0].astype(np.int32)
    return idx_lo, idx_hi


def sc_cfg_fingerprint(sc_cfg: dict[str, Any]) -> str:
    sc = _parse_sc_cfg(sc_cfg)
    core = dict(
        chi=float(sc["chi"]),
        Nd=float(sc["Nd"]),
        p0=float(sc["p0"]),
        volume_mm3=float(sc["volume_mm3"]),
        iters=int(sc["iters"]),
        omega=float(sc["omega"]),
        near_window=dict(sc["near_window"]),
        near_kernel=str(sc["near_kernel"]),
        subdip_n=int(sc["subdip_n"]),
    )
    if sc["near_kernel"] == "gl-double-mixed" and sc.get("gl_order") in (2, 3):
        core["gl_order"] = int(sc["gl_order"])
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_p_flat_self_consistent_jax(
    phi_rkn: NDArray[np.float64],
    r0_rkn: NDArray[np.float64],
    geom: Geometry,
    sc_cfg: dict[str, Any],
) -> NDArray[np.float64]:
    """
    Compute p_flat via self-consistent solver (JAX required).
    """
    _require_jax()
    import jax.numpy as jnp

    from halbach.autodiff.jax_self_consistent import (
        _gl_double_delta_table_n2,
        _gl_double_delta_table_n3,
        solve_p_easy_axis_near,
        solve_p_easy_axis_near_cellavg,
        solve_p_easy_axis_near_gl_double_mixed,
        solve_p_easy_axis_near_multi_dipole,
    )

    sc = _parse_sc_cfg(sc_cfg)
    volume_m3 = float(sc["volume_mm3"]) * 1e-9
    window = sc["near_window"]
    near = build_near_graph(
        int(geom.R),
        int(geom.K),
        int(geom.N),
        NearWindow(wr=int(window["wr"]), wz=int(window["wz"]), wphi=int(window["wphi"])),
    )

    phi_flat = np.asarray(phi_rkn, dtype=np.float64).reshape(-1)
    r0_flat = np.asarray(r0_rkn, dtype=np.float64).reshape(-1, 3)

    phi_j = jnp.asarray(phi_flat, dtype=jnp.float64)
    r0_j = jnp.asarray(r0_flat, dtype=jnp.float64)
    nbr_idx_j = jnp.asarray(near.nbr_idx, dtype=jnp.int32)
    nbr_mask_j = jnp.asarray(near.nbr_mask, dtype=bool)

    if float(sc["chi"]) == 0.0:
        p_flat = jnp.full((phi_j.shape[0],), float(sc["p0"]), dtype=jnp.float64)
    else:
        if sc["near_kernel"] == "dipole":
            p_flat = solve_p_easy_axis_near(
                phi_j,
                r0_j,
                nbr_idx_j,
                nbr_mask_j,
                p0=float(sc["p0"]),
                chi=float(sc["chi"]),
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
            )
        elif sc["near_kernel"] == "multi-dipole":
            p_flat = solve_p_easy_axis_near_multi_dipole(
                phi_j,
                r0_j,
                nbr_idx_j,
                nbr_mask_j,
                p0=float(sc["p0"]),
                chi=float(sc["chi"]),
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                subdip_n=int(sc["subdip_n"]),
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
            )
        elif sc["near_kernel"] == "cellavg":
            p_flat = solve_p_easy_axis_near_cellavg(
                phi_j,
                r0_j,
                nbr_idx_j,
                nbr_mask_j,
                p0=float(sc["p0"]),
                chi=float(sc["chi"]),
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
            )
        elif sc["near_kernel"] == "gl-double-mixed":
            i_edge_np, j_edge_np = edges_from_near(near.nbr_idx, near.nbr_mask)
            idx_lo_np, idx_hi_np = _edge_partition_face_to_face(
                i_edge_np,
                j_edge_np,
                R=int(geom.R),
                K=int(geom.K),
                N=int(geom.N),
            )
            i_edge_j = jnp.asarray(i_edge_np, dtype=jnp.int32)
            j_edge_j = jnp.asarray(j_edge_np, dtype=jnp.int32)
            idx_lo_j = jnp.asarray(idx_lo_np, dtype=jnp.int32)
            idx_hi_j = jnp.asarray(idx_hi_np, dtype=jnp.int32)
            cube_edge = float(volume_m3) ** (1.0 / 3.0)
            gl_order = sc.get("gl_order")
            if gl_order == 2:
                delta_lo_offsets, delta_lo_w = _gl_double_delta_table_n2(cube_edge)
                delta_hi_offsets, delta_hi_w = _gl_double_delta_table_n2(cube_edge)
            elif gl_order == 3:
                delta_lo_offsets, delta_lo_w = _gl_double_delta_table_n3(cube_edge)
                delta_hi_offsets, delta_hi_w = _gl_double_delta_table_n3(cube_edge)
            else:
                delta_lo_offsets, delta_lo_w = _gl_double_delta_table_n2(cube_edge)
                delta_hi_offsets, delta_hi_w = _gl_double_delta_table_n3(cube_edge)
            p_flat = solve_p_easy_axis_near_gl_double_mixed(
                phi_j,
                r0_j,
                p0=float(sc["p0"]),
                chi=float(sc["chi"]),
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
                i_edge=i_edge_j,
                j_edge=j_edge_j,
                idx_lo=idx_lo_j,
                idx_hi=idx_hi_j,
                delta_lo_offsets=delta_lo_offsets,
                delta_lo_w=delta_lo_w,
                delta_hi_offsets=delta_hi_offsets,
                delta_hi_w=delta_hi_w,
                implicit_diff=False,
            )
        else:
            raise ValueError(f"Unsupported near_kernel: {sc['near_kernel']}")

    return np.asarray(p_flat, dtype=np.float64)


def build_m_flat_from_phi_and_p(
    phi_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
) -> NDArray[np.float64]:
    u = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    return np.asarray(p_flat[:, None] * u, dtype=np.float64)


def compute_b_and_b0_from_m_flat(
    m_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    pts: NDArray[np.float64],
    *,
    factor: float = FACTOR,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
]:
    origin = np.zeros(3, dtype=np.float64)
    pts_f = np.asarray(pts, dtype=np.float64)
    r0_f = np.asarray(r0_flat, dtype=np.float64)
    m_f = np.asarray(m_flat, dtype=np.float64)
    B, B0 = compute_B_and_B0_from_m_flat(pts_f, r0_f, m_f, float(factor), origin)
    return (
        np.asarray(B[:, 0], dtype=np.float64),
        np.asarray(B[:, 1], dtype=np.float64),
        np.asarray(B[:, 2], dtype=np.float64),
        float(B0[0]),
        float(B0[1]),
        float(B0[2]),
    )


def compute_m_flat_from_run(
    run_dir: Path,
    geom: Geometry,
    phi_rkn: NDArray[np.float64],
    r0_rkn: NDArray[np.float64],
    *,
    sc_cfg_override: dict[str, Any] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    meta = _load_meta(run_dir)
    model_effective_meta, sc_cfg_meta = get_magnetization_config_from_meta(meta)
    if sc_cfg_override is not None:
        if not isinstance(sc_cfg_override, dict):
            raise TypeError("sc_cfg_override must be a dict when provided")
        model_effective = "self-consistent-easy-axis"
        sc_cfg = sc_cfg_override
    else:
        model_effective = model_effective_meta
        sc_cfg = sc_cfg_meta

    phi_flat = np.asarray(phi_rkn, dtype=np.float64).reshape(-1)
    if model_effective == "self-consistent-easy-axis":
        fp_expected = sc_cfg_fingerprint(sc_cfg)
        p_flat = None
        sc_source = "computed"
        results_path = run_dir / "results.npz"
        if results_path.is_file():
            try:
                with np.load(results_path, allow_pickle=False) as data:
                    if "sc_p_flat" in data and "sc_cfg_fingerprint" in data:
                        p_candidate = np.asarray(data["sc_p_flat"], dtype=np.float64)
                        fp_raw = data["sc_cfg_fingerprint"]
                        if isinstance(fp_raw, np.ndarray):
                            if fp_raw.shape == ():
                                fp_saved = str(fp_raw.item())
                            else:
                                fp_saved = str(fp_raw.reshape(-1)[0])
                        else:
                            fp_saved = str(fp_raw)
                        if fp_saved == fp_expected and p_candidate.shape == (phi_flat.size,):
                            p_flat = p_candidate
                            sc_source = "saved"
            except Exception:
                p_flat = None
        if p_flat is None:
            if importlib.util.find_spec("jax") is None:
                raise RuntimeError(
                    "self-consistent visualization requires JAX unless sc_p_flat is saved in results.npz"
                )
            p_flat = compute_p_flat_self_consistent_jax(phi_rkn, r0_rkn, geom, sc_cfg)
    else:
        p_flat = np.full(phi_flat.shape, float(m0), dtype=np.float64)

    assert p_flat is not None
    m_flat = build_m_flat_from_phi_and_p(phi_flat, p_flat)

    debug: dict[str, Any] = {"model_effective": model_effective}
    if sc_cfg_override is not None:
        debug["model_effective_meta"] = model_effective_meta
        debug["model_effective_eval"] = model_effective
    if model_effective == "self-consistent-easy-axis":
        p_min = float(np.min(p_flat))
        p_max = float(np.max(p_flat))
        p_mean = float(np.mean(p_flat))
        p_std = float(np.std(p_flat))
        p_rel_std = p_std / (abs(p_mean) + EPS)
        sc = _parse_sc_cfg(sc_cfg)
        debug.update(
            dict(
                sc_p_source=sc_source,
                sc_cfg_fingerprint=fp_expected,
                sc_p_min=p_min,
                sc_p_max=p_max,
                sc_p_mean=p_mean,
                sc_p_std=p_std,
                sc_p_rel_std=p_rel_std,
                sc_near_kernel=str(sc["near_kernel"]),
                sc_subdip_n=int(sc["subdip_n"]),
                sc_near_window=dict(sc["near_window"]),
                sc_iters=int(sc["iters"]),
                sc_omega=float(sc["omega"]),
                sc_chi=float(sc["chi"]),
                sc_Nd=float(sc["Nd"]),
                sc_p0=float(sc["p0"]),
                sc_volume_mm3=float(sc["volume_mm3"]),
            )
        )
        if sc["near_kernel"] == "gl-double-mixed":
            debug["sc_gl_order"] = sc.get("gl_order")
    return m_flat, debug


__all__ = [
    "get_magnetization_config_from_meta",
    "sc_cfg_fingerprint",
    "compute_p_flat_self_consistent_jax",
    "build_m_flat_from_phi_and_p",
    "compute_b_and_b0_from_m_flat",
    "compute_m_flat_from_run",
]
