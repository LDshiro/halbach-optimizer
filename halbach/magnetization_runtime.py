from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0
from halbach.near import NearWindow, build_near_graph
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
        near_kernel=str(sc_cfg.get("near_kernel", "dipole")),
        subdip_n=int(sc_cfg.get("subdip_n", 2)),
    )


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
        solve_p_easy_axis_near,
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
        else:
            raise ValueError(f"Unsupported near_kernel: {sc['near_kernel']}")

    return np.asarray(p_flat, dtype=np.float64)


def build_m_flat_from_phi_and_p(
    phi_flat: NDArray[np.float64],
    p_flat: NDArray[np.float64],
) -> NDArray[np.float64]:
    u = np.stack([np.cos(phi_flat), np.sin(phi_flat), np.zeros_like(phi_flat)], axis=1)
    return p_flat[:, None] * u


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
    origin = np.zeros((1, 3), dtype=np.float64)
    pts_all = np.concatenate([pts, origin], axis=0)

    r = pts_all[None, :, :] - r0_flat[:, None, :]
    r2 = np.sum(r * r, axis=2)
    rmag = np.sqrt(r2) + EPS
    invr3 = 1.0 / (rmag * r2 + EPS)
    rhat = r / rmag[:, :, None]
    mdotr = np.sum(m_flat[:, None, :] * rhat, axis=2)
    term = (3.0 * mdotr[:, :, None] * rhat - m_flat[:, None, :]) * invr3[:, :, None]
    B_all = float(factor) * np.sum(term, axis=0)
    B = B_all[:-1]
    B0 = B_all[-1]
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
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    meta = _load_meta(run_dir)
    model_effective, sc_cfg = get_magnetization_config_from_meta(meta)

    phi_flat = np.asarray(phi_rkn, dtype=np.float64).reshape(-1)
    if model_effective == "self-consistent-easy-axis":
        p_flat = compute_p_flat_self_consistent_jax(phi_rkn, r0_rkn, geom, sc_cfg)
    else:
        p_flat = np.full(phi_flat.shape, float(m0), dtype=np.float64)

    m_flat = build_m_flat_from_phi_and_p(phi_flat, p_flat)

    debug: dict[str, Any] = {"model_effective": model_effective}
    if model_effective == "self-consistent-easy-axis":
        p_min = float(np.min(p_flat))
        p_max = float(np.max(p_flat))
        p_mean = float(np.mean(p_flat))
        p_std = float(np.std(p_flat))
        p_rel_std = p_std / (abs(p_mean) + EPS)
        sc = _parse_sc_cfg(sc_cfg)
        debug.update(
            dict(
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
    return m_flat, debug


__all__ = [
    "get_magnetization_config_from_meta",
    "compute_p_flat_self_consistent_jax",
    "build_m_flat_from_phi_and_p",
    "compute_b_and_b0_from_m_flat",
    "compute_m_flat_from_run",
]
