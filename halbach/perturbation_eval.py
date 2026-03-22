from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import phi_rkn_from_run
from halbach.constants import FACTOR
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.magnetization_runtime import build_m_flat_from_phi_and_p
from halbach.near import NearWindow, build_near_graph, edges_from_near
from halbach.physics import compute_B_all_from_m_flat, compute_B_and_B0_from_m_flat
from halbach.run_types import RunBundle
from halbach.viz2d import ErrorMap2D

EPS = 1e-30
P0_FLOOR = 1e-12


@dataclass(frozen=True)
class PerturbationConfig:
    sigma_rel_pct: float
    sigma_phi_deg: float
    seed: int
    roi_radius_m: float
    roi_samples: int
    map_radius_m: float
    map_step_m: float
    sc_cfg: dict[str, Any]
    target_plane_z: float = 0.0


@dataclass(frozen=True)
class PerturbationResult:
    pts_roi: NDArray[np.float64]
    B_roi: NDArray[np.float64]
    B0_vec: NDArray[np.float64]
    Bnorm_roi: NDArray[np.float64]
    ppm_roi: NDArray[np.float64]
    map_xy: ErrorMap2D
    p_stats: dict[str, float]
    debug: dict[str, Any]


def _require_jax() -> None:
    if importlib.util.find_spec("jax") is None:
        raise RuntimeError(
            "JAX is required for self-consistent perturbation evaluation. "
            "Install `jax` and `jaxlib` (pip install jax jaxlib)."
        )


def _validate_cfg(cfg: PerturbationConfig) -> None:
    if cfg.roi_samples < 1:
        raise ValueError("roi_samples must be >= 1")
    if cfg.roi_radius_m <= 0.0:
        raise ValueError("roi_radius_m must be > 0")
    if cfg.map_radius_m <= 0.0:
        raise ValueError("map_radius_m must be > 0")
    if cfg.map_step_m <= 0.0:
        raise ValueError("map_step_m must be > 0")
    if cfg.sigma_rel_pct < 0.0:
        raise ValueError("sigma_rel_pct must be >= 0")
    if cfg.sigma_phi_deg < 0.0:
        raise ValueError("sigma_phi_deg must be >= 0")


def _normalize_sc_cfg(sc_cfg: dict[str, Any]) -> dict[str, Any]:
    near_window_raw = sc_cfg.get("near_window", {})
    if not isinstance(near_window_raw, dict):
        raise ValueError("sc_cfg.near_window must be a dict")

    near_kernel = str(sc_cfg.get("near_kernel", "dipole"))
    if near_kernel == "cube-average":
        near_kernel = "cellavg"

    wr = int(near_window_raw.get("wr", 0))
    wz = int(near_window_raw.get("wz", 1))
    wphi = int(near_window_raw.get("wphi", 2))
    if wr < 0 or wz < 0 or wphi < 0:
        raise ValueError("near_window values must be >= 0")

    chi = float(sc_cfg.get("chi", 0.0))
    Nd = float(sc_cfg.get("Nd", 1.0 / 3.0))
    p0 = float(sc_cfg.get("p0", 1.0))
    volume_mm3 = float(sc_cfg.get("volume_mm3", 1000.0))
    iters = int(sc_cfg.get("iters", 30))
    omega = float(sc_cfg.get("omega", 0.6))
    subdip_n = int(sc_cfg.get("subdip_n", 2))

    if not 0.0 <= Nd <= 1.0:
        raise ValueError("Nd must be in [0, 1]")
    if volume_mm3 <= 0.0:
        raise ValueError("volume_mm3 must be > 0")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")
    if subdip_n < 1:
        raise ValueError("subdip_n must be >= 1")

    gl_order_raw = sc_cfg.get("gl_order")
    gl_order: int | None = None
    if gl_order_raw is not None:
        gl_order = int(gl_order_raw)
        if gl_order not in (2, 3):
            raise ValueError("gl_order must be 2 or 3 when provided")

    kernel_choices = {"dipole", "multi-dipole", "cellavg", "gl-double-mixed"}
    if near_kernel not in kernel_choices:
        raise ValueError(f"Unsupported near_kernel: {near_kernel}")
    if near_kernel == "multi-dipole" and subdip_n < 2:
        raise ValueError("subdip_n must be >= 2 for multi-dipole")

    return {
        "chi": chi,
        "Nd": Nd,
        "p0": p0,
        "volume_mm3": volume_mm3,
        "iters": iters,
        "omega": omega,
        "near_window": {"wr": wr, "wz": wz, "wphi": wphi},
        "near_kernel": near_kernel,
        "subdip_n": subdip_n,
        "gl_order": gl_order,
    }


def _build_r0_rkn(run: RunBundle) -> NDArray[np.float64]:
    geom = run.geometry
    r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
    rho = r_bases[None, :] + np.asarray(geom.ring_offsets, dtype=np.float64)[:, None]
    px = rho[:, :, None] * np.asarray(geom.cth, dtype=np.float64)[None, None, :]
    py = rho[:, :, None] * np.asarray(geom.sth, dtype=np.float64)[None, None, :]
    pz = np.broadcast_to(np.asarray(geom.z_layers, dtype=np.float64)[None, :, None], px.shape)
    return np.stack([px, py, pz], axis=-1)


def _wrap_angle_rad(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray((phi + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def _axis_grid(roi_r: float, step: float) -> NDArray[np.float64]:
    n = int(2 * np.ceil(roi_r / step) + 1)
    return np.linspace(-roi_r, roi_r, n, dtype=np.float64)


def _build_xy_plane_points(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    z0: float,
    roi_r: float,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    mask = (X * X + Y * Y + z0 * z0) <= roi_r * roi_r
    pts = np.column_stack([X[mask], Y[mask], np.full(int(mask.sum()), z0)])
    return mask, np.asarray(pts, dtype=np.float64)


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


def run_perturbation_case(run: RunBundle, cfg: PerturbationConfig) -> PerturbationResult:
    _validate_cfg(cfg)
    _require_jax()

    import jax.numpy as jnp

    from halbach.autodiff.jax_self_consistent import (
        _gl_double_delta_table_n2,
        _gl_double_delta_table_n3,
        solve_p_easy_axis_near_gl_double_mixed,
        solve_p_easy_axis_near_multi_dipole_with_p0_flat,
        solve_p_easy_axis_near_with_p0_flat,
    )

    sc = _normalize_sc_cfg(cfg.sc_cfg)
    geom = run.geometry

    phi_rkn = phi_rkn_from_run(run)
    r0_rkn = _build_r0_rkn(run)
    phi_flat = np.asarray(phi_rkn, dtype=np.float64).reshape(-1)
    r0_flat = np.asarray(r0_rkn, dtype=np.float64).reshape(-1, 3)

    rng = np.random.default_rng(int(cfg.seed))
    sigma_rel = float(cfg.sigma_rel_pct) / 100.0
    eps_p = rng.normal(loc=0.0, scale=sigma_rel, size=phi_flat.shape[0])
    dphi_deg = rng.normal(loc=0.0, scale=float(cfg.sigma_phi_deg), size=phi_flat.shape[0])

    p0_nominal = float(sc["p0"])
    p0_flat = np.maximum(P0_FLOOR, p0_nominal * (1.0 + eps_p))
    phi_noisy = _wrap_angle_rad(phi_flat + np.deg2rad(dphi_deg))

    near_window = sc["near_window"]
    near = build_near_graph(
        int(geom.R),
        int(geom.K),
        int(geom.N),
        NearWindow(
            wr=int(near_window["wr"]),
            wz=int(near_window["wz"]),
            wphi=int(near_window["wphi"]),
        ),
    )

    chi = float(sc["chi"])
    if chi == 0.0:
        p_flat = p0_flat
    else:
        phi_j = jnp.asarray(phi_noisy, dtype=jnp.float64)
        r0_j = jnp.asarray(r0_flat, dtype=jnp.float64)
        p0_j = jnp.asarray(p0_flat, dtype=jnp.float64)
        nbr_idx_j = jnp.asarray(near.nbr_idx, dtype=jnp.int32)
        nbr_mask_j = jnp.asarray(near.nbr_mask, dtype=bool)
        volume_m3 = float(sc["volume_mm3"]) * 1e-9

        kernel = str(sc["near_kernel"])
        if kernel == "dipole":
            p_out = solve_p_easy_axis_near_with_p0_flat(
                phi_j,
                r0_j,
                nbr_idx_j,
                nbr_mask_j,
                p0_flat=p0_j,
                chi=chi,
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
                implicit_diff=False,
            )
        elif kernel == "multi-dipole":
            p_out = solve_p_easy_axis_near_multi_dipole_with_p0_flat(
                phi_j,
                r0_j,
                nbr_idx_j,
                nbr_mask_j,
                p0_flat=p0_j,
                chi=chi,
                Nd=float(sc["Nd"]),
                volume_m3=volume_m3,
                subdip_n=int(sc["subdip_n"]),
                iters=int(sc["iters"]),
                omega=float(sc["omega"]),
                implicit_diff=False,
            )
        elif kernel == "gl-double-mixed":
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
            p_out = solve_p_easy_axis_near_gl_double_mixed(
                phi_j,
                r0_j,
                p0=p0_j,
                chi=chi,
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
        elif kernel == "cellavg":
            raise ValueError(
                "near_kernel='cellavg' is not supported for per-magnet p0 variation in this prototype"
            )
        else:
            raise ValueError(f"Unsupported near_kernel: {kernel}")
        p_flat = np.asarray(p_out, dtype=np.float64)

    m_flat = build_m_flat_from_phi_and_p(phi_noisy, p_flat)

    pts_roi = sample_sphere_surface_fibonacci(
        int(cfg.roi_samples),
        float(cfg.roi_radius_m),
        seed=int(cfg.seed),
    )
    origin = np.zeros(3, dtype=np.float64)
    B_roi, B0 = compute_B_and_B0_from_m_flat(
        np.asarray(pts_roi, dtype=np.float64),
        np.asarray(r0_flat, dtype=np.float64),
        np.asarray(m_flat, dtype=np.float64),
        float(FACTOR),
        origin,
    )
    B_roi = np.asarray(B_roi, dtype=np.float64)
    B0_vec = np.asarray(B0, dtype=np.float64)
    Bnorm_roi = np.linalg.norm(B_roi, axis=1)
    B0_norm = float(np.linalg.norm(B0_vec))
    if B0_norm < 1e-15:
        raise ValueError("|B0| is too small for stable ppm normalization")
    ppm_roi = (Bnorm_roi - B0_norm) / B0_norm * 1e6

    xs = _axis_grid(float(cfg.map_radius_m), float(cfg.map_step_m))
    ys = _axis_grid(float(cfg.map_radius_m), float(cfg.map_step_m))
    mask_xy, pts_xy = _build_xy_plane_points(
        xs,
        ys,
        float(cfg.target_plane_z),
        float(cfg.map_radius_m),
    )
    B_xy = compute_B_all_from_m_flat(
        np.asarray(pts_xy, dtype=np.float64),
        np.asarray(r0_flat, dtype=np.float64),
        np.asarray(m_flat, dtype=np.float64),
        float(FACTOR),
    )
    Bnorm_xy = np.linalg.norm(np.asarray(B_xy, dtype=np.float64), axis=1)
    ppm_xy_vals = (Bnorm_xy - B0_norm) / B0_norm * 1e6
    ppm_xy = np.full(mask_xy.shape, np.nan, dtype=np.float64)
    ppm_xy[mask_xy] = ppm_xy_vals

    map_xy = ErrorMap2D(
        xs=np.asarray(xs, dtype=np.float64),
        ys=np.asarray(ys, dtype=np.float64),
        ppm=ppm_xy,
        mask=mask_xy,
        B0_T=B0_norm,
        plane="xy",
        coord0=float(cfg.target_plane_z),
    )

    p_mean = float(np.mean(p_flat))
    p_stats = {
        "sc_p_min": float(np.min(p_flat)),
        "sc_p_max": float(np.max(p_flat)),
        "sc_p_mean": p_mean,
        "sc_p_std": float(np.std(p_flat)),
        "sc_p_rel_std": float(np.std(p_flat) / (abs(p_mean) + EPS)),
    }

    debug: dict[str, Any] = {
        "model_effective": "self-consistent-easy-axis",
        "sc_near_kernel": str(sc["near_kernel"]),
        "sc_subdip_n": int(sc["subdip_n"]),
        "sc_near_window": dict(sc["near_window"]),
        "sc_near_deg_max": int(near.deg_max),
        "sc_iters": int(sc["iters"]),
        "sc_omega": float(sc["omega"]),
        "sc_chi": float(sc["chi"]),
        "sc_Nd": float(sc["Nd"]),
        "sc_p0": float(sc["p0"]),
        "sc_volume_mm3": float(sc["volume_mm3"]),
    }
    if sc.get("gl_order") is not None:
        debug["sc_gl_order"] = int(sc["gl_order"])

    return PerturbationResult(
        pts_roi=np.asarray(pts_roi, dtype=np.float64),
        B_roi=np.asarray(B_roi, dtype=np.float64),
        B0_vec=B0_vec,
        Bnorm_roi=np.asarray(Bnorm_roi, dtype=np.float64),
        ppm_roi=np.asarray(ppm_roi, dtype=np.float64),
        map_xy=map_xy,
        p_stats=p_stats,
        debug=debug,
    )


def _draw_map_png(result: PerturbationResult, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    m = result.map_xy
    data = np.where(np.isfinite(m.ppm), m.ppm, np.nan)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=180)
    im = ax.imshow(
        data,
        origin="lower",
        extent=(float(m.xs[0]), float(m.xs[-1]), float(m.ys[0]), float(m.ys[-1])),
        cmap="RdBu_r",
        aspect="equal",
    )
    finite_vals = m.ppm[m.mask]
    if finite_vals.size > 1:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmax > vmin:
            levels = np.linspace(vmin, vmax, 50, dtype=np.float64)
            X, Y = np.meshgrid(m.xs, m.ys, indexing="xy")
            ax.contour(X, Y, m.ppm, levels=levels, colors="black", linewidths=0.4)
    ax.set_title("XY(z=0) ppm error map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="ppm")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_perturbation_result(
    result: PerturbationResult,
    out_dir: Path,
    cfg: PerturbationConfig,
    run: RunBundle,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_npz_path = out_dir / "roi_vectors.npz"
    np.savez_compressed(
        roi_npz_path,
        pts_xyz_m=result.pts_roi,
        B_xyz_T=result.B_roi,
        Bnorm_T=result.Bnorm_roi,
        B0_xyz_T=result.B0_vec,
        ppm=result.ppm_roi,
        seed=np.array(int(cfg.seed), dtype=np.int64),
        sigma_rel_pct=np.array(float(cfg.sigma_rel_pct), dtype=np.float64),
        sigma_phi_deg=np.array(float(cfg.sigma_phi_deg), dtype=np.float64),
        roi_radius_m=np.array(float(cfg.roi_radius_m), dtype=np.float64),
        roi_samples=np.array(int(cfg.roi_samples), dtype=np.int64),
    )

    roi_csv_path = out_dir / "roi_vectors.csv"
    with roi_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "z_m", "Bx_T", "By_T", "Bz_T", "Bnorm_T", "ppm"])
        for idx in range(result.pts_roi.shape[0]):
            writer.writerow(
                [
                    float(result.pts_roi[idx, 0]),
                    float(result.pts_roi[idx, 1]),
                    float(result.pts_roi[idx, 2]),
                    float(result.B_roi[idx, 0]),
                    float(result.B_roi[idx, 1]),
                    float(result.B_roi[idx, 2]),
                    float(result.Bnorm_roi[idx]),
                    float(result.ppm_roi[idx]),
                ]
            )

    map_npz_path = out_dir / "map_xy_z0.npz"
    np.savez_compressed(
        map_npz_path,
        xs_m=np.asarray(result.map_xy.xs, dtype=np.float64),
        ys_m=np.asarray(result.map_xy.ys, dtype=np.float64),
        ppm=np.asarray(result.map_xy.ppm, dtype=np.float64),
        mask=np.asarray(result.map_xy.mask, dtype=bool),
        B0_T=np.array(float(result.map_xy.B0_T), dtype=np.float64),
        map_radius_m=np.array(float(cfg.map_radius_m), dtype=np.float64),
        map_step_m=np.array(float(cfg.map_step_m), dtype=np.float64),
    )

    map_png_path = out_dir / "map_xy_z0.png"
    _draw_map_png(result, map_png_path)

    run_info_path = out_dir / "run_info.json"
    run_info = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run.run_dir),
        "run_name": run.name,
        "config": asdict(cfg),
        "p_stats": dict(result.p_stats),
        "debug": dict(result.debug),
        "units": {
            "B": "T",
            "Bnorm": "T",
            "ppm": "ppm",
            "position": "m",
            "sigma_rel_pct": "%",
            "sigma_phi_deg": "deg",
            "volume": "mm^3",
        },
    }
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "roi_vectors_npz": roi_npz_path,
        "roi_vectors_csv": roi_csv_path,
        "map_xy_npz": map_npz_path,
        "map_xy_png": map_png_path,
        "run_info_json": run_info_path,
    }


__all__ = [
    "PerturbationConfig",
    "PerturbationResult",
    "run_perturbation_case",
    "save_perturbation_result",
]
