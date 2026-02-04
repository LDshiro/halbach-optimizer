from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import angle_model_from_run, phi_rkn_from_run
from halbach.constants import FACTOR, m0, phi0
from halbach.magnetization_runtime import (
    compute_b_and_b0_from_m_flat,
    compute_m_flat_from_run,
    get_magnetization_config_from_meta,
)
from halbach.physics import compute_B_and_B0, compute_B_and_B0_phi_rkn
from halbach.run_types import RunBundle
from halbach.types import FloatArray

Plane = Literal["xy", "xz", "yz"]
MagModelEval = Literal["auto", "fixed", "self-consistent-easy-axis"]


@dataclass(frozen=True)
class ErrorMap2D:
    xs: FloatArray
    ys: FloatArray
    ppm: FloatArray
    mask: NDArray[np.bool_]
    B0_T: float
    plane: Plane
    coord0: float


@dataclass(frozen=True)
class CrossSection1D:
    x: FloatArray
    ppm: FloatArray
    y0: float


def _axis_grid(roi_r: float, step: float) -> FloatArray:
    n = int(2 * np.ceil(roi_r / step) + 1)
    return np.linspace(-roi_r, roi_r, n, dtype=np.float64)


def _build_plane_points(
    xs: FloatArray, ys: FloatArray, plane: Plane, coord0: float, roi_r: float
) -> tuple[NDArray[np.bool_], FloatArray]:
    if plane == "xy":
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        mask = (X * X + Y * Y + coord0 * coord0) <= roi_r * roi_r
        pts = np.column_stack([X[mask], Y[mask], np.full(mask.sum(), coord0)])
    elif plane == "xz":
        X, Z = np.meshgrid(xs, ys, indexing="xy")
        mask = (X * X + coord0 * coord0 + Z * Z) <= roi_r * roi_r
        pts = np.column_stack([X[mask], np.full(mask.sum(), coord0), Z[mask]])
    elif plane == "yz":
        Y, Z = np.meshgrid(xs, ys, indexing="xy")
        mask = (coord0 * coord0 + Y * Y + Z * Z) <= roi_r * roi_r
        pts = np.column_stack([np.full(mask.sum(), coord0), Y[mask], Z[mask]])
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    return mask, np.asarray(pts, dtype=np.float64)


def _build_r0_rkn(run: RunBundle) -> NDArray[np.float64]:
    geom = run.geometry
    r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
    rho = r_bases[None, :] + np.asarray(geom.ring_offsets, dtype=np.float64)[:, None]
    px = rho[:, :, None] * np.asarray(geom.cth, dtype=np.float64)[None, None, :]
    py = rho[:, :, None] * np.asarray(geom.sth, dtype=np.float64)[None, None, :]
    pz = np.broadcast_to(np.asarray(geom.z_layers, dtype=np.float64)[None, :, None], px.shape)
    return np.stack([px, py, pz], axis=-1)


def _is_dc_run(run: RunBundle) -> bool:
    return run.meta.get("framework") == "dc"


def _dc_extract_arrays(
    run: RunBundle,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    extras = run.results.extras
    if "pts" not in extras or "r0_flat" not in extras:
        raise KeyError("DC run results must include pts and r0_flat")
    pts = np.asarray(extras["pts"], dtype=np.float64)
    r0_flat = np.asarray(extras["r0_flat"], dtype=np.float64)
    m_flat: NDArray[np.float64] | None = None
    if "m_flat" in extras:
        m_flat = np.asarray(extras["m_flat"], dtype=np.float64)
    elif "x_opt" in extras:
        x_opt = np.asarray(extras["x_opt"], dtype=np.float64).reshape(-1)
        if x_opt.size != 2 * r0_flat.shape[0]:
            raise ValueError("x_opt length does not match r0_flat")
        m_flat = np.zeros((r0_flat.shape[0], 3), dtype=np.float64)
        m_flat[:, 0] = x_opt[0::2]
        m_flat[:, 1] = x_opt[1::2]
    elif "x_sc_post" in extras:
        x_sc = np.asarray(extras["x_sc_post"], dtype=np.float64).reshape(-1)
        if x_sc.size != 2 * r0_flat.shape[0]:
            raise ValueError("x_sc_post length does not match r0_flat")
        m_flat = np.zeros((r0_flat.shape[0], 3), dtype=np.float64)
        m_flat[:, 0] = x_sc[0::2]
        m_flat[:, 1] = x_sc[1::2]
    if m_flat is None:
        raise KeyError("DC run results must include m_flat or x_opt/x_sc_post")
    if "center_idx" in extras:
        center_idx = int(np.asarray(extras["center_idx"]).reshape(-1)[0])
    else:
        center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    return pts, r0_flat, m_flat, center_idx


def _grid_from_pts_xy(
    pts: NDArray[np.float64],
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], NDArray[np.int_]]:
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    xs = np.sort(xs.astype(np.float64))
    ys = np.sort(ys.astype(np.float64))
    mask = np.zeros((ys.size, xs.size), dtype=bool)
    ix_list: list[int] = []
    iy_list: list[int] = []
    for x, y in pts[:, :2]:
        ix = int(np.argmin(np.abs(xs - x)))
        iy = int(np.argmin(np.abs(ys - y)))
        mask[iy, ix] = True
        ix_list.append(ix)
        iy_list.append(iy)
    idx_map = np.column_stack([iy_list, ix_list]).astype(np.int_)
    return xs, ys, mask, idx_map


def _compute_error_map_impl(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
    mag_model_eval: MagModelEval = "auto",
    sc_cfg_override: dict[str, Any] | None = None,
) -> tuple[ErrorMap2D, dict[str, object]]:
    if _is_dc_run(run):
        pts, r0_flat, m_flat, center_idx = _dc_extract_arrays(run)
        xs, ys, mask, idx_map = _grid_from_pts_xy(pts)
        factor = float(run.meta.get("factor", FACTOR))
        Bx, By, Bz, B0x, B0y, B0z = compute_b_and_b0_from_m_flat(
            m_flat, r0_flat, pts, factor=factor
        )
        Bnorm = np.sqrt(Bx * Bx + By * By + Bz * Bz)
        if 0 <= center_idx < Bnorm.size:
            B0_T = float(Bnorm[center_idx])
        else:
            B0_T = float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))
        if B0_T < 1e-15:
            raise ValueError("B0_T is too small for stable ppm normalization")
        ppm_vals = (Bnorm - B0_T) / B0_T * 1e6
        ppm = np.full(mask.shape, np.nan, dtype=np.float64)
        for k, (iy, ix) in enumerate(idx_map):
            ppm[iy, ix] = float(ppm_vals[k])
        debug_dc: dict[str, object] = {
            "model_effective": "dc",
            "framework": "dc",
            "center_idx": center_idx,
        }
        return (
            ErrorMap2D(
                xs=xs,
                ys=ys,
                ppm=ppm,
                mask=mask,
                B0_T=B0_T,
                plane="xy",
                coord0=0.0,
            ),
            debug_dc,
        )

    xs = _axis_grid(roi_r, step)
    ys = _axis_grid(roi_r, step)

    mask, pts = _build_plane_points(xs, ys, plane, coord0, roi_r)

    model_effective_meta, sc_cfg_meta = get_magnetization_config_from_meta(run.meta)
    if mag_model_eval == "auto":
        model_effective_eval = model_effective_meta
        sc_cfg_eval = sc_cfg_meta
    elif mag_model_eval == "self-consistent-easy-axis":
        model_effective_eval = "self-consistent-easy-axis"
        sc_cfg_eval = sc_cfg_override or sc_cfg_meta
    else:
        model_effective_eval = "fixed"
        sc_cfg_eval = {}

    debug: dict[str, object] = {
        "model_effective": model_effective_eval,
        "model_effective_meta": model_effective_meta,
        "model_effective_eval": model_effective_eval,
    }

    if model_effective_eval == "self-consistent-easy-axis":
        phi_rkn = phi_rkn_from_run(run, phi0=phi0)
        r0_rkn = _build_r0_rkn(run)
        sc_override = sc_cfg_eval if mag_model_eval != "auto" else None
        m_flat, sc_debug = compute_m_flat_from_run(
            run.run_dir,
            run.geometry,
            phi_rkn,
            r0_rkn,
            sc_cfg_override=sc_override,
        )
        r0_flat = r0_rkn.reshape(-1, 3)
        Bx, By, Bz, B0x, B0y, B0z = compute_b_and_b0_from_m_flat(
            m_flat, r0_flat, pts, factor=FACTOR
        )
        debug.update(sc_debug)
    else:
        model = angle_model_from_run(run)
        if model == "legacy-alpha":
            Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0(
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
            Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0_phi_rkn(
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

    B0_T = float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))
    if B0_T < 1e-15:
        raise ValueError("B0_T is too small for stable ppm normalization")

    Bnorm = np.sqrt(Bx * Bx + By * By + Bz * Bz)
    ppm_vals = (Bnorm - B0_T) / B0_T * 1e6

    ppm = np.full(mask.shape, np.nan, dtype=np.float64)
    ppm[mask] = np.asarray(ppm_vals, dtype=np.float64)

    return (
        ErrorMap2D(
            xs=xs,
            ys=ys,
            ppm=ppm,
            mask=mask,
            B0_T=B0_T,
            plane=plane,
            coord0=coord0,
        ),
        debug,
    )


def compute_error_map_ppm_plane(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
    mag_model_eval: MagModelEval = "auto",
    sc_cfg_override: dict[str, Any] | None = None,
) -> ErrorMap2D:
    m, _debug = _compute_error_map_impl(
        run,
        plane=plane,
        coord0=coord0,
        roi_r=roi_r,
        step=step,
        mag_model_eval=mag_model_eval,
        sc_cfg_override=sc_cfg_override,
    )
    return m


def compute_error_map_ppm_plane_with_debug(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
    mag_model_eval: MagModelEval = "auto",
    sc_cfg_override: dict[str, Any] | None = None,
) -> tuple[ErrorMap2D, dict[str, object]]:
    return _compute_error_map_impl(
        run,
        plane=plane,
        coord0=coord0,
        roi_r=roi_r,
        step=step,
        mag_model_eval=mag_model_eval,
        sc_cfg_override=sc_cfg_override,
    )


def compute_error_map_ppm_plane(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
) -> ErrorMap2D:
    m, _debug = _compute_error_map_impl(run, plane=plane, coord0=coord0, roi_r=roi_r, step=step)
    return m


def compute_error_map_ppm_plane_with_debug(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
) -> tuple[ErrorMap2D, dict[str, object]]:
    return _compute_error_map_impl(run, plane=plane, coord0=coord0, roi_r=roi_r, step=step)


def extract_cross_section_y0(m: ErrorMap2D) -> CrossSection1D:
    idx = int(np.argmin(np.abs(m.ys)))
    y0 = float(m.ys[idx])
    ppm_line = np.asarray(m.ppm[idx, :], dtype=np.float64)
    return CrossSection1D(x=m.xs, ppm=ppm_line, y0=y0)


def common_ppm_limits(
    maps: Sequence[ErrorMap2D],
    *,
    limit_ppm: float | None = 5000.0,
    symmetric: bool = True,
) -> tuple[float, float]:
    if limit_ppm is not None:
        limit = float(limit_ppm)
        return -limit, limit
    if not maps:
        raise ValueError("maps must be non-empty when limit_ppm is None")

    mins = [float(np.nanmin(m.ppm)) for m in maps]
    maxs = [float(np.nanmax(m.ppm)) for m in maps]
    vmin = float(np.nanmin(np.array(mins, dtype=np.float64)))
    vmax = float(np.nanmax(np.array(maxs, dtype=np.float64)))
    if symmetric:
        max_abs = max(abs(vmin), abs(vmax))
        return -max_abs, max_abs
    return vmin, vmax


def contour_levels_ppm(level: float = 1000.0) -> tuple[float, float]:
    level_f = float(level)
    return -level_f, level_f


__all__ = [
    "ErrorMap2D",
    "CrossSection1D",
    "compute_error_map_ppm_plane",
    "compute_error_map_ppm_plane_with_debug",
    "extract_cross_section_y0",
    "common_ppm_limits",
    "contour_levels_ppm",
]
