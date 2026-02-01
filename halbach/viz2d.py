from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import angle_model_from_run, phi_rkn_from_run
from halbach.constants import FACTOR, m0, phi0
from halbach.physics import compute_B_and_B0, compute_B_and_B0_phi_rkn
from halbach.run_types import RunBundle
from halbach.types import FloatArray

Plane = Literal["xy", "xz", "yz"]


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


def compute_error_map_ppm_plane(
    run: RunBundle,
    *,
    plane: Plane = "xy",
    coord0: float = 0.0,
    roi_r: float = 0.14,
    step: float = 0.001,
) -> ErrorMap2D:
    xs = _axis_grid(roi_r, step)
    ys = _axis_grid(roi_r, step)

    mask, pts = _build_plane_points(xs, ys, plane, coord0, roi_r)

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

    return ErrorMap2D(
        xs=xs,
        ys=ys,
        ppm=ppm,
        mask=mask,
        B0_T=B0_T,
        plane=plane,
        coord0=coord0,
    )


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
    "extract_cross_section_y0",
    "common_ppm_limits",
    "contour_levels_ppm",
]
