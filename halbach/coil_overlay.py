from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

Int64Array: TypeAlias = NDArray[np.int64]
UInt8Array: TypeAlias = NDArray[np.uint8]
BoolArray: TypeAlias = NDArray[np.bool_]

REQUIRED_KEYS = {
    "format_name",
    "format_version",
    "points_xyz",
    "polyline_start",
    "polyline_count",
    "polyline_surface",
    "polyline_sign",
    "polyline_closed",
    "polyline_periodic_closed",
    "polyline_color_rgba_u8",
    "source_contour_npz",
    "source_config_json",
    "input_kind",
    "coordinate_unit",
    "color_space",
}


@dataclass(frozen=True)
class CoilPolylineSet:
    points_xyz: FloatArray
    polyline_start: Int64Array
    polyline_count: Int64Array
    polyline_color_rgba_u8: UInt8Array
    bbox_min: FloatArray
    bbox_max: FloatArray

    @property
    def bbox_size(self) -> FloatArray:
        return np.asarray(self.bbox_max - self.bbox_min, dtype=np.float64)


@dataclass(frozen=True)
class CoilTraceGroup:
    x: FloatArray
    y: FloatArray
    z: FloatArray
    color_css: str
    color_rgba_u8: UInt8Array
    polyline_count: int


def _require_keys(npz_keys: set[str]) -> None:
    missing = sorted(REQUIRED_KEYS - npz_keys)
    if missing:
        items = ", ".join(missing)
        raise ValueError(f"Coil NPZ missing required keys: {items}")


def _load_scalar_string(npz: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"Coil NPZ key {key} must be a scalar string")
    return str(value.item())


def _validate_polyline_layout(
    points_xyz: FloatArray, starts: Int64Array, counts: Int64Array
) -> None:
    if np.any(starts < 0):
        raise ValueError("polyline_start must be >= 0")
    if np.any(counts <= 0):
        raise ValueError("polyline_count must be > 0")
    if int(np.sum(counts, dtype=np.int64)) != int(points_xyz.shape[0]):
        raise ValueError("sum(polyline_count) must equal len(points_xyz)")
    if starts.shape != counts.shape:
        raise ValueError("polyline_start and polyline_count must have the same shape")
    max_end = starts + counts
    if np.any(max_end > points_xyz.shape[0]):
        raise ValueError("polyline_start/polyline_count exceed points_xyz length")


def load_coil_polyline_npz(path: Path) -> CoilPolylineSet:
    with np.load(path, allow_pickle=False) as npz:
        _require_keys(set(npz.files))
        if _load_scalar_string(npz, "format_name") != "gradientcoil_3d_coil":
            raise ValueError("Unsupported coil NPZ format_name")
        if _load_scalar_string(npz, "format_version") != "1.0":
            raise ValueError("Unsupported coil NPZ format_version")
        if _load_scalar_string(npz, "coordinate_unit") != "m":
            raise ValueError("Unsupported coil coordinate_unit")
        if _load_scalar_string(npz, "color_space") != "srgba_u8":
            raise ValueError("Unsupported coil color_space")

        points_xyz = np.asarray(npz["points_xyz"], dtype=np.float64)
        starts = np.asarray(npz["polyline_start"], dtype=np.int64)
        counts = np.asarray(npz["polyline_count"], dtype=np.int64)
        colors = np.asarray(npz["polyline_color_rgba_u8"], dtype=np.uint8)
        polyline_closed = np.asarray(npz["polyline_closed"], dtype=np.bool_)
        polyline_periodic_closed = np.asarray(npz["polyline_periodic_closed"], dtype=np.bool_)
        polyline_surface = np.asarray(npz["polyline_surface"], dtype=np.int64)
        polyline_sign = np.asarray(npz["polyline_sign"], dtype=np.float64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if starts.ndim != 1 or counts.ndim != 1:
        raise ValueError("polyline_start and polyline_count must be 1D")
    n_poly = starts.shape[0]
    if colors.shape != (n_poly, 4):
        raise ValueError("polyline_color_rgba_u8 must have shape (L, 4)")
    if polyline_closed.shape != (n_poly,):
        raise ValueError("polyline_closed must have shape (L,)")
    if polyline_periodic_closed.shape != (n_poly,):
        raise ValueError("polyline_periodic_closed must have shape (L,)")
    if polyline_surface.shape != (n_poly,):
        raise ValueError("polyline_surface must have shape (L,)")
    if polyline_sign.shape != (n_poly,):
        raise ValueError("polyline_sign must have shape (L,)")

    _validate_polyline_layout(points_xyz, starts, counts)

    bbox_min = np.min(points_xyz, axis=0).astype(np.float64, copy=False)
    bbox_max = np.max(points_xyz, axis=0).astype(np.float64, copy=False)
    return CoilPolylineSet(
        points_xyz=points_xyz,
        polyline_start=starts,
        polyline_count=counts,
        polyline_color_rgba_u8=colors,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _rgba_u8_to_css(color_rgba_u8: UInt8Array) -> str:
    color = np.asarray(color_rgba_u8, dtype=np.uint8).reshape(4)
    alpha = float(color[3]) / 255.0
    return f"rgba({int(color[0])},{int(color[1])},{int(color[2])},{alpha:.3f})"


def build_plotly_polyline_groups(coils: CoilPolylineSet) -> list[CoilTraceGroup]:
    grouped_xyz: dict[
        tuple[int, int, int, int], tuple[list[float], list[float], list[float], int]
    ] = {}

    for idx, (start, count) in enumerate(
        zip(coils.polyline_start, coils.polyline_count, strict=False)
    ):
        pts = coils.points_xyz[int(start) : int(start + count)]
        rgba_row = coils.polyline_color_rgba_u8[idx]
        rgba_key = (
            int(rgba_row[0]),
            int(rgba_row[1]),
            int(rgba_row[2]),
            int(rgba_row[3]),
        )
        xs, ys, zs, poly_count = grouped_xyz.setdefault(rgba_key, ([], [], [], 0))
        xs.extend(float(v) for v in pts[:, 0])
        xs.append(np.nan)
        ys.extend(float(v) for v in pts[:, 1])
        ys.append(np.nan)
        zs.extend(float(v) for v in pts[:, 2])
        zs.append(np.nan)
        grouped_xyz[rgba_key] = (xs, ys, zs, poly_count + 1)

    groups: list[CoilTraceGroup] = []
    for rgba_key, (xs, ys, zs, poly_count) in grouped_xyz.items():
        color_rgba = np.asarray(rgba_key, dtype=np.uint8)
        groups.append(
            CoilTraceGroup(
                x=np.asarray(xs, dtype=np.float64),
                y=np.asarray(ys, dtype=np.float64),
                z=np.asarray(zs, dtype=np.float64),
                color_css=_rgba_u8_to_css(color_rgba),
                color_rgba_u8=color_rgba,
                polyline_count=poly_count,
            )
        )
    return groups


def rotate_coil_polyline_x90(coils: CoilPolylineSet, quarter_turns: int) -> CoilPolylineSet:
    turns = int(quarter_turns) % 4
    if turns == 0:
        return coils

    points_xyz = np.asarray(coils.points_xyz, dtype=np.float64).copy()
    x = points_xyz[:, 0].copy()
    y = points_xyz[:, 1].copy()
    z = points_xyz[:, 2].copy()

    if turns == 1:
        points_xyz[:, 0] = x
        points_xyz[:, 1] = -z
        points_xyz[:, 2] = y
    elif turns == 2:
        points_xyz[:, 0] = x
        points_xyz[:, 1] = -y
        points_xyz[:, 2] = -z
    else:
        points_xyz[:, 0] = x
        points_xyz[:, 1] = z
        points_xyz[:, 2] = -y

    bbox_min = np.min(points_xyz, axis=0).astype(np.float64, copy=False)
    bbox_max = np.max(points_xyz, axis=0).astype(np.float64, copy=False)
    return CoilPolylineSet(
        points_xyz=points_xyz,
        polyline_start=np.asarray(coils.polyline_start, dtype=np.int64),
        polyline_count=np.asarray(coils.polyline_count, dtype=np.int64),
        polyline_color_rgba_u8=np.asarray(coils.polyline_color_rgba_u8, dtype=np.uint8),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


__all__ = [
    "CoilPolylineSet",
    "CoilTraceGroup",
    "load_coil_polyline_npz",
    "build_plotly_polyline_groups",
    "rotate_coil_polyline_x90",
]
