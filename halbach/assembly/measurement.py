from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from halbach.assembly.types import MagnetError, MeasuredMagnet
from halbach.types import FloatArray

MIN_UZ_ABS = 1e-12


def _normalize_direction(direction: Sequence[float] | FloatArray) -> FloatArray:
    arr = np.asarray(direction, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"direction must have 3 components, got {arr.size}")
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction must have nonzero finite norm")
    return np.asarray(arr / norm, dtype=np.float64)


def measured_magnet_from_direction(
    moment_magnitude: float,
    direction: Sequence[float] | FloatArray,
    *,
    nominal_magnitude: float = 1.0,
    quality: float | None = None,
    cluster_id: str | None = None,
) -> MeasuredMagnet:
    """
    Convert a measurement-frame direction to Plan C magnet error coordinates.

    direction: shape (3,), measurement frame with +Z_meas as nominal magnetization axis
    moment_magnitude: measured moment magnitude, same unit as nominal_magnitude
    nominal_magnitude: positive nominal moment magnitude
    """
    moment = float(moment_magnitude)
    nominal = float(nominal_magnitude)
    if not math.isfinite(moment):
        raise ValueError("moment_magnitude must be finite")
    if not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("nominal_magnitude must be positive and finite")

    unit_direction = _normalize_direction(direction)
    uz = float(unit_direction[2])
    if abs(uz) < MIN_UZ_ABS:
        raise ValueError("direction z component is too small for small-angle conversion")

    error = MagnetError(
        epsilon_parallel=moment / nominal - 1.0,
        delta_perp_1=float(unit_direction[0]) / uz,
        delta_perp_2=float(unit_direction[1]) / uz,
    )
    return MeasuredMagnet(
        error=error,
        moment_magnitude=moment,
        direction=unit_direction,
        quality=None if quality is None else float(quality),
        cluster_id=cluster_id,
    )


__all__ = ["measured_magnet_from_direction"]
