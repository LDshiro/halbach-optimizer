from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from halbach.assembly.orientations import default_orientations
from halbach.assembly.sensitivity import (
    SENSITIVITY_SCHEMA_VERSION,
    compute_sensitivity_table,
    load_sensitivity_table,
    save_sensitivity_table,
)
from halbach.assembly.types import AssemblySlot, OrientationCandidate, SensitivityTable
from halbach.constants import FACTOR
from halbach.types import FloatArray

SENSITIVITY_CACHE_VERSION = 1


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> object: ...


@dataclass(frozen=True)
class CachedSensitivityTable:
    """Sensitivity table loaded from or written to the disk cache."""

    table: SensitivityTable
    cache_path: Path
    cache_hit: bool
    cache_key: str


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_array(hasher: _Hasher, value: NDArray[np.generic]) -> None:
    arr = np.ascontiguousarray(value)
    hasher.update(_json_bytes({"shape": arr.shape, "dtype": str(arr.dtype)}))
    hasher.update(arr.tobytes(order="C"))


def _float_array(value: FloatArray | None, *, default: FloatArray) -> FloatArray:
    arr = default if value is None else np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("origin must contain finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _orientation_payload(
    orientations: Sequence[OrientationCandidate],
) -> tuple[dict[str, object], ...]:
    if not orientations:
        raise ValueError("orientations must be non-empty")
    return tuple(
        {
            "id": orientation.id,
            "angle_deg": float(orientation.angle_deg),
        }
        for orientation in orientations
    )


def sensitivity_cache_key(
    slots: Sequence[AssemblySlot],
    roi_points: FloatArray,
    *,
    orientations: Sequence[OrientationCandidate] | None = None,
    finite_difference_step: float = 1e-6,
    factor: float = FACTOR,
    origin: FloatArray | None = None,
) -> str:
    """Return a stable hash for the numerical inputs of a sensitivity table."""
    if not slots:
        raise ValueError("slots must be non-empty")
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    if not math.isfinite(factor):
        raise ValueError("factor must be finite")
    roi = np.asarray(roi_points, dtype=np.float64)
    if roi.ndim != 2 or roi.shape[1] != 3 or roi.shape[0] == 0:
        raise ValueError("roi_points must have shape (P, 3) with P >= 1")
    if not np.all(np.isfinite(roi)):
        raise ValueError("roi_points must contain finite values")
    orientation_items = tuple(default_orientations() if orientations is None else orientations)
    origin_arr = _float_array(origin, default=np.zeros(3, dtype=np.float64))

    hasher = hashlib.sha256()
    hasher.update(
        _json_bytes(
            {
                "cache_version": SENSITIVITY_CACHE_VERSION,
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                "finite_difference_step": float(finite_difference_step),
                "factor": float(factor),
                "orientations": _orientation_payload(orientation_items),
            }
        )
    )
    _hash_array(hasher, np.asarray([slot.slot_flat_id for slot in slots], dtype=np.int64))
    _hash_array(hasher, np.asarray([slot.ring_id for slot in slots], dtype=np.int64))
    _hash_array(hasher, np.asarray([slot.layer_id for slot in slots], dtype=np.int64))
    _hash_array(hasher, np.asarray([slot.theta_id for slot in slots], dtype=np.int64))
    _hash_array(
        hasher,
        np.asarray([slot.center_m for slot in slots], dtype=np.float64),
    )
    _hash_array(
        hasher,
        np.asarray([slot.nominal_u for slot in slots], dtype=np.float64),
    )
    _hash_array(hasher, np.ascontiguousarray(roi, dtype=np.float64))
    _hash_array(hasher, origin_arr)
    return hasher.hexdigest()


def sensitivity_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    """Return the NPZ path for a sensitivity cache key."""
    if not cache_key:
        raise ValueError("cache_key must be non-empty")
    return Path(cache_dir) / f"sensitivity_{cache_key}.npz"


def _cache_metadata(
    *,
    cache_key: str,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    result = {} if metadata is None else dict(metadata)
    result.update(
        {
            "sensitivity_cache_key": cache_key,
            "sensitivity_cache_version": SENSITIVITY_CACHE_VERSION,
        }
    )
    return result


def _load_cached_table(cache_path: Path, cache_key: str) -> SensitivityTable | None:
    if not cache_path.exists():
        return None
    try:
        table = load_sensitivity_table(cache_path)
    except Exception:
        cache_path.unlink(missing_ok=True)
        return None
    if table.metadata.get("sensitivity_cache_key") != cache_key:
        cache_path.unlink(missing_ok=True)
        return None
    if table.metadata.get("sensitivity_cache_version") != SENSITIVITY_CACHE_VERSION:
        cache_path.unlink(missing_ok=True)
        return None
    return table


def load_or_compute_sensitivity_table(
    cache_dir: str | Path,
    slots: Sequence[AssemblySlot],
    roi_points: FloatArray,
    *,
    orientations: Sequence[OrientationCandidate] | None = None,
    finite_difference_step: float = 1e-6,
    factor: float = FACTOR,
    origin: FloatArray | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CachedSensitivityTable:
    """Load a sensitivity table from disk cache, or compute and cache it."""
    cache_key = sensitivity_cache_key(
        slots,
        roi_points,
        orientations=orientations,
        finite_difference_step=finite_difference_step,
        factor=factor,
        origin=origin,
    )
    cache_path = sensitivity_cache_path(cache_dir, cache_key)
    cached = _load_cached_table(cache_path, cache_key)
    if cached is not None:
        return CachedSensitivityTable(
            table=cached,
            cache_path=cache_path,
            cache_hit=True,
            cache_key=cache_key,
        )

    table = compute_sensitivity_table(
        slots,
        roi_points,
        orientations=orientations,
        finite_difference_step=finite_difference_step,
        factor=factor,
        origin=origin,
        metadata=_cache_metadata(cache_key=cache_key, metadata=metadata),
    )
    save_sensitivity_table(cache_path, table)
    return CachedSensitivityTable(
        table=table,
        cache_path=cache_path,
        cache_hit=False,
        cache_key=cache_key,
    )


__all__ = [
    "CachedSensitivityTable",
    "SENSITIVITY_CACHE_VERSION",
    "load_or_compute_sensitivity_table",
    "sensitivity_cache_key",
    "sensitivity_cache_path",
]
