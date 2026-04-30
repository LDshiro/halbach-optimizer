from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from halbach.assembly.types import MagnetError, MeasuredMagnet, VirtualMagnet
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


class MeasurementProvider(Protocol):
    """Sequential measured-magnet source for Plan C sessions."""

    @property
    def position(self) -> int:
        """Number of magnets already consumed."""
        ...

    def set_position(self, position: int) -> None:
        """Move the provider cursor, used by session resume."""
        ...

    def next_magnet(self) -> VirtualMagnet:
        """Return the next measured/simulated magnet."""
        ...


class SyntheticMeasurementProvider:
    """Measurement provider backed by an in-memory sequence of virtual magnets."""

    def __init__(self, magnets: Sequence[VirtualMagnet], *, start_index: int = 0) -> None:
        self._magnets = tuple(magnets)
        self._position = 0
        self.set_position(start_index)

    @property
    def position(self) -> int:
        return self._position

    def set_position(self, position: int) -> None:
        pos = int(position)
        if pos < 0 or pos > len(self._magnets):
            raise ValueError("provider position is out of range")
        self._position = pos

    def next_magnet(self) -> VirtualMagnet:
        if self._position >= len(self._magnets):
            raise StopIteration("no more synthetic magnets")
        magnet = self._magnets[self._position]
        self._position += 1
        return magnet


class ManualMeasurementProvider(SyntheticMeasurementProvider):
    """Manual provider MVP: callers append measurements before consuming them."""

    def __init__(self, magnets: Sequence[VirtualMagnet] | None = None) -> None:
        self._manual_magnets: list[VirtualMagnet] = list(magnets or [])
        super().__init__(self._manual_magnets)

    def submit_magnet(self, magnet: VirtualMagnet) -> None:
        self._manual_magnets.append(magnet)
        self._magnets = tuple(self._manual_magnets)


def _float_from_row(row: dict[str, str], key: str, default: float) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def _error_from_row(row: dict[str, str], prefix: str, fallback: MagnetError | None = None) -> MagnetError:
    eps_key = f"{prefix}epsilon_parallel"
    d1_key = f"{prefix}delta_perp_1"
    d2_key = f"{prefix}delta_perp_2"
    if eps_key in row or d1_key in row or d2_key in row:
        return MagnetError(
            epsilon_parallel=_float_from_row(
                row,
                eps_key,
                0.0 if fallback is None else fallback.epsilon_parallel,
            ),
            delta_perp_1=_float_from_row(
                row,
                d1_key,
                0.0 if fallback is None else fallback.delta_perp_1,
            ),
            delta_perp_2=_float_from_row(
                row,
                d2_key,
                0.0 if fallback is None else fallback.delta_perp_2,
            ),
        )
    if fallback is not None:
        return fallback
    return MagnetError(
        epsilon_parallel=_float_from_row(row, "epsilon_parallel", 0.0),
        delta_perp_1=_float_from_row(row, "delta_perp_1", 0.0),
        delta_perp_2=_float_from_row(row, "delta_perp_2", 0.0),
    )


def virtual_magnet_from_mapping(row: dict[str, str], *, default_magnet_id: int) -> VirtualMagnet:
    """Convert a CSV/JSON row to a VirtualMagnet."""
    magnet_id = int(row.get("magnet_id", str(default_magnet_id)))
    true_error = _error_from_row(row, "true_")
    measured_error = _error_from_row(row, "measured_", fallback=true_error)
    quality_raw = row.get("quality", "")
    quality = None if quality_raw == "" else float(quality_raw)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=true_error,
        measured_error=measured_error,
        quality=quality,
    )


class CsvMeasurementProvider(SyntheticMeasurementProvider):
    """
    CSV-backed provider.

    Supported columns:
    magnet_id, epsilon_parallel, delta_perp_1, delta_perp_2
    or true_* / measured_* variants, plus optional quality.
    """

    def __init__(self, path: str | Path, *, start_index: int = 0) -> None:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        magnets = [
            virtual_magnet_from_mapping(row, default_magnet_id=idx)
            for idx, row in enumerate(rows)
        ]
        super().__init__(magnets, start_index=start_index)


def _string_mapping_from_json(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("serial measurement line must decode to an object")
    return {str(key): str(value) for key, value in raw.items()}


class FakeSerialMeasurementProvider(SyntheticMeasurementProvider):
    """Fake serial provider backed by JSON lines for tests and interface wiring."""

    def __init__(self, lines: Sequence[str], *, start_index: int = 0) -> None:
        magnets = [
            virtual_magnet_from_mapping(
                _string_mapping_from_json(json.loads(line)),
                default_magnet_id=idx,
            )
            for idx, line in enumerate(lines)
        ]
        super().__init__(magnets, start_index=start_index)


__all__ = [
    "CsvMeasurementProvider",
    "FakeSerialMeasurementProvider",
    "ManualMeasurementProvider",
    "MeasurementProvider",
    "SyntheticMeasurementProvider",
    "measured_magnet_from_direction",
    "virtual_magnet_from_mapping",
]
