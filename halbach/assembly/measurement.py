from __future__ import annotations

import csv
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from halbach.assembly.types import MagnetError, MeasuredMagnet, QuarantineReason, VirtualMagnet
from halbach.types import FloatArray

MIN_UZ_ABS = 1e-12
DEFAULT_QUALITY_THRESHOLD = 0.90


class MeasurementProviderError(Exception):
    """Base class for recoverable measurement provider failures."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = self.__class__.__name__ if reason is None else reason
        self.quarantine_id: QuarantineReason | None = None


class MeasurementParseError(MeasurementProviderError):
    """A serial measurement line could not be parsed or validated."""


class MeasurementTimeoutError(MeasurementProviderError):
    """No complete measurement line was available before timeout."""


class MeasurementQualityError(MeasurementProviderError):
    """A parsed measurement is below the accepted quality threshold."""

    def __init__(self, quality: float, quality_threshold: float) -> None:
        super().__init__(
            f"measurement quality {quality:g} is below threshold {quality_threshold:g}",
            reason="Q_MEASUREMENT_UNSTABLE",
        )
        self.quality = float(quality)
        self.quality_threshold = float(quality_threshold)
        self.quarantine_id = "Q_MEASUREMENT_UNSTABLE"


class SerialDependencyError(MeasurementProviderError):
    """pyserial is required only when the real serial provider is instantiated."""


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


def _float_from_object(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise MeasurementParseError(f"serial measurement field {key} must be a float")
    if not isinstance(value, int | float | str | bytes | bytearray):
        raise MeasurementParseError(f"serial measurement field {key} must be a float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementParseError(f"serial measurement field {key} must be a float") from exc
    if not math.isfinite(out):
        raise MeasurementParseError(f"serial measurement field {key} must be finite")
    return out


def _float_from_mapping(raw: Mapping[str, object], key: str) -> float:
    if key not in raw:
        raise MeasurementParseError(f"serial measurement missing required field: {key}")
    return _float_from_object(raw[key], key)


def _optional_float_from_mapping(raw: Mapping[str, object], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return _float_from_object(raw[key], key)


def _int_from_mapping(raw: Mapping[str, object], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise MeasurementParseError(f"serial measurement field {key} must be an int")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str | bytes | bytearray):
        try:
            return int(value)
        except ValueError as exc:
            raise MeasurementParseError(f"serial measurement field {key} must be an int") from exc
    raise MeasurementParseError(f"serial measurement field {key} must be an int")


def _direction_from_mapping(raw: Mapping[str, object]) -> Sequence[float] | FloatArray:
    if "direction" not in raw:
        raise MeasurementParseError("serial measurement missing required field: direction")
    direction = raw["direction"]
    if isinstance(direction, str | bytes):
        raise MeasurementParseError("serial measurement direction must be a 3-element array")
    if not isinstance(direction, Sequence):
        raise MeasurementParseError("serial measurement direction must be a 3-element array")
    try:
        return [float(item) for item in direction]
    except (TypeError, ValueError) as exc:
        raise MeasurementParseError("serial measurement direction must contain floats") from exc


def _legacy_error_keys_present(raw: Mapping[str, object]) -> bool:
    keys = set(raw)
    return bool(
        keys
        & {
            "epsilon_parallel",
            "delta_perp_1",
            "delta_perp_2",
            "true_epsilon_parallel",
            "true_delta_perp_1",
            "true_delta_perp_2",
            "measured_epsilon_parallel",
            "measured_delta_perp_1",
            "measured_delta_perp_2",
        }
    )


def _enforce_quality(
    quality: float | None,
    *,
    quality_threshold: float,
) -> None:
    if quality is None:
        return
    if quality < float(quality_threshold):
        raise MeasurementQualityError(quality, float(quality_threshold))


def _object_from_json_line(line: str | bytes) -> dict[str, object]:
    if isinstance(line, bytes):
        text = line.decode("utf-8", errors="replace")
    else:
        text = str(line)
    if text.strip() == "":
        raise MeasurementTimeoutError("serial measurement timed out before a JSON line was received")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MeasurementParseError("serial measurement line is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise MeasurementParseError("serial measurement line must decode to an object")
    return {str(key): value for key, value in raw.items()}


def parse_serial_measurement_line(
    line: str | bytes,
    *,
    default_magnet_id: int,
    nominal_magnitude: float = 1.0,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> VirtualMagnet:
    """
    Parse one JSON line from a serial measurement stream.

    Standard JSON shape:
        {"moment_magnitude": float, "direction": [x, y, z], "quality": float?}

    Legacy JSON rows with epsilon/delta fields are also accepted for fake streams.
    """
    raw = _object_from_json_line(line)
    if _legacy_error_keys_present(raw):
        row = _string_mapping_from_json(raw)
        try:
            magnet = virtual_magnet_from_mapping(row, default_magnet_id=default_magnet_id)
        except (TypeError, ValueError) as exc:
            raise MeasurementParseError("legacy serial measurement fields are invalid") from exc
        _enforce_quality(magnet.quality, quality_threshold=quality_threshold)
        return magnet

    moment = _float_from_mapping(raw, "moment_magnitude")
    direction = _direction_from_mapping(raw)
    quality = _optional_float_from_mapping(raw, "quality")
    _enforce_quality(quality, quality_threshold=quality_threshold)
    try:
        measured = measured_magnet_from_direction(
            moment,
            direction,
            nominal_magnitude=float(nominal_magnitude),
            quality=quality,
        )
    except ValueError as exc:
        raise MeasurementParseError(str(exc)) from exc
    magnet_id = _int_from_mapping(raw, "magnet_id", default_magnet_id)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=measured.error,
        measured_error=measured.error,
        quality=measured.quality,
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
        raise MeasurementParseError("serial measurement line must decode to an object")
    return {str(key): str(value) for key, value in raw.items()}


class FakeSerialMeasurementProvider:
    """Fake serial provider backed by JSON lines for tests and interface wiring."""

    def __init__(
        self,
        lines: Sequence[str | bytes],
        *,
        start_index: int = 0,
        nominal_magnitude: float = 1.0,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> None:
        self._lines = tuple(lines)
        self._position = 0
        self.nominal_magnitude = float(nominal_magnitude)
        self.quality_threshold = float(quality_threshold)
        self.set_position(start_index)

    @property
    def position(self) -> int:
        return self._position

    def set_position(self, position: int) -> None:
        pos = int(position)
        if pos < 0 or pos > len(self._lines):
            raise ValueError("provider position is out of range")
        self._position = pos

    def next_magnet(self) -> VirtualMagnet:
        if self._position >= len(self._lines):
            raise MeasurementTimeoutError("fake serial stream has no next line")
        magnet = parse_serial_measurement_line(
            self._lines[self._position],
            default_magnet_id=self._position,
            nominal_magnitude=self.nominal_magnitude,
            quality_threshold=self.quality_threshold,
        )
        self._position += 1
        return magnet


class SerialMeasurementProvider:
    """Real serial JSON-line provider using optional pyserial."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        timeout_s: float = 2.0,
        nominal_magnitude: float = 1.0,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        start_index: int = 0,
    ) -> None:
        try:
            serial_module = importlib.import_module("serial")
        except ModuleNotFoundError as exc:
            raise SerialDependencyError(
                "pyserial is required for SerialMeasurementProvider; install it before using real serial input"
            ) from exc
        serial_class = getattr(serial_module, "Serial", None)
        if serial_class is None:
            raise SerialDependencyError("imported serial module does not provide serial.Serial")
        self._serial: Any = serial_class(
            port=str(port),
            baudrate=int(baudrate),
            timeout=float(timeout_s),
        )
        self._position = int(start_index)
        if self._position < 0:
            raise ValueError("start_index must be >= 0")
        self.nominal_magnitude = float(nominal_magnitude)
        self.quality_threshold = float(quality_threshold)

    @property
    def position(self) -> int:
        return self._position

    def set_position(self, position: int) -> None:
        pos = int(position)
        if pos < 0:
            raise ValueError("provider position is out of range")
        self._position = pos

    def next_magnet(self) -> VirtualMagnet:
        raw_line = self._serial.readline()
        magnet = parse_serial_measurement_line(
            raw_line,
            default_magnet_id=self._position,
            nominal_magnitude=self.nominal_magnitude,
            quality_threshold=self.quality_threshold,
        )
        self._position += 1
        return magnet

    def close(self) -> None:
        close_fn = getattr(self._serial, "close", None)
        if close_fn is not None:
            close_fn()


__all__ = [
    "CsvMeasurementProvider",
    "MeasurementParseError",
    "FakeSerialMeasurementProvider",
    "ManualMeasurementProvider",
    "MeasurementProviderError",
    "MeasurementProvider",
    "MeasurementQualityError",
    "MeasurementTimeoutError",
    "SerialDependencyError",
    "SerialMeasurementProvider",
    "SyntheticMeasurementProvider",
    "measured_magnet_from_direction",
    "parse_serial_measurement_line",
    "virtual_magnet_from_mapping",
]
