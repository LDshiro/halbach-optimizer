from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from halbach.assembly.field_eval import moment_vector_from_error
from halbach.assembly.orientations import default_orientations, rotate_error_for_orientation
from halbach.assembly.types import (
    AssemblySlot,
    IntArray,
    MagnetError,
    OrientationCandidate,
    SensitivityTable,
)
from halbach.constants import FACTOR
from halbach.physics import compute_B_and_B0_from_m_flat
from halbach.types import FloatArray

SENSITIVITY_SCHEMA_VERSION = 1


def _as_points(value: FloatArray) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("roi_points must have shape (P, 3)")
    if arr.shape[0] == 0:
        raise ValueError("roi_points must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("roi_points must contain finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_vector3(value: FloatArray, label: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{label} must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _orientation_ids(orientations: Sequence[OrientationCandidate]) -> tuple[str, ...]:
    if not orientations:
        raise ValueError("orientations must be non-empty")
    ids = tuple(orientation.id for orientation in orientations)
    if len(ids) != len(set(ids)):
        raise ValueError("orientation ids must be unique")
    return ids


def _component_error(component: int, value: float) -> MagnetError:
    if component == 0:
        return MagnetError(
            epsilon_parallel=value,
            delta_perp_1=0.0,
            delta_perp_2=0.0,
        )
    if component == 1:
        return MagnetError(
            epsilon_parallel=0.0,
            delta_perp_1=value,
            delta_perp_2=0.0,
        )
    if component == 2:
        return MagnetError(
            epsilon_parallel=0.0,
            delta_perp_1=0.0,
            delta_perp_2=value,
        )
    raise ValueError(f"unsupported component index: {component}")


def _nominal_model_arrays(slots: Sequence[AssemblySlot]) -> tuple[FloatArray, FloatArray]:
    if not slots:
        raise ValueError("slots must be non-empty")
    slot_ids = [slot.slot_flat_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")

    zero = MagnetError(
        epsilon_parallel=0.0,
        delta_perp_1=0.0,
        delta_perp_2=0.0,
    )
    r0_flat = np.empty((len(slots), 3), dtype=np.float64)
    m_flat = np.empty((len(slots), 3), dtype=np.float64)
    for idx, slot in enumerate(slots):
        r0_flat[idx, :] = _as_vector3(slot.center_m, "slot.center_m")
        m_flat[idx, :] = moment_vector_from_error(slot, zero)
    return (
        np.ascontiguousarray(r0_flat, dtype=np.float64),
        np.ascontiguousarray(m_flat, dtype=np.float64),
    )


def flatten_field_residual(B: FloatArray, B0: FloatArray) -> FloatArray:
    """Return flattened y = [B(p_1)-B0, ..., B(p_M)-B0]."""
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2 or B_arr.shape[1] != 3 or B_arr.shape[0] == 0:
        raise ValueError("B must have shape (P, 3) with P >= 1")
    B0_arr = _as_vector3(B0, "B0")
    return np.ascontiguousarray((B_arr - B0_arr[None, :]).reshape(-1), dtype=np.float64)


def _slot_field_residual_from_dm(
    slot: AssemblySlot,
    dm: FloatArray,
    roi_points: FloatArray,
    *,
    factor: float,
    origin: FloatArray,
) -> FloatArray:
    r0_one = np.ascontiguousarray(_as_vector3(slot.center_m, "slot.center_m")[None, :])
    m_one = np.ascontiguousarray(_as_vector3(dm, "dm")[None, :])
    B, B0 = compute_B_and_B0_from_m_flat(
        roi_points,
        r0_one,
        m_one,
        float(factor),
        origin,
    )
    return flatten_field_residual(B, B0)


def compute_sensitivity_table(
    slots: Sequence[AssemblySlot],
    roi_points: FloatArray,
    *,
    orientations: Sequence[OrientationCandidate] | None = None,
    finite_difference_step: float = 1e-6,
    factor: float = FACTOR,
    origin: FloatArray | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SensitivityTable:
    """
    Compute fixed-model Plan C linear sensitivity.

    C has shape (S, O, 3*P, 3) and maps a magnet error vector
    [epsilon_parallel, delta_perp_1, delta_perp_2] to the flattened ROI residual
    y = [B(p_1)-B0, ..., B(p_P)-B0].
    """
    if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    if not math.isfinite(factor):
        raise ValueError("factor must be finite")

    if not slots:
        raise ValueError("slots must be non-empty")
    slot_ids = [slot.slot_flat_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")

    roi = _as_points(roi_points)
    origin_arr = (
        np.zeros(3, dtype=np.float64)
        if origin is None
        else _as_vector3(origin, "origin")
    )
    origin_arr = np.ascontiguousarray(origin_arr, dtype=np.float64)

    orientation_items = tuple(default_orientations() if orientations is None else orientations)
    orientation_ids = _orientation_ids(orientation_items)
    residual_dim = int(roi.shape[0] * 3)
    C = np.empty((len(slots), len(orientation_items), residual_dim, 3), dtype=np.float64)

    r0_nominal, m_nominal = _nominal_model_arrays(slots)
    _B_nominal, normalization_b0 = compute_B_and_B0_from_m_flat(
        roi,
        r0_nominal,
        m_nominal,
        float(factor),
        origin_arr,
    )
    normalization_b0 = np.ascontiguousarray(normalization_b0, dtype=np.float64)

    step = float(finite_difference_step)
    for slot_idx, slot in enumerate(slots):
        for orientation_idx, orientation in enumerate(orientation_items):
            for component in range(3):
                err_plus = rotate_error_for_orientation(
                    _component_error(component, step),
                    orientation,
                )
                err_minus = rotate_error_for_orientation(
                    _component_error(component, -step),
                    orientation,
                )
                m_plus = moment_vector_from_error(slot, err_plus)
                m_minus = moment_vector_from_error(slot, err_minus)
                dm = np.ascontiguousarray((m_plus - m_minus) / (2.0 * step), dtype=np.float64)
                C[slot_idx, orientation_idx, :, component] = _slot_field_residual_from_dm(
                    slot,
                    dm,
                    roi,
                    factor=float(factor),
                    origin=origin_arr,
                )

    base_metadata: dict[str, object] = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "n_slots": len(slots),
        "n_orientations": len(orientation_items),
        "n_roi_points": int(roi.shape[0]),
        "residual_dim": residual_dim,
        "residual_order": "point_major_xyz",
        "finite_difference_step": step,
        "factor": float(factor),
        "projection": "none",
    }
    if metadata is not None:
        base_metadata.update(dict(metadata))

    return SensitivityTable(
        slot_flat_id=np.asarray([slot.slot_flat_id for slot in slots], dtype=np.int_),
        ring_id=np.asarray([slot.ring_id for slot in slots], dtype=np.int_),
        layer_id=np.asarray([slot.layer_id for slot in slots], dtype=np.int_),
        theta_id=np.asarray([slot.theta_id for slot in slots], dtype=np.int_),
        centers_m=np.ascontiguousarray(
            np.vstack([_as_vector3(slot.center_m, "slot.center_m") for slot in slots]),
            dtype=np.float64,
        ),
        nominal_u=np.ascontiguousarray(
            np.vstack([_as_vector3(slot.nominal_u, "slot.nominal_u") for slot in slots]),
            dtype=np.float64,
        ),
        orientation_id=orientation_ids,
        C=np.ascontiguousarray(C, dtype=np.float64),
        roi_points=roi,
        normalization_b0=normalization_b0,
        metadata=base_metadata,
        projection_basis=None,
    )


def _validate_sensitivity_table(table: SensitivityTable) -> None:
    n_slots = int(table.slot_flat_id.shape[0])
    if n_slots == 0:
        raise ValueError("sensitivity table must contain at least one slot")
    if len(table.orientation_id) == 0:
        raise ValueError("sensitivity table must contain at least one orientation")
    if len(set(table.orientation_id)) != len(table.orientation_id):
        raise ValueError("orientation ids must be unique")

    for label, arr in (
        ("ring_id", table.ring_id),
        ("layer_id", table.layer_id),
        ("theta_id", table.theta_id),
    ):
        if arr.shape != (n_slots,):
            raise ValueError(f"{label} must have shape ({n_slots},)")
    if table.centers_m.shape != (n_slots, 3):
        raise ValueError("centers_m must have shape (S, 3)")
    if table.nominal_u.shape != (n_slots, 3):
        raise ValueError("nominal_u must have shape (S, 3)")

    roi = _as_points(table.roi_points)
    residual_dim = int(roi.shape[0] * 3)
    expected_C_shape = (n_slots, len(table.orientation_id), residual_dim, 3)
    if table.C.shape != expected_C_shape:
        raise ValueError(f"C must have shape {expected_C_shape}")
    if not np.all(np.isfinite(table.C)):
        raise ValueError("C must contain finite values")
    _as_vector3(table.normalization_b0, "normalization_b0")
    if table.projection_basis is not None:
        basis = np.asarray(table.projection_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != residual_dim:
            raise ValueError("projection_basis must have shape (D, residual_dim)")


def save_sensitivity_table(path: str | Path, table: SensitivityTable) -> None:
    """Write a Plan C sensitivity table to an NPZ file."""
    _validate_sensitivity_table(table)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residual_dim = int(table.roi_points.shape[0] * 3)
    projection_basis = (
        np.zeros((0, residual_dim), dtype=np.float64)
        if table.projection_basis is None
        else np.asarray(table.projection_basis, dtype=np.float64)
    )
    metadata_json = np.array(json.dumps(table.metadata, sort_keys=True))
    np.savez_compressed(
        out_path,
        slot_flat_id=np.asarray(table.slot_flat_id, dtype=np.int_),
        ring_id=np.asarray(table.ring_id, dtype=np.int_),
        layer_id=np.asarray(table.layer_id, dtype=np.int_),
        theta_id=np.asarray(table.theta_id, dtype=np.int_),
        centers_m=np.asarray(table.centers_m, dtype=np.float64),
        nominal_u=np.asarray(table.nominal_u, dtype=np.float64),
        orientation_id=np.asarray(table.orientation_id),
        C=np.asarray(table.C, dtype=np.float64),
        roi_points=np.asarray(table.roi_points, dtype=np.float64),
        normalization_B0=np.asarray(table.normalization_b0, dtype=np.float64),
        projection_basis=projection_basis,
        metadata_json=metadata_json,
    )


def _metadata_from_json(value: object) -> dict[str, object]:
    raw = str(np.asarray(value).item())
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("metadata_json must decode to an object")
    return cast(dict[str, object], loaded)


def load_sensitivity_table(path: str | Path) -> SensitivityTable:
    """Read a Plan C sensitivity table from an NPZ file."""
    required = {
        "slot_flat_id",
        "ring_id",
        "layer_id",
        "theta_id",
        "centers_m",
        "nominal_u",
        "orientation_id",
        "C",
        "roi_points",
        "normalization_B0",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"sensitivity file is missing keys: {missing}")
        projection_basis = (
            np.asarray(data["projection_basis"], dtype=np.float64)
            if "projection_basis" in data.files
            else np.zeros((0, int(np.asarray(data["C"]).shape[2])), dtype=np.float64)
        )
        projection_or_none = (
            None
            if projection_basis.size == 0
            else np.ascontiguousarray(projection_basis, dtype=np.float64)
        )
        table = SensitivityTable(
            slot_flat_id=cast(IntArray, np.asarray(data["slot_flat_id"], dtype=np.int_)),
            ring_id=cast(IntArray, np.asarray(data["ring_id"], dtype=np.int_)),
            layer_id=cast(IntArray, np.asarray(data["layer_id"], dtype=np.int_)),
            theta_id=cast(IntArray, np.asarray(data["theta_id"], dtype=np.int_)),
            centers_m=cast(FloatArray, np.asarray(data["centers_m"], dtype=np.float64)),
            nominal_u=cast(FloatArray, np.asarray(data["nominal_u"], dtype=np.float64)),
            orientation_id=tuple(str(item) for item in np.asarray(data["orientation_id"]).tolist()),
            C=cast(FloatArray, np.asarray(data["C"], dtype=np.float64)),
            roi_points=cast(FloatArray, np.asarray(data["roi_points"], dtype=np.float64)),
            normalization_b0=cast(
                FloatArray,
                np.asarray(data["normalization_B0"], dtype=np.float64),
            ),
            metadata=_metadata_from_json(data["metadata_json"]),
            projection_basis=projection_or_none,
        )
    _validate_sensitivity_table(table)
    return table


def sensitivity_contribution(
    table: SensitivityTable,
    slot_flat_id: int,
    orientation_id: str,
    error: MagnetError,
) -> FloatArray:
    """Return C_{slot,orientation} @ [epsilon, delta1, delta2]."""
    _validate_sensitivity_table(table)
    slot_matches = np.flatnonzero(table.slot_flat_id == int(slot_flat_id))
    if slot_matches.size != 1:
        raise KeyError(f"unknown slot_flat_id: {slot_flat_id}")
    try:
        orientation_idx = table.orientation_id.index(orientation_id)
    except ValueError as exc:
        raise KeyError(f"unknown orientation_id: {orientation_id}") from exc

    x = np.array(
        [
            float(error.epsilon_parallel),
            float(error.delta_perp_1),
            float(error.delta_perp_2),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(x)):
        raise ValueError("error components must be finite")
    slot_idx = int(slot_matches[0])
    contribution = table.C[slot_idx, orientation_idx, :, :] @ x
    return np.ascontiguousarray(contribution, dtype=np.float64)


__all__ = [
    "SENSITIVITY_SCHEMA_VERSION",
    "compute_sensitivity_table",
    "flatten_field_residual",
    "load_sensitivity_table",
    "save_sensitivity_table",
    "sensitivity_contribution",
]
