from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from halbach.assembly.orientations import rotate_error_for_orientation
from halbach.assembly.types import (
    AssemblySlot,
    FieldEvaluation,
    FieldMetrics,
    MagnetError,
    Placement,
    VirtualMagnet,
)
from halbach.constants import FACTOR, m0
from halbach.physics import compute_B_and_B0_from_m_flat
from halbach.types import FloatArray


def _as_vector3(value: FloatArray, label: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{label} must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_points(value: FloatArray) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("pts must have shape (P, 3)")
    if arr.shape[0] == 0:
        raise ValueError("pts must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pts must contain finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _unit_vector(value: FloatArray, label: str) -> FloatArray:
    arr = _as_vector3(value, label)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{label} must be non-zero")
    return np.ascontiguousarray(arr / norm, dtype=np.float64)


def _local_transverse_basis(nominal_u: FloatArray) -> tuple[FloatArray, FloatArray]:
    u = _unit_vector(nominal_u, "nominal_u")
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(u, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1_raw = np.cross(reference, u)
    e1_norm = float(np.linalg.norm(e1_raw))
    if e1_norm <= 0.0:
        raise ValueError("could not construct transverse basis")
    e1 = np.ascontiguousarray(e1_raw / e1_norm, dtype=np.float64)
    e2 = np.ascontiguousarray(np.cross(u, e1), dtype=np.float64)
    return e1, e2


def _moment_vector_from_error(slot: AssemblySlot, error: MagnetError) -> FloatArray:
    scale = 1.0 + float(error.epsilon_parallel)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("moment magnitude scale must be finite and positive")
    if not math.isfinite(error.delta_perp_1) or not math.isfinite(error.delta_perp_2):
        raise ValueError("transverse error components must be finite")

    u = _unit_vector(slot.nominal_u, "slot.nominal_u")
    e1, e2 = _local_transverse_basis(u)
    direction_raw = (
        u
        + float(error.delta_perp_1) * e1
        + float(error.delta_perp_2) * e2
    )
    direction = _unit_vector(direction_raw, "perturbed magnet direction")
    return np.ascontiguousarray(float(m0) * scale * direction, dtype=np.float64)


def moment_vector_from_error(slot: AssemblySlot, error: MagnetError) -> FloatArray:
    """
    Convert a Plan C local magnet error into a dipole moment vector.

    The returned vector has shape (3,) and uses the same arbitrary dipole moment
    units as the existing fixed dipole model.
    """
    return _moment_vector_from_error(slot, error)


def _slot_dict(slots: Sequence[AssemblySlot]) -> dict[int, AssemblySlot]:
    if not slots:
        raise ValueError("slots must be non-empty")
    by_id: dict[int, AssemblySlot] = {}
    for slot in slots:
        if slot.slot_flat_id in by_id:
            raise ValueError(f"duplicate slot_flat_id: {slot.slot_flat_id}")
        by_id[slot.slot_flat_id] = slot
    return by_id


def _magnet_dict(magnets: Sequence[VirtualMagnet]) -> dict[int, VirtualMagnet]:
    if not magnets:
        raise ValueError("magnets must be non-empty")
    by_id: dict[int, VirtualMagnet] = {}
    for magnet in magnets:
        if magnet.magnet_id in by_id:
            raise ValueError(f"duplicate magnet_id: {magnet.magnet_id}")
        by_id[magnet.magnet_id] = magnet
    return by_id


def _validate_placements(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> tuple[dict[int, AssemblySlot], dict[int, VirtualMagnet]]:
    slot_by_id = _slot_dict(slots)
    magnet_by_id = _magnet_dict(magnets)
    if len(placements) != len(slot_by_id):
        raise ValueError("placements must contain exactly one assignment for every slot")

    placement_slot_ids = [placement.slot_flat_id for placement in placements]
    if len(placement_slot_ids) != len(set(placement_slot_ids)):
        raise ValueError("placements contain duplicate slot_flat_id values")
    unknown_slots = sorted(set(placement_slot_ids) - set(slot_by_id))
    missing_slots = sorted(set(slot_by_id) - set(placement_slot_ids))
    if unknown_slots or missing_slots:
        raise ValueError(
            f"placement slot coverage mismatch; unknown={unknown_slots}, missing={missing_slots}"
        )

    placement_magnet_ids = [placement.magnet_id for placement in placements]
    if len(placement_magnet_ids) != len(set(placement_magnet_ids)):
        raise ValueError("placements contain duplicate magnet_id values")
    unknown_magnets = sorted(set(placement_magnet_ids) - set(magnet_by_id))
    if unknown_magnets:
        raise ValueError(f"placements reference unknown magnet ids: {unknown_magnets}")
    return slot_by_id, magnet_by_id


def build_fixed_model_arrays(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> tuple[FloatArray, FloatArray]:
    """
    Build fixed dipole arrays from Plan C placement data.

    Returns:
        r0_flat: shape (M, 3), units m, magnet centers sorted by slot_flat_id
        m_flat: shape (M, 3), dipole moment vectors including true magnet error
    """
    slot_by_id, magnet_by_id = _validate_placements(slots, magnets, placements)
    ordered = sorted(placements, key=lambda placement: placement.slot_flat_id)
    r0_flat = np.empty((len(ordered), 3), dtype=np.float64)
    m_flat = np.empty((len(ordered), 3), dtype=np.float64)

    for row, placement in enumerate(ordered):
        slot = slot_by_id[placement.slot_flat_id]
        magnet = magnet_by_id[placement.magnet_id]
        true_error = rotate_error_for_orientation(
            magnet.true_error,
            placement.orientation_id,
        )
        r0_flat[row, :] = _as_vector3(slot.center_m, "slot.center_m")
        m_flat[row, :] = _moment_vector_from_error(slot, true_error)
    return (
        np.ascontiguousarray(r0_flat, dtype=np.float64),
        np.ascontiguousarray(m_flat, dtype=np.float64),
    )


def compute_field_metrics(
    B: FloatArray,
    B0: FloatArray,
    *,
    min_B0_norm: float = 1e-18,
) -> FieldMetrics:
    """Compute Plan C vector homogeneity metrics from B and B0."""
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2 or B_arr.shape[1] != 3 or B_arr.shape[0] == 0:
        raise ValueError("B must have shape (P, 3) with P >= 1")
    if not np.all(np.isfinite(B_arr)):
        raise ValueError("B must contain finite values")
    B0_arr = _as_vector3(B0, "B0")
    if not math.isfinite(min_B0_norm) or min_B0_norm < 0.0:
        raise ValueError("min_B0_norm must be finite and >= 0")

    B0_norm = float(np.linalg.norm(B0_arr))
    if B0_norm <= min_B0_norm:
        raise ValueError(f"B0_norm is too small for homogeneity metrics: {B0_norm}")

    diff = B_arr - B0_arr[None, :]
    diff_sq = np.sum(diff * diff, axis=1)
    diff_norm = np.sqrt(diff_sq)
    ppm = diff_norm / B0_norm * 1.0e6
    return FieldMetrics(
        rms_homogeneity_ppm=float(math.sqrt(float(np.mean(diff_sq))) / B0_norm * 1.0e6),
        max_homogeneity_ppm=float(np.max(ppm)),
        p95_homogeneity_ppm=float(np.percentile(ppm, 95.0)),
        p99_homogeneity_ppm=float(np.percentile(ppm, 99.0)),
        B0_norm=B0_norm,
        J_vector=float(np.mean(diff_sq)),
    )


def evaluate_fixed_placement(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    pts: FloatArray,
    *,
    factor: float = FACTOR,
    origin: FloatArray | None = None,
    min_B0_norm: float = 1e-18,
) -> FieldEvaluation:
    """
    Evaluate ROI field homogeneity for a fixed Plan C placement.

    pts: shape (P, 3), units m, ROI points relative to origin
    origin: shape (3,), units m; defaults to the design origin
    """
    pts_arr = _as_points(pts)
    origin_arr = (
        np.zeros(3, dtype=np.float64)
        if origin is None
        else _as_vector3(origin, "origin")
    )
    r0_flat, m_flat = build_fixed_model_arrays(slots, magnets, placements)
    B, B0 = compute_B_and_B0_from_m_flat(
        pts_arr,
        r0_flat,
        m_flat,
        float(factor),
        np.ascontiguousarray(origin_arr, dtype=np.float64),
    )
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    B0_arr = np.ascontiguousarray(B0, dtype=np.float64)
    metrics = compute_field_metrics(B_arr, B0_arr, min_B0_norm=min_B0_norm)
    return FieldEvaluation(pts=pts_arr, B=B_arr, B0=B0_arr, metrics=metrics)


__all__ = [
    "build_fixed_model_arrays",
    "compute_field_metrics",
    "evaluate_fixed_placement",
    "moment_vector_from_error",
]
