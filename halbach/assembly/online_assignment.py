from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from halbach.assembly.inventory import decrement_cluster
from halbach.assembly.types import (
    ClusterAssignment,
    ClusterInventory,
    LinearAssignmentResult,
    LinearCandidate,
    MagnetError,
    Placement,
    SensitivityTable,
    VirtualMagnet,
)
from halbach.types import FloatArray


def _validate_table(table: SensitivityTable) -> None:
    if table.C.ndim != 4 or table.C.shape[3] != 3:
        raise ValueError("sensitivity C must have shape (S, O, residual_dim, 3)")
    if table.C.shape[0] != table.slot_flat_id.shape[0]:
        raise ValueError("sensitivity slot ids and C slot dimension mismatch")
    if table.C.shape[1] != len(table.orientation_id):
        raise ValueError("sensitivity orientation ids and C orientation dimension mismatch")
    if table.C.shape[2] <= 0:
        raise ValueError("sensitivity residual dimension must be positive")
    if len(set(table.orientation_id)) != len(table.orientation_id):
        raise ValueError("sensitivity orientation ids must be unique")
    if len(set(int(item) for item in table.slot_flat_id.tolist())) != table.slot_flat_id.size:
        raise ValueError("sensitivity slot ids must be unique")
    if not np.all(np.isfinite(table.C)):
        raise ValueError("sensitivity C must contain finite values")


def _error_vector(error: MagnetError) -> FloatArray:
    x = np.array(
        [
            float(error.epsilon_parallel),
            float(error.delta_perp_1),
            float(error.delta_perp_2),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(x)):
        raise ValueError("magnet error components must be finite")
    return x


def _assignment_cluster_map(
    assignments: Sequence[ClusterAssignment] | None,
    magnets: Sequence[VirtualMagnet],
) -> dict[int, str | None]:
    if assignments is None:
        return {magnet.magnet_id: None for magnet in magnets}
    if len(assignments) != len(magnets):
        raise ValueError("assignments length must match magnets length")
    magnet_ids = {magnet.magnet_id for magnet in magnets}
    seen: set[int] = set()
    cluster_by_magnet: dict[int, str | None] = {}
    for assignment in assignments:
        if assignment.magnet_id in seen:
            raise ValueError("assignments contain duplicate magnet_id values")
        seen.add(assignment.magnet_id)
        if assignment.magnet_id not in magnet_ids:
            raise ValueError(f"assignment references unknown magnet_id: {assignment.magnet_id}")
        if assignment.quarantine_id is not None:
            raise ValueError("quarantined magnets are not supported by Step 5 MVP assignment")
        if assignment.cluster_id is None:
            raise ValueError("normal assignment must include cluster_id")
        cluster_by_magnet[assignment.magnet_id] = assignment.cluster_id
    missing = sorted(magnet_ids - seen)
    if missing:
        raise ValueError(f"assignments missing magnet ids: {missing}")
    return cluster_by_magnet


def _slot_index(table: SensitivityTable) -> dict[int, int]:
    return {int(slot_id): idx for idx, slot_id in enumerate(table.slot_flat_id.tolist())}


def _orientation_index(table: SensitivityTable) -> dict[str, int]:
    return {orientation_id: idx for idx, orientation_id in enumerate(table.orientation_id)}


def score_linear_candidates(
    table: SensitivityTable,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> list[LinearCandidate]:
    """
    Score all remaining slot/orientation candidates.

    score = ||residual + C[s,o] @ x||^2 where x is the measured magnet error.
    """
    _validate_table(table)
    residual_arr = np.asarray(residual, dtype=np.float64)
    if residual_arr.shape != (table.C.shape[2],):
        raise ValueError(f"residual must have shape ({table.C.shape[2]},)")
    if not np.all(np.isfinite(residual_arr)):
        raise ValueError("residual must contain finite values")
    if not remaining_slot_flat_ids:
        raise ValueError("remaining_slot_flat_ids must be non-empty")

    slot_to_idx = _slot_index(table)
    orientation_to_idx = _orientation_index(table)
    orientation_ids = (
        tuple(table.orientation_id)
        if allowed_orientation_ids is None
        else tuple(allowed_orientation_ids)
    )
    if not orientation_ids:
        raise ValueError("allowed_orientation_ids must be non-empty")

    x = _error_vector(error)
    candidates: list[LinearCandidate] = []
    for slot_flat_id in remaining_slot_flat_ids:
        slot_key = int(slot_flat_id)
        if slot_key not in slot_to_idx:
            raise KeyError(f"unknown slot_flat_id: {slot_key}")
        slot_idx = slot_to_idx[slot_key]
        for orientation_id in orientation_ids:
            if orientation_id not in orientation_to_idx:
                raise KeyError(f"unknown orientation_id: {orientation_id}")
            orientation_idx = orientation_to_idx[orientation_id]
            contribution = np.ascontiguousarray(
                table.C[slot_idx, orientation_idx, :, :] @ x,
                dtype=np.float64,
            )
            updated = residual_arr + contribution
            score = float(np.dot(updated, updated))
            candidates.append(
                LinearCandidate(
                    slot_flat_id=slot_key,
                    orientation_id=orientation_id,
                    score=score,
                    contribution=contribution,
                )
            )
    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.slot_flat_id,
            table.orientation_id.index(candidate.orientation_id),
        )
    )
    return candidates


def choose_best_linear_candidate(
    table: SensitivityTable,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> LinearCandidate:
    """Return the minimum-score slot/orientation candidate."""
    return score_linear_candidates(
        table,
        residual,
        remaining_slot_flat_ids,
        error,
        allowed_orientation_ids=allowed_orientation_ids,
    )[0]


def run_linear_sensitivity_assignment(
    table: SensitivityTable,
    magnets: Sequence[VirtualMagnet],
    *,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
) -> LinearAssignmentResult:
    """
    Greedily place magnets using the Step 5 linear sensitivity MVP.

    Decisions use measured_error, while downstream fixed-field evaluation should still
    use true_error through the resulting Placement records.
    """
    _validate_table(table)
    if len(magnets) != table.slot_flat_id.shape[0]:
        raise ValueError("magnets count must match sensitivity slot count")
    magnet_by_id = {magnet.magnet_id: magnet for magnet in magnets}
    if len(magnet_by_id) != len(magnets):
        raise ValueError("magnet_id values must be unique")

    if magnet_order is None:
        ordered_magnets = list(magnets)
    else:
        if len(magnet_order) != len(magnets):
            raise ValueError("magnet_order length must match magnets length")
        if len(set(int(item) for item in magnet_order)) != len(magnet_order):
            raise ValueError("magnet_order contains duplicate magnet ids")
        unknown = sorted(set(int(item) for item in magnet_order) - set(magnet_by_id))
        if unknown:
            raise ValueError(f"magnet_order references unknown magnet ids: {unknown}")
        ordered_magnets = [magnet_by_id[int(magnet_id)] for magnet_id in magnet_order]

    cluster_by_magnet = _assignment_cluster_map(assignments, magnets)
    residual = np.zeros(table.C.shape[2], dtype=np.float64)
    remaining_slot_ids = [int(slot_id) for slot_id in table.slot_flat_id.tolist()]
    placements: list[Placement] = []
    current_inventory = inventory

    for insert_order, magnet in enumerate(ordered_magnets):
        candidate = choose_best_linear_candidate(
            table,
            residual,
            remaining_slot_ids,
            magnet.measured_error,
            allowed_orientation_ids=allowed_orientation_ids,
        )
        cluster_id = cluster_by_magnet.get(magnet.magnet_id)
        if current_inventory is not None and cluster_id is not None:
            current_inventory = decrement_cluster(current_inventory, cluster_id)
        residual = np.ascontiguousarray(residual + candidate.contribution, dtype=np.float64)
        remaining_slot_ids.remove(candidate.slot_flat_id)
        placements.append(
            Placement(
                slot_flat_id=candidate.slot_flat_id,
                magnet_id=magnet.magnet_id,
                orientation_id=candidate.orientation_id,
                cluster_requested=cluster_id,
                insert_order=insert_order,
                decision_engine="linear_sensitivity",
            )
        )

    linear_score = float(np.dot(residual, residual))
    if not math.isfinite(linear_score):
        raise ValueError("linear assignment produced a non-finite score")
    return LinearAssignmentResult(
        placements=tuple(placements),
        residual=residual,
        linear_score=linear_score,
        remaining_slot_flat_ids=tuple(remaining_slot_ids),
        inventory=current_inventory,
    )


def cluster_usage_from_placements(placements: Sequence[Placement]) -> dict[str, int]:
    """Count cluster_requested values in a completed placement sequence."""
    usage: dict[str, int] = {}
    for placement in placements:
        if placement.cluster_requested is None:
            continue
        usage[placement.cluster_requested] = usage.get(placement.cluster_requested, 0) + 1
    return dict(sorted(usage.items()))


def planned_cluster_counts(assignments: Sequence[ClusterAssignment]) -> dict[str, int]:
    """Count normal cluster assignments."""
    counts: dict[str, int] = {}
    for assignment in assignments:
        if assignment.cluster_id is None:
            continue
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "choose_best_linear_candidate",
    "cluster_usage_from_placements",
    "planned_cluster_counts",
    "run_linear_sensitivity_assignment",
    "score_linear_candidates",
]
