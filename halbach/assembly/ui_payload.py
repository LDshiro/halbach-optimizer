from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from halbach.assembly.orientations import default_orientations
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    ClusterInventory,
    FieldMetrics,
    MagnetError,
)
from halbach.assembly.session import SessionSnapshot


def _error_dict(error: MagnetError) -> dict[str, float]:
    return {
        "epsilon_parallel": error.epsilon_parallel,
        "delta_perp_1": error.delta_perp_1,
        "delta_perp_2": error.delta_perp_2,
    }


def _metrics_dict(metrics: FieldMetrics | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {
        "rms_homogeneity_ppm": metrics.rms_homogeneity_ppm,
        "max_homogeneity_ppm": metrics.max_homogeneity_ppm,
        "p95_homogeneity_ppm": metrics.p95_homogeneity_ppm,
        "p99_homogeneity_ppm": metrics.p99_homogeneity_ppm,
        "B0_norm": metrics.B0_norm,
        "J_vector": metrics.J_vector,
    }


def _slot_by_id(slots: Sequence[AssemblySlot]) -> dict[int, AssemblySlot]:
    by_id: dict[int, AssemblySlot] = {}
    for slot in slots:
        if slot.slot_flat_id in by_id:
            raise ValueError(f"duplicate slot_flat_id: {slot.slot_flat_id}")
        by_id[slot.slot_flat_id] = slot
    return by_id


def _assignment_map(assignments: Sequence[ClusterAssignment] | None) -> dict[int, str | None]:
    if assignments is None:
        return {}
    mapping: dict[int, str | None] = {}
    for assignment in assignments:
        mapping[assignment.magnet_id] = assignment.cluster_id
    return mapping


def _cluster_inventory_payload(inventory: ClusterInventory | None) -> dict[str, object]:
    if inventory is None:
        return {
            "clusters": {},
            "quarantine": {},
            "total_available": None,
            "quarantine_count": 0,
        }
    clusters = {
        cluster_id: {
            "count": stats.count,
            "mean": stats.mean.tolist(),
        }
        for cluster_id, stats in sorted(inventory.clusters.items())
    }
    quarantine = dict(sorted(inventory.quarantine.items()))
    return {
        "clusters": clusters,
        "quarantine": quarantine,
        "total_available": int(
            sum(stats.count for stats in inventory.clusters.values())
            + sum(inventory.quarantine.values())
        ),
        "quarantine_count": int(sum(inventory.quarantine.values())),
    }


def _orientation_instruction(orientation_id: str | None) -> str | None:
    if orientation_id is None:
        return None
    for orientation in default_orientations():
        if orientation.id == orientation_id:
            return orientation.instruction
    return None


def _current_slot(
    snapshot: SessionSnapshot,
    slot_map: Mapping[int, AssemblySlot],
) -> AssemblySlot | None:
    if snapshot.pending_candidate is not None:
        return slot_map.get(snapshot.pending_candidate.slot_flat_id)
    if snapshot.placements:
        return slot_map.get(snapshot.placements[-1].slot_flat_id)
    if snapshot.remaining_slot_flat_ids:
        return slot_map.get(snapshot.remaining_slot_flat_ids[0])
    return None


def build_slot_display_rows(
    snapshot: SessionSnapshot,
    slots: Sequence[AssemblySlot],
) -> list[dict[str, object]]:
    """
    Build ring/slot display rows with occupied/empty/recommended states.

    The rows are intentionally plain dictionaries so Streamlit can render them as
    a table without importing domain internals.
    """
    occupied = {placement.slot_flat_id for placement in snapshot.placements}
    recommended = (
        None
        if snapshot.pending_candidate is None
        else snapshot.pending_candidate.slot_flat_id
    )
    rows: list[dict[str, object]] = []
    for slot in sorted(slots, key=lambda item: (item.layer_id, item.ring_id, item.theta_id)):
        if slot.slot_flat_id == recommended:
            state = "recommended"
        elif slot.slot_flat_id in occupied:
            state = "occupied"
        else:
            state = "empty"
        rows.append(
            {
                "slot_flat_id": slot.slot_flat_id,
                "ring_id": slot.ring_id,
                "layer_id": slot.layer_id,
                "theta_id": slot.theta_id,
                "physical_slot_number": slot.physical_slot_number,
                "work_unit_id": slot.work_unit_id,
                "mirror_pair_id": slot.mirror_pair_id,
                "state": state,
                "highlight": state == "recommended",
            }
        )
    return rows


def build_session_ui_payload(
    snapshot: SessionSnapshot,
    slots: Sequence[AssemblySlot],
    *,
    mode: str = "simulation_step_by_step",
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    metrics: FieldMetrics | None = None,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Build the minimal payload consumed by Plan C Streamlit pages."""
    slot_map = _slot_by_id(slots)
    current_slot = _current_slot(snapshot, slot_map)
    assignment_by_magnet = _assignment_map(assignments)
    pending_magnet = snapshot.pending_magnet
    recommended_slot_id = (
        None
        if snapshot.pending_candidate is None
        else snapshot.pending_candidate.slot_flat_id
    )
    recommended_orientation_id = (
        None
        if snapshot.pending_candidate is None
        else snapshot.pending_candidate.orientation_id
    )
    cluster_id = (
        None
        if pending_magnet is None
        else assignment_by_magnet.get(pending_magnet.magnet_id)
    )
    residual_norm = float(np.linalg.norm(snapshot.residual))
    log_saved = False if log_path is None else Path(log_path).exists()
    slot_rows = build_slot_display_rows(snapshot, slots)

    return {
        "mode": mode,
        "state": snapshot.state,
        "sub_state": snapshot.sub_state,
        "current_work_unit_id": None if current_slot is None else current_slot.work_unit_id,
        "current_mirror_pair_id": None if current_slot is None else current_slot.mirror_pair_id,
        "current_ring_id": None if current_slot is None else current_slot.ring_id,
        "remaining_slot_count": len(snapshot.remaining_slot_flat_ids),
        "placed_count": len(snapshot.placements),
        "next_cluster_id": cluster_id,
        "cluster_inventory": _cluster_inventory_payload(inventory),
        "measurement": (
            None
            if pending_magnet is None
            else {
                "magnet_id": pending_magnet.magnet_id,
                "measured_error": _error_dict(pending_magnet.measured_error),
                "quality": pending_magnet.quality,
            }
        ),
        "recommended_ring_id": None if current_slot is None else current_slot.ring_id,
        "recommended_slot_flat_id": recommended_slot_id,
        "recommended_physical_slot_number": (
            None if current_slot is None else current_slot.physical_slot_number
        ),
        "recommended_orientation_id": recommended_orientation_id,
        "orientation_instruction": _orientation_instruction(recommended_orientation_id),
        "predicted_linear_score": (
            None if snapshot.pending_candidate is None else snapshot.pending_candidate.score
        ),
        "residual_norm": residual_norm,
        "current_metrics": _metrics_dict(metrics),
        "quarantine_count": _cluster_inventory_payload(inventory)["quarantine_count"],
        "log_saved": log_saved,
        "recommended_highlight_slot_ids": (
            [] if recommended_slot_id is None else [recommended_slot_id]
        ),
        "slot_rows": slot_rows,
    }


def build_summary_ui_payload(summary: dict[str, object]) -> dict[str, object]:
    """Normalize simulation_auto_run summary JSON for display."""
    metadata = summary.get("metadata", {})
    summary_block = summary.get("summary", {})
    trials = summary.get("trials", [])
    return {
        "schema_version": summary.get("schema_version"),
        "engine": metadata.get("engine") if isinstance(metadata, dict) else None,
        "trials": summary_block.get("trials") if isinstance(summary_block, dict) else None,
        "rms_ratio_mean": (
            summary_block.get("rms_ratio_mean") if isinstance(summary_block, dict) else None
        ),
        "linear_improved_count": (
            summary_block.get("linear_improved_count")
            if isinstance(summary_block, dict)
            else None
        ),
        "trial_rows": trials if isinstance(trials, list) else [],
    }


__all__ = [
    "build_session_ui_payload",
    "build_slot_display_rows",
    "build_summary_ui_payload",
]
