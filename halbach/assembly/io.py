from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from halbach.assembly.orientations import default_orientations
from halbach.assembly.ring_summary import (
    ring_pair_summary_from_ring_summaries,
    ring_summary_from_placements,
    timeline_from_placements,
)
from halbach.assembly.types import (
    AssemblySlot,
    AssemblyTimelineEvent,
    ClusterAssignment,
    FieldMetrics,
    Placement,
    RingKey,
    RingPairSummary,
    RingQuotaPlan,
    RingSummary,
    SimulationComparisonResult,
    VirtualMagnet,
)

SCHEMA_VERSION = 1

FINAL_PLACEMENT_COLUMNS = [
    "work_unit_id",
    "mirror_pair_id",
    "installed_layer_id",
    "installed_side",
    "ring_id",
    "layer_id",
    "theta_id",
    "slot_flat_id",
    "physical_slot_number",
    "cluster_requested",
    "epsilon_parallel",
    "delta_perp_1",
    "delta_perp_2",
    "orientation_id",
    "orientation_angle_deg",
    "insert_order",
    "decision_engine",
]

SIMULATION_TRIAL_COLUMNS = [
    "trial_id",
    "seed",
    "random_rms_homogeneity_ppm",
    "linear_rms_homogeneity_ppm",
    "random_max_homogeneity_ppm",
    "linear_max_homogeneity_ppm",
    "random_p95_homogeneity_ppm",
    "linear_p95_homogeneity_ppm",
    "random_p99_homogeneity_ppm",
    "linear_p99_homogeneity_ppm",
    "random_B0_norm",
    "linear_B0_norm",
    "random_J_vector",
    "linear_J_vector",
    "rms_ratio_linear_over_random",
    "j_ratio_linear_over_random",
    "linear_score_measured",
]

RING_SUMMARY_COLUMNS = [
    "trial_id",
    "work_unit_id",
    "layer_id",
    "ring_id",
    "count",
    "mean_epsilon",
    "std_epsilon",
    "min_epsilon",
    "max_epsilon",
    "mean_angle_error",
    "std_angle_error",
    "mean_true_epsilon",
    "mean_true_delta_perp_1",
    "mean_true_delta_perp_2",
    "mean_measured_epsilon",
    "mean_measured_delta_perp_1",
    "mean_measured_delta_perp_2",
    "cluster_counts_json",
    "B0_norm_after_ring",
    "rms_homogeneity_ppm_after_ring",
    "J_vector_after_ring",
]

RING_PAIR_SUMMARY_COLUMNS = [
    "trial_id",
    "pair_id",
    "pair_index",
    "ring_id",
    "lower_layer_id",
    "upper_layer_id",
    "lower_count",
    "upper_count",
    "lower_mean_epsilon",
    "upper_mean_epsilon",
    "mean_epsilon_difference",
    "mean_angle_error_difference",
    "residual_norm_after_lower",
    "residual_norm_after_upper",
    "residual_norm_after_pair",
    "pair_complete",
]

CLUSTER_QUOTA_PLAN_COLUMNS = [
    "trial_id",
    "work_unit_id",
    "layer_id",
    "ring_id",
    "target_count",
    "target_mean_epsilon",
    "ring_importance",
    "expected_mean_epsilon",
    "expected_mean_angle_bin",
    "expected_angle_error",
    "mirror_pair_id",
    "allowed_clusters_json",
    "quota_by_cluster_json",
]

CLUSTER_PICKUP_LOG_COLUMNS = [
    "trial_id",
    "insert_order",
    "decision_engine",
    "work_unit_id",
    "layer_id",
    "ring_id",
    "theta_id",
    "slot_flat_id",
    "physical_slot_number",
    "center_x_m",
    "center_y_m",
    "nominal_phi_rad",
    "magnet_id",
    "assignment_cluster_id",
    "cluster_requested",
    "epsilon_parallel",
    "delta_perp_1",
    "delta_perp_2",
    "measured_epsilon_parallel",
    "measured_delta_perp_1",
    "measured_delta_perp_2",
    "orientation_id",
]


@dataclass(frozen=True)
class SimulationTrialArtifacts:
    """All per-trial data needed for Plan C simulation output files."""

    trial_id: int
    seed: int
    result: SimulationComparisonResult
    magnets: Sequence[VirtualMagnet]
    assignments: Sequence[ClusterAssignment]
    quota_plans: Sequence[RingQuotaPlan] | None = None


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def field_metrics_to_dict(metrics: FieldMetrics) -> dict[str, float]:
    """Serialize vector homogeneity metrics."""
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


def _magnet_by_id(magnets: Sequence[VirtualMagnet]) -> dict[int, VirtualMagnet]:
    by_id: dict[int, VirtualMagnet] = {}
    for magnet in magnets:
        if magnet.magnet_id in by_id:
            raise ValueError(f"duplicate magnet_id: {magnet.magnet_id}")
        by_id[magnet.magnet_id] = magnet
    return by_id


def _orientation_angle_by_id() -> dict[str, float]:
    return {orientation.id: orientation.angle_deg for orientation in default_orientations()}


def _installed_side(slot: AssemblySlot, max_layer_id: int) -> str:
    if slot.mirror_pair_id is None:
        return "center"
    return "lower" if slot.layer_id < max_layer_id / 2.0 else "upper"


def final_placement_rows(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> list[dict[str, object]]:
    """Build rows for final_placement_trial_XXX.csv."""
    slot_by_id = _slot_by_id(slots)
    magnet_by_id = _magnet_by_id(magnets)
    orientation_angles = _orientation_angle_by_id()
    max_layer_id = max(slot.layer_id for slot in slots)
    rows: list[dict[str, object]] = []
    seen_slots: set[int] = set()
    for placement in sorted(placements, key=lambda item: item.insert_order):
        if placement.slot_flat_id in seen_slots:
            raise ValueError(f"duplicate placed slot_flat_id: {placement.slot_flat_id}")
        seen_slots.add(placement.slot_flat_id)
        if placement.slot_flat_id not in slot_by_id:
            raise ValueError(f"unknown placed slot_flat_id: {placement.slot_flat_id}")
        if placement.magnet_id not in magnet_by_id:
            raise ValueError(f"unknown placed magnet_id: {placement.magnet_id}")
        if placement.orientation_id not in orientation_angles:
            raise ValueError(f"unknown orientation_id: {placement.orientation_id}")
        slot = slot_by_id[placement.slot_flat_id]
        error = magnet_by_id[placement.magnet_id].true_error
        rows.append(
            {
                "work_unit_id": slot.work_unit_id,
                "mirror_pair_id": slot.mirror_pair_id or "",
                "installed_layer_id": slot.layer_id,
                "installed_side": _installed_side(slot, max_layer_id),
                "ring_id": slot.ring_id,
                "layer_id": slot.layer_id,
                "theta_id": slot.theta_id,
                "slot_flat_id": slot.slot_flat_id,
                "physical_slot_number": slot.physical_slot_number,
                "cluster_requested": placement.cluster_requested or "",
                "epsilon_parallel": error.epsilon_parallel,
                "delta_perp_1": error.delta_perp_1,
                "delta_perp_2": error.delta_perp_2,
                "orientation_id": placement.orientation_id,
                "orientation_angle_deg": orientation_angles[placement.orientation_id],
                "insert_order": placement.insert_order,
                "decision_engine": placement.decision_engine,
            }
        )
    return rows


def write_final_placement_csv(
    path: str | Path,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> None:
    """Write final placement CSV using the specification 13.5 columns."""
    rows = final_placement_rows(slots, magnets, placements)
    _write_csv(Path(path), FINAL_PLACEMENT_COLUMNS, rows)


def validate_final_placement_csv(path: str | Path) -> tuple[int, ...]:
    """Validate required placement columns and duplicate-free slot usage."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FINAL_PLACEMENT_COLUMNS:
            raise ValueError("final placement CSV columns do not match the Step 6 schema")
        slot_ids: list[int] = []
        for row in reader:
            slot_ids.append(int(row["slot_flat_id"]))
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("final placement CSV contains duplicate slot_flat_id values")
    return tuple(slot_ids)


def _planned_cluster_counts(assignments: Sequence[ClusterAssignment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for assignment in assignments:
        if assignment.cluster_id is None:
            continue
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return counts


def _used_cluster_counts(placements: Sequence[Placement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for placement in placements:
        if placement.cluster_requested is None:
            continue
        counts[placement.cluster_requested] = counts.get(placement.cluster_requested, 0) + 1
    return counts


def _assignment_cluster_by_magnet(
    assignments: Sequence[ClusterAssignment],
) -> dict[int, str | None]:
    cluster_by_magnet: dict[int, str | None] = {}
    for assignment in assignments:
        if assignment.magnet_id in cluster_by_magnet:
            raise ValueError(f"duplicate assignment magnet_id: {assignment.magnet_id}")
        cluster_by_magnet[assignment.magnet_id] = assignment.cluster_id
    return cluster_by_magnet


def write_cluster_usage_csv(
    path: str | Path,
    assignments: Sequence[ClusterAssignment],
    placements: Sequence[Placement],
) -> None:
    """Write cluster usage summary for a trial."""
    planned = _planned_cluster_counts(assignments)
    used = _used_cluster_counts(placements)
    cluster_ids = sorted(set(planned) | set(used))
    rows = [
        {
            "cluster_id": cluster_id,
            "planned_count": planned.get(cluster_id, 0),
            "used_count": used.get(cluster_id, 0),
            "remaining_count": planned.get(cluster_id, 0) - used.get(cluster_id, 0),
        }
        for cluster_id in cluster_ids
    ]
    _write_csv(
        Path(path),
        ["cluster_id", "planned_count", "used_count", "remaining_count"],
        rows,
    )


def _ring_work_unit_by_key(slots: Sequence[AssemblySlot]) -> dict[RingKey, str]:
    by_key: dict[RingKey, str] = {}
    for slot in slots:
        key = RingKey(layer_id=slot.layer_id, ring_id=slot.ring_id)
        work_unit_id = slot.work_unit_id or "W_UNASSIGNED"
        if key in by_key and by_key[key] != work_unit_id:
            # A mirror-pair work unit can intentionally share one id across two rings,
            # but one physical ring should still have one work unit id.
            raise ValueError(
                f"physical ring {key} appears in multiple work units: "
                f"{by_key[key]} and {work_unit_id}"
            )
        by_key[key] = work_unit_id
    return by_key


def ring_summary_rows(
    trial_id: int,
    slots: Sequence[AssemblySlot],
    summaries: Sequence[RingSummary],
) -> list[dict[str, object]]:
    """Build rows for ring_summary_trial_XXX.csv."""
    work_unit_by_key = _ring_work_unit_by_key(slots)
    rows: list[dict[str, object]] = []
    for summary in summaries:
        true_mean = np.asarray(summary.mean_true_error, dtype=np.float64).reshape(3)
        measured_mean = np.asarray(summary.mean_measured_error, dtype=np.float64).reshape(3)
        rows.append(
            {
                "trial_id": int(trial_id),
                "work_unit_id": work_unit_by_key.get(summary.ring_key, ""),
                "layer_id": summary.layer_id,
                "ring_id": summary.ring_id,
                "count": summary.count,
                "mean_epsilon": summary.mean_epsilon,
                "std_epsilon": summary.std_epsilon,
                "min_epsilon": summary.min_epsilon,
                "max_epsilon": summary.max_epsilon,
                "mean_angle_error": summary.mean_angle_error,
                "std_angle_error": summary.std_angle_error,
                "mean_true_epsilon": float(true_mean[0]),
                "mean_true_delta_perp_1": float(true_mean[1]),
                "mean_true_delta_perp_2": float(true_mean[2]),
                "mean_measured_epsilon": float(measured_mean[0]),
                "mean_measured_delta_perp_1": float(measured_mean[1]),
                "mean_measured_delta_perp_2": float(measured_mean[2]),
                "cluster_counts_json": _json_cell(summary.cluster_counts),
                "B0_norm_after_ring": summary.B0_norm_after_ring,
                "rms_homogeneity_ppm_after_ring": summary.rms_homogeneity_ppm_after_ring,
                "J_vector_after_ring": summary.J_vector_after_ring,
            }
        )
    return rows


def write_ring_summary_csv(
    path: str | Path,
    trial_id: int,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> tuple[RingSummary, ...]:
    """Write per-physical-ring magnet-error statistics for visualization."""
    summaries = ring_summary_from_placements(slots, magnets, placements)
    _write_csv(
        Path(path),
        RING_SUMMARY_COLUMNS,
        ring_summary_rows(trial_id, slots, summaries),
    )
    return summaries


def ring_pair_summary_rows(
    trial_id: int,
    pair_summaries: Sequence[RingPairSummary],
) -> list[dict[str, object]]:
    """Build rows for ring_pair_summary_trial_XXX.csv."""
    rows: list[dict[str, object]] = []
    for summary in pair_summaries:
        rows.append(
            {
                "trial_id": int(trial_id),
                "pair_id": summary.pair_id,
                "pair_index": summary.pair_index,
                "ring_id": summary.ring_id,
                "lower_layer_id": summary.lower_ring.layer_id,
                "upper_layer_id": (
                    "" if summary.upper_ring is None else summary.upper_ring.layer_id
                ),
                "lower_count": summary.lower_count,
                "upper_count": summary.upper_count,
                "lower_mean_epsilon": summary.lower_mean_epsilon,
                "upper_mean_epsilon": summary.upper_mean_epsilon,
                "mean_epsilon_difference": summary.mean_epsilon_difference,
                "mean_angle_error_difference": summary.mean_angle_error_difference,
                "residual_norm_after_lower": summary.residual_norm_after_lower,
                "residual_norm_after_upper": summary.residual_norm_after_upper,
                "residual_norm_after_pair": summary.residual_norm_after_pair,
                "pair_complete": summary.pair_complete,
            }
        )
    return rows


def write_ring_pair_summary_csv(
    path: str | Path,
    trial_id: int,
    ring_summaries: Sequence[RingSummary],
    *,
    pair_summaries: Sequence[RingPairSummary] | None = None,
) -> tuple[RingPairSummary, ...]:
    """Write mirror-pair ring-balance summary for visualization."""
    summaries = (
        tuple(pair_summaries)
        if pair_summaries is not None and len(pair_summaries) > 0
        else ring_pair_summary_from_ring_summaries(ring_summaries)
    )
    _write_csv(
        Path(path),
        RING_PAIR_SUMMARY_COLUMNS,
        ring_pair_summary_rows(trial_id, summaries),
    )
    return summaries


def _timeline_event_to_dict(
    trial_id: int,
    event: AssemblyTimelineEvent,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": int(trial_id),
        "step": event.step,
        "event": event.event,
        "result_label": event.result_label,
        "work_unit_id": event.work_unit_id,
        "layer_id": event.layer_id,
        "ring_id": event.ring_id,
        "theta_id": event.theta_id,
        "slot_flat_id": event.slot_flat_id,
        "physical_slot_number": event.physical_slot_number,
        "magnet_id": event.magnet_id,
        "cluster_requested": event.cluster_requested,
        "epsilon_parallel": event.epsilon_parallel,
        "angle_error": event.angle_error,
        "orientation_id": event.orientation_id,
        "insert_order": event.insert_order,
        "decision_engine": event.decision_engine,
        "ring_count_so_far": event.ring_count_so_far,
        "ring_mean_epsilon_so_far": event.ring_mean_epsilon_so_far,
        "ring_mean_angle_error_so_far": event.ring_mean_angle_error_so_far,
        "residual_norm": event.residual_norm,
        "B0_norm": event.B0_norm,
        "rms_homogeneity_ppm": event.rms_homogeneity_ppm,
        "J_vector": event.J_vector,
    }


def write_assembly_timeline_jsonl(
    path: str | Path,
    trial_id: int,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    *,
    result_label: str = "linear",
) -> tuple[AssemblyTimelineEvent, ...]:
    """Write one insert event per line for ring-by-ring playback."""
    events = timeline_from_placements(
        slots,
        magnets,
        placements,
        result_label=result_label,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    _timeline_event_to_dict(trial_id, event),
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
    return events


def cluster_quota_plan_rows(
    trial_id: int,
    quota_plans: Sequence[RingQuotaPlan] | None,
) -> list[dict[str, object]]:
    """Build rows for cluster_quota_plan_trial_XXX.csv."""
    if quota_plans is None:
        return []
    rows: list[dict[str, object]] = []
    for plan in quota_plans:
        rows.append(
            {
                "trial_id": int(trial_id),
                "work_unit_id": plan.work_unit_id,
                "layer_id": plan.layer_id,
                "ring_id": plan.ring_id,
                "target_count": plan.target_count,
                "target_mean_epsilon": plan.target_mean_epsilon,
                "ring_importance": plan.ring_importance,
                "expected_mean_epsilon": plan.expected_mean_epsilon,
                "expected_mean_angle_bin": plan.expected_mean_angle_bin,
                "expected_angle_error": plan.expected_angle_error,
                "mirror_pair_id": plan.mirror_pair_id or "",
                "allowed_clusters_json": _json_cell(list(plan.allowed_clusters)),
                "quota_by_cluster_json": _json_cell(plan.quota_by_cluster),
            }
        )
    return rows


def write_cluster_quota_plan_csv(
    path: str | Path,
    trial_id: int,
    quota_plans: Sequence[RingQuotaPlan] | None,
) -> None:
    """Write planned per-ring cluster quotas; writes headers even when no plan exists."""
    _write_csv(
        Path(path),
        CLUSTER_QUOTA_PLAN_COLUMNS,
        cluster_quota_plan_rows(trial_id, quota_plans),
    )


def cluster_pickup_log_rows(
    trial_id: int,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    assignments: Sequence[ClusterAssignment],
    placements: Sequence[Placement],
) -> list[dict[str, object]]:
    """Build one pickup row per placed magnet for playback and inventory analysis."""
    slot_by_id = _slot_by_id(slots)
    magnet_by_id = _magnet_by_id(magnets)
    cluster_by_magnet = _assignment_cluster_by_magnet(assignments)
    rows: list[dict[str, object]] = []
    for placement in sorted(placements, key=lambda item: item.insert_order):
        if placement.slot_flat_id not in slot_by_id:
            raise ValueError(f"unknown placed slot_flat_id: {placement.slot_flat_id}")
        if placement.magnet_id not in magnet_by_id:
            raise ValueError(f"unknown placed magnet_id: {placement.magnet_id}")
        slot = slot_by_id[placement.slot_flat_id]
        magnet = magnet_by_id[placement.magnet_id]
        true_error = magnet.true_error
        measured_error = magnet.measured_error
        rows.append(
            {
                "trial_id": int(trial_id),
                "insert_order": placement.insert_order,
                "decision_engine": placement.decision_engine,
                "work_unit_id": slot.work_unit_id,
                "layer_id": slot.layer_id,
                "ring_id": slot.ring_id,
                "theta_id": slot.theta_id,
                "slot_flat_id": slot.slot_flat_id,
                "physical_slot_number": slot.physical_slot_number,
                "center_x_m": float(slot.center_m[0]),
                "center_y_m": float(slot.center_m[1]),
                "nominal_phi_rad": slot.nominal_phi_rad,
                "magnet_id": placement.magnet_id,
                "assignment_cluster_id": cluster_by_magnet.get(placement.magnet_id) or "",
                "cluster_requested": placement.cluster_requested or "",
                "epsilon_parallel": true_error.epsilon_parallel,
                "delta_perp_1": true_error.delta_perp_1,
                "delta_perp_2": true_error.delta_perp_2,
                "measured_epsilon_parallel": measured_error.epsilon_parallel,
                "measured_delta_perp_1": measured_error.delta_perp_1,
                "measured_delta_perp_2": measured_error.delta_perp_2,
                "orientation_id": placement.orientation_id,
            }
        )
    return rows


def write_cluster_pickup_log_csv(
    path: str | Path,
    trial_id: int,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    assignments: Sequence[ClusterAssignment],
    placements: Sequence[Placement],
) -> None:
    """Write the simulated cluster pickup sequence for one trial."""
    _write_csv(
        Path(path),
        CLUSTER_PICKUP_LOG_COLUMNS,
        cluster_pickup_log_rows(trial_id, slots, magnets, assignments, placements),
    )


def write_work_unit_summary_csv(
    path: str | Path,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
) -> None:
    """Write per-work-unit aggregate error statistics."""
    slot_by_id = _slot_by_id(slots)
    magnet_by_id = _magnet_by_id(magnets)
    errors_by_work_unit: dict[str, list[tuple[float, float, float]]] = {}
    for placement in placements:
        slot = slot_by_id[placement.slot_flat_id]
        magnet = magnet_by_id[placement.magnet_id]
        work_unit_id = slot.work_unit_id or "W_UNASSIGNED"
        error = magnet.true_error
        errors_by_work_unit.setdefault(work_unit_id, []).append(
            (error.epsilon_parallel, error.delta_perp_1, error.delta_perp_2)
        )

    rows: list[dict[str, object]] = []
    for work_unit_id, errors in sorted(errors_by_work_unit.items()):
        arr = np.asarray(errors, dtype=np.float64)
        transverse_norm = np.sqrt(arr[:, 1] * arr[:, 1] + arr[:, 2] * arr[:, 2])
        rows.append(
            {
                "work_unit_id": work_unit_id,
                "slots_used": int(arr.shape[0]),
                "mean_epsilon_parallel": float(np.mean(arr[:, 0])),
                "mean_delta_perp_1": float(np.mean(arr[:, 1])),
                "mean_delta_perp_2": float(np.mean(arr[:, 2])),
                "rms_transverse_error": float(np.sqrt(np.mean(transverse_norm * transverse_norm))),
            }
        )
    _write_csv(
        Path(path),
        [
            "work_unit_id",
            "slots_used",
            "mean_epsilon_parallel",
            "mean_delta_perp_1",
            "mean_delta_perp_2",
            "rms_transverse_error",
        ],
        rows,
    )


def write_field_metrics_json(path: str | Path, artifact: SimulationTrialArtifacts) -> None:
    """Write per-trial field metrics JSON."""
    result = artifact.result
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "trial_id": artifact.trial_id,
        "seed": artifact.seed,
        "units": {
            "homogeneity": "ppm",
            "B0_norm": "model field units",
            "J_vector": "model field units squared",
        },
        "random_baseline": field_metrics_to_dict(result.random_baseline.evaluation.metrics),
        "linear_sensitivity": field_metrics_to_dict(result.linear.evaluation.metrics),
        "rms_ratio_linear_over_random": result.rms_ratio_linear_over_random,
        "j_ratio_linear_over_random": result.j_ratio_linear_over_random,
        "linear_score_measured": result.linear.assignment.linear_score,
    }
    if result.self_consistent is not None:
        payload["sequential_self_consistent"] = field_metrics_to_dict(
            result.self_consistent.evaluation.metrics
        )
        payload["rms_ratio_self_consistent_over_linear"] = (
            result.rms_ratio_self_consistent_over_linear
        )
        payload["j_ratio_self_consistent_over_linear"] = result.j_ratio_self_consistent_over_linear
        payload["self_consistent_evaluated_count"] = result.self_consistent.evaluated_count
    _write_json(Path(path), payload)


def write_session_log_jsonl(
    path: str | Path,
    artifact: SimulationTrialArtifacts,
) -> None:
    """Write a minimal JSONL event log that later UI/session code can replay."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        events: list[dict[str, object]] = [
            {
                "schema_version": SCHEMA_VERSION,
                "trial_id": artifact.trial_id,
                "event": "session_started",
                "seed": artifact.seed,
            }
        ]
        for placement in artifact.result.linear.assignment.placements:
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "trial_id": artifact.trial_id,
                    "event": "insert_confirmed",
                    "slot_flat_id": placement.slot_flat_id,
                    "orientation_id": placement.orientation_id,
                    "insert_order": placement.insert_order,
                    "decision_engine": placement.decision_engine,
                }
            )
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "trial_id": artifact.trial_id,
                "event": "session_completed",
                "placements": len(artifact.result.linear.assignment.placements),
            }
        )
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")


def simulation_trial_row(artifact: SimulationTrialArtifacts) -> dict[str, object]:
    """Build one row for simulation_trials.csv."""
    result = artifact.result
    random_metrics = result.random_baseline.evaluation.metrics
    linear_metrics = result.linear.evaluation.metrics
    row: dict[str, object] = {
        "trial_id": artifact.trial_id,
        "seed": artifact.seed,
        "random_rms_homogeneity_ppm": random_metrics.rms_homogeneity_ppm,
        "linear_rms_homogeneity_ppm": linear_metrics.rms_homogeneity_ppm,
        "random_max_homogeneity_ppm": random_metrics.max_homogeneity_ppm,
        "linear_max_homogeneity_ppm": linear_metrics.max_homogeneity_ppm,
        "random_p95_homogeneity_ppm": random_metrics.p95_homogeneity_ppm,
        "linear_p95_homogeneity_ppm": linear_metrics.p95_homogeneity_ppm,
        "random_p99_homogeneity_ppm": random_metrics.p99_homogeneity_ppm,
        "linear_p99_homogeneity_ppm": linear_metrics.p99_homogeneity_ppm,
        "random_B0_norm": random_metrics.B0_norm,
        "linear_B0_norm": linear_metrics.B0_norm,
        "random_J_vector": random_metrics.J_vector,
        "linear_J_vector": linear_metrics.J_vector,
        "rms_ratio_linear_over_random": result.rms_ratio_linear_over_random,
        "j_ratio_linear_over_random": result.j_ratio_linear_over_random,
        "linear_score_measured": result.linear.assignment.linear_score,
    }
    if result.self_consistent is not None:
        sc_metrics = result.self_consistent.evaluation.metrics
        row.update(
            {
                "self_consistent_rms_homogeneity_ppm": sc_metrics.rms_homogeneity_ppm,
                "self_consistent_max_homogeneity_ppm": sc_metrics.max_homogeneity_ppm,
                "self_consistent_p95_homogeneity_ppm": sc_metrics.p95_homogeneity_ppm,
                "self_consistent_p99_homogeneity_ppm": sc_metrics.p99_homogeneity_ppm,
                "self_consistent_B0_norm": sc_metrics.B0_norm,
                "self_consistent_J_vector": sc_metrics.J_vector,
                "rms_ratio_self_consistent_over_linear": (
                    result.rms_ratio_self_consistent_over_linear
                ),
                "j_ratio_self_consistent_over_linear": (result.j_ratio_self_consistent_over_linear),
                "self_consistent_evaluated_count": result.self_consistent.evaluated_count,
            }
        )
    return row


def build_simulation_summary_payload(
    artifacts: Sequence[SimulationTrialArtifacts],
    summary: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build simulation_summary.json payload."""
    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {
            "length": "m",
            "angle": "rad",
            "homogeneity": "ppm",
            "magnet_error": "[relative strength, rad-equivalent transverse components]",
        },
        "metadata": dict(metadata or {}),
        "summary": summary,
        "trials": [simulation_trial_row(artifact) for artifact in artifacts],
    }


def write_simulation_outputs(
    out_dir: str | Path,
    artifacts: Sequence[SimulationTrialArtifacts],
    slots: Sequence[AssemblySlot],
    summary: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Write all Step 6 Plan C simulation output files."""
    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    summary_path = out_path / "simulation_summary.json"
    _write_json(
        summary_path,
        build_simulation_summary_payload(artifacts, summary, metadata=metadata),
    )
    written["simulation_summary"] = summary_path

    trials_path = out_path / "simulation_trials.csv"
    _write_csv(
        trials_path,
        SIMULATION_TRIAL_COLUMNS,
        [simulation_trial_row(artifact) for artifact in artifacts],
    )
    written["simulation_trials"] = trials_path

    for artifact in artifacts:
        suffix = f"trial_{artifact.trial_id:03d}"
        placements = artifact.result.linear.assignment.placements
        final_path = out_path / f"final_placement_{suffix}.csv"
        write_final_placement_csv(
            final_path,
            slots,
            artifact.magnets,
            placements,
        )
        written[f"final_placement_{suffix}"] = final_path

        cluster_path = out_path / f"cluster_usage_{suffix}.csv"
        write_cluster_usage_csv(
            cluster_path,
            artifact.assignments,
            placements,
        )
        written[f"cluster_usage_{suffix}"] = cluster_path

        work_unit_path = out_path / f"work_unit_summary_{suffix}.csv"
        write_work_unit_summary_csv(
            work_unit_path,
            slots,
            artifact.magnets,
            placements,
        )
        written[f"work_unit_summary_{suffix}"] = work_unit_path

        ring_summary_path = out_path / f"ring_summary_{suffix}.csv"
        ring_summaries = write_ring_summary_csv(
            ring_summary_path,
            artifact.trial_id,
            slots,
            artifact.magnets,
            placements,
        )
        written[f"ring_summary_{suffix}"] = ring_summary_path

        ring_pair_path = out_path / f"ring_pair_summary_{suffix}.csv"
        write_ring_pair_summary_csv(
            ring_pair_path,
            artifact.trial_id,
            ring_summaries,
            pair_summaries=artifact.result.linear.assignment.mirror_pair_summaries,
        )
        written[f"ring_pair_summary_{suffix}"] = ring_pair_path

        timeline_path = out_path / f"assembly_timeline_{suffix}.jsonl"
        write_assembly_timeline_jsonl(
            timeline_path,
            artifact.trial_id,
            slots,
            artifact.magnets,
            placements,
        )
        written[f"assembly_timeline_{suffix}"] = timeline_path

        quota_path = out_path / f"cluster_quota_plan_{suffix}.csv"
        write_cluster_quota_plan_csv(
            quota_path,
            artifact.trial_id,
            artifact.quota_plans,
        )
        written[f"cluster_quota_plan_{suffix}"] = quota_path

        pickup_log_path = out_path / f"cluster_pickup_log_{suffix}.csv"
        write_cluster_pickup_log_csv(
            pickup_log_path,
            artifact.trial_id,
            slots,
            artifact.magnets,
            artifact.assignments,
            placements,
        )
        written[f"cluster_pickup_log_{suffix}"] = pickup_log_path

        metrics_path = out_path / f"field_metrics_{suffix}.json"
        write_field_metrics_json(metrics_path, artifact)
        written[f"field_metrics_{suffix}"] = metrics_path

        log_path = out_path / f"streamlit_session_log_{suffix}.jsonl"
        write_session_log_jsonl(log_path, artifact)
        written[f"streamlit_session_log_{suffix}"] = log_path

    return written


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries for tests and lightweight analysis."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file containing one JSON object per non-empty line."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object")
        rows.append(raw)
    return rows


def load_json_dict(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON file must contain an object")
    return raw


__all__ = [
    "FINAL_PLACEMENT_COLUMNS",
    "CLUSTER_PICKUP_LOG_COLUMNS",
    "CLUSTER_QUOTA_PLAN_COLUMNS",
    "RING_PAIR_SUMMARY_COLUMNS",
    "RING_SUMMARY_COLUMNS",
    "SCHEMA_VERSION",
    "SIMULATION_TRIAL_COLUMNS",
    "SimulationTrialArtifacts",
    "build_simulation_summary_payload",
    "cluster_pickup_log_rows",
    "cluster_quota_plan_rows",
    "field_metrics_to_dict",
    "final_placement_rows",
    "load_json_dict",
    "read_csv_dicts",
    "read_jsonl_dicts",
    "ring_pair_summary_rows",
    "ring_summary_rows",
    "simulation_trial_row",
    "validate_final_placement_csv",
    "write_assembly_timeline_jsonl",
    "write_cluster_pickup_log_csv",
    "write_cluster_quota_plan_csv",
    "write_cluster_usage_csv",
    "write_field_metrics_json",
    "write_final_placement_csv",
    "write_ring_pair_summary_csv",
    "write_ring_summary_csv",
    "write_session_log_jsonl",
    "write_simulation_outputs",
    "write_work_unit_summary_csv",
]
