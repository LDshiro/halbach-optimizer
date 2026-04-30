from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from halbach.assembly.orientations import default_orientations
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    FieldMetrics,
    Placement,
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


@dataclass(frozen=True)
class SimulationTrialArtifacts:
    """All per-trial data needed for Step 6 output files."""

    trial_id: int
    seed: int
    result: SimulationComparisonResult
    magnets: Sequence[VirtualMagnet]
    assignments: Sequence[ClusterAssignment]


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
    return {
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
        final_path = out_path / f"final_placement_{suffix}.csv"
        write_final_placement_csv(
            final_path,
            slots,
            artifact.magnets,
            artifact.result.linear.assignment.placements,
        )
        written[f"final_placement_{suffix}"] = final_path

        cluster_path = out_path / f"cluster_usage_{suffix}.csv"
        write_cluster_usage_csv(
            cluster_path,
            artifact.assignments,
            artifact.result.linear.assignment.placements,
        )
        written[f"cluster_usage_{suffix}"] = cluster_path

        work_unit_path = out_path / f"work_unit_summary_{suffix}.csv"
        write_work_unit_summary_csv(
            work_unit_path,
            slots,
            artifact.magnets,
            artifact.result.linear.assignment.placements,
        )
        written[f"work_unit_summary_{suffix}"] = work_unit_path

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


def load_json_dict(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON file must contain an object")
    return raw


__all__ = [
    "FINAL_PLACEMENT_COLUMNS",
    "SCHEMA_VERSION",
    "SIMULATION_TRIAL_COLUMNS",
    "SimulationTrialArtifacts",
    "build_simulation_summary_payload",
    "field_metrics_to_dict",
    "final_placement_rows",
    "load_json_dict",
    "read_csv_dicts",
    "simulation_trial_row",
    "validate_final_placement_csv",
    "write_cluster_usage_csv",
    "write_field_metrics_json",
    "write_final_placement_csv",
    "write_session_log_jsonl",
    "write_simulation_outputs",
    "write_work_unit_summary_csv",
]
