import json
from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory, inventory_total_count
from halbach.assembly.measurement import FakeSerialMeasurementProvider, SyntheticMeasurementProvider
from halbach.assembly.online_assignment import run_linear_sensitivity_assignment
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.session import (
    PlanCSession,
    SessionLogError,
    load_latest_session_snapshot,
    run_session_to_completion,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import ClusterAssignment, Placement
from halbach.assembly.variation import generate_virtual_magnets
from halbach.cli.plan_c_session import main as session_main
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"session_run_N{N}_R{R}_K{K}"
    results = generate_halbach_initial(
        N=N,
        R=R,
        end_R=None,
        end_layers_per_side=0,
        K=K,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-session-test",
        schema_version=1,
        generator_params={},
        description="plan c session test run",
    )
    return load_run(run_dir)


def _case(tmp_path: Path):
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0005,
        measurement_noise=None,
    )
    assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
    inventory = build_cluster_inventory(magnets, assignments)
    return table, magnets, assignments, inventory


def _advance_to_measurement_wait(session: PlanCSession) -> None:
    session.step()
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "WAIT_FOR_MAGNET_MEASUREMENT")


def _advance_to_insert_confirmation(session: PlanCSession) -> None:
    while session.state != "INSTALL_OR_CONFIRM_PAIR":
        session.step()


def test_session_state_transitions_to_insert_confirmation(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )

    assert session.state == "PREPARE_SESSION"
    session.step()
    assert (session.state, session.sub_state) == ("SELECT_WORK_UNIT", "CHOOSE_CLUSTER")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "WAIT_FOR_MAGNET_MEASUREMENT")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "VALIDATE_MEASUREMENT")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "SOLVE_PLACEMENT")
    session.step()
    assert session.state == "EVALUATE_MIRROR_PAIR_SWAP"
    session.step()
    assert session.state == "INSTALL_OR_CONFIRM_PAIR"


def test_step_by_step_session_matches_auto_linear_assignment(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    auto = run_linear_sensitivity_assignment(
        table,
        magnets,
        assignments=assignments,
        inventory=inventory,
    )
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )

    final = run_session_to_completion(session)

    assert final.state == "COMPLETE"
    assert final.placements == auto.placements
    assert inventory_total_count(session.current_inventory) == 0


def test_session_undo_restores_last_pending_insert(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )
    while session.state != "INSTALL_OR_CONFIRM_PAIR":
        session.step()
    pending_slot = session.pending_candidate.slot_flat_id
    session.confirm_insert()

    assert len(session.placements) == 1
    session.undo_last_insert()

    assert len(session.placements) == 0
    assert pending_slot in session.remaining_slot_flat_ids
    assert session.pending_candidate is not None
    assert session.state == "INSTALL_OR_CONFIRM_PAIR"


def test_session_resume_from_log_completes_without_duplicate_slots(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    log_path = tmp_path / "session.jsonl"
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
        log_path=log_path,
    )
    while len(session.placements) < 1:
        if session.state == "INSTALL_OR_CONFIRM_PAIR":
            session.confirm_insert()
        else:
            session.step()

    resumed = PlanCSession.resume_from_log(
        log_path,
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )
    final = run_session_to_completion(resumed)
    slot_ids = [placement.slot_flat_id for placement in final.placements]

    assert len(slot_ids) == len(set(slot_ids))
    assert len(slot_ids) == len(magnets)
    assert final.state == "COMPLETE"


def test_plan_c_session_cli_writes_log_and_final_placement(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    out_dir = tmp_path / "session_cli"

    session_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_dir),
            "--seed",
            "5",
            "--strength-sigma",
            "0.001",
            "--direction-sigma",
            "0.0005",
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "5",
        ]
    )

    assert (out_dir / "session_log.jsonl").exists()
    assert (out_dir / "placement_final.csv").exists()
    assert (out_dir / "session_summary.json").exists()


def test_bad_measurement_keeps_session_waiting_and_preserves_state(tmp_path: Path) -> None:
    table, _magnets, _assignments, _inventory = _case(tmp_path)
    log_path = tmp_path / "bad_measurement.jsonl"
    session = PlanCSession(
        sensitivity_table=table,
        provider=FakeSerialMeasurementProvider(["{bad-json"]),
        log_path=log_path,
    )
    _advance_to_measurement_wait(session)
    residual_before = np.array(session.residual, copy=True)

    session.step()

    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "WAIT_FOR_MAGNET_MEASUREMENT")
    assert session.pending_magnet is None
    assert session.pending_candidate is None
    assert session.placements == ()
    np.testing.assert_allclose(session.residual, residual_before)
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event") == "measurement_rejected" for event in events)


def test_low_quality_measurement_logs_quarantine_reason(tmp_path: Path) -> None:
    table, _magnets, _assignments, _inventory = _case(tmp_path)
    log_path = tmp_path / "low_quality.jsonl"
    provider = FakeSerialMeasurementProvider(
        [
            json.dumps(
                {
                    "moment_magnitude": 1.0,
                    "direction": [0.0, 0.0, 1.0],
                    "quality": 0.5,
                }
            )
        ]
    )
    session = PlanCSession(sensitivity_table=table, provider=provider, log_path=log_path)
    _advance_to_measurement_wait(session)

    session.step()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    rejected = [event for event in events if event.get("event") == "measurement_rejected"]
    assert rejected
    assert rejected[-1]["quarantine_id"] == "Q_MEASUREMENT_UNSTABLE"
    assert rejected[-1]["reason"] == "Q_MEASUREMENT_UNSTABLE"


def test_corrupt_session_log_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "corrupt.jsonl"
    log_path.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(SessionLogError):
        load_latest_session_snapshot(log_path)


def test_confirm_rejects_duplicate_or_inconsistent_slot_state(tmp_path: Path) -> None:
    table, magnets, _assignments, _inventory = _case(tmp_path)
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
    )
    _advance_to_insert_confirmation(session)
    assert session.pending_candidate is not None
    slot_id = session.pending_candidate.slot_flat_id
    session.placements = (
        Placement(
            slot_flat_id=slot_id,
            magnet_id=999,
            orientation_id="O0",
            insert_order=0,
            decision_engine="test",
        ),
    )

    with pytest.raises(ValueError, match="slot"):
        session.confirm_insert()


def test_session_init_rejects_inventory_assignment_mismatch(tmp_path: Path) -> None:
    table, magnets, _assignments, inventory = _case(tmp_path)

    with pytest.raises(ValueError, match="inventory"):
        PlanCSession(
            sensitivity_table=table,
            provider=SyntheticMeasurementProvider(magnets),
            assignments=[ClusterAssignment(magnet_id=magnets[0].magnet_id, cluster_id="MISSING")],
            inventory=inventory,
        )


def test_manual_override_updates_pending_candidate_and_logs_event(tmp_path: Path) -> None:
    table, magnets, _assignments, _inventory = _case(tmp_path)
    log_path = tmp_path / "override.jsonl"
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        log_path=log_path,
    )
    _advance_to_insert_confirmation(session)
    assert session.pending_candidate is not None
    original_slot = session.pending_candidate.slot_flat_id
    override_slot = next(slot_id for slot_id in session.remaining_slot_flat_ids if slot_id != original_slot)
    override_orientation = "O180"

    session.override_pending_candidate(
        override_slot,
        override_orientation,
        reason="fixture test",
        operator="tester",
    )
    assert session.pending_candidate is not None
    assert session.pending_candidate.slot_flat_id == override_slot
    assert session.pending_candidate.orientation_id == override_orientation

    snapshot = session.confirm_insert()

    assert snapshot.placements[-1].slot_flat_id == override_slot
    assert snapshot.placements[-1].orientation_id == override_orientation
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    overrides = [event for event in events if event.get("event") == "manual_override"]
    assert overrides
    assert overrides[-1]["slot_flat_id"] == override_slot
    assert overrides[-1]["old_slot_flat_id"] == original_slot
