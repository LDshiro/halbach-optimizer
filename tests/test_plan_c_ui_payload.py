import numpy as np

from halbach.assembly.ui_payload import (
    build_session_ui_payload,
    build_slot_display_rows,
    build_summary_ui_payload,
)
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    ClusterInventory,
    ClusterStats,
    FieldMetrics,
    LinearCandidate,
    MagnetError,
    Placement,
    VirtualMagnet,
)
from halbach.assembly.session import SessionSnapshot


def _slot(slot_id: int, layer_id: int, theta_id: int) -> AssemblySlot:
    return AssemblySlot(
        slot_flat_id=slot_id,
        ring_id=0,
        layer_id=layer_id,
        theta_id=theta_id,
        center_m=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        nominal_phi_rad=0.0,
        physical_slot_number=theta_id + 1,
        work_unit_id="W_ALL",
        mirror_pair_id="P000_001",
    )


def _snapshot() -> SessionSnapshot:
    error = MagnetError(0.01, 0.001, -0.002)
    return SessionSnapshot(
        state="INSTALL_OR_CONFIRM_PAIR",
        sub_state="WAIT_FOR_INSERT_CONFIRMATION",
        residual=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        remaining_slot_flat_ids=(2,),
        placements=(
            Placement(
                slot_flat_id=1,
                magnet_id=10,
                orientation_id="O0",
                insert_order=0,
                decision_engine="linear_sensitivity",
            ),
        ),
        measurement_index=2,
        pending_magnet=VirtualMagnet(
            magnet_id=11,
            true_error=error,
            measured_error=error,
            quality=0.98,
        ),
        pending_candidate=LinearCandidate(
            slot_flat_id=2,
            orientation_id="O180",
            score=12.5,
            contribution=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        ),
    )


def test_session_ui_payload_contains_required_fields_and_instruction(tmp_path) -> None:
    slots = [_slot(1, 0, 0), _slot(2, 1, 1)]
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("", encoding="utf-8")
    inventory = ClusterInventory(
        clusters={
            "S00_A00": ClusterStats(
                cluster_id="S00_A00",
                count=3,
                mean=np.zeros(3, dtype=np.float64),
                cov=np.zeros((3, 3), dtype=np.float64),
            )
        },
        quarantine={"Q_DIRECTION_OUTLIER": 1},
    )
    metrics = FieldMetrics(
        rms_homogeneity_ppm=1.0,
        max_homogeneity_ppm=2.0,
        p95_homogeneity_ppm=1.5,
        p99_homogeneity_ppm=1.8,
        B0_norm=0.1,
        J_vector=1e-6,
    )

    payload = build_session_ui_payload(
        _snapshot(),
        slots,
        assignments=[ClusterAssignment(magnet_id=11, cluster_id="S00_A00")],
        inventory=inventory,
        metrics=metrics,
        log_path=log_path,
    )

    assert payload["mode"] == "simulation_step_by_step"
    assert payload["state"] == "INSTALL_OR_CONFIRM_PAIR"
    assert payload["current_work_unit_id"] == "W_ALL"
    assert payload["current_mirror_pair_id"] == "P000_001"
    assert payload["remaining_slot_count"] == 1
    assert payload["next_cluster_id"] == "S00_A00"
    assert payload["cluster_inventory"]["clusters"]["S00_A00"]["count"] == 3
    assert payload["measurement"]["magnet_id"] == 11
    assert payload["recommended_slot_flat_id"] == 2
    assert payload["recommended_orientation_id"] == "O180"
    assert payload["orientation_instruction"]
    assert payload["current_metrics"]["rms_homogeneity_ppm"] == 1.0
    assert payload["quarantine_count"] == 1
    assert payload["log_saved"] is True


def test_slot_display_rows_distinguish_occupied_empty_and_recommended() -> None:
    slots = [_slot(1, 0, 0), _slot(2, 1, 1), _slot(3, 1, 2)]

    rows = build_slot_display_rows(_snapshot(), slots)
    state_by_slot = {row["slot_flat_id"]: row["state"] for row in rows}
    highlight_by_slot = {row["slot_flat_id"]: row["highlight"] for row in rows}

    assert state_by_slot == {1: "occupied", 2: "recommended", 3: "empty"}
    assert highlight_by_slot[2] is True
    assert highlight_by_slot[1] is False
    assert highlight_by_slot[3] is False


def test_summary_ui_payload_normalizes_simulation_summary() -> None:
    payload = build_summary_ui_payload(
        {
            "schema_version": 1,
            "metadata": {"engine": "linear_sensitivity"},
            "summary": {
                "trials": 2,
                "rms_ratio_mean": 0.8,
                "linear_improved_count": 1,
            },
            "trials": [{"trial_id": 0}, {"trial_id": 1}],
        }
    )

    assert payload["schema_version"] == 1
    assert payload["engine"] == "linear_sensitivity"
    assert payload["trials"] == 2
    assert payload["rms_ratio_mean"] == 0.8
    assert payload["linear_improved_count"] == 1
    assert len(payload["trial_rows"]) == 2
