import numpy as np
import pytest

from halbach.assembly.online_assignment import (
    run_linear_sensitivity_assignment,
    run_ring_constrained_linear_assignment,
)
from halbach.assembly.simulation import build_random_placements, run_simulation_trial
from halbach.assembly.types import (
    AssemblySlot,
    MagnetError,
    SensitivityTable,
    VirtualMagnet,
    WorkUnit,
)


def _slot(slot_id: int, layer_id: int, theta_id: int) -> AssemblySlot:
    return AssemblySlot(
        slot_flat_id=slot_id,
        ring_id=0,
        layer_id=layer_id,
        theta_id=theta_id,
        center_m=np.array([0.10 + 0.01 * theta_id, 0.02, -0.03 + 0.06 * layer_id]),
        nominal_u=np.array([0.0, 1.0, 0.0]),
        nominal_phi_rad=0.0,
        physical_slot_number=theta_id + 1,
        work_unit_id=f"W_K{layer_id:03d}_R000",
    )


def _slots() -> list[AssemblySlot]:
    return [
        _slot(10, 0, 0),
        _slot(11, 0, 1),
        _slot(20, 1, 0),
        _slot(21, 1, 1),
    ]


def _table(slots: list[AssemblySlot]) -> SensitivityTable:
    C = np.zeros((4, 1, 1, 3), dtype=np.float64)
    C[:, 0, 0, 0] = np.array([10.0, 9.0, 0.1, 0.2], dtype=np.float64)
    return SensitivityTable(
        slot_flat_id=np.array([slot.slot_flat_id for slot in slots], dtype=np.int_),
        ring_id=np.array([slot.ring_id for slot in slots], dtype=np.int_),
        layer_id=np.array([slot.layer_id for slot in slots], dtype=np.int_),
        theta_id=np.array([slot.theta_id for slot in slots], dtype=np.int_),
        centers_m=np.vstack([slot.center_m for slot in slots]).astype(np.float64),
        nominal_u=np.vstack([slot.nominal_u for slot in slots]).astype(np.float64),
        orientation_id=("O0",),
        C=C,
        roi_points=np.array([[0.0, 0.0, 0.02]], dtype=np.float64),
        normalization_b0=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata={"case": "ring-constrained-assignment-test"},
    )


def _magnets() -> list[VirtualMagnet]:
    magnets: list[VirtualMagnet] = []
    for magnet_id in range(4):
        error = MagnetError(1.0, 0.0, 0.0)
        magnets.append(
            VirtualMagnet(
                magnet_id=magnet_id,
                true_error=error,
                measured_error=error,
                quality=1.0,
            )
        )
    return magnets


def _work_units() -> list[WorkUnit]:
    return [
        WorkUnit(
            work_unit_id="W_K000_R000",
            mode="ring_by_ring_outer_to_inner",
            slot_flat_ids=(10, 11),
            label="outer layer",
        ),
        WorkUnit(
            work_unit_id="W_K001_R000",
            mode="ring_by_ring_outer_to_inner",
            slot_flat_ids=(20, 21),
            label="inner layer",
        ),
    ]


def test_ring_constrained_linear_assignment_finishes_current_unit_before_next() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = _magnets()

    global_result = run_linear_sensitivity_assignment(table, magnets)
    constrained = run_ring_constrained_linear_assignment(table, magnets, _work_units())

    assert global_result.placements[0].slot_flat_id == 20
    assert [placement.slot_flat_id for placement in constrained.placements[:2]] == [11, 10]
    assert [placement.slot_flat_id for placement in constrained.placements[2:]] == [20, 21]
    assert all(
        placement.decision_engine == "ring_constrained_linear_sensitivity"
        for placement in constrained.placements
    )
    assert constrained.remaining_slot_flat_ids == ()


def test_all_slot_work_unit_matches_existing_global_linear_assignment() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = _magnets()
    all_slots = [
        WorkUnit(
            work_unit_id="W_ALL",
            mode="all_slots",
            slot_flat_ids=tuple(slot.slot_flat_id for slot in slots),
            label="all slots",
        )
    ]

    global_result = run_linear_sensitivity_assignment(table, magnets)
    constrained = run_ring_constrained_linear_assignment(table, magnets, all_slots)

    assert constrained == global_result


def test_ring_constrained_assignment_rejects_bad_work_unit_coverage() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = _magnets()

    duplicate_units = [
        WorkUnit("W0", "ring_by_ring_outer_to_inner", (10, 11), "first"),
        WorkUnit("W1", "ring_by_ring_outer_to_inner", (11, 20, 21), "second"),
    ]
    with pytest.raises(ValueError, match="multiple work units"):
        run_ring_constrained_linear_assignment(table, magnets, duplicate_units)

    missing_units = [WorkUnit("W0", "ring_by_ring_outer_to_inner", (10, 11), "first")]
    with pytest.raises(ValueError, match="coverage mismatch"):
        run_ring_constrained_linear_assignment(table, magnets, missing_units)


def test_random_placements_can_follow_work_unit_order() -> None:
    slots = _slots()
    magnets = _magnets()
    reversed_units = [
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (20, 21), "inner"),
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "outer"),
    ]

    placements = build_random_placements(
        slots,
        magnets,
        seed=5,
        orientation_mode="fixed_o0",
        work_units=reversed_units,
    )

    assert [placement.slot_flat_id for placement in placements] == [20, 21, 10, 11]
    assert [placement.insert_order for placement in placements] == [0, 1, 2, 3]


def test_simulation_trial_applies_same_work_unit_order_to_random_and_linear() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = _magnets()
    work_units = _work_units()
    pts = np.array([[0.0, 0.0, 0.02], [0.0, 0.01, -0.01]], dtype=np.float64)

    result = run_simulation_trial(
        slots,
        magnets,
        table,
        pts,
        trial_id=0,
        seed=8,
        random_orientation_mode="fixed_o0",
        work_units=work_units,
        allowed_orientation_ids=("O0",),
    )

    assert [placement.slot_flat_id for placement in result.random_baseline.placements[:2]] == [
        10,
        11,
    ]
    assert [placement.slot_flat_id for placement in result.linear.assignment.placements[:2]] == [
        11,
        10,
    ]
    assert all(
        placement.slot_flat_id in {10, 11} for placement in result.random_baseline.placements[:2]
    )
    assert all(
        placement.slot_flat_id in {10, 11} for placement in result.linear.assignment.placements[:2]
    )
