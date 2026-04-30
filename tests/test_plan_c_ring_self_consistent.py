import numpy as np

from halbach.assembly.online_assignment import run_ring_constrained_linear_assignment
from halbach.assembly.self_consistent_assignment import (
    SelfConsistentConfig,
    run_ring_constrained_self_consistent_assignment,
    run_self_consistent_assignment,
)
from halbach.assembly.simulation import run_simulation_trial
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
        center_m=np.array([0.06 + 0.005 * theta_id, 0.01 * layer_id, 0.002], dtype=np.float64),
        nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        nominal_phi_rad=np.pi / 2.0,
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
    C = np.zeros((len(slots), 2, 2, 3), dtype=np.float64)
    for slot_idx in range(len(slots)):
        C[slot_idx, :, 0, 0] = float(slot_idx + 1)
        C[slot_idx, 0, 1, 1] = 1.0
        C[slot_idx, 1, 1, 1] = -1.0
    return SensitivityTable(
        slot_flat_id=np.array([slot.slot_flat_id for slot in slots], dtype=np.int_),
        ring_id=np.array([slot.ring_id for slot in slots], dtype=np.int_),
        layer_id=np.array([slot.layer_id for slot in slots], dtype=np.int_),
        theta_id=np.array([slot.theta_id for slot in slots], dtype=np.int_),
        centers_m=np.vstack([slot.center_m for slot in slots]),
        nominal_u=np.vstack([slot.nominal_u for slot in slots]),
        orientation_id=("O0", "O180"),
        C=C,
        roi_points=_points(),
        normalization_b0=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata={"case": "ring-self-consistent-test"},
    )


def _points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.004, 0.0],
            [0.0, 0.0, 0.004],
        ],
        dtype=np.float64,
    )


def _magnets() -> list[VirtualMagnet]:
    values = [0.003, -0.002, 0.001, -0.0015]
    magnets: list[VirtualMagnet] = []
    for magnet_id, eps in enumerate(values):
        error = MagnetError(eps, 0.0002 * ((magnet_id % 2) * 2 - 1), 0.0)
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
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (20, 21), "inner first"),
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "outer second"),
    ]


def test_ring_constrained_self_consistent_uses_same_work_unit_order_as_linear() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = _magnets()
    work_units = _work_units()

    linear = run_ring_constrained_linear_assignment(table, magnets, work_units)
    self_consistent = run_ring_constrained_self_consistent_assignment(
        table,
        slots,
        magnets,
        _points(),
        SelfConsistentConfig(chi=0.0, max_linear_candidates=4),
        work_units,
    )

    assert {placement.slot_flat_id for placement in linear.placements[:2]} == {20, 21}
    assert {placement.slot_flat_id for placement in self_consistent.placements[:2]} == {20, 21}
    assert {placement.slot_flat_id for placement in linear.placements[2:]} == {10, 11}
    assert {placement.slot_flat_id for placement in self_consistent.placements[2:]} == {10, 11}
    assert all(
        evaluation.candidate.slot_flat_id in {20, 21}
        for decision in self_consistent.decisions[:2]
        for evaluation in decision.evaluations
    )
    assert all(
        evaluation.candidate.slot_flat_id in {10, 11}
        for decision in self_consistent.decisions[2:]
        for evaluation in decision.evaluations
    )


def test_self_consistent_assignment_can_record_completed_work_unit_evaluations() -> None:
    slots = _slots()
    result = run_self_consistent_assignment(
        _table(slots),
        slots,
        _magnets(),
        _points(),
        SelfConsistentConfig(chi=0.0, max_linear_candidates=2),
        work_units=_work_units(),
        evaluate_completed_work_units=True,
        work_unit_evaluation_stride=1,
    )

    assert [item.work_unit_id for item in result.work_unit_evaluations] == [
        "W_K001_R000",
        "W_K000_R000",
    ]
    assert [item.completed_placement_count for item in result.work_unit_evaluations] == [2, 4]
    assert result.work_unit_evaluations[0].evaluation.B.shape == (3, 3)
    assert np.isfinite(result.work_unit_evaluations[-1].evaluation.metrics.J_vector)


def test_simulation_self_consistent_branch_preserves_ring_order() -> None:
    slots = _slots()
    result = run_simulation_trial(
        slots,
        _magnets(),
        _table(slots),
        _points(),
        trial_id=0,
        seed=3,
        random_orientation_mode="fixed_o0",
        work_units=_work_units(),
        include_self_consistent=True,
        self_consistent_config=SelfConsistentConfig(chi=0.0, max_linear_candidates=2),
    )

    assert result.self_consistent is not None
    assert {placement.slot_flat_id for placement in result.linear.assignment.placements[:2]} == {
        20,
        21,
    }
    assert {placement.slot_flat_id for placement in result.self_consistent.placements[:2]} == {
        20,
        21,
    }
