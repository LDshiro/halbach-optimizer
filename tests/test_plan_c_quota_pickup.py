import numpy as np
import pytest

from halbach.assembly.inventory import build_cluster_inventory, inventory_total_count
from halbach.assembly.online_assignment import (
    build_quota_ordered_magnet_order,
    cluster_usage_from_placements,
    run_quota_ordered_ring_constrained_linear_assignment,
)
from halbach.assembly.ring_summary import ring_summary_from_placements
from halbach.assembly.simulation import run_simulation_trial
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    MagnetError,
    RingKey,
    RingQuotaPlan,
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
        metadata={"case": "quota-pickup-test"},
    )


def _magnet(magnet_id: int, eps: float) -> VirtualMagnet:
    error = MagnetError(eps, 0.0, 0.0)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=error,
        measured_error=error,
        quality=1.0,
    )


def _magnets() -> list[VirtualMagnet]:
    return [
        _magnet(0, -0.02),
        _magnet(1, 0.02),
        _magnet(2, -0.02),
        _magnet(3, 0.02),
    ]


def _assignments() -> list[ClusterAssignment]:
    return [
        ClusterAssignment(magnet_id=0, cluster_id="S00_A00"),
        ClusterAssignment(magnet_id=1, cluster_id="S01_A00"),
        ClusterAssignment(magnet_id=2, cluster_id="S00_A00"),
        ClusterAssignment(magnet_id=3, cluster_id="S01_A00"),
    ]


def _work_units() -> list[WorkUnit]:
    return [
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "outer"),
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (20, 21), "inner"),
    ]


def _quota_plans() -> list[RingQuotaPlan]:
    return [
        RingQuotaPlan(
            ring_key=RingKey(layer_id=0, ring_id=0),
            layer_id=0,
            ring_id=0,
            work_unit_id="W_K000_R000",
            target_count=2,
            target_mean_epsilon=0.0,
            ring_importance=0.0,
            allowed_clusters=("S00_A00", "S01_A00"),
            quota_by_cluster={"S00_A00": 1, "S01_A00": 1},
            expected_mean_epsilon=0.0,
            expected_mean_angle_bin=0.0,
            expected_angle_error=0.0,
            mirror_pair_id="P000_R000",
        ),
        RingQuotaPlan(
            ring_key=RingKey(layer_id=1, ring_id=0),
            layer_id=1,
            ring_id=0,
            work_unit_id="W_K001_R000",
            target_count=2,
            target_mean_epsilon=0.0,
            ring_importance=1.0,
            allowed_clusters=("S00_A00", "S01_A00"),
            quota_by_cluster={"S00_A00": 1, "S01_A00": 1},
            expected_mean_epsilon=0.0,
            expected_mean_angle_bin=0.0,
            expected_angle_error=0.0,
            mirror_pair_id="P000_R000",
        ),
    ]


def _cluster_by_magnet(assignments: list[ClusterAssignment]) -> dict[int, str]:
    return {
        assignment.magnet_id: str(assignment.cluster_id)
        for assignment in assignments
        if assignment.cluster_id is not None
    }


def test_quota_ordered_magnet_order_consumes_cluster_pools_by_plan() -> None:
    magnets = _magnets()
    assignments = _assignments()
    cluster_by_magnet = _cluster_by_magnet(assignments)

    order = build_quota_ordered_magnet_order(
        magnets,
        assignments,
        _quota_plans(),
        seed=12,
    )

    assert [cluster_by_magnet[magnet_id] for magnet_id in order[:2]] == [
        "S00_A00",
        "S01_A00",
    ]
    assert [cluster_by_magnet[magnet_id] for magnet_id in order[2:]] == [
        "S00_A00",
        "S01_A00",
    ]
    assert set(order) == {magnet.magnet_id for magnet in magnets}


def test_quota_ordered_assignment_uses_ring_quota_and_decrements_inventory() -> None:
    slots = _slots()
    magnets = _magnets()
    assignments = _assignments()
    inventory = build_cluster_inventory(magnets, assignments)

    result = run_quota_ordered_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        _quota_plans(),
        assignments=assignments,
        inventory=inventory,
        allowed_orientation_ids=("O0",),
        seed=3,
    )

    assert inventory_total_count(result.inventory) == 0
    assert cluster_usage_from_placements(result.placements[:2]) == {
        "S00_A00": 1,
        "S01_A00": 1,
    }
    assert cluster_usage_from_placements(result.placements[2:]) == {
        "S00_A00": 1,
        "S01_A00": 1,
    }
    assert {placement.slot_flat_id for placement in result.placements[:2]} == {10, 11}
    assert {placement.slot_flat_id for placement in result.placements[2:]} == {20, 21}


def test_quota_ordered_assignment_ring_mean_matches_expected_quota_mean() -> None:
    slots = _slots()
    magnets = _magnets()
    assignments = _assignments()
    quota_plans = _quota_plans()

    result = run_quota_ordered_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        quota_plans,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        allowed_orientation_ids=("O0",),
        seed=7,
    )
    summaries = ring_summary_from_placements(slots, magnets, result.placements)
    summary_by_ring = {summary.ring_key: summary for summary in summaries}

    for plan in quota_plans:
        assert summary_by_ring[plan.ring_key].mean_epsilon == pytest.approx(
            plan.expected_mean_epsilon
        )


def test_quota_ordered_assignment_rejects_cluster_shortage() -> None:
    quota_plans = _quota_plans()
    quota_plans[0] = RingQuotaPlan(
        ring_key=quota_plans[0].ring_key,
        layer_id=quota_plans[0].layer_id,
        ring_id=quota_plans[0].ring_id,
        work_unit_id=quota_plans[0].work_unit_id,
        target_count=2,
        target_mean_epsilon=0.0,
        ring_importance=0.0,
        allowed_clusters=("S00_A00",),
        quota_by_cluster={"S00_A00": 2},
        expected_mean_epsilon=-0.02,
        expected_mean_angle_bin=0.0,
        expected_angle_error=0.0,
    )

    with pytest.raises(ValueError, match="no assigned magnets remain"):
        build_quota_ordered_magnet_order(
            _magnets(),
            _assignments(),
            quota_plans,
            seed=0,
        )


def test_simulation_trial_can_use_quota_ordered_cluster_pickup() -> None:
    slots = _slots()
    magnets = _magnets()
    assignments = _assignments()
    pts = np.array([[0.0, 0.0, 0.02], [0.0, 0.01, -0.01]], dtype=np.float64)

    result = run_simulation_trial(
        slots,
        magnets,
        _table(slots),
        pts,
        trial_id=0,
        seed=11,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        random_orientation_mode="fixed_o0",
        allowed_orientation_ids=("O0",),
        work_units=_work_units(),
        cluster_pickup_policy="quota_ordered",
        quota_plans=_quota_plans(),
    )

    assert cluster_usage_from_placements(result.linear.assignment.placements[:2]) == {
        "S00_A00": 1,
        "S01_A00": 1,
    }
    assert cluster_usage_from_placements(result.linear.assignment.placements[2:]) == {
        "S00_A00": 1,
        "S01_A00": 1,
    }
