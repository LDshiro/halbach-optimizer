import numpy as np
import pytest

from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.online_assignment import (
    run_cluster_mpc_ring_constrained_linear_assignment,
    score_cluster_for_current_ring,
)
from halbach.assembly.simulation import run_simulation_trial
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    ClusterMPCConfig,
    ClusterStats,
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
        center_m=np.array([0.1, 0.0, -0.03 + 0.06 * layer_id], dtype=np.float64),
        nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        nominal_phi_rad=0.0,
        physical_slot_number=theta_id + 1,
        work_unit_id=f"W_K{layer_id:03d}_R000",
    )


def _slots() -> list[AssemblySlot]:
    return [_slot(10, 0, 0), _slot(11, 0, 1), _slot(20, 1, 0), _slot(21, 1, 1)]


def _table(slots: list[AssemblySlot]) -> SensitivityTable:
    C = np.zeros((len(slots), 1, 1, 3), dtype=np.float64)
    C[:, 0, 0, 0] = 1.0
    return SensitivityTable(
        slot_flat_id=np.array([slot.slot_flat_id for slot in slots], dtype=np.int_),
        ring_id=np.array([slot.ring_id for slot in slots], dtype=np.int_),
        layer_id=np.array([slot.layer_id for slot in slots], dtype=np.int_),
        theta_id=np.array([slot.theta_id for slot in slots], dtype=np.int_),
        centers_m=np.vstack([slot.center_m for slot in slots]),
        nominal_u=np.vstack([slot.nominal_u for slot in slots]),
        orientation_id=("O0",),
        C=C,
        roi_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        normalization_b0=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata={"case": "cluster-mpc-test"},
    )


def _stats(
    cluster_id: str,
    eps: float,
    *,
    d1: float = 0.0,
    d2: float = 0.0,
    cov_eps: float = 0.0,
    count: int = 1,
) -> ClusterStats:
    cov = np.zeros((3, 3), dtype=np.float64)
    cov[0, 0] = cov_eps
    return ClusterStats(
        cluster_id=cluster_id,
        count=count,
        mean=np.array([eps, d1, d2], dtype=np.float64),
        cov=cov,
    )


def _magnet(magnet_id: int, eps: float) -> VirtualMagnet:
    error = MagnetError(eps, 0.0, 0.0)
    return VirtualMagnet(magnet_id, error, error, quality=1.0)


def _quota(
    work_unit_id: str,
    layer_id: int,
    quota_by_cluster: dict[str, int],
    *,
    expected_angle_bin: float = 0.0,
) -> RingQuotaPlan:
    target_count = sum(quota_by_cluster.values())
    expected_eps = 0.0
    return RingQuotaPlan(
        ring_key=RingKey(layer_id=layer_id, ring_id=0),
        layer_id=layer_id,
        ring_id=0,
        work_unit_id=work_unit_id,
        target_count=target_count,
        target_mean_epsilon=0.0,
        ring_importance=0.0,
        allowed_clusters=tuple(sorted(quota_by_cluster)),
        quota_by_cluster=quota_by_cluster,
        expected_mean_epsilon=expected_eps,
        expected_mean_angle_bin=expected_angle_bin,
        expected_angle_error=0.0,
        mirror_pair_id="P000_R000",
    )


def _work_units() -> list[WorkUnit]:
    return [
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "outer"),
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (20, 21), "inner"),
    ]


def test_score_cluster_prefers_expected_contribution_against_residual() -> None:
    slots = _slots()
    table = _table(slots)
    plan = _quota("W_K000_R000", 0, {"S00_A00": 1, "S01_A00": 1})
    config = ClusterMPCConfig(
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
    )

    good = score_cluster_for_current_ring(
        table,
        np.array([1.0], dtype=np.float64),
        [10, 11],
        _stats("S00_A00", -1.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
        allowed_orientation_ids=("O0",),
    )
    bad = score_cluster_for_current_ring(
        table,
        np.array([1.0], dtype=np.float64),
        [10, 11],
        _stats("S01_A00", 1.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
        allowed_orientation_ids=("O0",),
    )

    assert good.field_cost < bad.field_cost
    assert good.total_score < bad.total_score
    assert good.best_slot_flat_id == 10
    assert good.best_orientation_id == "O0"


def test_score_cluster_penalizes_large_covariance() -> None:
    table = _table(_slots())
    plan = _quota("W_K000_R000", 0, {"S00_A00": 1, "S01_A00": 1})
    config = ClusterMPCConfig(
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
    )

    low_cov = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S00_A00", 0.0, cov_eps=0.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
    )
    high_cov = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S01_A00", 0.0, cov_eps=2.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
    )

    assert high_cov.field_cost > low_cov.field_cost


def test_score_cluster_angle_cost_uses_mean_angle_error_not_bin_number() -> None:
    table = _table(_slots())
    plan = _quota(
        "W_K000_R000",
        0,
        {"S00_A00": 1, "S09_A09": 1},
        expected_angle_bin=0.0,
    )
    plan = RingQuotaPlan(
        **{
            **plan.__dict__,
            "expected_angle_error": 0.1,
        }
    )
    config = ClusterMPCConfig(
        lambda_field=0.0,
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=1.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
        lambda_central_reserve=0.0,
    )

    matching = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S09_A09", 0.0, d1=0.1),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S09_A09": 1},
        {},
        config,
    )
    mismatched = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S00_A00", 0.0, d1=1.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S09_A09": 1},
        {},
        config,
    )

    assert matching.angle_cost < mismatched.angle_cost
    assert matching.total_score < mismatched.total_score


def test_legacy_mpc_angle_cost_uses_angle_bin_number() -> None:
    table = _table(_slots())
    plan = _quota(
        "W_K000_R000",
        0,
        {"S00_A00": 1, "S09_A09": 1},
        expected_angle_bin=0.0,
    )
    plan = RingQuotaPlan(
        **{
            **plan.__dict__,
            "expected_angle_error": 0.1,
        }
    )
    config = ClusterMPCConfig(
        strategy="legacy",
        lambda_field=0.0,
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=1.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
        lambda_central_reserve=1.0,
        future_neighbor_radius_bins=1,
    )

    high_bin_good_angle = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S09_A09", 0.0, d1=0.1),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S09_A09": 1},
        {},
        config,
    )
    low_bin_bad_angle = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S00_A00", 0.0, d1=1.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S09_A09": 1},
        {},
        config,
    )

    assert high_bin_good_angle.angle_cost > low_bin_bad_angle.angle_cost
    assert high_bin_good_angle.total_score > low_bin_bad_angle.total_score


def test_central_reserve_penalizes_using_future_center_inventory() -> None:
    table = _table(_slots())
    plan = _quota("W_K000_R000", 0, {"S00_A04": 1, "S05_A00": 1})
    config = ClusterMPCConfig(
        lambda_field=0.0,
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
        lambda_central_reserve=1.0,
    )

    center = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S05_A00", 0.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A04": 1, "S05_A00": 1},
        {"S05_A00": 1},
        config,
    )
    tail = score_cluster_for_current_ring(
        table,
        np.zeros(1, dtype=np.float64),
        [10],
        _stats("S00_A04", 0.0),
        plan,
        {},
        0.0,
        0,
        {"S00_A04": 1, "S05_A00": 1},
        {"S05_A00": 1},
        config,
    )

    assert center.central_reserve_cost > tail.central_reserve_cost
    assert center.total_score > tail.total_score


def test_future_neighbor_radius_treats_nearby_sigma_bins_as_reserve() -> None:
    table = _table(_slots())
    plan = _quota("W_K000_R000", 0, {"S04_A00": 1})
    common = {
        "table": table,
        "residual": np.zeros(1, dtype=np.float64),
        "remaining_slot_flat_ids": [10],
        "cluster_stats": _stats("S04_A00", 0.0),
        "quota_plan": plan,
        "current_ring_cluster_usage": {},
        "current_ring_epsilon_sum": 0.0,
        "current_ring_count": 0,
        "remaining_cluster_counts": {"S04_A00": 1, "S05_A00": 0},
        "future_cluster_demand": {"S05_A00": 1},
    }

    exact = score_cluster_for_current_ring(
        **common,
        config=ClusterMPCConfig(
            lambda_field=0.0,
            lambda_quota=0.0,
            lambda_ring_mean=0.0,
            lambda_angle=0.0,
            lambda_future=1.0,
            lambda_mirror=0.0,
            lambda_central_reserve=0.0,
            future_neighbor_radius_bins=0,
        ),
    )
    neighbor = score_cluster_for_current_ring(
        **common,
        config=ClusterMPCConfig(
            lambda_field=0.0,
            lambda_quota=0.0,
            lambda_ring_mean=0.0,
            lambda_angle=0.0,
            lambda_future=1.0,
            lambda_mirror=0.0,
            lambda_central_reserve=0.0,
            future_neighbor_radius_bins=1,
        ),
    )

    assert exact.future_cost == 0.0
    assert neighbor.future_cost > exact.future_cost


def test_cluster_mpc_assignment_respects_work_unit_order() -> None:
    slots = _slots()
    magnets = [_magnet(0, -0.1), _magnet(1, 0.0), _magnet(2, -0.1), _magnet(3, 0.0)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
        ClusterAssignment(2, "S00_A00"),
        ClusterAssignment(3, "S01_A00"),
    ]
    inventory = build_cluster_inventory(magnets, assignments)
    quota_plans = [
        _quota("W_K000_R000", 0, {"S00_A00": 1, "S01_A00": 1}),
        _quota("W_K001_R000", 1, {"S00_A00": 1, "S01_A00": 1}),
    ]

    result = run_cluster_mpc_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        quota_plans,
        assignments=assignments,
        inventory=inventory,
        config=ClusterMPCConfig(lambda_angle=0.0, lambda_future=0.0),
        allowed_orientation_ids=("O0",),
        seed=2,
    )

    assert {placement.slot_flat_id for placement in result.placements[:2]} == {10, 11}
    assert {placement.slot_flat_id for placement in result.placements[2:]} == {20, 21}
    assert all(placement.decision_engine == "cluster_mpc" for placement in result.placements)


def test_future_reserve_keeps_future_quota_cluster_available() -> None:
    slots = _slots()[:2]
    work_units = [
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10,), "outer"),
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (11,), "inner"),
    ]
    table = _table(slots)
    magnets = [_magnet(0, 0.1), _magnet(1, 0.0)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
    ]
    quota_plans = [
        _quota("W_K000_R000", 0, {"S00_A00": 1}),
        _quota("W_K001_R000", 1, {"S01_A00": 1}),
    ]

    reserved = run_cluster_mpc_ring_constrained_linear_assignment(
        table,
        magnets,
        work_units,
        quota_plans,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        config=ClusterMPCConfig(
            lambda_quota=0.0,
            lambda_ring_mean=0.0,
            lambda_angle=0.0,
            lambda_future=1.0,
            lambda_mirror=0.0,
        ),
        seed=0,
    )
    greedy = run_cluster_mpc_ring_constrained_linear_assignment(
        table,
        magnets,
        work_units,
        quota_plans,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        config=ClusterMPCConfig(
            lambda_quota=0.0,
            lambda_ring_mean=0.0,
            lambda_angle=0.0,
            lambda_future=0.0,
            lambda_mirror=0.0,
        ),
        seed=0,
    )

    assert reserved.placements[0].cluster_requested == "S00_A00"
    assert greedy.placements[0].cluster_requested == "S01_A00"


def test_soft_quota_can_choose_out_of_quota_cluster_for_field_benefit() -> None:
    slots = _slots()[:2]
    work_units = [WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "ring")]
    table = _table(slots)
    magnets = [_magnet(0, 0.1), _magnet(1, 0.0)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
    ]
    quota_plans = [_quota("W_K000_R000", 0, {"S00_A00": 2})]

    result = run_cluster_mpc_ring_constrained_linear_assignment(
        table,
        magnets,
        work_units,
        quota_plans,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        config=ClusterMPCConfig(
            lambda_quota=0.001,
            lambda_ring_mean=0.0,
            lambda_angle=0.0,
            lambda_future=0.0,
            lambda_mirror=0.0,
        ),
        seed=0,
    )

    assert result.placements[0].cluster_requested == "S01_A00"


def test_cluster_mpc_rejects_bad_inputs() -> None:
    slots = _slots()[:2]
    table = _table(slots)
    magnets = [_magnet(0, 0.0), _magnet(1, 0.1)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
    ]
    work_units = [WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10,), "short")]
    quota_plans = [_quota("W_K000_R000", 0, {"S00_A00": 1})]

    with pytest.raises(ValueError, match="coverage mismatch"):
        run_cluster_mpc_ring_constrained_linear_assignment(
            table,
            magnets,
            work_units,
            quota_plans,
            assignments=assignments,
            inventory=build_cluster_inventory(magnets, assignments),
        )

    bad_assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, None, quarantine_id="Q_DIRECTION_OUTLIER"),
    ]
    good_work_units = [WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "ring")]
    with pytest.raises(ValueError, match="quarantined"):
        run_cluster_mpc_ring_constrained_linear_assignment(
            table,
            magnets,
            good_work_units,
            [_quota("W_K000_R000", 0, {"S00_A00": 2})],
            assignments=bad_assignments,
            inventory=build_cluster_inventory(magnets, bad_assignments),
        )


def test_simulation_trial_can_use_cluster_mpc() -> None:
    slots = _slots()
    table = _table(slots)
    magnets = [_magnet(0, -0.1), _magnet(1, 0.0), _magnet(2, -0.1), _magnet(3, 0.0)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
        ClusterAssignment(2, "S00_A00"),
        ClusterAssignment(3, "S01_A00"),
    ]
    quota_plans = [
        _quota("W_K000_R000", 0, {"S00_A00": 1, "S01_A00": 1}),
        _quota("W_K001_R000", 1, {"S00_A00": 1, "S01_A00": 1}),
    ]

    result = run_simulation_trial(
        slots,
        magnets,
        table,
        np.array([[0.0, 0.0, 0.02]], dtype=np.float64),
        trial_id=0,
        seed=4,
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        random_orientation_mode="fixed_o0",
        allowed_orientation_ids=("O0",),
        work_units=_work_units(),
        cluster_pickup_policy="cluster_mpc",
        cluster_mpc_config=ClusterMPCConfig(lambda_angle=0.0, lambda_future=0.0),
        quota_plans=quota_plans,
    )

    assert result.linear.assignment.placements
    assert all(
        placement.decision_engine == "cluster_mpc"
        for placement in result.linear.assignment.placements
    )


def test_simulation_cluster_mpc_requires_inventory() -> None:
    slots = _slots()[:2]
    magnets = [_magnet(0, 0.0), _magnet(1, 0.1)]
    assignments = [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S01_A00"),
    ]

    with pytest.raises(ValueError, match="requires inventory"):
        run_simulation_trial(
            slots,
            magnets,
            _table(slots),
            np.array([[0.0, 0.0, 0.02]], dtype=np.float64),
            trial_id=0,
            seed=0,
            assignments=assignments,
            inventory=None,
            work_units=[WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10, 11), "ring")],
            cluster_pickup_policy="cluster_mpc",
            quota_plans=[_quota("W_K000_R000", 0, {"S00_A00": 1, "S01_A00": 1})],
        )
