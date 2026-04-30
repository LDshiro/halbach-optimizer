import numpy as np
import pytest

from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.online_assignment import (
    run_cluster_mpc_ring_constrained_linear_assignment,
    score_cluster_for_current_ring,
)
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


def _slot(slot_id: int, layer_id: int) -> AssemblySlot:
    return AssemblySlot(
        slot_flat_id=slot_id,
        ring_id=0,
        layer_id=layer_id,
        theta_id=0,
        center_m=np.array([0.1, 0.0, -0.03 + 0.03 * layer_id], dtype=np.float64),
        nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        nominal_phi_rad=0.0,
        physical_slot_number=1,
        work_unit_id=f"W_K{layer_id:03d}_R000",
        mirror_pair_id="P000_R000" if layer_id in (0, 2) else None,
    )


def _slots() -> list[AssemblySlot]:
    return [_slot(10, 0), _slot(20, 2), _slot(30, 1)]


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
        metadata={"case": "mirror-balance-test"},
    )


def _magnet(magnet_id: int, eps: float) -> VirtualMagnet:
    error = MagnetError(eps, 0.0, 0.0)
    return VirtualMagnet(magnet_id, error, error, quality=1.0)


def _stats(cluster_id: str, eps: float, count: int = 1) -> ClusterStats:
    return ClusterStats(
        cluster_id=cluster_id,
        count=count,
        mean=np.array([eps, 0.0, 0.0], dtype=np.float64),
        cov=np.zeros((3, 3), dtype=np.float64),
    )


def _quota(
    layer_id: int,
    quota_by_cluster: dict[str, int],
    *,
    mirror_pair_id: str | None,
) -> RingQuotaPlan:
    target_count = sum(quota_by_cluster.values())
    return RingQuotaPlan(
        ring_key=RingKey(layer_id=layer_id, ring_id=0),
        layer_id=layer_id,
        ring_id=0,
        work_unit_id=f"W_K{layer_id:03d}_R000",
        target_count=target_count,
        target_mean_epsilon=0.0,
        ring_importance=0.0,
        allowed_clusters=tuple(sorted(quota_by_cluster)),
        quota_by_cluster=quota_by_cluster,
        expected_mean_epsilon=0.0,
        expected_mean_angle_bin=0.0,
        expected_angle_error=0.0,
        mirror_pair_id=mirror_pair_id,
    )


def _work_units() -> list[WorkUnit]:
    return [
        WorkUnit("W_K000_R000", "ring_by_ring_outer_to_inner", (10,), "lower mirror"),
        WorkUnit("W_K002_R000", "ring_by_ring_outer_to_inner", (20,), "upper mirror"),
        WorkUnit("W_K001_R000", "ring_by_ring_outer_to_inner", (30,), "center"),
    ]


def _quota_plans() -> list[RingQuotaPlan]:
    return [
        _quota(0, {"S00_A00": 1}, mirror_pair_id="P000_R000"),
        _quota(2, {"S00_A00": 1, "S01_A00": 0}, mirror_pair_id="P000_R000"),
        _quota(1, {"S01_A00": 1}, mirror_pair_id=None),
    ]


def _magnets() -> list[VirtualMagnet]:
    return [_magnet(0, 0.2), _magnet(1, 0.2), _magnet(2, -0.2)]


def _assignments() -> list[ClusterAssignment]:
    return [
        ClusterAssignment(0, "S00_A00"),
        ClusterAssignment(1, "S00_A00"),
        ClusterAssignment(2, "S01_A00"),
    ]


def test_score_cluster_mirror_penalty_prefers_matching_completed_ring_mean() -> None:
    table = _table(_slots())
    plan = _quota(2, {"S00_A00": 1, "S01_A00": 0}, mirror_pair_id="P000_R000")
    config = ClusterMPCConfig(
        lambda_field=0.0,
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=1.0,
    )

    matching = score_cluster_for_current_ring(
        table,
        np.array([0.2], dtype=np.float64),
        [20],
        _stats("S00_A00", 0.2),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
        mirror_mean_epsilon=0.2,
    )
    opposing = score_cluster_for_current_ring(
        table,
        np.array([0.2], dtype=np.float64),
        [20],
        _stats("S01_A00", -0.2),
        plan,
        {},
        0.0,
        0,
        {"S00_A00": 1, "S01_A00": 1},
        {},
        config,
        mirror_mean_epsilon=0.2,
    )

    assert matching.mirror_cost == pytest.approx(0.0)
    assert opposing.mirror_cost == pytest.approx(0.16)
    assert matching.total_score < opposing.total_score


def test_cluster_mpc_mirror_penalty_reduces_pair_mean_difference_and_reports_summary() -> None:
    slots = _slots()
    magnets = _magnets()
    assignments = _assignments()
    no_mirror = ClusterMPCConfig(
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=0.0,
    )
    strong_mirror = ClusterMPCConfig(
        lambda_quota=0.0,
        lambda_ring_mean=0.0,
        lambda_angle=0.0,
        lambda_future=0.0,
        lambda_mirror=100.0,
    )

    unbalanced = run_cluster_mpc_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        _quota_plans(),
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        config=no_mirror,
        seed=0,
    )
    balanced = run_cluster_mpc_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        _quota_plans(),
        assignments=assignments,
        inventory=build_cluster_inventory(magnets, assignments),
        config=strong_mirror,
        seed=0,
    )

    assert unbalanced.placements[1].cluster_requested == "S01_A00"
    assert balanced.placements[1].cluster_requested == "S00_A00"
    assert len(unbalanced.mirror_pair_summaries) == 1
    assert len(balanced.mirror_pair_summaries) == 1
    assert unbalanced.mirror_pair_summaries[0].pair_complete
    assert balanced.mirror_pair_summaries[0].pair_complete
    assert abs(unbalanced.mirror_pair_summaries[0].mean_epsilon_difference) > abs(
        balanced.mirror_pair_summaries[0].mean_epsilon_difference
    )
    assert balanced.mirror_pair_summaries[0].mean_epsilon_difference == pytest.approx(0.0)


def test_cluster_mpc_carries_outer_residual_into_inner_compensation() -> None:
    slots = _slots()
    magnets = _magnets()
    assignments = _assignments()
    result = run_cluster_mpc_ring_constrained_linear_assignment(
        _table(slots),
        magnets,
        _work_units(),
        _quota_plans(),
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

    pair = result.mirror_pair_summaries[0]
    assert result.placements[0].cluster_requested == "S00_A00"
    assert result.placements[1].cluster_requested == "S01_A00"
    assert pair.residual_norm_after_lower == pytest.approx(0.2)
    assert pair.residual_norm_after_pair == pytest.approx(0.0)
