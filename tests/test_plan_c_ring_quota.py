import numpy as np
import pytest

from halbach.assembly.ring_quota import (
    compute_inventory_target_mean_epsilon,
    compute_ring_importance,
    plan_ring_cluster_quotas,
    plan_work_unit_cluster_quotas,
)
from halbach.assembly.types import (
    AssemblySlot,
    ClusterInventory,
    ClusterStats,
    RingKey,
    RingQuotaPlannerConfig,
)
from halbach.assembly.work_units import build_work_units


def _slots(*, K: int, N: int, R: int = 1) -> list[AssemblySlot]:
    if K == 1:
        z_layers = [0.0]
    else:
        z_layers = np.linspace(-1.0, 1.0, K).tolist()
    slots: list[AssemblySlot] = []
    for layer_id, z_value in enumerate(z_layers):
        for ring_id in range(R):
            for theta_id in range(N):
                slot_id = (ring_id * K + layer_id) * N + theta_id
                slots.append(
                    AssemblySlot(
                        slot_flat_id=slot_id,
                        ring_id=ring_id,
                        layer_id=layer_id,
                        theta_id=theta_id,
                        center_m=np.array([0.1 + ring_id * 0.01, 0.0, z_value]),
                        nominal_u=np.array([0.0, 1.0, 0.0]),
                        nominal_phi_rad=0.0,
                        physical_slot_number=theta_id + 1,
                        work_unit_id=f"W_K{layer_id:03d}_R{ring_id:03d}",
                    )
                )
    return slots


def _stats(cluster_id: str, *, count: int, eps: float, angle_score: float) -> ClusterStats:
    return ClusterStats(
        cluster_id=cluster_id,
        count=count,
        mean=np.array([eps, angle_score, 0.0], dtype=np.float64),
        cov=np.zeros((3, 3), dtype=np.float64),
    )


def _matrix_inventory(
    strength_eps: list[float],
    angle_bins: list[int],
    *,
    count_per_cluster: int,
) -> ClusterInventory:
    clusters: dict[str, ClusterStats] = {}
    for strength_bin, eps in enumerate(strength_eps):
        for angle_bin in angle_bins:
            cluster_id = f"S{strength_bin:02d}_A{angle_bin:02d}"
            clusters[cluster_id] = _stats(
                cluster_id,
                count=count_per_cluster,
                eps=eps,
                angle_score=float(angle_bin),
            )
    return ClusterInventory(clusters=clusters, quarantine={})


def _cluster_usage(plans) -> dict[str, int]:
    usage: dict[str, int] = {}
    for plan in plans:
        for cluster_id, count in plan.quota_by_cluster.items():
            usage[cluster_id] = usage.get(cluster_id, 0) + count
    return usage


def test_compute_ring_importance_is_highest_at_stack_center() -> None:
    slots = _slots(K=5, N=2)

    importance = compute_ring_importance(slots)

    assert importance[RingKey(layer_id=0, ring_id=0)] == pytest.approx(0.0)
    assert importance[RingKey(layer_id=4, ring_id=0)] == pytest.approx(0.0)
    assert importance[RingKey(layer_id=2, ring_id=0)] == pytest.approx(1.0)
    assert importance[RingKey(layer_id=1, ring_id=0)] == pytest.approx(0.5)
    assert importance[RingKey(layer_id=3, ring_id=0)] == pytest.approx(0.5)


def test_inventory_target_mean_epsilon_uses_usable_weighted_mean() -> None:
    inventory = ClusterInventory(
        clusters={
            "S00_A00": _stats("S00_A00", count=1, eps=-1.0, angle_score=0.0),
            "S01_A00": _stats("S01_A00", count=3, eps=1.0, angle_score=0.0),
        },
        quarantine={"Q_DIRECTION_OUTLIER": 10},
    )

    assert compute_inventory_target_mean_epsilon(inventory) == pytest.approx(0.5)


def test_plan_ring_cluster_quotas_matches_ring_counts_and_inventory_limits() -> None:
    slots = _slots(K=4, N=4)
    inventory = _matrix_inventory(
        [-0.03, -0.01, 0.01, 0.03],
        [0, 1, 2, 3],
        count_per_cluster=1,
    )

    plans = plan_ring_cluster_quotas(slots, inventory)

    assert [plan.ring_key for plan in plans] == [
        RingKey(layer_id=0, ring_id=0),
        RingKey(layer_id=3, ring_id=0),
        RingKey(layer_id=1, ring_id=0),
        RingKey(layer_id=2, ring_id=0),
    ]
    assert sum(plan.target_count for plan in plans) == len(slots)
    assert all(sum(plan.quota_by_cluster.values()) == plan.target_count for plan in plans)
    usage = _cluster_usage(plans)
    assert usage == {cluster_id: 1 for cluster_id in inventory.clusters}
    assert all(
        abs(plan.expected_mean_epsilon - plan.target_mean_epsilon) < 1.0e-12 for plan in plans
    )


def test_angle_quality_schedule_spends_larger_angle_bins_on_outer_rings() -> None:
    slots = _slots(K=5, N=2)
    inventory = _matrix_inventory([0.0], [0, 1, 2, 3, 4], count_per_cluster=2)

    plans = plan_ring_cluster_quotas(slots, inventory)
    plan_by_ring = {plan.ring_key: plan for plan in plans}

    outer_average = 0.5 * (
        plan_by_ring[RingKey(layer_id=0, ring_id=0)].expected_mean_angle_bin
        + plan_by_ring[RingKey(layer_id=4, ring_id=0)].expected_mean_angle_bin
    )
    center_angle = plan_by_ring[RingKey(layer_id=2, ring_id=0)].expected_mean_angle_bin

    assert outer_average > center_angle
    assert plan_by_ring[RingKey(layer_id=0, ring_id=0)].expected_mean_angle_bin == pytest.approx(
        4.0
    )
    assert center_angle == pytest.approx(0.0)


def test_mirror_pair_plans_have_balanced_strength_means() -> None:
    slots = _slots(K=4, N=2)
    inventory = _matrix_inventory([-0.02, 0.02], [0, 1, 2, 3], count_per_cluster=1)

    plans = plan_ring_cluster_quotas(slots, inventory)
    plan_by_ring = {plan.ring_key: plan for plan in plans}

    left = plan_by_ring[RingKey(layer_id=0, ring_id=0)]
    right = plan_by_ring[RingKey(layer_id=3, ring_id=0)]
    assert left.mirror_pair_id == "P000_R000"
    assert right.mirror_pair_id == "P000_R000"
    assert left.expected_mean_epsilon == pytest.approx(right.expected_mean_epsilon)


def test_plan_work_unit_cluster_quotas_aggregates_mirror_pair_units() -> None:
    slots = _slots(K=4, N=2)
    inventory = _matrix_inventory([-0.02, 0.02], [0, 1, 2, 3], count_per_cluster=1)
    work_units = build_work_units(slots, "mirror_ring_pair")

    base_plans = plan_ring_cluster_quotas(slots, inventory)
    aligned = plan_work_unit_cluster_quotas(slots, inventory, work_units)

    assert [plan.work_unit_id for plan in aligned] == ["W_PAIR000_R000", "W_PAIR001_R000"]
    assert [plan.target_count for plan in aligned] == [4, 4]
    assert all(plan.mirror_pair_id is None for plan in aligned)
    assert sum(plan.target_count for plan in aligned) == len(slots)
    assert _cluster_usage(aligned) == _cluster_usage(base_plans)

    outer_pair = [
        plan
        for plan in base_plans
        if plan.ring_key in {RingKey(layer_id=0, ring_id=0), RingKey(layer_id=3, ring_id=0)}
    ]
    expected_outer_mean = sum(
        plan.expected_mean_epsilon * plan.target_count for plan in outer_pair
    ) / sum(plan.target_count for plan in outer_pair)
    assert aligned[0].expected_mean_epsilon == pytest.approx(expected_outer_mean)


def test_plan_work_unit_cluster_quotas_aggregates_layer_units() -> None:
    slots = _slots(K=3, N=2, R=2)
    inventory = _matrix_inventory([-0.01, 0.01], [0, 1, 2], count_per_cluster=2)
    work_units = build_work_units(slots, "layer_by_layer_outer_to_inner")

    aligned = plan_work_unit_cluster_quotas(slots, inventory, work_units)

    assert [plan.work_unit_id for plan in aligned] == [
        "W_LAYER000",
        "W_LAYER002",
        "W_LAYER001",
    ]
    assert [plan.target_count for plan in aligned] == [4, 4, 4]
    assert sum(plan.target_count for plan in aligned) == len(slots)
    assert all(sum(plan.quota_by_cluster.values()) == plan.target_count for plan in aligned)


def test_plan_work_unit_cluster_quotas_aggregates_all_slots_unit() -> None:
    slots = _slots(K=3, N=2)
    inventory = _matrix_inventory([-0.01, 0.01], [0, 1, 2], count_per_cluster=1)
    work_units = build_work_units(slots, "all_slots")

    aligned = plan_work_unit_cluster_quotas(slots, inventory, work_units)

    assert len(aligned) == 1
    assert aligned[0].work_unit_id == "W_ALL"
    assert aligned[0].target_count == len(slots)
    assert sum(aligned[0].quota_by_cluster.values()) == len(slots)
    assert aligned[0].mirror_pair_id is None


def test_plan_ring_cluster_quotas_rejects_insufficient_inventory() -> None:
    slots = _slots(K=2, N=2)
    inventory = ClusterInventory(
        clusters={"S00_A00": _stats("S00_A00", count=3, eps=0.0, angle_score=0.0)},
        quarantine={},
    )

    with pytest.raises(ValueError, match="enough usable magnets"):
        plan_ring_cluster_quotas(slots, inventory)


def test_plan_ring_cluster_quotas_rejects_invalid_config_and_cluster_ids() -> None:
    slots = _slots(K=1, N=1)
    inventory = ClusterInventory(
        clusters={"bad": _stats("bad", count=1, eps=0.0, angle_score=0.0)},
        quarantine={},
    )

    with pytest.raises(ValueError, match="Sxx_Ayy"):
        plan_ring_cluster_quotas(slots, inventory)

    valid_inventory = ClusterInventory(
        clusters={"S00_A00": _stats("S00_A00", count=1, eps=0.0, angle_score=0.0)},
        quarantine={},
    )
    with pytest.raises(ValueError, match="lambda_angle"):
        plan_ring_cluster_quotas(
            slots,
            valid_inventory,
            RingQuotaPlannerConfig(lambda_angle=-1.0),
        )
