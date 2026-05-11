from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from halbach.assembly.types import (
    AssemblySlot,
    ClusterInventory,
    ClusterStats,
    RingKey,
    RingQuotaPlan,
    RingQuotaPlannerConfig,
    SensitivityTable,
    WorkUnit,
)
from halbach.assembly.work_units import outer_to_inner_layer_order

_CLUSTER_ID_RE = re.compile(r"^S(?P<strength>\d+)_A(?P<angle>\d+)$")


@dataclass(frozen=True)
class _ClusterInfo:
    cluster_id: str
    count: int
    mean_epsilon: float
    angle_error: float
    strength_bin: int
    angle_bin: int


def _parse_cluster_id(cluster_id: str) -> tuple[int, int]:
    match = _CLUSTER_ID_RE.fullmatch(cluster_id)
    if match is None:
        raise ValueError(f"cluster_id must use Sxx_Ayy format: {cluster_id}")
    return int(match.group("strength")), int(match.group("angle"))


def _cluster_info(cluster_id: str, stats: ClusterStats) -> _ClusterInfo:
    if stats.count < 0:
        raise ValueError(f"cluster {cluster_id} has negative count")
    if stats.mean.shape != (3,):
        raise ValueError(f"cluster {cluster_id} mean must have shape (3,)")
    strength_bin, angle_bin = _parse_cluster_id(cluster_id)
    return _ClusterInfo(
        cluster_id=cluster_id,
        count=int(stats.count),
        mean_epsilon=float(stats.mean[0]),
        angle_error=float(math.hypot(float(stats.mean[1]), float(stats.mean[2]))),
        strength_bin=strength_bin,
        angle_bin=angle_bin,
    )


def _usable_clusters(inventory: ClusterInventory) -> dict[str, _ClusterInfo]:
    clusters = {
        cluster_id: _cluster_info(cluster_id, stats)
        for cluster_id, stats in sorted(inventory.clusters.items())
        if stats.count > 0
    }
    if not clusters:
        raise ValueError("inventory must contain at least one usable cluster")
    return clusters


def _validate_slots(slots: Sequence[AssemblySlot]) -> None:
    if not slots:
        raise ValueError("slots must be non-empty")
    slot_ids = [slot.slot_flat_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")


def _ring_slots(slots: Sequence[AssemblySlot]) -> dict[RingKey, list[AssemblySlot]]:
    rings: dict[RingKey, list[AssemblySlot]] = {}
    for slot in slots:
        ring_key = RingKey(layer_id=slot.layer_id, ring_id=slot.ring_id)
        rings.setdefault(ring_key, []).append(slot)
    for ring in rings.values():
        ring.sort(key=lambda slot: slot.theta_id)
    return rings


def _ordered_ring_keys(slots: Sequence[AssemblySlot]) -> tuple[RingKey, ...]:
    rings = _ring_slots(slots)
    layers = sorted({slot.layer_id for slot in slots})
    ordered_layers = [layers[index] for index in outer_to_inner_layer_order(len(layers))]
    ring_ids = sorted({slot.ring_id for slot in slots})
    ordered: list[RingKey] = []
    for layer_id in ordered_layers:
        for ring_id in ring_ids:
            ring_key = RingKey(layer_id=layer_id, ring_id=ring_id)
            if ring_key in rings:
                ordered.append(ring_key)
    return tuple(ordered)


def compute_ring_importance(slots: Sequence[AssemblySlot]) -> dict[RingKey, float]:
    """Return center-weighted importance per physical ring in [0, 1]."""
    _validate_slots(slots)
    rings = _ring_slots(slots)
    z_by_layer: dict[int, float] = {}
    for layer_id in sorted({slot.layer_id for slot in slots}):
        z_values = [float(slot.center_m[2]) for slot in slots if slot.layer_id == layer_id]
        z_by_layer[layer_id] = float(np.mean(np.asarray(z_values, dtype=np.float64)))
    max_abs_z = max(abs(z_value) for z_value in z_by_layer.values())
    layer_importance: dict[int, float] = {}
    for layer_id, z_value in z_by_layer.items():
        if max_abs_z <= 0.0:
            importance = 1.0
        else:
            importance = 1.0 - abs(z_value) / max_abs_z
        layer_importance[layer_id] = min(1.0, max(0.0, float(importance)))
    return {ring_key: layer_importance[ring_key.layer_id] for ring_key in rings}


def _normalize_importance(values: Mapping[RingKey, float]) -> dict[RingKey, float]:
    if not values:
        raise ValueError("importance values must be non-empty")
    finite = {key: float(value) for key, value in values.items()}
    for key, value in finite.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"ring importance for {key} must be finite and >= 0")
    max_value = max(finite.values())
    if max_value <= 0.0:
        return {key: 1.0 for key in finite}
    return {key: min(1.0, max(0.0, value / max_value)) for key, value in finite.items()}


def compute_ring_sensitivity_importance(table: SensitivityTable) -> dict[RingKey, float]:
    """Return ring importance from average sensitivity Frobenius norms."""
    C = np.asarray(table.C, dtype=np.float64)
    if C.ndim != 4:
        raise ValueError(f"sensitivity C must have shape (S, O, D, 3), got {C.shape}")
    slot_count = C.shape[0]
    if table.layer_id.shape != (slot_count,) or table.ring_id.shape != (slot_count,):
        raise ValueError("sensitivity table ring_id/layer_id shapes must match slot count")
    per_slot = np.linalg.norm(C.reshape(slot_count, C.shape[1], -1), axis=2).mean(axis=1)
    totals: dict[RingKey, list[float]] = {}
    for idx, value in enumerate(per_slot):
        key = RingKey(layer_id=int(table.layer_id[idx]), ring_id=int(table.ring_id[idx]))
        totals.setdefault(key, []).append(float(value))
    return _normalize_importance(
        {
            key: float(np.mean(np.asarray(values, dtype=np.float64)))
            for key, values in totals.items()
        }
    )


def _resolve_ring_importance(
    slots: Sequence[AssemblySlot],
    *,
    sensitivity_table: SensitivityTable | None,
    importance_by_ring: Mapping[RingKey, float] | None,
) -> dict[RingKey, float]:
    rings = _ring_slots(slots)
    if importance_by_ring is not None:
        importance = _normalize_importance(dict(importance_by_ring))
    elif sensitivity_table is not None:
        importance = compute_ring_sensitivity_importance(sensitivity_table)
    else:
        importance = compute_ring_importance(slots)
    missing = sorted(set(rings) - set(importance), key=lambda key: (key.layer_id, key.ring_id))
    if missing:
        raise ValueError(f"ring importance is missing ring keys: {missing}")
    return {ring_key: importance[ring_key] for ring_key in rings}


def compute_inventory_target_mean_epsilon(inventory: ClusterInventory) -> float:
    """Return the usable-inventory weighted mean epsilon_parallel."""
    clusters = _usable_clusters(inventory)
    total_count = sum(info.count for info in clusters.values())
    if total_count <= 0:
        raise ValueError("inventory must contain usable cluster count")
    weighted_sum = sum(info.mean_epsilon * info.count for info in clusters.values())
    return float(weighted_sum / total_count)


def _validate_config(config: RingQuotaPlannerConfig) -> None:
    if config.target_mean_epsilon is not None and not math.isfinite(config.target_mean_epsilon):
        raise ValueError("target_mean_epsilon must be finite when provided")
    weights = {
        "lambda_ring_mean": config.lambda_ring_mean,
        "lambda_angle": config.lambda_angle,
        "lambda_inventory": config.lambda_inventory,
        "lambda_mirror_mean": config.lambda_mirror_mean,
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")


def _mirror_maps(
    slots: Sequence[AssemblySlot],
) -> tuple[dict[RingKey, RingKey], dict[RingKey, str]]:
    layers = sorted({slot.layer_id for slot in slots})
    layer_to_index = {layer_id: index for index, layer_id in enumerate(layers)}
    mirror_layer = {
        layer_id: layers[len(layers) - 1 - index] for layer_id, index in layer_to_index.items()
    }
    mirror_keys: dict[RingKey, RingKey] = {}
    pair_ids: dict[RingKey, str] = {}
    for slot in slots:
        key = RingKey(layer_id=slot.layer_id, ring_id=slot.ring_id)
        mirror = RingKey(layer_id=mirror_layer[slot.layer_id], ring_id=slot.ring_id)
        pair_index = min(
            layer_to_index[slot.layer_id],
            layer_to_index[mirror_layer[slot.layer_id]],
        )
        mirror_keys[key] = mirror
        pair_ids[key] = f"P{pair_index:03d}_R{slot.ring_id:03d}"
    return mirror_keys, pair_ids


def _target_angle_bin(
    *,
    ring_importance: float,
    max_angle_bin: int,
) -> float:
    if max_angle_bin <= 0:
        return 0.0
    return float((1.0 - ring_importance) * max_angle_bin)


def _candidate_cost(
    info: _ClusterInfo,
    *,
    remaining_count: int,
    picked_count: int,
    target_count: int,
    epsilon_sum: float,
    target_mean_epsilon: float,
    target_angle_bin: float,
    max_angle_bin: int,
    mirror_mean_epsilon: float | None,
    config: RingQuotaPlannerConfig,
) -> float:
    future_count = target_count - picked_count - 1
    projected_mean = (
        epsilon_sum + info.mean_epsilon + target_mean_epsilon * future_count
    ) / target_count
    strength_deviation = projected_mean - target_mean_epsilon
    if max_angle_bin <= 0:
        angle_deviation = 0.0
    else:
        angle_deviation = (info.angle_bin - target_angle_bin) / max_angle_bin
    scarcity = 1.0 / remaining_count
    mirror_deviation = 0.0
    if mirror_mean_epsilon is not None and config.mirror_balance:
        mirror_deviation = projected_mean - mirror_mean_epsilon
    return float(
        config.lambda_ring_mean * strength_deviation * strength_deviation
        + config.lambda_angle * angle_deviation * angle_deviation
        + config.lambda_inventory * scarcity
        + config.lambda_mirror_mean * mirror_deviation * mirror_deviation
    )


def _build_plan_for_ring(
    ring_key: RingKey,
    *,
    target_count: int,
    ring_importance: float,
    target_mean_epsilon: float,
    clusters: Mapping[str, _ClusterInfo],
    remaining: dict[str, int],
    max_angle_bin: int,
    mirror_mean_epsilon: float | None,
    mirror_pair_id: str | None,
    config: RingQuotaPlannerConfig,
) -> RingQuotaPlan:
    quota_by_cluster: dict[str, int] = {}
    epsilon_sum = 0.0
    angle_bin_sum = 0.0
    angle_error_sum = 0.0
    target_angle = _target_angle_bin(
        ring_importance=ring_importance,
        max_angle_bin=max_angle_bin,
    )
    for picked_count in range(target_count):
        candidates = [
            info for cluster_id, info in clusters.items() if remaining.get(cluster_id, 0) > 0
        ]
        if not candidates:
            raise ValueError("inventory does not contain enough usable magnets for all slots")
        best = min(
            candidates,
            key=lambda info: (
                _candidate_cost(
                    info,
                    remaining_count=remaining[info.cluster_id],
                    picked_count=picked_count,
                    target_count=target_count,
                    epsilon_sum=epsilon_sum,
                    target_mean_epsilon=target_mean_epsilon,
                    target_angle_bin=target_angle,
                    max_angle_bin=max_angle_bin,
                    mirror_mean_epsilon=mirror_mean_epsilon,
                    config=config,
                ),
                info.cluster_id,
            ),
        )
        remaining[best.cluster_id] -= 1
        quota_by_cluster[best.cluster_id] = quota_by_cluster.get(best.cluster_id, 0) + 1
        epsilon_sum += best.mean_epsilon
        angle_bin_sum += best.angle_bin
        angle_error_sum += best.angle_error

    expected_mean_epsilon = epsilon_sum / target_count
    expected_mean_angle_bin = angle_bin_sum / target_count
    expected_angle_error = angle_error_sum / target_count
    sorted_quota = dict(sorted(quota_by_cluster.items()))
    return RingQuotaPlan(
        ring_key=ring_key,
        layer_id=ring_key.layer_id,
        ring_id=ring_key.ring_id,
        work_unit_id=f"W_K{ring_key.layer_id:03d}_R{ring_key.ring_id:03d}",
        target_count=target_count,
        target_mean_epsilon=target_mean_epsilon,
        ring_importance=ring_importance,
        allowed_clusters=tuple(sorted_quota),
        quota_by_cluster=sorted_quota,
        expected_mean_epsilon=float(expected_mean_epsilon),
        expected_mean_angle_bin=float(expected_mean_angle_bin),
        expected_angle_error=float(expected_angle_error),
        mirror_pair_id=mirror_pair_id,
    )


def plan_ring_cluster_quotas(
    slots: Sequence[AssemblySlot],
    inventory: ClusterInventory,
    config: RingQuotaPlannerConfig | None = None,
    *,
    sensitivity_table: SensitivityTable | None = None,
    importance_by_ring: Mapping[RingKey, float] | None = None,
) -> tuple[RingQuotaPlan, ...]:
    """Plan Level 1 cluster quotas for each physical ring."""
    _validate_slots(slots)
    planner_config = RingQuotaPlannerConfig() if config is None else config
    _validate_config(planner_config)
    clusters = _usable_clusters(inventory)
    rings = _ring_slots(slots)
    total_slots = len(slots)
    total_usable = sum(info.count for info in clusters.values())
    if total_usable < total_slots:
        raise ValueError(
            "inventory does not contain enough usable magnets for all slots; "
            f"usable={total_usable}, required={total_slots}"
        )
    target_mean_epsilon = (
        compute_inventory_target_mean_epsilon(inventory)
        if planner_config.target_mean_epsilon is None
        else float(planner_config.target_mean_epsilon)
    )
    resolved_importance = _resolve_ring_importance(
        slots,
        sensitivity_table=sensitivity_table,
        importance_by_ring=importance_by_ring,
    )
    max_angle_bin = max(info.angle_bin for info in clusters.values())
    remaining = {cluster_id: info.count for cluster_id, info in clusters.items()}
    mirror_keys, mirror_pair_ids = _mirror_maps(slots)
    completed: dict[RingKey, RingQuotaPlan] = {}
    plans: list[RingQuotaPlan] = []
    for ring_key in _ordered_ring_keys(slots):
        mirror_plan = completed.get(mirror_keys[ring_key])
        plan = _build_plan_for_ring(
            ring_key,
            target_count=len(rings[ring_key]),
            ring_importance=resolved_importance[ring_key],
            target_mean_epsilon=target_mean_epsilon,
            clusters=clusters,
            remaining=remaining,
            max_angle_bin=max_angle_bin,
            mirror_mean_epsilon=(
                None if mirror_plan is None else mirror_plan.expected_mean_epsilon
            ),
            mirror_pair_id=mirror_pair_ids.get(ring_key),
            config=planner_config,
        )
        plans.append(plan)
        completed[ring_key] = plan
    return tuple(plans)


def _plan_weighted_mean(
    plans: Sequence[RingQuotaPlan],
    getter: Callable[[RingQuotaPlan], float],
) -> float:
    total_count = sum(plan.target_count for plan in plans)
    if total_count <= 0:
        raise ValueError("aggregated quota plan target_count must be positive")
    weighted_sum = sum(float(getter(plan)) * plan.target_count for plan in plans)
    return float(weighted_sum / total_count)


def _merge_quota_by_cluster(plans: Sequence[RingQuotaPlan]) -> dict[str, int]:
    quota: dict[str, int] = {}
    for plan in plans:
        for cluster_id, count in plan.quota_by_cluster.items():
            if count < 0:
                raise ValueError(f"quota count for {cluster_id} must be non-negative")
            quota[cluster_id] = quota.get(cluster_id, 0) + int(count)
    return dict(sorted((cluster_id, count) for cluster_id, count in quota.items() if count > 0))


def _unit_ring_keys(
    unit: WorkUnit,
    *,
    slot_to_ring: Mapping[int, RingKey],
    ring_slot_ids: Mapping[RingKey, set[int]],
) -> tuple[RingKey, ...]:
    unit_slot_ids = set(int(slot_id) for slot_id in unit.slot_flat_ids)
    if len(unit_slot_ids) != len(unit.slot_flat_ids):
        raise ValueError(f"work unit {unit.work_unit_id} contains duplicate slots")
    ring_keys = tuple(sorted({slot_to_ring[slot_id] for slot_id in unit_slot_ids}))
    for ring_key in ring_keys:
        ring_ids = ring_slot_ids[ring_key]
        covered = unit_slot_ids & ring_ids
        if covered != ring_ids:
            raise ValueError(
                "quota aggregation requires work units to contain complete physical rings; "
                f"work_unit_id={unit.work_unit_id}, ring={ring_key}"
            )
    expected_slots = set().union(*(ring_slot_ids[ring_key] for ring_key in ring_keys))
    if unit_slot_ids != expected_slots:
        raise ValueError(
            "quota aggregation found slot coverage mismatch for work unit " f"{unit.work_unit_id}"
        )
    return ring_keys


def _aggregate_quota_plans(
    unit: WorkUnit,
    plans: Sequence[RingQuotaPlan],
) -> RingQuotaPlan:
    if not plans:
        raise ValueError(f"work unit {unit.work_unit_id} has no source quota plans")
    target_count = sum(plan.target_count for plan in plans)
    if target_count != len(unit.slot_flat_ids):
        raise ValueError(
            f"aggregated quota target_count mismatch for {unit.work_unit_id}; "
            f"target={target_count}, unit_slots={len(unit.slot_flat_ids)}"
        )
    quota_by_cluster = _merge_quota_by_cluster(plans)
    if sum(quota_by_cluster.values()) != target_count:
        raise ValueError(
            f"aggregated quota count mismatch for {unit.work_unit_id}; "
            f"target={target_count}, quota_sum={sum(quota_by_cluster.values())}"
        )
    first = plans[0]
    if len(plans) == 1:
        mirror_pair_id = first.mirror_pair_id
    else:
        mirror_pair_id = None
    return RingQuotaPlan(
        ring_key=first.ring_key,
        layer_id=first.layer_id,
        ring_id=first.ring_id,
        work_unit_id=unit.work_unit_id,
        target_count=target_count,
        target_mean_epsilon=_plan_weighted_mean(
            plans,
            lambda plan: plan.target_mean_epsilon,
        ),
        ring_importance=_plan_weighted_mean(plans, lambda plan: plan.ring_importance),
        allowed_clusters=tuple(quota_by_cluster),
        quota_by_cluster=quota_by_cluster,
        expected_mean_epsilon=_plan_weighted_mean(
            plans,
            lambda plan: plan.expected_mean_epsilon,
        ),
        expected_mean_angle_bin=_plan_weighted_mean(
            plans,
            lambda plan: plan.expected_mean_angle_bin,
        ),
        expected_angle_error=_plan_weighted_mean(plans, lambda plan: plan.expected_angle_error),
        mirror_pair_id=mirror_pair_id,
    )


def plan_work_unit_cluster_quotas(
    slots: Sequence[AssemblySlot],
    inventory: ClusterInventory,
    work_units: Sequence[WorkUnit],
    config: RingQuotaPlannerConfig | None = None,
    *,
    sensitivity_table: SensitivityTable | None = None,
    importance_by_ring: Mapping[RingKey, float] | None = None,
) -> tuple[RingQuotaPlan, ...]:
    """
    Return cluster quota plans aligned 1:1 with assembly work units.

    The Level 1 planner works per physical ring. Ring-by-ring work units can use those
    plans directly, while paired/grouped work units need the corresponding ring plans
    summed into one quota plan per work unit.
    """
    _validate_slots(slots)
    if not work_units:
        raise ValueError("work_units must be non-empty")
    base_plans = plan_ring_cluster_quotas(
        slots,
        inventory,
        config,
        sensitivity_table=sensitivity_table,
        importance_by_ring=importance_by_ring,
    )
    plan_by_ring = {plan.ring_key: plan for plan in base_plans}
    if len(plan_by_ring) != len(base_plans):
        raise ValueError("ring quota plans contain duplicate ring keys")

    rings = _ring_slots(slots)
    ring_slot_ids = {
        ring_key: {slot.slot_flat_id for slot in ring_slots}
        for ring_key, ring_slots in rings.items()
    }
    slot_to_ring: dict[int, RingKey] = {}
    for ring_key, slot_ids in ring_slot_ids.items():
        for slot_id in slot_ids:
            slot_to_ring[slot_id] = ring_key

    table_slot_ids = set(slot_to_ring)
    seen_slots: set[int] = set()
    aligned_plans: list[RingQuotaPlan] = []
    for unit in work_units:
        unit_slot_ids = set(int(slot_id) for slot_id in unit.slot_flat_ids)
        unknown = sorted(unit_slot_ids - table_slot_ids)
        if unknown:
            raise ValueError(f"work unit {unit.work_unit_id} references unknown slots: {unknown}")
        duplicate_across_units = sorted(seen_slots & unit_slot_ids)
        if duplicate_across_units:
            raise ValueError(
                f"work unit {unit.work_unit_id} reuses slots: {duplicate_across_units}"
            )
        seen_slots.update(unit_slot_ids)
        ring_keys = _unit_ring_keys(
            unit,
            slot_to_ring=slot_to_ring,
            ring_slot_ids=ring_slot_ids,
        )
        aligned_plans.append(
            _aggregate_quota_plans(unit, tuple(plan_by_ring[ring_key] for ring_key in ring_keys))
        )

    if seen_slots != table_slot_ids:
        missing = sorted(table_slot_ids - seen_slots)
        extra = sorted(seen_slots - table_slot_ids)
        raise ValueError(f"work unit slot coverage mismatch; missing={missing}, extra={extra}")
    return tuple(aligned_plans)


__all__ = [
    "compute_inventory_target_mean_epsilon",
    "compute_ring_importance",
    "compute_ring_sensitivity_importance",
    "plan_ring_cluster_quotas",
    "plan_work_unit_cluster_quotas",
]
