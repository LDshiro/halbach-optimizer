from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from halbach.assembly.types import (
    AssemblySlot,
    ClusterInventory,
    ClusterStats,
    RingKey,
    RingQuotaPlan,
    RingQuotaPlannerConfig,
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
    importance_by_ring = compute_ring_importance(slots)
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
            ring_importance=importance_by_ring[ring_key],
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


__all__ = [
    "compute_inventory_target_mean_epsilon",
    "compute_ring_importance",
    "plan_ring_cluster_quotas",
]
