from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from halbach.assembly.inventory import decrement_cluster
from halbach.assembly.types import (
    ClusterAssignment,
    ClusterInventory,
    ClusterMPCConfig,
    ClusterMPCDecision,
    ClusterStats,
    LinearAssignmentResult,
    LinearCandidate,
    MagnetError,
    Placement,
    RingKey,
    RingPairSummary,
    RingQuotaPlan,
    SensitivityTable,
    VirtualMagnet,
    WorkUnit,
)
from halbach.types import FloatArray


def _validate_table(table: SensitivityTable) -> None:
    if table.C.ndim != 4 or table.C.shape[3] != 3:
        raise ValueError("sensitivity C must have shape (S, O, residual_dim, 3)")
    if table.C.shape[0] != table.slot_flat_id.shape[0]:
        raise ValueError("sensitivity slot ids and C slot dimension mismatch")
    if table.C.shape[1] != len(table.orientation_id):
        raise ValueError("sensitivity orientation ids and C orientation dimension mismatch")
    if table.C.shape[2] <= 0:
        raise ValueError("sensitivity residual dimension must be positive")
    if len(set(table.orientation_id)) != len(table.orientation_id):
        raise ValueError("sensitivity orientation ids must be unique")
    if len(set(int(item) for item in table.slot_flat_id.tolist())) != table.slot_flat_id.size:
        raise ValueError("sensitivity slot ids must be unique")
    if not np.all(np.isfinite(table.C)):
        raise ValueError("sensitivity C must contain finite values")


def _error_vector(error: MagnetError) -> FloatArray:
    x = np.array(
        [
            float(error.epsilon_parallel),
            float(error.delta_perp_1),
            float(error.delta_perp_2),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(x)):
        raise ValueError("magnet error components must be finite")
    return x


def _angle_error(error: MagnetError) -> float:
    return float(math.hypot(float(error.delta_perp_1), float(error.delta_perp_2)))


@dataclass(frozen=True)
class _MirrorRingCompletion:
    pair_id: str
    ring_key: RingKey
    work_unit_id: str
    count: int
    epsilon_sum: float
    angle_error_sum: float
    residual_norm_after_ring: float
    cluster_counts: dict[str, int]

    @property
    def mean_epsilon(self) -> float:
        return self.epsilon_sum / self.count

    @property
    def mean_angle_error(self) -> float:
        return self.angle_error_sum / self.count


def _pair_index_from_pair_id(pair_id: str) -> int:
    head = pair_id.split("_", maxsplit=1)[0]
    if not head.startswith("P"):
        return 0
    try:
        return int(head[1:])
    except ValueError:
        return 0


class _MirrorPairTracker:
    """Track completed mirror rings during online cluster MPC assignment."""

    def __init__(self) -> None:
        self._completed: dict[str, list[_MirrorRingCompletion]] = {}

    def mirror_mean_for(self, quota_plan: RingQuotaPlan) -> float | None:
        if quota_plan.mirror_pair_id is None:
            return None
        for completion in self._completed.get(quota_plan.mirror_pair_id, []):
            if completion.ring_key != quota_plan.ring_key:
                return completion.mean_epsilon
        return None

    def complete_ring(
        self,
        quota_plan: RingQuotaPlan,
        *,
        count: int,
        epsilon_sum: float,
        angle_error_sum: float,
        residual: FloatArray,
        cluster_counts: dict[str, int],
    ) -> None:
        if quota_plan.mirror_pair_id is None:
            return
        if count <= 0:
            raise ValueError("completed mirror ring count must be positive")
        if not math.isfinite(epsilon_sum):
            raise ValueError("completed mirror ring epsilon_sum must be finite")
        if not math.isfinite(angle_error_sum):
            raise ValueError("completed mirror ring angle_error_sum must be finite")
        residual_arr = np.asarray(residual, dtype=np.float64)
        if residual_arr.ndim != 1 or not np.all(np.isfinite(residual_arr)):
            raise ValueError("completed mirror ring residual must be a finite vector")
        pair_id = quota_plan.mirror_pair_id
        completions = self._completed.setdefault(pair_id, [])
        if any(completion.ring_key == quota_plan.ring_key for completion in completions):
            raise ValueError(f"mirror pair {pair_id} already contains ring {quota_plan.ring_key}")
        if len(completions) >= 2:
            raise ValueError(f"mirror pair {pair_id} already has two completed rings")
        completions.append(
            _MirrorRingCompletion(
                pair_id=pair_id,
                ring_key=quota_plan.ring_key,
                work_unit_id=quota_plan.work_unit_id,
                count=count,
                epsilon_sum=float(epsilon_sum),
                angle_error_sum=float(angle_error_sum),
                residual_norm_after_ring=float(np.linalg.norm(residual_arr)),
                cluster_counts=dict(sorted(cluster_counts.items())),
            )
        )

    def summaries(self) -> tuple[RingPairSummary, ...]:
        summaries: list[RingPairSummary] = []
        for pair_id in sorted(self._completed):
            completions = sorted(
                self._completed[pair_id],
                key=lambda item: (item.ring_key.layer_id, item.ring_key.ring_id),
            )
            lower = completions[0]
            upper = completions[1] if len(completions) > 1 else None
            summaries.append(
                RingPairSummary(
                    pair_id=pair_id,
                    pair_index=_pair_index_from_pair_id(pair_id),
                    ring_id=lower.ring_key.ring_id,
                    lower_ring=lower.ring_key,
                    upper_ring=None if upper is None else upper.ring_key,
                    lower_count=lower.count,
                    upper_count=0 if upper is None else upper.count,
                    mean_epsilon_difference=(
                        None if upper is None else lower.mean_epsilon - upper.mean_epsilon
                    ),
                    mean_angle_error_difference=(
                        None if upper is None else lower.mean_angle_error - upper.mean_angle_error
                    ),
                    lower_mean_epsilon=lower.mean_epsilon,
                    upper_mean_epsilon=None if upper is None else upper.mean_epsilon,
                    residual_norm_after_lower=lower.residual_norm_after_ring,
                    residual_norm_after_upper=(
                        None if upper is None else upper.residual_norm_after_ring
                    ),
                    residual_norm_after_pair=(
                        None if upper is None else upper.residual_norm_after_ring
                    ),
                    pair_complete=upper is not None,
                )
            )
        return tuple(summaries)


@dataclass(frozen=True)
class _TableLookup:
    table: SensitivityTable
    slot_to_idx: dict[int, int]
    orientation_to_idx: dict[str, int]
    orientation_order: dict[str, int]
    gram: FloatArray


def _table_lookup(table: SensitivityTable) -> _TableLookup:
    _validate_table(table)
    C = np.asarray(table.C, dtype=np.float64)
    return _TableLookup(
        table=table,
        slot_to_idx=_slot_index(table),
        orientation_to_idx=_orientation_index(table),
        orientation_order={
            orientation_id: idx for idx, orientation_id in enumerate(table.orientation_id)
        },
        gram=np.ascontiguousarray(np.einsum("sodc,sode->soce", C, C), dtype=np.float64),
    )


def _assignment_cluster_map(
    assignments: Sequence[ClusterAssignment] | None,
    magnets: Sequence[VirtualMagnet],
) -> dict[int, str | None]:
    if assignments is None:
        return {magnet.magnet_id: None for magnet in magnets}
    if len(assignments) != len(magnets):
        raise ValueError("assignments length must match magnets length")
    magnet_ids = {magnet.magnet_id for magnet in magnets}
    seen: set[int] = set()
    cluster_by_magnet: dict[int, str | None] = {}
    for assignment in assignments:
        if assignment.magnet_id in seen:
            raise ValueError("assignments contain duplicate magnet_id values")
        seen.add(assignment.magnet_id)
        if assignment.magnet_id not in magnet_ids:
            raise ValueError(f"assignment references unknown magnet_id: {assignment.magnet_id}")
        if assignment.quarantine_id is not None:
            raise ValueError("quarantined magnets are not supported by Step 5 MVP assignment")
        if assignment.cluster_id is None:
            raise ValueError("normal assignment must include cluster_id")
        cluster_by_magnet[assignment.magnet_id] = assignment.cluster_id
    missing = sorted(magnet_ids - seen)
    if missing:
        raise ValueError(f"assignments missing magnet ids: {missing}")
    return cluster_by_magnet


def _slot_index(table: SensitivityTable) -> dict[int, int]:
    return {int(slot_id): idx for idx, slot_id in enumerate(table.slot_flat_id.tolist())}


def _orientation_index(table: SensitivityTable) -> dict[str, int]:
    return {orientation_id: idx for idx, orientation_id in enumerate(table.orientation_id)}


def _angle_bin_from_cluster_id(cluster_id: str) -> int:
    try:
        angle_part = cluster_id.split("_A", maxsplit=1)[1]
    except IndexError as exc:
        raise ValueError(f"cluster_id must use Sxx_Ayy format: {cluster_id}") from exc
    return int(angle_part)


def _validate_cluster_stats(stats: ClusterStats) -> None:
    if stats.count < 0:
        raise ValueError(f"cluster {stats.cluster_id} count must be non-negative")
    if stats.mean.shape != (3,):
        raise ValueError(f"cluster {stats.cluster_id} mean must have shape (3,)")
    if stats.cov.shape != (3, 3):
        raise ValueError(f"cluster {stats.cluster_id} cov must have shape (3, 3)")
    if not np.all(np.isfinite(stats.mean)):
        raise ValueError(f"cluster {stats.cluster_id} mean must contain finite values")
    if not np.all(np.isfinite(stats.cov)):
        raise ValueError(f"cluster {stats.cluster_id} cov must contain finite values")


def _validate_cluster_mpc_config(config: ClusterMPCConfig) -> None:
    weights = {
        "lambda_field": config.lambda_field,
        "lambda_quota": config.lambda_quota,
        "lambda_ring_mean": config.lambda_ring_mean,
        "lambda_angle": config.lambda_angle,
        "lambda_future": config.lambda_future,
        "lambda_mirror": config.lambda_mirror,
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")


def _allowed_orientation_ids(
    lookup: _TableLookup,
    allowed_orientation_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    orientation_ids = (
        tuple(lookup.table.orientation_id)
        if allowed_orientation_ids is None
        else tuple(allowed_orientation_ids)
    )
    if not orientation_ids:
        raise ValueError("allowed_orientation_ids must be non-empty")
    for orientation_id in orientation_ids:
        if orientation_id not in lookup.orientation_to_idx:
            raise KeyError(f"unknown orientation_id: {orientation_id}")
    return orientation_ids


def _score_linear_candidates_cached(
    lookup: _TableLookup,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> list[LinearCandidate]:
    """
    Score all remaining slot/orientation candidates.

    score = ||residual + C[s,o] @ x||^2 where x is the measured magnet error.
    """
    residual_arr = np.asarray(residual, dtype=np.float64)
    if residual_arr.shape != (lookup.table.C.shape[2],):
        raise ValueError(f"residual must have shape ({lookup.table.C.shape[2]},)")
    if not np.all(np.isfinite(residual_arr)):
        raise ValueError("residual must contain finite values")
    if not remaining_slot_flat_ids:
        raise ValueError("remaining_slot_flat_ids must be non-empty")

    orientation_ids = _allowed_orientation_ids(lookup, allowed_orientation_ids)
    x = _error_vector(error)
    candidates: list[LinearCandidate] = []
    for slot_flat_id in remaining_slot_flat_ids:
        slot_key = int(slot_flat_id)
        if slot_key not in lookup.slot_to_idx:
            raise KeyError(f"unknown slot_flat_id: {slot_key}")
        slot_idx = lookup.slot_to_idx[slot_key]
        for orientation_id in orientation_ids:
            orientation_idx = lookup.orientation_to_idx[orientation_id]
            contribution = np.ascontiguousarray(
                lookup.table.C[slot_idx, orientation_idx, :, :] @ x,
                dtype=np.float64,
            )
            updated = residual_arr + contribution
            score = float(np.dot(updated, updated))
            candidates.append(
                LinearCandidate(
                    slot_flat_id=slot_key,
                    orientation_id=orientation_id,
                    score=score,
                    contribution=contribution,
                )
            )
    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.slot_flat_id,
            lookup.orientation_order[candidate.orientation_id],
        )
    )
    return candidates


def score_linear_candidates(
    table: SensitivityTable,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> list[LinearCandidate]:
    """
    Score all remaining slot/orientation candidates.

    score = ||residual + C[s,o] @ x||^2 where x is the measured magnet error.
    """
    return _score_linear_candidates_cached(
        _table_lookup(table),
        residual,
        remaining_slot_flat_ids,
        error,
        allowed_orientation_ids=allowed_orientation_ids,
    )


def choose_best_linear_candidate(
    table: SensitivityTable,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> LinearCandidate:
    """Return the minimum-score slot/orientation candidate."""
    return score_linear_candidates(
        table,
        residual,
        remaining_slot_flat_ids,
        error,
        allowed_orientation_ids=allowed_orientation_ids,
    )[0]


def _choose_best_linear_candidate_cached(
    lookup: _TableLookup,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    error: MagnetError,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
) -> LinearCandidate:
    return _score_linear_candidates_cached(
        lookup,
        residual,
        remaining_slot_flat_ids,
        error,
        allowed_orientation_ids=allowed_orientation_ids,
    )[0]


def _score_cluster_for_current_ring_cached(
    lookup: _TableLookup,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    cluster_stats: ClusterStats,
    quota_plan: RingQuotaPlan,
    current_ring_cluster_usage: dict[str, int],
    current_ring_epsilon_sum: float,
    current_ring_count: int,
    remaining_cluster_counts: dict[str, int],
    future_cluster_demand: dict[str, int],
    config: ClusterMPCConfig,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
    mirror_mean_epsilon: float | None = None,
) -> ClusterMPCDecision:
    """Score one cluster as the next pickup candidate for the current ring."""
    _validate_cluster_stats(cluster_stats)
    _validate_cluster_mpc_config(config)
    residual_arr = np.asarray(residual, dtype=np.float64)
    if residual_arr.shape != (lookup.table.C.shape[2],):
        raise ValueError(f"residual must have shape ({lookup.table.C.shape[2]},)")
    if not np.all(np.isfinite(residual_arr)):
        raise ValueError("residual must contain finite values")
    if current_ring_count < 0:
        raise ValueError("current_ring_count must be non-negative")
    if not math.isfinite(current_ring_epsilon_sum):
        raise ValueError("current_ring_epsilon_sum must be finite")
    if not remaining_slot_flat_ids:
        raise ValueError("remaining_slot_flat_ids must be non-empty")
    if remaining_cluster_counts.get(cluster_stats.cluster_id, 0) <= 0:
        raise ValueError(f"cluster {cluster_stats.cluster_id} has no remaining magnets")

    orientation_ids = _allowed_orientation_ids(lookup, allowed_orientation_ids)
    mean = np.asarray(cluster_stats.mean, dtype=np.float64)
    cov = np.asarray(cluster_stats.cov, dtype=np.float64)
    best_field_cost = math.inf
    best_slot_id = int(remaining_slot_flat_ids[0])
    best_orientation_id = orientation_ids[0]
    for slot_flat_id in remaining_slot_flat_ids:
        slot_key = int(slot_flat_id)
        if slot_key not in lookup.slot_to_idx:
            raise KeyError(f"unknown slot_flat_id: {slot_key}")
        slot_idx = lookup.slot_to_idx[slot_key]
        for orientation_id in orientation_ids:
            orientation_idx = lookup.orientation_to_idx[orientation_id]
            C_j = np.asarray(lookup.table.C[slot_idx, orientation_idx, :, :], dtype=np.float64)
            expected = residual_arr + C_j @ mean
            trace_term = float(np.einsum("ij,ji->", lookup.gram[slot_idx, orientation_idx], cov))
            field_cost = float(np.dot(expected, expected) + trace_term)
            if field_cost < best_field_cost:
                best_field_cost = field_cost
                best_slot_id = slot_key
                best_orientation_id = orientation_id

    cluster_id = cluster_stats.cluster_id
    planned_count = int(quota_plan.quota_by_cluster.get(cluster_id, 0))
    used_count = int(current_ring_cluster_usage.get(cluster_id, 0))
    quota_overuse = max(0, used_count + 1 - planned_count)
    quota_cost = float(quota_overuse * quota_overuse)

    projected_count = current_ring_count + 1
    projected_mean = (current_ring_epsilon_sum + float(cluster_stats.mean[0])) / projected_count
    mean_delta = projected_mean - float(quota_plan.target_mean_epsilon)
    ring_mean_cost = float(mean_delta * mean_delta)

    angle_delta = float(_angle_bin_from_cluster_id(cluster_id)) - float(
        quota_plan.expected_mean_angle_bin
    )
    angle_cost = float(angle_delta * angle_delta)

    projected_remaining = int(remaining_cluster_counts.get(cluster_id, 0)) - 1
    future_shortage = max(0, int(future_cluster_demand.get(cluster_id, 0)) - projected_remaining)
    future_cost = float(future_shortage * future_shortage)

    mirror_cost = 0.0
    if mirror_mean_epsilon is not None:
        mirror_delta = projected_mean - float(mirror_mean_epsilon)
        mirror_cost = float(mirror_delta * mirror_delta)

    total_score = float(
        config.lambda_field * best_field_cost
        + config.lambda_quota * quota_cost
        + config.lambda_ring_mean * ring_mean_cost
        + config.lambda_angle * angle_cost
        + config.lambda_future * future_cost
        + config.lambda_mirror * mirror_cost
    )
    return ClusterMPCDecision(
        cluster_id=cluster_id,
        total_score=total_score,
        field_cost=best_field_cost,
        quota_cost=quota_cost,
        ring_mean_cost=ring_mean_cost,
        angle_cost=angle_cost,
        future_cost=future_cost,
        mirror_cost=mirror_cost,
        best_slot_flat_id=best_slot_id,
        best_orientation_id=best_orientation_id,
    )


def score_cluster_for_current_ring(
    table: SensitivityTable,
    residual: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    cluster_stats: ClusterStats,
    quota_plan: RingQuotaPlan,
    current_ring_cluster_usage: dict[str, int],
    current_ring_epsilon_sum: float,
    current_ring_count: int,
    remaining_cluster_counts: dict[str, int],
    future_cluster_demand: dict[str, int],
    config: ClusterMPCConfig,
    *,
    allowed_orientation_ids: Sequence[str] | None = None,
    mirror_mean_epsilon: float | None = None,
) -> ClusterMPCDecision:
    """Score one cluster as the next pickup candidate for the current ring."""
    return _score_cluster_for_current_ring_cached(
        _table_lookup(table),
        residual,
        remaining_slot_flat_ids,
        cluster_stats,
        quota_plan,
        current_ring_cluster_usage,
        current_ring_epsilon_sum,
        current_ring_count,
        remaining_cluster_counts,
        future_cluster_demand,
        config,
        allowed_orientation_ids=allowed_orientation_ids,
        mirror_mean_epsilon=mirror_mean_epsilon,
    )


def run_linear_sensitivity_assignment(
    table: SensitivityTable,
    magnets: Sequence[VirtualMagnet],
    *,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
) -> LinearAssignmentResult:
    """
    Greedily place magnets using the Step 5 linear sensitivity MVP.

    Decisions use measured_error, while downstream fixed-field evaluation should still
    use true_error through the resulting Placement records.
    """
    lookup = _table_lookup(table)
    if len(magnets) != table.slot_flat_id.shape[0]:
        raise ValueError("magnets count must match sensitivity slot count")
    magnet_by_id = {magnet.magnet_id: magnet for magnet in magnets}
    if len(magnet_by_id) != len(magnets):
        raise ValueError("magnet_id values must be unique")

    if magnet_order is None:
        ordered_magnets = list(magnets)
    else:
        if len(magnet_order) != len(magnets):
            raise ValueError("magnet_order length must match magnets length")
        if len(set(int(item) for item in magnet_order)) != len(magnet_order):
            raise ValueError("magnet_order contains duplicate magnet ids")
        unknown = sorted(set(int(item) for item in magnet_order) - set(magnet_by_id))
        if unknown:
            raise ValueError(f"magnet_order references unknown magnet ids: {unknown}")
        ordered_magnets = [magnet_by_id[int(magnet_id)] for magnet_id in magnet_order]

    cluster_by_magnet = _assignment_cluster_map(assignments, magnets)
    residual = np.zeros(table.C.shape[2], dtype=np.float64)
    remaining_slot_ids = [int(slot_id) for slot_id in table.slot_flat_id.tolist()]
    placements: list[Placement] = []
    current_inventory = inventory

    for insert_order, magnet in enumerate(ordered_magnets):
        candidate = _choose_best_linear_candidate_cached(
            lookup,
            residual,
            remaining_slot_ids,
            magnet.measured_error,
            allowed_orientation_ids=allowed_orientation_ids,
        )
        cluster_id = cluster_by_magnet.get(magnet.magnet_id)
        if current_inventory is not None and cluster_id is not None:
            current_inventory = decrement_cluster(current_inventory, cluster_id)
        residual = np.ascontiguousarray(residual + candidate.contribution, dtype=np.float64)
        remaining_slot_ids.remove(candidate.slot_flat_id)
        placements.append(
            Placement(
                slot_flat_id=candidate.slot_flat_id,
                magnet_id=magnet.magnet_id,
                orientation_id=candidate.orientation_id,
                cluster_requested=cluster_id,
                insert_order=insert_order,
                decision_engine="linear_sensitivity",
            )
        )

    linear_score = float(np.dot(residual, residual))
    if not math.isfinite(linear_score):
        raise ValueError("linear assignment produced a non-finite score")
    return LinearAssignmentResult(
        placements=tuple(placements),
        residual=residual,
        linear_score=linear_score,
        remaining_slot_flat_ids=tuple(remaining_slot_ids),
        inventory=current_inventory,
    )


def _ordered_unit_slot_ids(
    table: SensitivityTable,
    work_units: Sequence[WorkUnit],
) -> tuple[tuple[int, ...], ...]:
    if not work_units:
        raise ValueError("work_units must be non-empty")
    table_slot_ids = tuple(int(slot_id) for slot_id in table.slot_flat_id.tolist())
    table_slot_set = set(table_slot_ids)
    seen: set[int] = set()
    ordered_units: list[tuple[int, ...]] = []
    for unit in work_units:
        if not unit.slot_flat_ids:
            raise ValueError(f"work unit {unit.work_unit_id} has no slots")
        unit_slots = tuple(int(slot_id) for slot_id in unit.slot_flat_ids)
        for slot_id in unit_slots:
            if slot_id in seen:
                raise ValueError(f"slot_flat_id {slot_id} appears in multiple work units")
            if slot_id not in table_slot_set:
                raise ValueError(f"work unit references unknown slot_flat_id: {slot_id}")
            seen.add(slot_id)
        ordered_units.append(unit_slots)
    if seen != table_slot_set:
        missing = sorted(table_slot_set - seen)
        extra = sorted(seen - table_slot_set)
        raise ValueError(f"work unit slot coverage mismatch; missing={missing}, extra={extra}")
    return tuple(ordered_units)


def _assignment_cluster_pools(
    assignments: Sequence[ClusterAssignment],
    magnets: Sequence[VirtualMagnet],
    *,
    seed: int,
) -> dict[str, list[int]]:
    cluster_by_magnet = _assignment_cluster_map(assignments, magnets)
    pools: dict[str, list[int]] = {}
    for magnet in magnets:
        cluster_id = cluster_by_magnet[magnet.magnet_id]
        if cluster_id is None:
            raise ValueError("quota pickup requires normal cluster assignments")
        pools.setdefault(cluster_id, []).append(magnet.magnet_id)
    rng = np.random.default_rng(int(seed))
    for cluster_id, magnet_ids in pools.items():
        shuffled = rng.permutation(np.asarray(magnet_ids, dtype=np.int64))
        pools[cluster_id] = [int(magnet_id) for magnet_id in shuffled.tolist()]
    return pools


def _expanded_quota_clusters(plan: RingQuotaPlan) -> list[str]:
    if plan.target_count <= 0:
        raise ValueError(f"quota plan {plan.work_unit_id} target_count must be positive")
    if not plan.quota_by_cluster:
        raise ValueError(f"quota plan {plan.work_unit_id} has no cluster quota")
    sequence: list[str] = []
    for cluster_id, count in sorted(plan.quota_by_cluster.items()):
        if count < 0:
            raise ValueError(f"quota count for {cluster_id} must be non-negative")
        sequence.extend([cluster_id] * int(count))
    if len(sequence) != plan.target_count:
        raise ValueError(
            f"quota plan {plan.work_unit_id} count mismatch; "
            f"target={plan.target_count}, quota_sum={len(sequence)}"
        )
    return sequence


def _validate_quota_work_units(
    work_units: Sequence[WorkUnit],
    quota_plans: Sequence[RingQuotaPlan],
) -> None:
    if len(work_units) != len(quota_plans):
        raise ValueError("quota_plans length must match work_units length")
    for unit, plan in zip(work_units, quota_plans, strict=True):
        if unit.work_unit_id != plan.work_unit_id:
            raise ValueError(
                "quota plan order must match work unit order; "
                f"unit={unit.work_unit_id}, plan={plan.work_unit_id}"
            )
        if len(unit.slot_flat_ids) != plan.target_count:
            raise ValueError(
                f"quota plan {plan.work_unit_id} target_count mismatch; "
                f"unit_slots={len(unit.slot_flat_ids)}, target={plan.target_count}"
            )


def _future_quota_demand(
    quota_plans: Sequence[RingQuotaPlan],
    start_index: int,
) -> dict[str, int]:
    demand: dict[str, int] = {}
    for plan in quota_plans[start_index:]:
        for cluster_id, count in plan.quota_by_cluster.items():
            if count < 0:
                raise ValueError(f"quota count for {cluster_id} must be non-negative")
            demand[cluster_id] = demand.get(cluster_id, 0) + int(count)
    return demand


def build_quota_ordered_magnet_order(
    magnets: Sequence[VirtualMagnet],
    assignments: Sequence[ClusterAssignment],
    quota_plans: Sequence[RingQuotaPlan],
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """
    Build a magnet pickup order by expanding each ring quota into cluster requests.

    Each requested cluster consumes one magnet id from that cluster's measured-assignment
    pool. Pools are shuffled with seed to simulate arbitrary pickup within a cluster.
    """
    if not quota_plans:
        raise ValueError("quota_plans must be non-empty")
    pools = _assignment_cluster_pools(assignments, magnets, seed=seed)
    order: list[int] = []
    for plan in quota_plans:
        for cluster_id in _expanded_quota_clusters(plan):
            pool = pools.get(cluster_id)
            if not pool:
                raise ValueError(
                    f"quota requests cluster {cluster_id} but no assigned magnets remain"
                )
            order.append(pool.pop(0))
    if len(order) != len(magnets):
        raise ValueError(
            "quota pickup must consume exactly one magnet per slot; "
            f"picked={len(order)}, magnets={len(magnets)}"
        )
    return tuple(order)


def run_ring_constrained_linear_assignment(
    table: SensitivityTable,
    magnets: Sequence[VirtualMagnet],
    work_units: Sequence[WorkUnit],
    *,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
) -> LinearAssignmentResult:
    """
    Greedily place magnets while restricting candidate slots to one work unit at a time.

    The residual is global, but each magnet can only choose among the remaining slots in
    the current work unit. Once that unit is full, assignment advances to the next unit.
    A single all-slot work unit delegates to the existing global implementation.
    """
    lookup = _table_lookup(table)
    ordered_unit_slots = _ordered_unit_slot_ids(table, work_units)
    if len(ordered_unit_slots) == 1:
        return run_linear_sensitivity_assignment(
            table,
            magnets,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            magnet_order=magnet_order,
        )
    if len(magnets) != table.slot_flat_id.shape[0]:
        raise ValueError("magnets count must match sensitivity slot count")
    magnet_by_id = {magnet.magnet_id: magnet for magnet in magnets}
    if len(magnet_by_id) != len(magnets):
        raise ValueError("magnet_id values must be unique")

    if magnet_order is None:
        ordered_magnets = list(magnets)
    else:
        if len(magnet_order) != len(magnets):
            raise ValueError("magnet_order length must match magnets length")
        if len(set(int(item) for item in magnet_order)) != len(magnet_order):
            raise ValueError("magnet_order contains duplicate magnet ids")
        unknown = sorted(set(int(item) for item in magnet_order) - set(magnet_by_id))
        if unknown:
            raise ValueError(f"magnet_order references unknown magnet ids: {unknown}")
        ordered_magnets = [magnet_by_id[int(magnet_id)] for magnet_id in magnet_order]

    cluster_by_magnet = _assignment_cluster_map(assignments, magnets)
    residual = np.zeros(table.C.shape[2], dtype=np.float64)
    placements: list[Placement] = []
    current_inventory = inventory
    magnet_index = 0
    for unit_slots in ordered_unit_slots:
        remaining_unit_slot_ids = list(unit_slots)
        while remaining_unit_slot_ids:
            magnet = ordered_magnets[magnet_index]
            candidate = _choose_best_linear_candidate_cached(
                lookup,
                residual,
                remaining_unit_slot_ids,
                magnet.measured_error,
                allowed_orientation_ids=allowed_orientation_ids,
            )
            cluster_id = cluster_by_magnet.get(magnet.magnet_id)
            if current_inventory is not None and cluster_id is not None:
                current_inventory = decrement_cluster(current_inventory, cluster_id)
            residual = np.ascontiguousarray(residual + candidate.contribution, dtype=np.float64)
            remaining_unit_slot_ids.remove(candidate.slot_flat_id)
            placements.append(
                Placement(
                    slot_flat_id=candidate.slot_flat_id,
                    magnet_id=magnet.magnet_id,
                    orientation_id=candidate.orientation_id,
                    cluster_requested=cluster_id,
                    insert_order=len(placements),
                    decision_engine="ring_constrained_linear_sensitivity",
                )
            )
            magnet_index += 1

    linear_score = float(np.dot(residual, residual))
    if not math.isfinite(linear_score):
        raise ValueError("linear assignment produced a non-finite score")
    return LinearAssignmentResult(
        placements=tuple(placements),
        residual=residual,
        linear_score=linear_score,
        remaining_slot_flat_ids=(),
        inventory=current_inventory,
    )


def run_quota_ordered_ring_constrained_linear_assignment(
    table: SensitivityTable,
    magnets: Sequence[VirtualMagnet],
    work_units: Sequence[WorkUnit],
    quota_plans: Sequence[RingQuotaPlan],
    *,
    assignments: Sequence[ClusterAssignment],
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    seed: int = 0,
) -> LinearAssignmentResult:
    """
    Run ring-constrained assignment after simulating quota-ordered cluster pickup.

    The quota controls only which magnet cluster is picked next. The picked magnet's
    measured_error still decides the slot/orientation inside the current work unit.
    """
    _ordered_unit_slot_ids(table, work_units)
    _validate_quota_work_units(work_units, quota_plans)
    magnet_order = build_quota_ordered_magnet_order(
        magnets,
        assignments,
        quota_plans,
        seed=seed,
    )
    return run_ring_constrained_linear_assignment(
        table,
        magnets,
        work_units,
        assignments=assignments,
        inventory=inventory,
        allowed_orientation_ids=allowed_orientation_ids,
        magnet_order=magnet_order,
    )


def run_cluster_mpc_ring_constrained_linear_assignment(
    table: SensitivityTable,
    magnets: Sequence[VirtualMagnet],
    work_units: Sequence[WorkUnit],
    quota_plans: Sequence[RingQuotaPlan],
    *,
    assignments: Sequence[ClusterAssignment],
    inventory: ClusterInventory,
    config: ClusterMPCConfig | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    seed: int = 0,
) -> LinearAssignmentResult:
    """Run Step R6 cluster-level MPC pickup with ring-constrained placement."""
    lookup = _table_lookup(table)
    mpc_config = ClusterMPCConfig() if config is None else config
    _validate_cluster_mpc_config(mpc_config)
    ordered_unit_slots = _ordered_unit_slot_ids(table, work_units)
    _validate_quota_work_units(work_units, quota_plans)
    if len(magnets) != table.slot_flat_id.shape[0]:
        raise ValueError("magnets count must match sensitivity slot count")
    magnet_by_id = {magnet.magnet_id: magnet for magnet in magnets}
    if len(magnet_by_id) != len(magnets):
        raise ValueError("magnet_id values must be unique")

    pools = _assignment_cluster_pools(assignments, magnets, seed=seed)
    residual = np.zeros(table.C.shape[2], dtype=np.float64)
    placements: list[Placement] = []
    current_inventory = inventory
    mirror_tracker = _MirrorPairTracker()

    for unit_index, (unit_slots, quota_plan) in enumerate(
        zip(ordered_unit_slots, quota_plans, strict=True)
    ):
        remaining_unit_slot_ids = list(unit_slots)
        current_usage: dict[str, int] = {}
        ring_epsilon_sum = 0.0
        ring_angle_sum = 0.0
        ring_count = 0
        future_demand = _future_quota_demand(quota_plans, unit_index + 1)
        mirror_mean = mirror_tracker.mirror_mean_for(quota_plan)
        while remaining_unit_slot_ids:
            remaining_counts = {
                cluster_id: len(pool) for cluster_id, pool in pools.items() if len(pool) > 0
            }
            if not remaining_counts:
                raise ValueError("cluster_mpc exhausted all cluster pools before filling slots")

            decisions: list[ClusterMPCDecision] = []
            for cluster_id in sorted(remaining_counts):
                if cluster_id not in current_inventory.clusters:
                    raise ValueError(f"cluster {cluster_id} is missing from inventory")
                decision = _score_cluster_for_current_ring_cached(
                    lookup,
                    residual,
                    remaining_unit_slot_ids,
                    current_inventory.clusters[cluster_id],
                    quota_plan,
                    current_usage,
                    ring_epsilon_sum,
                    ring_count,
                    remaining_counts,
                    future_demand,
                    mpc_config,
                    allowed_orientation_ids=allowed_orientation_ids,
                    mirror_mean_epsilon=mirror_mean,
                )
                decisions.append(decision)
            selected = min(
                decisions, key=lambda decision: (decision.total_score, decision.cluster_id)
            )
            pool = pools[selected.cluster_id]
            if not pool:
                raise ValueError(
                    f"cluster_mpc selected cluster {selected.cluster_id} but pool is empty"
                )
            magnet_id = pool.pop(0)
            magnet = magnet_by_id[magnet_id]
            candidate = _choose_best_linear_candidate_cached(
                lookup,
                residual,
                remaining_unit_slot_ids,
                magnet.measured_error,
                allowed_orientation_ids=allowed_orientation_ids,
            )
            current_inventory = decrement_cluster(current_inventory, selected.cluster_id)
            residual = np.ascontiguousarray(residual + candidate.contribution, dtype=np.float64)
            remaining_unit_slot_ids.remove(candidate.slot_flat_id)
            current_usage[selected.cluster_id] = current_usage.get(selected.cluster_id, 0) + 1
            ring_epsilon_sum += float(magnet.measured_error.epsilon_parallel)
            ring_angle_sum += _angle_error(magnet.measured_error)
            ring_count += 1
            placements.append(
                Placement(
                    slot_flat_id=candidate.slot_flat_id,
                    magnet_id=magnet.magnet_id,
                    orientation_id=candidate.orientation_id,
                    cluster_requested=selected.cluster_id,
                    insert_order=len(placements),
                    decision_engine="cluster_mpc",
                )
            )
        mirror_tracker.complete_ring(
            quota_plan,
            count=ring_count,
            epsilon_sum=ring_epsilon_sum,
            angle_error_sum=ring_angle_sum,
            residual=residual,
            cluster_counts=current_usage,
        )

    linear_score = float(np.dot(residual, residual))
    if not math.isfinite(linear_score):
        raise ValueError("cluster_mpc assignment produced a non-finite score")
    return LinearAssignmentResult(
        placements=tuple(placements),
        residual=residual,
        linear_score=linear_score,
        remaining_slot_flat_ids=(),
        inventory=current_inventory,
        mirror_pair_summaries=mirror_tracker.summaries(),
    )


def cluster_usage_from_placements(placements: Sequence[Placement]) -> dict[str, int]:
    """Count cluster_requested values in a completed placement sequence."""
    usage: dict[str, int] = {}
    for placement in placements:
        if placement.cluster_requested is None:
            continue
        usage[placement.cluster_requested] = usage.get(placement.cluster_requested, 0) + 1
    return dict(sorted(usage.items()))


def planned_cluster_counts(assignments: Sequence[ClusterAssignment]) -> dict[str, int]:
    """Count normal cluster assignments."""
    counts: dict[str, int] = {}
    for assignment in assignments:
        if assignment.cluster_id is None:
            continue
        counts[assignment.cluster_id] = counts.get(assignment.cluster_id, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "build_quota_ordered_magnet_order",
    "choose_best_linear_candidate",
    "cluster_usage_from_placements",
    "planned_cluster_counts",
    "run_cluster_mpc_ring_constrained_linear_assignment",
    "run_quota_ordered_ring_constrained_linear_assignment",
    "run_ring_constrained_linear_assignment",
    "run_linear_sensitivity_assignment",
    "score_cluster_for_current_ring",
    "score_linear_candidates",
]
