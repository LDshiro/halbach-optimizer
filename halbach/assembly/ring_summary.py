from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from halbach.assembly.types import (
    AssemblySlot,
    AssemblyTimelineEvent,
    FieldMetrics,
    MagnetError,
    Placement,
    RingKey,
    RingPairSummary,
    RingSummary,
    SimulationComparisonResult,
    VirtualMagnet,
)
from halbach.types import FloatArray


def _slot_by_id(slots: Sequence[AssemblySlot]) -> dict[int, AssemblySlot]:
    if not slots:
        raise ValueError("slots must be non-empty")
    by_id: dict[int, AssemblySlot] = {}
    for slot in slots:
        if slot.slot_flat_id in by_id:
            raise ValueError(f"duplicate slot_flat_id: {slot.slot_flat_id}")
        by_id[slot.slot_flat_id] = slot
    return by_id


def _magnet_by_id(magnets: Sequence[VirtualMagnet]) -> dict[int, VirtualMagnet]:
    if not magnets:
        raise ValueError("magnets must be non-empty")
    by_id: dict[int, VirtualMagnet] = {}
    for magnet in magnets:
        if magnet.magnet_id in by_id:
            raise ValueError(f"duplicate magnet_id: {magnet.magnet_id}")
        by_id[magnet.magnet_id] = magnet
    return by_id


def _error_vector(error: MagnetError) -> FloatArray:
    return np.array(
        [
            float(error.epsilon_parallel),
            float(error.delta_perp_1),
            float(error.delta_perp_2),
        ],
        dtype=np.float64,
    )


def _angle_error(error: MagnetError) -> float:
    return float(math.hypot(float(error.delta_perp_1), float(error.delta_perp_2)))


def _metrics_for_ring(
    metrics_by_ring: Mapping[RingKey, FieldMetrics] | None,
    ring_key: RingKey,
) -> FieldMetrics | None:
    if metrics_by_ring is None:
        return None
    return metrics_by_ring.get(ring_key)


def ring_summary_from_placements(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    *,
    use_measured_errors: bool = False,
    metrics_by_ring: Mapping[RingKey, FieldMetrics] | None = None,
) -> tuple[RingSummary, ...]:
    """
    Summarize assigned magnet errors per physical ring.

    Summary scalar fields use true errors by default for simulation analysis. Set
    use_measured_errors=True for an operator-facing view based only on measured data.
    """
    slot_by_id = _slot_by_id(slots)
    magnet_by_id = _magnet_by_id(magnets)
    seen_slots: set[int] = set()
    ring_rows: dict[RingKey, list[tuple[MagnetError, MagnetError, str | None]]] = {}
    for placement in placements:
        if placement.slot_flat_id in seen_slots:
            raise ValueError(f"duplicate placed slot_flat_id: {placement.slot_flat_id}")
        seen_slots.add(placement.slot_flat_id)
        if placement.slot_flat_id not in slot_by_id:
            raise ValueError(f"unknown placed slot_flat_id: {placement.slot_flat_id}")
        if placement.magnet_id not in magnet_by_id:
            raise ValueError(f"unknown placed magnet_id: {placement.magnet_id}")
        slot = slot_by_id[placement.slot_flat_id]
        magnet = magnet_by_id[placement.magnet_id]
        ring_key = RingKey(layer_id=slot.layer_id, ring_id=slot.ring_id)
        ring_rows.setdefault(ring_key, []).append(
            (magnet.true_error, magnet.measured_error, placement.cluster_requested)
        )

    summaries: list[RingSummary] = []
    for ring_key in sorted(ring_rows):
        rows = ring_rows[ring_key]
        true_arr = np.asarray([_error_vector(row[0]) for row in rows], dtype=np.float64)
        measured_arr = np.asarray([_error_vector(row[1]) for row in rows], dtype=np.float64)
        source_errors = [row[1] if use_measured_errors else row[0] for row in rows]
        eps = np.asarray([error.epsilon_parallel for error in source_errors], dtype=np.float64)
        angle = np.asarray([_angle_error(error) for error in source_errors], dtype=np.float64)
        cluster_counts: dict[str, int] = {}
        for _true_error, _measured_error, cluster_id in rows:
            if cluster_id is None:
                continue
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        metrics = _metrics_for_ring(metrics_by_ring, ring_key)
        summaries.append(
            RingSummary(
                ring_key=ring_key,
                layer_id=ring_key.layer_id,
                ring_id=ring_key.ring_id,
                count=len(rows),
                mean_epsilon=float(np.mean(eps)),
                std_epsilon=float(np.std(eps)),
                min_epsilon=float(np.min(eps)),
                max_epsilon=float(np.max(eps)),
                mean_angle_error=float(np.mean(angle)),
                std_angle_error=float(np.std(angle)),
                cluster_counts=dict(sorted(cluster_counts.items())),
                mean_true_error=np.ascontiguousarray(np.mean(true_arr, axis=0), dtype=np.float64),
                mean_measured_error=np.ascontiguousarray(
                    np.mean(measured_arr, axis=0),
                    dtype=np.float64,
                ),
                B0_norm_after_ring=None if metrics is None else metrics.B0_norm,
                rms_homogeneity_ppm_after_ring=(
                    None if metrics is None else metrics.rms_homogeneity_ppm
                ),
                J_vector_after_ring=None if metrics is None else metrics.J_vector,
            )
        )
    return tuple(summaries)


def ring_pair_summary_from_ring_summaries(
    summaries: Sequence[RingSummary],
) -> tuple[RingPairSummary, ...]:
    """Build mirror-pair summaries from per-ring summaries."""
    by_ring: dict[int, dict[int, RingSummary]] = {}
    for summary in summaries:
        by_ring.setdefault(summary.ring_id, {})[summary.layer_id] = summary

    pairs: list[RingPairSummary] = []
    for ring_id in sorted(by_ring):
        layer_map = by_ring[ring_id]
        layers = sorted(layer_map)
        left = 0
        right = len(layers) - 1
        pair_index = 0
        while left <= right:
            lower_layer = layers[left]
            upper_layer = layers[right]
            lower = layer_map[lower_layer]
            if lower_layer == upper_layer:
                pairs.append(
                    RingPairSummary(
                        pair_id=f"P{pair_index:03d}_R{ring_id:03d}",
                        pair_index=pair_index,
                        ring_id=ring_id,
                        lower_ring=lower.ring_key,
                        upper_ring=None,
                        lower_count=lower.count,
                        upper_count=0,
                        mean_epsilon_difference=None,
                        mean_angle_error_difference=None,
                        lower_mean_epsilon=lower.mean_epsilon,
                        upper_mean_epsilon=None,
                        pair_complete=True,
                    )
                )
            else:
                upper = layer_map[upper_layer]
                pairs.append(
                    RingPairSummary(
                        pair_id=f"P{pair_index:03d}_R{ring_id:03d}",
                        pair_index=pair_index,
                        ring_id=ring_id,
                        lower_ring=lower.ring_key,
                        upper_ring=upper.ring_key,
                        lower_count=lower.count,
                        upper_count=upper.count,
                        mean_epsilon_difference=lower.mean_epsilon - upper.mean_epsilon,
                        mean_angle_error_difference=(
                            lower.mean_angle_error - upper.mean_angle_error
                        ),
                        lower_mean_epsilon=lower.mean_epsilon,
                        upper_mean_epsilon=upper.mean_epsilon,
                        pair_complete=True,
                    )
                )
            left += 1
            right -= 1
            pair_index += 1
    return tuple(pairs)


def timeline_from_placements(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    *,
    result_label: str,
    use_measured_errors: bool = False,
    residual_norm_by_step: Mapping[int, float] | None = None,
    field_metrics_by_step: Mapping[int, FieldMetrics] | None = None,
) -> tuple[AssemblyTimelineEvent, ...]:
    """Build replayable insert events from a placement sequence."""
    slot_by_id = _slot_by_id(slots)
    magnet_by_id = _magnet_by_id(magnets)
    ring_eps: dict[RingKey, list[float]] = {}
    ring_angle: dict[RingKey, list[float]] = {}
    events: list[AssemblyTimelineEvent] = []
    seen_slots: set[int] = set()
    for step, placement in enumerate(sorted(placements, key=lambda item: item.insert_order)):
        if placement.slot_flat_id in seen_slots:
            raise ValueError(f"duplicate placed slot_flat_id: {placement.slot_flat_id}")
        seen_slots.add(placement.slot_flat_id)
        if placement.slot_flat_id not in slot_by_id:
            raise ValueError(f"unknown placed slot_flat_id: {placement.slot_flat_id}")
        if placement.magnet_id not in magnet_by_id:
            raise ValueError(f"unknown placed magnet_id: {placement.magnet_id}")
        slot = slot_by_id[placement.slot_flat_id]
        magnet = magnet_by_id[placement.magnet_id]
        error = magnet.measured_error if use_measured_errors else magnet.true_error
        eps = float(error.epsilon_parallel)
        angle = _angle_error(error)
        ring_key = RingKey(layer_id=slot.layer_id, ring_id=slot.ring_id)
        ring_eps.setdefault(ring_key, []).append(eps)
        ring_angle.setdefault(ring_key, []).append(angle)
        metrics = (
            None
            if field_metrics_by_step is None
            else field_metrics_by_step.get(placement.insert_order)
        )
        residual_norm = (
            None
            if residual_norm_by_step is None
            else residual_norm_by_step.get(placement.insert_order)
        )
        events.append(
            AssemblyTimelineEvent(
                step=step,
                event="insert_confirmed",
                result_label=result_label,
                work_unit_id=slot.work_unit_id,
                layer_id=slot.layer_id,
                ring_id=slot.ring_id,
                theta_id=slot.theta_id,
                slot_flat_id=slot.slot_flat_id,
                physical_slot_number=slot.physical_slot_number,
                magnet_id=placement.magnet_id,
                cluster_requested=placement.cluster_requested,
                epsilon_parallel=eps,
                angle_error=angle,
                orientation_id=placement.orientation_id,
                insert_order=placement.insert_order,
                decision_engine=placement.decision_engine,
                ring_count_so_far=len(ring_eps[ring_key]),
                ring_mean_epsilon_so_far=float(np.mean(ring_eps[ring_key])),
                ring_mean_angle_error_so_far=float(np.mean(ring_angle[ring_key])),
                residual_norm=residual_norm,
                B0_norm=None if metrics is None else metrics.B0_norm,
                rms_homogeneity_ppm=None if metrics is None else metrics.rms_homogeneity_ppm,
                J_vector=None if metrics is None else metrics.J_vector,
            )
        )
    return tuple(events)


def timeline_from_simulation_result(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    result: SimulationComparisonResult,
    *,
    result_label: str = "linear",
    use_measured_errors: bool = False,
) -> tuple[AssemblyTimelineEvent, ...]:
    """Build a timeline from one branch of a simulation comparison result."""
    if result_label == "random":
        placements = result.random_baseline.placements
        evaluation = result.random_baseline.evaluation
    elif result_label == "linear":
        placements = result.linear.assignment.placements
        evaluation = result.linear.evaluation
    elif result_label == "self_consistent":
        if result.self_consistent is None:
            raise ValueError("simulation result does not contain self_consistent placements")
        placements = result.self_consistent.placements
        evaluation = result.self_consistent.evaluation
    else:
        raise ValueError(f"unsupported result_label: {result_label}")
    if not placements:
        return ()
    final_step = max(placement.insert_order for placement in placements)
    return timeline_from_placements(
        slots,
        magnets,
        placements,
        result_label=result_label,
        use_measured_errors=use_measured_errors,
        field_metrics_by_step={final_step: evaluation.metrics},
    )


__all__ = [
    "ring_pair_summary_from_ring_summaries",
    "ring_summary_from_placements",
    "timeline_from_placements",
    "timeline_from_simulation_result",
]
