from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from halbach.assembly.field_eval import (
    compute_field_metrics,
    evaluate_fixed_placement,
    moment_vector_from_error,
)
from halbach.assembly.inventory import decrement_cluster
from halbach.assembly.online_assignment import score_linear_candidates
from halbach.assembly.orientations import rotate_error_for_orientation
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    ClusterInventory,
    FieldEvaluation,
    FieldMetrics,
    LinearCandidate,
    MagnetError,
    Placement,
    SensitivityTable,
    VirtualMagnet,
)
from halbach.constants import FACTOR, mu0
from halbach.physics import compute_B_and_B0_from_m_flat
from halbach.types import FloatArray


@dataclass(frozen=True)
class SelfConsistentConfig:
    """
    Small-run sequential self-consistent candidate re-evaluation config.

    volume_m3: magnet volume used by the easy-axis update, units m^3
    max_linear_candidates: number of linear top-k candidates to re-evaluate
    """

    chi: float = 0.0
    Nd: float = 1.0 / 3.0
    volume_m3: float = 1.0e-9
    iters: int = 30
    omega: float = 0.6
    factor: float = FACTOR
    max_linear_candidates: int = 8
    min_b0_norm: float = 1.0e-18


@dataclass(frozen=True)
class SelfConsistentCandidateEvaluation:
    """Self-consistent score for one linear-pruned candidate."""

    candidate: LinearCandidate
    metrics: FieldMetrics
    score: float
    linear_score: float
    elapsed_s: float


@dataclass(frozen=True)
class SelfConsistentDecision:
    """Selected candidate and all evaluated candidates for one placement step."""

    selected: SelfConsistentCandidateEvaluation
    evaluations: tuple[SelfConsistentCandidateEvaluation, ...]
    evaluated_count: int
    linear_best_slot_flat_id: int
    linear_best_orientation_id: str


@dataclass(frozen=True)
class SelfConsistentAssignmentResult:
    """Completed sequential self-consistent assignment."""

    placements: tuple[Placement, ...]
    decisions: tuple[SelfConsistentDecision, ...]
    evaluated_count: int
    final_evaluation: FieldEvaluation
    inventory: ClusterInventory | None = None


def _validate_config(config: SelfConsistentConfig) -> None:
    if not math.isfinite(config.chi) or config.chi < 0.0:
        raise ValueError("chi must be finite and >= 0")
    if not math.isfinite(config.Nd) or config.Nd < 0.0:
        raise ValueError("Nd must be finite and >= 0")
    if not math.isfinite(config.volume_m3) or config.volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be finite and positive")
    if config.iters < 0:
        raise ValueError("iters must be >= 0")
    if not math.isfinite(config.omega) or config.omega < 0.0 or config.omega > 1.0:
        raise ValueError("omega must be in [0, 1]")
    if not math.isfinite(config.factor) or config.factor < 0.0:
        raise ValueError("factor must be finite and >= 0")
    if config.max_linear_candidates <= 0:
        raise ValueError("max_linear_candidates must be positive")
    if not math.isfinite(config.min_b0_norm) or config.min_b0_norm < 0.0:
        raise ValueError("min_b0_norm must be finite and >= 0")


def _slot_order(slots: Sequence[AssemblySlot]) -> list[AssemblySlot]:
    if not slots:
        raise ValueError("slots must be non-empty")
    ids = [slot.slot_flat_id for slot in slots]
    if len(ids) != len(set(ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")
    return sorted(slots, key=lambda slot: slot.slot_flat_id)


def _magnet_map(magnets: Sequence[VirtualMagnet]) -> dict[int, VirtualMagnet]:
    by_id: dict[int, VirtualMagnet] = {}
    for magnet in magnets:
        if magnet.magnet_id in by_id:
            raise ValueError(f"duplicate magnet_id: {magnet.magnet_id}")
        by_id[magnet.magnet_id] = magnet
    return by_id


def _zero_error() -> MagnetError:
    return MagnetError(
        epsilon_parallel=0.0,
        delta_perp_1=0.0,
        delta_perp_2=0.0,
    )


def _placement_by_slot(placements: Sequence[Placement]) -> dict[int, Placement]:
    by_slot: dict[int, Placement] = {}
    for placement in placements:
        if placement.slot_flat_id in by_slot:
            raise ValueError(f"duplicate placement slot_flat_id: {placement.slot_flat_id}")
        by_slot[placement.slot_flat_id] = placement
    return by_slot


def _model_arrays_for_candidate(
    slots: Sequence[AssemblySlot],
    placed_placements: Sequence[Placement],
    placed_magnets: Mapping[int, VirtualMagnet],
    current_magnet: VirtualMagnet | None,
    candidate: LinearCandidate | None,
    *,
    use_measured_errors: bool,
) -> tuple[FloatArray, FloatArray]:
    ordered_slots = _slot_order(slots)
    placed_by_slot = _placement_by_slot(placed_placements)
    slot_ids = {slot.slot_flat_id for slot in ordered_slots}
    unknown_slots = sorted(set(placed_by_slot) - slot_ids)
    if unknown_slots:
        raise ValueError(f"placements reference unknown slot ids: {unknown_slots}")
    if candidate is not None and candidate.slot_flat_id in placed_by_slot:
        raise ValueError("candidate slot is already occupied")
    if candidate is not None and candidate.slot_flat_id not in slot_ids:
        raise ValueError(f"candidate references unknown slot id: {candidate.slot_flat_id}")

    r0_flat = np.empty((len(ordered_slots), 3), dtype=np.float64)
    m_flat = np.empty((len(ordered_slots), 3), dtype=np.float64)
    zero = _zero_error()
    for idx, slot in enumerate(ordered_slots):
        orientation_id = "O0"
        error = zero
        placement = placed_by_slot.get(slot.slot_flat_id)
        if placement is not None:
            if placement.magnet_id not in placed_magnets:
                raise ValueError(f"missing placed magnet id: {placement.magnet_id}")
            magnet = placed_magnets[placement.magnet_id]
            error = magnet.measured_error if use_measured_errors else magnet.true_error
            orientation_id = placement.orientation_id
        elif candidate is not None and slot.slot_flat_id == candidate.slot_flat_id:
            if current_magnet is None:
                raise ValueError("current_magnet is required when candidate is provided")
            error = current_magnet.measured_error if use_measured_errors else current_magnet.true_error
            orientation_id = candidate.orientation_id

        rotated_error = rotate_error_for_orientation(error, orientation_id)
        r0_flat[idx, :] = np.asarray(slot.center_m, dtype=np.float64)
        m_flat[idx, :] = moment_vector_from_error(slot, rotated_error)
    return (
        np.ascontiguousarray(r0_flat, dtype=np.float64),
        np.ascontiguousarray(m_flat, dtype=np.float64),
    )


def _validate_p0_flat_override(
    p0_flat_override: FloatArray | None,
    *,
    expected_size: int,
) -> FloatArray | None:
    if p0_flat_override is None:
        return None
    p0 = np.asarray(p0_flat_override, dtype=np.float64).reshape(-1)
    if p0.shape != (expected_size,):
        raise ValueError(f"p0_flat_override must have shape ({expected_size},)")
    if not np.all(np.isfinite(p0)):
        raise ValueError("p0_flat_override must contain finite values")
    if np.any(p0 <= 0.0):
        raise ValueError("p0_flat_override values must be positive")
    return np.ascontiguousarray(p0, dtype=np.float64)


def _solve_easy_axis_p_full(
    r0_flat: FloatArray,
    u_flat: FloatArray,
    p0_flat: FloatArray,
    config: SelfConsistentConfig,
) -> FloatArray:
    if config.chi == 0.0 or config.iters == 0:
        return np.ascontiguousarray(p0_flat, dtype=np.float64)

    p = np.asarray(p0_flat, dtype=np.float64).copy()
    denom = 1.0 + float(config.chi) * float(config.Nd)
    h_factor = float(config.factor) / float(mu0)
    for _ in range(int(config.iters)):
        h_axis = np.zeros_like(p)
        m_flat = p[:, None] * u_flat
        for i in range(p.size):
            ui = u_flat[i]
            ri = r0_flat[i]
            total = 0.0
            for j in range(p.size):
                if i == j:
                    continue
                d = ri - r0_flat[j]
                r2 = float(np.dot(d, d))
                if r2 <= 0.0:
                    continue
                rmag = math.sqrt(r2)
                rhat = d / rmag
                mj = m_flat[j]
                invr3 = 1.0 / (rmag * r2)
                field = h_factor * (3.0 * float(np.dot(mj, rhat)) * rhat - mj) * invr3
                total += float(np.dot(field, ui))
            h_axis[i] = total
        p_new = (p0_flat + float(config.chi) * float(config.volume_m3) * h_axis) / denom
        p = (1.0 - float(config.omega)) * p + float(config.omega) * p_new
    return np.ascontiguousarray(p, dtype=np.float64)


def evaluate_self_consistent_arrays(
    r0_flat: FloatArray,
    m_fixed: FloatArray,
    pts: FloatArray,
    config: SelfConsistentConfig,
    *,
    p0_flat_override: FloatArray | None = None,
) -> FieldEvaluation:
    """Evaluate a fixed-axis self-consistent model from full candidate arrays."""
    _validate_config(config)
    r0 = np.asarray(r0_flat, dtype=np.float64)
    m0 = np.asarray(m_fixed, dtype=np.float64)
    points = np.asarray(pts, dtype=np.float64)
    if r0.ndim != 2 or r0.shape[1] != 3:
        raise ValueError("r0_flat must have shape (M, 3)")
    if m0.shape != r0.shape:
        raise ValueError("m_fixed must have the same shape as r0_flat")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("pts must have shape (P, 3) with P >= 1")

    base_p0 = np.linalg.norm(m0, axis=1)
    if np.any(base_p0 <= 0.0):
        raise ValueError("all fixed moment vectors must be non-zero")
    override = _validate_p0_flat_override(p0_flat_override, expected_size=r0.shape[0])
    p0 = override if override is not None else base_p0
    if np.any(p0 <= 0.0):
        raise ValueError("all p0 values must be positive")
    u = np.ascontiguousarray(m0 / base_p0[:, None], dtype=np.float64)
    p = _solve_easy_axis_p_full(r0, u, np.ascontiguousarray(p0, dtype=np.float64), config)
    m_sc = np.ascontiguousarray(p[:, None] * u, dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64)
    B, B0 = compute_B_and_B0_from_m_flat(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(r0, dtype=np.float64),
        m_sc,
        float(config.factor),
        origin,
    )
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    B0_arr = np.ascontiguousarray(B0, dtype=np.float64)
    metrics = compute_field_metrics(B_arr, B0_arr, min_B0_norm=config.min_b0_norm)
    return FieldEvaluation(
        pts=np.ascontiguousarray(points, dtype=np.float64),
        B=B_arr,
        B0=B0_arr,
        metrics=metrics,
    )


def evaluate_self_consistent_placement(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    pts: FloatArray,
    config: SelfConsistentConfig,
    *,
    use_measured_errors: bool = False,
    p0_flat_override: FloatArray | None = None,
) -> FieldEvaluation:
    """Evaluate a complete placement with fixed-axis self-consistent moments."""
    slot_ids = {slot.slot_flat_id for slot in slots}
    placement_slot_ids = [placement.slot_flat_id for placement in placements]
    if len(placement_slot_ids) != len(slot_ids):
        raise ValueError("placements must contain exactly one assignment for every slot")
    if len(placement_slot_ids) != len(set(placement_slot_ids)):
        raise ValueError("placements contain duplicate slot_flat_id values")
    unknown_slots = sorted(set(placement_slot_ids) - slot_ids)
    missing_slots = sorted(slot_ids - set(placement_slot_ids))
    if unknown_slots or missing_slots:
        raise ValueError(
            f"placement slot coverage mismatch; unknown={unknown_slots}, missing={missing_slots}"
        )
    if config.chi == 0.0 and p0_flat_override is None and not use_measured_errors:
        return evaluate_fixed_placement(
            slots,
            magnets,
            placements,
            pts,
            factor=config.factor,
            min_B0_norm=config.min_b0_norm,
        )
    magnet_by_id = _magnet_map(magnets)
    r0_flat, m_flat = _model_arrays_for_candidate(
        slots,
        placements,
        magnet_by_id,
        None,
        None,
        use_measured_errors=use_measured_errors,
    )
    return evaluate_self_consistent_arrays(
        r0_flat,
        m_flat,
        pts,
        config,
        p0_flat_override=p0_flat_override,
    )


def evaluate_self_consistent_candidate(
    slots: Sequence[AssemblySlot],
    placed_placements: Sequence[Placement],
    placed_magnets: Mapping[int, VirtualMagnet],
    current_magnet: VirtualMagnet,
    candidate: LinearCandidate,
    pts: FloatArray,
    config: SelfConsistentConfig,
    *,
    p0_flat_override: FloatArray | None = None,
) -> FieldEvaluation:
    """Evaluate a provisional placement with unfilled slots kept nominal."""
    r0_flat, m_flat = _model_arrays_for_candidate(
        slots,
        placed_placements,
        placed_magnets,
        current_magnet,
        candidate,
        use_measured_errors=True,
    )
    return evaluate_self_consistent_arrays(
        r0_flat,
        m_flat,
        pts,
        config,
        p0_flat_override=p0_flat_override,
    )


def choose_self_consistent_candidate(
    table: SensitivityTable,
    slots: Sequence[AssemblySlot],
    placed_placements: Sequence[Placement],
    placed_magnets: Mapping[int, VirtualMagnet],
    current_magnet: VirtualMagnet,
    pts: FloatArray,
    remaining_slot_flat_ids: Sequence[int],
    config: SelfConsistentConfig,
    *,
    residual: FloatArray | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    p0_flat_override: FloatArray | None = None,
) -> SelfConsistentDecision:
    """Prune with linear sensitivity, then re-rank by self-consistent field metrics."""
    _validate_config(config)
    residual_arr = (
        np.zeros(table.C.shape[2], dtype=np.float64)
        if residual is None
        else np.asarray(residual, dtype=np.float64)
    )
    if residual_arr.shape != (table.C.shape[2],):
        raise ValueError(f"residual must have shape ({table.C.shape[2]},)")
    if not np.all(np.isfinite(residual_arr)):
        raise ValueError("residual must contain finite values")
    linear_candidates = score_linear_candidates(
        table,
        np.ascontiguousarray(residual_arr, dtype=np.float64),
        remaining_slot_flat_ids,
        current_magnet.measured_error,
        allowed_orientation_ids=allowed_orientation_ids,
    )
    if not linear_candidates:
        raise ValueError("linear pruning produced no candidates")
    selected_linear = linear_candidates[: int(config.max_linear_candidates)]
    evaluations: list[SelfConsistentCandidateEvaluation] = []
    for candidate in selected_linear:
        t0 = perf_counter()
        evaluation = evaluate_self_consistent_candidate(
            slots,
            placed_placements,
            placed_magnets,
            current_magnet,
            candidate,
            pts,
            config,
            p0_flat_override=p0_flat_override,
        )
        elapsed_s = perf_counter() - t0
        evaluations.append(
            SelfConsistentCandidateEvaluation(
                candidate=candidate,
                metrics=evaluation.metrics,
                score=evaluation.metrics.J_vector,
                linear_score=candidate.score,
                elapsed_s=elapsed_s,
            )
        )
    evaluations.sort(
        key=lambda item: (
            item.score,
            item.linear_score,
            item.candidate.slot_flat_id,
            table.orientation_id.index(item.candidate.orientation_id),
        )
    )
    best_linear = linear_candidates[0]
    return SelfConsistentDecision(
        selected=evaluations[0],
        evaluations=tuple(evaluations),
        evaluated_count=len(evaluations),
        linear_best_slot_flat_id=best_linear.slot_flat_id,
        linear_best_orientation_id=best_linear.orientation_id,
    )


def _validate_slot_table_coverage(
    slots: Sequence[AssemblySlot],
    table: SensitivityTable,
) -> None:
    slot_ids = {slot.slot_flat_id for slot in slots}
    table_slot_ids = {int(slot_id) for slot_id in table.slot_flat_id.tolist()}
    if slot_ids != table_slot_ids:
        missing_in_table = sorted(slot_ids - table_slot_ids)
        missing_in_slots = sorted(table_slot_ids - slot_ids)
        raise ValueError(
            "slot/table coverage mismatch; "
            f"missing_in_table={missing_in_table}, missing_in_slots={missing_in_slots}"
        )


def _cluster_by_magnet(
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
            raise ValueError("quarantined magnets are not supported by self-consistent assignment")
        if assignment.cluster_id is None:
            raise ValueError("normal assignment must include cluster_id")
        cluster_by_magnet[assignment.magnet_id] = assignment.cluster_id
    missing = sorted(magnet_ids - seen)
    if missing:
        raise ValueError(f"assignments missing magnet ids: {missing}")
    return cluster_by_magnet


def run_self_consistent_assignment(
    table: SensitivityTable,
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    pts: FloatArray,
    config: SelfConsistentConfig,
    *,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
    p0_flat_override: FloatArray | None = None,
) -> SelfConsistentAssignmentResult:
    """Sequential Plan C assignment using self-consistent candidate re-evaluation."""
    _validate_config(config)
    _validate_slot_table_coverage(slots, table)
    if len(magnets) != len(slots):
        raise ValueError("magnets count must match slot count")
    magnet_by_id = _magnet_map(magnets)
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

    remaining_slot_ids = [int(slot_id) for slot_id in table.slot_flat_id.tolist()]
    placed_magnets: dict[int, VirtualMagnet] = {}
    placements: list[Placement] = []
    decisions: list[SelfConsistentDecision] = []
    current_inventory = inventory
    clusters = _cluster_by_magnet(assignments, magnets)
    residual = np.zeros(table.C.shape[2], dtype=np.float64)

    for insert_order, magnet in enumerate(ordered_magnets):
        decision = choose_self_consistent_candidate(
            table,
            slots,
            placements,
            placed_magnets,
            magnet,
            pts,
            remaining_slot_ids,
            config,
            residual=residual,
            allowed_orientation_ids=allowed_orientation_ids,
            p0_flat_override=p0_flat_override,
        )
        cluster_id = clusters.get(magnet.magnet_id)
        if current_inventory is not None and cluster_id is not None:
            current_inventory = decrement_cluster(current_inventory, cluster_id)
        candidate = decision.selected.candidate
        placement = Placement(
            slot_flat_id=candidate.slot_flat_id,
            magnet_id=magnet.magnet_id,
            orientation_id=candidate.orientation_id,
            cluster_requested=cluster_id,
            insert_order=insert_order,
            decision_engine="sequential_self_consistent",
        )
        placements.append(placement)
        placed_magnets[magnet.magnet_id] = magnet
        residual = np.ascontiguousarray(
            residual + candidate.contribution,
            dtype=np.float64,
        )
        remaining_slot_ids.remove(candidate.slot_flat_id)
        decisions.append(decision)

    final_evaluation = evaluate_self_consistent_placement(
        slots,
        magnets,
        placements,
        pts,
        config,
        p0_flat_override=p0_flat_override,
    )
    return SelfConsistentAssignmentResult(
        placements=tuple(placements),
        decisions=tuple(decisions),
        evaluated_count=sum(decision.evaluated_count for decision in decisions),
        final_evaluation=final_evaluation,
        inventory=current_inventory,
    )


__all__ = [
    "SelfConsistentAssignmentResult",
    "SelfConsistentCandidateEvaluation",
    "SelfConsistentConfig",
    "SelfConsistentDecision",
    "choose_self_consistent_candidate",
    "evaluate_self_consistent_arrays",
    "evaluate_self_consistent_candidate",
    "evaluate_self_consistent_placement",
    "run_self_consistent_assignment",
]
