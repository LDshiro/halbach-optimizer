from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from halbach.assembly.field_eval import evaluate_fixed_placement
from halbach.assembly.online_assignment import (
    run_cluster_mpc_ring_constrained_linear_assignment,
    run_linear_sensitivity_assignment,
    run_quota_ordered_ring_constrained_linear_assignment,
    run_ring_constrained_linear_assignment,
)
from halbach.assembly.orientations import default_orientations
from halbach.assembly.self_consistent_assignment import (
    SelfConsistentConfig,
    evaluate_self_consistent_placement,
    run_self_consistent_assignment,
)
from halbach.assembly.types import (
    AssemblySlot,
    ClusterAssignment,
    ClusterInventory,
    ClusterMPCConfig,
    ClusterPickupPolicy,
    EvaluationModel,
    FieldEvaluation,
    LinearSimulationResult,
    Placement,
    PlacementOrientationMode,
    RandomBaselineResult,
    RingQuotaPlan,
    SelfConsistentSimulationResult,
    SensitivityTable,
    SimulationComparisonResult,
    VirtualMagnet,
    WorkUnit,
)
from halbach.constants import FACTOR
from halbach.types import FloatArray


def _validate_random_inputs(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
) -> None:
    if not slots:
        raise ValueError("slots must be non-empty")
    if not magnets:
        raise ValueError("magnets must be non-empty")
    if len(slots) != len(magnets):
        raise ValueError("random baseline requires the same number of slots and magnets")
    slot_ids = [slot.slot_flat_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")
    magnet_ids = [magnet.magnet_id for magnet in magnets]
    if len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("magnets contain duplicate magnet_id values")


def _slots_in_work_unit_order(
    slots: Sequence[AssemblySlot],
    work_units: Sequence[WorkUnit] | None,
) -> list[AssemblySlot]:
    if work_units is None:
        return list(slots)
    if not work_units:
        raise ValueError("work_units must be non-empty")
    slot_by_id = {slot.slot_flat_id: slot for slot in slots}
    if len(slot_by_id) != len(slots):
        raise ValueError("slots contain duplicate slot_flat_id values")
    ordered: list[AssemblySlot] = []
    seen: set[int] = set()
    for unit in work_units:
        if not unit.slot_flat_ids:
            raise ValueError(f"work unit {unit.work_unit_id} has no slots")
        for raw_slot_id in unit.slot_flat_ids:
            slot_id = int(raw_slot_id)
            if slot_id in seen:
                raise ValueError(f"slot_flat_id {slot_id} appears in multiple work units")
            if slot_id not in slot_by_id:
                raise ValueError(f"work unit references unknown slot_flat_id: {slot_id}")
            seen.add(slot_id)
            ordered.append(slot_by_id[slot_id])
    slot_ids = set(slot_by_id)
    if seen != slot_ids:
        missing = sorted(slot_ids - seen)
        extra = sorted(seen - slot_ids)
        raise ValueError(f"work unit slot coverage mismatch; missing={missing}, extra={extra}")
    return ordered


def build_random_placements(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    *,
    seed: int,
    orientation_mode: PlacementOrientationMode = "fixed_o0",
    work_units: Sequence[WorkUnit] | None = None,
) -> list[Placement]:
    """
    Build a reproducible random one-to-one placement baseline.

    fixed_o0 keeps all magnets at O0. random_discrete4 samples O0/O90/O180/O270
    independently for each placed magnet.
    """
    _validate_random_inputs(slots, magnets)
    if orientation_mode not in ("fixed_o0", "random_discrete4"):
        raise ValueError(f"unsupported orientation_mode: {orientation_mode}")

    ordered_slots = _slots_in_work_unit_order(slots, work_units)
    rng = np.random.default_rng(int(seed))
    magnet_ids = np.asarray([magnet.magnet_id for magnet in magnets], dtype=np.int64)
    shuffled_magnet_ids = rng.permutation(magnet_ids)

    if orientation_mode == "fixed_o0":
        orientation_ids = ["O0"] * len(ordered_slots)
    else:
        candidates = default_orientations()
        indices = rng.integers(0, len(candidates), size=len(ordered_slots))
        orientation_ids = [candidates[int(idx)].id for idx in indices]

    placements: list[Placement] = []
    for insert_order, (slot, magnet_id, orientation_id) in enumerate(
        zip(ordered_slots, shuffled_magnet_ids, orientation_ids, strict=True)
    ):
        placements.append(
            Placement(
                slot_flat_id=slot.slot_flat_id,
                magnet_id=int(magnet_id),
                orientation_id=orientation_id,
                cluster_requested=None,
                insert_order=insert_order,
                decision_engine="random_baseline",
            )
        )
    return placements


def _evaluate_placement(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    placements: Sequence[Placement],
    pts: FloatArray,
    *,
    evaluation_model: EvaluationModel,
    self_consistent_evaluation_config: SelfConsistentConfig | None,
    factor: float,
    min_B0_norm: float,
) -> FieldEvaluation:
    if evaluation_model == "fixed":
        return evaluate_fixed_placement(
            slots,
            magnets,
            placements,
            pts,
            factor=float(factor),
            min_B0_norm=min_B0_norm,
        )
    config = (
        SelfConsistentConfig(factor=float(factor), min_b0_norm=min_B0_norm)
        if self_consistent_evaluation_config is None
        else self_consistent_evaluation_config
    )
    return evaluate_self_consistent_placement(
        slots,
        magnets,
        placements,
        pts,
        config,
    )


def run_random_baseline(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    pts: FloatArray,
    *,
    seed: int,
    orientation_mode: PlacementOrientationMode = "fixed_o0",
    evaluation_model: EvaluationModel = "fixed",
    self_consistent_evaluation_config: SelfConsistentConfig | None = None,
    work_units: Sequence[WorkUnit] | None = None,
    factor: float = FACTOR,
    min_B0_norm: float = 1e-18,
) -> RandomBaselineResult:
    """Generate a random placement baseline and evaluate its ROI field."""
    placements = build_random_placements(
        slots,
        magnets,
        seed=seed,
        orientation_mode=orientation_mode,
        work_units=work_units,
    )
    evaluation = _evaluate_placement(
        slots,
        magnets,
        placements,
        pts,
        evaluation_model=evaluation_model,
        self_consistent_evaluation_config=self_consistent_evaluation_config,
        factor=float(factor),
        min_B0_norm=min_B0_norm,
    )
    return RandomBaselineResult(placements=tuple(placements), evaluation=evaluation)


def _validate_slot_table_coverage(
    slots: Sequence[AssemblySlot],
    table: SensitivityTable,
) -> None:
    slot_ids = [slot.slot_flat_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slots contain duplicate slot_flat_id values")
    table_slot_ids = [int(slot_id) for slot_id in table.slot_flat_id.tolist()]
    if set(slot_ids) != set(table_slot_ids):
        missing_in_table = sorted(set(slot_ids) - set(table_slot_ids))
        missing_in_slots = sorted(set(table_slot_ids) - set(slot_ids))
        raise ValueError(
            "slot/table coverage mismatch; "
            f"missing_in_table={missing_in_table}, missing_in_slots={missing_in_slots}"
        )


def run_linear_sensitivity_baseline(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    sensitivity_table: SensitivityTable,
    pts: FloatArray,
    *,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
    work_units: Sequence[WorkUnit] | None = None,
    cluster_pickup_policy: ClusterPickupPolicy | None = None,
    cluster_mpc_config: ClusterMPCConfig | None = None,
    quota_plans: Sequence[RingQuotaPlan] | None = None,
    pickup_seed: int = 0,
    evaluation_model: EvaluationModel = "fixed",
    self_consistent_evaluation_config: SelfConsistentConfig | None = None,
    factor: float = FACTOR,
    min_B0_norm: float = 1e-18,
) -> LinearSimulationResult:
    """Run the Step 5 greedy Plan C linear sensitivity placement and evaluate it."""
    _validate_slot_table_coverage(slots, sensitivity_table)
    if cluster_pickup_policy not in (None, "quota_ordered", "cluster_mpc"):
        raise ValueError(f"unsupported cluster_pickup_policy: {cluster_pickup_policy}")
    if cluster_pickup_policy == "quota_ordered":
        if work_units is None:
            raise ValueError("quota_ordered pickup requires work_units")
        if quota_plans is None:
            raise ValueError("quota_ordered pickup requires quota_plans")
        if assignments is None:
            raise ValueError("quota_ordered pickup requires cluster assignments")
        assignment = run_quota_ordered_ring_constrained_linear_assignment(
            sensitivity_table,
            magnets,
            work_units,
            quota_plans,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            seed=pickup_seed,
        )
    elif cluster_pickup_policy == "cluster_mpc":
        if work_units is None:
            raise ValueError("cluster_mpc pickup requires work_units")
        if quota_plans is None:
            raise ValueError("cluster_mpc pickup requires quota_plans")
        if assignments is None:
            raise ValueError("cluster_mpc pickup requires cluster assignments")
        if inventory is None:
            raise ValueError("cluster_mpc pickup requires inventory")
        assignment = run_cluster_mpc_ring_constrained_linear_assignment(
            sensitivity_table,
            magnets,
            work_units,
            quota_plans,
            assignments=assignments,
            inventory=inventory,
            config=cluster_mpc_config,
            allowed_orientation_ids=allowed_orientation_ids,
            seed=pickup_seed,
        )
    elif work_units is None:
        assignment = run_linear_sensitivity_assignment(
            sensitivity_table,
            magnets,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            magnet_order=magnet_order,
        )
    else:
        assignment = run_ring_constrained_linear_assignment(
            sensitivity_table,
            magnets,
            work_units,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            magnet_order=magnet_order,
        )
    evaluation = _evaluate_placement(
        slots,
        magnets,
        assignment.placements,
        pts,
        evaluation_model=evaluation_model,
        self_consistent_evaluation_config=self_consistent_evaluation_config,
        factor=float(factor),
        min_B0_norm=min_B0_norm,
    )
    return LinearSimulationResult(assignment=assignment, evaluation=evaluation)


def run_self_consistent_baseline(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    sensitivity_table: SensitivityTable,
    pts: FloatArray,
    *,
    config: SelfConsistentConfig | None = None,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    allowed_orientation_ids: Sequence[str] | None = None,
    magnet_order: Sequence[int] | None = None,
    work_units: Sequence[WorkUnit] | None = None,
    evaluate_completed_work_units: bool = False,
    work_unit_evaluation_stride: int = 1,
    factor: float = FACTOR,
    min_B0_norm: float = 1e-18,
) -> SelfConsistentSimulationResult:
    """Run the Step 9 sequential self-consistent placement and evaluate it."""
    _validate_slot_table_coverage(slots, sensitivity_table)
    sc_config = (
        SelfConsistentConfig(factor=float(factor), min_b0_norm=min_B0_norm)
        if config is None
        else config
    )
    assignment = run_self_consistent_assignment(
        sensitivity_table,
        slots,
        magnets,
        pts,
        sc_config,
        assignments=assignments,
        inventory=inventory,
        allowed_orientation_ids=allowed_orientation_ids,
        magnet_order=magnet_order,
        work_units=work_units,
        evaluate_completed_work_units=evaluate_completed_work_units,
        work_unit_evaluation_stride=work_unit_evaluation_stride,
    )
    return SelfConsistentSimulationResult(
        placements=assignment.placements,
        evaluation=assignment.final_evaluation,
        evaluated_count=assignment.evaluated_count,
    )


def run_simulation_trial(
    slots: Sequence[AssemblySlot],
    magnets: Sequence[VirtualMagnet],
    sensitivity_table: SensitivityTable,
    pts: FloatArray,
    *,
    trial_id: int,
    seed: int,
    assignments: Sequence[ClusterAssignment] | None = None,
    inventory: ClusterInventory | None = None,
    random_orientation_mode: PlacementOrientationMode = "random_discrete4",
    allowed_orientation_ids: Sequence[str] | None = None,
    work_units: Sequence[WorkUnit] | None = None,
    cluster_pickup_policy: ClusterPickupPolicy | None = None,
    cluster_mpc_config: ClusterMPCConfig | None = None,
    quota_plans: Sequence[RingQuotaPlan] | None = None,
    include_self_consistent: bool = False,
    self_consistent_config: SelfConsistentConfig | None = None,
    self_consistent_evaluate_completed_work_units: bool = False,
    self_consistent_work_unit_evaluation_stride: int = 1,
    evaluation_model: EvaluationModel = "fixed",
    self_consistent_evaluation_config: SelfConsistentConfig | None = None,
    factor: float = FACTOR,
    min_B0_norm: float = 1e-18,
) -> SimulationComparisonResult:
    """Compare random, linear sensitivity, and optional self-consistent Plan C placements."""
    random_result = run_random_baseline(
        slots,
        magnets,
        pts,
        seed=seed,
        orientation_mode=random_orientation_mode,
        evaluation_model=evaluation_model,
        self_consistent_evaluation_config=self_consistent_evaluation_config,
        work_units=work_units,
        factor=float(factor),
        min_B0_norm=min_B0_norm,
    )
    linear_result = run_linear_sensitivity_baseline(
        slots,
        magnets,
        sensitivity_table,
        pts,
        assignments=assignments,
        inventory=inventory,
        allowed_orientation_ids=allowed_orientation_ids,
        work_units=work_units,
        cluster_pickup_policy=cluster_pickup_policy,
        cluster_mpc_config=cluster_mpc_config,
        quota_plans=quota_plans,
        pickup_seed=seed,
        evaluation_model=evaluation_model,
        self_consistent_evaluation_config=self_consistent_evaluation_config,
        factor=float(factor),
        min_B0_norm=min_B0_norm,
    )

    random_rms = random_result.evaluation.metrics.rms_homogeneity_ppm
    random_j = random_result.evaluation.metrics.J_vector
    linear_rms = linear_result.evaluation.metrics.rms_homogeneity_ppm
    linear_j = linear_result.evaluation.metrics.J_vector
    rms_ratio = float("inf") if random_rms <= 0.0 else float(linear_rms / random_rms)
    j_ratio = float("inf") if random_j <= 0.0 else float(linear_j / random_j)
    self_consistent_result = None
    sc_rms_ratio = None
    sc_j_ratio = None
    if include_self_consistent:
        self_consistent_result = run_self_consistent_baseline(
            slots,
            magnets,
            sensitivity_table,
            pts,
            config=self_consistent_config or self_consistent_evaluation_config,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            work_units=work_units,
            evaluate_completed_work_units=self_consistent_evaluate_completed_work_units,
            work_unit_evaluation_stride=self_consistent_work_unit_evaluation_stride,
            factor=float(factor),
            min_B0_norm=min_B0_norm,
        )
        sc_rms = self_consistent_result.evaluation.metrics.rms_homogeneity_ppm
        sc_j = self_consistent_result.evaluation.metrics.J_vector
        sc_rms_ratio = float("inf") if linear_rms <= 0.0 else float(sc_rms / linear_rms)
        sc_j_ratio = float("inf") if linear_j <= 0.0 else float(sc_j / linear_j)
    return SimulationComparisonResult(
        trial_id=int(trial_id),
        random_baseline=random_result,
        linear=linear_result,
        rms_ratio_linear_over_random=rms_ratio,
        j_ratio_linear_over_random=j_ratio,
        self_consistent=self_consistent_result,
        rms_ratio_self_consistent_over_linear=sc_rms_ratio,
        j_ratio_self_consistent_over_linear=sc_j_ratio,
    )


def summarize_comparison_results(
    results: Sequence[SimulationComparisonResult],
) -> dict[str, object]:
    """Return compact aggregate metrics for Plan C simulation comparisons."""
    if not results:
        raise ValueError("results must be non-empty")
    random_rms = np.array(
        [result.random_baseline.evaluation.metrics.rms_homogeneity_ppm for result in results],
        dtype=np.float64,
    )
    linear_rms = np.array(
        [result.linear.evaluation.metrics.rms_homogeneity_ppm for result in results],
        dtype=np.float64,
    )
    ratios = np.array(
        [result.rms_ratio_linear_over_random for result in results],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "trials": len(results),
        "random_rms_ppm_mean": float(np.mean(random_rms)),
        "linear_rms_ppm_mean": float(np.mean(linear_rms)),
        "rms_ratio_mean": float(np.mean(ratios)),
        "rms_ratio_median": float(np.median(ratios)),
        "linear_improved_count": int(np.sum(linear_rms < random_rms)),
    }
    sc_results = [result for result in results if result.self_consistent is not None]
    if sc_results:
        sc_rms = np.array(
            [
                result.self_consistent.evaluation.metrics.rms_homogeneity_ppm
                for result in sc_results
                if result.self_consistent is not None
            ],
            dtype=np.float64,
        )
        sc_ratios = np.array(
            [
                result.rms_ratio_self_consistent_over_linear
                for result in sc_results
                if result.rms_ratio_self_consistent_over_linear is not None
            ],
            dtype=np.float64,
        )
        linear_subset = np.array(
            [result.linear.evaluation.metrics.rms_homogeneity_ppm for result in sc_results],
            dtype=np.float64,
        )
        summary.update(
            {
                "self_consistent_trials": len(sc_results),
                "self_consistent_rms_ppm_mean": float(np.mean(sc_rms)),
                "rms_ratio_self_consistent_over_linear_mean": float(np.mean(sc_ratios)),
                "rms_ratio_self_consistent_over_linear_median": float(np.median(sc_ratios)),
                "self_consistent_improved_over_linear_count": int(np.sum(sc_rms < linear_subset)),
            }
        )
    return summary


__all__ = [
    "build_random_placements",
    "run_linear_sensitivity_baseline",
    "run_random_baseline",
    "run_self_consistent_baseline",
    "run_simulation_trial",
    "summarize_comparison_results",
]
