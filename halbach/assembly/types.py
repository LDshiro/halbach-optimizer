from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

WorkUnitMode = Literal[
    "all_slots",
    "single_physical_ring",
    "ring_group",
    "ring_by_ring_outer_to_inner",
    "mirror_ring_pair",
]
BuildWorkUnitMode = Literal[
    "all_slots",
    "single_physical_ring",
    "ring_group",
    "ring_by_ring_outer_to_inner",
    "mirror_ring_pair",
    "auto",
]
PlacementOrientationMode = Literal["fixed_o0", "random_discrete4"]
EvaluationModel = Literal["fixed", "self_consistent"]
ClusterPickupPolicy = Literal["quota_ordered"]
QuarantineReason = Literal[
    "Q_MEASUREMENT_UNSTABLE",
    "Q_DIRECTION_OUTLIER",
    "Q_STRENGTH_OUTLIER",
]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class AssemblySlot:
    """
    Active magnet slot in an optimized Halbach run.

    center_m: shape (3,), units m
    nominal_u: shape (3,), unitless nominal magnetization direction
    nominal_phi_rad: units rad
    physical_slot_number: 1-based theta order within a physical ring
    """

    slot_flat_id: int
    ring_id: int
    layer_id: int
    theta_id: int
    center_m: FloatArray
    nominal_u: FloatArray
    nominal_phi_rad: float
    physical_slot_number: int
    work_unit_id: str = ""
    mirror_pair_id: str | None = None


@dataclass(frozen=True)
class MagnetError:
    """
    Magnet error in the local nominal-magnetization frame.

    epsilon_parallel: relative moment magnitude error
    delta_perp_1: small-angle transverse component 1, rad-equivalent
    delta_perp_2: small-angle transverse component 2, rad-equivalent
    """

    epsilon_parallel: float
    delta_perp_1: float
    delta_perp_2: float


@dataclass(frozen=True)
class MeasuredMagnet:
    """
    Magnet measurement converted to Plan C error coordinates.

    direction: shape (3,), normalized measurement-frame direction
    """

    error: MagnetError
    moment_magnitude: float
    direction: FloatArray
    quality: float | None = None
    cluster_id: str | None = None


@dataclass(frozen=True)
class OrientationCandidate:
    """Allowed insertion orientation around the measured nominal magnetization axis."""

    id: str
    angle_deg: float
    instruction: str


@dataclass(frozen=True)
class WorkUnit:
    """Collection of slot ids assigned to one online assembly unit."""

    work_unit_id: str
    mode: WorkUnitMode
    slot_flat_ids: tuple[int, ...]
    label: str


@dataclass(frozen=True, order=True)
class RingKey:
    """Physical ring identifier used by ring-by-ring assembly summaries."""

    layer_id: int
    ring_id: int


@dataclass(frozen=True)
class RingSummary:
    """
    Aggregate magnet-error statistics for one physical ring.

    mean_true_error and mean_measured_error: shape (3,), components
    [epsilon_parallel, delta_perp_1, delta_perp_2].
    Field metric fields are optional until per-ring evaluation is available.
    """

    ring_key: RingKey
    layer_id: int
    ring_id: int
    count: int
    mean_epsilon: float
    std_epsilon: float
    min_epsilon: float
    max_epsilon: float
    mean_angle_error: float
    std_angle_error: float
    cluster_counts: dict[str, int]
    mean_true_error: FloatArray
    mean_measured_error: FloatArray
    B0_norm_after_ring: float | None = None
    rms_homogeneity_ppm_after_ring: float | None = None
    J_vector_after_ring: float | None = None


@dataclass(frozen=True)
class RingPairSummary:
    """Left/right mirror-pair summary for one ring_id and layer pair."""

    pair_id: str
    pair_index: int
    ring_id: int
    lower_ring: RingKey
    upper_ring: RingKey | None
    lower_count: int
    upper_count: int
    mean_epsilon_difference: float | None
    mean_angle_error_difference: float | None


@dataclass(frozen=True)
class RingQuotaPlannerConfig:
    """Weights and targets for Level 1 ring cluster quota planning."""

    target_mean_epsilon: float | None = None
    lambda_ring_mean: float = 1.0
    lambda_angle: float = 1.0
    lambda_inventory: float = 0.0
    lambda_mirror_mean: float = 1.0
    mirror_balance: bool = True


@dataclass(frozen=True)
class RingQuotaPlan:
    """
    Planned cluster quota for one physical ring.

    quota_by_cluster maps cluster id to the number of magnets planned for the ring.
    expected_mean_angle_bin is the weighted mean of Axx bin ids in the quota.
    """

    ring_key: RingKey
    layer_id: int
    ring_id: int
    work_unit_id: str
    target_count: int
    target_mean_epsilon: float
    ring_importance: float
    allowed_clusters: tuple[str, ...]
    quota_by_cluster: dict[str, int]
    expected_mean_epsilon: float
    expected_mean_angle_bin: float
    expected_angle_error: float
    mirror_pair_id: str | None = None


@dataclass(frozen=True)
class AssemblyTimelineEvent:
    """One replayable ring assembly event for visualization and later JSONL output."""

    step: int
    event: str
    result_label: str
    work_unit_id: str
    layer_id: int
    ring_id: int
    theta_id: int
    slot_flat_id: int
    physical_slot_number: int
    magnet_id: int
    cluster_requested: str | None
    epsilon_parallel: float
    angle_error: float
    orientation_id: str
    insert_order: int
    decision_engine: str
    ring_count_so_far: int
    ring_mean_epsilon_so_far: float
    ring_mean_angle_error_so_far: float
    residual_norm: float | None = None
    B0_norm: float | None = None
    rms_homogeneity_ppm: float | None = None
    J_vector: float | None = None


@dataclass(frozen=True)
class Placement:
    """
    Assignment of one measured/simulated magnet to one active slot.

    slot_flat_id: flattened active slot id from the design run
    magnet_id: measured/simulated magnet id
    orientation_id: one of the discrete Plan C insertion orientations
    """

    slot_flat_id: int
    magnet_id: int
    orientation_id: str
    cluster_requested: str | None = None
    insert_order: int = 0
    decision_engine: str = "random"


@dataclass(frozen=True)
class VirtualMagnet:
    """Synthetic magnet used by Plan C simulations."""

    magnet_id: int
    true_error: MagnetError
    measured_error: MagnetError
    quality: float | None = None


@dataclass(frozen=True)
class ClusterAssignment:
    """Cluster or quarantine assignment for one virtual/measured magnet."""

    magnet_id: int
    cluster_id: str | None
    quarantine_id: QuarantineReason | None = None


@dataclass(frozen=True)
class ClusterStats:
    """
    Cluster inventory statistics.

    mean: shape (3,), measured error vector [epsilon, delta1, delta2]
    cov: shape (3, 3), measured error covariance
    """

    cluster_id: str
    count: int
    mean: FloatArray
    cov: FloatArray


@dataclass(frozen=True)
class ClusterInventory:
    """Normal cluster counts/stats plus quarantined magnet counts."""

    clusters: dict[str, ClusterStats]
    quarantine: dict[QuarantineReason, int]


@dataclass(frozen=True)
class FieldMetrics:
    """
    Plan C vector field homogeneity metrics.

    ppm metrics use ||B(p)-B0|| / ||B0|| * 1e6.
    J_vector: mean(||B(p)-B0||^2), SI field units squared
    B0_norm: ||B0|| at the evaluation origin
    """

    rms_homogeneity_ppm: float
    max_homogeneity_ppm: float
    p95_homogeneity_ppm: float
    p99_homogeneity_ppm: float
    B0_norm: float
    J_vector: float


@dataclass(frozen=True)
class FieldEvaluation:
    """
    Fixed-model field evaluation for a placement.

    pts: shape (P, 3), units m, ROI points relative to origin
    B: shape (P, 3), magnetic flux density at pts + origin
    B0: shape (3,), magnetic flux density at origin
    """

    pts: FloatArray
    B: FloatArray
    B0: FloatArray
    metrics: FieldMetrics


@dataclass(frozen=True)
class RandomBaselineResult:
    """Random placement baseline and its field evaluation."""

    placements: tuple[Placement, ...]
    evaluation: FieldEvaluation


@dataclass(frozen=True)
class LinearCandidate:
    """One scored slot/orientation candidate for linear sensitivity assignment."""

    slot_flat_id: int
    orientation_id: str
    score: float
    contribution: FloatArray


@dataclass(frozen=True)
class LinearAssignmentResult:
    """Completed greedy linear sensitivity assignment."""

    placements: tuple[Placement, ...]
    residual: FloatArray
    linear_score: float
    remaining_slot_flat_ids: tuple[int, ...]
    inventory: ClusterInventory | None = None


@dataclass(frozen=True)
class LinearSimulationResult:
    """Linear sensitivity placement and its field evaluation."""

    assignment: LinearAssignmentResult
    evaluation: FieldEvaluation


@dataclass(frozen=True)
class SelfConsistentSimulationResult:
    """Self-consistent placement and its self-consistent field evaluation."""

    placements: tuple[Placement, ...]
    evaluation: FieldEvaluation
    evaluated_count: int


@dataclass(frozen=True)
class SimulationComparisonResult:
    """Random baseline versus Plan C linear sensitivity for one trial."""

    trial_id: int
    random_baseline: RandomBaselineResult
    linear: LinearSimulationResult
    rms_ratio_linear_over_random: float
    j_ratio_linear_over_random: float
    self_consistent: SelfConsistentSimulationResult | None = None
    rms_ratio_self_consistent_over_linear: float | None = None
    j_ratio_self_consistent_over_linear: float | None = None


@dataclass(frozen=True)
class SensitivityTable:
    """
    Precomputed Plan C linear sensitivity table.

    C: shape (S, O, residual_dim, 3), maps [epsilon, delta1, delta2] to
       flattened y = [B(p_1)-B0, ..., B(p_M)-B0]
    roi_points: shape (P, 3), units m
    normalization_b0: shape (3,), nominal B0 vector used for normalization metadata
    """

    slot_flat_id: IntArray
    ring_id: IntArray
    layer_id: IntArray
    theta_id: IntArray
    centers_m: FloatArray
    nominal_u: FloatArray
    orientation_id: tuple[str, ...]
    C: FloatArray
    roi_points: FloatArray
    normalization_b0: FloatArray
    metadata: dict[str, object]
    projection_basis: FloatArray | None = None


__all__ = [
    "AssemblySlot",
    "AssemblyTimelineEvent",
    "BuildWorkUnitMode",
    "ClusterAssignment",
    "ClusterInventory",
    "ClusterPickupPolicy",
    "ClusterStats",
    "FieldEvaluation",
    "FieldMetrics",
    "EvaluationModel",
    "IntArray",
    "LinearAssignmentResult",
    "LinearCandidate",
    "LinearSimulationResult",
    "MagnetError",
    "MeasuredMagnet",
    "OrientationCandidate",
    "Placement",
    "PlacementOrientationMode",
    "QuarantineReason",
    "RandomBaselineResult",
    "RingKey",
    "RingPairSummary",
    "RingQuotaPlan",
    "RingQuotaPlannerConfig",
    "RingSummary",
    "SensitivityTable",
    "SelfConsistentSimulationResult",
    "SimulationComparisonResult",
    "VirtualMagnet",
    "WorkUnit",
    "WorkUnitMode",
]
