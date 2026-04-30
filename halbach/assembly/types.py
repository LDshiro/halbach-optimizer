from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

WorkUnitMode = Literal["all_slots", "single_physical_ring", "ring_group"]
BuildWorkUnitMode = Literal["all_slots", "single_physical_ring", "ring_group", "auto"]
PlacementOrientationMode = Literal["fixed_o0", "random_discrete4"]
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
    """Random placement baseline and its fixed-model field evaluation."""

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
    """Linear sensitivity placement and its fixed-model field evaluation."""

    assignment: LinearAssignmentResult
    evaluation: FieldEvaluation


@dataclass(frozen=True)
class SimulationComparisonResult:
    """Random baseline versus Plan C linear sensitivity for one trial."""

    trial_id: int
    random_baseline: RandomBaselineResult
    linear: LinearSimulationResult
    rms_ratio_linear_over_random: float
    j_ratio_linear_over_random: float


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
    "BuildWorkUnitMode",
    "ClusterAssignment",
    "ClusterInventory",
    "ClusterStats",
    "FieldEvaluation",
    "FieldMetrics",
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
    "SensitivityTable",
    "SimulationComparisonResult",
    "VirtualMagnet",
    "WorkUnit",
    "WorkUnitMode",
]
