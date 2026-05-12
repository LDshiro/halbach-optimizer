import numpy as np
import pytest

from halbach.assembly.ring_summary import (
    ring_pair_summary_from_ring_summaries,
    ring_summary_from_placements,
    timeline_from_placements,
    timeline_from_simulation_result,
)
from halbach.assembly.types import (
    AssemblySlot,
    FieldEvaluation,
    FieldMetrics,
    LinearAssignmentResult,
    LinearSimulationResult,
    MagnetError,
    Placement,
    RandomBaselineResult,
    RingKey,
    RingSummary,
    SimulationComparisonResult,
    VirtualMagnet,
)


def _slot(slot_id: int, layer_id: int, theta_id: int, *, ring_id: int = 0) -> AssemblySlot:
    return AssemblySlot(
        slot_flat_id=slot_id,
        ring_id=ring_id,
        layer_id=layer_id,
        theta_id=theta_id,
        center_m=np.array([0.1, 0.0, 0.01 * layer_id], dtype=np.float64),
        nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        nominal_phi_rad=0.0,
        physical_slot_number=theta_id + 1,
        work_unit_id=f"W_K{layer_id:03d}_R{ring_id:03d}",
        mirror_pair_id=f"P{min(layer_id, 3 - layer_id):03d}_R{ring_id:03d}",
    )


def _magnet(
    magnet_id: int,
    true_error: MagnetError,
    *,
    measured_error: MagnetError | None = None,
) -> VirtualMagnet:
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=true_error,
        measured_error=true_error if measured_error is None else measured_error,
        quality=1.0,
    )


def _metrics(B0_norm: float, rms: float, J_vector: float) -> FieldMetrics:
    return FieldMetrics(
        rms_homogeneity_ppm=rms,
        max_homogeneity_ppm=rms * 2.0,
        p95_homogeneity_ppm=rms * 1.5,
        p99_homogeneity_ppm=rms * 1.8,
        B0_norm=B0_norm,
        J_vector=J_vector,
    )


def _evaluation(metrics: FieldMetrics) -> FieldEvaluation:
    return FieldEvaluation(
        pts=np.zeros((1, 3), dtype=np.float64),
        B=np.zeros((1, 3), dtype=np.float64),
        B0=np.array([metrics.B0_norm, 0.0, 0.0], dtype=np.float64),
        metrics=metrics,
    )


def _summary(layer_id: int, mean_epsilon: float, mean_angle: float) -> RingSummary:
    return RingSummary(
        ring_key=RingKey(layer_id=layer_id, ring_id=0),
        layer_id=layer_id,
        ring_id=0,
        count=2,
        mean_epsilon=mean_epsilon,
        std_epsilon=0.0,
        min_epsilon=mean_epsilon,
        max_epsilon=mean_epsilon,
        mean_angle_error=mean_angle,
        std_angle_error=0.0,
        cluster_counts={},
        mean_true_error=np.array([mean_epsilon, 0.0, 0.0], dtype=np.float64),
        mean_measured_error=np.array([mean_epsilon, 0.0, 0.0], dtype=np.float64),
    )


def test_ring_summary_from_placements_computes_ring_statistics_and_clusters() -> None:
    slots = [_slot(0, 0, 0), _slot(1, 0, 1), _slot(2, 1, 0)]
    magnets = [
        _magnet(10, MagnetError(0.10, 0.003, 0.004), measured_error=MagnetError(0.20, 0.0, 0.0)),
        _magnet(
            11,
            MagnetError(-0.10, 0.0, 0.0),
            measured_error=MagnetError(0.00, 0.0, 0.0),
        ),
        _magnet(12, MagnetError(0.30, 0.006, 0.008)),
    ]
    placements = [
        Placement(0, 10, "O0", cluster_requested="S00_A01", insert_order=0),
        Placement(1, 11, "O90", cluster_requested="S00_A01", insert_order=1),
        Placement(2, 12, "O180", cluster_requested="S02_A04", insert_order=2),
    ]
    metrics = _metrics(B0_norm=0.12, rms=5.0, J_vector=1.0e-6)

    summaries = ring_summary_from_placements(
        slots,
        magnets,
        placements,
        metrics_by_ring={RingKey(layer_id=0, ring_id=0): metrics},
    )

    assert [summary.ring_key for summary in summaries] == [
        RingKey(layer_id=0, ring_id=0),
        RingKey(layer_id=1, ring_id=0),
    ]
    first = summaries[0]
    assert first.count == 2
    assert first.mean_epsilon == pytest.approx(0.0)
    assert first.std_epsilon == pytest.approx(0.10)
    assert first.min_epsilon == pytest.approx(-0.10)
    assert first.max_epsilon == pytest.approx(0.10)
    assert first.mean_angle_error == pytest.approx(0.0025)
    assert first.cluster_counts == {"S00_A01": 2}
    np.testing.assert_allclose(first.mean_true_error, np.array([0.0, 0.0015, 0.002]))
    np.testing.assert_allclose(first.mean_measured_error, np.array([0.10, 0.0, 0.0]))
    assert first.B0_norm_after_ring == pytest.approx(0.12)
    assert first.rms_homogeneity_ppm_after_ring == pytest.approx(5.0)
    assert first.J_vector_after_ring == pytest.approx(1.0e-6)


def test_ring_summary_can_use_measured_errors_for_operator_view() -> None:
    slots = [_slot(0, 0, 0), _slot(1, 0, 1)]
    magnets = [
        _magnet(10, MagnetError(0.10, 0.0, 0.0), measured_error=MagnetError(0.20, 0.0, 0.0)),
        _magnet(11, MagnetError(-0.10, 0.0, 0.0), measured_error=MagnetError(0.00, 0.0, 0.0)),
    ]
    placements = [
        Placement(0, 10, "O0", insert_order=0),
        Placement(1, 11, "O0", insert_order=1),
    ]

    (summary,) = ring_summary_from_placements(
        slots,
        magnets,
        placements,
        use_measured_errors=True,
    )

    assert summary.mean_epsilon == pytest.approx(0.10)
    np.testing.assert_allclose(summary.mean_true_error, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(summary.mean_measured_error, np.array([0.10, 0.0, 0.0]))


def test_ring_pair_summary_reports_mirror_differences_and_center_layer() -> None:
    summaries = [
        _summary(0, 0.10, 0.010),
        _summary(4, 0.04, 0.004),
        _summary(1, 0.08, 0.008),
        _summary(3, 0.05, 0.005),
        _summary(2, 0.06, 0.006),
    ]

    pairs = ring_pair_summary_from_ring_summaries(summaries)

    assert [
        (pair.lower_ring.layer_id, None if pair.upper_ring is None else pair.upper_ring.layer_id)
        for pair in pairs
    ] == [
        (0, 4),
        (1, 3),
        (2, None),
    ]
    assert pairs[0].pair_id == "P000_R000"
    assert pairs[0].mean_epsilon_difference == pytest.approx(0.06)
    assert pairs[0].mean_angle_error_difference == pytest.approx(0.006)
    assert pairs[2].upper_count == 0
    assert pairs[2].mean_epsilon_difference is None
    assert pairs[2].mean_angle_error_difference is None


def test_timeline_from_placements_sorts_by_insert_order_and_tracks_ring_running_mean() -> None:
    slots = [_slot(0, 0, 0), _slot(1, 0, 1), _slot(2, 1, 0)]
    magnets = [
        _magnet(10, MagnetError(0.10, 0.003, 0.004)),
        _magnet(11, MagnetError(-0.02, 0.0, 0.0)),
        _magnet(12, MagnetError(0.30, 0.006, 0.008)),
    ]
    placements = [
        Placement(2, 12, "O0", cluster_requested="S02_A04", insert_order=2),
        Placement(
            0,
            10,
            "O90",
            cluster_requested="S00_A01",
            insert_order=0,
            decision_engine="ring_constrained_linear",
        ),
        Placement(1, 11, "O180", cluster_requested="S00_A00", insert_order=1),
    ]
    metrics = _metrics(B0_norm=0.11, rms=3.0, J_vector=2.0e-6)

    events = timeline_from_placements(
        slots,
        magnets,
        placements,
        result_label="linear",
        residual_norm_by_step={1: 4.5},
        field_metrics_by_step={2: metrics},
    )

    assert [event.insert_order for event in events] == [0, 1, 2]
    assert [event.step for event in events] == [0, 1, 2]
    assert events[0].slot_flat_id == 0
    assert events[0].ring_count_so_far == 1
    assert events[0].ring_mean_epsilon_so_far == pytest.approx(0.10)
    assert events[0].angle_error == pytest.approx(0.005)
    assert events[1].ring_count_so_far == 2
    assert events[1].ring_mean_epsilon_so_far == pytest.approx(0.04)
    assert events[1].residual_norm == pytest.approx(4.5)
    assert events[2].B0_norm == pytest.approx(0.11)
    assert events[2].rms_homogeneity_ppm == pytest.approx(3.0)
    assert events[2].J_vector == pytest.approx(2.0e-6)


def test_timeline_from_simulation_result_selects_requested_branch_and_final_metrics() -> None:
    slots = [_slot(0, 0, 0), _slot(1, 0, 1)]
    magnets = [
        _magnet(10, MagnetError(0.10, 0.0, 0.0)),
        _magnet(11, MagnetError(-0.05, 0.0, 0.0)),
    ]
    random_placements = (
        Placement(0, 11, "O0", insert_order=0, decision_engine="random_baseline"),
        Placement(1, 10, "O0", insert_order=1, decision_engine="random_baseline"),
    )
    linear_placements = (
        Placement(0, 10, "O90", insert_order=0, decision_engine="linear_sensitivity"),
        Placement(1, 11, "O180", insert_order=1, decision_engine="linear_sensitivity"),
    )
    linear_eval = _evaluation(_metrics(B0_norm=0.20, rms=2.5, J_vector=5.0e-7))
    result = SimulationComparisonResult(
        trial_id=0,
        random_baseline=RandomBaselineResult(
            placements=random_placements,
            evaluation=_evaluation(_metrics(B0_norm=0.10, rms=5.0, J_vector=1.0e-6)),
        ),
        linear=LinearSimulationResult(
            assignment=LinearAssignmentResult(
                placements=linear_placements,
                residual=np.zeros(3, dtype=np.float64),
                linear_score=0.0,
                remaining_slot_flat_ids=(),
            ),
            evaluation=linear_eval,
        ),
        rms_ratio_linear_over_random=0.5,
        j_ratio_linear_over_random=0.5,
    )

    events = timeline_from_simulation_result(slots, magnets, result, result_label="linear")

    assert [event.result_label for event in events] == ["linear", "linear"]
    assert [event.magnet_id for event in events] == [10, 11]
    assert events[-1].B0_norm == pytest.approx(0.20)
    assert events[-1].rms_homogeneity_ppm == pytest.approx(2.5)
    assert events[-1].J_vector == pytest.approx(5.0e-7)


def test_ring_summary_rejects_duplicate_placement_slot() -> None:
    slots = [_slot(0, 0, 0), _slot(1, 0, 1)]
    magnets = [
        _magnet(10, MagnetError(0.0, 0.0, 0.0)),
        _magnet(11, MagnetError(0.0, 0.0, 0.0)),
    ]
    placements = [
        Placement(0, 10, "O0", insert_order=0),
        Placement(0, 11, "O0", insert_order=1),
    ]

    with pytest.raises(ValueError, match="duplicate placed slot"):
        ring_summary_from_placements(slots, magnets, placements)
