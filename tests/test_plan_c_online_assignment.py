import numpy as np
import pytest

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory, inventory_total_count
from halbach.assembly.online_assignment import (
    choose_best_linear_candidate,
    cluster_usage_from_placements,
    planned_cluster_counts,
    run_linear_sensitivity_assignment,
)
from halbach.assembly.types import MagnetError, SensitivityTable, VirtualMagnet


def _error(eps: float = 0.0, d1: float = 0.0, d2: float = 0.0) -> MagnetError:
    return MagnetError(
        epsilon_parallel=eps,
        delta_perp_1=d1,
        delta_perp_2=d2,
    )


def _magnet(magnet_id: int, eps: float) -> VirtualMagnet:
    error = _error(eps)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=error,
        measured_error=error,
        quality=1.0,
    )


def _table() -> SensitivityTable:
    C = np.zeros((2, 4, 2, 3), dtype=np.float64)
    C[0, :, :, 0] = np.array([1.0, 0.0], dtype=np.float64)
    C[1, :, :, 0] = np.array([-1.0, 0.0], dtype=np.float64)
    return SensitivityTable(
        slot_flat_id=np.array([10, 20], dtype=np.int_),
        ring_id=np.array([0, 0], dtype=np.int_),
        layer_id=np.array([0, 1], dtype=np.int_),
        theta_id=np.array([0, 0], dtype=np.int_),
        centers_m=np.zeros((2, 3), dtype=np.float64),
        nominal_u=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (2, 1)),
        orientation_id=("O0", "O90", "O180", "O270"),
        C=C,
        roi_points=np.zeros((1, 3), dtype=np.float64),
        normalization_b0=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata={"case": "online-assignment-test"},
    )


def test_choose_best_candidate_minimizes_residual_score() -> None:
    table = _table()
    candidate = choose_best_linear_candidate(
        table,
        np.array([1.0, 0.0], dtype=np.float64),
        [10, 20],
        _error(eps=1.0),
    )

    assert candidate.slot_flat_id == 20
    assert candidate.orientation_id == "O0"
    assert candidate.score == pytest.approx(0.0)
    np.testing.assert_allclose(candidate.contribution, [-1.0, 0.0])


def test_orientation_is_selected_from_allowed_candidates() -> None:
    table = _table()
    table.C[0, 1, :, 0] = np.array([-1.0, 0.0], dtype=np.float64)
    candidate = choose_best_linear_candidate(
        table,
        np.array([1.0, 0.0], dtype=np.float64),
        [10],
        _error(eps=1.0),
    )

    assert candidate.orientation_id == "O90"
    assert candidate.orientation_id in table.orientation_id


def test_linear_assignment_updates_residual_and_uses_each_slot_once() -> None:
    table = _table()
    magnets = [_magnet(0, 1.0), _magnet(1, 1.0)]

    result = run_linear_sensitivity_assignment(table, magnets)

    assert [placement.slot_flat_id for placement in result.placements] == [10, 20]
    assert len({placement.slot_flat_id for placement in result.placements}) == 2
    assert result.remaining_slot_flat_ids == ()
    np.testing.assert_allclose(result.residual, np.zeros(2, dtype=np.float64))
    assert result.linear_score == pytest.approx(0.0)
    assert {placement.orientation_id for placement in result.placements} <= set(table.orientation_id)


def test_inventory_counts_are_decremented_and_never_negative() -> None:
    table = _table()
    magnets = [_magnet(0, 1.0), _magnet(1, 1.0)]
    assignments = assign_quantile_clusters(magnets, strength_count=1, angle_count=1)
    inventory = build_cluster_inventory(magnets, assignments)

    result = run_linear_sensitivity_assignment(
        table,
        magnets,
        assignments=assignments,
        inventory=inventory,
    )

    assert inventory_total_count(result.inventory) == 0
    assert cluster_usage_from_placements(result.placements) == planned_cluster_counts(assignments)

    underfilled = build_cluster_inventory(magnets[:1], assignments[:1])
    with pytest.raises(ValueError, match="remaining inventory"):
        run_linear_sensitivity_assignment(
            table,
            magnets,
            assignments=assignments,
            inventory=underfilled,
        )
