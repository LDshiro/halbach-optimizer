import numpy as np
import pytest

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import (
    build_cluster_inventory,
    decrement_cluster,
    increment_cluster,
    inventory_total_count,
)
from halbach.assembly.types import ClusterAssignment, MagnetError, VirtualMagnet


def _magnet(magnet_id: int, eps: float, d1: float, d2: float) -> VirtualMagnet:
    error = MagnetError(epsilon_parallel=eps, delta_perp_1=d1, delta_perp_2=d2)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=error,
        measured_error=error,
        quality=1.0,
    )


def test_build_cluster_inventory_counts_stats_and_quarantine() -> None:
    magnets = [
        _magnet(0, 0.0, 0.0, 0.0),
        _magnet(1, 1.0, 0.0, 0.0),
        _magnet(2, 0.0, 2.0, 0.0),
        _magnet(3, 0.0, 0.0, 3.0),
    ]
    assignments = [
        ClusterAssignment(magnet_id=0, cluster_id="S00_A00"),
        ClusterAssignment(magnet_id=1, cluster_id="S00_A00"),
        ClusterAssignment(magnet_id=2, cluster_id="S01_A00"),
        ClusterAssignment(magnet_id=3, cluster_id=None, quarantine_id="Q_DIRECTION_OUTLIER"),
    ]

    inventory = build_cluster_inventory(magnets, assignments)

    assert inventory_total_count(inventory) == 4
    assert inventory.quarantine == {"Q_DIRECTION_OUTLIER": 1}
    assert set(inventory.clusters) == {"S00_A00", "S01_A00"}
    stats = inventory.clusters["S00_A00"]
    assert stats.count == 2
    assert stats.mean.shape == (3,)
    assert stats.cov.shape == (3, 3)
    np.testing.assert_allclose(stats.mean, [0.5, 0.0, 0.0])

    singleton = inventory.clusters["S01_A00"]
    np.testing.assert_allclose(singleton.cov, np.zeros((3, 3)))


def test_inventory_decrement_and_increment_return_updated_copies() -> None:
    magnets = [_magnet(idx, float(idx), 0.0, 0.0) for idx in range(3)]
    assignments = assign_quantile_clusters(magnets, strength_count=1, angle_count=1)
    inventory = build_cluster_inventory(magnets, assignments)

    decremented = decrement_cluster(inventory, "S00_A00")
    assert inventory.clusters["S00_A00"].count == 3
    assert decremented.clusters["S00_A00"].count == 2

    incremented = increment_cluster(decremented, "S00_A00")
    assert incremented.clusters["S00_A00"].count == 3


def test_decrement_rejects_empty_or_unknown_cluster() -> None:
    magnets = [_magnet(0, 0.0, 0.0, 0.0)]
    assignments = assign_quantile_clusters(magnets, strength_count=1, angle_count=1)
    inventory = build_cluster_inventory(magnets, assignments)

    empty = decrement_cluster(inventory, "S00_A00")
    assert empty.clusters["S00_A00"].count == 0
    with pytest.raises(ValueError):
        decrement_cluster(empty, "S00_A00")
    with pytest.raises(KeyError):
        decrement_cluster(inventory, "S99_A99")


def test_build_cluster_inventory_rejects_bad_assignment_coverage() -> None:
    magnets = [_magnet(0, 0.0, 0.0, 0.0)]
    with pytest.raises(ValueError):
        build_cluster_inventory(magnets, [])
    with pytest.raises(ValueError):
        build_cluster_inventory(
            magnets,
            [ClusterAssignment(magnet_id=1, cluster_id="S00_A00")],
        )
