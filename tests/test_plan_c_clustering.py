import re

import pytest

from halbach.assembly.clustering import assign_quantile_clusters, isolate_outliers
from halbach.assembly.types import MagnetError, VirtualMagnet


def _magnet(
    magnet_id: int,
    *,
    eps: float,
    d1: float,
    d2: float,
    quality: float | None = 1.0,
) -> VirtualMagnet:
    error = MagnetError(epsilon_parallel=eps, delta_perp_1=d1, delta_perp_2=d2)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=error,
        measured_error=error,
        quality=quality,
    )


def test_assign_quantile_clusters_has_stable_cluster_id_format_and_limits() -> None:
    magnets = [
        _magnet(idx, eps=float(idx), d1=float(idx % 3), d2=0.0)
        for idx in range(12)
    ]

    assignments = assign_quantile_clusters(magnets, strength_count=4, angle_count=3)

    assert len(assignments) == len(magnets)
    assert all(item.quarantine_id is None for item in assignments)
    assert all(item.cluster_id is not None for item in assignments)
    assert all(re.fullmatch(r"S\d{2}_A\d{2}", item.cluster_id or "") for item in assignments)
    assert len({item.cluster_id for item in assignments}) <= 4 * 3


def test_assign_quantile_clusters_applies_quarantine_exactly_once() -> None:
    magnets = [_magnet(idx, eps=float(idx), d1=0.0, d2=0.0) for idx in range(5)]
    quarantine = {1: "Q_MEASUREMENT_UNSTABLE", 3: "Q_STRENGTH_OUTLIER"}

    assignments = assign_quantile_clusters(magnets, quarantine=quarantine)

    by_id = {assignment.magnet_id: assignment for assignment in assignments}
    assert by_id[1].cluster_id is None
    assert by_id[1].quarantine_id == "Q_MEASUREMENT_UNSTABLE"
    assert by_id[3].cluster_id is None
    assert by_id[3].quarantine_id == "Q_STRENGTH_OUTLIER"
    assert all(
        (assignment.cluster_id is None) != (assignment.quarantine_id is None)
        for assignment in assignments
    )


def test_transverse_2_weight_changes_angle_bin_ordering() -> None:
    magnets = [
        _magnet(0, eps=0.0, d1=0.00, d2=0.30),
        _magnet(1, eps=0.0, d1=0.10, d2=0.00),
        _magnet(2, eps=0.0, d1=0.20, d2=0.00),
    ]

    no_d2_weight = assign_quantile_clusters(
        magnets,
        strength_count=1,
        angle_count=3,
        transverse_2_weight=0.0,
    )
    high_d2_weight = assign_quantile_clusters(
        magnets,
        strength_count=1,
        angle_count=3,
        transverse_2_weight=10.0,
    )

    bins_without = {item.magnet_id: item.cluster_id for item in no_d2_weight}
    bins_with = {item.magnet_id: item.cluster_id for item in high_d2_weight}
    assert bins_without[0] == "S00_A00"
    assert bins_with[0] == "S00_A02"


def test_isolate_outliers_respects_limit_and_quality_priority() -> None:
    magnets = [
        _magnet(0, eps=0.0, d1=10.0, d2=0.0, quality=1.0),
        _magnet(1, eps=10.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(2, eps=0.0, d1=0.0, d2=0.0, quality=0.2),
        _magnet(3, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(4, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(5, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(6, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(7, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(8, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
        _magnet(9, eps=0.0, d1=0.0, d2=0.0, quality=1.0),
    ]

    quarantine = isolate_outliers(magnets, max_fraction=0.10, quality_threshold=0.90)

    assert quarantine == {2: "Q_MEASUREMENT_UNSTABLE"}
    assert len(quarantine) <= 1


def test_isolate_outliers_uses_direction_before_strength_when_quality_is_ok() -> None:
    magnets = [
        _magnet(0, eps=10.0, d1=0.0, d2=0.0),
        _magnet(1, eps=0.0, d1=5.0, d2=0.0),
        _magnet(2, eps=0.0, d1=0.0, d2=0.0),
        _magnet(3, eps=0.0, d1=0.0, d2=0.0),
        _magnet(4, eps=0.0, d1=0.0, d2=0.0),
        _magnet(5, eps=0.0, d1=0.0, d2=0.0),
        _magnet(6, eps=0.0, d1=0.0, d2=0.0),
        _magnet(7, eps=0.0, d1=0.0, d2=0.0),
        _magnet(8, eps=0.0, d1=0.0, d2=0.0),
        _magnet(9, eps=0.0, d1=0.0, d2=0.0),
    ]

    quarantine = isolate_outliers(magnets, max_fraction=0.10)

    assert quarantine == {1: "Q_DIRECTION_OUTLIER"}


def test_isolate_outliers_falls_back_to_strength_when_no_direction_outlier() -> None:
    magnets = [
        _magnet(idx, eps=float(idx), d1=0.0, d2=0.0)
        for idx in range(10)
    ]

    quarantine = isolate_outliers(magnets, max_fraction=0.10)

    assert quarantine == {9: "Q_STRENGTH_OUTLIER"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength_count": 0},
        {"angle_count": 0},
        {"transverse_2_weight": -1.0},
    ],
)
def test_assign_quantile_clusters_rejects_invalid_counts(kwargs: dict[str, object]) -> None:
    magnets = [_magnet(0, eps=0.0, d1=0.0, d2=0.0)]
    with pytest.raises(ValueError):
        assign_quantile_clusters(magnets, **kwargs)
