from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from halbach.assembly.types import (
    ClusterAssignment,
    ClusterBinningMode,
    QuarantineReason,
    VirtualMagnet,
)


def _validate_magnets(magnets: list[VirtualMagnet]) -> None:
    if not magnets:
        raise ValueError("magnets must not be empty")
    ids = [magnet.magnet_id for magnet in magnets]
    if len(ids) != len(set(ids)):
        raise ValueError("magnet_id values must be unique")


def _angle_score(magnet: VirtualMagnet, transverse_2_weight: float) -> float:
    error = magnet.measured_error
    return math.sqrt(
        error.delta_perp_1 * error.delta_perp_1
        + transverse_2_weight * error.delta_perp_2 * error.delta_perp_2
    )


def _quality_sort_value(magnet: VirtualMagnet) -> float:
    if magnet.quality is None:
        raise ValueError("quality is missing for unstable magnet sorting")
    return float(magnet.quality)


def _rank_bins(values: list[tuple[int, float]], count: int) -> dict[int, int]:
    if count <= 0:
        raise ValueError("bin count must be positive")
    n = len(values)
    if n == 0:
        return {}
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    bins: dict[int, int] = {}
    for rank, (magnet_id, _value) in enumerate(ordered):
        bins[magnet_id] = int((rank * count) // n)
    return bins


def _validate_cluster_inputs(
    magnets: list[VirtualMagnet],
    *,
    strength_count: int,
    angle_count: int,
    transverse_2_weight: float,
    quarantine: Mapping[int, QuarantineReason] | None,
) -> tuple[dict[int, QuarantineReason], list[VirtualMagnet]]:
    _validate_magnets(magnets)
    if strength_count <= 0:
        raise ValueError("strength_count must be positive")
    if angle_count <= 0:
        raise ValueError("angle_count must be positive")
    if not math.isfinite(transverse_2_weight) or transverse_2_weight < 0.0:
        raise ValueError("transverse_2_weight must be finite and >= 0")

    quarantine_map = dict(quarantine or {})
    magnet_ids = {magnet.magnet_id for magnet in magnets}
    unknown_quarantine_ids = sorted(set(quarantine_map) - magnet_ids)
    if unknown_quarantine_ids:
        raise ValueError(f"quarantine contains unknown magnet ids: {unknown_quarantine_ids}")
    return quarantine_map, [magnet for magnet in magnets if magnet.magnet_id not in quarantine_map]


def _cluster_assignments_from_bins(
    magnets: list[VirtualMagnet],
    *,
    strength_bins: Mapping[int, int],
    angle_bins: Mapping[int, int],
    quarantine_map: Mapping[int, QuarantineReason],
) -> list[ClusterAssignment]:
    assignments: list[ClusterAssignment] = []
    for magnet in magnets:
        quarantine_id = quarantine_map.get(magnet.magnet_id)
        if quarantine_id is not None:
            assignments.append(
                ClusterAssignment(
                    magnet_id=magnet.magnet_id,
                    cluster_id=None,
                    quarantine_id=quarantine_id,
                )
            )
            continue

        strength_bin = strength_bins[magnet.magnet_id]
        angle_bin = angle_bins[magnet.magnet_id]
        assignments.append(
            ClusterAssignment(
                magnet_id=magnet.magnet_id,
                cluster_id=f"S{strength_bin:02d}_A{angle_bin:02d}",
                quarantine_id=None,
            )
        )
    return assignments


def assign_quantile_clusters(
    magnets: list[VirtualMagnet],
    *,
    strength_count: int = 10,
    angle_count: int = 3,
    transverse_2_weight: float = 1.0,
    quarantine: Mapping[int, QuarantineReason] | None = None,
) -> list[ClusterAssignment]:
    """Assign measured magnets to strength/angle quantile clusters."""
    quarantine_map, active = _validate_cluster_inputs(
        magnets,
        strength_count=strength_count,
        angle_count=angle_count,
        transverse_2_weight=transverse_2_weight,
        quarantine=quarantine,
    )
    strength_bins = _rank_bins(
        [(magnet.magnet_id, magnet.measured_error.epsilon_parallel) for magnet in active],
        strength_count,
    )
    angle_bins = _rank_bins(
        [(magnet.magnet_id, _angle_score(magnet, transverse_2_weight)) for magnet in active],
        angle_count,
    )
    return _cluster_assignments_from_bins(
        magnets,
        strength_bins=strength_bins,
        angle_bins=angle_bins,
        quarantine_map=quarantine_map,
    )


def _sigma_strength_bin(z_score: float, count: int, step: float) -> int:
    if count == 1:
        return 0
    lower_edge = -0.5 * float(step) * float(count - 2)
    if z_score < lower_edge:
        return 0
    return max(0, min(count - 1, int(math.floor((z_score - lower_edge) / step)) + 1))


def _sigma_angle_bin(normalized_score: float, count: int, step: float) -> int:
    if count == 1:
        return 0
    return max(0, min(count - 1, int(math.floor(normalized_score / step))))


def assign_sigma_band_clusters(
    magnets: list[VirtualMagnet],
    *,
    strength_count: int = 10,
    angle_count: int = 5,
    transverse_2_weight: float = 1.0,
    strength_sigma_step: float = 0.5,
    angle_sigma_step: float = 0.5,
    quarantine: Mapping[int, QuarantineReason] | None = None,
) -> list[ClusterAssignment]:
    """Assign measured magnets to fixed-width sigma bands."""
    if not math.isfinite(strength_sigma_step) or strength_sigma_step <= 0.0:
        raise ValueError("strength_sigma_step must be finite and > 0")
    if not math.isfinite(angle_sigma_step) or angle_sigma_step <= 0.0:
        raise ValueError("angle_sigma_step must be finite and > 0")
    quarantine_map, active = _validate_cluster_inputs(
        magnets,
        strength_count=strength_count,
        angle_count=angle_count,
        transverse_2_weight=transverse_2_weight,
        quarantine=quarantine,
    )
    if not active:
        return _cluster_assignments_from_bins(
            magnets,
            strength_bins={},
            angle_bins={},
            quarantine_map=quarantine_map,
        )

    eps = np.asarray(
        [magnet.measured_error.epsilon_parallel for magnet in active], dtype=np.float64
    )
    eps_mean = float(np.mean(eps))
    eps_sigma = float(np.std(eps))
    angle_scores = np.asarray(
        [_angle_score(magnet, transverse_2_weight) for magnet in active], dtype=np.float64
    )
    angle_sigma = float(
        math.sqrt(
            np.mean(
                [
                    magnet.measured_error.delta_perp_1 * magnet.measured_error.delta_perp_1
                    + transverse_2_weight
                    * magnet.measured_error.delta_perp_2
                    * magnet.measured_error.delta_perp_2
                    for magnet in active
                ]
            )
        )
    )

    central_strength_bin = strength_count // 2
    strength_bins: dict[int, int] = {}
    angle_bins: dict[int, int] = {}
    for magnet, angle_score in zip(active, angle_scores, strict=True):
        if eps_sigma <= 0.0:
            strength_bins[magnet.magnet_id] = central_strength_bin
        else:
            z_score = (float(magnet.measured_error.epsilon_parallel) - eps_mean) / eps_sigma
            strength_bins[magnet.magnet_id] = _sigma_strength_bin(
                z_score, strength_count, strength_sigma_step
            )
        if angle_sigma <= 0.0:
            angle_bins[magnet.magnet_id] = 0
        else:
            angle_bins[magnet.magnet_id] = _sigma_angle_bin(
                float(angle_score) / angle_sigma, angle_count, angle_sigma_step
            )
    return _cluster_assignments_from_bins(
        magnets,
        strength_bins=strength_bins,
        angle_bins=angle_bins,
        quarantine_map=quarantine_map,
    )


def assign_clusters(
    magnets: list[VirtualMagnet],
    *,
    binning: ClusterBinningMode = "quantile",
    strength_count: int = 10,
    angle_count: int = 3,
    transverse_2_weight: float = 1.0,
    strength_sigma_step: float = 0.5,
    angle_sigma_step: float = 0.5,
    quarantine: Mapping[int, QuarantineReason] | None = None,
) -> list[ClusterAssignment]:
    """Assign clusters using the selected binning strategy."""
    if binning == "quantile":
        return assign_quantile_clusters(
            magnets,
            strength_count=strength_count,
            angle_count=angle_count,
            transverse_2_weight=transverse_2_weight,
            quarantine=quarantine,
        )
    if binning == "sigma_band":
        return assign_sigma_band_clusters(
            magnets,
            strength_count=strength_count,
            angle_count=angle_count,
            transverse_2_weight=transverse_2_weight,
            strength_sigma_step=strength_sigma_step,
            angle_sigma_step=angle_sigma_step,
            quarantine=quarantine,
        )
    raise ValueError(f"unsupported cluster binning mode: {binning}")


def isolate_outliers(
    magnets: list[VirtualMagnet],
    *,
    max_fraction: float = 0.10,
    quality_threshold: float = 0.90,
    direction_weight: float = 1.0,
) -> dict[int, QuarantineReason]:
    """Select up to max_fraction magnets for quarantine using Plan C priority rules."""
    _validate_magnets(magnets)
    if not math.isfinite(max_fraction) or max_fraction < 0.0 or max_fraction > 1.0:
        raise ValueError("max_fraction must be in [0, 1]")
    if not math.isfinite(quality_threshold):
        raise ValueError("quality_threshold must be finite")
    if not math.isfinite(direction_weight) or direction_weight < 0.0:
        raise ValueError("direction_weight must be finite and >= 0")

    limit = int(math.floor(len(magnets) * max_fraction))
    if limit <= 0:
        return {}

    selected: dict[int, QuarantineReason] = {}

    unstable = [
        magnet
        for magnet in magnets
        if magnet.quality is not None and float(magnet.quality) < quality_threshold
    ]
    unstable.sort(key=lambda magnet: (_quality_sort_value(magnet), magnet.magnet_id))
    for magnet in unstable[:limit]:
        selected[magnet.magnet_id] = "Q_MEASUREMENT_UNSTABLE"
    if len(selected) >= limit:
        return selected

    remaining = [magnet for magnet in magnets if magnet.magnet_id not in selected]
    direction_candidates = [
        (magnet, _angle_score(magnet, direction_weight)) for magnet in remaining
    ]
    direction_candidates = [
        (magnet, score) for magnet, score in direction_candidates if score > 0.0
    ]
    direction_candidates.sort(key=lambda item: (-item[1], item[0].magnet_id))
    for magnet, _score in direction_candidates:
        if len(selected) >= limit:
            return selected
        selected[magnet.magnet_id] = "Q_DIRECTION_OUTLIER"

    remaining = [magnet for magnet in magnets if magnet.magnet_id not in selected]
    strength_candidates = [
        (magnet, abs(float(magnet.measured_error.epsilon_parallel))) for magnet in remaining
    ]
    strength_candidates = [(magnet, score) for magnet, score in strength_candidates if score > 0.0]
    strength_candidates.sort(key=lambda item: (-item[1], item[0].magnet_id))
    for magnet, _score in strength_candidates:
        if len(selected) >= limit:
            return selected
        selected[magnet.magnet_id] = "Q_STRENGTH_OUTLIER"

    return selected


__all__ = [
    "assign_clusters",
    "assign_quantile_clusters",
    "assign_sigma_band_clusters",
    "isolate_outliers",
]
