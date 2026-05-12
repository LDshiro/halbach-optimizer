from __future__ import annotations

from dataclasses import replace

import numpy as np

from halbach.assembly.types import (
    ClusterAssignment,
    ClusterInventory,
    ClusterStats,
    MagnetError,
    QuarantineReason,
    VirtualMagnet,
)
from halbach.types import FloatArray


def _error_vector(error: MagnetError) -> FloatArray:
    return np.array(
        [
            float(error.epsilon_parallel),
            float(error.delta_perp_1),
            float(error.delta_perp_2),
        ],
        dtype=np.float64,
    )


def _cluster_stats(cluster_id: str, errors: list[MagnetError]) -> ClusterStats:
    if not errors:
        raise ValueError("errors must not be empty")
    rows = np.vstack([_error_vector(error) for error in errors]).astype(np.float64)
    mean = np.asarray(np.mean(rows, axis=0), dtype=np.float64)
    if rows.shape[0] == 1:
        cov = np.zeros((3, 3), dtype=np.float64)
    else:
        cov = np.asarray(np.cov(rows, rowvar=False), dtype=np.float64)
        if cov.shape != (3, 3):
            cov = np.reshape(cov, (3, 3)).astype(np.float64)
    return ClusterStats(
        cluster_id=cluster_id,
        count=int(rows.shape[0]),
        mean=mean,
        cov=cov,
    )


def build_cluster_inventory(
    magnets: list[VirtualMagnet],
    assignments: list[ClusterAssignment],
) -> ClusterInventory:
    """Build cluster inventory from measured errors and cluster/quarantine assignments."""
    magnet_by_id = {magnet.magnet_id: magnet for magnet in magnets}
    if len(magnet_by_id) != len(magnets):
        raise ValueError("magnet_id values must be unique")
    if len(assignments) != len(magnets):
        raise ValueError("assignments length must match magnets length")

    assignment_ids = [assignment.magnet_id for assignment in assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("assignment magnet_id values must be unique")
    unknown = sorted(set(assignment_ids) - set(magnet_by_id))
    missing = sorted(set(magnet_by_id) - set(assignment_ids))
    if unknown or missing:
        raise ValueError(f"assignment coverage mismatch; unknown={unknown}, missing={missing}")

    errors_by_cluster: dict[str, list[MagnetError]] = {}
    quarantine: dict[QuarantineReason, int] = {}
    for assignment in assignments:
        if assignment.cluster_id is None and assignment.quarantine_id is None:
            raise ValueError("assignment must have cluster_id or quarantine_id")
        if assignment.cluster_id is not None and assignment.quarantine_id is not None:
            raise ValueError("assignment cannot have both cluster_id and quarantine_id")

        if assignment.quarantine_id is not None:
            quarantine[assignment.quarantine_id] = quarantine.get(assignment.quarantine_id, 0) + 1
            continue

        assert assignment.cluster_id is not None
        magnet = magnet_by_id[assignment.magnet_id]
        errors_by_cluster.setdefault(assignment.cluster_id, []).append(magnet.measured_error)

    clusters = {
        cluster_id: _cluster_stats(cluster_id, errors)
        for cluster_id, errors in sorted(errors_by_cluster.items())
    }
    return ClusterInventory(clusters=clusters, quarantine=quarantine)


def inventory_total_count(inventory: ClusterInventory) -> int:
    """Return normal inventory count plus quarantine count."""
    return int(
        sum(stats.count for stats in inventory.clusters.values())
        + sum(inventory.quarantine.values())
    )


def decrement_cluster(inventory: ClusterInventory, cluster_id: str) -> ClusterInventory:
    """Return a copy of inventory with one item consumed from a normal cluster."""
    if cluster_id not in inventory.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    stats = inventory.clusters[cluster_id]
    if stats.count <= 0:
        raise ValueError(f"cluster {cluster_id} has no remaining inventory")
    clusters = dict(inventory.clusters)
    clusters[cluster_id] = replace(stats, count=stats.count - 1)
    return ClusterInventory(clusters=clusters, quarantine=dict(inventory.quarantine))


def increment_cluster(inventory: ClusterInventory, cluster_id: str) -> ClusterInventory:
    """Return a copy of inventory with one item restored to a normal cluster."""
    if cluster_id not in inventory.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    stats = inventory.clusters[cluster_id]
    clusters = dict(inventory.clusters)
    clusters[cluster_id] = replace(stats, count=stats.count + 1)
    return ClusterInventory(clusters=clusters, quarantine=dict(inventory.quarantine))


__all__ = [
    "build_cluster_inventory",
    "decrement_cluster",
    "increment_cluster",
    "inventory_total_count",
]
