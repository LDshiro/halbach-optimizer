from __future__ import annotations

import numpy as np

from halbach.angles_runtime import phi_rkn_from_run, u_rkn_from_run
from halbach.assembly.types import AssemblySlot
from halbach.radial_profile import radial_profile_from_run
from halbach.run_types import RunBundle


def flatten_slot_id(r: int, k: int, n: int, K: int, N: int) -> int:
    """Flatten slot index with C-order `(r, k, n)` layout."""
    if K <= 0 or N <= 0:
        raise ValueError("K and N must be positive")
    if r < 0 or k < 0 or n < 0:
        raise ValueError("slot indices must be non-negative")
    if k >= K or n >= N:
        raise ValueError("slot index out of range")
    return (int(r) * int(K) + int(k)) * int(N) + int(n)


def _mirror_pair_id(layer_id: int, K: int) -> str | None:
    mate = K - 1 - layer_id
    if mate == layer_id:
        return None
    low = min(layer_id, mate)
    high = max(layer_id, mate)
    return f"P{low:03d}_{high:03d}"


def build_assembly_slots(run: RunBundle) -> list[AssemblySlot]:
    """
    Enumerate active Plan C assembly slots from a standard Halbach run.

    Returned slots preserve dense `(r, k, n)` traversal order while skipping
    inactive ring/layer pairs. Coordinates are in m and angles in rad.
    """
    geom = run.geometry
    R = int(geom.R)
    K = int(geom.K)
    N = int(geom.N)

    radial_profile = radial_profile_from_run(run)
    active_mask = np.asarray(radial_profile.ring_active_mask, dtype=np.bool_)
    if active_mask.shape != (R, K):
        raise ValueError(f"ring_active_mask shape {active_mask.shape} does not match {(R, K)}")

    phi_rkn = np.asarray(phi_rkn_from_run(run), dtype=np.float64)
    u_rkn = np.asarray(u_rkn_from_run(run), dtype=np.float64)
    if phi_rkn.shape != (R, K, N):
        raise ValueError(f"phi_rkn shape {phi_rkn.shape} does not match {(R, K, N)}")
    if u_rkn.shape != (R, K, N, 3):
        raise ValueError(f"u_rkn shape {u_rkn.shape} does not match {(R, K, N, 3)}")

    r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
    ring_offsets = np.asarray(geom.ring_offsets, dtype=np.float64)
    cth = np.asarray(geom.cth, dtype=np.float64)
    sth = np.asarray(geom.sth, dtype=np.float64)
    z_layers = np.asarray(geom.z_layers, dtype=np.float64)

    slots: list[AssemblySlot] = []
    for r in range(R):
        for k in range(K):
            if not bool(active_mask[r, k]):
                continue
            rho = float(r_bases[k] + ring_offsets[r])
            mirror_pair_id = _mirror_pair_id(k, K)
            for n in range(N):
                center = np.array(
                    [
                        rho * float(cth[n]),
                        rho * float(sth[n]),
                        float(z_layers[k]),
                    ],
                    dtype=np.float64,
                )
                slots.append(
                    AssemblySlot(
                        slot_flat_id=flatten_slot_id(r, k, n, K, N),
                        ring_id=r,
                        layer_id=k,
                        theta_id=n,
                        center_m=center,
                        nominal_u=np.array(u_rkn[r, k, n], dtype=np.float64, copy=True),
                        nominal_phi_rad=float(phi_rkn[r, k, n]),
                        physical_slot_number=n + 1,
                        mirror_pair_id=mirror_pair_id,
                    )
                )
    return slots


__all__ = ["build_assembly_slots", "flatten_slot_id"]
