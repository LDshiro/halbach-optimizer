from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]


@dataclass(frozen=True)
class MagnetExportData:
    """
    Flattened magnet geometry export for downstream CAD/CAM consumers.

    centers_m: shape (M, 3), units m
    phi_rad: shape (M,), units rad
    beta_rad: shape (M,), units rad
    u: shape (M, 3), unitless
    ring_id: shape (M,), ring index per magnet
    layer_id: shape (M,), layer index per magnet
    theta_id: shape (M,), theta-sample index per magnet
    """

    centers_m: FloatArray
    phi_rad: FloatArray
    beta_rad: FloatArray
    u: FloatArray
    ring_id: IntArray
    layer_id: IntArray
    theta_id: IntArray


def equivalent_cube_dimensions_from_volume_mm3(
    volume_mm3: float,
) -> tuple[FloatArray, FloatArray]:
    """
    Convert scalar volume to equivalent cube dimensions.

    Returns:
        dims_m: shape (3,), units m
        dims_mm: shape (3,), units mm
    """
    volume_mm3_f = float(volume_mm3)
    if volume_mm3_f <= 0.0:
        raise ValueError("volume_mm3 must be > 0")
    edge_mm = volume_mm3_f ** (1.0 / 3.0)
    dims_mm = np.full((3,), edge_mm, dtype=np.float64)
    dims_m = dims_mm * 1e-3
    return dims_m, dims_mm


def build_magnet_export_data(
    phi_rkn: FloatArray,
    r0_rkn: FloatArray,
    *,
    ring_active_mask: NDArray[np.bool_] | None = None,
    beta_rk: FloatArray | None = None,
) -> MagnetExportData:
    """
    Flatten active magnets into per-magnet center/orientation records.

    phi_rkn: shape (R, K, N), units rad
    r0_rkn: shape (R, K, N, 3), units m
    ring_active_mask: shape (R, K), active ring/layer mask
    beta_rk: shape (R, K), units rad
    """
    phi_arr = np.asarray(phi_rkn, dtype=np.float64)
    r0_arr = np.asarray(r0_rkn, dtype=np.float64)
    if phi_arr.ndim != 3:
        raise ValueError(f"phi_rkn must have shape (R, K, N), got {phi_arr.shape}")
    if r0_arr.shape != phi_arr.shape + (3,):
        raise ValueError(f"r0_rkn must have shape {(phi_arr.shape + (3,))}, got {r0_arr.shape}")

    R, K, N = phi_arr.shape
    if ring_active_mask is None:
        active_rk = np.ones((R, K), dtype=np.bool_)
    else:
        active_rk = np.asarray(ring_active_mask, dtype=np.bool_)
        if active_rk.shape != (R, K):
            raise ValueError(f"ring_active_mask must have shape {(R, K)}, got {active_rk.shape}")

    if beta_rk is None:
        beta_arr = np.zeros((R, K), dtype=np.float64)
    else:
        beta_arr = np.asarray(beta_rk, dtype=np.float64)
        if beta_arr.shape != (R, K):
            raise ValueError(f"beta_rk must have shape {(R, K)}, got {beta_arr.shape}")

    active_rkn = np.broadcast_to(active_rk[:, :, None], (R, K, N))
    centers = r0_arr[active_rkn].reshape(-1, 3)
    phi_flat = phi_arr[active_rkn].reshape(-1)
    beta_rkn = np.broadcast_to(beta_arr[:, :, None], (R, K, N))
    beta_flat = beta_rkn[active_rkn].reshape(-1)

    ring_id_grid = np.broadcast_to(np.arange(R, dtype=np.int32)[:, None, None], (R, K, N))
    layer_id_grid = np.broadcast_to(np.arange(K, dtype=np.int32)[None, :, None], (R, K, N))
    theta_id_grid = np.broadcast_to(np.arange(N, dtype=np.int32)[None, None, :], (R, K, N))

    cos_beta = np.cos(beta_flat)
    u = np.column_stack(
        [
            cos_beta * np.cos(phi_flat),
            cos_beta * np.sin(phi_flat),
            np.sin(beta_flat),
        ]
    ).astype(np.float64)

    return MagnetExportData(
        centers_m=np.asarray(centers, dtype=np.float64),
        phi_rad=np.asarray(phi_flat, dtype=np.float64),
        beta_rad=np.asarray(beta_flat, dtype=np.float64),
        u=u,
        ring_id=np.asarray(ring_id_grid[active_rkn], dtype=np.int32),
        layer_id=np.asarray(layer_id_grid[active_rkn], dtype=np.int32),
        theta_id=np.asarray(theta_id_grid[active_rkn], dtype=np.int32),
    )


def build_magnet_export_payload(
    phi_rkn: FloatArray,
    r0_rkn: FloatArray,
    *,
    ring_active_mask: NDArray[np.bool_] | None = None,
    beta_rk: FloatArray | None = None,
    dimensions_m: FloatArray | None = None,
    dimensions_mm: FloatArray | None = None,
) -> dict[str, NDArray[Any]]:
    """
    Build results.npz payload entries for per-magnet export.
    """
    export = build_magnet_export_data(
        phi_rkn,
        r0_rkn,
        ring_active_mask=ring_active_mask,
        beta_rk=beta_rk,
    )
    payload: dict[str, NDArray[Any]] = {
        "magnet_centers_m": export.centers_m,
        "magnet_phi_rad": export.phi_rad,
        "magnet_beta_rad": export.beta_rad,
        "magnet_u": export.u,
        "magnet_ring_id": export.ring_id,
        "magnet_layer_id": export.layer_id,
        "magnet_theta_id": export.theta_id,
    }
    if dimensions_m is not None:
        dims_m = np.asarray(dimensions_m, dtype=np.float64).reshape(-1)
        if dims_m.shape != (3,):
            raise ValueError(f"dimensions_m must have shape (3,), got {dims_m.shape}")
        payload["magnet_dimensions_m"] = dims_m
    if dimensions_mm is not None:
        dims_mm = np.asarray(dimensions_mm, dtype=np.float64).reshape(-1)
        if dims_mm.shape != (3,):
            raise ValueError(f"dimensions_mm must have shape (3,), got {dims_mm.shape}")
        payload["magnet_dimensions_mm"] = dims_mm
    return payload


__all__ = [
    "MagnetExportData",
    "build_magnet_export_data",
    "build_magnet_export_payload",
    "equivalent_cube_dimensions_from_volume_mm3",
]
