from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

from halbach.demag_cellavg import demag_N_cellavg

EPS = 1e-30
FOUR_PI = 4.0 * math.pi


def edges_from_near(
    nbr_idx: NDArray[np.int32], nbr_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    M, deg = nbr_idx.shape
    i_all = np.repeat(np.arange(M, dtype=np.int32), deg)
    j_all = nbr_idx.reshape(-1).astype(np.int32)
    m_all = nbr_mask.reshape(-1).astype(bool)
    i = i_all[m_all]
    j = j_all[m_all]
    return i, j


def u_xy_from_phi(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64], np.stack([np.cos(phi), np.sin(phi)], axis=1).astype(np.float64)
    )


def _subdip_offsets(subdip_n: int, a: float) -> NDArray[np.float64]:
    coords = (np.arange(subdip_n, dtype=np.float64) + 0.5) / subdip_n - 0.5
    coords = coords * float(a)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return cast(NDArray[np.float64], np.stack([xx, yy, zz], axis=-1).reshape(-1, 3))


def build_T2_edges(
    r0_flat: NDArray[np.float64],
    i: NDArray[np.int32],
    j: NDArray[np.int32],
    *,
    near_kernel: str,
    volume_m3: float,
    subdip_n: int = 2,
) -> NDArray[np.float64]:
    near_kernel_norm = "cellavg" if near_kernel == "cube-average" else str(near_kernel)
    s = np.asarray(r0_flat[i] - r0_flat[j], dtype=np.float64)
    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]
    r2 = np.maximum(x * x + y * y + z * z, EPS)
    rmag = np.sqrt(r2)
    inv_r3 = 1.0 / (rmag * r2)
    inv_r5 = inv_r3 / r2

    pref = float(volume_m3) / FOUR_PI

    if near_kernel_norm == "dipole":
        Txx = pref * (3.0 * x * x * inv_r5 - inv_r3)
        Txy = pref * (3.0 * x * y * inv_r5)
        Tyx = Txy
        Tyy = pref * (3.0 * y * y * inv_r5 - inv_r3)
        T2 = np.stack([np.stack([Txx, Txy], axis=-1), np.stack([Tyx, Tyy], axis=-1)], axis=-2)
        return np.asarray(T2, dtype=np.float64)

    if near_kernel_norm == "multi-dipole":
        a = float(volume_m3) ** (1.0 / 3.0)
        offsets = _subdip_offsets(int(subdip_n), a)
        r_sub = s[:, None, :] - offsets[None, :, :]
        x_sub = r_sub[..., 0]
        y_sub = r_sub[..., 1]
        z_sub = r_sub[..., 2]
        r2_sub = np.maximum(x_sub * x_sub + y_sub * y_sub + z_sub * z_sub, EPS)
        rmag_sub = np.sqrt(r2_sub)
        inv_r3_sub = 1.0 / (rmag_sub * r2_sub)
        inv_r5_sub = inv_r3_sub / r2_sub

        Txx = 3.0 * x_sub * x_sub * inv_r5_sub - inv_r3_sub
        Txy = 3.0 * x_sub * y_sub * inv_r5_sub
        Tyx = Txy
        Tyy = 3.0 * y_sub * y_sub * inv_r5_sub - inv_r3_sub

        Txx = pref * np.mean(Txx, axis=1)
        Txy = pref * np.mean(Txy, axis=1)
        Tyx = pref * np.mean(Tyx, axis=1)
        Tyy = pref * np.mean(Tyy, axis=1)
        T2 = np.stack([np.stack([Txx, Txy], axis=-1), np.stack([Tyx, Tyy], axis=-1)], axis=-2)
        return np.asarray(T2, dtype=np.float64)

    if near_kernel_norm == "cellavg":
        a = float(volume_m3) ** (1.0 / 3.0)
        N = demag_N_cellavg(s, (a, a, a))
        return np.asarray(-N[:, 0:2, 0:2], dtype=np.float64)

    raise ValueError(f"Unsupported near_kernel: {near_kernel}")


def build_C_edges_from_phi(
    phi_flat: NDArray[np.float64],
    i: NDArray[np.int32],
    j: NDArray[np.int32],
    T2: NDArray[np.float64],
) -> NDArray[np.float64]:
    u2 = u_xy_from_phi(phi_flat)
    ui = u2[i]
    uj = u2[j]
    tmp = np.einsum("eab,eb->ea", T2, uj)
    C_edge = np.einsum("ea,ea->e", ui, tmp)
    return np.asarray(C_edge, dtype=np.float64)


def build_C_sparse(
    M: int, i: NDArray[np.int32], j: NDArray[np.int32], C_edge: NDArray[np.float64]
) -> csr_matrix:
    C = csr_matrix((C_edge, (i, j)), shape=(M, M), dtype=np.float64)
    C.sum_duplicates()
    return C


def build_A_sparse(C: csr_matrix, chi: float, Nd: float) -> csr_matrix:
    denom = 1.0 + float(chi) * float(Nd)
    I_mat = identity(C.shape[0], format="csr", dtype=np.float64)
    A = denom * I_mat - float(chi) * C
    return cast(csr_matrix, A)


def build_b(p0: float | NDArray[np.float64], M: int) -> NDArray[np.float64]:
    if np.ndim(p0) == 0:
        return np.full(M, float(p0), dtype=np.float64)
    p0_arr = np.asarray(p0, dtype=np.float64)
    if p0_arr.shape != (M,):
        raise ValueError("p0 must be scalar or shape (M,)")
    return p0_arr


def solve_p_linear(A: csr_matrix, b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(spsolve(A, b), dtype=np.float64)


def residual_norm(A: csr_matrix, p: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    r = A.dot(p) - b
    b_norm = float(np.linalg.norm(b))
    return float(np.linalg.norm(r)) / max(b_norm, EPS)


def solve_p_easy_axis_linear_system(
    phi_flat: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    nbr_idx: NDArray[np.int32],
    nbr_mask: NDArray[np.bool_],
    *,
    near_kernel: str,
    volume_m3: float,
    p0: float | NDArray[np.float64],
    chi: float,
    Nd: float,
    subdip_n: int = 2,
) -> tuple[NDArray[np.float64], dict[str, float | int | str]]:
    M = int(phi_flat.shape[0])
    i, j = edges_from_near(nbr_idx, nbr_mask)
    T2 = build_T2_edges(
        r0_flat,
        i,
        j,
        near_kernel=near_kernel,
        volume_m3=float(volume_m3),
        subdip_n=int(subdip_n),
    )
    C_edge = build_C_edges_from_phi(phi_flat, i, j, T2)
    C = build_C_sparse(M, i, j, C_edge)
    A = build_A_sparse(C, float(chi), float(Nd))
    b = build_b(p0, M)
    p = solve_p_linear(A, b)

    C_max_abs = float(np.max(np.abs(C_edge))) if C_edge.size > 0 else 0.0
    C_abs = C.copy()
    C_abs.data = np.abs(C_abs.data)
    row_sum = np.asarray(C_abs.sum(axis=1)).ravel()
    C_row_abs_sum_max = float(np.max(row_sum)) if row_sum.size > 0 else 0.0

    stats: dict[str, float | int | str] = {
        "M": int(M),
        "E": int(C_edge.size),
        "near_kernel": "cellavg" if near_kernel == "cube-average" else str(near_kernel),
        "denom": float(1.0 + float(chi) * float(Nd)),
        "C_max_abs": C_max_abs,
        "C_row_abs_sum_max": C_row_abs_sum_max,
        "residual_norm": residual_norm(A, p, b),
    }
    return p, stats


def export_edges_npz(
    path: str | Path,
    i: NDArray[np.int32],
    j: NDArray[np.int32],
    T2: NDArray[np.float64],
    *,
    meta: dict[str, Any],
) -> None:
    i_arr = np.asarray(i, dtype=np.int32)
    j_arr = np.asarray(j, dtype=np.int32)
    T2_arr = np.asarray(T2, dtype=np.float64)
    meta_json = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, i=i_arr, j=j_arr, T2=T2_arr, meta_json=meta_json)


__all__ = [
    "edges_from_near",
    "u_xy_from_phi",
    "build_T2_edges",
    "build_C_edges_from_phi",
    "build_C_sparse",
    "build_A_sparse",
    "build_b",
    "solve_p_linear",
    "residual_norm",
    "solve_p_easy_axis_linear_system",
    "export_edges_npz",
]
