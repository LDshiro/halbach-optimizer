from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from halbach.numba_compat import njit

EPS = 1e-30


@njit(cache=True)
def compute_B_and_B0(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    theta: NDArray[np.float64],
    sin2: NDArray[np.float64],
    cth: NDArray[np.float64],
    sth: NDArray[np.float64],
    z_layers: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    phi0: float,
    m0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, float]:
    """
    Compute B at ROI points and B0 at the origin in a single pass.
    """
    Rloc = alphas.shape[0]
    Kloc = alphas.shape[1]
    Nloc = theta.shape[0]
    M = pts.shape[0]
    Bx = np.zeros(M, dtype=np.float64)
    By = np.zeros(M, dtype=np.float64)
    Bz = np.zeros(M, dtype=np.float64)
    B0x = 0.0
    B0y = 0.0
    B0z = 0.0

    for k in range(Kloc):
        z0 = z_layers[k]
        rb = r_bases[k]
        for r in range(Rloc):
            ak = alphas[r, k]
            rho = rb + ring_offsets[r]
            for i in range(Nloc):
                th = theta[i]
                c = cth[i]
                s = sth[i]
                s2 = sin2[i]
                phi = 2.0 * th + phi0 + ak * s2
                mx = m0 * math.cos(phi)
                my = m0 * math.sin(phi)
                px = rho * c
                py = rho * s
                for p in range(M):
                    dx = pts[p, 0] - px
                    dy = pts[p, 1] - py
                    dz = pts[p, 2] - z0
                    r2 = dx * dx + dy * dy + dz * dz
                    rmag = math.sqrt(r2) + 1e-30
                    invr3 = 1.0 / (rmag * r2 + 1e-30)
                    rx = dx / rmag
                    ry = dy / rmag
                    rz = dz / rmag
                    mdotr = mx * rx + my * ry
                    Bx[p] += factor * (3.0 * mdotr * rx - mx) * invr3
                    By[p] += factor * (3.0 * mdotr * ry - my) * invr3
                    Bz[p] += factor * (3.0 * mdotr * rz - 0.0) * invr3
                dx0 = -px
                dy0 = -py
                dz0 = -z0
                r20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0
                r0mag = math.sqrt(r20) + 1e-30
                invr30 = 1.0 / (r0mag * r20 + 1e-30)
                rx0 = dx0 / r0mag
                ry0 = dy0 / r0mag
                rz0 = dz0 / r0mag
                mdotr0 = mx * rx0 + my * ry0
                B0x += factor * (3.0 * mdotr0 * rx0 - mx) * invr30
                B0y += factor * (3.0 * mdotr0 * ry0 - my) * invr30
                B0z += factor * (3.0 * mdotr0 * rz0 - 0.0) * invr30
    return Bx, By, Bz, B0x, B0y, B0z


@njit(cache=True)
def compute_B_and_B0_phi_rkn(
    phi_rkn: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    cth: NDArray[np.float64],
    sth: NDArray[np.float64],
    z_layers: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    m0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, float]:
    """
    Compute B at ROI points and B0 at the origin using explicit phi_rkn.
    """
    Rloc = phi_rkn.shape[0]
    Kloc = phi_rkn.shape[1]
    Nloc = phi_rkn.shape[2]
    M = pts.shape[0]
    Bx = np.zeros(M, dtype=np.float64)
    By = np.zeros(M, dtype=np.float64)
    Bz = np.zeros(M, dtype=np.float64)
    B0x = 0.0
    B0y = 0.0
    B0z = 0.0

    for k in range(Kloc):
        z0 = z_layers[k]
        rb = r_bases[k]
        for r in range(Rloc):
            rho = rb + ring_offsets[r]
            for i in range(Nloc):
                phi = phi_rkn[r, k, i]
                mx = m0 * math.cos(phi)
                my = m0 * math.sin(phi)
                px = rho * cth[i]
                py = rho * sth[i]
                for p in range(M):
                    dx = pts[p, 0] - px
                    dy = pts[p, 1] - py
                    dz = pts[p, 2] - z0
                    r2 = dx * dx + dy * dy + dz * dz
                    rmag = math.sqrt(r2) + 1e-30
                    invr3 = 1.0 / (rmag * r2 + 1e-30)
                    rx = dx / rmag
                    ry = dy / rmag
                    rz = dz / rmag
                    mdotr = mx * rx + my * ry
                    Bx[p] += factor * (3.0 * mdotr * rx - mx) * invr3
                    By[p] += factor * (3.0 * mdotr * ry - my) * invr3
                    Bz[p] += factor * (3.0 * mdotr * rz - 0.0) * invr3
                dx0 = -px
                dy0 = -py
                dz0 = -z0
                r20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0
                r0mag = math.sqrt(r20) + 1e-30
                invr30 = 1.0 / (r0mag * r20 + 1e-30)
                rx0 = dx0 / r0mag
                ry0 = dy0 / r0mag
                rz0 = dz0 / r0mag
                mdotr0 = mx * rx0 + my * ry0
                B0x += factor * (3.0 * mdotr0 * rx0 - mx) * invr30
                B0y += factor * (3.0 * mdotr0 * ry0 - my) * invr30
                B0z += factor * (3.0 * mdotr0 * rz0 - 0.0) * invr30
    return Bx, By, Bz, B0x, B0y, B0z


@njit(cache=True)
def objective_only_phi_rkn(
    phi_rkn: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    cth: NDArray[np.float64],
    sth: NDArray[np.float64],
    z_layers: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    m0: float,
) -> float:
    """
    Objective J(x) = mean_p ||B(p) - B0||^2 using explicit phi_rkn.
    """
    Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0_phi_rkn(
        phi_rkn,
        r_bases,
        cth,
        sth,
        z_layers,
        ring_offsets,
        pts,
        factor,
        m0,
    )
    DBx = Bx - B0x
    DBy = By - B0y
    DBz = Bz - B0z
    return float(np.mean(DBx * DBx + DBy * DBy + DBz * DBz))


@njit(cache=True)
def objective_only(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    theta: NDArray[np.float64],
    sin2: NDArray[np.float64],
    cth: NDArray[np.float64],
    sth: NDArray[np.float64],
    z_layers: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    phi0: float,
    m0: float,
) -> float:
    """
    Objective J(x) = mean_p ||B(p) - B0||^2.
    """
    Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0(
        alphas, r_bases, theta, sin2, cth, sth, z_layers, ring_offsets, pts, factor, phi0, m0
    )
    DBx = Bx - B0x
    DBy = By - B0y
    DBz = Bz - B0z
    return float(np.mean(DBx * DBx + DBy * DBy + DBz * DBz))


@njit(cache=True, parallel=True)
def compute_B_all_from_m_flat(
    pts: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    m_flat: NDArray[np.float64],
    factor: float,
) -> NDArray[np.float64]:
    """
    Compute B at arbitrary points for per-magnet m vectors.
    """
    P = pts.shape[0]
    M = r0_flat.shape[0]
    out = np.zeros((P, 3), dtype=np.float64)
    for p in range(P):
        Bx = 0.0
        By = 0.0
        Bz = 0.0
        px = pts[p, 0]
        py = pts[p, 1]
        pz = pts[p, 2]
        for i in range(M):
            dx = px - r0_flat[i, 0]
            dy = py - r0_flat[i, 1]
            dz = pz - r0_flat[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            rmag = math.sqrt(r2) + EPS
            invr3 = 1.0 / (rmag * r2 + EPS)
            rx = dx / rmag
            ry = dy / rmag
            rz = dz / rmag
            mx = m_flat[i, 0]
            my = m_flat[i, 1]
            mz = m_flat[i, 2]
            mdotr = mx * rx + my * ry + mz * rz
            Bx += factor * (3.0 * mdotr * rx - mx) * invr3
            By += factor * (3.0 * mdotr * ry - my) * invr3
            Bz += factor * (3.0 * mdotr * rz - mz) * invr3
        out[p, 0] = Bx
        out[p, 1] = By
        out[p, 2] = Bz
    return out


@njit(cache=True, parallel=True)
def compute_B_and_B0_from_m_flat(
    pts: NDArray[np.float64],
    r0_flat: NDArray[np.float64],
    m_flat: NDArray[np.float64],
    factor: float,
    origin: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute B at pts+origin and B0 at origin for per-magnet m vectors.
    """
    P = pts.shape[0]
    pts_all = np.empty((P + 1, 3), dtype=np.float64)
    ox = origin[0]
    oy = origin[1]
    oz = origin[2]
    for p in range(P):
        pts_all[p, 0] = pts[p, 0] + ox
        pts_all[p, 1] = pts[p, 1] + oy
        pts_all[p, 2] = pts[p, 2] + oz
    pts_all[P, 0] = ox
    pts_all[P, 1] = oy
    pts_all[P, 2] = oz

    B_all = compute_B_all_from_m_flat(pts_all, r0_flat, m_flat, factor)
    B = B_all[:-1]
    B0 = B_all[-1]
    return B, B0


__all__ = [
    "compute_B_and_B0",
    "compute_B_and_B0_phi_rkn",
    "compute_B_all_from_m_flat",
    "compute_B_and_B0_from_m_flat",
    "objective_only",
    "objective_only_phi_rkn",
]
