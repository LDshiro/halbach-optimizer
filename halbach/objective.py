from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, phi0
from halbach.physics import compute_B_and_B0

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:

    def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...

else:
    from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def grad_alpha_and_radius_fixed(
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
    DBx: NDArray[np.float64],
    DBy: NDArray[np.float64],
    DBz: NDArray[np.float64],
    sumDBx: float,
    sumDBy: float,
    sumDBz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Analytic gradient with respect to alphas and r_bases.
    """
    Rloc = alphas.shape[0]
    Kloc = alphas.shape[1]
    Nloc = theta.shape[0]
    M = DBx.shape[0]
    invM = 1.0 / M
    g_alpha = np.zeros((Rloc, Kloc), dtype=np.float64)
    g_rbase = np.zeros(Kloc, dtype=np.float64)

    dB0_dalpha_x = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_dalpha_y = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_dalpha_z = np.zeros((Rloc, Kloc), dtype=np.float64)
    dB0_drb_x = np.zeros(Kloc, dtype=np.float64)
    dB0_drb_y = np.zeros(Kloc, dtype=np.float64)
    dB0_drb_z = np.zeros(Kloc, dtype=np.float64)

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
                s2i = sin2[i]

                phi = 2.0 * th + phi0 + ak * s2i
                mx = m0 * math.cos(phi)
                my = m0 * math.sin(phi)
                dmx = -m0 * math.sin(phi)
                dmy = m0 * math.cos(phi)

                px = rho * c
                py = rho * s
                dx0 = -px
                dy0 = -py
                dz0 = -z0
                r20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0
                r0mag = math.sqrt(r20) + 1e-30
                invr30 = 1.0 / (r0mag * r20 + 1e-30)
                rx0 = dx0 / r0mag
                ry0 = dy0 / r0mag
                rz0 = dz0 / r0mag

                dmdotr0 = dmx * rx0 + dmy * ry0
                dBx0 = factor * (3.0 * dmdotr0 * rx0 - dmx) * invr30
                dBy0 = factor * (3.0 * dmdotr0 * ry0 - dmy) * invr30
                dBz0 = factor * (3.0 * dmdotr0 * rz0 - 0.0) * invr30

                dB0_dalpha_x[r, k] += dBx0 * s2i
                dB0_dalpha_y[r, k] += dBy0 * s2i
                dB0_dalpha_z[r, k] += dBz0 * s2i

                r2 = r20
                rmag = r0mag
                r5 = rmag * r2 * r2
                r7 = r5 * r2
                S = mx * dx0 + my * dy0

                Jx0 = factor * (
                    (3.0 * mx * dx0 + 3.0 * mx * dx0) / r5
                    + 3.0 * S / r5
                    - 15.0 * S * dx0 * dx0 / r7
                )
                Jx1 = factor * ((3.0 * mx * dy0 + 3.0 * my * dx0) / r5 - 15.0 * S * dy0 * dx0 / r7)
                Jx2 = factor * ((3.0 * mx * dz0 + 3.0 * 0.0 * dx0) / r5 - 15.0 * S * dz0 * dx0 / r7)
                Jy0 = factor * ((3.0 * my * dx0 + 3.0 * mx * dy0) / r5 - 15.0 * S * dx0 * dy0 / r7)
                Jy1 = factor * (
                    (3.0 * my * dy0 + 3.0 * my * dy0) / r5
                    + 3.0 * S / r5
                    - 15.0 * S * dy0 * dy0 / r7
                )
                Jy2 = factor * ((3.0 * my * dz0 + 3.0 * 0.0 * dy0) / r5 - 15.0 * S * dz0 * dy0 / r7)

                dB0_drb_x[k] += -(Jx0 * c + Jy0 * s)
                dB0_drb_y[k] += -(Jx1 * c + Jy1 * s)
                dB0_drb_z[k] += -(Jx2 * c + Jy2 * s)

    for k in range(Kloc):
        z0 = z_layers[k]
        rb = r_bases[k]
        for r in range(Rloc):
            ak = alphas[r, k]
            rho = rb + ring_offsets[r]
            sum_dot_alpha = 0.0
            sum_dot_rbase = 0.0
            for i in range(Nloc):
                th = theta[i]
                c = cth[i]
                s = sth[i]
                s2i = sin2[i]
                phi = 2.0 * th + phi0 + ak * s2i
                mx = m0 * math.cos(phi)
                my = m0 * math.sin(phi)
                dmx = -m0 * math.sin(phi)
                dmy = m0 * math.cos(phi)
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

                    dmdotr = dmx * rx + dmy * ry
                    dBx = factor * (3.0 * dmdotr * rx - dmx) * invr3
                    dBy = factor * (3.0 * dmdotr * ry - dmy) * invr3
                    dBz = factor * (3.0 * dmdotr * rz - 0.0) * invr3
                    sum_dot_alpha += (DBx[p] * dBx + DBy[p] * dBy + DBz[p] * dBz) * s2i

                    r5 = rmag * r2 * r2
                    r7 = r5 * r2
                    S = mx * dx + my * dy
                    Jx0 = factor * (
                        (3.0 * mx * dx + 3.0 * mx * dx) / r5
                        + 3.0 * S / r5
                        - 15.0 * S * dx * dx / r7
                    )
                    Jx1 = factor * ((3.0 * mx * dy + 3.0 * my * dx) / r5 - 15.0 * S * dy * dx / r7)
                    Jx2 = factor * ((3.0 * mx * dz + 3.0 * 0.0 * dx) / r5 - 15.0 * S * dz * dx / r7)
                    Jy0 = factor * ((3.0 * my * dx + 3.0 * mx * dy) / r5 - 15.0 * S * dx * dy / r7)
                    Jy1 = factor * (
                        (3.0 * my * dy + 3.0 * my * dy) / r5
                        + 3.0 * S / r5
                        - 15.0 * S * dy * dy / r7
                    )
                    Jy2 = factor * ((3.0 * my * dz + 3.0 * 0.0 * dy) / r5 - 15.0 * S * dz * dy / r7)
                    dBrx = -Jx0 * c - Jy0 * s
                    dBry = -Jx1 * c - Jy1 * s
                    dBrz = -Jx2 * c - Jy2 * s
                    sum_dot_rbase += DBx[p] * dBrx + DBy[p] * dBry + DBz[p] * dBrz

            g_alpha[r, k] += (
                2.0
                * invM
                * (
                    sum_dot_alpha
                    - (
                        sumDBx * dB0_dalpha_x[r, k]
                        + sumDBy * dB0_dalpha_y[r, k]
                        + sumDBz * dB0_dalpha_z[r, k]
                    )
                )
            )
            g_rbase[k] += 2.0 * invM * (sum_dot_rbase)

        g_rbase[k] += (
            -2.0 * invM * (sumDBx * dB0_drb_x[k] + sumDBy * dB0_drb_y[k] + sumDBz * dB0_drb_z[k])
        )

    return g_alpha, g_rbase


def objective_with_grads_fixed(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: dict[str, Any],
    pts: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
    """
    Objective J and y-space gradients, plus |B0|.
    """
    Bx, By, Bz, B0x, B0y, B0z = compute_B_and_B0(
        alphas,
        r_bases,
        geom["theta"],
        geom["sin2"],
        geom["cth"],
        geom["sth"],
        geom["z_layers"],
        geom["ring_offsets"],
        pts,
        FACTOR,
        phi0,
        m0,
    )
    DBx = Bx - B0x
    DBy = By - B0y
    DBz = Bz - B0z
    J = float(np.mean(DBx * DBx + DBy * DBy + DBz * DBz))
    sumDBx = np.sum(DBx)
    sumDBy = np.sum(DBy)
    sumDBz = np.sum(DBz)

    g_alpha, g_rbase = grad_alpha_and_radius_fixed(
        alphas,
        r_bases,
        geom["theta"],
        geom["sin2"],
        geom["cth"],
        geom["sth"],
        geom["z_layers"],
        geom["ring_offsets"],
        pts,
        FACTOR,
        phi0,
        m0,
        DBx,
        DBy,
        DBz,
        sumDBx,
        sumDBy,
        sumDBz,
    )

    B0n = float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))
    return J, g_alpha, g_rbase, B0n


__all__ = ["grad_alpha_and_radius_fixed", "objective_with_grads_fixed"]
