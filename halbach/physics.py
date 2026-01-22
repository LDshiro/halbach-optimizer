from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:

    def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...

else:
    from numba import njit  # type: ignore[import-untyped]


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


__all__ = ["compute_B_and_B0", "objective_only"]
