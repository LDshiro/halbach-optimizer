from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from halbach.geom import build_r_bases_from_vars, unpack_x
from halbach.objective import objective_with_grads_fixed
from halbach.types import Geometry

if TYPE_CHECKING:
    import optype.numpy as onp

    Float1DArray: TypeAlias = onp.Array1D[np.float64]
else:
    Float1DArray: TypeAlias = NDArray[np.float64]


def hvp_y(
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    geom: Geometry,
    pts: NDArray[np.float64],
    v_y: NDArray[np.float64],
    eps_hvp: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    y-space Hessian-vector product via central differences.
    """
    P_A = geom.R * geom.K
    vA = v_y[:P_A].reshape(geom.R, geom.K)
    vR = v_y[P_A:]

    vmax = float(np.max(np.abs(v_y)) + 1e-16)
    h = eps_hvp / vmax

    a_p = alphas + h * vA
    r_p = r_bases + h * vR
    a_m = alphas - h * vA
    r_m = r_bases - h * vR

    _, gA_p, gR_p, _ = objective_with_grads_fixed(a_p, r_p, geom, pts)
    _, gA_m, gR_m, _ = objective_with_grads_fixed(a_m, r_m, geom, pts)

    hvA = (gA_p - gA_m) / (2.0 * h)
    hvR = (gR_p - gR_m) / (2.0 * h)
    return hvA.ravel(), hvR


def fun_grad_gradnorm_fixed(
    x: Float1DArray,
    geom: Geometry,
    pts: NDArray[np.float64],
    sigma_alpha: float,
    sigma_r: float,
    rho_gn: float,
    eps_hvp: float,
    r0: float,
    lower_var: NDArray[np.int_],
    upper_var: NDArray[np.int_],
) -> tuple[float, Float1DArray, float, float, float]:
    """
    Robust objective and x-space gradient using y-space GN formulation.
    """
    alphas, r_vars = unpack_x(x, geom.R, geom.K)
    r_bases = build_r_bases_from_vars(r_vars, geom.K, r0, lower_var, upper_var)

    J, gA_y, gRb_y, B0n = objective_with_grads_fixed(alphas, r_bases, geom, pts)

    g_y = np.concatenate([gA_y.ravel(), gRb_y])
    P_A = geom.R * geom.K
    gn2 = (sigma_alpha**2) * np.dot(g_y[:P_A], g_y[:P_A]) + (sigma_r**2) * np.dot(
        g_y[P_A:], g_y[P_A:]
    )
    v_y = np.concatenate([(sigma_alpha**2) * g_y[:P_A], (sigma_r**2) * g_y[P_A:]])

    hvA, hvR = hvp_y(alphas, r_bases, geom, pts, v_y, eps_hvp)

    Jgn = J + 0.5 * rho_gn * gn2

    gA_x = gA_y.ravel() + rho_gn * hvA
    gR_x = np.zeros_like(r_vars)
    for j, k_low in enumerate(lower_var):
        k_up = upper_var[j]
        gR_x[j] = (gRb_y[k_low] + gRb_y[k_up]) + rho_gn * (hvR[k_low] + hvR[k_up])

    grad_x = cast(Float1DArray, np.concatenate([gA_x, gR_x]))
    return float(Jgn), grad_x, float(B0n), float(J), float(gn2)


__all__ = ["hvp_y", "fun_grad_gradnorm_fixed", "Float1DArray"]
