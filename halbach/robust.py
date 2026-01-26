from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR
from halbach.geom import ParamMap, pack_grad, unpack_x
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
    factor: float = FACTOR,
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

    _, gA_p, gR_p, _ = objective_with_grads_fixed(a_p, r_p, geom, pts, factor=factor)
    _, gA_m, gR_m, _ = objective_with_grads_fixed(a_m, r_m, geom, pts, factor=factor)

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
    param_map: ParamMap,
    alphas0: NDArray[np.float64],
    r_bases0: NDArray[np.float64],
    factor: float = FACTOR,
) -> tuple[float, Float1DArray, float, float, float]:
    """
    Robust objective and x-space gradient using y-space GN formulation.
    """
    alphas, r_bases = unpack_x(x, alphas0, r_bases0, param_map)

    J, gA_y, gRb_y, B0n = objective_with_grads_fixed(alphas, r_bases, geom, pts, factor=factor)

    gA_flat = gA_y.ravel()
    gR_masked = gRb_y * param_map.free_r_mask
    g_y = np.concatenate([gA_flat, gR_masked])
    P_A = geom.R * geom.K
    gn2 = (sigma_alpha**2) * np.dot(g_y[:P_A], g_y[:P_A]) + (sigma_r**2) * np.dot(
        g_y[P_A:], g_y[P_A:]
    )
    v_y = np.concatenate([(sigma_alpha**2) * g_y[:P_A], (sigma_r**2) * g_y[P_A:]])

    hvA, hvR = hvp_y(alphas, r_bases, geom, pts, v_y, eps_hvp, factor=factor)

    scale = factor / FACTOR if FACTOR != 0.0 else 1.0
    rho_gn_eff = rho_gn / (scale * scale)
    Jgn = J + 0.5 * rho_gn_eff * gn2

    gA_full = (gA_flat + rho_gn_eff * hvA).reshape(geom.R, geom.K)
    gR_full = gRb_y + rho_gn_eff * hvR
    grad_x = cast(Float1DArray, pack_grad(gA_full, gR_full, param_map))
    return float(Jgn), grad_x, float(B0n), float(J), float(gn2)


__all__ = ["hvp_y", "fun_grad_gradnorm_fixed", "Float1DArray"]
