from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def solve_qp_relaxed_mxy(
    Ax: NDArray[np.float64],
    Ay: NDArray[np.float64],
    Az: NDArray[np.float64],
    *,
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    pmax: float,
    reg: float = 0.0,
    solver: str = "OSQP",
    verbose: bool = False,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    cp = cast(Any, importlib.import_module("cvxpy"))

    D = int(Ax.shape[1])
    Ay_diff = Ay - Ay[int(center_idx) : int(center_idx) + 1, :]

    x = cp.Variable(D)
    obj = 0.0
    if wy > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wy)) * (Ay_diff @ x))
    if wx > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wx)) * (Ax @ x))
    if wz > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wz)) * (Az @ x))
    if reg > 0.0:
        obj += float(reg) * cp.sum_squares(x)

    constraints = [x <= float(pmax), x >= -float(pmax)]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    if solver.upper() == "OSQP":
        prob.solve(
            solver=cp.OSQP,
            verbose=verbose,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=20000,
        )
    else:
        prob.solve(solver=solver, verbose=verbose)

    if x.value is None:
        raise RuntimeError("QP solve failed to produce a solution")
    x_opt = np.asarray(x.value, dtype=np.float64).reshape(-1)

    info: dict[str, Any] = {
        "status": str(prob.status),
        "obj_value": float(prob.value) if prob.value is not None else float("nan"),
        "solver": str(solver),
        "D": int(D),
    }
    return x_opt, info


def derive_phi_p_from_mxy(
    x_opt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mx = np.asarray(x_opt[0::2], dtype=np.float64)
    my = np.asarray(x_opt[1::2], dtype=np.float64)
    phi = np.arctan2(my, mx)
    p = np.sqrt(mx * mx + my * my)
    return phi, p


__all__ = ["solve_qp_relaxed_mxy", "derive_phi_p_from_mxy"]
