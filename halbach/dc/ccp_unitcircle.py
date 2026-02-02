from __future__ import annotations

import importlib
import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

EPS = 1e-30


def make_phi_nominal_halbach(
    *,
    N: int,
    K: int,
    R: int,
    phi0: float = 0.0,
    mode: str = "2theta",
) -> NDArray[np.float64]:
    if mode != "2theta":
        raise ValueError("Only mode='2theta' is supported")
    if N < 1 or K < 1 or R < 1:
        raise ValueError("N,K,R must be >= 1")
    theta = 2.0 * math.pi * np.arange(N, dtype=np.float64) / float(N)
    phi_n = 2.0 * theta + float(phi0)
    phi_rkn = np.zeros((R, K, N), dtype=np.float64)
    phi_rkn[:, :, :] = phi_n[None, None, :]
    return phi_rkn


def phi_to_zvec(phi_flat: NDArray[np.float64]) -> NDArray[np.float64]:
    phi = np.asarray(phi_flat, dtype=np.float64)
    M = int(phi.shape[0])
    z = np.empty(2 * M, dtype=np.float64)
    z[0::2] = np.cos(phi)
    z[1::2] = np.sin(phi)
    return z


def zvec_to_phi_pnorm(
    z: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z_arr = np.asarray(z, dtype=np.float64)
    c = z_arr[0::2]
    s = z_arr[1::2]
    phi = np.arctan2(s, c)
    pnorm = np.sqrt(c * c + s * s)
    return phi, pnorm


def solve_ccp_subproblem(
    *,
    Axd: NDArray[np.float64],
    Ayd: NDArray[np.float64],
    Azd: NDArray[np.float64],
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    reg: float,
    tau: float,
    z_prev: NDArray[np.float64],
    z_nom: NDArray[np.float64],
    cos_delta_nom: float,
    cos_delta_step: float | None,
    solver: str = "ECOS",
    verbose: bool = False,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    cp = cast(Any, importlib.import_module("cvxpy"))

    D = int(Axd.shape[1])
    M = D // 2
    Ay_diff = Ayd - Ayd[int(center_idx) : int(center_idx) + 1, :]

    z = cp.Variable(D)
    obj = 0.0
    if wy > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wy)) * (Ay_diff @ z))
    if wx > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wx)) * (Axd @ z))
    if wz > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wz)) * (Azd @ z))
    if reg > 0.0:
        obj += float(reg) * cp.sum_squares(z)

    lin = 0.0
    for i in range(M):
        zi = z[2 * i : 2 * i + 2]
        zpi = z_prev[2 * i : 2 * i + 2]
        lin += zpi[0] * zi[0] + zpi[1] * zi[1]
    obj_total = obj - 2.0 * float(tau) * lin

    constraints = []
    for i in range(M):
        zi = z[2 * i : 2 * i + 2]
        constraints.append(cp.norm(zi, 2) <= 1.0)
        zni = z_nom[2 * i : 2 * i + 2]
        constraints.append(zni[0] * zi[0] + zni[1] * zi[1] >= float(cos_delta_nom))

    if cos_delta_step is not None:
        for i in range(M):
            zi = z[2 * i : 2 * i + 2]
            zpi = z_prev[2 * i : 2 * i + 2]
            constraints.append(zpi[0] * zi[0] + zpi[1] * zi[1] >= float(cos_delta_step))

    prob = cp.Problem(cp.Minimize(obj_total), constraints)
    solver_key = str(solver).upper()
    if solver_key == "ECOS":
        try:
            prob.solve(
                solver=cp.ECOS,
                verbose=verbose,
                max_iters=5000,
                abstol=1e-8,
                reltol=1e-8,
            )
        except Exception:
            prob.solve(
                solver=cp.SCS,
                verbose=verbose,
                max_iters=20000,
                eps=1e-4,
            )
    elif solver_key == "SCS":
        prob.solve(
            solver=cp.SCS,
            verbose=verbose,
            max_iters=20000,
            eps=1e-4,
        )
    else:
        prob.solve(solver=solver, verbose=verbose)

    if z.value is None:
        raise RuntimeError("CCP subproblem failed to produce a solution")

    z_new = np.asarray(z.value, dtype=np.float64).reshape(-1)
    info: dict[str, Any] = {
        "status": str(prob.status),
        "obj_total": float(prob.value) if prob.value is not None else float("nan"),
        "solver": str(solver),
        "D": int(D),
    }
    return z_new, info


def run_ccp_unitcircle(
    *,
    Axd: NDArray[np.float64],
    Ayd: NDArray[np.float64],
    Azd: NDArray[np.float64],
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    reg: float,
    tau0: float,
    tau_mult: float,
    tau_max: float,
    iters: int,
    tol: float,
    tol_f: float,
    phi_nom_flat: NDArray[np.float64],
    delta_nom_deg: float,
    delta_step_deg: float | None,
    z_init: NDArray[np.float64] | None = None,
    solver: str,
    verbose: bool,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    z_nom = phi_to_zvec(phi_nom_flat)
    if z_init is None:
        z_prev = np.asarray(z_nom, dtype=np.float64)
    else:
        z_prev = np.asarray(z_init, dtype=np.float64).reshape(-1)
        if z_prev.shape != z_nom.shape:
            raise ValueError("z_init shape does not match z_nom")
    cos_delta_nom = math.cos(math.radians(float(delta_nom_deg)))
    cos_delta_step = (
        math.cos(math.radians(float(delta_step_deg))) if delta_step_deg is not None else None
    )

    Ay_diff = Ayd - Ayd[int(center_idx) : int(center_idx) + 1, :]

    def f_eval(z: NDArray[np.float64]) -> float:
        f_val = 0.0
        if wy > 0.0:
            tmp = Ay_diff @ z
            f_val += float(wy) * float(np.sum(tmp * tmp))
        if wx > 0.0:
            tmp = Axd @ z
            f_val += float(wx) * float(np.sum(tmp * tmp))
        if wz > 0.0:
            tmp = Azd @ z
            f_val += float(wz) * float(np.sum(tmp * tmp))
        if reg > 0.0:
            f_val += float(reg) * float(np.sum(z * z))
        return float(f_val)

    trace: dict[str, Any] = {"iters": []}
    tau = float(tau0)
    f_prev = f_eval(z_prev)

    for t in range(int(iters)):
        z_new, info = solve_ccp_subproblem(
            Axd=Axd,
            Ayd=Ayd,
            Azd=Azd,
            center_idx=int(center_idx),
            wx=float(wx),
            wy=float(wy),
            wz=float(wz),
            reg=float(reg),
            tau=float(tau),
            z_prev=z_prev,
            z_nom=z_nom,
            cos_delta_nom=float(cos_delta_nom),
            cos_delta_step=cos_delta_step,
            solver=str(solver),
            verbose=bool(verbose),
        )
        f_new = f_eval(z_new)

        c = z_new[0::2]
        s = z_new[1::2]
        norms = np.sqrt(c * c + s * s)
        norm_min = float(np.min(norms)) if norms.size else 0.0
        norm_mean = float(np.mean(norms)) if norms.size else 0.0
        norm_max = float(np.max(norms)) if norms.size else 0.0

        rel_change = float(np.linalg.norm(z_new - z_prev)) / max(float(np.linalg.norm(z_prev)), EPS)
        rel_f = abs(f_new - f_prev) / max(abs(f_prev), EPS)

        trace["iters"].append(
            {
                "t": int(t),
                "tau": float(tau),
                "status": str(info.get("status")),
                "f": float(f_new),
                "rel_change": float(rel_change),
                "rel_f": float(rel_f),
                "norm_min": float(norm_min),
                "norm_mean": float(norm_mean),
                "norm_max": float(norm_max),
            }
        )

        if rel_change < float(tol) and rel_f < float(tol_f):
            z_prev = z_new
            break

        z_prev = z_new
        f_prev = f_new
        tau = min(float(tau) * float(tau_mult), float(tau_max))

    return z_prev, trace


__all__ = [
    "make_phi_nominal_halbach",
    "phi_to_zvec",
    "zvec_to_phi_pnorm",
    "solve_ccp_subproblem",
    "run_ccp_unitcircle",
]
