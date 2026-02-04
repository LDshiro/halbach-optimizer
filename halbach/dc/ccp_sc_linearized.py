from __future__ import annotations

import importlib
import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

EPS = 1e-30


def z_to_blocks(z: NDArray[np.float64], M: int) -> NDArray[np.float64]:
    return np.asarray(z, dtype=np.float64).reshape(M, 2)


def p_to_diag_blocks(
    p: NDArray[np.float64], M: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p_arr = np.asarray(p, dtype=np.float64).reshape(M)
    p_rep = np.repeat(p_arr, 2)
    return p_rep, p_arr


def eval_g_true(
    *,
    M: int,
    i_edge: NDArray[np.int32],
    j_edge: NDArray[np.int32],
    T2_edge: NDArray[np.float64],
    z: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    Z = z_to_blocks(z, M)
    X = z_to_blocks(x, M)
    zi = Z[i_edge]
    xj = X[j_edge]
    tmp = np.einsum("eab,eb->ea", T2_edge, xj)
    val = np.einsum("ea,ea->e", zi, tmp)
    g = np.zeros((M,), dtype=np.float64)
    np.add.at(g, i_edge, val)
    return g


def linearize_sc_terms(
    *,
    M: int,
    i_edge: NDArray[np.int32],
    j_edge: NDArray[np.int32],
    T2_edge: NDArray[np.float64],
    z_prev: NDArray[np.float64],
    x_prev: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    Zp = z_to_blocks(z_prev, M)
    Xp = z_to_blocks(x_prev, M)
    zpi = Zp[i_edge]
    xpj = Xp[j_edge]

    a = np.einsum("eab,eb->ea", np.swapaxes(T2_edge, 1, 2), zpi)
    b = np.einsum("eab,eb->ea", T2_edge, xpj)
    k = np.einsum("ea,ea->e", zpi, b)

    rows = np.concatenate([i_edge, i_edge])
    cols = np.concatenate([2 * j_edge, 2 * j_edge + 1])
    data = np.concatenate([a[:, 0], a[:, 1]])

    A_x = coo_matrix((data, (rows, cols)), shape=(M, 2 * M)).tocsr()
    A_x.sum_duplicates()

    bz = np.zeros((M, 2), dtype=np.float64)
    const = np.zeros((M,), dtype=np.float64)
    np.add.at(bz, i_edge, b)
    np.add.at(const, i_edge, k)

    bz0 = bz[:, 0]
    bz1 = bz[:, 1]
    return A_x, bz0, bz1, const


def _interleave_expr(zc: Any, zs: Any, M: int) -> Any:
    cp = cast(Any, importlib.import_module("cvxpy"))
    parts = []
    for i in range(M):
        parts.append(zc[i])
        parts.append(zs[i])
    return cp.hstack(parts)


def linearized_x_from_pz(
    *,
    z: Any,
    p: Any,
    z_prev: NDArray[np.float64],
    p_prev: NDArray[np.float64],
) -> Any:
    cp = cast(Any, importlib.import_module("cvxpy"))
    zc = z[0::2]
    zs = z[1::2]
    zpc = z_prev[0::2]
    zps = z_prev[1::2]
    pprev = p_prev

    xlin_c = cp.multiply(pprev, zc) + cp.multiply(p, zpc) - (pprev * zpc)
    xlin_s = cp.multiply(pprev, zs) + cp.multiply(p, zps) - (pprev * zps)
    return _interleave_expr(xlin_c, xlin_s, int(pprev.shape[0]))


def solve_ccp_sc_subproblem(
    *,
    Ax: NDArray[np.float64],
    Ay: NDArray[np.float64],
    Az: NDArray[np.float64],
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    reg_x: float,
    reg_p: float,
    reg_z: float,
    tau: float,
    z_prev: NDArray[np.float64],
    p_prev: NDArray[np.float64],
    x_prev: NDArray[np.float64],
    z_nom: NDArray[np.float64],
    cos_delta_nom: float,
    cos_delta_step: float | None,
    p_bounds: tuple[float, float],
    sc_eq_enabled: bool,
    sc_params: dict[str, Any],
    sc_lin: tuple[csr_matrix, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None,
    p_fix_value: float | None,
    solver: str,
    verbose: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    cp = cast(Any, importlib.import_module("cvxpy"))
    D = int(Ax.shape[1])
    M = D // 2
    Ay_diff = Ay - Ay[int(center_idx) : int(center_idx) + 1, :]

    z = cp.Variable(D)
    p = cp.Variable(M)
    x = cp.Variable(D)

    obj = 0.0
    if wy > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wy)) * (Ay_diff @ x))
    if wx > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wx)) * (Ax @ x))
    if wz > 0.0:
        obj += cp.sum_squares(np.sqrt(float(wz)) * (Az @ x))
    if reg_x > 0.0:
        obj += float(reg_x) * cp.sum_squares(x)
    if reg_p > 0.0:
        obj += float(reg_p) * cp.sum_squares(p)
    if reg_z > 0.0:
        obj += float(reg_z) * cp.sum_squares(z)

    obj_total = obj - 2.0 * float(tau) * cp.sum(cp.multiply(z_prev, z))

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

    pmin, pmax = p_bounds
    constraints.append(p >= float(pmin))
    constraints.append(p <= float(pmax))

    x_lin = linearized_x_from_pz(z=z, p=p, z_prev=z_prev, p_prev=p_prev)
    constraints.append(x == x_lin)

    if sc_eq_enabled:
        if sc_lin is None:
            raise ValueError("sc_lin must be provided when sc_eq_enabled is True")
        A_x, bz0, bz1, const = sc_lin
        zc = z[0::2]
        zs = z[1::2]
        g_lin = A_x @ x + cp.multiply(bz0, zc) + cp.multiply(bz1, zs) - const

        chi = float(sc_params["chi"])
        Nd = float(sc_params["Nd"])
        denom = 1.0 + chi * Nd
        p0 = sc_params["p0"]
        if np.ndim(p0) == 0:
            p0_vec = float(p0) * np.ones(M, dtype=np.float64)
        else:
            p0_vec = np.asarray(p0, dtype=np.float64)
            if p0_vec.shape != (M,):
                raise ValueError("p0 must be scalar or shape (M,)")

        constraints.append(denom * p - chi * g_lin == p0_vec)
    else:
        if p_fix_value is None:
            raise ValueError("p_fix_value must be set when sc_eq_enabled is False")
        constraints.append(p == float(p_fix_value))

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
            prob.solve(solver=cp.SCS, verbose=verbose, max_iters=20000, eps=1e-4)
    elif solver_key == "SCS":
        prob.solve(solver=cp.SCS, verbose=verbose, max_iters=20000, eps=1e-4)
    else:
        prob.solve(solver=solver, verbose=verbose)

    if z.value is None or p.value is None or x.value is None:
        raise RuntimeError("CCP subproblem failed to produce a solution")

    z_new = np.asarray(z.value, dtype=np.float64).reshape(-1)
    p_new = np.asarray(p.value, dtype=np.float64).reshape(-1)
    x_new = np.asarray(x.value, dtype=np.float64).reshape(-1)

    info: dict[str, Any] = {
        "status": str(prob.status),
        "obj": float(prob.value) if prob.value is not None else float("nan"),
        "solver": str(solver),
        "D": int(D),
    }
    return z_new, p_new, x_new, info


def _eval_objective(
    Ax: NDArray[np.float64],
    Ay: NDArray[np.float64],
    Az: NDArray[np.float64],
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    reg_x: float,
    reg_p: float,
    reg_z: float,
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    z: NDArray[np.float64],
) -> float:
    Ay_diff = Ay - Ay[int(center_idx) : int(center_idx) + 1, :]
    f_val = 0.0
    if wy > 0.0:
        tmp = Ay_diff @ x
        f_val += float(wy) * float(np.sum(tmp * tmp))
    if wx > 0.0:
        tmp = Ax @ x
        f_val += float(wx) * float(np.sum(tmp * tmp))
    if wz > 0.0:
        tmp = Az @ x
        f_val += float(wz) * float(np.sum(tmp * tmp))
    if reg_x > 0.0:
        f_val += float(reg_x) * float(np.sum(x * x))
    if reg_p > 0.0:
        f_val += float(reg_p) * float(np.sum(p * p))
    if reg_z > 0.0:
        f_val += float(reg_z) * float(np.sum(z * z))
    return float(f_val)


def run_ccp_sc(
    *,
    Ax: NDArray[np.float64],
    Ay: NDArray[np.float64],
    Az: NDArray[np.float64],
    center_idx: int,
    wx: float,
    wy: float,
    wz: float,
    reg_x: float,
    reg_p: float,
    reg_z: float,
    tau0: float,
    tau_mult: float,
    tau_max: float,
    iters: int,
    tol: float,
    tol_f: float,
    z_nom: NDArray[np.float64],
    z_init: NDArray[np.float64] | None = None,
    delta_nom_deg: float,
    delta_step_deg: float | None,
    p_init: NDArray[np.float64],
    sc_eq_enabled: bool,
    sc_params: dict[str, Any],
    p_bounds: tuple[float, float],
    p_fix_value: float | None,
    i_edge: NDArray[np.int32],
    j_edge: NDArray[np.int32],
    T2_edge: NDArray[np.float64],
    solver: str,
    verbose: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    M = int(p_init.shape[0])
    if z_init is None:
        z_prev = np.asarray(z_nom, dtype=np.float64)
    else:
        z_prev = np.asarray(z_init, dtype=np.float64).reshape(-1)
        if z_prev.shape != z_nom.shape:
            raise ValueError("z_init shape does not match z_nom")
    p_prev = np.asarray(p_init, dtype=np.float64)

    x_prev = np.empty(2 * M, dtype=np.float64)
    x_prev[0::2] = p_prev * z_prev[0::2]
    x_prev[1::2] = p_prev * z_prev[1::2]

    tau = float(tau0)
    cos_delta_nom = math.cos(math.radians(float(delta_nom_deg)))
    cos_delta_step = (
        math.cos(math.radians(float(delta_step_deg))) if delta_step_deg is not None else None
    )

    trace: dict[str, Any] = {"iters": []}
    f_prev = _eval_objective(
        Ax,
        Ay,
        Az,
        center_idx,
        wx,
        wy,
        wz,
        reg_x,
        reg_p,
        reg_z,
        x_prev,
        p_prev,
        z_prev,
    )

    chi = float(sc_params.get("chi", 0.0))
    Nd = float(sc_params.get("Nd", 0.0))
    p0 = sc_params.get("p0", 0.0)
    if np.ndim(p0) == 0:
        p0_vec = float(p0) * np.ones(M, dtype=np.float64)
    else:
        p0_vec = np.asarray(p0, dtype=np.float64)
        if p0_vec.shape != (M,):
            raise ValueError("p0 must be scalar or shape (M,)")

    for t in range(int(iters)):
        sc_lin = None
        if sc_eq_enabled:
            sc_lin = linearize_sc_terms(
                M=M,
                i_edge=i_edge,
                j_edge=j_edge,
                T2_edge=T2_edge,
                z_prev=z_prev,
                x_prev=x_prev,
            )

        z_new, p_new, x_new, info = solve_ccp_sc_subproblem(
            Ax=Ax,
            Ay=Ay,
            Az=Az,
            center_idx=center_idx,
            wx=wx,
            wy=wy,
            wz=wz,
            reg_x=reg_x,
            reg_p=reg_p,
            reg_z=reg_z,
            tau=tau,
            z_prev=z_prev,
            p_prev=p_prev,
            x_prev=x_prev,
            z_nom=z_nom,
            cos_delta_nom=cos_delta_nom,
            cos_delta_step=cos_delta_step,
            p_bounds=p_bounds,
            sc_eq_enabled=sc_eq_enabled,
            sc_params=sc_params,
            sc_lin=sc_lin,
            p_fix_value=p_fix_value,
            solver=solver,
            verbose=verbose,
        )

        f_new = _eval_objective(
            Ax,
            Ay,
            Az,
            center_idx,
            wx,
            wy,
            wz,
            reg_x,
            reg_p,
            reg_z,
            x_new,
            p_new,
            z_new,
        )

        Z = z_to_blocks(z_new, M)
        norms = np.sqrt(Z[:, 0] * Z[:, 0] + Z[:, 1] * Z[:, 1])
        norm_min = float(np.min(norms)) if norms.size else 0.0
        norm_mean = float(np.mean(norms)) if norms.size else 0.0
        norm_max = float(np.max(norms)) if norms.size else 0.0

        X = z_to_blocks(x_new, M)
        prod_err = np.sqrt(
            (X[:, 0] - p_new * Z[:, 0]) ** 2 + (X[:, 1] - p_new * Z[:, 1]) ** 2
        ) / np.maximum(p_new, EPS)
        r_prod = float(np.max(prod_err)) if prod_err.size else 0.0

        denom = 1.0 + chi * Nd
        g_true = eval_g_true(
            M=M,
            i_edge=i_edge,
            j_edge=j_edge,
            T2_edge=T2_edge,
            z=z_new,
            x=x_new,
        )
        r_sc_vec = denom * p_new - (p0_vec + chi * g_true)
        r_sc = float(np.linalg.norm(r_sc_vec)) / max(float(np.linalg.norm(p0_vec)), EPS)

        rel_change = float(np.linalg.norm(z_new - z_prev)) / max(float(np.linalg.norm(z_prev)), EPS)
        rel_f = abs(f_new - f_prev) / max(abs(f_prev), EPS)

        trace["iters"].append(
            {
                "t": int(t),
                "status": str(info.get("status")),
                "tau": float(tau),
                "f": float(f_new),
                "rel_change": float(rel_change),
                "rel_f": float(rel_f),
                "norm_min": float(norm_min),
                "norm_mean": float(norm_mean),
                "norm_max": float(norm_max),
                "r_prod": float(r_prod),
                "r_sc": float(r_sc),
            }
        )
        if verbose:
            print(
                f"[CCP] t={t} tau={tau:.3e} status={info.get('status')} "
                f"f={f_new:.6e} rel_change={rel_change:.3e} rel_f={rel_f:.3e} "
                f"r_prod={r_prod:.3e} r_sc={r_sc:.3e} norm_mean={norm_mean:.3f}",
                flush=True,
            )

        if rel_change < float(tol) and rel_f < float(tol_f):
            z_prev = z_new
            p_prev = p_new
            x_prev = x_new
            break

        z_prev = z_new
        p_prev = p_new
        x_prev = x_new
        f_prev = f_new
        tau = min(float(tau) * float(tau_mult), float(tau_max))

    return z_prev, p_prev, x_prev, trace


__all__ = [
    "z_to_blocks",
    "p_to_diag_blocks",
    "eval_g_true",
    "linearize_sc_terms",
    "linearized_x_from_pz",
    "solve_ccp_sc_subproblem",
    "run_ccp_sc",
]
