from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from halbach.dc.ccp_sc_linearized import run_ccp_sc
from halbach.dc.ccp_unitcircle import make_phi_nominal_halbach, phi_to_zvec, zvec_to_phi_pnorm
from halbach.dc.dc_io import write_dc_run
from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.init_from_lbfgs import load_phi_flat_from_lbfgs_run
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane
from halbach.near import NearWindow, build_near_graph
from halbach.sc_linear_system import (
    build_T2_edges,
    edges_from_near,
    solve_p_easy_axis_linear_system,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DC CCP with optional self-consistent linear equalities (on/off)"
    )
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--length-m", type=float, required=True)
    parser.add_argument("--roi-radius-m", type=float, required=True)
    parser.add_argument("--roi-grid-n", type=int, required=True)
    parser.add_argument("--wx", type=float, default=0.0)
    parser.add_argument("--wy", type=float, default=1.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--factor", type=float, default=1e-7)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--delta-nom-deg", type=float, default=5.0)
    parser.add_argument("--delta-step-deg", type=float, default=2.0)
    parser.add_argument("--tau0", type=float, default=1e-4)
    parser.add_argument("--tau-mult", type=float, default=1.2)
    parser.add_argument("--tau-max", type=float, default=1e-1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--tol-f", type=float, default=1e-9)
    parser.add_argument("--reg-x", type=float, default=0.0)
    parser.add_argument("--reg-p", type=float, default=0.0)
    parser.add_argument("--reg-z", type=float, default=0.0)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sc-eq", dest="sc_eq", action="store_true")
    group.add_argument("--no-sc-eq", dest="sc_eq", action="store_false")
    parser.set_defaults(sc_eq=True)

    parser.add_argument("--p-fix", type=float, default=None)

    parser.add_argument("--sc-chi", type=float, default=0.05)
    parser.add_argument("--sc-Nd", type=float, default=1.0 / 3.0)
    parser.add_argument("--sc-p0", type=float, default=1.0)
    parser.add_argument("--sc-volume-mm3", type=float, default=1000.0)
    parser.add_argument("--sc-near-wr", type=int, default=0)
    parser.add_argument("--sc-near-wz", type=int, default=1)
    parser.add_argument("--sc-near-wphi", type=int, default=2)
    parser.add_argument(
        "--sc-near-kernel",
        type=str,
        default="dipole",
        choices=["dipole", "multi-dipole", "cellavg", "cube-average"],
    )
    parser.add_argument("--sc-subdip-n", type=int, default=2)
    parser.add_argument("--pmin", type=float, default=0.0)
    parser.add_argument("--pmax", type=float, default=2.0)
    parser.add_argument("--solver", type=str, default="ECOS")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--init-run", type=str, default=None)
    return parser.parse_args()


def _interleave_x(p: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.empty_like(z)
    x[0::2] = p * z[0::2]
    x[1::2] = p * z[1::2]
    return x


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


def main() -> None:
    args = _parse_args()

    sc_near_kernel = "cellavg" if args.sc_near_kernel == "cube-average" else args.sc_near_kernel

    z_positions = make_z_positions_uniform(int(args.K), float(args.length_m))
    r0_rkn, _info = build_ring_stack_positions(
        N=int(args.N),
        K=int(args.K),
        R=int(args.R),
        radii_m=float(args.radius_m),
        z_positions_m=z_positions,
    )
    r0_flat = r0_rkn.reshape(-1, 3)
    M = int(r0_flat.shape[0])

    pts = make_roi_points_xy_plane(
        radius_m=float(args.roi_radius_m),
        grid_n=int(args.roi_grid_n),
        include_center=True,
    )
    center_idx = find_center_index(pts)
    P = int(pts.shape[0])

    Ax, Ay, Az = build_B_mxy_matrices(pts, r0_flat, factor=float(args.factor))

    phi_rkn = make_phi_nominal_halbach(
        N=int(args.N), K=int(args.K), R=int(args.R), phi0=float(args.phi0)
    )
    phi_nom_flat = phi_rkn.reshape(-1)
    z_nom = phi_to_zvec(phi_nom_flat)

    phi_init_flat = None
    z_init = None
    init_model = None
    if args.init_run:
        phi_init_flat, init_model = load_phi_flat_from_lbfgs_run(
            Path(args.init_run),
            expected_R=int(args.R),
            expected_K=int(args.K),
            expected_N=int(args.N),
        )
        z_init = phi_to_zvec(phi_init_flat)

    delta_step = float(args.delta_step_deg)
    delta_step_deg = None if delta_step < 0.0 else float(delta_step)

    near = build_near_graph(
        int(args.R),
        int(args.K),
        int(args.N),
        NearWindow(wr=int(args.sc_near_wr), wz=int(args.sc_near_wz), wphi=int(args.sc_near_wphi)),
    )
    i_edge, j_edge = edges_from_near(near.nbr_idx, near.nbr_mask)
    volume_m3 = float(args.sc_volume_mm3) * 1e-9
    T2_edge = build_T2_edges(
        r0_flat,
        i_edge,
        j_edge,
        near_kernel=str(sc_near_kernel),
        volume_m3=volume_m3,
        subdip_n=int(args.sc_subdip_n),
    )

    sc_params = {
        "chi": float(args.sc_chi),
        "Nd": float(args.sc_Nd),
        "p0": float(args.sc_p0),
    }

    if bool(args.sc_eq):
        phi_seed = phi_init_flat if phi_init_flat is not None else phi_nom_flat
        p_init, p_stats = solve_p_easy_axis_linear_system(
            phi_seed,
            r0_flat,
            near.nbr_idx,
            near.nbr_mask,
            near_kernel=str(sc_near_kernel),
            volume_m3=volume_m3,
            p0=float(args.sc_p0),
            chi=float(args.sc_chi),
            Nd=float(args.sc_Nd),
            subdip_n=int(args.sc_subdip_n),
        )
    else:
        p_fix_val = float(args.p_fix) if args.p_fix is not None else float(args.sc_p0)
        p_init = np.full(M, p_fix_val, dtype=np.float64)
        p_stats = None

    p_fix_value = (
        None
        if bool(args.sc_eq)
        else float(args.p_fix) if args.p_fix is not None else float(args.sc_p0)
    )

    z_opt, p_opt, x_opt, trace = run_ccp_sc(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        center_idx=int(center_idx),
        wx=float(args.wx),
        wy=float(args.wy),
        wz=float(args.wz),
        reg_x=float(args.reg_x),
        reg_p=float(args.reg_p),
        reg_z=float(args.reg_z),
        tau0=float(args.tau0),
        tau_mult=float(args.tau_mult),
        tau_max=float(args.tau_max),
        iters=int(args.iters),
        tol=float(args.tol),
        tol_f=float(args.tol_f),
        z_nom=z_nom,
        z_init=z_init,
        delta_nom_deg=float(args.delta_nom_deg),
        delta_step_deg=delta_step_deg,
        p_init=p_init,
        sc_eq_enabled=bool(args.sc_eq),
        sc_params=sc_params,
        p_bounds=(float(args.pmin), float(args.pmax)),
        p_fix_value=p_fix_value,
        i_edge=i_edge,
        j_edge=j_edge,
        T2_edge=T2_edge,
        solver=str(args.solver),
        verbose=bool(args.verbose),
    )

    phi_opt, z_norm = zvec_to_phi_pnorm(z_opt)

    By = Ay @ x_opt
    By_diff = By - By[int(center_idx)]
    std_by = float(np.std(By_diff))

    p_sc_post, p_sc_stats = solve_p_easy_axis_linear_system(
        phi_opt,
        r0_flat,
        near.nbr_idx,
        near.nbr_mask,
        near_kernel=str(sc_near_kernel),
        volume_m3=volume_m3,
        p0=float(args.sc_p0),
        chi=float(args.sc_chi),
        Nd=float(args.sc_Nd),
        subdip_n=int(args.sc_subdip_n),
    )
    z_unit = phi_to_zvec(phi_opt)
    x_sc_post = _interleave_x(p_sc_post, z_unit)
    f_sc_post = _eval_objective(
        Ax,
        Ay,
        Az,
        int(center_idx),
        float(args.wx),
        float(args.wy),
        float(args.wz),
        float(args.reg_x),
        float(args.reg_p),
        float(args.reg_z),
        x_sc_post,
        p_sc_post,
        z_unit,
    )

    meta = {
        "framework": "dc",
        "dc_model": "ccp_sc_linearized",
        "sc_eq_enabled": bool(args.sc_eq),
        "sc_cfg": {
            "chi": float(args.sc_chi),
            "Nd": float(args.sc_Nd),
            "p0": float(args.sc_p0),
            "volume_mm3": float(args.sc_volume_mm3),
            "near_window": {
                "wr": int(args.sc_near_wr),
                "wz": int(args.sc_near_wz),
                "wphi": int(args.sc_near_wphi),
            },
            "near_kernel": str(sc_near_kernel),
            "subdip_n": int(args.sc_subdip_n),
        },
        "geom": {
            "R": int(args.R),
            "K": int(args.K),
            "N": int(args.N),
            "radius_m": float(args.radius_m),
            "length_m": float(args.length_m),
        },
        "roi": {
            "type": "xy_plane",
            "radius_m": float(args.roi_radius_m),
            "grid_n": int(args.roi_grid_n),
            "P": int(P),
            "center_idx": int(center_idx),
        },
        "objective": {
            "wx": float(args.wx),
            "wy": float(args.wy),
            "wz": float(args.wz),
            "reg_x": float(args.reg_x),
            "reg_p": float(args.reg_p),
            "reg_z": float(args.reg_z),
        },
        "ccp": {
            "tau0": float(args.tau0),
            "tau_mult": float(args.tau_mult),
            "tau_max": float(args.tau_max),
            "iters": int(args.iters),
            "tol": float(args.tol),
            "tol_f": float(args.tol_f),
            "delta_nom_deg": float(args.delta_nom_deg),
            "delta_step_deg": None if delta_step_deg is None else float(delta_step_deg),
            "phi0": float(args.phi0),
        },
        "p_bounds": {"pmin": float(args.pmin), "pmax": float(args.pmax)},
        "p_fix": None if p_fix_value is None else float(p_fix_value),
        "factor": float(args.factor),
        "solver": str(args.solver),
    }
    if args.init_run:
        meta["init_source"] = "lbfgs"
        meta["init_from_run"] = str(args.init_run)
        meta["init_angle_model"] = init_model

    trace_json = {
        "ccp_trace": trace,
        "std_By": float(std_by),
        "p_init_stats": p_stats,
        "p_sc_post_stats": p_sc_stats,
        "f_sc_post": float(f_sc_post),
    }

    m_flat = np.zeros((M, 3), dtype=np.float64)
    m_flat[:, 0] = x_opt[0::2]
    m_flat[:, 1] = x_opt[1::2]

    results = {
        "z_opt": np.asarray(z_opt, dtype=np.float64),
        "p_opt": np.asarray(p_opt, dtype=np.float64),
        "x_opt": np.asarray(x_opt, dtype=np.float64),
        "p_init": np.asarray(p_init, dtype=np.float64),
        "phi_nom_flat": np.asarray(phi_nom_flat, dtype=np.float64),
        "phi_opt": np.asarray(phi_opt, dtype=np.float64),
        "z_norm": np.asarray(z_norm, dtype=np.float64),
        "m_flat": np.asarray(m_flat, dtype=np.float64),
        "r0_flat": np.asarray(r0_flat, dtype=np.float64),
        "pts": np.asarray(pts, dtype=np.float64),
        "By": np.asarray(By, dtype=np.float64),
        "By_diff": np.asarray(By_diff, dtype=np.float64),
        "center_idx": np.array(int(center_idx), dtype=np.int32),
        "p_sc_post": np.asarray(p_sc_post, dtype=np.float64),
        "x_sc_post": np.asarray(x_sc_post, dtype=np.float64),
        "f_sc_post": np.array(float(f_sc_post), dtype=np.float64),
    }
    if z_init is not None:
        results["z_init"] = np.asarray(z_init, dtype=np.float64)
    if phi_init_flat is not None:
        results["phi_init_flat"] = np.asarray(phi_init_flat, dtype=np.float64)

    out_dir = Path(args.out)
    write_dc_run(out_dir, meta=meta, trace=trace_json, results=results)

    status = None
    if isinstance(trace.get("iters"), list) and trace["iters"]:
        status = trace["iters"][-1].get("status")
        last_f = trace["iters"][-1].get("f")
    else:
        last_f = float("nan")

    print(f"status={status} f={last_f} std(By_diff)={std_by:.3e} f_sc_post={float(f_sc_post):.3e}")
    print(f"center_idx={center_idx} out_dir={out_dir}")
    if delta_step_deg is not None:
        print(f"delta_step_deg={delta_step_deg} cos={math.cos(math.radians(delta_step_deg)):.6f}")


if __name__ == "__main__":
    main()
