from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from halbach.dc.ccp_unitcircle import (
    make_phi_nominal_halbach,
    run_ccp_unitcircle,
    zvec_to_phi_pnorm,
)
from halbach.dc.dc_io import write_dc_run
from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.init_from_lbfgs import load_phi_flat_from_lbfgs_run
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DC CCP: unit-circle angle optimization")
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
    parser.add_argument("--p-mag", type=float, required=True)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--delta-nom-deg", type=float, default=5.0)
    parser.add_argument("--delta-step-deg", type=float, default=2.0)
    parser.add_argument("--tau0", type=float, default=1e-4)
    parser.add_argument("--tau-mult", type=float, default=1.2)
    parser.add_argument("--tau-max", type=float, default=1e-1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--tol-f", type=float, default=1e-9)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--solver", type=str, default="ECOS")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--init-run", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
    Axd = float(args.p_mag) * Ax
    Ayd = float(args.p_mag) * Ay
    Azd = float(args.p_mag) * Az

    phi_rkn = make_phi_nominal_halbach(
        N=int(args.N), K=int(args.K), R=int(args.R), phi0=float(args.phi0)
    )
    phi_nom_flat = phi_rkn.reshape(-1)

    delta_step = float(args.delta_step_deg)
    delta_step_deg = None if delta_step < 0.0 else float(delta_step)

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
        from halbach.dc.ccp_unitcircle import phi_to_zvec

        z_init = phi_to_zvec(phi_init_flat)

    z_opt, trace = run_ccp_unitcircle(
        Axd=Axd,
        Ayd=Ayd,
        Azd=Azd,
        center_idx=int(center_idx),
        wx=float(args.wx),
        wy=float(args.wy),
        wz=float(args.wz),
        reg=float(args.reg),
        tau0=float(args.tau0),
        tau_mult=float(args.tau_mult),
        tau_max=float(args.tau_max),
        iters=int(args.iters),
        tol=float(args.tol),
        tol_f=float(args.tol_f),
        phi_nom_flat=phi_nom_flat,
        delta_nom_deg=float(args.delta_nom_deg),
        delta_step_deg=delta_step_deg,
        z_init=z_init,
        solver=str(args.solver),
        verbose=bool(args.verbose),
    )

    phi_opt, norm_opt = zvec_to_phi_pnorm(z_opt)
    x_opt = float(args.p_mag) * z_opt
    m_flat = np.zeros((M, 3), dtype=np.float64)
    m_flat[:, 0] = x_opt[0::2]
    m_flat[:, 1] = x_opt[1::2]

    By = Ayd @ z_opt
    By_diff = By - By[int(center_idx)]
    std_by = float(np.std(By_diff))

    norms = np.sqrt(z_opt[0::2] ** 2 + z_opt[1::2] ** 2)
    norm_min = float(np.min(norms)) if norms.size else 0.0
    norm_mean = float(np.mean(norms)) if norms.size else 0.0
    norm_max = float(np.max(norms)) if norms.size else 0.0

    meta = {
        "framework": "dc",
        "dc_model": "ccp_unitcircle_angle",
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
            "reg": float(args.reg),
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
            "p_mag": float(args.p_mag),
            "phi0": float(args.phi0),
        },
        "factor": float(args.factor),
    }
    if args.init_run:
        meta["init_source"] = "lbfgs"
        meta["init_from_run"] = str(args.init_run)
        meta["init_angle_model"] = init_model
    trace_json = {
        "ccp_trace": trace,
        "std_By": float(std_by),
    }
    results = {
        "z_opt": np.asarray(z_opt, dtype=np.float64),
        "x_opt": np.asarray(x_opt, dtype=np.float64),
        "phi_nom_flat": np.asarray(phi_nom_flat, dtype=np.float64),
        "phi_opt": np.asarray(phi_opt, dtype=np.float64),
        "norm_opt": np.asarray(norm_opt, dtype=np.float64),
        "m_flat": np.asarray(m_flat, dtype=np.float64),
        "r0_flat": np.asarray(r0_flat, dtype=np.float64),
        "pts": np.asarray(pts, dtype=np.float64),
        "By": np.asarray(By, dtype=np.float64),
        "By_diff": np.asarray(By_diff, dtype=np.float64),
        "center_idx": np.array(int(center_idx), dtype=np.int32),
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

    print(
        "status={} f={} std(By_diff)={:.3e} norm_min={:.3f} norm_mean={:.3f} norm_max={:.3f}".format(
            status,
            trace["iters"][-1]["f"] if trace.get("iters") else float("nan"),
            std_by,
            norm_min,
            norm_mean,
            norm_max,
        )
    )
    print(f"center_idx={center_idx} out_dir={out_dir}")
    if delta_step_deg is not None:
        print(f"delta_step_deg={delta_step_deg} cos={math.cos(math.radians(delta_step_deg)):.6f}")


if __name__ == "__main__":
    main()
