from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from halbach.dc.dc_io import write_dc_run
from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.qp_relaxed_mxy import derive_phi_p_from_mxy, solve_qp_relaxed_mxy
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DC baseline: convex QP (relaxed mxy)")
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
    parser.add_argument("--pmax", type=float, required=True)
    parser.add_argument("--factor", type=float, default=1e-7)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
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

    x_opt, info = solve_qp_relaxed_mxy(
        Ax,
        Ay,
        Az,
        center_idx=center_idx,
        wx=float(args.wx),
        wy=float(args.wy),
        wz=float(args.wz),
        pmax=float(args.pmax),
        reg=float(args.reg),
        verbose=bool(args.verbose),
    )

    phi_flat, p_flat = derive_phi_p_from_mxy(x_opt)
    m_flat = np.zeros((M, 3), dtype=np.float64)
    m_flat[:, 0] = x_opt[0::2]
    m_flat[:, 1] = x_opt[1::2]

    meta = {
        "framework": "dc",
        "dc_model": "qp_relaxed_mxy",
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
        "constraints": {"type": "box", "pmax": float(args.pmax)},
        "factor": float(args.factor),
    }
    trace = {"solver_info": info}
    results = {
        "x_opt": np.asarray(x_opt, dtype=np.float64),
        "phi_flat": np.asarray(phi_flat, dtype=np.float64),
        "p_flat": np.asarray(p_flat, dtype=np.float64),
        "m_flat": np.asarray(m_flat, dtype=np.float64),
        "r0_flat": np.asarray(r0_flat, dtype=np.float64),
        "pts": np.asarray(pts, dtype=np.float64),
    }

    out_dir = Path(args.out)
    write_dc_run(out_dir, meta=meta, trace=trace, results=results)

    By = Ay @ x_opt
    By_diff = By - By[center_idx]
    std_by = float(np.std(By_diff))

    print(
        f"status={info.get('status')} obj={info.get('obj_value')} M={M} P={P} std(By_diff)={std_by:.3e}"
    )
    print(f"center_idx={center_idx} out_dir={out_dir}")


if __name__ == "__main__":
    main()
