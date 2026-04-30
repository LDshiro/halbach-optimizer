from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from halbach.assembly.sensitivity import compute_sensitivity_table, save_sensitivity_table
from halbach.assembly.slots import build_assembly_slots
from halbach.constants import FACTOR
from halbach.geom import RoiMode, build_roi_points
from halbach.run_io import load_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute a Plan C linear sensitivity NPZ file")
    ap.add_argument("--run", required=True, help="input Halbach run directory or results.npz")
    ap.add_argument("--out", required=True, help="output plan_c_sensitivity.npz path")
    ap.add_argument("--roi-r", type=float, required=True, help="ROI radius [m]")
    ap.add_argument(
        "--roi-step",
        type=float,
        default=0.01,
        help="ROI grid step [m] for volume-grid / volume-subsample",
    )
    ap.add_argument(
        "--roi-mode",
        choices=[
            "volume-grid",
            "volume-subsample",
            "surface-fibonacci",
            "surface-random",
        ],
        default="surface-fibonacci",
        help="ROI point generation mode",
    )
    ap.add_argument(
        "--roi-samples",
        type=int,
        default=300,
        help="sample count for surface/subsample ROI modes",
    )
    ap.add_argument("--roi-seed", type=int, default=0, help="ROI sampling seed")
    ap.add_argument(
        "--finite-difference-step",
        type=float,
        default=1e-6,
        help="finite difference step for error components",
    )
    ap.add_argument("--factor", type=float, default=FACTOR, help="dipole field factor")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_path = Path(str(args.run))
    out_path = Path(str(args.out))
    run = load_run(run_path)
    slots = build_assembly_slots(run)
    roi_mode = cast(RoiMode, str(args.roi_mode))
    roi_points = build_roi_points(
        float(args.roi_r),
        float(args.roi_step),
        mode=roi_mode,
        n_samples=int(args.roi_samples),
        seed=int(args.roi_seed),
    )
    table = compute_sensitivity_table(
        slots,
        roi_points,
        finite_difference_step=float(args.finite_difference_step),
        factor=float(args.factor),
        metadata={
            "run_path": str(run_path),
            "roi_r": float(args.roi_r),
            "roi_step": float(args.roi_step),
            "roi_mode": str(args.roi_mode),
            "roi_samples": int(args.roi_samples),
            "roi_seed": int(args.roi_seed),
        },
    )
    save_sensitivity_table(out_path, table)
    print(
        "saved Plan C sensitivity: "
        f"{out_path} slots={len(slots)} roi_points={roi_points.shape[0]} "
        f"C_shape={table.C.shape}"
    )


if __name__ == "__main__":
    main()
