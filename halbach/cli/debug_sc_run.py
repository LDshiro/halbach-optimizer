from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import phi_rkn_from_run
from halbach.constants import FACTOR
from halbach.run_io import load_run
from halbach.sc_debug import make_sc_debug_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate self-consistent debug bundle for a run.")
    ap.add_argument("--run", dest="run_path", required=True, help="run directory")
    ap.add_argument(
        "--out", dest="out_dir", default=None, help="output directory (default: run dir)"
    )
    ap.add_argument("--plane-z", type=float, default=None, help="plane z coordinate for ROI points")
    ap.add_argument("--roi-samples", type=int, default=64, help="ROI sample count for debug checks")
    ap.add_argument(
        "--field-scale-check", dest="field_scale_check", action="store_true", default=True
    )
    ap.add_argument(
        "--no-field-scale-check",
        dest="field_scale_check",
        action="store_false",
        help="skip field-scale checks",
    )
    ap.add_argument("--scale-factor", type=float, default=10.0, help="scale factor for checks")
    ap.add_argument(
        "--force-compute-p",
        dest="force_compute_p",
        action="store_true",
        default=False,
        help="ignore saved p and recompute",
    )
    ap.add_argument(
        "--override-sc-near-kernel",
        type=str,
        choices=["dipole", "multi-dipole", "cellavg"],
        default=None,
        help="override near_kernel in sc_cfg",
    )
    ap.add_argument(
        "--override-sc-subdip-n",
        type=int,
        default=None,
        help="override subdip_n in sc_cfg",
    )
    ap.add_argument(
        "--linear-check",
        dest="linear_check",
        action="store_true",
        default=False,
        help="enable linear system check",
    )
    ap.add_argument(
        "--linear-check-max-M",
        type=int,
        default=256,
        help="max M for linear check",
    )
    return ap.parse_args(argv)


def _build_plane_points(roi_r: float, plane_z: float, roi_samples: int) -> NDArray[np.float64]:
    n = max(2, int(np.sqrt(max(1, roi_samples))))
    xs = np.linspace(-roi_r, roi_r, n, dtype=np.float64)
    ys = np.linspace(-roi_r, roi_r, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    mask = (X * X + Y * Y + plane_z * plane_z) <= roi_r * roi_r
    pts = np.column_stack([X[mask], Y[mask], np.full(mask.sum(), plane_z)])
    return np.asarray(pts, dtype=np.float64)


def _build_r0_rkn(run: Any) -> NDArray[np.float64]:
    geom = run.geometry
    r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
    rho = r_bases[None, :] + np.asarray(geom.ring_offsets, dtype=np.float64)[:, None]
    px = rho[:, :, None] * np.asarray(geom.cth, dtype=np.float64)[None, None, :]
    py = rho[:, :, None] * np.asarray(geom.sth, dtype=np.float64)[None, None, :]
    pz = np.broadcast_to(np.asarray(geom.z_layers, dtype=np.float64)[None, :, None], px.shape)
    return np.stack([px, py, pz], axis=-1)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = Path(args.run_path)
    run = load_run(run_path)
    out_dir = Path(args.out_dir) if args.out_dir else run.run_dir

    phi_rkn = phi_rkn_from_run(run)
    r0_rkn = _build_r0_rkn(run)

    pts = None
    factor = None
    if args.plane_z is not None:
        roi_r = 0.05
        if isinstance(run.meta.get("roi"), dict) and "roi_r" in run.meta["roi"]:
            roi_r = float(run.meta["roi"]["roi_r"])
        pts = _build_plane_points(roi_r, float(args.plane_z), int(args.roi_samples))
        factor = float(FACTOR)

    sc_cfg_override: dict[str, Any] | None = None
    if args.override_sc_near_kernel is not None or args.override_sc_subdip_n is not None:
        sc_cfg_override = {}
        if args.override_sc_near_kernel is not None:
            sc_cfg_override["near_kernel"] = str(args.override_sc_near_kernel)
        if args.override_sc_subdip_n is not None:
            sc_cfg_override["subdip_n"] = int(args.override_sc_subdip_n)

    debug_dir = make_sc_debug_bundle(
        run_dir=run.run_dir,
        out_dir=out_dir,
        geom=run.geometry,
        phi_rkn=np.asarray(phi_rkn, dtype=np.float64),
        r0_rkn=np.asarray(r0_rkn, dtype=np.float64),
        pts=pts,
        factor=factor,
        field_scale_check=bool(args.field_scale_check),
        scale_factor=float(args.scale_factor),
        force_compute_p=bool(args.force_compute_p),
        sc_cfg_override=sc_cfg_override,
        linear_check=bool(args.linear_check),
        linear_check_max_M=int(args.linear_check_max_M),
    )

    report_path = debug_dir / "check_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = "PASS" if report.get("pass", False) else "FAIL"
        failures = report.get("failures", [])
        print(f"[sc_debug] {status}")
        if failures:
            print(f"failures: {failures}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
