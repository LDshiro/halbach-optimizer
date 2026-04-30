from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.io import write_final_placement_csv
from halbach.assembly.measurement import SyntheticMeasurementProvider
from halbach.assembly.sensitivity import compute_sensitivity_table, load_sensitivity_table
from halbach.assembly.session import PlanCSession, run_session_to_completion
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.constants import FACTOR
from halbach.geom import RoiMode, build_roi_points
from halbach.run_io import load_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a Plan C step-by-step session without UI")
    ap.add_argument("--run", required=True, help="input Halbach run directory or results.npz")
    ap.add_argument("--out", required=True, help="output session directory")
    ap.add_argument("--sensitivity", default=None, help="existing plan_c_sensitivity.npz")
    ap.add_argument("--seed", type=int, default=0, help="synthetic measurement seed")
    ap.add_argument("--strength-sigma", type=float, default=0.01, help="iid strength error sigma")
    ap.add_argument("--direction-sigma", type=float, default=0.001, help="transverse error sigma")
    ap.add_argument("--roi-r", type=float, default=0.05, help="ROI radius [m]")
    ap.add_argument("--roi-step", type=float, default=0.01, help="ROI grid step [m]")
    ap.add_argument(
        "--roi-mode",
        choices=[
            "volume-grid",
            "volume-subsample",
            "surface-fibonacci",
            "surface-random",
        ],
        default="surface-fibonacci",
        help="ROI mode when --sensitivity is omitted",
    )
    ap.add_argument("--roi-samples", type=int, default=100, help="ROI sample count")
    ap.add_argument("--finite-difference-step", type=float, default=1e-6)
    ap.add_argument("--factor", type=float, default=FACTOR)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_path = Path(str(args.run))
    out_dir = Path(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(run_path)
    slots = build_assembly_slots(run)

    if args.sensitivity is None:
        roi_points = build_roi_points(
            float(args.roi_r),
            float(args.roi_step),
            mode=cast(RoiMode, str(args.roi_mode)),
            n_samples=int(args.roi_samples),
            seed=int(args.seed),
        )
        sensitivity_table = compute_sensitivity_table(
            slots,
            roi_points,
            finite_difference_step=float(args.finite_difference_step),
            factor=float(args.factor),
            metadata={"run_path": str(run_path), "session_cli": True},
        )
    else:
        sensitivity_table = load_sensitivity_table(Path(str(args.sensitivity)))

    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=int(args.seed),
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": float(args.strength_sigma)},
        direction_sigma_1=float(args.direction_sigma),
        direction_sigma_2=float(args.direction_sigma),
        measurement_noise=None,
    )
    assignments = assign_quantile_clusters(magnets)
    inventory = build_cluster_inventory(magnets, assignments)
    log_path = out_dir / "session_log.jsonl"
    session = PlanCSession(
        sensitivity_table=sensitivity_table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
        log_path=log_path,
    )
    snapshot = run_session_to_completion(session)
    placement_path = out_dir / "placement_final.csv"
    write_final_placement_csv(placement_path, slots, magnets, snapshot.placements)
    summary_path = out_dir / "session_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_path": str(run_path),
                "seed": int(args.seed),
                "placements": len(snapshot.placements),
                "state": snapshot.state,
                "session_log": str(log_path),
                "placement_final": str(placement_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"saved Plan C session: {summary_path} placements={len(snapshot.placements)}")


if __name__ == "__main__":
    main()
