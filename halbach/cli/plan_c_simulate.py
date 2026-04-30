from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.io import SimulationTrialArtifacts, write_simulation_outputs
from halbach.assembly.sensitivity import compute_sensitivity_table, load_sensitivity_table
from halbach.assembly.self_consistent_assignment import (
    SelfConsistentConfig,
    self_consistent_config_from_run,
)
from halbach.assembly.simulation import run_simulation_trial, summarize_comparison_results
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import EvaluationModel
from halbach.assembly.variation import generate_virtual_magnets
from halbach.constants import FACTOR
from halbach.geom import RoiMode, build_roi_points
from halbach.run_io import load_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Plan C random vs linear sensitivity simulation")
    ap.add_argument("--run", required=True, help="input Halbach run directory or results.npz")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sensitivity", default=None, help="existing plan_c_sensitivity.npz")
    ap.add_argument("--trials", type=int, default=1, help="number of simulation trials")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed")
    ap.add_argument(
        "--engine",
        choices=["linear_sensitivity", "sequential_self_consistent"],
        default="linear_sensitivity",
        help="Plan C assignment engine",
    )
    ap.add_argument(
        "--evaluation-model",
        choices=["fixed", "self_consistent_from_run"],
        default="fixed",
        help="field model used for final random/linear placement homogeneity evaluation",
    )
    ap.add_argument(
        "--sc-source",
        choices=["run", "manual"],
        default="run",
        help="source for sequential_self_consistent parameters",
    )
    ap.add_argument(
        "--sc-chi",
        type=float,
        default=0.0,
        help="sequential_self_consistent susceptibility-like chi",
    )
    ap.add_argument(
        "--sc-Nd",
        type=float,
        default=1.0 / 3.0,
        help="sequential_self_consistent demagnetization factor for manual source",
    )
    ap.add_argument(
        "--sc-p0",
        type=float,
        default=1.0,
        help="sequential_self_consistent nominal moment for manual source",
    )
    ap.add_argument(
        "--sc-volume-mm3",
        type=float,
        default=1000.0,
        help="sequential_self_consistent magnet volume for manual source [mm^3]",
    )
    ap.add_argument(
        "--sc-iters",
        type=int,
        default=30,
        help="sequential_self_consistent fixed-point iterations",
    )
    ap.add_argument(
        "--sc-omega",
        type=float,
        default=0.6,
        help="sequential_self_consistent relaxation factor for manual source",
    )
    ap.add_argument(
        "--sc-max-linear-candidates",
        type=int,
        default=8,
        help="linear top-k candidates re-evaluated by sequential_self_consistent",
    )
    ap.add_argument("--strength-mu", type=float, default=0.0, help="iid strength error mean")
    ap.add_argument("--strength-sigma", type=float, default=0.01, help="iid strength error sigma")
    ap.add_argument(
        "--direction-sigma",
        type=float,
        default=0.001,
        help="shared transverse direction error sigma",
    )
    ap.add_argument(
        "--direction-sigma-1",
        type=float,
        default=None,
        help="transverse component 1 sigma; overrides --direction-sigma",
    )
    ap.add_argument(
        "--direction-sigma-2",
        type=float,
        default=None,
        help="transverse component 2 sigma; overrides --direction-sigma",
    )
    ap.add_argument("--roi-r", type=float, default=0.05, help="ROI radius [m]")
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
        help="ROI point generation mode when --sensitivity is omitted",
    )
    ap.add_argument("--roi-samples", type=int, default=100, help="ROI sample count")
    ap.add_argument("--roi-seed", type=int, default=0, help="ROI sampling seed")
    ap.add_argument(
        "--finite-difference-step",
        type=float,
        default=1e-6,
        help="finite difference step when computing sensitivity inline",
    )
    ap.add_argument("--factor", type=float, default=FACTOR, help="dipole field factor")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.trials) <= 0:
        raise ValueError("--trials must be positive")
    engine = str(args.engine)
    include_self_consistent = engine == "sequential_self_consistent"

    run_path = Path(str(args.run))
    out_dir = Path(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(run_path)
    slots = build_assembly_slots(run)
    sc_source = str(args.sc_source)
    evaluation_model_arg = str(args.evaluation_model)
    evaluation_model: EvaluationModel = (
        "self_consistent" if evaluation_model_arg == "self_consistent_from_run" else "fixed"
    )
    self_consistent_evaluation_config = (
        self_consistent_config_from_run(
            run,
            factor=float(args.factor),
            max_linear_candidates=int(args.sc_max_linear_candidates),
            require=True,
        )
        if evaluation_model == "self_consistent"
        else None
    )
    if include_self_consistent and sc_source == "run":
        self_consistent_config = (
            self_consistent_evaluation_config
            if self_consistent_evaluation_config is not None
            else self_consistent_config_from_run(
                run,
                factor=float(args.factor),
                max_linear_candidates=int(args.sc_max_linear_candidates),
                require=True,
            )
        )
    elif include_self_consistent:
        self_consistent_config = SelfConsistentConfig(
            chi=float(args.sc_chi),
            Nd=float(args.sc_Nd),
            p0=float(args.sc_p0),
            volume_m3=float(args.sc_volume_mm3) * 1.0e-9,
            iters=int(args.sc_iters),
            omega=float(args.sc_omega),
            max_linear_candidates=int(args.sc_max_linear_candidates),
            factor=float(args.factor),
        )
    else:
        self_consistent_config = None

    if args.sensitivity is None:
        roi_mode = cast(RoiMode, str(args.roi_mode))
        roi_points = build_roi_points(
            float(args.roi_r),
            float(args.roi_step),
            mode=roi_mode,
            n_samples=int(args.roi_samples),
            seed=int(args.roi_seed),
        )
        sensitivity_table = compute_sensitivity_table(
            slots,
            roi_points,
            finite_difference_step=float(args.finite_difference_step),
            factor=float(args.factor),
            metadata={
                "run_path": str(run_path),
                "roi_r": float(args.roi_r),
                "roi_mode": str(args.roi_mode),
                "roi_samples": int(args.roi_samples),
                "engine": str(args.engine),
                "evaluation_model": evaluation_model_arg,
                "sc_source": sc_source if include_self_consistent else None,
            },
        )
    else:
        sensitivity_table = load_sensitivity_table(Path(str(args.sensitivity)))
        roi_points = sensitivity_table.roi_points

    sigma1 = (
        float(args.direction_sigma)
        if args.direction_sigma_1 is None
        else float(args.direction_sigma_1)
    )
    sigma2 = (
        float(args.direction_sigma)
        if args.direction_sigma_2 is None
        else float(args.direction_sigma_2)
    )

    artifacts: list[SimulationTrialArtifacts] = []
    for trial_id in range(int(args.trials)):
        trial_seed = int(args.seed) + trial_id
        magnets = generate_virtual_magnets(
            count=len(slots),
            seed=trial_seed,
            strength_model={
                "mode": "iid_normal",
                "mu": float(args.strength_mu),
                "sigma": float(args.strength_sigma),
            },
            direction_sigma_1=sigma1,
            direction_sigma_2=sigma2,
            measurement_noise=None,
        )
        assignments = assign_quantile_clusters(magnets)
        inventory = build_cluster_inventory(magnets, assignments)
        result = run_simulation_trial(
            slots,
            magnets,
            sensitivity_table,
            roi_points,
            trial_id=trial_id,
            seed=trial_seed,
            assignments=assignments,
            inventory=inventory,
            include_self_consistent=include_self_consistent,
            self_consistent_config=self_consistent_config,
            evaluation_model=evaluation_model,
            self_consistent_evaluation_config=self_consistent_evaluation_config,
            factor=float(args.factor),
        )
        artifacts.append(
            SimulationTrialArtifacts(
                trial_id=trial_id,
                seed=trial_seed,
                result=result,
                magnets=tuple(magnets),
                assignments=tuple(assignments),
            )
        )

    summary = summarize_comparison_results([artifact.result for artifact in artifacts])
    written = write_simulation_outputs(
        out_dir,
        artifacts,
        slots,
        summary,
        metadata={
            "engine": str(args.engine),
            "evaluation_model": evaluation_model_arg,
            "run_path": str(run_path),
            "n_slots": len(slots),
            "n_roi_points": int(roi_points.shape[0]),
            "seed": int(args.seed),
            "strength_model": {
                "mode": "iid_normal",
                "mu": float(args.strength_mu),
                "sigma": float(args.strength_sigma),
            },
            "direction_sigma_1": sigma1,
            "direction_sigma_2": sigma2,
            "self_consistent": (
                None
                if self_consistent_config is None
                else {
                    "chi": self_consistent_config.chi,
                    "Nd": self_consistent_config.Nd,
                    "p0": self_consistent_config.p0,
                    "volume_m3": self_consistent_config.volume_m3,
                    "iters": self_consistent_config.iters,
                    "omega": self_consistent_config.omega,
                    "max_linear_candidates": self_consistent_config.max_linear_candidates,
                    "source": sc_source,
                }
            ),
            "self_consistent_evaluation": (
                None
                if self_consistent_evaluation_config is None
                else {
                    "chi": self_consistent_evaluation_config.chi,
                    "Nd": self_consistent_evaluation_config.Nd,
                    "p0": self_consistent_evaluation_config.p0,
                    "volume_m3": self_consistent_evaluation_config.volume_m3,
                    "iters": self_consistent_evaluation_config.iters,
                    "omega": self_consistent_evaluation_config.omega,
                    "source": "run",
                }
            ),
        },
    )
    summary_path = written["simulation_summary"]
    print(
        "saved Plan C simulation summary: "
        f"{summary_path} trials={summary['trials']} "
        f"rms_ratio_mean={summary['rms_ratio_mean']:.6g}"
    )


if __name__ == "__main__":
    main()
