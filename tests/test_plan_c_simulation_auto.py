import json
from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory, inventory_total_count
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.simulation import (
    run_simulation_trial,
    summarize_comparison_results,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.cli.plan_c_simulate import main as simulate_main
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"simulation_auto_run_N{N}_R{R}_K{K}"
    results = generate_halbach_initial(
        N=N,
        R=R,
        end_R=None,
        end_layers_per_side=0,
        K=K,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-simulation-auto-test",
        schema_version=1,
        generator_params={},
        description="plan c simulation auto test run",
    )
    return load_run(run_dir)


def _write_self_consistent_meta(run: RunBundle) -> None:
    assert run.meta_path is not None
    meta = dict(run.meta)
    meta["magnetization"] = {
        "model_effective": "self-consistent-easy-axis",
        "self_consistent": {
            "chi": 0.012,
            "Nd": 0.22,
            "p0": 1.3,
            "volume_mm3": 300.0,
            "iters": 3,
            "omega": 0.5,
            "near_kernel": "dipole",
            "near_window": {"wr": 0, "wz": 1, "wphi": 1},
        },
    }
    run.meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _magnets(count: int, seed: int):
    return generate_virtual_magnets(
        count=count,
        seed=seed,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.002},
        direction_sigma_1=0.001,
        direction_sigma_2=0.001,
        measurement_noise=None,
    )


def test_simulation_trial_linear_sensitivity_completes_and_is_reproducible(
    tmp_path: Path,
) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(6, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = _magnets(len(slots), seed=123)
    assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
    inventory = build_cluster_inventory(magnets, assignments)

    result_a = run_simulation_trial(
        slots,
        magnets,
        table,
        pts,
        trial_id=0,
        seed=99,
        assignments=assignments,
        inventory=inventory,
    )
    result_b = run_simulation_trial(
        slots,
        magnets,
        table,
        pts,
        trial_id=0,
        seed=99,
        assignments=assignments,
        inventory=inventory,
    )

    assert result_a.linear.assignment.placements == result_b.linear.assignment.placements
    assert result_a.random_baseline.placements == result_b.random_baseline.placements
    assert len(result_a.linear.assignment.placements) == len(slots)
    assert len(
        {placement.slot_flat_id for placement in result_a.linear.assignment.placements}
    ) == len(slots)
    assert inventory_total_count(result_a.linear.assignment.inventory) == 0
    assert result_a.linear.evaluation.B.shape == (6, 3)
    assert np.isfinite(result_a.rms_ratio_linear_over_random)


def test_simulation_summary_reports_ratio_without_requiring_improvement(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=2)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    results = []
    for trial_id in range(2):
        magnets = _magnets(len(slots), seed=200 + trial_id)
        assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
        inventory = build_cluster_inventory(magnets, assignments)
        results.append(
            run_simulation_trial(
                slots,
                magnets,
                table,
                pts,
                trial_id=trial_id,
                seed=300 + trial_id,
                assignments=assignments,
                inventory=inventory,
            )
        )

    summary = summarize_comparison_results(results)

    assert summary["trials"] == 2
    assert "rms_ratio_mean" in summary
    assert "linear_improved_count" in summary


def test_plan_c_simulate_cli_writes_summary_json(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    out_dir = tmp_path / "plan_c_sim"

    simulate_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_dir),
            "--trials",
            "2",
            "--seed",
            "7",
            "--strength-sigma",
            "0.001",
            "--direction-sigma",
            "0.0005",
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "5",
        ]
    )

    payload = json.loads((out_dir / "simulation_summary.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["engine"] == "linear_sensitivity"
    assert payload["summary"]["trials"] == 2
    assert len(payload["trials"]) == 2
    assert payload["metadata"]["n_slots"] == len(build_assembly_slots(run))
    assert (out_dir / "simulation_trials.csv").exists()
    assert (out_dir / "final_placement_trial_000.csv").exists()


def test_plan_c_simulate_cli_can_include_sequential_self_consistent(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    run = _write_generated_run(tmp_path, N=4, R=1, K=2)
    _write_self_consistent_meta(run)
    out_dir = tmp_path / "plan_c_sim_sc"

    simulate_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_dir),
            "--engine",
            "sequential_self_consistent",
            "--trials",
            "1",
            "--seed",
            "9",
            "--strength-sigma",
            "0.001",
            "--direction-sigma",
            "0.0005",
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "3",
            "--sc-source",
            "run",
            "--sc-max-linear-candidates",
            "2",
        ]
    )

    payload = json.loads((out_dir / "simulation_summary.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["engine"] == "sequential_self_consistent"
    assert payload["metadata"]["self_consistent"]["source"] == "run"
    assert payload["metadata"]["self_consistent"]["chi"] == 0.012
    assert payload["metadata"]["self_consistent"]["Nd"] == 0.22
    assert payload["metadata"]["self_consistent"]["p0"] == 1.3
    assert np.isclose(payload["metadata"]["self_consistent"]["volume_m3"], 300.0e-9)
    assert payload["metadata"]["self_consistent"]["iters"] == 3
    assert payload["metadata"]["self_consistent"]["omega"] == 0.5
    assert payload["metadata"]["self_consistent"]["backend"] == "jax"
    assert payload["metadata"]["self_consistent"]["near_kernel"] == "dipole"
    assert payload["metadata"]["self_consistent"]["near_window"] == {
        "wr": 0,
        "wz": 1,
        "wphi": 1,
    }
    assert payload["summary"]["self_consistent_trials"] == 1
    assert "rms_ratio_self_consistent_over_linear_mean" in payload["summary"]
    assert payload["trials"][0]["self_consistent_evaluated_count"] > 0
    assert "self_consistent_rms_homogeneity_ppm" in payload["trials"][0]


def test_plan_c_simulate_cli_can_evaluate_linear_with_run_self_consistent(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jax")
    run = _write_generated_run(tmp_path, N=4, R=1, K=2)
    _write_self_consistent_meta(run)
    out_dir = tmp_path / "plan_c_sim_linear_sc_eval"

    simulate_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_dir),
            "--engine",
            "linear_sensitivity",
            "--evaluation-model",
            "self_consistent_from_run",
            "--trials",
            "1",
            "--seed",
            "11",
            "--strength-sigma",
            "0.001",
            "--direction-sigma",
            "0.0005",
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "3",
        ]
    )

    payload = json.loads((out_dir / "simulation_summary.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["engine"] == "linear_sensitivity"
    assert payload["metadata"]["evaluation_model"] == "self_consistent_from_run"
    assert payload["metadata"]["self_consistent"] is None
    assert payload["metadata"]["self_consistent_evaluation"]["source"] == "run"
    assert payload["metadata"]["self_consistent_evaluation"]["backend"] == "jax"
    assert payload["metadata"]["self_consistent_evaluation"]["near_kernel"] == "dipole"
    assert payload["metadata"]["self_consistent_evaluation"]["chi"] == 0.012
    assert payload["summary"]["trials"] == 1
    assert "self_consistent_trials" not in payload["summary"]
    assert payload["trials"][0]["linear_B0_norm"] > 0.0
