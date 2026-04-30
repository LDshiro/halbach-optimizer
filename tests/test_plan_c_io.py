import json
from pathlib import Path

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.io import (
    FINAL_PLACEMENT_COLUMNS,
    SIMULATION_TRIAL_COLUMNS,
    SimulationTrialArtifacts,
    load_json_dict,
    read_csv_dicts,
    validate_final_placement_csv,
    write_simulation_outputs,
)
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.simulation import run_simulation_trial, summarize_comparison_results
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.assembly.work_units import assign_work_unit_ids, build_work_units
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"io_run_N{N}_R{R}_K{K}"
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
        name="plan-c-io-test",
        schema_version=1,
        generator_params={},
        description="plan c io test run",
    )
    return load_run(run_dir)


def _artifact(tmp_path: Path) -> tuple[list, SimulationTrialArtifacts, dict[str, object]]:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    slots = assign_work_unit_ids(slots, build_work_units(slots, "all_slots"))
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=11,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0005,
        measurement_noise=None,
    )
    assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
    inventory = build_cluster_inventory(magnets, assignments)
    result = run_simulation_trial(
        slots,
        magnets,
        table,
        pts,
        trial_id=0,
        seed=123,
        assignments=assignments,
        inventory=inventory,
    )
    artifact = SimulationTrialArtifacts(
        trial_id=0,
        seed=123,
        result=result,
        magnets=tuple(magnets),
        assignments=tuple(assignments),
    )
    summary = summarize_comparison_results([result])
    return slots, artifact, summary


def test_write_simulation_outputs_creates_step6_files(tmp_path: Path) -> None:
    slots, artifact, summary = _artifact(tmp_path)
    out_dir = tmp_path / "out"

    written = write_simulation_outputs(
        out_dir,
        [artifact],
        slots,
        summary,
        metadata={"engine": "linear_sensitivity"},
    )

    expected_names = {
        "simulation_summary.json",
        "simulation_trials.csv",
        "final_placement_trial_000.csv",
        "cluster_usage_trial_000.csv",
        "work_unit_summary_trial_000.csv",
        "field_metrics_trial_000.json",
        "streamlit_session_log_trial_000.jsonl",
    }
    assert {path.name for path in written.values()} == expected_names
    assert all(path.exists() for path in written.values())


def test_summary_json_and_trials_csv_are_readable(tmp_path: Path) -> None:
    slots, artifact, summary = _artifact(tmp_path)
    out_dir = tmp_path / "out"
    write_simulation_outputs(out_dir, [artifact], slots, summary)

    summary_payload = load_json_dict(out_dir / "simulation_summary.json")
    assert summary_payload["schema_version"] == 1
    assert summary_payload["units"]["length"] == "m"
    assert summary_payload["summary"]["trials"] == 1

    trial_rows = read_csv_dicts(out_dir / "simulation_trials.csv")
    assert len(trial_rows) == 1
    assert list(trial_rows[0]) == SIMULATION_TRIAL_COLUMNS
    assert trial_rows[0]["trial_id"] == "0"


def test_final_placement_csv_columns_and_duplicate_validation(tmp_path: Path) -> None:
    slots, artifact, summary = _artifact(tmp_path)
    out_dir = tmp_path / "out"
    write_simulation_outputs(out_dir, [artifact], slots, summary)
    final_path = out_dir / "final_placement_trial_000.csv"

    rows = read_csv_dicts(final_path)
    assert list(rows[0]) == FINAL_PLACEMENT_COLUMNS
    assert len(validate_final_placement_csv(final_path)) == len(slots)
    assert len({row["slot_flat_id"] for row in rows}) == len(slots)
    assert all(row["work_unit_id"] == "W_ALL" for row in rows)


def test_field_metrics_json_and_session_log_have_schema_version(tmp_path: Path) -> None:
    slots, artifact, summary = _artifact(tmp_path)
    out_dir = tmp_path / "out"
    write_simulation_outputs(out_dir, [artifact], slots, summary)

    metrics = load_json_dict(out_dir / "field_metrics_trial_000.json")
    assert metrics["schema_version"] == 1
    assert "random_baseline" in metrics
    assert "linear_sensitivity" in metrics

    log_lines = (out_dir / "streamlit_session_log_trial_000.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert json.loads(log_lines[0])["event"] == "session_started"
    assert json.loads(log_lines[-1])["event"] == "session_completed"
    assert all(json.loads(line)["schema_version"] == 1 for line in log_lines)
