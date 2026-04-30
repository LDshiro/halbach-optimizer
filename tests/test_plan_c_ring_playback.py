from pathlib import Path

import numpy as np

from app.plan_c_ring_playback import (
    cluster_inventory_counts,
    cluster_inventory_matrix,
    discover_trial_ids,
    load_ring_trial_bundle,
    playback_state_at_step,
    plot_active_ring_polar_view,
    plot_cluster_inventory_heatmap,
    plot_mirror_pair_imbalance,
    plot_ring_metric_heatmap,
    plot_side_stack_view,
    ring_completion_steps,
    ring_metric_matrix,
)
from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.io import SimulationTrialArtifacts, write_simulation_outputs
from halbach.assembly.ring_quota import plan_ring_cluster_quotas
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.simulation import run_simulation_trial, summarize_comparison_results
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.assembly.work_units import assign_work_unit_ids, build_work_units
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run


def _write_ring_outputs(tmp_path: Path) -> tuple[Path, int]:
    run_dir = tmp_path / "ring_playback_run"
    results = generate_halbach_initial(
        N=4,
        R=1,
        end_R=None,
        end_layers_per_side=0,
        K=3,
        Lz=0.18,
        diameter_m=0.35,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-ring-playback-test",
        schema_version=1,
        generator_params={},
        description="plan c ring playback test run",
    )
    run = load_run(run_dir)
    slots = build_assembly_slots(run)
    work_units = build_work_units(slots, "ring_by_ring_outer_to_inner")
    slots = assign_work_unit_ids(slots, work_units)
    pts = sample_sphere_surface_fibonacci(4, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=22,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0005,
        measurement_noise=None,
    )
    assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
    inventory = build_cluster_inventory(magnets, assignments)
    quota_plans = plan_ring_cluster_quotas(slots, inventory)
    result = run_simulation_trial(
        slots,
        magnets,
        table,
        pts,
        trial_id=0,
        seed=44,
        assignments=assignments,
        inventory=inventory,
        random_orientation_mode="fixed_o0",
        work_units=work_units,
        cluster_pickup_policy="quota_ordered",
        quota_plans=quota_plans,
    )
    artifact = SimulationTrialArtifacts(
        trial_id=0,
        seed=44,
        result=result,
        magnets=tuple(magnets),
        assignments=tuple(assignments),
        quota_plans=quota_plans,
    )
    out_dir = tmp_path / "out"
    write_simulation_outputs(
        out_dir,
        [artifact],
        slots,
        summarize_comparison_results([result]),
        metadata={"engine": "linear_sensitivity", "work_unit_mode": "ring_by_ring_outer_to_inner"},
    )
    return out_dir, len(slots)


def test_discover_and_load_ring_trial_bundle(tmp_path: Path) -> None:
    out_dir, slot_count = _write_ring_outputs(tmp_path)

    assert discover_trial_ids(out_dir) == (0,)
    bundle = load_ring_trial_bundle(out_dir, 0)

    assert bundle.trial_id == 0
    assert bundle.summary["schema_version"] == 1
    assert len(bundle.timeline) == slot_count
    assert bundle.ring_summary
    assert bundle.ring_pair_summary
    assert bundle.quota_plan
    assert bundle.pickup_log


def test_playback_state_replays_completed_slots_and_ring_changes(tmp_path: Path) -> None:
    out_dir, _slot_count = _write_ring_outputs(tmp_path)
    bundle = load_ring_trial_bundle(out_dir, 0)

    first = playback_state_at_step(bundle, 0)
    second_ring_step = ring_completion_steps(bundle)[0] + 1
    after_change = playback_state_at_step(bundle, second_ring_step)

    assert first.placed_count == 1
    assert first.current_slot_flat_id in first.placed_slot_ids
    assert first.current_orientation_id is not None
    assert after_change.active_layer_id != first.active_layer_id
    assert len(after_change.placed_slot_ids) == second_ring_step + 1


def test_ring_metric_matrix_matches_csv_value(tmp_path: Path) -> None:
    out_dir, _slot_count = _write_ring_outputs(tmp_path)
    bundle = load_ring_trial_bundle(out_dir, 0)
    matrix = ring_metric_matrix(bundle, "mean_epsilon")
    row = bundle.ring_summary[0]
    layer_idx = matrix.y.index(int(row["layer_id"]))
    ring_idx = matrix.x.index(int(row["ring_id"]))

    assert np.isclose(matrix.z[layer_idx, ring_idx], float(row["mean_epsilon"]))


def test_cluster_inventory_and_quota_matrices_match_counts(tmp_path: Path) -> None:
    out_dir, slot_count = _write_ring_outputs(tmp_path)
    bundle = load_ring_trial_bundle(out_dir, 0)

    initial = cluster_inventory_counts(bundle, "initial")
    used = cluster_inventory_counts(bundle, "used")
    remaining = cluster_inventory_counts(bundle, "remaining")
    planned = cluster_inventory_counts(bundle, "planned")

    assert sum(initial.values()) == slot_count
    assert sum(used.values()) == slot_count
    assert sum(planned.values()) == slot_count
    assert sum(remaining.values()) == 0
    assert cluster_inventory_matrix(bundle, "used").z.sum() == slot_count
    assert cluster_inventory_matrix(bundle, "planned").z.sum() == slot_count


def test_ring_playback_plot_helpers_return_non_empty_figures(tmp_path: Path) -> None:
    out_dir, _slot_count = _write_ring_outputs(tmp_path)
    bundle = load_ring_trial_bundle(out_dir, 0)
    state = playback_state_at_step(bundle, 0)

    figures = [
        plot_side_stack_view(bundle, state),
        plot_active_ring_polar_view(bundle, state),
        plot_ring_metric_heatmap(bundle, "mean_epsilon"),
        plot_mirror_pair_imbalance(bundle, "mean_epsilon_difference"),
        plot_cluster_inventory_heatmap(bundle, "used"),
    ]

    assert all(len(fig.data) > 0 for fig in figures)
