from pathlib import Path

import numpy as np

from app.plan_c_ring_playback import (
    RingTrialBundle,
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


def test_active_layer_polar_view_shows_all_rings_in_layer(tmp_path: Path) -> None:
    pickup_log: list[dict[str, str]] = []
    timeline: list[dict[str, object]] = []
    insert_order = 0
    for ring_id in (0, 1):
        for theta_id in range(4):
            row = {
                "insert_order": str(insert_order),
                "layer_id": "0",
                "ring_id": str(ring_id),
                "theta_id": str(theta_id),
                "slot_flat_id": str(100 + insert_order),
                "physical_slot_number": str(theta_id + 1),
                "nominal_phi_rad": "0.0",
                "magnet_id": str(insert_order),
                "cluster_requested": "S00_A00",
                "epsilon_parallel": "0.001",
                "delta_perp_1": "0.0001",
                "delta_perp_2": "0.0002",
                "orientation_id": "O0",
            }
            pickup_log.append(row)
            timeline.append(
                {
                    **row,
                    "event": "insert_confirmed",
                    "work_unit_id": f"W_K000_R{ring_id:03d}",
                    "step": insert_order,
                    "ring_count_so_far": theta_id + 1,
                    "ring_mean_epsilon_so_far": 0.001,
                    "ring_mean_angle_error_so_far": 0.0002,
                }
            )
            insert_order += 1
    bundle = RingTrialBundle(
        out_dir=tmp_path,
        trial_id=0,
        summary={},
        ring_summary=[
            {
                "layer_id": "0",
                "ring_id": str(ring_id),
                "mean_epsilon": "0.0",
            }
            for ring_id in (0, 1)
        ],
        ring_pair_summary=[],
        timeline=timeline,
        quota_plan=[],
        pickup_log=pickup_log,
    )
    state = playback_state_at_step(bundle, 4)

    fig = plot_active_ring_polar_view(bundle, state)
    stack = plot_side_stack_view(bundle, state)
    completed = [trace for trace in fig.data if trace.name == "completed"]
    pending = [trace for trace in fig.data if trace.name == "pending"]
    labels = next(trace for trace in fig.data if trace.name == "frame_numbers")

    assert len(completed) == 5
    assert len(pending) == 3
    assert list(labels.text).count("1") == 2
    assert max(completed[0].x) - min(completed[0].x) > max(completed[0].y) - min(completed[0].y)
    assert np.asarray(stack.data[0].z).tolist() == [[2.0, 2.0]]
    assert fig.layout.title.text.startswith("Active Layer 0 Rings")


def test_active_layer_view_uses_real_geometry_dimensions_and_orientation_labels(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "insert_order": "0",
            "layer_id": "0",
            "ring_id": "0",
            "theta_id": "0",
            "slot_flat_id": "10",
            "physical_slot_number": "1",
            "center_x_m": "0.5",
            "center_y_m": "0.0",
            "nominal_phi_rad": str(np.pi / 2.0),
            "magnet_id": "1",
            "cluster_requested": "S00_A00",
            "epsilon_parallel": "0.001",
            "measured_epsilon_parallel": "0.0011",
            "delta_perp_1": "0.0",
            "delta_perp_2": "0.0",
            "orientation_id": "O180",
        },
        {
            "insert_order": "1",
            "layer_id": "0",
            "ring_id": "0",
            "theta_id": "1",
            "slot_flat_id": "11",
            "physical_slot_number": "2",
            "center_x_m": "0.0",
            "center_y_m": "0.5",
            "nominal_phi_rad": "0.0",
            "magnet_id": "2",
            "cluster_requested": "S00_A00",
            "epsilon_parallel": "-0.001",
            "measured_epsilon_parallel": "-0.0009",
            "delta_perp_1": "0.0",
            "delta_perp_2": "0.0",
            "orientation_id": "O90",
        },
    ]
    bundle = RingTrialBundle(
        out_dir=tmp_path,
        trial_id=0,
        summary={
            "metadata": {
                "visualization_geometry": {
                    "magnet_dimensions_m": [0.2, 0.08, 0.04],
                    "magnet_dimensions_source": "test_dimensions",
                }
            }
        },
        ring_summary=[{"layer_id": "0", "ring_id": "0", "mean_epsilon": "0.0"}],
        ring_pair_summary=[],
        timeline=[
            {
                **rows[0],
                "event": "insert_confirmed",
                "work_unit_id": "W_K000_R000",
                "step": 0,
            }
        ],
        quota_plan=[],
        pickup_log=rows,
    )
    state = playback_state_at_step(bundle, 0)

    fig = plot_active_ring_polar_view(bundle, state)
    completed = next(trace for trace in fig.data if trace.name == "completed")
    orientation = next(trace for trace in fig.data if trace.name == "orientation_patterns")
    frame_numbers = next(trace for trace in fig.data if trace.name == "frame_numbers")

    center_x = 0.5 * (max(completed.x) + min(completed.x))
    center_y = 0.5 * (max(completed.y) + min(completed.y))

    assert np.isclose(center_x, 0.5)
    assert np.isclose(center_y, 0.0)
    assert np.isclose(max(completed.x) - min(completed.x), 0.08)
    assert np.isclose(max(completed.y) - min(completed.y), 0.2)
    assert list(orientation.text) == ["3"]
    assert list(frame_numbers.text) == ["1", "2"]
    assert "real center geometry" in fig.layout.title.text
