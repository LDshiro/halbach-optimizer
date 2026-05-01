from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from app.plan_c_ring_playback import (
    ClusterMatrixMode,
    discover_trial_ids,
    load_ring_trial_bundle,
    playback_state_at_step,
    plot_active_ring_polar_view,
    plot_cluster_inventory_heatmap,
    plot_mirror_pair_imbalance,
    plot_ring_metric_heatmap,
    plot_side_stack_view,
    ring_completion_steps,
)
from app.plan_c_run_selector import (
    MANUAL_RUN_CHOICE,
    default_plan_c_child_output_dir,
    list_plan_c_run_result_choices,
    resolve_selected_run_path,
)
from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.io import SimulationTrialArtifacts, write_simulation_outputs
from halbach.assembly.ring_quota import plan_work_unit_cluster_quotas
from halbach.assembly.self_consistent_assignment import self_consistent_config_from_run
from halbach.assembly.sensitivity_cache import load_or_compute_sensitivity_table
from halbach.assembly.simulation import run_simulation_trial, summarize_comparison_results
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import (
    BuildWorkUnitMode,
    ClusterMPCConfig,
    ClusterPickupPolicy,
    EvaluationModel,
)
from halbach.assembly.ui_payload import build_summary_ui_payload
from halbach.assembly.variation import generate_virtual_magnets
from halbach.assembly.work_units import assign_work_unit_ids, build_work_units
from halbach.constants import FACTOR
from halbach.geom import build_roi_points
from halbach.run_io import load_run


def _run_ring_simulation(
    *,
    run_path: str,
    out_dir: str,
    work_unit_mode: BuildWorkUnitMode,
    cluster_pickup_policy: ClusterPickupPolicy,
    strength_count: int,
    angle_count: int,
    mpc_config: ClusterMPCConfig,
    evaluation_model_label: str,
    trials: int,
    seed: int,
    roi_r: float,
    roi_samples: int,
    strength_sigma: float,
    direction_sigma: float,
    measurement_strength_sigma: float,
    measurement_direction_sigma_1: float,
    measurement_direction_sigma_2: float,
    trial_workers: int,
) -> None:
    run = load_run(Path(run_path))
    raw_slots = build_assembly_slots(run)
    work_units = build_work_units(raw_slots, work_unit_mode)
    slots = assign_work_unit_ids(raw_slots, work_units)
    roi_points = build_roi_points(
        float(roi_r),
        0.01,
        mode="surface-fibonacci",
        n_samples=int(roi_samples),
        seed=int(seed),
    )
    sensitivity_cache = load_or_compute_sensitivity_table(
        Path(out_dir) / "sensitivity_cache",
        slots,
        roi_points,
        finite_difference_step=1.0e-6,
        factor=FACTOR,
        metadata={
            "run_path": run_path,
            "work_unit_mode": work_unit_mode,
            "cluster_pickup_policy": cluster_pickup_policy,
            "roi_mode": "surface-fibonacci",
        },
    )
    sensitivity_table = sensitivity_cache.table
    evaluation_model: EvaluationModel = (
        "self_consistent" if evaluation_model_label == "self_consistent_from_run" else "fixed"
    )
    sc_eval_config = (
        self_consistent_config_from_run(run, factor=FACTOR, max_linear_candidates=8, require=True)
        if evaluation_model == "self_consistent"
        else None
    )

    def run_one_trial(trial_id: int) -> SimulationTrialArtifacts:
        trial_seed = int(seed) + trial_id
        magnets = generate_virtual_magnets(
            count=len(slots),
            seed=trial_seed,
            strength_model={
                "mode": "iid_normal",
                "mu": 0.0,
                "sigma": float(strength_sigma),
            },
            direction_sigma_1=float(direction_sigma),
            direction_sigma_2=float(direction_sigma),
            measurement_noise={
                "strength_sigma": float(measurement_strength_sigma),
                "transverse_component_1_sigma": float(measurement_direction_sigma_1),
                "transverse_component_2_sigma": float(measurement_direction_sigma_2),
            },
        )
        assignments = assign_quantile_clusters(
            magnets,
            strength_count=int(strength_count),
            angle_count=int(angle_count),
        )
        inventory = build_cluster_inventory(magnets, assignments)
        quota_plans = plan_work_unit_cluster_quotas(slots, inventory, work_units)

        result = run_simulation_trial(
            slots,
            magnets,
            sensitivity_table,
            roi_points,
            trial_id=trial_id,
            seed=trial_seed,
            assignments=assignments,
            inventory=inventory,
            random_orientation_mode="fixed_o0",
            work_units=work_units,
            cluster_pickup_policy=cluster_pickup_policy,
            cluster_mpc_config=mpc_config if cluster_pickup_policy == "cluster_mpc" else None,
            quota_plans=quota_plans,
            evaluation_model=evaluation_model,
            self_consistent_evaluation_config=sc_eval_config,
            factor=FACTOR,
        )
        return SimulationTrialArtifacts(
            trial_id=trial_id,
            seed=trial_seed,
            result=result,
            magnets=tuple(magnets),
            assignments=tuple(assignments),
            quota_plans=quota_plans,
        )

    trial_count = int(trials)
    worker_count = min(max(1, int(trial_workers)), trial_count)
    if worker_count == 1:
        artifacts = [run_one_trial(trial_id) for trial_id in range(trial_count)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            artifacts = list(executor.map(run_one_trial, range(trial_count)))
        artifacts.sort(key=lambda artifact: artifact.trial_id)

    summary = summarize_comparison_results([artifact.result for artifact in artifacts])
    write_simulation_outputs(
        out_dir,
        artifacts,
        slots,
        summary,
        metadata={
            "engine": "linear_sensitivity",
            "evaluation_model": evaluation_model_label,
            "run_path": run_path,
            "work_unit_mode": work_unit_mode,
            "cluster_pickup_policy": cluster_pickup_policy,
            "strength_count": int(strength_count),
            "angle_count": int(angle_count),
            "measurement_strength_sigma": float(measurement_strength_sigma),
            "measurement_direction_sigma_1": float(measurement_direction_sigma_1),
            "measurement_direction_sigma_2": float(measurement_direction_sigma_2),
            "sensitivity_cache_hit": sensitivity_cache.cache_hit,
            "sensitivity_cache_key": sensitivity_cache.cache_key,
            "sensitivity_cache_path": str(sensitivity_cache.cache_path),
            "trial_workers": worker_count,
        },
    )


def _trial_file_rows(out_dir: Path, trial_id: int) -> list[dict[str, object]]:
    names = {
        "final_placement": "csv",
        "ring_summary": "csv",
        "ring_pair_summary": "csv",
        "assembly_timeline": "jsonl",
        "cluster_quota_plan": "csv",
        "cluster_pickup_log": "csv",
        "field_metrics": "json",
        "streamlit_session_log": "jsonl",
    }
    rows: list[dict[str, object]] = []
    for name, suffix in names.items():
        path = out_dir / f"{name}_trial_{trial_id:03d}.{suffix}"
        rows.append(
            {
                "file": path.name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "path": str(path),
            }
        )
    return rows


st.set_page_config(page_title="Plan C Ring Simulation", layout="wide")
st.title("Plan C Ring Simulation")

with st.sidebar:
    run_choices = list_plan_c_run_result_choices()
    if run_choices:
        selected_run = st.selectbox(
            "Run result under runs/",
            [MANUAL_RUN_CHOICE, *run_choices],
            index=1,
        )
        st.caption(f"Found {len(run_choices)} result file(s) under runs/.")
    else:
        selected_run = MANUAL_RUN_CHOICE
        st.caption("No result files found under runs/. Use a manual path.")
    manual_run_path = st.text_input("Manual run path", value="runs/demo_opt")
    run_path = resolve_selected_run_path(str(selected_run), manual_run_path)
    out_dir = st.text_input(
        "Output directory",
        value=default_plan_c_child_output_dir(run_path, "plan_c_ring"),
    )
    work_unit_mode = cast(
        BuildWorkUnitMode,
        st.selectbox(
            "Work unit mode",
            [
                "layer_by_layer_outer_to_inner",
                "auto",
                "mirror_ring_pair",
                "all_slots",
                "ring_by_ring_outer_to_inner",
            ],
            index=0,
        ),
    )
    cluster_pickup_policy = cast(
        ClusterPickupPolicy,
        st.selectbox("Cluster pickup policy", ["cluster_mpc", "quota_ordered"], index=0),
    )
    strength_count = st.number_input(
        "Strength clusters", min_value=1, max_value=50, value=10, step=1
    )
    angle_count = st.number_input("Angle clusters", min_value=1, max_value=20, value=5, step=1)
    evaluation_model_label = st.selectbox(
        "Final evaluation model",
        ["fixed", "self_consistent_from_run"],
        index=0,
    )
    trials = st.number_input("Trials", min_value=1, max_value=50, value=1, step=1)
    trial_workers = st.number_input(
        "Trial workers",
        min_value=1,
        max_value=max(1, int(trials)),
        value=min(4, max(1, int(trials))),
        step=1,
    )
    seed = st.number_input("Seed", min_value=0, value=1234, step=1)
    roi_r = st.number_input("ROI radius [m]", min_value=0.001, value=0.05, step=0.01)
    roi_samples = st.number_input("ROI samples", min_value=1, value=100, step=10)
    strength_sigma = st.number_input("Strength sigma", min_value=0.0, value=0.01, step=0.001)
    direction_sigma = st.number_input(
        "Direction sigma",
        min_value=0.0,
        value=0.001,
        step=0.0001,
        format="%.5f",
    )
    st.markdown("**Measurement noise**")
    measurement_strength_sigma = st.number_input(
        "Measurement strength sigma",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    measurement_direction_sigma_1 = st.number_input(
        "Measurement d1 sigma",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    measurement_direction_sigma_2 = st.number_input(
        "Measurement d2 sigma",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f",
    )
    st.markdown("**MPC weights**")
    lambda_field = st.number_input("lambda field", min_value=0.0, value=1.0, step=0.1)
    lambda_quota = st.number_input("lambda quota", min_value=0.0, value=1.0, step=0.1)
    lambda_ring_mean = st.number_input("lambda ring mean", min_value=0.0, value=1.0, step=0.1)
    lambda_angle = st.number_input("lambda angle", min_value=0.0, value=1.0, step=0.1)
    lambda_future = st.number_input("lambda future", min_value=0.0, value=1.0, step=0.1)
    lambda_mirror = st.number_input("lambda mirror", min_value=0.0, value=1.0, step=0.1)
    run_clicked = st.button("Run Ring Simulation", type="primary")

if run_clicked:
    try:
        with st.spinner("Running Plan C ring simulation..."):
            _run_ring_simulation(
                run_path=run_path,
                out_dir=out_dir,
                work_unit_mode=work_unit_mode,
                cluster_pickup_policy=cluster_pickup_policy,
                strength_count=int(strength_count),
                angle_count=int(angle_count),
                mpc_config=ClusterMPCConfig(
                    lambda_field=float(lambda_field),
                    lambda_quota=float(lambda_quota),
                    lambda_ring_mean=float(lambda_ring_mean),
                    lambda_angle=float(lambda_angle),
                    lambda_future=float(lambda_future),
                    lambda_mirror=float(lambda_mirror),
                ),
                evaluation_model_label=str(evaluation_model_label),
                trials=int(trials),
                seed=int(seed),
                roi_r=float(roi_r),
                roi_samples=int(roi_samples),
                strength_sigma=float(strength_sigma),
                direction_sigma=float(direction_sigma),
                measurement_strength_sigma=float(measurement_strength_sigma),
                measurement_direction_sigma_1=float(measurement_direction_sigma_1),
                measurement_direction_sigma_2=float(measurement_direction_sigma_2),
                trial_workers=int(trial_workers),
            )
        st.success("Ring simulation completed")
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))

out_path = Path(out_dir)
summary_path = out_path / "simulation_summary.json"
trial_ids = discover_trial_ids(out_path)
if not summary_path.exists() or not trial_ids:
    st.info(
        "Run a ring simulation or choose an output directory containing R9 visualization files."
    )
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
payload = build_summary_ui_payload(summary)
trial_id = int(st.selectbox("Trial", trial_ids, index=0))
bundle = load_ring_trial_bundle(out_path, trial_id)

summary_tab, playback_tab, ring_tab, inventory_tab, files_tab = st.tabs(
    ["Summary", "Assembly Playback", "Ring Error Map", "Cluster Inventory", "Trial Files"]
)

with summary_tab:
    cols = st.columns(5)
    cols[0].metric("Trials", payload["trials"] or 0)
    cols[1].metric("Engine", payload["engine"] or "linear_sensitivity")
    cols[2].metric("Evaluation", payload["evaluation_model"] or "fixed")
    cols[3].metric(
        "RMS Ratio Mean",
        "n/a" if payload["rms_ratio_mean"] is None else f"{float(payload['rms_ratio_mean']):.4g}",
    )
    cols[4].metric("Linear Improved", payload["linear_improved_count"] or 0)
    if payload["trial_rows"]:
        st.dataframe(pd.DataFrame(payload["trial_rows"]), use_container_width=True, hide_index=True)

with playback_tab:
    max_step = max(0, len(bundle.timeline) - 1)
    state_key = f"plan_c_ring_step_{trial_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    button_cols = st.columns([1, 1, 3])
    if button_cols[0].button("Prev", disabled=int(st.session_state[state_key]) <= 0):
        st.session_state[state_key] = max(0, int(st.session_state[state_key]) - 1)
    if button_cols[1].button("Next", disabled=int(st.session_state[state_key]) >= max_step):
        st.session_state[state_key] = min(max_step, int(st.session_state[state_key]) + 1)
    selected_completion = button_cols[2].selectbox(
        "Ring completion step",
        ring_completion_steps(bundle) or (0,),
        format_func=lambda value: f"step {value}",
    )
    jump_cols = st.columns([1, 5])
    if jump_cols[0].button("Jump"):
        st.session_state[state_key] = int(selected_completion)
    step = st.slider(
        "Insert step",
        min_value=0,
        max_value=max_step,
        value=int(st.session_state[state_key]),
        key=f"plan_c_ring_slider_{trial_id}",
    )
    st.session_state[state_key] = int(step)
    state = playback_state_at_step(bundle, int(step))
    info_cols = st.columns(5)
    info_cols[0].metric("Placed", state.placed_count)
    info_cols[1].metric("Layer", "-" if state.active_layer_id is None else state.active_layer_id)
    info_cols[2].metric(
        "Current Ring",
        "-" if state.active_ring_id is None else state.active_ring_id,
    )
    info_cols[3].metric("Cluster", state.current_cluster or "-")
    info_cols[4].metric(
        "Residual",
        "n/a" if state.residual_norm is None else f"{state.residual_norm:.4g}",
    )
    plot_cols = st.columns(2)
    plot_cols[0].plotly_chart(plot_side_stack_view(bundle, state), use_container_width=True)
    plot_cols[1].plotly_chart(plot_active_ring_polar_view(bundle, state), use_container_width=True)
    if state.current_event is not None:
        st.dataframe(pd.DataFrame([state.current_event]), use_container_width=True, hide_index=True)

with ring_tab:
    metric = st.selectbox(
        "Ring metric",
        ["mean_epsilon", "std_epsilon", "mean_angle_error", "std_angle_error"],
        index=0,
    )
    st.plotly_chart(plot_ring_metric_heatmap(bundle, str(metric)), use_container_width=True)
    imbalance_metric = st.selectbox(
        "Mirror imbalance",
        ["mean_epsilon_difference", "mean_angle_error_difference"],
        index=0,
    )
    imbalance_metric_key = (
        "mean_angle_error_difference"
        if str(imbalance_metric) == "mean_angle_error_difference"
        else "mean_epsilon_difference"
    )
    st.plotly_chart(
        plot_mirror_pair_imbalance(bundle, imbalance_metric_key),
        use_container_width=True,
    )
    st.dataframe(pd.DataFrame(bundle.ring_summary), use_container_width=True, hide_index=True)

with inventory_tab:
    mode = cast(
        ClusterMatrixMode,
        st.selectbox("Inventory mode", ["initial", "used", "remaining", "planned"], index=1),
    )
    st.plotly_chart(plot_cluster_inventory_heatmap(bundle, mode), use_container_width=True)
    cols = st.columns(2)
    cols[0].dataframe(pd.DataFrame(bundle.pickup_log), use_container_width=True, hide_index=True)
    cols[1].dataframe(pd.DataFrame(bundle.quota_plan), use_container_width=True, hide_index=True)

with files_tab:
    files = _trial_file_rows(out_path, trial_id)
    st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)
    for row in files:
        path = Path(str(row["path"]))
        if path.exists():
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/plain",
            )
