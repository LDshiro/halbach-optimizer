from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.plan_c_run_selector import (
    MANUAL_RUN_CHOICE,
    default_plan_c_child_output_dir,
    list_plan_c_run_result_choices,
    resolve_selected_run_path,
)
from halbach.assembly.ui_payload import build_summary_ui_payload
from halbach.cli.plan_c_simulate import main as simulate_main


def _run_simulation(
    run_path: str,
    out_dir: str,
    trials: int,
    seed: int,
    roi_r: float,
    roi_samples: int,
    strength_sigma: float,
    direction_sigma: float,
) -> None:
    simulate_main(
        [
            "--run",
            run_path,
            "--out",
            out_dir,
            "--trials",
            str(trials),
            "--seed",
            str(seed),
            "--roi-r",
            str(roi_r),
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            str(roi_samples),
            "--strength-sigma",
            str(strength_sigma),
            "--direction-sigma",
            str(direction_sigma),
        ]
    )


st.set_page_config(page_title="Plan C Simulation", layout="wide")
st.title("Plan C Simulation")

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
        value=default_plan_c_child_output_dir(run_path, "plan_c_sim"),
    )
    trials = st.number_input("Trials", min_value=1, max_value=100, value=3, step=1)
    seed = st.number_input("Seed", min_value=0, value=1234, step=1)
    roi_r = st.number_input("ROI radius [m]", min_value=0.001, value=0.05, step=0.01)
    roi_samples = st.number_input("ROI samples", min_value=1, value=100, step=10)
    strength_sigma = st.number_input("Strength sigma", min_value=0.0, value=0.01, step=0.001)
    direction_sigma = st.number_input("Direction sigma", min_value=0.0, value=0.001, step=0.0001, format="%.5f")
    run_clicked = st.button("Run Simulation", type="primary")

if run_clicked:
    try:
        with st.spinner("Running Plan C simulation..."):
            _run_simulation(
                run_path,
                out_dir,
                int(trials),
                int(seed),
                float(roi_r),
                int(roi_samples),
                float(strength_sigma),
                float(direction_sigma),
            )
        st.success("Simulation completed")
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))

summary_path = Path(out_dir) / "simulation_summary.json"
if not summary_path.exists():
    st.info("Run a simulation or choose an output directory containing simulation_summary.json.")
    st.stop()

payload = build_summary_ui_payload(json.loads(summary_path.read_text(encoding="utf-8")))

cols = st.columns(4)
cols[0].metric("Trials", payload["trials"] or 0)
cols[1].metric("RMS Ratio Mean", "n/a" if payload["rms_ratio_mean"] is None else f"{float(payload['rms_ratio_mean']):.4g}")
cols[2].metric("Linear Improved", payload["linear_improved_count"] or 0)
cols[3].metric("Engine", payload["engine"] or "unknown")

trial_rows = payload["trial_rows"]
if trial_rows:
    st.dataframe(pd.DataFrame(trial_rows), use_container_width=True, hide_index=True)

trials_csv = Path(out_dir) / "simulation_trials.csv"
if trials_csv.exists():
    st.download_button(
        "Export trials CSV",
        data=trials_csv.read_bytes(),
        file_name="simulation_trials.csv",
        mime="text/csv",
    )
