from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.plan_c_run_selector import (
    MANUAL_RUN_CHOICE,
    default_plan_c_child_output_dir,
    list_plan_c_run_result_choices,
    resolve_selected_run_path,
)
from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory
from halbach.assembly.measurement import ManualMeasurementProvider, SyntheticMeasurementProvider
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.session import PlanCSession
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import MagnetError, VirtualMagnet
from halbach.assembly.ui_payload import build_session_ui_payload
from halbach.assembly.variation import generate_virtual_magnets
from halbach.geom import build_roi_points
from halbach.run_io import load_run


def _start_session(
    run_path: str,
    log_path: str,
    mode: str,
    seed: int,
    roi_r: float,
    roi_samples: int,
    strength_sigma: float,
    direction_sigma: float,
) -> dict[str, object]:
    run = load_run(Path(run_path))
    slots = build_assembly_slots(run)
    roi_points = build_roi_points(
        float(roi_r),
        0.01,
        mode="surface-fibonacci",
        n_samples=int(roi_samples),
        seed=int(seed),
    )
    table = compute_sensitivity_table(slots, roi_points, finite_difference_step=1e-6)

    if mode == "assembly_real":
        provider = ManualMeasurementProvider()
        assignments = None
        inventory = None
        magnets = []
    else:
        magnets = generate_virtual_magnets(
            count=len(slots),
            seed=int(seed),
            strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": float(strength_sigma)},
            direction_sigma_1=float(direction_sigma),
            direction_sigma_2=float(direction_sigma),
            measurement_noise=None,
        )
        assignments = assign_quantile_clusters(magnets)
        inventory = build_cluster_inventory(magnets, assignments)
        provider = SyntheticMeasurementProvider(magnets)

    session = PlanCSession(
        sensitivity_table=table,
        provider=provider,
        assignments=assignments,
        inventory=inventory,
        log_path=log_path,
    )
    return {
        "session": session,
        "slots": slots,
        "assignments": assignments,
        "inventory": inventory,
        "provider": provider,
        "manual_next_id": 0,
        "magnets": magnets,
    }


st.set_page_config(page_title="Plan C Assembly", layout="wide")
st.title("Plan C Assembly")

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
    log_path = st.text_input(
        "Session log",
        value=str(Path(default_plan_c_child_output_dir(run_path, "plan_c_session")) / "session_log.jsonl"),
    )
    mode = st.selectbox(
        "Mode",
        ["simulation_step_by_step", "assembly_real"],
        index=0,
    )
    seed = st.number_input("Seed", min_value=0, value=1234, step=1)
    roi_r = st.number_input("ROI radius [m]", min_value=0.001, value=0.05, step=0.01)
    roi_samples = st.number_input("ROI samples", min_value=1, value=80, step=10)
    strength_sigma = st.number_input("Strength sigma", min_value=0.0, value=0.01, step=0.001)
    direction_sigma = st.number_input("Direction sigma", min_value=0.0, value=0.001, step=0.0001, format="%.5f")
    if st.button("Start Session", type="primary"):
        try:
            with st.spinner("Preparing session..."):
                st.session_state["plan_c_assembly"] = _start_session(
                    run_path,
                    log_path,
                    mode,
                    int(seed),
                    float(roi_r),
                    int(roi_samples),
                    float(strength_sigma),
                    float(direction_sigma),
                )
            st.success("Session started")
        except Exception as exc:  # pragma: no cover - UI safety net
            st.error(str(exc))

ctx = st.session_state.get("plan_c_assembly")
if ctx is None:
    st.info("Start a session from the sidebar.")
    st.stop()

session: PlanCSession = ctx["session"]
provider = ctx["provider"]

if mode == "assembly_real" and isinstance(provider, ManualMeasurementProvider):
    with st.expander("Manual Measurement", expanded=session.sub_state == "WAIT_FOR_MAGNET_MEASUREMENT"):
        cols = st.columns(5)
        magnet_id = cols[0].number_input("Magnet ID", min_value=0, value=int(ctx["manual_next_id"]), step=1)
        eps = cols[1].number_input("epsilon", value=0.0, step=0.001, format="%.6f")
        d1 = cols[2].number_input("delta 1", value=0.0, step=0.0001, format="%.6f")
        d2 = cols[3].number_input("delta 2", value=0.0, step=0.0001, format="%.6f")
        quality = cols[4].number_input("quality", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        if st.button("Submit Measurement"):
            error = MagnetError(
                epsilon_parallel=float(eps),
                delta_perp_1=float(d1),
                delta_perp_2=float(d2),
            )
            provider.submit_magnet(
                VirtualMagnet(
                    magnet_id=int(magnet_id),
                    true_error=error,
                    measured_error=error,
                    quality=float(quality),
                )
            )
            ctx["manual_next_id"] = int(magnet_id) + 1
            st.success("Measurement queued")

buttons = st.columns(5)
if buttons[0].button("Next"):
    try:
        session.step()
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))
if buttons[1].button("Confirm"):
    try:
        session.confirm_insert()
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))
if buttons[2].button("Undo"):
    try:
        session.undo_last_insert()
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))
if buttons[3].button("Pause"):
    session.pause()
if buttons[4].button("Resume From Log"):
    try:
        ctx["session"] = PlanCSession.resume_from_log(
            log_path,
            sensitivity_table=session.sensitivity_table,
            provider=provider,
            assignments=ctx["assignments"],
            inventory=ctx["inventory"],
        )
        session = ctx["session"]
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(str(exc))

payload = build_session_ui_payload(
    session.snapshot(),
    ctx["slots"],
    mode=mode,
    assignments=ctx["assignments"],
    inventory=session.current_inventory,
    log_path=log_path,
)

top = st.columns(5)
top[0].metric("State", payload["state"])
top[1].metric("Sub State", payload["sub_state"])
top[2].metric("Remaining", payload["remaining_slot_count"])
top[3].metric("Placed", payload["placed_count"])
top[4].metric("Residual", f"{float(payload['residual_norm']):.4g}")

rec = st.columns(4)
rec[0].metric("Ring", payload["recommended_ring_id"] if payload["recommended_ring_id"] is not None else "-")
rec[1].metric("Slot", payload["recommended_physical_slot_number"] if payload["recommended_physical_slot_number"] is not None else "-")
rec[2].metric("Orientation", payload["recommended_orientation_id"] or "-")
rec[3].metric("Cluster", payload["next_cluster_id"] or "-")

if payload["orientation_instruction"]:
    st.info(str(payload["orientation_instruction"]))

st.dataframe(pd.DataFrame(payload["slot_rows"]), use_container_width=True, hide_index=True)
