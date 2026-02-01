from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from halbach.run_types import RunBundle
    from halbach.viz2d import ErrorMap2D

PlotlyMode = Literal["fast", "pretty", "cubes", "cubes_arrows"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _scan_runs(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        return []
    candidates: list[str] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "results.npz").is_file() or any(entry.glob("*results*.npz")):
            try:
                candidates.append(str(entry.relative_to(ROOT)))
            except ValueError:
                candidates.append(str(entry))
    return candidates


def _resolve_path(selected: str, manual: str) -> str:
    manual_clean = manual.strip()
    if manual_clean:
        return manual_clean
    return selected.strip()


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", tag.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _default_out_dir(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"{stamp}_{_sanitize_tag(tag)}"


def _resolve_results_path(path: Path) -> Path:
    if path.is_dir():
        primary = path / "results.npz"
        if primary.is_file():
            return primary
        matches = sorted(path.glob("*results*.npz"))
        if not matches:
            raise FileNotFoundError(f"No results .npz found in {path}")
        if len(matches) > 1:
            items = ", ".join(str(p) for p in matches)
            raise ValueError(f"Multiple results files found in {path}: {items}")
        return matches[0]
    if path.is_file() and path.suffix == ".npz":
        return path
    raise FileNotFoundError(f"Run path not found: {path}")


def _results_mtime(path: Path) -> float:
    results_path = _resolve_results_path(path)
    return float(results_path.stat().st_mtime)


def _meta_mtime(path: Path) -> float:
    if path.is_dir():
        primary = path / "meta.json"
        if primary.is_file():
            return float(primary.stat().st_mtime)
        matches = sorted(path.glob("*meta*.json"))
        if len(matches) == 1:
            return float(matches[0].stat().st_mtime)
        return 0.0
    if path.is_file() and path.suffix == ".npz":
        candidate = path.with_name("meta.json")
        if candidate.is_file():
            return float(candidate.stat().st_mtime)
        matches = sorted(path.parent.glob("*meta*.json"))
        if len(matches) == 1:
            return float(matches[0].stat().st_mtime)
    return 0.0


def _magnetization_cache_key(meta: dict[str, object]) -> str:
    magnetization = meta.get("magnetization", {})
    try:
        return json.dumps(magnetization, sort_keys=True, default=str)
    except Exception:
        return ""


def _jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


def _apply_pending_selection_updates() -> None:
    pending_init_select = st.session_state.pop("pending_init_select", None)
    if pending_init_select is not None:
        st.session_state["init_select"] = pending_init_select
    pending_init_path = st.session_state.pop("pending_init_path", None)
    if pending_init_path is not None:
        st.session_state["init_path"] = pending_init_path
    pending_opt_select = st.session_state.pop("pending_opt_select", None)
    if pending_opt_select is not None:
        st.session_state["opt_select"] = pending_opt_select
    pending_opt_path = st.session_state.pop("pending_opt_path", None)
    if pending_opt_path is not None:
        st.session_state["opt_path"] = pending_opt_path


@st.cache_data(show_spinner=False)
def _cached_load_run(path_text: str, mtime: float, _meta_mtime: float) -> RunBundle:
    from halbach.run_io import load_run

    return load_run(Path(path_text))


@st.cache_data(show_spinner=False)
def _cached_error_map(
    path_text: str, mtime: float, _meta_key: str, roi_r: float, step: float
) -> tuple[ErrorMap2D, dict[str, object]]:
    from halbach.run_io import load_run
    from halbach.viz2d import compute_error_map_ppm_plane_with_debug

    run = load_run(Path(path_text))
    return compute_error_map_ppm_plane_with_debug(
        run, plane="xy", coord0=0.0, roi_r=roi_r, step=step
    )


@st.cache_data(show_spinner=False)
def _cached_b0(path_text: str, mtime: float, _meta_key: str) -> float:
    from halbach.angles_runtime import phi_rkn_from_run
    from halbach.constants import FACTOR, m0, phi0
    from halbach.magnetization_runtime import (
        compute_b_and_b0_from_m_flat,
        compute_m_flat_from_run,
        get_magnetization_config_from_meta,
    )
    from halbach.physics import compute_B_and_B0, compute_B_and_B0_phi_rkn
    from halbach.run_io import load_run

    run = load_run(Path(path_text))
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    model_effective, _sc_cfg = get_magnetization_config_from_meta(run.meta)
    if model_effective == "self-consistent-easy-axis":
        phi_rkn = phi_rkn_from_run(run, phi0=phi0)
        r_bases = np.asarray(run.results.r_bases, dtype=np.float64)
        rho = r_bases[None, :] + np.asarray(run.geometry.ring_offsets, dtype=np.float64)[:, None]
        px = rho[:, :, None] * np.asarray(run.geometry.cth, dtype=np.float64)[None, None, :]
        py = rho[:, :, None] * np.asarray(run.geometry.sth, dtype=np.float64)[None, None, :]
        pz = np.broadcast_to(
            np.asarray(run.geometry.z_layers, dtype=np.float64)[None, :, None], px.shape
        )
        r0_rkn = np.stack([px, py, pz], axis=-1)
        m_flat, _debug = compute_m_flat_from_run(run.run_dir, run.geometry, phi_rkn, r0_rkn)
        r0_flat = r0_rkn.reshape(-1, 3)
        _, _, _, B0x, B0y, B0z = compute_b_and_b0_from_m_flat(m_flat, r0_flat, pts, factor=FACTOR)
    else:
        if run.meta.get("angle_model", "legacy-alpha") == "legacy-alpha":
            _, _, _, B0x, B0y, B0z = compute_B_and_B0(
                run.results.alphas,
                run.results.r_bases,
                run.geometry.theta,
                run.geometry.sin2,
                run.geometry.cth,
                run.geometry.sth,
                run.geometry.z_layers,
                run.geometry.ring_offsets,
                pts,
                FACTOR,
                phi0,
                m0,
            )
        else:
            phi_rkn = phi_rkn_from_run(run, phi0=phi0)
            _, _, _, B0x, B0y, B0z = compute_B_and_B0_phi_rkn(
                phi_rkn,
                run.results.r_bases,
                run.geometry.cth,
                run.geometry.sth,
                run.geometry.z_layers,
                run.geometry.ring_offsets,
                pts,
                FACTOR,
                m0,
            )
    return float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))


def _try_load(
    path_text: str,
) -> tuple[RunBundle | None, float | None, float | None, str | None]:
    if not path_text:
        return None, None, None, None
    try:
        mtime = _results_mtime(Path(path_text))
        meta_mtime = _meta_mtime(Path(path_text))
        run = _cached_load_run(path_text, mtime, meta_mtime)
        return run, mtime, meta_mtime, None
    except Exception as exc:
        return None, None, None, str(exc)


def _ppm_stats(m: ErrorMap2D) -> tuple[float, float]:
    return float(np.nanmean(m.ppm)), float(np.nanmax(np.abs(m.ppm)))


def _fmt_num(value: object, fmt: str = ".4f") -> str:
    if isinstance(value, int | float | np.floating):
        return f"{float(value):{fmt}}"
    return "n/a"


def _sample_colorscale(name: str, levels: int) -> list[tuple[float, str]]:
    import plotly.colors as pc

    steps = max(2, int(levels))
    scale = pc.get_colorscale(name)
    values = [i / (steps - 1) for i in range(steps)]
    colors = pc.sample_colorscale(scale, values)
    return list(zip(values, colors, strict=True))


def _plot_error_map(
    m: ErrorMap2D, vmin: float, vmax: float, contour_level: float, title: str
) -> go.Figure:
    from halbach.viz2d import contour_levels_ppm

    xs_mm = m.xs * 1000.0
    ys_mm = m.ys * 1000.0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs_mm,
            y=ys_mm,
            z=m.ppm,
            zmin=vmin,
            zmax=vmax,
            zmid=0.0,
            colorscale=_sample_colorscale("RdBu", 50),
            colorbar={"title": "ppm"},
        )
    )
    level_neg, level_pos = contour_levels_ppm(contour_level)
    contour_steps = 50
    contour_size = (level_pos - level_neg) / contour_steps
    fig.add_trace(
        go.Contour(
            x=xs_mm,
            y=ys_mm,
            z=m.ppm,
            contours={"start": level_neg, "end": level_pos, "size": contour_size},
            line={"color": "black", "width": 1},
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        height=420,
        margin={"l": 40, "r": 10, "b": 40, "t": 40},
    )
    return fig


def _plot_cross_section(m: ErrorMap2D, vmin: float, vmax: float, title: str) -> go.Figure:
    from halbach.viz2d import extract_cross_section_y0

    line = extract_cross_section_y0(m)
    x_mm = line.x * 1000.0
    fig = go.Figure(go.Scatter(x=x_mm, y=line.ppm, mode="lines", line={"color": "#1f77b4"}))
    fig.update_layout(
        title=f"{title} (y0={line.y0 * 1000.0:.2f} mm)",
        xaxis_title="x (mm)",
        yaxis_title="ppm",
        yaxis={"range": [vmin, vmax]},
        height=240,
        margin={"l": 40, "r": 10, "b": 40, "t": 40},
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Halbach Run Viewer", layout="wide")
    st.title("Halbach Run Viewer")

    runs_dir = ROOT / "runs"
    candidates = _scan_runs(runs_dir)

    if "opt_job" not in st.session_state:
        st.session_state["opt_job"] = None
    if "gen_run_code" not in st.session_state:
        st.session_state["gen_run_code"] = None
        st.session_state["gen_run_output"] = ""
        st.session_state["gen_run_dir"] = ""
    if "flash_message" not in st.session_state:
        st.session_state["flash_message"] = ""

    _apply_pending_selection_updates()

    with st.sidebar:
        st.header("Run Selection")
        if candidates:
            st.caption(f"Found {len(candidates)} run(s) under {runs_dir}")
        else:
            st.caption(f"No runs found under {runs_dir}")
        init_select = st.selectbox("Initial run (runs/)", [""] + candidates, key="init_select")
        init_path_text = st.text_input("Initial run path", value="", key="init_path")
        opt_select = st.selectbox("Optimized run (runs/)", [""] + candidates, key="opt_select")
        opt_path_text = st.text_input("Optimized run path", value="", key="opt_path")
        if st.button("Reload cache"):
            st.cache_data.clear()

        st.header("2D Settings")
        roi_r = float(
            st.number_input(
                "ROI radius (m)",
                min_value=0.001,
                max_value=0.2,
                value=0.14,
                step=0.001,
                format="%.4f",
            )
        )
        step = float(
            st.number_input(
                "Step (m)",
                min_value=0.001,
                max_value=0.02,
                value=0.002,
                step=0.001,
                format="%.4f",
            )
        )
        auto_limit = st.checkbox("Auto ppm limit", value=False)
        ppm_limit: float | None
        if auto_limit:
            ppm_limit = None
        else:
            ppm_limit = float(
                st.number_input(
                    "PPM limit", min_value=5.0, max_value=20000.0, value=5000.0, step=500.0
                )
            )
        contour_level = float(
            st.number_input(
                "Contour level (ppm)", min_value=1.0, max_value=20000.0, value=1000.0, step=100.0
            )
        )

        st.header("3D Settings")
        view_target = st.radio("3D run", ["initial", "optimized"], index=1)
        mode = cast(
            PlotlyMode, st.selectbox("Mode", ["fast", "pretty", "cubes", "cubes_arrows"], index=0)
        )
        stride = int(st.number_input("Stride", min_value=1, max_value=64, value=2, step=1))
        hide_x_negative = st.checkbox("Hide x < 0", value=False)
        magnet_size_mm = float(
            st.number_input("Magnet size (mm)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        )
        magnet_thickness_mm = float(
            st.number_input(
                "Magnet thickness (mm)", min_value=0.5, max_value=20.0, value=10.0, step=0.5
            )
        )
        arrow_length_mm = float(
            st.number_input(
                "Arrow length (mm)",
                min_value=1.0,
                max_value=200.0,
                value=30.0,
                step=1.0,
            )
        )
        arrow_head_angle_deg = float(
            st.number_input(
                "Arrow head angle (deg)",
                min_value=5.0,
                max_value=90.0,
                value=30.0,
                step=1.0,
            )
        )

    init_path = _resolve_path(init_select, init_path_text)
    opt_path = _resolve_path(opt_select, opt_path_text)

    init_run, init_mtime, _init_meta_mtime, init_err = _try_load(init_path)
    opt_run, opt_mtime, _opt_meta_mtime, opt_err = _try_load(opt_path)

    if init_err:
        st.error(f"Initial run load error: {init_err}")
    if opt_err:
        st.error(f"Optimized run load error: {opt_err}")

    init_map: ErrorMap2D | None = None
    opt_map: ErrorMap2D | None = None
    init_debug: dict[str, object] | None = None
    opt_debug: dict[str, object] | None = None
    if init_run is not None and init_mtime is not None:
        try:
            init_key = _magnetization_cache_key(init_run.meta)
            init_map, init_debug = _cached_error_map(init_path, init_mtime, init_key, roi_r, step)
        except Exception as exc:
            st.error(f"Initial 2D map error: {exc}")
    if opt_run is not None and opt_mtime is not None:
        try:
            opt_key = _magnetization_cache_key(opt_run.meta)
            opt_map, opt_debug = _cached_error_map(opt_path, opt_mtime, opt_key, roi_r, step)
        except Exception as exc:
            st.error(f"Optimized 2D map error: {exc}")

    vmin = vmax = 0.0
    if init_map is not None and opt_map is not None:
        from halbach.viz2d import common_ppm_limits

        vmin, vmax = common_ppm_limits([init_map, opt_map], limit_ppm=ppm_limit, symmetric=True)

    tabs = st.tabs(["Overview", "2D", "3D", "Optimize"])

    with tabs[0]:
        st.subheader("Runs")
        cols = st.columns(2)

        def render_summary(
            run: RunBundle | None,
            m: ErrorMap2D | None,
            label: str,
            debug: dict[str, object] | None,
        ) -> None:
            if run is None:
                st.info(f"{label} run is not loaded.")
                return
            b0 = None
            try:
                mtime = _results_mtime(run.results_path)
                meta_key = _magnetization_cache_key(run.meta)
                b0 = _cached_b0(str(run.results_path), mtime, meta_key)
            except Exception:
                b0 = None
            st.markdown(f"**{label} name**: {run.name}")
            st.markdown(f"**Results**: `{run.results_path}`")
            st.markdown(f"**Meta**: `{run.meta_path}`")
            if b0 is not None:
                st.metric("B0_T (mT)", f"{b0 * 1e3:.3f}")
            rb_min = float(np.min(run.results.r_bases))
            rb_max = float(np.max(run.results.r_bases))
            al_min = float(np.min(run.results.alphas))
            al_max = float(np.max(run.results.alphas))
            st.write(f"r_bases min/max: {rb_min:.6f}, {rb_max:.6f}")
            st.write(f"alphas min/max: {al_min:.6f}, {al_max:.6f}")
            if m is not None:
                ppm_mean, ppm_maxabs = _ppm_stats(m)
                st.write(f"ppm mean: {ppm_mean:.2f}")
                st.write(f"ppm max|.|: {ppm_maxabs:.2f}")
            if debug is not None and debug.get("model_effective") == "self-consistent-easy-axis":
                st.markdown("**Self-consistent diagnostics**")
                st.write(
                    f"near_kernel: {debug.get('sc_near_kernel')} "
                    f"subdip_n: {debug.get('sc_subdip_n')} "
                    f"near_window: {debug.get('sc_near_window')}"
                )
                st.write(
                    f"iters: {debug.get('sc_iters')} omega: {debug.get('sc_omega')} "
                    f"chi: {debug.get('sc_chi')} Nd: {debug.get('sc_Nd')}"
                )
                st.write(
                    f"p stats (min/max/mean/std/rel_std): "
                    f"{_fmt_num(debug.get('sc_p_min'))} / "
                    f"{_fmt_num(debug.get('sc_p_max'))} / "
                    f"{_fmt_num(debug.get('sc_p_mean'))} / "
                    f"{_fmt_num(debug.get('sc_p_std'))} / "
                    f"{_fmt_num(debug.get('sc_p_rel_std'))}"
                )

        with cols[0]:
            render_summary(init_run, init_map, "Initial", init_debug)
        with cols[1]:
            render_summary(opt_run, opt_map, "Optimized", opt_debug)

        if init_run is not None and opt_run is not None:
            st.subheader("Delta (optimized - initial)")
            if init_run.results.alphas.shape == opt_run.results.alphas.shape:
                dalphas = opt_run.results.alphas - init_run.results.alphas
                dr_bases = opt_run.results.r_bases - init_run.results.r_bases
                st.write(
                    f"alphas delta min/max: {float(np.min(dalphas)):.6f}, "
                    f"{float(np.max(dalphas)):.6f}"
                )
                st.write(
                    f"r_bases delta min/max: {float(np.min(dr_bases)):.6f}, "
                    f"{float(np.max(dr_bases)):.6f}"
                )
            else:
                st.write("Run shapes differ; delta summary skipped.")

    with tabs[1]:
        st.subheader("2D Error Maps (ppm)")
        if init_map is None or opt_map is None:
            st.info("Both initial and optimized runs are required for 2D comparison.")
        else:
            map_cols = st.columns(2)
            with map_cols[0]:
                fig_init = _plot_error_map(init_map, vmin, vmax, contour_level, "Initial")
                st.plotly_chart(fig_init, use_container_width=True)
            with map_cols[1]:
                fig_opt = _plot_error_map(opt_map, vmin, vmax, contour_level, "Optimized")
                st.plotly_chart(fig_opt, use_container_width=True)

            line_cols = st.columns(2)
            with line_cols[0]:
                fig_line_init = _plot_cross_section(init_map, vmin, vmax, "Initial y=0")
                st.plotly_chart(fig_line_init, use_container_width=True)
            with line_cols[1]:
                fig_line_opt = _plot_cross_section(opt_map, vmin, vmax, "Optimized y=0")
                st.plotly_chart(fig_line_opt, use_container_width=True)

    with tabs[2]:
        st.subheader("3D Magnet View")
        view_tabs = st.tabs(["Single", "Compare"])
        with view_tabs[0]:
            target_run = init_run if view_target == "initial" else opt_run
            if target_run is None:
                st.info("Select a run for 3D rendering.")
            else:
                from halbach.viz3d import build_magnet_figure

                fig = build_magnet_figure(
                    target_run,
                    stride=stride,
                    hide_x_negative=hide_x_negative,
                    mode=mode,
                    magnet_size_m=magnet_size_mm / 1000.0,
                    magnet_thickness_m=magnet_thickness_mm / 1000.0,
                    arrow_length_m=arrow_length_mm / 1000.0,
                    arrow_head_angle_deg=arrow_head_angle_deg,
                    height=700,
                )
                st.plotly_chart(fig, use_container_width=True)
        with view_tabs[1]:
            if init_run is None or opt_run is None:
                st.info("Both initial and optimized runs are required for 3D comparison.")
            else:
                from halbach.viz3d import (
                    build_magnet_figure,
                    compute_scene_ranges,
                    enumerate_magnets,
                )

                camera_presets = {
                    "Isometric": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
                    "Top": {"eye": {"x": 0.0, "y": 0.0, "z": 2.0}},
                    "Side +X": {"eye": {"x": 2.0, "y": 0.0, "z": 0.0}},
                    "Side +Y": {"eye": {"x": 0.0, "y": 2.0, "z": 0.0}},
                }
                preset = st.selectbox(
                    "Camera preset",
                    list(camera_presets.keys()),
                    index=0,
                    key="compare_camera_preset",
                )
                scene_camera = camera_presets[preset]

                centers_init, _phi_init, _ring_init, _layer_init = enumerate_magnets(
                    init_run, stride=stride, hide_x_negative=hide_x_negative
                )
                centers_opt, _phi_opt, _ring_opt, _layer_opt = enumerate_magnets(
                    opt_run, stride=stride, hide_x_negative=hide_x_negative
                )
                scene_ranges = compute_scene_ranges([centers_init, centers_opt])

                fig_init = build_magnet_figure(
                    init_run,
                    stride=stride,
                    hide_x_negative=hide_x_negative,
                    mode=mode,
                    magnet_size_m=magnet_size_mm / 1000.0,
                    magnet_thickness_m=magnet_thickness_mm / 1000.0,
                    arrow_length_m=arrow_length_mm / 1000.0,
                    arrow_head_angle_deg=arrow_head_angle_deg,
                    scene_camera=scene_camera,
                    scene_ranges=scene_ranges,
                    height=700,
                )
                fig_opt = build_magnet_figure(
                    opt_run,
                    stride=stride,
                    hide_x_negative=hide_x_negative,
                    mode=mode,
                    magnet_size_m=magnet_size_mm / 1000.0,
                    magnet_thickness_m=magnet_thickness_mm / 1000.0,
                    arrow_length_m=arrow_length_mm / 1000.0,
                    arrow_head_angle_deg=arrow_head_angle_deg,
                    scene_camera=scene_camera,
                    scene_ranges=scene_ranges,
                    height=700,
                )
                compare_cols = st.columns(2)
                with compare_cols[0]:
                    st.caption("Initial")
                    st.plotly_chart(fig_init, use_container_width=True)
                with compare_cols[1]:
                    st.caption("Optimized")
                    st.plotly_chart(fig_opt, use_container_width=True)

    with tabs[3]:
        st.subheader("Optimize (L-BFGS-B)")
        from halbach.gui.opt_job import (
            build_command,
            build_generate_command,
            build_generate_out_dir,
            poll_opt_job,
            run_generate_command,
            start_opt_job,
            stop_opt_job,
            tail_log,
        )

        flash_message = st.session_state.get("flash_message", "")
        if flash_message:
            st.success(flash_message)
            st.session_state["flash_message"] = ""

        job = st.session_state.get("opt_job")
        exit_code = poll_opt_job(job) if job is not None else None
        job_running = job is not None and exit_code is None

        input_choice = st.radio("Input run", ["initial", "optimized", "custom"], index=0)
        custom_input = ""
        if input_choice == "custom":
            custom_input = st.text_input("Custom input path", value="")
        if input_choice == "initial":
            opt_in_path = init_path
        elif input_choice == "optimized":
            opt_in_path = opt_path
        else:
            opt_in_path = custom_input.strip()

        st.caption(f"Input run: {opt_in_path or '(not set)'}")

        maxiter = int(st.number_input("maxiter", min_value=50, max_value=2000, value=900, step=50))
        gtol = float(
            st.number_input("gtol", min_value=1e-16, max_value=1e-6, value=1e-12, format="%.1e")
        )
        roi_r_opt = float(
            st.number_input(
                "ROI radius (m)",
                min_value=0.001,
                max_value=0.2,
                value=0.14,
                step=0.001,
                format="%.4f",
            )
        )
        roi_step_opt = float(
            st.number_input(
                "ROI step (m)",
                min_value=0.001,
                max_value=0.02,
                value=0.02,
                step=0.001,
                format="%.4f",
            )
        )
        angle_model = st.selectbox(
            "Angle model",
            ["legacy-alpha", "delta-rep-x0", "fourier-x0"],
            index=0,
            key="angle_model",
        )
        angle_init = st.selectbox(
            "Angle init",
            ["from-run", "zeros"],
            index=0,
            key="angle_init",
        )
        jax_available = _jax_available()
        if angle_model == "legacy-alpha":
            grad_backend = st.selectbox(
                "Grad backend",
                ["analytic", "jax"],
                index=0,
                key="grad_backend",
            )
            fourier_H = 4
            lambda0 = 0.0
            lambda_theta = 0.0
            lambda_z = 0.0
        else:
            grad_backend = "jax"
            st.caption("Grad backend is fixed to jax for delta/fourier models.")
            if angle_model == "fourier-x0":
                fourier_H = int(
                    st.number_input(
                        "Fourier H",
                        min_value=0,
                        max_value=64,
                        value=4,
                        step=1,
                        key="fourier_H",
                    )
                )
            else:
                fourier_H = 4
            lambda0 = float(
                st.number_input(
                    "lambda0",
                    min_value=0.0,
                    max_value=1e3,
                    value=0.0,
                    step=0.1,
                    key="lambda0",
                )
            )
            lambda_theta = float(
                st.number_input(
                    "lambda_theta",
                    min_value=0.0,
                    max_value=1e3,
                    value=0.0,
                    step=0.1,
                    key="lambda_theta",
                )
            )
            lambda_z = float(
                st.number_input(
                    "lambda_z",
                    min_value=0.0,
                    max_value=1e3,
                    value=0.0,
                    step=0.1,
                    key="lambda_z",
                )
            )
        jax_issue: str | None = None
        if (angle_model != "legacy-alpha" or grad_backend == "jax") and not jax_available:
            jax_issue = (
                "JAX is required for the selected angle model/backend. "
                "Install `jax` and `jaxlib`, or switch to legacy-alpha + analytic."
            )
            st.error(jax_issue)
        can_start = jax_issue is None
        fix_center_radius_layers = int(
            st.selectbox(
                "Fixed center radius layers (z≈0)",
                [0, 2, 4],
                index=1,
                key="fix_center_radius_layers",
            )
        )

        st.markdown("**Radius bounds**")
        r_bounds_enabled = st.checkbox("Enable radius bounds", value=True, key="r_bounds_enabled")
        if r_bounds_enabled:
            r_bound_mode = st.radio(
                "Radius bounds mode",
                ["relative", "absolute"],
                index=0,
                horizontal=True,
                key="r_bound_mode",
            )
            if r_bound_mode == "relative":
                r_lower_delta_mm = float(
                    st.number_input(
                        "Lower delta (mm)",
                        min_value=0.0,
                        max_value=200.0,
                        value=30.0,
                        step=1.0,
                        key="r_lower_delta_mm",
                    )
                )
                r_upper_delta_mm = float(
                    st.number_input(
                        "Upper delta (mm)",
                        min_value=0.0,
                        max_value=200.0,
                        value=30.0,
                        step=1.0,
                        key="r_upper_delta_mm",
                    )
                )
                r_no_upper = st.checkbox("No upper bound", value=False, key="r_no_upper")
                r_min_mm = 0.0
                r_max_mm = 1e9
            else:
                r_min_mm = float(
                    st.number_input(
                        "Min radius (mm)",
                        min_value=0.0,
                        max_value=1e9,
                        value=0.0,
                        step=1.0,
                        key="r_min_mm",
                    )
                )
                r_max_mm = float(
                    st.number_input(
                        "Max radius (mm)",
                        min_value=0.0,
                        max_value=1e9,
                        value=1e9,
                        step=1.0,
                        key="r_max_mm",
                    )
                )
                r_lower_delta_mm = 30.0
                r_upper_delta_mm = 30.0
                r_no_upper = False
        else:
            r_bound_mode = "none"
            r_lower_delta_mm = 30.0
            r_upper_delta_mm = 30.0
            r_no_upper = False
            r_min_mm = 0.0
            r_max_mm = 1e9

        tag = st.text_input("Output tag", value="opt")
        preview_dir = _default_out_dir(tag)
        st.caption(f"Output dir: {preview_dir}")

        with st.expander("Generate initial run", expanded=False):
            gen_cols = st.columns(3)
            with gen_cols[0]:
                gen_N = int(
                    st.number_input("N", min_value=1, max_value=512, value=48, step=1, key="gen_N")
                )
                gen_Lz = float(
                    st.number_input(
                        "Lz (m)", min_value=0.01, max_value=2.0, value=0.64, step=0.01, key="gen_Lz"
                    )
                )
            with gen_cols[1]:
                gen_R = int(
                    st.number_input("R", min_value=1, max_value=32, value=3, step=1, key="gen_R")
                )
                gen_diameter_mm = float(
                    st.number_input(
                        "Diameter (mm)",
                        min_value=10.0,
                        max_value=2000.0,
                        value=400.0,
                        step=10.0,
                        key="gen_diameter_mm",
                    )
                )
            with gen_cols[2]:
                gen_K = int(
                    st.number_input("K", min_value=1, max_value=256, value=24, step=1, key="gen_K")
                )
                gen_ring_offset_step_mm = float(
                    st.number_input(
                        "Ring offset step (mm)",
                        min_value=0.0,
                        max_value=100.0,
                        value=12.0,
                        step=1.0,
                        key="gen_ring_offset_step_mm",
                    )
                )

            gen_tag = st.text_input("Generate output tag", value="init", key="gen_tag")
            gen_use_as_input = st.checkbox(
                "Use generated run as input", value=True, key="gen_use_as_input"
            )
            gen_start_opt = st.checkbox(
                "Generate and start optimization", value=False, key="gen_start_opt"
            )

            gen_out_dir = build_generate_out_dir(
                ROOT / "runs", tag=gen_tag, N=gen_N, R=gen_R, K=gen_K
            )
            st.caption(f"Output dir: {gen_out_dir}")

            gen_clicked = st.button("Generate", key="gen_button")
            if gen_clicked:
                cmd = build_generate_command(
                    gen_out_dir,
                    N=gen_N,
                    R=gen_R,
                    K=gen_K,
                    Lz=gen_Lz,
                    diameter_mm=gen_diameter_mm,
                    ring_offset_step_mm=gen_ring_offset_step_mm,
                )
                code, output = run_generate_command(cmd, cwd=ROOT)
                st.session_state["gen_run_code"] = code
                st.session_state["gen_run_output"] = output
                st.session_state["gen_run_dir"] = str(gen_out_dir)
                if code != 0:
                    st.error(f"Generate failed (exit {code}).")
                else:
                    if gen_use_as_input:
                        try:
                            rel_path = str(gen_out_dir.relative_to(ROOT))
                        except ValueError:
                            rel_path = str(gen_out_dir)
                        st.session_state["pending_init_select"] = rel_path
                        st.session_state["pending_init_path"] = ""
                    if gen_start_opt:
                        if not can_start:
                            st.error(jax_issue or "JAX is required for this configuration.")
                        elif job_running:
                            st.error("Optimization is already running.")
                        else:
                            opt_out_dir = _default_out_dir(f"{gen_tag}_opt")
                            try:
                                job = start_opt_job(
                                    gen_out_dir,
                                    opt_out_dir,
                                    maxiter=maxiter,
                                    gtol=gtol,
                                    roi_r=roi_r_opt,
                                    roi_step=roi_step_opt,
                                    angle_model=angle_model,
                                    grad_backend=grad_backend,
                                    fourier_H=fourier_H,
                                    lambda0=lambda0,
                                    lambda_theta=lambda_theta,
                                    lambda_z=lambda_z,
                                    angle_init=angle_init,
                                    r_bound_mode=r_bound_mode,
                                    r_lower_delta_mm=r_lower_delta_mm,
                                    r_upper_delta_mm=r_upper_delta_mm,
                                    r_no_upper=r_no_upper,
                                    r_min_mm=r_min_mm,
                                    r_max_mm=r_max_mm,
                                    fix_center_radius_layers=fix_center_radius_layers,
                                    repo_root=ROOT,
                                )
                                st.session_state["opt_job"] = job
                                st.session_state["opt_job_fix_center_radius_layers"] = (
                                    fix_center_radius_layers
                                )
                                st.success(f"Started optimization: {job.out_dir}")
                            except Exception as exc:
                                st.error(f"Failed to start optimization: {exc}")
                    st.rerun()

            gen_code = st.session_state.get("gen_run_code")
            if gen_code is not None:
                if gen_code == 0:
                    st.success(f"Generated run: {st.session_state.get('gen_run_dir')}")
                else:
                    st.error(f"Generate failed (exit {gen_code}).")
            gen_output = st.session_state.get("gen_run_output", "")
            if gen_output:
                st.text_area("generate_run output", gen_output, height=160)

        start_cols = st.columns(2)
        with start_cols[0]:
            start_clicked = st.button("Start", disabled=job_running or not can_start)
        with start_cols[1]:
            stop_clicked = st.button("Stop", disabled=not job_running)

        if start_clicked:
            if not can_start:
                st.error(jax_issue or "JAX is required for this configuration.")
            elif job_running:
                st.error("Optimization is already running.")
            elif not opt_in_path:
                st.error("Input run path is empty.")
            else:
                out_dir = _default_out_dir(tag)
                try:
                    job = start_opt_job(
                        opt_in_path,
                        out_dir,
                        maxiter=maxiter,
                        gtol=gtol,
                        roi_r=roi_r_opt,
                        roi_step=roi_step_opt,
                        angle_model=angle_model,
                        grad_backend=grad_backend,
                        fourier_H=fourier_H,
                        lambda0=lambda0,
                        lambda_theta=lambda_theta,
                        lambda_z=lambda_z,
                        angle_init=angle_init,
                        r_bound_mode=r_bound_mode,
                        r_lower_delta_mm=r_lower_delta_mm,
                        r_upper_delta_mm=r_upper_delta_mm,
                        r_no_upper=r_no_upper,
                        r_min_mm=r_min_mm,
                        r_max_mm=r_max_mm,
                        fix_center_radius_layers=fix_center_radius_layers,
                        repo_root=ROOT,
                    )
                    st.session_state["opt_job"] = job
                    st.session_state["opt_job_fix_center_radius_layers"] = fix_center_radius_layers
                    st.success(f"Started optimization: {job.out_dir}")
                except Exception as exc:
                    st.error(f"Failed to start optimization: {exc}")

        if stop_clicked and job is not None:
            stop_opt_job(job)
            st.warning("Terminate signal sent.")

        auto_refresh = False
        refresh_secs = 1.0
        if job_running:
            auto_refresh = st.checkbox("Auto-refresh log", value=True)
            refresh_secs = float(
                st.number_input(
                    "Log refresh (s)", min_value=0.5, max_value=10.0, value=1.0, step=0.5
                )
            )

        if job is not None:
            exit_code = poll_opt_job(job)
            st.write(f"Status: {'running' if exit_code is None else f'exit {exit_code}'}")
            st.write(f"Out dir: `{job.out_dir}`")
            cmd = build_command(
                opt_in_path,
                job.out_dir,
                maxiter=maxiter,
                gtol=gtol,
                roi_r=roi_r_opt,
                roi_step=roi_step_opt,
                angle_model=angle_model,
                grad_backend=grad_backend,
                fourier_H=fourier_H,
                lambda0=lambda0,
                lambda_theta=lambda_theta,
                lambda_z=lambda_z,
                angle_init=angle_init,
                r_bound_mode=r_bound_mode,
                r_lower_delta_mm=r_lower_delta_mm,
                r_upper_delta_mm=r_upper_delta_mm,
                r_no_upper=r_no_upper,
                r_min_mm=r_min_mm,
                r_max_mm=r_max_mm,
                fix_center_radius_layers=fix_center_radius_layers,
            )
            st.code(" ".join(cmd))

            fixed_layers = st.session_state.get("opt_job_fix_center_radius_layers")
            if fixed_layers is not None:
                st.write(f"Fixed center radius layers: {fixed_layers}")

            log_text = tail_log(job.log_path, n_lines=200)
            st.text_area("opt.log (tail)", log_text, height=240)

            if exit_code is not None:
                set_cols = st.columns(2)
                with set_cols[0]:
                    if st.button("Set optimized run"):
                        st.session_state["pending_opt_path"] = str(job.out_dir)
                        st.session_state["pending_opt_select"] = ""
                        st.session_state["flash_message"] = "Optimized run path updated."
                        st.rerun()
                with set_cols[1]:
                    if st.button("Clear job"):
                        st.session_state["opt_job"] = None

        if job_running and auto_refresh:
            time.sleep(max(0.1, refresh_secs))
            st.rerun()


if __name__ == "__main__":
    main()
