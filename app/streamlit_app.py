from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from halbach.run_types import RunBundle
    from halbach.viz2d import ErrorMap2D

PlotlyMode = Literal["fast", "pretty"]

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


@st.cache_data(show_spinner=False)
def _cached_load_run(path_text: str, mtime: float) -> RunBundle:
    from halbach.run_io import load_run

    return load_run(Path(path_text))


@st.cache_data(show_spinner=False)
def _cached_error_map(path_text: str, mtime: float, roi_r: float, step: float) -> ErrorMap2D:
    from halbach.run_io import load_run
    from halbach.viz2d import compute_error_map_ppm_plane

    run = load_run(Path(path_text))
    return compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=roi_r, step=step)


@st.cache_data(show_spinner=False)
def _cached_b0(path_text: str, mtime: float) -> float:
    from halbach.constants import FACTOR, m0, phi0
    from halbach.physics import compute_B_and_B0
    from halbach.run_io import load_run

    run = load_run(Path(path_text))
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
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
    return float(np.sqrt(B0x * B0x + B0y * B0y + B0z * B0z))


def _try_load(path_text: str) -> tuple[RunBundle | None, float | None, str | None]:
    if not path_text:
        return None, None, None
    try:
        mtime = _results_mtime(Path(path_text))
        run = _cached_load_run(path_text, mtime)
        return run, mtime, None
    except Exception as exc:
        return None, None, str(exc)


def _ppm_stats(m: ErrorMap2D) -> tuple[float, float]:
    return float(np.nanmean(m.ppm)), float(np.nanmax(np.abs(m.ppm)))


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
            colorscale="RdBu",
            colorbar={"title": "ppm"},
        )
    )
    level_neg, level_pos = contour_levels_ppm(contour_level)
    fig.add_trace(
        go.Contour(
            x=xs_mm,
            y=ys_mm,
            z=m.ppm,
            contours={"start": level_neg, "end": level_pos, "size": level_pos - level_neg},
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
            st.number_input("ROI radius (m)", min_value=0.05, max_value=0.2, value=0.14, step=0.01)
        )
        step = float(
            st.number_input("Step (m)", min_value=0.001, max_value=0.02, value=0.002, step=0.001)
        )
        auto_limit = st.checkbox("Auto ppm limit", value=False)
        ppm_limit: float | None
        if auto_limit:
            ppm_limit = None
        else:
            ppm_limit = float(
                st.number_input(
                    "PPM limit", min_value=500.0, max_value=20000.0, value=5000.0, step=500.0
                )
            )
        contour_level = float(
            st.number_input(
                "Contour level (ppm)", min_value=100.0, max_value=20000.0, value=1000.0, step=100.0
            )
        )

        st.header("3D Settings")
        view_target = st.radio("3D run", ["initial", "optimized"], index=1)
        mode = cast(PlotlyMode, st.selectbox("Mode", ["fast", "pretty"], index=0))
        stride = int(st.number_input("Stride", min_value=1, max_value=64, value=2, step=1))
        hide_x_negative = st.checkbox("Hide x < 0", value=False)
        magnet_size_mm = float(
            st.number_input("Magnet size (mm)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        )
        magnet_thickness_mm = float(
            st.number_input(
                "Magnet thickness (mm)", min_value=0.5, max_value=20.0, value=2.0, step=0.5
            )
        )

    init_path = _resolve_path(init_select, init_path_text)
    opt_path = _resolve_path(opt_select, opt_path_text)

    init_run, init_mtime, init_err = _try_load(init_path)
    opt_run, opt_mtime, opt_err = _try_load(opt_path)

    if init_err:
        st.error(f"Initial run load error: {init_err}")
    if opt_err:
        st.error(f"Optimized run load error: {opt_err}")

    init_map: ErrorMap2D | None = None
    opt_map: ErrorMap2D | None = None
    if init_run is not None and init_mtime is not None:
        try:
            init_map = _cached_error_map(init_path, init_mtime, roi_r, step)
        except Exception as exc:
            st.error(f"Initial 2D map error: {exc}")
    if opt_run is not None and opt_mtime is not None:
        try:
            opt_map = _cached_error_map(opt_path, opt_mtime, roi_r, step)
        except Exception as exc:
            st.error(f"Optimized 2D map error: {exc}")

    vmin = vmax = 0.0
    if init_map is not None and opt_map is not None:
        from halbach.viz2d import common_ppm_limits

        vmin, vmax = common_ppm_limits([init_map, opt_map], limit_ppm=ppm_limit, symmetric=True)

    tabs = st.tabs(["Overview", "2D", "3D"])

    with tabs[0]:
        st.subheader("Runs")
        cols = st.columns(2)

        def render_summary(run: RunBundle | None, m: ErrorMap2D | None, label: str) -> None:
            if run is None:
                st.info(f"{label} run is not loaded.")
                return
            b0 = None
            try:
                mtime = _results_mtime(run.results_path)
                b0 = _cached_b0(str(run.results_path), mtime)
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

        with cols[0]:
            render_summary(init_run, init_map, "Initial")
        with cols[1]:
            render_summary(opt_run, opt_map, "Optimized")

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
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
