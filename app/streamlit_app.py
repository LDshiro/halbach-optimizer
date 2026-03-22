from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from halbach.perturbation_eval import PerturbationResult
    from halbach.run_types import RunBundle
    from halbach.viz2d import ErrorMap2D

PlotlyMode = Literal["fast", "pretty", "cubes", "cubes_arrows"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HUMAN_OVERLAY_OBJ = "10688_GenericMale_v2.obj"
DEFAULT_COIL_OVERLAY_DIR = "coil"


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


def _default_human_overlay_obj_path() -> Path:
    return ROOT / DEFAULT_HUMAN_OVERLAY_OBJ


def _resolve_human_overlay_obj_path(path_text: str) -> Path:
    raw = path_text.strip()
    if not raw:
        return _default_human_overlay_obj_path()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _human_overlay_unit_scale_m(unit_mode: str) -> float:
    unit_scale = {
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
    }
    try:
        return float(unit_scale[unit_mode])
    except KeyError as exc:
        raise ValueError(f"Unsupported OBJ unit: {unit_mode}") from exc


def _human_overlay_transform_key(
    obj_path_text: str,
    obj_mtime: float,
    unit_mode: str,
    uniform_scale: float,
    rotation_xyz_deg: tuple[float, float, float],
    translation_xyz_m: tuple[float, float, float],
) -> str:
    return json.dumps(
        {
            "obj_path": obj_path_text,
            "obj_mtime": float(obj_mtime),
            "unit_mode": unit_mode,
            "uniform_scale": float(uniform_scale),
            "rotation_xyz_deg": [float(v) for v in rotation_xyz_deg],
            "translation_xyz_m": [float(v) for v in translation_xyz_m],
        },
        sort_keys=True,
    )


def _camera_presets() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "Isometric": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "Top": {"eye": {"x": 0.0, "y": 0.0, "z": 2.0}},
        "Side +X": {"eye": {"x": 2.0, "y": 0.0, "z": 0.0}},
        "Side +Y": {"eye": {"x": 0.0, "y": 2.0, "z": 0.0}},
    }


def _magnet_surface_color_options() -> dict[str, str]:
    return {
        "Current gray": "rgb(229,229,229)",
        "Cadmium yellow": "rgb(255,246,0)",
        "Lemon yellow": "rgb(255,255,102)",
    }


def _scan_coil_npz_files(coil_dir: Path) -> list[str]:
    if not coil_dir.is_dir():
        return []
    candidates: list[str] = []
    for entry in sorted(coil_dir.rglob("*.npz")):
        if not entry.is_file():
            continue
        try:
            candidates.append(str(entry.relative_to(ROOT)))
        except ValueError:
            candidates.append(str(entry))
    return candidates


def _resolve_coil_overlay_path(selected: str) -> Path:
    candidate = Path(selected.strip())
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _coil_x_rotation_quarter_turns(rotation_deg: int) -> int:
    allowed = {0, 90, 180, 270}
    if rotation_deg not in allowed:
        raise ValueError(f"Unsupported coil X rotation: {rotation_deg}")
    return rotation_deg // 90


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


def _is_dc_run(run: RunBundle | None) -> bool:
    return run is not None and run.meta.get("framework") == "dc"


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
    path_text: str,
    mtime: float,
    _meta_key: str,
    roi_r: float,
    step: float,
    mag_model_eval: str,
    sc_cfg_key: str,
) -> tuple[ErrorMap2D, dict[str, object]]:
    from halbach.run_io import load_run
    from halbach.viz2d import compute_error_map_ppm_plane_with_debug

    run = load_run(Path(path_text))
    sc_cfg_override = json.loads(sc_cfg_key) if sc_cfg_key else None
    return compute_error_map_ppm_plane_with_debug(
        run,
        plane="xy",
        coord0=0.0,
        roi_r=roi_r,
        step=step,
        mag_model_eval=cast(Literal["auto", "fixed", "self-consistent-easy-axis"], mag_model_eval),
        sc_cfg_override=sc_cfg_override,
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


@st.cache_data(show_spinner=False)
def _cached_variation_eval(
    path_text: str,
    mtime: float,
    _meta_key: str,
    cfg_key: str,
) -> tuple[PerturbationResult, dict[str, object]]:
    from halbach.perturbation_eval import PerturbationConfig, run_perturbation_case
    from halbach.run_io import load_run

    run = load_run(Path(path_text))
    cfg_raw = json.loads(cfg_key)
    cfg = PerturbationConfig(
        sigma_rel_pct=float(cfg_raw["sigma_rel_pct"]),
        sigma_phi_deg=float(cfg_raw["sigma_phi_deg"]),
        seed=int(cfg_raw["seed"]),
        roi_radius_m=float(cfg_raw["roi_radius_m"]),
        roi_samples=int(cfg_raw["roi_samples"]),
        map_radius_m=float(cfg_raw["map_radius_m"]),
        map_step_m=float(cfg_raw["map_step_m"]),
        sc_cfg=cast(dict[str, Any], cfg_raw["sc_cfg"]),
        target_plane_z=float(cfg_raw.get("target_plane_z", 0.0)),
    )
    result = run_perturbation_case(run, cfg)
    debug = {
        "run_dir": str(run.run_dir),
        "run_name": run.name,
    }
    return result, debug


@st.cache_data(show_spinner=False)
def _cached_load_obj_mesh(path_text: str, mtime: float):
    from halbach.obj_mesh import load_obj_mesh

    return load_obj_mesh(Path(path_text))


@st.cache_data(show_spinner=False)
def _cached_transform_obj_mesh(path_text: str, mtime: float, transform_key: str):
    from halbach.obj_mesh import transform_obj_mesh

    raw_mesh = _cached_load_obj_mesh(path_text, mtime)
    cfg = json.loads(transform_key)
    rotation_xyz_deg = tuple(float(v) for v in cfg["rotation_xyz_deg"])
    translation_xyz_m = tuple(float(v) for v in cfg["translation_xyz_m"])
    return transform_obj_mesh(
        raw_mesh,
        unit_scale_m=_human_overlay_unit_scale_m(str(cfg["unit_mode"])),
        uniform_scale=float(cfg["uniform_scale"]),
        rotation_xyz_deg=cast(tuple[float, float, float], rotation_xyz_deg),
        translation_xyz_m=cast(tuple[float, float, float], translation_xyz_m),
        anchor_mode="bbox_center",
    )


@st.cache_data(show_spinner=False)
def _cached_load_coil_overlay(path_text: str, mtime: float):
    from halbach.coil_overlay import load_coil_polyline_npz

    return load_coil_polyline_npz(Path(path_text))


@st.cache_data(show_spinner=False)
def _cached_transform_coil_overlay(path_text: str, mtime: float, rotation_x_deg: int):
    from halbach.coil_overlay import rotate_coil_polyline_x90

    coils = _cached_load_coil_overlay(path_text, mtime)
    return rotate_coil_polyline_x90(coils, _coil_x_rotation_quarter_turns(int(rotation_x_deg)))


def _default_variation_sc_cfg(run: RunBundle | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "chi": 0.0,
        "Nd": 1.0 / 3.0,
        "p0": 1.0,
        "volume_mm3": 1000.0,
        "iters": 30,
        "omega": 0.6,
        "near_window": {"wr": 0, "wz": 1, "wphi": 2},
        "near_kernel": "dipole",
        "subdip_n": 2,
    }
    if run is None:
        return defaults
    magnetization = run.meta.get("magnetization")
    if not isinstance(magnetization, dict):
        return defaults
    sc_cfg = magnetization.get("self_consistent")
    if not isinstance(sc_cfg, dict):
        return defaults

    near_window_raw = sc_cfg.get("near_window")
    if isinstance(near_window_raw, dict):
        defaults["near_window"] = {
            "wr": int(near_window_raw.get("wr", 0)),
            "wz": int(near_window_raw.get("wz", 1)),
            "wphi": int(near_window_raw.get("wphi", 2)),
        }
    defaults["chi"] = float(sc_cfg.get("chi", defaults["chi"]))
    defaults["Nd"] = float(sc_cfg.get("Nd", defaults["Nd"]))
    defaults["p0"] = float(sc_cfg.get("p0", defaults["p0"]))
    defaults["volume_mm3"] = float(sc_cfg.get("volume_mm3", defaults["volume_mm3"]))
    defaults["iters"] = int(sc_cfg.get("iters", defaults["iters"]))
    defaults["omega"] = float(sc_cfg.get("omega", defaults["omega"]))
    near_kernel = str(sc_cfg.get("near_kernel", defaults["near_kernel"]))
    if near_kernel == "cube-average":
        near_kernel = "cellavg"
    defaults["near_kernel"] = near_kernel
    defaults["subdip_n"] = int(sc_cfg.get("subdip_n", defaults["subdip_n"]))
    if sc_cfg.get("gl_order") is not None:
        defaults["gl_order"] = int(sc_cfg["gl_order"])
    return defaults


def _seed_variation_sc_state(run_key: str, defaults: dict[str, Any]) -> None:
    if st.session_state.get("variation_sc_seed_run") == run_key:
        return
    near_window = cast(dict[str, Any], defaults.get("near_window", {}))
    st.session_state["variation_sc_chi"] = float(defaults.get("chi", 0.0))
    st.session_state["variation_sc_Nd"] = float(defaults.get("Nd", 1.0 / 3.0))
    st.session_state["variation_sc_p0"] = float(defaults.get("p0", 1.0))
    st.session_state["variation_sc_volume_mm3"] = float(defaults.get("volume_mm3", 1000.0))
    st.session_state["variation_sc_iters"] = int(defaults.get("iters", 30))
    st.session_state["variation_sc_omega"] = float(defaults.get("omega", 0.6))
    st.session_state["variation_sc_near_wr"] = int(near_window.get("wr", 0))
    st.session_state["variation_sc_near_wz"] = int(near_window.get("wz", 1))
    st.session_state["variation_sc_near_wphi"] = int(near_window.get("wphi", 2))
    st.session_state["variation_sc_near_kernel"] = str(defaults.get("near_kernel", "dipole"))
    st.session_state["variation_sc_subdip_n"] = int(defaults.get("subdip_n", 2))
    if "gl_order" in defaults and defaults["gl_order"] is not None:
        st.session_state["variation_sc_gl_order"] = str(defaults["gl_order"])
    else:
        st.session_state["variation_sc_gl_order"] = "mixed (2/3)"
    st.session_state["variation_sc_seed_run"] = run_key


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


def _fmt_vec3(value: np.ndarray, fmt: str = ".4f") -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return ", ".join(f"{float(v):{fmt}}" for v in arr)


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
        st.header("2D Magnetization Eval")
        mag_model_eval = st.selectbox(
            "Magnetization model (2D)",
            ["auto", "fixed", "self-consistent-easy-axis"],
            index=0,
            key="map_mag_model_eval",
        )
        sc_cfg_eval: dict[str, object] | None = None
        sc_cfg_key = ""
        if mag_model_eval == "self-consistent-easy-axis":
            with st.expander("Self-consistent eval settings", expanded=True):
                sc_chi = float(
                    st.number_input(
                        "chi (2D eval)", min_value=0.0, max_value=10.0, value=0.0, step=0.01
                    )
                )
                sc_Nd = float(
                    st.number_input(
                        "Nd (2D eval)",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / 3.0,
                        step=0.01,
                    )
                )
                sc_p0 = float(
                    st.number_input(
                        "p0 (2D eval)", min_value=0.0, max_value=10.0, value=1.0, step=0.01
                    )
                )
                sc_volume_mm3 = float(
                    st.number_input(
                        "volume_mm3 (2D eval)",
                        min_value=1.0,
                        max_value=1e6,
                        value=1000.0,
                        step=10.0,
                    )
                )
                sc_iters = int(
                    st.number_input("iters (2D eval)", min_value=1, max_value=500, value=30, step=1)
                )
                sc_omega = float(
                    st.number_input(
                        "omega (2D eval)", min_value=0.01, max_value=1.0, value=0.6, step=0.01
                    )
                )
                st.markdown("**Near window (2D eval)**")
                sc_near_wr = int(
                    st.number_input("wr (2D eval)", min_value=0, max_value=10, value=0, step=1)
                )
                sc_near_wz = int(
                    st.number_input("wz (2D eval)", min_value=0, max_value=10, value=1, step=1)
                )
                sc_near_wphi = int(
                    st.number_input("wphi (2D eval)", min_value=0, max_value=10, value=2, step=1)
                )
                sc_near_kernel = st.selectbox(
                    "near kernel (2D eval)",
                    ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                    index=0,
                    key="map_sc_near_kernel",
                )
                sc_subdip_n = 2
                if sc_near_kernel == "multi-dipole":
                    sc_subdip_n = int(
                        st.number_input(
                            "subdip_n (2D eval)",
                            min_value=2,
                            max_value=10,
                            value=2,
                            step=1,
                        )
                    )
                sc_gl_order: int | None = None
                if sc_near_kernel == "gl-double-mixed":
                    gl_choice = st.selectbox(
                        "gl order (2D eval)",
                        ["mixed (2/3)", "2", "3"],
                        index=0,
                        key="map_sc_gl_order",
                    )
                    if gl_choice in ("2", "3"):
                        sc_gl_order = int(gl_choice)
            sc_cfg_eval = dict(
                chi=sc_chi,
                Nd=sc_Nd,
                p0=sc_p0,
                volume_mm3=sc_volume_mm3,
                iters=sc_iters,
                omega=sc_omega,
                near_window=dict(wr=sc_near_wr, wz=sc_near_wz, wphi=sc_near_wphi),
                near_kernel=sc_near_kernel,
                subdip_n=sc_subdip_n,
            )
            if sc_gl_order is not None:
                sc_cfg_eval["gl_order"] = sc_gl_order
            sc_cfg_key = json.dumps(sc_cfg_eval, sort_keys=True)
            if not _jax_available():
                st.error(
                    "JAX is required for self-consistent 2D evaluation unless sc_p_flat is saved."
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
        magnet_surface_palette = _magnet_surface_color_options()
        magnet_surface_label = st.selectbox(
            "Magnet surface color",
            list(magnet_surface_palette.keys()),
            index=0,
        )
        magnet_surface_color = magnet_surface_palette[magnet_surface_label]

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
            init_map, init_debug = _cached_error_map(
                init_path,
                init_mtime,
                init_key,
                roi_r,
                step,
                mag_model_eval,
                sc_cfg_key,
            )
        except Exception as exc:
            st.error(f"Initial 2D map error: {exc}")
    if opt_run is not None and opt_mtime is not None:
        try:
            opt_key = _magnetization_cache_key(opt_run.meta)
            opt_map, opt_debug = _cached_error_map(
                opt_path,
                opt_mtime,
                opt_key,
                roi_r,
                step,
                mag_model_eval,
                sc_cfg_key,
            )
        except Exception as exc:
            st.error(f"Optimized 2D map error: {exc}")

    vmin = vmax = 0.0
    available_maps = [m for m in (init_map, opt_map) if m is not None]
    if available_maps:
        from halbach.viz2d import common_ppm_limits

        vmin, vmax = common_ppm_limits(available_maps, limit_ppm=ppm_limit, symmetric=True)

    tabs = st.tabs(["Overview", "2D", "3D", "Human Overlay", "Variation", "Optimize"])

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
            is_dc = _is_dc_run(run)
            b0 = None
            st.markdown(f"**{label} name**: {run.name}")
            st.markdown(f"**Results**: `{run.results_path}`")
            st.markdown(f"**Meta**: `{run.meta_path}`")
            if is_dc:
                st.markdown("**Framework**: DC/CCP")
                if run.meta.get("dc_model") is not None:
                    st.write(f"dc_model: {run.meta.get('dc_model')}")

                extras = run.results.extras

                def _stats(name: str, key: str) -> None:
                    if key not in extras:
                        return
                    arr = np.asarray(extras[key], dtype=np.float64).reshape(-1)
                    if arr.size == 0:
                        return
                    st.write(
                        f"{name} min/mean/max: {float(np.min(arr)):.6f}, "
                        f"{float(np.mean(arr)):.6f}, {float(np.max(arr)):.6f}"
                    )

                _stats("p_opt", "p_opt")
                _stats("p_sc_post", "p_sc_post")
                _stats("z_norm", "z_norm")
                if "By_diff" in extras:
                    by_diff = np.asarray(extras["By_diff"], dtype=np.float64).reshape(-1)
                    if by_diff.size:
                        st.write(f"By_diff std: {float(np.std(by_diff)):.6e}")
                if "x_opt" in extras:
                    x_opt = np.asarray(extras["x_opt"], dtype=np.float64).reshape(-1)
                    if x_opt.size:
                        st.write(f"x_opt max|.|: {float(np.max(np.abs(x_opt))):.6f}")
                if "r0_flat" in extras:
                    r0_flat = np.asarray(extras["r0_flat"])
                    st.write(f"r0_flat shape: {r0_flat.shape}")
                if "pts" in extras:
                    pts = np.asarray(extras["pts"])
                    st.write(f"pts shape: {pts.shape}")
                return

            try:
                mtime = _results_mtime(run.results_path)
                meta_key = _magnetization_cache_key(run.meta)
                b0 = _cached_b0(str(run.results_path), mtime, meta_key)
            except Exception:
                b0 = None
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
                if debug.get("sc_p_source") is not None:
                    st.write(f"p source: {debug.get('sc_p_source')}")
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
            if _is_dc_run(init_run) or _is_dc_run(opt_run):
                st.write("Delta summary is not available for DC/CCP runs.")
            elif init_run.results.alphas.shape == opt_run.results.alphas.shape:
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
        if init_map is None and opt_map is None:
            st.info("Select a run to render 2D maps.")
        else:
            maps_to_show: list[tuple[str, ErrorMap2D, str]] = []
            if init_map is not None:
                maps_to_show.append(("Initial", init_map, "init"))
            if opt_map is not None:
                maps_to_show.append(("Optimized", opt_map, "opt"))

            if len(maps_to_show) == 1:
                label, map_data, key_suffix = maps_to_show[0]
                fig_map = _plot_error_map(map_data, vmin, vmax, contour_level, label)
                st.plotly_chart(fig_map, use_container_width=True, key=f"map_{key_suffix}")
                fig_line = _plot_cross_section(map_data, vmin, vmax, f"{label} y=0")
                st.plotly_chart(fig_line, use_container_width=True, key=f"line_{key_suffix}")
            else:
                map_cols = st.columns(2)
                for col, (label, map_data, key_suffix) in zip(map_cols, maps_to_show, strict=False):
                    with col:
                        fig_map = _plot_error_map(map_data, vmin, vmax, contour_level, label)
                        st.plotly_chart(fig_map, use_container_width=True, key=f"map_{key_suffix}")

                line_cols = st.columns(2)
                for col, (label, map_data, key_suffix) in zip(
                    line_cols, maps_to_show, strict=False
                ):
                    with col:
                        fig_line = _plot_cross_section(map_data, vmin, vmax, f"{label} y=0")
                        st.plotly_chart(
                            fig_line, use_container_width=True, key=f"line_{key_suffix}"
                        )

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
                    magnet_surface_color=magnet_surface_color,
                )
                st.plotly_chart(fig, use_container_width=True, key="magnet_single")
        with view_tabs[1]:
            if init_run is None or opt_run is None:
                st.info("Both initial and optimized runs are required for 3D comparison.")
            else:
                from halbach.viz3d import (
                    build_magnet_figure,
                    compute_scene_ranges,
                    enumerate_magnets,
                )

                camera_presets = _camera_presets()
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
                    magnet_surface_color=magnet_surface_color,
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
                    magnet_surface_color=magnet_surface_color,
                )
                compare_cols = st.columns(2)
                with compare_cols[0]:
                    st.caption("Initial")
                    st.plotly_chart(fig_init, use_container_width=True, key="magnet_compare_init")
                with compare_cols[1]:
                    st.caption("Optimized")
                    st.plotly_chart(fig_opt, use_container_width=True, key="magnet_compare_opt")

    with tabs[3]:
        st.subheader("Human Overlay")
        st.caption(
            "Overlay a Wavefront OBJ body mesh on top of the current magnet layout for fit and scale checks."
        )
        overlay_run_target = st.radio(
            "Overlay run",
            ["initial", "optimized"],
            index=1,
            horizontal=True,
            key="human_overlay_run_target",
        )
        overlay_run = init_run if overlay_run_target == "initial" else opt_run

        obj_path_input = st.text_input(
            "OBJ path",
            value=DEFAULT_HUMAN_OVERLAY_OBJ,
            key="human_overlay_obj_path",
        )
        overlay_obj_path = _resolve_human_overlay_obj_path(obj_path_input)
        st.caption(f"Resolved OBJ path: `{overlay_obj_path}`")

        overlay_cols = st.columns(3)
        with overlay_cols[0]:
            overlay_unit = st.selectbox(
                "OBJ unit",
                ["mm", "cm", "m"],
                index=1,
                key="human_overlay_unit",
            )
        with overlay_cols[1]:
            overlay_opacity = float(
                st.slider(
                    "Body opacity",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.35,
                    step=0.05,
                    key="human_overlay_opacity",
                )
            )
        with overlay_cols[2]:
            overlay_scale = float(
                st.number_input(
                    "Uniform scale",
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0,
                    step=0.01,
                    key="human_overlay_scale",
                )
            )

        st.markdown("**Position / rotation**")
        transform_cols = st.columns(3)
        with transform_cols[0]:
            overlay_tx = float(
                st.number_input(
                    "Translate X (m)",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.01,
                    format="%.3f",
                    key="human_overlay_tx_m",
                )
            )
            overlay_rx = float(
                st.number_input(
                    "Rotate X (deg)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    step=1.0,
                    key="human_overlay_rx_deg",
                )
            )
        with transform_cols[1]:
            overlay_ty = float(
                st.number_input(
                    "Translate Y (m)",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.01,
                    format="%.3f",
                    key="human_overlay_ty_m",
                )
            )
            overlay_ry = float(
                st.number_input(
                    "Rotate Y (deg)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    step=1.0,
                    key="human_overlay_ry_deg",
                )
            )
        with transform_cols[2]:
            overlay_tz = float(
                st.number_input(
                    "Translate Z (m)",
                    min_value=-5.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.01,
                    format="%.3f",
                    key="human_overlay_tz_m",
                )
            )
            overlay_rz = float(
                st.number_input(
                    "Rotate Z (deg)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    step=1.0,
                    key="human_overlay_rz_deg",
                )
            )

        camera_presets = _camera_presets()
        overlay_camera_preset = st.selectbox(
            "Camera preset",
            list(camera_presets.keys()),
            index=0,
            key="human_overlay_camera_preset",
        )
        show_coil_overlay = st.checkbox(
            "Show coil overlay",
            value=True,
            key="human_overlay_show_coil",
        )
        coil_candidates = _scan_coil_npz_files(ROOT / DEFAULT_COIL_OVERLAY_DIR)
        coil_selected = ""
        coil_line_width = 4.0
        coil_rotation_x_deg = 0
        if show_coil_overlay:
            if coil_candidates:
                coil_selected = st.selectbox(
                    "Coil file",
                    coil_candidates,
                    index=0,
                    key="human_overlay_coil_file",
                )
                coil_line_width = float(
                    st.number_input(
                        "Coil line width",
                        min_value=1.0,
                        max_value=12.0,
                        value=4.0,
                        step=0.5,
                        key="human_overlay_coil_line_width",
                    )
                )
                coil_rotation_x_deg = int(
                    st.selectbox(
                        "Coil X rotation",
                        [0, 90, 180, 270],
                        index=0,
                        key="human_overlay_coil_rotation_x_deg",
                    )
                )
            else:
                st.warning(f"No coil NPZ files found under `{ROOT / DEFAULT_COIL_OVERLAY_DIR}`.")

        st.caption(
            "v1 limitation: OBJ shape only. `.mtl`, textures, normals, and material colors are ignored."
        )

        raw_overlay_mesh = None
        transformed_overlay_mesh = None
        coil_overlay = None
        overlay_load_error: str | None = None
        if not overlay_obj_path.is_file():
            overlay_load_error = f"OBJ file not found: {overlay_obj_path}"
        else:
            try:
                overlay_obj_mtime = float(overlay_obj_path.stat().st_mtime)
                rotation_xyz_deg = (overlay_rx, overlay_ry, overlay_rz)
                translation_xyz_m = (overlay_tx, overlay_ty, overlay_tz)
                transform_key = _human_overlay_transform_key(
                    str(overlay_obj_path),
                    overlay_obj_mtime,
                    overlay_unit,
                    overlay_scale,
                    rotation_xyz_deg,
                    translation_xyz_m,
                )
                raw_overlay_mesh = _cached_load_obj_mesh(str(overlay_obj_path), overlay_obj_mtime)
                transformed_overlay_mesh = _cached_transform_obj_mesh(
                    str(overlay_obj_path),
                    overlay_obj_mtime,
                    transform_key,
                )
                if show_coil_overlay and coil_selected:
                    coil_path = _resolve_coil_overlay_path(coil_selected)
                    if not coil_path.is_file():
                        raise FileNotFoundError(f"Coil file not found: {coil_path}")
                    coil_mtime = float(coil_path.stat().st_mtime)
                    coil_overlay = _cached_transform_coil_overlay(
                        str(coil_path),
                        coil_mtime,
                        coil_rotation_x_deg,
                    )
            except Exception as exc:
                overlay_load_error = str(exc)

        if overlay_load_error is not None:
            st.error(overlay_load_error)
        elif raw_overlay_mesh is not None and transformed_overlay_mesh is not None:
            unit_scale_m = _human_overlay_unit_scale_m(overlay_unit)
            converted_size_m = raw_overlay_mesh.bbox_size * unit_scale_m * overlay_scale

            dims_cols = st.columns(2)
            with dims_cols[0]:
                st.markdown("**Raw OBJ bbox**")
                st.write(f"size (OBJ units): {_fmt_vec3(raw_overlay_mesh.bbox_size)}")
                st.write(f"min (OBJ units): {_fmt_vec3(raw_overlay_mesh.bbox_min)}")
                st.write(f"max (OBJ units): {_fmt_vec3(raw_overlay_mesh.bbox_max)}")
            with dims_cols[1]:
                st.markdown("**Converted / transformed bbox**")
                st.write(f"size after unit+scale (m): {_fmt_vec3(converted_size_m)}")
                st.write(f"transformed min (m): {_fmt_vec3(transformed_overlay_mesh.bbox_min)}")
                st.write(f"transformed max (m): {_fmt_vec3(transformed_overlay_mesh.bbox_max)}")
                st.write(f"transformed size (m): {_fmt_vec3(transformed_overlay_mesh.bbox_size)}")

            if coil_overlay is not None:
                coil_cols = st.columns(2)
                with coil_cols[0]:
                    st.markdown("**Coil overlay**")
                    st.write(f"file: `{coil_selected}`")
                    st.write(f"x rotation (deg): {coil_rotation_x_deg}")
                    st.write(f"polyline count: {int(coil_overlay.polyline_start.shape[0])}")
                    st.write(f"point count: {int(coil_overlay.points_xyz.shape[0])}")
                with coil_cols[1]:
                    st.markdown("**Displayed coil bbox**")
                    st.write(
                        "unique colors: "
                        f"{int(np.unique(coil_overlay.polyline_color_rgba_u8, axis=0).shape[0])}"
                    )
                    st.write(f"bbox min (m): {_fmt_vec3(coil_overlay.bbox_min)}")
                    st.write(f"bbox max (m): {_fmt_vec3(coil_overlay.bbox_max)}")
                    st.write(f"bbox size (m): {_fmt_vec3(coil_overlay.bbox_size)}")

            if overlay_run is None:
                st.info("Select a run to overlay the body mesh on the magnet figure.")
            else:
                from halbach.viz3d import (
                    build_magnet_figure,
                    compute_scene_ranges,
                    enumerate_magnets,
                )

                centers_overlay, _phi_overlay, _ring_overlay, _layer_overlay = enumerate_magnets(
                    overlay_run,
                    stride=stride,
                    hide_x_negative=hide_x_negative,
                )
                overlay_range_sources = [centers_overlay, transformed_overlay_mesh.vertices]
                if coil_overlay is not None:
                    overlay_range_sources.append(coil_overlay.points_xyz)
                overlay_scene_ranges = compute_scene_ranges(overlay_range_sources)
                overlay_fig = build_magnet_figure(
                    overlay_run,
                    stride=stride,
                    hide_x_negative=hide_x_negative,
                    mode=mode,
                    magnet_size_m=magnet_size_mm / 1000.0,
                    magnet_thickness_m=magnet_thickness_mm / 1000.0,
                    arrow_length_m=arrow_length_mm / 1000.0,
                    arrow_head_angle_deg=arrow_head_angle_deg,
                    scene_camera=camera_presets[overlay_camera_preset],
                    scene_ranges=overlay_scene_ranges,
                    height=1520,
                    overlay_mesh=transformed_overlay_mesh,
                    overlay_opacity=overlay_opacity,
                    coil_overlay=coil_overlay,
                    coil_line_width=coil_line_width,
                    magnet_surface_color=magnet_surface_color,
                )
                st.plotly_chart(
                    overlay_fig,
                    use_container_width=True,
                    key="human_overlay_figure",
                )

    with tabs[4]:
        st.subheader("Magnetization Variation Eval (prototype)")
        st.caption(
            "One realization: per-magnet amplitude/angle perturbation + self-consistent solve, "
            "ROI vectors and XY(z=0) map export."
        )
        variation_source = st.radio(
            "Run source",
            ["initial", "optimized", "custom"],
            index=1,
            key="variation_run_source",
            horizontal=True,
        )
        variation_custom_path = ""
        if variation_source == "custom":
            variation_custom_path = st.text_input(
                "Custom run path",
                value="",
                key="variation_custom_path",
            )
        if variation_source == "initial":
            variation_path = init_path
            variation_run = init_run
            variation_mtime = init_mtime
        elif variation_source == "optimized":
            variation_path = opt_path
            variation_run = opt_run
            variation_mtime = opt_mtime
        else:
            variation_path = variation_custom_path.strip()
            variation_run, variation_mtime, _variation_meta_mtime, variation_load_err = _try_load(
                variation_path
            )
            if variation_load_err:
                st.error(f"Variation run load error: {variation_load_err}")

        st.caption(f"Target run: {variation_path or '(not set)'}")
        if variation_run is None:
            st.info("Select a run to evaluate variation.")
        else:
            run_key = str(variation_run.run_dir)
            sc_defaults = _default_variation_sc_cfg(variation_run)
            _seed_variation_sc_state(run_key, sc_defaults)

            cfg_cols = st.columns(3)
            with cfg_cols[0]:
                sigma_rel_pct = float(
                    st.number_input(
                        "sigma_rel_pct (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=2.0,
                        step=0.1,
                        key="variation_sigma_rel_pct",
                    )
                )
                sigma_phi_deg = float(
                    st.number_input(
                        "sigma_phi_deg (deg)",
                        min_value=0.0,
                        max_value=180.0,
                        value=1.0,
                        step=0.1,
                        key="variation_sigma_phi_deg",
                    )
                )
                seed = int(
                    st.number_input(
                        "seed",
                        min_value=0,
                        max_value=2_147_483_647,
                        value=1234,
                        step=1,
                        key="variation_seed",
                    )
                )
            with cfg_cols[1]:
                roi_radius_var = float(
                    st.number_input(
                        "ROI radius (m)",
                        min_value=0.001,
                        max_value=0.5,
                        value=0.14,
                        step=0.001,
                        format="%.4f",
                        key="variation_roi_radius_m",
                    )
                )
                roi_samples_var = int(
                    st.number_input(
                        "ROI samples (Fibonacci)",
                        min_value=1,
                        max_value=200000,
                        value=512,
                        step=1,
                        key="variation_roi_samples",
                    )
                )
            with cfg_cols[2]:
                map_radius_var = float(
                    st.number_input(
                        "Map radius (m)",
                        min_value=0.001,
                        max_value=0.5,
                        value=0.14,
                        step=0.001,
                        format="%.4f",
                        key="variation_map_radius_m",
                    )
                )
                map_step_var = float(
                    st.number_input(
                        "Map step (m)",
                        min_value=0.001,
                        max_value=0.05,
                        value=0.002,
                        step=0.001,
                        format="%.4f",
                        key="variation_map_step_m",
                    )
                )
                st.caption("Map plane: XY (z=0)")

            with st.expander("Self-consistent settings (editable)", expanded=False):
                sc_chi_var = float(
                    st.number_input(
                        "chi",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                        key="variation_sc_chi",
                    )
                )
                sc_Nd_var = float(
                    st.number_input(
                        "Nd",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        key="variation_sc_Nd",
                    )
                )
                sc_p0_var = float(
                    st.number_input(
                        "p0",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                        key="variation_sc_p0",
                    )
                )
                sc_volume_mm3_var = float(
                    st.number_input(
                        "volume_mm3",
                        min_value=1.0,
                        max_value=1e7,
                        step=10.0,
                        key="variation_sc_volume_mm3",
                    )
                )
                sc_iters_var = int(
                    st.number_input(
                        "iters",
                        min_value=1,
                        max_value=1000,
                        step=1,
                        key="variation_sc_iters",
                    )
                )
                sc_omega_var = float(
                    st.number_input(
                        "omega",
                        min_value=0.01,
                        max_value=1.0,
                        step=0.01,
                        key="variation_sc_omega",
                    )
                )
                sc_wr_var = int(
                    st.number_input(
                        "near wr",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="variation_sc_near_wr",
                    )
                )
                sc_wz_var = int(
                    st.number_input(
                        "near wz",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="variation_sc_near_wz",
                    )
                )
                sc_wphi_var = int(
                    st.number_input(
                        "near wphi",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="variation_sc_near_wphi",
                    )
                )
                sc_kernel_var = st.selectbox(
                    "near kernel",
                    ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                    index=0,
                    key="variation_sc_near_kernel",
                )
                sc_subdip_n_var = 2
                if sc_kernel_var == "multi-dipole":
                    sc_subdip_n_var = int(
                        st.number_input(
                            "subdip_n",
                            min_value=2,
                            max_value=10,
                            step=1,
                            key="variation_sc_subdip_n",
                        )
                    )
                else:
                    sc_subdip_n_var = int(st.session_state.get("variation_sc_subdip_n", 2))
                sc_gl_order_var: int | None = None
                if sc_kernel_var == "gl-double-mixed":
                    gl_choice_var = st.selectbox(
                        "gl order",
                        ["mixed (2/3)", "2", "3"],
                        index=0,
                        key="variation_sc_gl_order",
                    )
                    if gl_choice_var in ("2", "3"):
                        sc_gl_order_var = int(gl_choice_var)

            sc_cfg_eval_var: dict[str, object] = {
                "chi": sc_chi_var,
                "Nd": sc_Nd_var,
                "p0": sc_p0_var,
                "volume_mm3": sc_volume_mm3_var,
                "iters": sc_iters_var,
                "omega": sc_omega_var,
                "near_window": {"wr": sc_wr_var, "wz": sc_wz_var, "wphi": sc_wphi_var},
                "near_kernel": sc_kernel_var,
                "subdip_n": sc_subdip_n_var,
            }
            if sc_gl_order_var is not None:
                sc_cfg_eval_var["gl_order"] = sc_gl_order_var

            variation_errors: list[str] = []
            if not _jax_available():
                variation_errors.append(
                    "JAX is required for variation evaluation. Install `jax` and `jaxlib`."
                )
            if roi_samples_var < 1:
                variation_errors.append("ROI samples must be >= 1.")
            if map_step_var <= 0.0:
                variation_errors.append("Map step must be > 0.")
            if sc_kernel_var == "multi-dipole" and sc_subdip_n_var < 2:
                variation_errors.append("subdip_n must be >= 2 for multi-dipole.")
            if sc_kernel_var == "cellavg":
                variation_errors.append(
                    "near kernel 'cellavg' is not supported for per-magnet p0 variation in this prototype."
                )
            for msg in variation_errors:
                st.error(msg)

            variation_cfg_payload = {
                "sigma_rel_pct": sigma_rel_pct,
                "sigma_phi_deg": sigma_phi_deg,
                "seed": seed,
                "roi_radius_m": roi_radius_var,
                "roi_samples": roi_samples_var,
                "map_radius_m": map_radius_var,
                "map_step_m": map_step_var,
                "target_plane_z": 0.0,
                "sc_cfg": sc_cfg_eval_var,
            }
            variation_cfg_key = json.dumps(variation_cfg_payload, sort_keys=True)

            variation_run_key = re.sub(r"[^0-9A-Za-z_-]+", "_", run_key)
            result_state_key = f"variation_result::{variation_run_key}"
            debug_state_key = f"variation_result_debug::{variation_run_key}"

            run_eval_clicked = st.button(
                "Run variation evaluation",
                key="variation_run_button",
                disabled=bool(variation_errors),
            )
            if run_eval_clicked:
                try:
                    if variation_mtime is None:
                        raise ValueError("Could not resolve results mtime for selected run.")
                    meta_key = _magnetization_cache_key(variation_run.meta)
                    variation_result, variation_debug = _cached_variation_eval(
                        variation_path,
                        variation_mtime,
                        meta_key,
                        variation_cfg_key,
                    )
                    st.session_state[result_state_key] = variation_result
                    st.session_state[debug_state_key] = variation_debug
                    st.success("Variation evaluation completed.")
                except Exception as exc:
                    st.error(f"Variation evaluation failed: {exc}")

            variation_result = st.session_state.get(result_state_key)
            variation_debug = st.session_state.get(debug_state_key, {})
            if variation_result is not None:
                from halbach.perturbation_eval import PerturbationConfig, save_perturbation_result

                result_obj = variation_result
                B0_norm_var = float(np.linalg.norm(result_obj.B0_vec))
                ppm_mean_var = float(np.mean(result_obj.ppm_roi))
                ppm_max_var = float(np.max(np.abs(result_obj.ppm_roi)))
                st.markdown(
                    f"**|B0|**: {B0_norm_var:.6e} T  |  "
                    f"**ppm mean**: {ppm_mean_var:.3f}  |  "
                    f"**ppm max|.|**: {ppm_max_var:.3f}"
                )
                st.write(
                    f"near_kernel: {result_obj.debug.get('sc_near_kernel')}  "
                    f"subdip_n: {result_obj.debug.get('sc_subdip_n')}  "
                    f"near_window: {result_obj.debug.get('sc_near_window')}"
                )
                st.write(
                    "p stats min/max/mean/std/rel_std: "
                    f"{_fmt_num(result_obj.p_stats.get('sc_p_min'))} / "
                    f"{_fmt_num(result_obj.p_stats.get('sc_p_max'))} / "
                    f"{_fmt_num(result_obj.p_stats.get('sc_p_mean'))} / "
                    f"{_fmt_num(result_obj.p_stats.get('sc_p_std'))} / "
                    f"{_fmt_num(result_obj.p_stats.get('sc_p_rel_std'))}"
                )
                if variation_debug:
                    st.caption(
                        f"run: {variation_debug.get('run_name')} ({variation_debug.get('run_dir')})"
                    )

                map_obj = result_obj.map_xy
                map_vals = map_obj.ppm[map_obj.mask]
                if map_vals.size > 0:
                    map_abs = float(np.max(np.abs(map_vals)))
                else:
                    map_abs = 1.0
                fig_var_map = _plot_error_map(
                    map_obj,
                    -map_abs,
                    map_abs,
                    contour_level,
                    "Variation XY(z=0) ppm map",
                )
                st.plotly_chart(fig_var_map, use_container_width=True, key="variation_map_plot")

                save_cols = st.columns(2)
                with save_cols[0]:
                    save_tag_var = st.text_input(
                        "Save tag",
                        value=f"seed{seed}",
                        key="variation_save_tag",
                    )
                with save_cols[1]:
                    default_var_dir = (
                        variation_run.run_dir
                        / "variation_eval"
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_sanitize_tag(save_tag_var)}"
                    )
                    st.caption(f"Output dir: {default_var_dir}")
                if st.button("Save outputs", key="variation_save_button"):
                    try:
                        cfg_obj = PerturbationConfig(
                            sigma_rel_pct=float(variation_cfg_payload["sigma_rel_pct"]),
                            sigma_phi_deg=float(variation_cfg_payload["sigma_phi_deg"]),
                            seed=int(variation_cfg_payload["seed"]),
                            roi_radius_m=float(variation_cfg_payload["roi_radius_m"]),
                            roi_samples=int(variation_cfg_payload["roi_samples"]),
                            map_radius_m=float(variation_cfg_payload["map_radius_m"]),
                            map_step_m=float(variation_cfg_payload["map_step_m"]),
                            target_plane_z=float(variation_cfg_payload["target_plane_z"]),
                            sc_cfg=cast(dict[str, Any], variation_cfg_payload["sc_cfg"]),
                        )
                        saved = save_perturbation_result(
                            result_obj,
                            default_var_dir,
                            cfg_obj,
                            variation_run,
                        )
                        st.success(f"Saved variation outputs to: {default_var_dir}")
                        for name, p in saved.items():
                            st.write(f"{name}: `{p}`")
                    except Exception as exc:
                        st.error(f"Save failed: {exc}")

    with tabs[5]:
        st.subheader("Optimize")
        opt_mode = st.selectbox(
            "Optimization mode",
            ["L-BFGS-B", "DC/CCP (self-consistent linear)"],
            index=0,
            key="opt_mode",
        )
        if opt_mode == "L-BFGS-B":
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

            maxiter = int(
                st.number_input("maxiter", min_value=50, max_value=2000, value=900, step=50)
            )
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
            beta_cols = st.columns([1, 1])
            with beta_cols[0]:
                enable_beta_tilt_x = st.checkbox(
                    "Enable beta_tilt_x",
                    value=False,
                    key="enable_beta_tilt_x",
                    help="Ring-wise local X-axis tilt angle (legacy-alpha + jax only).",
                )
            with beta_cols[1]:
                beta_tilt_x_bound_deg = float(
                    st.number_input(
                        "beta_tilt_x bound (deg)",
                        min_value=0.1,
                        max_value=89.0,
                        value=20.0,
                        step=0.5,
                        key="beta_tilt_x_bound_deg",
                    )
                )
            if enable_beta_tilt_x:
                st.caption("beta_tilt_x requires angle_model=legacy-alpha and grad_backend=jax.")
            st.markdown("**Magnetization model**")
            mag_model = st.selectbox(
                "Magnetization model",
                ["fixed", "self-consistent-easy-axis"],
                index=0,
                key="mag_model",
            )
            sc_chi = 0.0
            sc_Nd = 1.0 / 3.0
            sc_p0 = 1.0
            sc_volume_mm3 = 1000.0
            sc_iters = 30
            sc_omega = 0.6
            sc_near_wr = 0
            sc_near_wz = 1
            sc_near_wphi = 2
            sc_near_kernel = "dipole"
            sc_subdip_n = 2
            sc_gl_order: int | None = None
            if mag_model == "self-consistent-easy-axis":
                with st.expander("Self-consistent settings", expanded=True):
                    sc_chi = float(
                        st.number_input("chi", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
                    )
                    sc_Nd = float(
                        st.number_input(
                            "Nd", min_value=0.0, max_value=1.0, value=1.0 / 3.0, step=0.01
                        )
                    )
                    sc_p0 = float(
                        st.number_input("p0", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
                    )
                    sc_volume_mm3 = float(
                        st.number_input(
                            "volume_mm3",
                            min_value=1.0,
                            max_value=1e6,
                            value=1000.0,
                            step=10.0,
                        )
                    )
                    sc_iters = int(
                        st.number_input("iters", min_value=1, max_value=500, value=30, step=1)
                    )
                    sc_omega = float(
                        st.number_input(
                            "omega", min_value=0.01, max_value=1.0, value=0.6, step=0.01
                        )
                    )
                    st.markdown("**Near window**")
                    sc_near_wr = int(
                        st.number_input("wr", min_value=0, max_value=10, value=0, step=1)
                    )
                    sc_near_wz = int(
                        st.number_input("wz", min_value=0, max_value=10, value=1, step=1)
                    )
                    sc_near_wphi = int(
                        st.number_input("wphi", min_value=0, max_value=10, value=2, step=1)
                    )
                    sc_near_kernel = st.selectbox(
                        "near kernel",
                        ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                        index=0,
                        key="sc_near_kernel",
                    )
                    if sc_near_kernel == "multi-dipole":
                        sc_subdip_n = int(
                            st.number_input("subdip_n", min_value=2, max_value=10, value=2, step=1)
                        )
                    if sc_near_kernel == "gl-double-mixed":
                        gl_choice = st.selectbox(
                            "gl order",
                            ["mixed (2/3)", "2", "3"],
                            index=0,
                            key="sc_gl_order",
                        )
                        if gl_choice in ("2", "3"):
                            sc_gl_order = int(gl_choice)
            jax_issue: str | None = None
            if (angle_model != "legacy-alpha" or grad_backend == "jax") and not jax_available:
                jax_issue = (
                    "JAX is required for the selected angle model/backend. "
                    "Install `jax` and `jaxlib`, or switch to legacy-alpha + analytic."
                )
                st.error(jax_issue)
            sc_errors: list[str] = []
            if mag_model == "self-consistent-easy-axis":
                if not jax_available:
                    sc_errors.append("JAX is required for self-consistent magnetization.")
                if angle_model == "legacy-alpha" and grad_backend != "jax":
                    sc_errors.append("Self-consistent legacy-alpha requires grad_backend=jax.")
                if sc_near_kernel == "multi-dipole" and sc_subdip_n < 2:
                    sc_errors.append("subdip_n must be >= 2 for multi-dipole.")
            if enable_beta_tilt_x:
                if angle_model != "legacy-alpha" or grad_backend != "jax":
                    sc_errors.append(
                        "beta_tilt_x is supported only with angle_model=legacy-alpha and grad_backend=jax."
                    )
                if beta_tilt_x_bound_deg <= 0.0:
                    sc_errors.append("beta_tilt_x bound must be > 0 deg.")
                if mag_model == "self-consistent-easy-axis" and sc_near_kernel not in {
                    "dipole",
                    "multi-dipole",
                }:
                    sc_errors.append(
                        "beta_tilt_x with self-consistent supports near kernel only in {dipole, multi-dipole}."
                    )
            for msg in sc_errors:
                st.error(msg)
            can_start = jax_issue is None and not sc_errors
            fix_center_radius_layers = int(
                st.selectbox(
                    "Fixed center radius layers (z≈0)",
                    [0, 2, 4],
                    index=1,
                    key="fix_center_radius_layers",
                )
            )

            st.markdown("**Radius bounds**")
            r_bounds_enabled = st.checkbox(
                "Enable radius bounds", value=True, key="r_bounds_enabled"
            )
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
                        st.number_input(
                            "N", min_value=1, max_value=512, value=48, step=1, key="gen_N"
                        )
                    )
                    gen_Lz = float(
                        st.number_input(
                            "Lz (m)",
                            min_value=0.01,
                            max_value=2.0,
                            value=0.64,
                            step=0.01,
                            key="gen_Lz",
                        )
                    )
                with gen_cols[1]:
                    gen_R = int(
                        st.number_input(
                            "R", min_value=1, max_value=32, value=3, step=1, key="gen_R"
                        )
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
                        st.number_input(
                            "K", min_value=1, max_value=256, value=24, step=1, key="gen_K"
                        )
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
                                        enable_beta_tilt_x=enable_beta_tilt_x,
                                        beta_tilt_x_bound_deg=beta_tilt_x_bound_deg,
                                        mag_model=mag_model,
                                        sc_chi=sc_chi,
                                        sc_Nd=sc_Nd,
                                        sc_p0=sc_p0,
                                        sc_volume_mm3=sc_volume_mm3,
                                        sc_iters=sc_iters,
                                        sc_omega=sc_omega,
                                        sc_near_wr=sc_near_wr,
                                        sc_near_wz=sc_near_wz,
                                        sc_near_wphi=sc_near_wphi,
                                        sc_near_kernel=sc_near_kernel,
                                        sc_subdip_n=sc_subdip_n,
                                        sc_gl_order=sc_gl_order,
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
                            enable_beta_tilt_x=enable_beta_tilt_x,
                            beta_tilt_x_bound_deg=beta_tilt_x_bound_deg,
                            mag_model=mag_model,
                            sc_chi=sc_chi,
                            sc_Nd=sc_Nd,
                            sc_p0=sc_p0,
                            sc_volume_mm3=sc_volume_mm3,
                            sc_iters=sc_iters,
                            sc_omega=sc_omega,
                            sc_near_wr=sc_near_wr,
                            sc_near_wz=sc_near_wz,
                            sc_near_wphi=sc_near_wphi,
                            sc_near_kernel=sc_near_kernel,
                            sc_subdip_n=sc_subdip_n,
                            sc_gl_order=sc_gl_order,
                            repo_root=ROOT,
                        )
                        st.session_state["opt_job"] = job
                        st.session_state["opt_job_fix_center_radius_layers"] = (
                            fix_center_radius_layers
                        )
                        st.success(f"Started optimization: {job.out_dir}")
                        st.rerun()
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
                    enable_beta_tilt_x=enable_beta_tilt_x,
                    beta_tilt_x_bound_deg=beta_tilt_x_bound_deg,
                    mag_model=mag_model,
                    sc_chi=sc_chi,
                    sc_Nd=sc_Nd,
                    sc_p0=sc_p0,
                    sc_volume_mm3=sc_volume_mm3,
                    sc_iters=sc_iters,
                    sc_omega=sc_omega,
                    sc_near_wr=sc_near_wr,
                    sc_near_wz=sc_near_wz,
                    sc_near_wphi=sc_near_wphi,
                    sc_near_kernel=sc_near_kernel,
                    sc_subdip_n=sc_subdip_n,
                    sc_gl_order=sc_gl_order,
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

        else:
            st.subheader("Optimize (DC/CCP)")
            from halbach.gui.opt_job import (
                build_dc_ccp_sc_command,
                poll_opt_job,
                start_dc_ccp_sc_job,
                stop_opt_job,
                tail_log,
            )

            cvxpy_available = importlib.util.find_spec("cvxpy") is not None
            installed_solvers: list[str] = []
            if cvxpy_available:
                try:
                    import cvxpy as cp

                    installed_solvers = list(cp.installed_solvers())
                except Exception:
                    installed_solvers = []
            if not cvxpy_available:
                st.error("cvxpy is required for DC/CCP optimization.")
            elif not installed_solvers:
                st.error("cvxpy is available but no solvers were detected.")

            dc_job = st.session_state.get("dc_job")
            dc_exit = poll_opt_job(dc_job) if dc_job is not None else None
            dc_job_running = dc_job is not None and dc_exit is None

            st.markdown("**Geometry**")
            dc_N = int(
                st.number_input("N", min_value=1, max_value=512, value=32, step=1, key="dc_N")
            )
            dc_K = int(
                st.number_input("K", min_value=1, max_value=256, value=60, step=1, key="dc_K")
            )
            dc_R = int(st.number_input("R", min_value=1, max_value=32, value=1, step=1, key="dc_R"))
            dc_radius_m = float(
                st.number_input(
                    "Radius (m)",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.2,
                    step=0.01,
                    key="dc_radius_m",
                )
            )
            dc_length_m = float(
                st.number_input(
                    "Length (m)",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.6,
                    step=0.01,
                    key="dc_length_m",
                )
            )

            st.markdown("**ROI**")
            dc_roi_radius_m = float(
                st.number_input(
                    "ROI radius (m)",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.12,
                    step=0.001,
                    key="dc_roi_radius_m",
                )
            )
            dc_roi_grid_n = int(
                st.number_input(
                    "ROI grid N", min_value=3, max_value=201, value=41, step=2, key="dc_roi_grid_n"
                )
            )

            st.markdown("**Objective weights**")
            dc_wx = float(
                st.number_input(
                    "wx", min_value=0.0, max_value=1e3, value=0.0, step=0.1, key="dc_wx"
                )
            )
            dc_wy = float(
                st.number_input(
                    "wy", min_value=0.0, max_value=1e3, value=1.0, step=0.1, key="dc_wy"
                )
            )
            dc_wz = float(
                st.number_input(
                    "wz", min_value=0.0, max_value=1e3, value=0.0, step=0.1, key="dc_wz"
                )
            )

            st.markdown("**CCP settings**")
            dc_phi0 = float(
                st.number_input(
                    "phi0", min_value=-6.3, max_value=6.3, value=0.0, step=0.1, key="dc_phi0"
                )
            )
            dc_delta_nom = float(
                st.number_input(
                    "delta_nom_deg",
                    min_value=0.1,
                    max_value=60.0,
                    value=5.0,
                    step=0.5,
                    key="dc_delta_nom",
                )
            )
            dc_step_enable = st.checkbox("Enable step trust", value=True, key="dc_step_enable")
            dc_delta_step = float(
                st.number_input(
                    "delta_step_deg",
                    min_value=0.1,
                    max_value=60.0,
                    value=2.0,
                    step=0.5,
                    key="dc_delta_step",
                )
            )
            if not dc_step_enable:
                dc_delta_step_val = None
            else:
                dc_delta_step_val = dc_delta_step

            dc_tau0 = float(
                st.number_input(
                    "tau0", min_value=0.0, max_value=1.0, value=1e-4, format="%.1e", key="dc_tau0"
                )
            )
            dc_tau_mult = float(
                st.number_input(
                    "tau_mult", min_value=1.0, max_value=5.0, value=1.2, step=0.1, key="dc_tau_mult"
                )
            )
            dc_tau_max = float(
                st.number_input(
                    "tau_max",
                    min_value=0.0,
                    max_value=1.0,
                    value=1e-1,
                    format="%.1e",
                    key="dc_tau_max",
                )
            )
            dc_iters = int(
                st.number_input(
                    "iters", min_value=1, max_value=200, value=20, step=1, key="dc_iters"
                )
            )
            dc_tol = float(
                st.number_input(
                    "tol", min_value=1e-10, max_value=1e-2, value=1e-6, format="%.1e", key="dc_tol"
                )
            )
            dc_tol_f = float(
                st.number_input(
                    "tol_f",
                    min_value=1e-12,
                    max_value=1e-3,
                    value=1e-9,
                    format="%.1e",
                    key="dc_tol_f",
                )
            )
            st.markdown("**Initial guess (from L-BFGS run)**")
            dc_init_select = st.selectbox(
                "Init run (optional)",
                ["(none)"] + candidates,
                index=0,
                key="dc_init_select",
            )
            dc_init_manual = st.text_input(
                "Init run path (optional)", value="", key="dc_init_manual"
            )
            dc_init_select_val = "" if dc_init_select == "(none)" else dc_init_select
            dc_init_run = _resolve_path(dc_init_select_val, dc_init_manual)
            st.caption("Only delta-rep-x0 / fourier-x0 angle models are accepted.")

            st.markdown("**Regularization**")
            dc_reg_x = float(
                st.number_input(
                    "reg_x", min_value=0.0, max_value=1e3, value=0.0, step=0.1, key="dc_reg_x"
                )
            )
            dc_reg_p = float(
                st.number_input(
                    "reg_p", min_value=0.0, max_value=1e3, value=0.0, step=0.1, key="dc_reg_p"
                )
            )
            dc_reg_z = float(
                st.number_input(
                    "reg_z", min_value=0.0, max_value=1e3, value=0.0, step=0.1, key="dc_reg_z"
                )
            )

            st.markdown("**Self-consistent equality**")
            dc_sc_eq = st.checkbox("Enable self-consistent equalities", value=True, key="dc_sc_eq")
            dc_p_fix = None
            if not dc_sc_eq:
                dc_p_fix = float(
                    st.number_input(
                        "p_fix", min_value=0.0, max_value=10.0, value=1.0, step=0.01, key="dc_p_fix"
                    )
                )

            st.markdown("**Self-consistent parameters**")
            dc_sc_chi = float(
                st.number_input(
                    "sc_chi", min_value=0.0, max_value=10.0, value=0.05, step=0.01, key="dc_sc_chi"
                )
            )
            dc_sc_Nd = float(
                st.number_input(
                    "sc_Nd",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / 3.0,
                    step=0.01,
                    key="dc_sc_Nd",
                )
            )
            dc_sc_p0 = float(
                st.number_input(
                    "sc_p0", min_value=0.0, max_value=10.0, value=1.0, step=0.01, key="dc_sc_p0"
                )
            )
            dc_sc_volume_mm3 = float(
                st.number_input(
                    "sc_volume_mm3",
                    min_value=1.0,
                    max_value=1e6,
                    value=1000.0,
                    step=10.0,
                    key="dc_sc_volume_mm3",
                )
            )

            st.markdown("**Near window**")
            dc_sc_near_wr = int(
                st.number_input(
                    "wr", min_value=0, max_value=10, value=0, step=1, key="dc_sc_near_wr"
                )
            )
            dc_sc_near_wz = int(
                st.number_input(
                    "wz", min_value=0, max_value=10, value=1, step=1, key="dc_sc_near_wz"
                )
            )
            dc_sc_near_wphi = int(
                st.number_input(
                    "wphi", min_value=0, max_value=10, value=2, step=1, key="dc_sc_near_wphi"
                )
            )
            dc_sc_near_kernel = st.selectbox(
                "near kernel",
                ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                index=0,
                key="dc_sc_near_kernel",
            )
            dc_sc_subdip_n = 2
            dc_sc_gl_order: int | None = None
            if dc_sc_near_kernel == "multi-dipole":
                dc_sc_subdip_n = int(
                    st.number_input(
                        "subdip_n", min_value=2, max_value=10, value=2, step=1, key="dc_sc_subdip_n"
                    )
                )
            if dc_sc_near_kernel == "gl-double-mixed":
                gl_choice = st.selectbox(
                    "gl order",
                    ["mixed (2/3)", "2", "3"],
                    index=0,
                    key="dc_sc_gl_order",
                )
                if gl_choice in ("2", "3"):
                    dc_sc_gl_order = int(gl_choice)

            st.markdown("**p bounds**")
            dc_pmin = float(
                st.number_input(
                    "pmin", min_value=0.0, max_value=10.0, value=0.0, step=0.01, key="dc_pmin"
                )
            )
            dc_pmax = float(
                st.number_input(
                    "pmax", min_value=0.0, max_value=10.0, value=2.0, step=0.01, key="dc_pmax"
                )
            )

            st.markdown("**Factor**")
            dc_factor = float(
                st.number_input(
                    "factor",
                    min_value=0.0,
                    max_value=1.0,
                    value=1e-7,
                    format="%.1e",
                    key="dc_factor",
                )
            )

            solver_options = [s for s in ["ECOS", "SCS"] if s in installed_solvers] or ["SCS"]
            dc_solver = st.selectbox("Solver", solver_options, index=0, key="dc_solver")
            dc_verbose = st.checkbox("Verbose log", value=True, key="dc_verbose")

            dc_tag = st.text_input("Output tag", value="dc_ccp_sc", key="dc_tag")
            dc_out_dir = _default_out_dir(dc_tag)
            st.caption(f"Output dir: {dc_out_dir}")

            dc_errors: list[str] = []
            if not cvxpy_available or not installed_solvers:
                dc_errors.append("cvxpy + a solver (ECOS/SCS) is required.")
            if not dc_sc_eq and dc_p_fix is None:
                dc_errors.append("p_fix is required when self-consistent equality is disabled.")
            if dc_sc_near_kernel == "multi-dipole" and dc_sc_subdip_n < 2:
                dc_errors.append("subdip_n must be >= 2 for multi-dipole.")
            if dc_pmin > dc_pmax:
                dc_errors.append("pmin must be <= pmax.")
            for msg in dc_errors:
                st.error(msg)

            dc_can_start = not dc_errors and not dc_job_running

            dc_cols = st.columns(2)
            with dc_cols[0]:
                dc_start = st.button("Start DC/CCP", disabled=not dc_can_start)
            with dc_cols[1]:
                dc_stop = st.button("Stop DC/CCP", disabled=not dc_job_running)

            if dc_start:
                if dc_job_running:
                    st.error("DC/CCP optimization is already running.")
                elif dc_errors:
                    st.error("Cannot start due to validation errors.")
                else:
                    try:
                        job = start_dc_ccp_sc_job(
                            dc_out_dir,
                            N=dc_N,
                            K=dc_K,
                            R=dc_R,
                            radius_m=dc_radius_m,
                            length_m=dc_length_m,
                            roi_radius_m=dc_roi_radius_m,
                            roi_grid_n=dc_roi_grid_n,
                            wx=dc_wx,
                            wy=dc_wy,
                            wz=dc_wz,
                            factor=dc_factor,
                            phi0=dc_phi0,
                            delta_nom_deg=dc_delta_nom,
                            delta_step_deg=dc_delta_step_val,
                            tau0=dc_tau0,
                            tau_mult=dc_tau_mult,
                            tau_max=dc_tau_max,
                            iters=dc_iters,
                            tol=dc_tol,
                            tol_f=dc_tol_f,
                            reg_x=dc_reg_x,
                            reg_p=dc_reg_p,
                            reg_z=dc_reg_z,
                            sc_eq=dc_sc_eq,
                            p_fix=dc_p_fix,
                            sc_chi=dc_sc_chi,
                            sc_Nd=dc_sc_Nd,
                            sc_p0=dc_sc_p0,
                            sc_volume_mm3=dc_sc_volume_mm3,
                            sc_near_wr=dc_sc_near_wr,
                            sc_near_wz=dc_sc_near_wz,
                            sc_near_wphi=dc_sc_near_wphi,
                            sc_near_kernel=dc_sc_near_kernel,
                            sc_subdip_n=dc_sc_subdip_n,
                            sc_gl_order=dc_sc_gl_order,
                            pmin=dc_pmin,
                            pmax=dc_pmax,
                            solver=dc_solver,
                            init_run=dc_init_run or None,
                            verbose=dc_verbose,
                            repo_root=ROOT,
                        )
                        st.session_state["dc_job"] = job
                        st.success(f"Started DC/CCP optimization: {job.out_dir}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to start DC/CCP optimization: {exc}")

            if dc_stop and dc_job is not None:
                stop_opt_job(dc_job)
                st.warning("Terminate signal sent.")

            if dc_job is not None:
                dc_exit = poll_opt_job(dc_job)
                st.write(f"Status: {'running' if dc_exit is None else f'exit {dc_exit}'}")
                st.write(f"Out dir: `{dc_job.out_dir}`")
                cmd = build_dc_ccp_sc_command(
                    dc_job.out_dir,
                    N=dc_N,
                    K=dc_K,
                    R=dc_R,
                    radius_m=dc_radius_m,
                    length_m=dc_length_m,
                    roi_radius_m=dc_roi_radius_m,
                    roi_grid_n=dc_roi_grid_n,
                    wx=dc_wx,
                    wy=dc_wy,
                    wz=dc_wz,
                    factor=dc_factor,
                    phi0=dc_phi0,
                    delta_nom_deg=dc_delta_nom,
                    delta_step_deg=dc_delta_step_val,
                    tau0=dc_tau0,
                    tau_mult=dc_tau_mult,
                    tau_max=dc_tau_max,
                    iters=dc_iters,
                    tol=dc_tol,
                    tol_f=dc_tol_f,
                    reg_x=dc_reg_x,
                    reg_p=dc_reg_p,
                    reg_z=dc_reg_z,
                    sc_eq=dc_sc_eq,
                    p_fix=dc_p_fix,
                    sc_chi=dc_sc_chi,
                    sc_Nd=dc_sc_Nd,
                    sc_p0=dc_sc_p0,
                    sc_volume_mm3=dc_sc_volume_mm3,
                    sc_near_wr=dc_sc_near_wr,
                    sc_near_wz=dc_sc_near_wz,
                    sc_near_wphi=dc_sc_near_wphi,
                    sc_near_kernel=dc_sc_near_kernel,
                    sc_subdip_n=dc_sc_subdip_n,
                    sc_gl_order=dc_sc_gl_order,
                    pmin=dc_pmin,
                    pmax=dc_pmax,
                    solver=dc_solver,
                    init_run=dc_init_run or None,
                    verbose=dc_verbose,
                )
                st.code(" ".join(cmd))

                dc_log = tail_log(dc_job.log_path, n_lines=200)
                st.text_area("dc_ccp_sc.log (tail)", dc_log, height=240)

            if dc_job_running:
                dc_auto_refresh = st.checkbox("Auto-refresh log", value=True, key="dc_auto_refresh")
                dc_refresh_secs = float(
                    st.number_input(
                        "Log refresh (s)",
                        min_value=0.5,
                        max_value=10.0,
                        value=1.0,
                        step=0.5,
                        key="dc_refresh_secs",
                    )
                )
                if dc_auto_refresh:
                    time.sleep(max(0.1, dc_refresh_secs))
                    st.rerun()


if __name__ == "__main__":
    main()
