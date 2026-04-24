from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.gui_config import (
    GUI_CONFIG_KEYS,
    GUI_DEFAULTS_PATH,
    apply_gui_config_values,
    default_gui_config_export_path,
    list_gui_config_export_paths,
    load_gui_config_values,
    merge_gui_config_values,
    normalize_gui_choice,
    write_gui_config,
)
from app.plotly_views import (
    build_scene_figure,
    combine_scene_ranges,
    common_ppm_limits,
    plot_cross_section,
    plot_error_map,
)
from halbach.app_services import (
    Map2DPayload,
    Map2DRequest,
    RunSummary,
    Scene3DPayload,
    Scene3DRequest,
    build_run_delta_summary,
    build_run_summary,
    list_run_candidates,
    load_map2d_payload,
    load_run_bundle,
    load_scene3d_payload,
)
from halbach.app_services import magnetization_cache_key as service_magnetization_cache_key
from halbach.app_services import meta_mtime as service_meta_mtime
from halbach.app_services import resolve_results_path as service_resolve_results_path
from halbach.app_services import resolve_selected_path
from halbach.app_services import results_mtime as service_results_mtime
from halbach.app_services import try_load_run_bundle

if TYPE_CHECKING:
    from halbach.perturbation_eval import PerturbationResult
    from halbach.run_types import RunBundle

PlotlyMode = Literal["fast", "pretty", "cubes", "cubes_arrows"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HUMAN_OVERLAY_OBJ = "10688_GenericMale_v2.obj"
DEFAULT_COIL_OVERLAY_DIR = "coil"
_GUI_CONFIG_DEFAULTS_STATE_KEY = "_gui_config_defaults"
_GUI_CONFIG_WARNING_STATE_KEY = "_gui_config_warnings"
_GUI_CONFIG_WARNED_KEYS_STATE_KEY = "_gui_config_warned_keys"
ChoiceT = TypeVar("ChoiceT")


def _get_gui_defaults() -> dict[str, object]:
    defaults = st.session_state.get(_GUI_CONFIG_DEFAULTS_STATE_KEY)
    if defaults is None:
        defaults = load_gui_config_values(GUI_DEFAULTS_PATH)
        st.session_state[_GUI_CONFIG_DEFAULTS_STATE_KEY] = defaults
    return cast(dict[str, object], defaults)


def _seed_gui_state(defaults: Mapping[str, object]) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _queue_gui_config_warning(key: str, message: str) -> None:
    warned_keys = cast(
        list[str], st.session_state.setdefault(_GUI_CONFIG_WARNED_KEYS_STATE_KEY, [])
    )
    if key in warned_keys:
        return
    warned_keys.append(key)
    warnings = cast(list[str], st.session_state.setdefault(_GUI_CONFIG_WARNING_STATE_KEY, []))
    warnings.append(message)


def _consume_gui_config_warnings() -> list[str]:
    warnings = st.session_state.pop(_GUI_CONFIG_WARNING_STATE_KEY, [])
    if not isinstance(warnings, list):
        return []
    return [str(message) for message in warnings]


def _resolve_gui_config_import_path(saved_config: str, custom_path: str) -> Path | None:
    raw = custom_path.strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return candidate
    if not saved_config or saved_config == "(none)":
        return None
    candidate = Path(saved_config)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _ensure_gui_choice_state(
    key: str,
    options: Sequence[ChoiceT],
    *,
    fallback: ChoiceT,
    label: str,
    warn_on_blank: bool = False,
) -> None:
    current = st.session_state.get(key)
    resolved, should_warn = normalize_gui_choice(
        current,
        options,
        fallback=fallback,
        warn_on_blank=warn_on_blank,
    )
    if current == resolved:
        return
    st.session_state[key] = resolved
    if should_warn:
        _queue_gui_config_warning(
            key,
            f"{label} config value {current!r} is not available; falling back to {resolved!r}.",
        )


def _gui_selectbox(
    label: str,
    options: Sequence[ChoiceT],
    *,
    key: str,
    fallback: ChoiceT,
    warn_on_blank: bool = False,
    **kwargs: Any,
) -> ChoiceT:
    _ensure_gui_choice_state(
        key,
        options,
        fallback=fallback,
        label=label,
        warn_on_blank=warn_on_blank,
    )
    return cast(ChoiceT, st.selectbox(label, options, key=key, **kwargs))


def _gui_radio(
    label: str,
    options: Sequence[ChoiceT],
    *,
    key: str,
    fallback: ChoiceT,
    warn_on_blank: bool = False,
    **kwargs: Any,
) -> ChoiceT:
    _ensure_gui_choice_state(
        key,
        options,
        fallback=fallback,
        label=label,
        warn_on_blank=warn_on_blank,
    )
    return cast(ChoiceT, st.radio(label, options, key=key, **kwargs))


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
    return resolve_selected_path(selected, manual)


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", tag.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _default_out_dir(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"{stamp}_{_sanitize_tag(tag)}"


def _resolve_results_path(path: Path) -> Path:
    return service_resolve_results_path(path)


def _results_mtime(path: Path) -> float:
    return service_results_mtime(path)


def _meta_mtime(path: Path) -> float:
    return service_meta_mtime(path)


def _magnetization_cache_key(meta: dict[str, object]) -> str:
    return service_magnetization_cache_key(meta)


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

    pending_gui_config_values = st.session_state.pop("pending_gui_config_values", None)
    pending_gui_config_keys = st.session_state.pop("pending_gui_config_keys", None)
    if pending_gui_config_values is not None:
        try:
            apply_gui_config_values(
                st.session_state,
                cast(Mapping[str, object], pending_gui_config_values),
                selected_keys=cast(Sequence[str] | None, pending_gui_config_keys),
            )
        except Exception as exc:
            _queue_gui_config_warning(
                "gui_config_import_apply",
                f"GUI config apply failed: {exc}",
            )


@st.cache_data(show_spinner=False)
def _cached_load_run(path_text: str, mtime: float, _meta_mtime: float) -> RunBundle:
    return load_run_bundle(path_text)


@st.cache_data(show_spinner=False)
def _cached_run_summary(path_text: str, mtime: float, _meta_mtime: float) -> RunSummary:
    run = load_run_bundle(path_text)
    return build_run_summary(run)


@st.cache_data(show_spinner=False)
def _cached_error_map(
    path_text: str,
    mtime: float,
    _meta_key: str,
    roi_r: float,
    step: float,
    mag_model_eval: str,
    sc_cfg_key: str,
) -> Map2DPayload:
    sc_cfg_override = cast(dict[str, object] | None, json.loads(sc_cfg_key) if sc_cfg_key else None)
    return load_map2d_payload(
        Map2DRequest(
            run_path=path_text,
            roi_r=roi_r,
            step=step,
            mag_model_eval=cast(
                Literal["auto", "fixed", "self-consistent-easy-axis"], mag_model_eval
            ),
            sc_cfg_override=sc_cfg_override,
        )
    )


@st.cache_data(show_spinner=False)
def _cached_scene_payload(
    primary_path: str,
    primary_mtime: float,
    primary_meta_mtime: float,
    secondary_path: str,
    secondary_mtime: float,
    secondary_meta_mtime: float,
    stride: int,
    hide_x_negative: bool,
) -> Scene3DPayload:
    _ = (primary_mtime, primary_meta_mtime, secondary_mtime, secondary_meta_mtime)
    return load_scene3d_payload(
        Scene3DRequest(
            primary_path=primary_path,
            secondary_path=secondary_path or None,
            stride=stride,
            hide_x_negative=hide_x_negative,
        )
    )


@st.cache_data(show_spinner=False)
def _cached_variation_eval(
    path_text: str,
    mtime: float,
    _meta_key: str,
    cfg_key: str,
) -> tuple[PerturbationResult, dict[str, object]]:
    from halbach.perturbation_eval import PerturbationConfig, run_perturbation_case

    run = load_run_bundle(path_text)
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
    near_window = cast(dict[str, Any], defaults.get("near_window", {}))
    seed_values: dict[str, object] = {
        "variation_sc_chi": float(defaults.get("chi", 0.0)),
        "variation_sc_Nd": float(defaults.get("Nd", 1.0 / 3.0)),
        "variation_sc_p0": float(defaults.get("p0", 1.0)),
        "variation_sc_volume_mm3": float(defaults.get("volume_mm3", 1000.0)),
        "variation_sc_iters": int(defaults.get("iters", 30)),
        "variation_sc_omega": float(defaults.get("omega", 0.6)),
        "variation_sc_near_wr": int(near_window.get("wr", 0)),
        "variation_sc_near_wz": int(near_window.get("wz", 1)),
        "variation_sc_near_wphi": int(near_window.get("wphi", 2)),
        "variation_sc_near_kernel": str(defaults.get("near_kernel", "dipole")),
        "variation_sc_subdip_n": int(defaults.get("subdip_n", 2)),
        "variation_sc_gl_order": (
            str(defaults["gl_order"])
            if "gl_order" in defaults and defaults["gl_order"] is not None
            else "mixed (2/3)"
        ),
    }
    if st.session_state.get("variation_sc_seed_run") == run_key:
        return

    should_seed = False
    previous_seed = st.session_state.get("variation_sc_seed_values")
    if isinstance(previous_seed, dict):
        should_seed = all(
            st.session_state.get(key) == value for key, value in previous_seed.items()
        )
    else:
        should_seed = not any(key in st.session_state for key in seed_values)

    if should_seed:
        for key, value in seed_values.items():
            st.session_state[key] = value

    st.session_state["variation_sc_seed_run"] = run_key
    st.session_state["variation_sc_seed_values"] = seed_values


def _try_load(
    path_text: str,
) -> tuple[RunBundle | None, float | None, float | None, str | None]:
    return try_load_run_bundle(path_text)


def _ppm_stats(m: Map2DPayload) -> tuple[float, float]:
    return float(m.summary_stats["ppm_mean"]), float(m.summary_stats["ppm_max_abs"])


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
    m: Map2DPayload, vmin: float, vmax: float, contour_level: float, title: str
) -> go.Figure:
    return plot_error_map(m, vmin=vmin, vmax=vmax, contour_level=contour_level, title=title)


def _plot_cross_section(m: Map2DPayload, vmin: float, vmax: float, title: str) -> go.Figure:
    return plot_cross_section(m, vmin=vmin, vmax=vmax, title=title)


def _render_readonly_sidebar(candidates: list[str]) -> dict[str, object]:
    with st.sidebar:
        st.header("Run Selection")
        runs_dir = ROOT / "runs"
        if candidates:
            st.caption(f"Found {len(candidates)} run(s) under {runs_dir}")
        else:
            st.caption(f"No runs found under {runs_dir}")
        init_select = _gui_selectbox(
            "Initial run (runs/)",
            [""] + candidates,
            key="init_select",
            fallback="",
        )
        init_path_text = st.text_input("Initial run path", key="init_path")
        opt_select = _gui_selectbox(
            "Optimized run (runs/)",
            [""] + candidates,
            key="opt_select",
            fallback="",
        )
        opt_path_text = st.text_input("Optimized run path", key="opt_path")
        if st.button("Reload cache"):
            st.cache_data.clear()

        st.header("2D Settings")
        roi_r = float(
            st.number_input(
                "ROI radius (m)",
                min_value=0.001,
                max_value=0.2,
                step=0.001,
                format="%.4f",
                key="map_roi_radius_m",
            )
        )
        step = float(
            st.number_input(
                "Step (m)",
                min_value=0.001,
                max_value=0.02,
                step=0.001,
                format="%.4f",
                key="map_step_m",
            )
        )
        auto_limit = st.checkbox("Auto ppm limit", key="map_auto_limit")
        ppm_limit: float | None
        if auto_limit:
            ppm_limit = None
        else:
            ppm_limit = float(
                st.number_input(
                    "PPM limit",
                    min_value=5.0,
                    max_value=20000.0,
                    step=500.0,
                    key="map_ppm_limit",
                )
            )
        contour_level = float(
            st.number_input(
                "Contour level (ppm)",
                min_value=1.0,
                max_value=20000.0,
                step=100.0,
                key="map_contour_level_ppm",
            )
        )
        st.header("2D Magnetization Eval")
        mag_model_eval = _gui_selectbox(
            "Magnetization model (2D)",
            ["auto", "fixed", "self-consistent-easy-axis"],
            key="map_mag_model_eval",
            fallback="auto",
        )
        sc_cfg_eval: dict[str, object] | None = None
        sc_cfg_key = ""
        if mag_model_eval == "self-consistent-easy-axis":
            with st.expander("Self-consistent eval settings", expanded=True):
                sc_chi = float(
                    st.number_input(
                        "chi (2D eval)",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                        key="map_sc_chi",
                    )
                )
                sc_Nd = float(
                    st.number_input(
                        "Nd (2D eval)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        key="map_sc_Nd",
                    )
                )
                sc_p0 = float(
                    st.number_input(
                        "p0 (2D eval)",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                        key="map_sc_p0",
                    )
                )
                sc_volume_mm3 = float(
                    st.number_input(
                        "volume_mm3 (2D eval)",
                        min_value=1.0,
                        max_value=1e6,
                        step=10.0,
                        key="map_sc_volume_mm3",
                    )
                )
                sc_iters = int(
                    st.number_input(
                        "iters (2D eval)",
                        min_value=1,
                        max_value=500,
                        step=1,
                        key="map_sc_iters",
                    )
                )
                sc_omega = float(
                    st.number_input(
                        "omega (2D eval)",
                        min_value=0.01,
                        max_value=1.0,
                        step=0.01,
                        key="map_sc_omega",
                    )
                )
                st.markdown("**Near window (2D eval)**")
                sc_near_wr = int(
                    st.number_input(
                        "wr (2D eval)",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="map_sc_near_wr",
                    )
                )
                sc_near_wz = int(
                    st.number_input(
                        "wz (2D eval)",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="map_sc_near_wz",
                    )
                )
                sc_near_wphi = int(
                    st.number_input(
                        "wphi (2D eval)",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key="map_sc_near_wphi",
                    )
                )
                sc_near_kernel = _gui_selectbox(
                    "near kernel (2D eval)",
                    ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                    key="map_sc_near_kernel",
                    fallback="dipole",
                )
                sc_subdip_n = 2
                if sc_near_kernel == "multi-dipole":
                    sc_subdip_n = int(
                        st.number_input(
                            "subdip_n (2D eval)",
                            min_value=2,
                            max_value=10,
                            step=1,
                            key="map_sc_subdip_n",
                        )
                    )
                sc_gl_order: int | None = None
                if sc_near_kernel == "gl-double-mixed":
                    gl_choice = _gui_selectbox(
                        "gl order (2D eval)",
                        ["mixed (2/3)", "2", "3"],
                        key="map_sc_gl_order",
                        fallback="mixed (2/3)",
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
        view_target = _gui_radio(
            "3D run",
            ["initial", "optimized"],
            key="view_target",
            fallback="optimized",
        )
        mode = cast(
            PlotlyMode,
            _gui_selectbox(
                "Mode",
                ["fast", "pretty", "cubes", "cubes_arrows"],
                key="view_mode",
                fallback="fast",
            ),
        )
        stride = int(
            st.number_input("Stride", min_value=1, max_value=64, step=1, key="view_stride")
        )
        hide_x_negative = st.checkbox("Hide x < 0", key="view_hide_x_negative")
        magnet_size_mm = float(
            st.number_input(
                "Magnet size (mm)",
                min_value=1.0,
                max_value=50.0,
                step=1.0,
                key="view_magnet_size_mm",
            )
        )
        magnet_thickness_mm = float(
            st.number_input(
                "Magnet thickness (mm)",
                min_value=0.5,
                max_value=20.0,
                step=0.5,
                key="view_magnet_thickness_mm",
            )
        )
        arrow_length_mm = float(
            st.number_input(
                "Arrow length (mm)",
                min_value=1.0,
                max_value=200.0,
                step=1.0,
                key="view_arrow_length_mm",
            )
        )
        arrow_head_angle_deg = float(
            st.number_input(
                "Arrow head angle (deg)",
                min_value=5.0,
                max_value=90.0,
                step=1.0,
                key="view_arrow_head_angle_deg",
            )
        )
        magnet_surface_palette = _magnet_surface_color_options()
        magnet_surface_label = _gui_selectbox(
            "Magnet surface color",
            list(magnet_surface_palette.keys()),
            key="view_magnet_surface_label",
            fallback="Current gray",
        )

    return {
        "init_select": init_select,
        "init_path_text": init_path_text,
        "opt_select": opt_select,
        "opt_path_text": opt_path_text,
        "roi_r": roi_r,
        "step": step,
        "ppm_limit": ppm_limit,
        "contour_level": contour_level,
        "mag_model_eval": mag_model_eval,
        "sc_cfg_key": sc_cfg_key,
        "view_target": view_target,
        "mode": mode,
        "stride": stride,
        "hide_x_negative": hide_x_negative,
        "magnet_size_mm": magnet_size_mm,
        "magnet_thickness_mm": magnet_thickness_mm,
        "arrow_length_mm": arrow_length_mm,
        "arrow_head_angle_deg": arrow_head_angle_deg,
        "magnet_surface_color": magnet_surface_palette[magnet_surface_label],
    }


def _render_overview_tab(
    init_summary: RunSummary | None,
    opt_summary: RunSummary | None,
    init_map: Map2DPayload | None,
    opt_map: Map2DPayload | None,
    delta_summary: dict[str, object] | None,
) -> None:
    st.subheader("Runs")
    cols = st.columns(2)

    def render_summary(
        summary: RunSummary | None,
        m: Map2DPayload | None,
        label: str,
    ) -> None:
        if summary is None:
            st.info(f"{label} run is not loaded.")
            return

        st.markdown(f"**{label} name**: {summary.name}")
        st.markdown(f"**Results**: `{summary.results_path}`")
        st.markdown(f"**Meta**: `{summary.meta_path}`")

        if summary.framework == "dc":
            st.markdown("**Framework**: DC/CCP")
            for key in (
                "p_opt_min",
                "p_opt_mean",
                "p_opt_max",
                "p_sc_post_min",
                "p_sc_post_mean",
                "p_sc_post_max",
                "z_norm_min",
                "z_norm_mean",
                "z_norm_max",
            ):
                if key in summary.key_stats:
                    st.write(f"{key}: {_fmt_num(summary.key_stats[key], '.6f')}")
            return

        b0 = summary.key_stats.get("B0_T")
        if isinstance(b0, int | float):
            st.metric("B0_T (mT)", f"{float(b0) * 1e3:.3f}")
        st.write(
            f"r_bases min/max: {_fmt_num(summary.key_stats.get('r_bases_min'), '.6f')}, "
            f"{_fmt_num(summary.key_stats.get('r_bases_max'), '.6f')}"
        )
        st.write(
            f"alphas min/max: {_fmt_num(summary.key_stats.get('alphas_min'), '.6f')}, "
            f"{_fmt_num(summary.key_stats.get('alphas_max'), '.6f')}"
        )
        if m is not None:
            ppm_mean, ppm_maxabs = _ppm_stats(m)
            st.write(f"ppm mean: {ppm_mean:.2f}")
            st.write(f"ppm max|.|: {ppm_maxabs:.2f}")
            debug = m.magnetization_debug
            if debug.get("model_effective") == "self-consistent-easy-axis":
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
        render_summary(init_summary, init_map, "Initial")
    with cols[1]:
        render_summary(opt_summary, opt_map, "Optimized")

    if init_summary is not None and opt_summary is not None:
        st.subheader("Delta (optimized - initial)")
        if delta_summary is None or not bool(delta_summary.get("available", False)):
            if delta_summary is None:
                st.write("Delta summary is not available.")
            elif delta_summary.get("reason") == "dc-framework":
                st.write("Delta summary is not available for DC/CCP runs.")
            else:
                st.write("Run shapes differ; delta summary skipped.")
        else:
            st.write(
                f"alphas delta min/max: {_fmt_num(delta_summary.get('alphas_delta_min'), '.6f')}, "
                f"{_fmt_num(delta_summary.get('alphas_delta_max'), '.6f')}"
            )
            st.write(
                f"r_bases delta min/max: {_fmt_num(delta_summary.get('r_bases_delta_min'), '.6f')}, "
                f"{_fmt_num(delta_summary.get('r_bases_delta_max'), '.6f')}"
            )


def _render_2d_tab(
    init_map: Map2DPayload | None,
    opt_map: Map2DPayload | None,
    *,
    vmin: float,
    vmax: float,
    contour_level: float,
) -> None:
    st.subheader("2D Error Maps (ppm)")
    if init_map is None and opt_map is None:
        st.info("Select a run to render 2D maps.")
        return

    maps_to_show: list[tuple[str, Map2DPayload, str]] = []
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
        return

    map_cols = st.columns(2)
    for col, (label, map_data, key_suffix) in zip(map_cols, maps_to_show, strict=False):
        with col:
            fig_map = _plot_error_map(map_data, vmin, vmax, contour_level, label)
            st.plotly_chart(fig_map, use_container_width=True, key=f"map_{key_suffix}")

    line_cols = st.columns(2)
    for col, (label, map_data, key_suffix) in zip(line_cols, maps_to_show, strict=False):
        with col:
            fig_line = _plot_cross_section(map_data, vmin, vmax, f"{label} y=0")
            st.plotly_chart(fig_line, use_container_width=True, key=f"line_{key_suffix}")


def _render_3d_tab(
    init_scene: Scene3DPayload | None,
    opt_scene: Scene3DPayload | None,
    compare_scene: Scene3DPayload | None,
    *,
    view_target: str,
    mode: PlotlyMode,
    magnet_size_mm: float,
    magnet_thickness_mm: float,
    arrow_length_mm: float,
    arrow_head_angle_deg: float,
    magnet_surface_color: str,
) -> None:
    st.subheader("3D Magnet View")
    view_tabs = st.tabs(["Single", "Compare"])
    with view_tabs[0]:
        target_scene = init_scene if view_target == "initial" else opt_scene
        if target_scene is None:
            st.info("Select a run for 3D rendering.")
        else:
            fig = build_scene_figure(
                target_scene,
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
        if compare_scene is None or init_scene is None or opt_scene is None:
            st.info("Both initial and optimized runs are required for 3D comparison.")
        else:
            camera_presets = _camera_presets()
            preset = _gui_selectbox(
                "Camera preset",
                list(camera_presets.keys()),
                key="compare_camera_preset",
                fallback="Isometric",
            )
            scene_camera = camera_presets[preset]
            fig_compare = build_scene_figure(
                compare_scene,
                mode=mode,
                magnet_size_m=magnet_size_mm / 1000.0,
                magnet_thickness_m=magnet_thickness_mm / 1000.0,
                arrow_length_m=arrow_length_mm / 1000.0,
                arrow_head_angle_deg=arrow_head_angle_deg,
                scene_camera=scene_camera,
                height=700,
                magnet_surface_color=magnet_surface_color,
            )
            st.caption("Initial and optimized runs are overlaid in a shared 3D scene.")
            st.plotly_chart(fig_compare, use_container_width=True, key="magnet_compare")


def main() -> None:
    st.set_page_config(page_title="Halbach Run Viewer", layout="wide")
    st.title("Halbach Run Viewer")

    try:
        gui_defaults = _get_gui_defaults()
    except Exception as exc:
        st.error(f"GUI config load error: {exc}")
        return
    _seed_gui_state(gui_defaults)

    candidates = [candidate.path for candidate in list_run_candidates(ROOT / "runs")]

    if "opt_job" not in st.session_state:
        st.session_state["opt_job"] = None
    if "gen_run_code" not in st.session_state:
        st.session_state["gen_run_code"] = None
        st.session_state["gen_run_output"] = ""
        st.session_state["gen_run_dir"] = ""
    if "flash_message" not in st.session_state:
        st.session_state["flash_message"] = ""

    _apply_pending_selection_updates()

    sidebar_state = _render_readonly_sidebar(candidates)

    init_path = _resolve_path(
        cast(str, sidebar_state["init_select"]),
        cast(str, sidebar_state["init_path_text"]),
    )
    opt_path = _resolve_path(
        cast(str, sidebar_state["opt_select"]),
        cast(str, sidebar_state["opt_path_text"]),
    )

    init_run, init_mtime, init_meta_mtime, init_err = _try_load(init_path)
    opt_run, opt_mtime, opt_meta_mtime, opt_err = _try_load(opt_path)

    if init_err:
        st.error(f"Initial run load error: {init_err}")
    if opt_err:
        st.error(f"Optimized run load error: {opt_err}")

    init_summary: RunSummary | None = None
    opt_summary: RunSummary | None = None
    init_map: Map2DPayload | None = None
    opt_map: Map2DPayload | None = None
    init_scene: Scene3DPayload | None = None
    opt_scene: Scene3DPayload | None = None
    compare_scene: Scene3DPayload | None = None

    roi_r = float(sidebar_state["roi_r"])
    step = float(sidebar_state["step"])
    mag_model_eval = cast(str, sidebar_state["mag_model_eval"])
    sc_cfg_key = cast(str, sidebar_state["sc_cfg_key"])
    stride = int(sidebar_state["stride"])
    hide_x_negative = bool(sidebar_state["hide_x_negative"])

    if init_run is not None and init_mtime is not None and init_meta_mtime is not None:
        try:
            init_summary = _cached_run_summary(init_path, init_mtime, init_meta_mtime)
        except Exception as exc:
            st.error(f"Initial summary error: {exc}")
        try:
            init_key = _magnetization_cache_key(init_run.meta)
            init_map = _cached_error_map(
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
        try:
            init_scene = _cached_scene_payload(
                init_path,
                init_mtime,
                init_meta_mtime,
                "",
                0.0,
                0.0,
                stride,
                hide_x_negative,
            )
        except Exception as exc:
            st.error(f"Initial 3D scene error: {exc}")

    if opt_run is not None and opt_mtime is not None and opt_meta_mtime is not None:
        try:
            opt_summary = _cached_run_summary(opt_path, opt_mtime, opt_meta_mtime)
        except Exception as exc:
            st.error(f"Optimized summary error: {exc}")
        try:
            opt_key = _magnetization_cache_key(opt_run.meta)
            opt_map = _cached_error_map(
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
        try:
            opt_scene = _cached_scene_payload(
                opt_path,
                opt_mtime,
                opt_meta_mtime,
                "",
                0.0,
                0.0,
                stride,
                hide_x_negative,
            )
        except Exception as exc:
            st.error(f"Optimized 3D scene error: {exc}")

    delta_summary = (
        None if init_run is None or opt_run is None else build_run_delta_summary(init_run, opt_run)
    )

    if (
        init_run is not None
        and opt_run is not None
        and init_mtime is not None
        and init_meta_mtime is not None
        and opt_mtime is not None
        and opt_meta_mtime is not None
    ):
        try:
            compare_scene = _cached_scene_payload(
                init_path,
                init_mtime,
                init_meta_mtime,
                opt_path,
                opt_mtime,
                opt_meta_mtime,
                stride,
                hide_x_negative,
            )
        except Exception as exc:
            st.error(f"3D compare scene error: {exc}")

    vmin = vmax = 0.0
    available_maps = [payload for payload in (init_map, opt_map) if payload is not None]
    if available_maps:
        vmin, vmax = common_ppm_limits(
            available_maps,
            limit_ppm=cast(float | None, sidebar_state["ppm_limit"]),
            symmetric=True,
        )

    tabs = st.tabs(["Overview", "2D", "3D", "Human Overlay", "Variation", "Optimize"])

    with tabs[0]:
        _render_overview_tab(init_summary, opt_summary, init_map, opt_map, delta_summary)

    with tabs[1]:
        _render_2d_tab(
            init_map,
            opt_map,
            vmin=vmin,
            vmax=vmax,
            contour_level=float(sidebar_state["contour_level"]),
        )

    with tabs[2]:
        _render_3d_tab(
            init_scene,
            opt_scene,
            compare_scene,
            view_target=cast(str, sidebar_state["view_target"]),
            mode=cast(PlotlyMode, sidebar_state["mode"]),
            magnet_size_mm=float(sidebar_state["magnet_size_mm"]),
            magnet_thickness_mm=float(sidebar_state["magnet_thickness_mm"]),
            arrow_length_mm=float(sidebar_state["arrow_length_mm"]),
            arrow_head_angle_deg=float(sidebar_state["arrow_head_angle_deg"]),
            magnet_surface_color=cast(str, sidebar_state["magnet_surface_color"]),
        )

    with tabs[3]:
        st.subheader("Human Overlay")
        st.caption(
            "Overlay a Wavefront OBJ body mesh on top of the current magnet layout for fit and scale checks."
        )
        overlay_run_target = _gui_radio(
            "Overlay run",
            ["initial", "optimized"],
            key="human_overlay_run_target",
            fallback="optimized",
            horizontal=True,
        )
        overlay_run = init_run if overlay_run_target == "initial" else opt_run
        overlay_path = init_path if overlay_run_target == "initial" else opt_path
        overlay_mtime = init_mtime if overlay_run_target == "initial" else opt_mtime
        overlay_meta = init_meta_mtime if overlay_run_target == "initial" else opt_meta_mtime

        obj_path_input = st.text_input(
            "OBJ path",
            key="human_overlay_obj_path",
        )
        overlay_obj_path = _resolve_human_overlay_obj_path(obj_path_input)
        st.caption(f"Resolved OBJ path: `{overlay_obj_path}`")

        overlay_cols = st.columns(3)
        with overlay_cols[0]:
            overlay_unit = _gui_selectbox(
                "OBJ unit",
                ["mm", "cm", "m"],
                key="human_overlay_unit",
                fallback="cm",
            )
        with overlay_cols[1]:
            overlay_opacity = float(
                st.slider(
                    "Body opacity",
                    min_value=0.05,
                    max_value=1.0,
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
                    step=1.0,
                    key="human_overlay_rz_deg",
                )
            )

        camera_presets = _camera_presets()
        overlay_camera_preset = _gui_selectbox(
            "Camera preset",
            list(camera_presets.keys()),
            key="human_overlay_camera_preset",
            fallback="Isometric",
        )
        show_coil_overlay = st.checkbox("Show coil overlay", key="human_overlay_show_coil")
        coil_candidates = _scan_coil_npz_files(ROOT / DEFAULT_COIL_OVERLAY_DIR)
        coil_selected = ""
        coil_line_width = 4.0
        coil_rotation_x_deg = 0
        if show_coil_overlay:
            if coil_candidates:
                coil_selected = _gui_selectbox(
                    "Coil file",
                    coil_candidates,
                    key="human_overlay_coil_file",
                    fallback=coil_candidates[0],
                )
                coil_line_width = float(
                    st.number_input(
                        "Coil line width",
                        min_value=1.0,
                        max_value=12.0,
                        step=0.5,
                        key="human_overlay_coil_line_width",
                    )
                )
                coil_rotation_x_deg = int(
                    _gui_selectbox(
                        "Coil X rotation",
                        [0, 90, 180, 270],
                        key="human_overlay_coil_rotation_x_deg",
                        fallback=0,
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
                overlay_scene = None
                if overlay_mtime is not None and overlay_meta is not None:
                    overlay_scene = _cached_scene_payload(
                        overlay_path,
                        overlay_mtime,
                        overlay_meta,
                        "",
                        0.0,
                        0.0,
                        stride,
                        hide_x_negative,
                    )
                if overlay_scene is None:
                    st.info("Could not build a 3D scene for the selected overlay run.")
                else:
                    overlay_range_sources = [
                        np.asarray(overlay_scene.primary.centers, dtype=np.float64),
                        np.asarray(transformed_overlay_mesh.vertices, dtype=np.float64),
                    ]
                    if coil_overlay is not None:
                        overlay_range_sources.append(
                            np.asarray(coil_overlay.points_xyz, dtype=np.float64)
                        )
                    overlay_scene_payload = Scene3DPayload(
                        primary=overlay_scene.primary,
                        secondary=overlay_scene.secondary,
                        scene_ranges=combine_scene_ranges(overlay_range_sources),
                        view_metadata=dict(overlay_scene.view_metadata),
                    )
                    overlay_fig = build_scene_figure(
                        overlay_scene_payload,
                        mode=cast(PlotlyMode, sidebar_state["mode"]),
                        magnet_size_m=float(sidebar_state["magnet_size_mm"]) / 1000.0,
                        magnet_thickness_m=float(sidebar_state["magnet_thickness_mm"]) / 1000.0,
                        arrow_length_m=float(sidebar_state["arrow_length_mm"]) / 1000.0,
                        arrow_head_angle_deg=float(sidebar_state["arrow_head_angle_deg"]),
                        scene_camera=camera_presets[overlay_camera_preset],
                        height=1520,
                        overlay_mesh=transformed_overlay_mesh,
                        overlay_opacity=overlay_opacity,
                        coil_overlay=coil_overlay,
                        coil_line_width=coil_line_width,
                        magnet_surface_color=cast(str, sidebar_state["magnet_surface_color"]),
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
        variation_source = _gui_radio(
            "Run source",
            ["initial", "optimized", "custom"],
            key="variation_run_source",
            fallback="optimized",
            horizontal=True,
        )
        variation_custom_path = ""
        if variation_source == "custom":
            variation_custom_path = st.text_input(
                "Custom run path",
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
                        step=0.1,
                        key="variation_sigma_rel_pct",
                    )
                )
                sigma_phi_deg = float(
                    st.number_input(
                        "sigma_phi_deg (deg)",
                        min_value=0.0,
                        max_value=180.0,
                        step=0.1,
                        key="variation_sigma_phi_deg",
                    )
                )
                seed = int(
                    st.number_input(
                        "seed",
                        min_value=0,
                        max_value=2_147_483_647,
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
                sc_kernel_var = _gui_selectbox(
                    "near kernel",
                    ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                    key="variation_sc_near_kernel",
                    fallback="dipole",
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
                    gl_choice_var = _gui_selectbox(
                        "gl order",
                        ["mixed (2/3)", "2", "3"],
                        key="variation_sc_gl_order",
                        fallback="mixed (2/3)",
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
                    float(st.session_state.get("map_contour_level_ppm", 1000.0)),
                    "Variation XY(z=0) ppm map",
                )
                st.plotly_chart(fig_var_map, use_container_width=True, key="variation_map_plot")

                save_cols = st.columns(2)
                with save_cols[0]:
                    save_tag_var = st.text_input(
                        "Save tag",
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
        opt_mode = _gui_selectbox(
            "Optimization mode",
            ["L-BFGS-B", "DC/CCP (self-consistent linear)"],
            key="opt_mode",
            fallback="L-BFGS-B",
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

            input_choice = _gui_radio(
                "Input run",
                ["initial", "optimized", "custom"],
                key="opt_input_choice",
                fallback="initial",
            )
            custom_input = ""
            if input_choice == "custom":
                custom_input = st.text_input("Custom input path", key="opt_custom_input")
            if input_choice == "initial":
                opt_in_path = init_path
            elif input_choice == "optimized":
                opt_in_path = opt_path
            else:
                opt_in_path = custom_input.strip()

            st.caption(f"Input run: {opt_in_path or '(not set)'}")

            maxiter = int(
                st.number_input("maxiter", min_value=50, max_value=2000, step=50, key="opt_maxiter")
            )
            gtol = float(
                st.number_input(
                    "gtol", min_value=1e-16, max_value=1e-6, format="%.1e", key="opt_gtol"
                )
            )
            roi_r_opt = float(
                st.number_input(
                    "ROI radius (m)",
                    min_value=0.001,
                    max_value=0.2,
                    step=0.001,
                    format="%.4f",
                    key="opt_roi_radius_m",
                )
            )
            roi_step_opt = float(
                st.number_input(
                    "ROI step (m)",
                    min_value=0.001,
                    max_value=0.02,
                    step=0.001,
                    format="%.4f",
                    key="opt_roi_step_m",
                )
            )
            roi_samples_opt = int(
                st.number_input(
                    "ROI samples",
                    min_value=1,
                    max_value=200000,
                    step=1,
                    help="Used by the default surface-fibonacci ROI sampling mode.",
                    key="opt_roi_samples",
                )
            )
            angle_model = _gui_selectbox(
                "Angle model",
                ["legacy-alpha", "delta-rep-x0", "fourier-x0"],
                key="angle_model",
                fallback="legacy-alpha",
            )
            angle_init = _gui_selectbox(
                "Angle init",
                ["from-run", "zeros"],
                key="angle_init",
                fallback="from-run",
            )
            jax_available = _jax_available()
            if angle_model == "legacy-alpha":
                grad_backend = _gui_selectbox(
                    "Grad backend",
                    ["analytic", "jax"],
                    key="grad_backend",
                    fallback="analytic",
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
                        step=0.1,
                        key="lambda0",
                    )
                )
                lambda_theta = float(
                    st.number_input(
                        "lambda_theta",
                        min_value=0.0,
                        max_value=1e3,
                        step=0.1,
                        key="lambda_theta",
                    )
                )
                lambda_z = float(
                    st.number_input(
                        "lambda_z",
                        min_value=0.0,
                        max_value=1e3,
                        step=0.1,
                        key="lambda_z",
                    )
                )
            beta_cols = st.columns([1, 1])
            with beta_cols[0]:
                enable_beta_tilt_x = st.checkbox(
                    "Enable beta_tilt_x",
                    key="enable_beta_tilt_x",
                    help="Ring-wise local X-axis tilt angle (legacy-alpha + jax only).",
                )
            with beta_cols[1]:
                beta_tilt_x_bound_deg = float(
                    st.number_input(
                        "beta_tilt_x bound (deg)",
                        min_value=0.1,
                        max_value=89.0,
                        step=0.5,
                        key="beta_tilt_x_bound_deg",
                    )
                )
            if enable_beta_tilt_x:
                st.caption("beta_tilt_x requires angle_model=legacy-alpha and grad_backend=jax.")
            st.markdown("**Magnetization model**")
            mag_model = _gui_selectbox(
                "Magnetization model",
                ["fixed", "self-consistent-easy-axis"],
                key="mag_model",
                fallback="fixed",
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
                        st.number_input(
                            "chi",
                            min_value=0.0,
                            max_value=10.0,
                            step=0.01,
                            key="opt_sc_chi",
                        )
                    )
                    sc_Nd = float(
                        st.number_input(
                            "Nd",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            key="opt_sc_Nd",
                        )
                    )
                    sc_p0 = float(
                        st.number_input(
                            "p0",
                            min_value=0.0,
                            max_value=10.0,
                            step=0.01,
                            key="opt_sc_p0",
                        )
                    )
                    sc_volume_mm3 = float(
                        st.number_input(
                            "volume_mm3",
                            min_value=1.0,
                            max_value=1e6,
                            step=10.0,
                            key="opt_sc_volume_mm3",
                        )
                    )
                    sc_iters = int(
                        st.number_input(
                            "iters",
                            min_value=1,
                            max_value=500,
                            step=1,
                            key="opt_sc_iters",
                        )
                    )
                    sc_omega = float(
                        st.number_input(
                            "omega",
                            min_value=0.01,
                            max_value=1.0,
                            step=0.01,
                            key="opt_sc_omega",
                        )
                    )
                    st.markdown("**Near window**")
                    sc_near_wr = int(
                        st.number_input(
                            "wr",
                            min_value=0,
                            max_value=10,
                            step=1,
                            key="opt_sc_near_wr",
                        )
                    )
                    sc_near_wz = int(
                        st.number_input(
                            "wz",
                            min_value=0,
                            max_value=10,
                            step=1,
                            key="opt_sc_near_wz",
                        )
                    )
                    sc_near_wphi = int(
                        st.number_input(
                            "wphi",
                            min_value=0,
                            max_value=10,
                            step=1,
                            key="opt_sc_near_wphi",
                        )
                    )
                    sc_near_kernel = _gui_selectbox(
                        "near kernel",
                        ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                        key="sc_near_kernel",
                        fallback="dipole",
                    )
                    if sc_near_kernel == "multi-dipole":
                        sc_subdip_n = int(
                            st.number_input(
                                "subdip_n",
                                min_value=2,
                                max_value=10,
                                step=1,
                                key="opt_sc_subdip_n",
                            )
                        )
                    if sc_near_kernel == "gl-double-mixed":
                        gl_choice = _gui_selectbox(
                            "gl order",
                            ["mixed (2/3)", "2", "3"],
                            key="sc_gl_order",
                            fallback="mixed (2/3)",
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
            fix_radius_layer_mode = cast(
                Literal["center", "ends"],
                _gui_selectbox(
                    "Fixed radius layer mode",
                    ["center", "ends"],
                    key="fix_radius_layer_mode",
                    fallback="center",
                ),
            )
            fix_center_radius_layers = int(
                _gui_selectbox(
                    "Fixed radius layers",
                    [0, 2, 4],
                    key="fix_center_radius_layers",
                    fallback=2,
                )
            )

            st.markdown("**Radius bounds**")
            r_bounds_enabled = st.checkbox("Enable radius bounds", key="r_bounds_enabled")
            if r_bounds_enabled:
                r_bound_mode = _gui_radio(
                    "Radius bounds mode",
                    ["relative", "absolute"],
                    key="r_bound_mode",
                    fallback="relative",
                    horizontal=True,
                )
                if r_bound_mode == "relative":
                    r_lower_delta_mm = float(
                        st.number_input(
                            "Lower delta (mm)",
                            min_value=0.0,
                            max_value=200.0,
                            step=1.0,
                            key="r_lower_delta_mm",
                        )
                    )
                    r_upper_delta_mm = float(
                        st.number_input(
                            "Upper delta (mm)",
                            min_value=0.0,
                            max_value=200.0,
                            step=1.0,
                            key="r_upper_delta_mm",
                        )
                    )
                    r_no_upper = st.checkbox("No upper bound", key="r_no_upper")
                    r_min_mm = 0.0
                    r_max_mm = 1e9
                else:
                    r_min_mm = float(
                        st.number_input(
                            "Min radius (mm)",
                            min_value=0.0,
                            max_value=1e9,
                            step=1.0,
                            key="r_min_mm",
                        )
                    )
                    r_max_mm = float(
                        st.number_input(
                            "Max radius (mm)",
                            min_value=0.0,
                            max_value=1e9,
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

            tag = st.text_input("Output tag", key="opt_tag")
            preview_dir = _default_out_dir(tag)
            st.caption(f"Output dir: {preview_dir}")

            with st.expander("Generate initial run", expanded=False):
                gen_cols = st.columns(3)
                with gen_cols[0]:
                    gen_N = int(
                        st.number_input("N", min_value=1, max_value=512, step=1, key="gen_N")
                    )
                    gen_Lz = float(
                        st.number_input(
                            "Lz (m)",
                            min_value=0.01,
                            max_value=2.0,
                            step=0.01,
                            key="gen_Lz",
                        )
                    )
                with gen_cols[1]:
                    gen_R = int(
                        st.number_input("R", min_value=1, max_value=32, step=1, key="gen_R")
                    )
                    gen_diameter_mm = float(
                        st.number_input(
                            "Diameter (mm)",
                            min_value=10.0,
                            max_value=2000.0,
                            step=10.0,
                            key="gen_diameter_mm",
                        )
                    )
                with gen_cols[2]:
                    gen_K = int(
                        st.number_input("K", min_value=1, max_value=256, step=1, key="gen_K")
                    )
                    gen_ring_offset_step_mm = float(
                        st.number_input(
                            "Ring offset step (mm)",
                            min_value=0.0,
                            max_value=100.0,
                            step=1.0,
                            key="gen_ring_offset_step_mm",
                        )
                    )
                gen_profile_cols = st.columns(2)
                with gen_profile_cols[0]:
                    gen_end_R = int(
                        st.number_input(
                            "End-layer R",
                            min_value=1,
                            max_value=32,
                            step=1,
                            key="gen_end_R",
                        )
                    )
                with gen_profile_cols[1]:
                    gen_end_layers = int(
                        st.number_input(
                            "End layers / side",
                            min_value=0,
                            max_value=max(0, gen_K // 2),
                            step=1,
                            key="gen_end_layers",
                        )
                    )

                gen_tag = st.text_input("Generate output tag", key="gen_tag")
                gen_use_as_input = st.checkbox("Use generated run as input", key="gen_use_as_input")
                gen_start_opt = st.checkbox("Generate and start optimization", key="gen_start_opt")

                gen_out_dir = build_generate_out_dir(
                    ROOT / "runs",
                    tag=gen_tag,
                    N=gen_N,
                    R=gen_R,
                    K=gen_K,
                    end_R=gen_end_R,
                    end_layers_per_side=gen_end_layers,
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
                        end_R=gen_end_R,
                        end_layers_per_side=gen_end_layers,
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
                                        roi_samples=roi_samples_opt,
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
                                        fix_radius_layer_mode=fix_radius_layer_mode,
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
                                    st.session_state["opt_job_fix_radius_layer_mode"] = (
                                        fix_radius_layer_mode
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
                            roi_samples=roi_samples_opt,
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
                            fix_radius_layer_mode=fix_radius_layer_mode,
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
                        st.session_state["opt_job_fix_radius_layer_mode"] = fix_radius_layer_mode
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
                auto_refresh = st.checkbox("Auto-refresh log", key="opt_auto_refresh")
                refresh_secs = float(
                    st.number_input(
                        "Log refresh (s)",
                        min_value=0.5,
                        max_value=10.0,
                        step=0.5,
                        key="opt_log_refresh_secs",
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
                    roi_samples=roi_samples_opt,
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
                    fix_radius_layer_mode=fix_radius_layer_mode,
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
                fixed_mode = st.session_state.get("opt_job_fix_radius_layer_mode", "center")
                if fixed_layers is not None:
                    st.write(f"Fixed radius layers: {fixed_layers} (mode={fixed_mode})")

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
            dc_N = int(st.number_input("N", min_value=1, max_value=512, step=1, key="dc_N"))
            dc_K = int(st.number_input("K", min_value=1, max_value=256, step=1, key="dc_K"))
            dc_R = int(st.number_input("R", min_value=1, max_value=32, step=1, key="dc_R"))
            dc_radius_m = float(
                st.number_input(
                    "Radius (m)",
                    min_value=0.01,
                    max_value=5.0,
                    step=0.01,
                    key="dc_radius_m",
                )
            )
            dc_length_m = float(
                st.number_input(
                    "Length (m)",
                    min_value=0.01,
                    max_value=5.0,
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
                    step=0.001,
                    key="dc_roi_radius_m",
                )
            )
            dc_roi_grid_n = int(
                st.number_input(
                    "ROI grid N", min_value=3, max_value=201, step=2, key="dc_roi_grid_n"
                )
            )

            st.markdown("**Objective weights**")
            dc_wx = float(
                st.number_input("wx", min_value=0.0, max_value=1e3, step=0.1, key="dc_wx")
            )
            dc_wy = float(
                st.number_input("wy", min_value=0.0, max_value=1e3, step=0.1, key="dc_wy")
            )
            dc_wz = float(
                st.number_input("wz", min_value=0.0, max_value=1e3, step=0.1, key="dc_wz")
            )

            st.markdown("**CCP settings**")
            dc_phi0 = float(
                st.number_input("phi0", min_value=-6.3, max_value=6.3, step=0.1, key="dc_phi0")
            )
            dc_delta_nom = float(
                st.number_input(
                    "delta_nom_deg",
                    min_value=0.1,
                    max_value=60.0,
                    step=0.5,
                    key="dc_delta_nom",
                )
            )
            dc_step_enable = st.checkbox("Enable step trust", key="dc_step_enable")
            dc_delta_step = float(
                st.number_input(
                    "delta_step_deg",
                    min_value=0.1,
                    max_value=60.0,
                    step=0.5,
                    key="dc_delta_step",
                )
            )
            if not dc_step_enable:
                dc_delta_step_val = None
            else:
                dc_delta_step_val = dc_delta_step

            dc_tau0 = float(
                st.number_input("tau0", min_value=0.0, max_value=1.0, format="%.1e", key="dc_tau0")
            )
            dc_tau_mult = float(
                st.number_input(
                    "tau_mult", min_value=1.0, max_value=5.0, step=0.1, key="dc_tau_mult"
                )
            )
            dc_tau_max = float(
                st.number_input(
                    "tau_max",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1e",
                    key="dc_tau_max",
                )
            )
            dc_iters = int(
                st.number_input("iters", min_value=1, max_value=200, step=1, key="dc_iters")
            )
            dc_tol = float(
                st.number_input("tol", min_value=1e-10, max_value=1e-2, format="%.1e", key="dc_tol")
            )
            dc_tol_f = float(
                st.number_input(
                    "tol_f",
                    min_value=1e-12,
                    max_value=1e-3,
                    format="%.1e",
                    key="dc_tol_f",
                )
            )
            st.markdown("**Initial guess (from L-BFGS run)**")
            dc_init_select = _gui_selectbox(
                "Init run (optional)",
                ["(none)"] + candidates,
                key="dc_init_select",
                fallback="(none)",
            )
            dc_init_manual = st.text_input("Init run path (optional)", key="dc_init_manual")
            dc_init_select_val = "" if dc_init_select == "(none)" else dc_init_select
            dc_init_run = _resolve_path(dc_init_select_val, dc_init_manual)
            st.caption("Only delta-rep-x0 / fourier-x0 angle models are accepted.")

            st.markdown("**Regularization**")
            dc_reg_x = float(
                st.number_input("reg_x", min_value=0.0, max_value=1e3, step=0.1, key="dc_reg_x")
            )
            dc_reg_p = float(
                st.number_input("reg_p", min_value=0.0, max_value=1e3, step=0.1, key="dc_reg_p")
            )
            dc_reg_z = float(
                st.number_input("reg_z", min_value=0.0, max_value=1e3, step=0.1, key="dc_reg_z")
            )

            st.markdown("**Self-consistent equality**")
            dc_sc_eq = st.checkbox("Enable self-consistent equalities", key="dc_sc_eq")
            dc_p_fix = None
            if not dc_sc_eq:
                dc_p_fix = float(
                    st.number_input(
                        "p_fix", min_value=0.0, max_value=10.0, step=0.01, key="dc_p_fix"
                    )
                )

            st.markdown("**Self-consistent parameters**")
            dc_sc_chi = float(
                st.number_input("sc_chi", min_value=0.0, max_value=10.0, step=0.01, key="dc_sc_chi")
            )
            dc_sc_Nd = float(
                st.number_input(
                    "sc_Nd",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="dc_sc_Nd",
                )
            )
            dc_sc_p0 = float(
                st.number_input("sc_p0", min_value=0.0, max_value=10.0, step=0.01, key="dc_sc_p0")
            )
            dc_sc_volume_mm3 = float(
                st.number_input(
                    "sc_volume_mm3",
                    min_value=1.0,
                    max_value=1e6,
                    step=10.0,
                    key="dc_sc_volume_mm3",
                )
            )

            st.markdown("**Near window**")
            dc_sc_near_wr = int(
                st.number_input("wr", min_value=0, max_value=10, step=1, key="dc_sc_near_wr")
            )
            dc_sc_near_wz = int(
                st.number_input("wz", min_value=0, max_value=10, step=1, key="dc_sc_near_wz")
            )
            dc_sc_near_wphi = int(
                st.number_input("wphi", min_value=0, max_value=10, step=1, key="dc_sc_near_wphi")
            )
            dc_sc_near_kernel = _gui_selectbox(
                "near kernel",
                ["dipole", "multi-dipole", "cellavg", "gl-double-mixed"],
                key="dc_sc_near_kernel",
                fallback="dipole",
            )
            dc_sc_subdip_n = 2
            dc_sc_gl_order: int | None = None
            if dc_sc_near_kernel == "multi-dipole":
                dc_sc_subdip_n = int(
                    st.number_input(
                        "subdip_n", min_value=2, max_value=10, step=1, key="dc_sc_subdip_n"
                    )
                )
            if dc_sc_near_kernel == "gl-double-mixed":
                gl_choice = _gui_selectbox(
                    "gl order",
                    ["mixed (2/3)", "2", "3"],
                    key="dc_sc_gl_order",
                    fallback="mixed (2/3)",
                )
                if gl_choice in ("2", "3"):
                    dc_sc_gl_order = int(gl_choice)

            st.markdown("**p bounds**")
            dc_pmin = float(
                st.number_input("pmin", min_value=0.0, max_value=10.0, step=0.01, key="dc_pmin")
            )
            dc_pmax = float(
                st.number_input("pmax", min_value=0.0, max_value=10.0, step=0.01, key="dc_pmax")
            )

            st.markdown("**Factor**")
            dc_factor = float(
                st.number_input(
                    "factor",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1e",
                    key="dc_factor",
                )
            )

            solver_options = [s for s in ["ECOS", "SCS"] if s in installed_solvers] or ["SCS"]
            dc_solver = _gui_selectbox(
                "Solver",
                solver_options,
                key="dc_solver",
                fallback="SCS",
            )
            dc_verbose = st.checkbox("Verbose log", key="dc_verbose")

            dc_tag = st.text_input("Output tag", key="dc_tag")
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
                dc_auto_refresh = st.checkbox("Auto-refresh log", key="dc_auto_refresh")
                dc_refresh_secs = float(
                    st.number_input(
                        "Log refresh (s)",
                        min_value=0.5,
                        max_value=10.0,
                        step=0.5,
                        key="dc_refresh_secs",
                    )
                )
                if dc_auto_refresh:
                    time.sleep(max(0.1, dc_refresh_secs))
                    st.rerun()

    with st.sidebar:
        st.header("GUI Config")
        st.caption(f"Defaults file: `{GUI_DEFAULTS_PATH}`")
        for warning in _consume_gui_config_warnings():
            st.warning(warning)

        if "gui_config_import_keys" not in st.session_state:
            st.session_state["gui_config_import_keys"] = list(GUI_CONFIG_KEYS)

        saved_config_paths = ["(none)", *list_gui_config_export_paths()]
        _gui_selectbox(
            "Saved config",
            saved_config_paths,
            key="gui_config_import_select",
            fallback="(none)",
        )
        st.text_input(
            "Import path (optional)",
            value="",
            key="gui_config_import_path",
            help="Set this to override the saved-config selection.",
        )
        import_path = _resolve_gui_config_import_path(
            str(st.session_state["gui_config_import_select"]),
            str(st.session_state["gui_config_import_path"]),
        )
        if import_path is not None:
            st.caption(f"Import path: `{import_path}`")
            try:
                import_values = load_gui_config_values(import_path)
            except Exception as exc:
                st.error(f"GUI config load failed: {exc}")
            else:
                select_cols = st.columns(2)
                with select_cols[0]:
                    if st.button("Select all", key="gui_config_import_select_all"):
                        st.session_state["gui_config_import_keys"] = list(GUI_CONFIG_KEYS)
                with select_cols[1]:
                    if st.button("Clear selection", key="gui_config_import_clear"):
                        st.session_state["gui_config_import_keys"] = []

                selected_import_keys = cast(
                    list[str],
                    st.multiselect(
                        "Settings to apply",
                        list(GUI_CONFIG_KEYS),
                        key="gui_config_import_keys",
                    ),
                )
                st.caption(
                    f"Selected {len(selected_import_keys)} / {len(GUI_CONFIG_KEYS)} settings."
                )
                if selected_import_keys:
                    with st.expander("Preview selected values", expanded=False):
                        st.json({key: import_values[key] for key in selected_import_keys})

                if st.button(
                    "Apply imported settings",
                    key="gui_config_import_apply_button",
                    disabled=not selected_import_keys,
                ):
                    try:
                        st.session_state["pending_gui_config_values"] = dict(import_values)
                        st.session_state["pending_gui_config_keys"] = list(selected_import_keys)
                        st.session_state["flash_message"] = (
                            f"Applied {len(selected_import_keys)} GUI setting(s) from: {import_path}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"GUI config apply failed: {exc}")

        export_tag = st.text_input("Export tag", value="gui", key="gui_config_export_tag")
        export_path = default_gui_config_export_path(export_tag)
        st.caption(f"Export path: `{export_path}`")
        if st.button("Export current settings", key="gui_config_export_button"):
            try:
                export_values = merge_gui_config_values(gui_defaults, st.session_state)
                write_gui_config(export_path, export_values)
                st.success(f"Exported GUI settings to: {export_path}")
            except Exception as exc:
                st.error(f"GUI config export failed: {exc}")


if __name__ == "__main__":
    main()
