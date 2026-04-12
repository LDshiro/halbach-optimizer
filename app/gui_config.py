from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar, cast

ROOT = Path(__file__).resolve().parents[1]
GUI_CONFIG_SCHEMA_VERSION = 1
GUI_DEFAULTS_PATH = ROOT / "app" / "gui_defaults.json"
GUI_CONFIG_EXPORT_DIR = ROOT / "runs" / "gui_configs"

GUI_CONFIG_KEYS = (
    "init_select",
    "init_path",
    "opt_select",
    "opt_path",
    "map_roi_radius_m",
    "map_step_m",
    "map_auto_limit",
    "map_ppm_limit",
    "map_contour_level_ppm",
    "map_mag_model_eval",
    "map_sc_chi",
    "map_sc_Nd",
    "map_sc_p0",
    "map_sc_volume_mm3",
    "map_sc_iters",
    "map_sc_omega",
    "map_sc_near_wr",
    "map_sc_near_wz",
    "map_sc_near_wphi",
    "map_sc_near_kernel",
    "map_sc_subdip_n",
    "map_sc_gl_order",
    "view_target",
    "view_mode",
    "view_stride",
    "view_hide_x_negative",
    "view_magnet_size_mm",
    "view_magnet_thickness_mm",
    "view_arrow_length_mm",
    "view_arrow_head_angle_deg",
    "view_magnet_surface_label",
    "compare_camera_preset",
    "human_overlay_run_target",
    "human_overlay_obj_path",
    "human_overlay_unit",
    "human_overlay_opacity",
    "human_overlay_scale",
    "human_overlay_tx_m",
    "human_overlay_rx_deg",
    "human_overlay_ty_m",
    "human_overlay_ry_deg",
    "human_overlay_tz_m",
    "human_overlay_rz_deg",
    "human_overlay_camera_preset",
    "human_overlay_show_coil",
    "human_overlay_coil_file",
    "human_overlay_coil_line_width",
    "human_overlay_coil_rotation_x_deg",
    "variation_run_source",
    "variation_custom_path",
    "variation_sigma_rel_pct",
    "variation_sigma_phi_deg",
    "variation_seed",
    "variation_roi_radius_m",
    "variation_roi_samples",
    "variation_map_radius_m",
    "variation_map_step_m",
    "variation_sc_chi",
    "variation_sc_Nd",
    "variation_sc_p0",
    "variation_sc_volume_mm3",
    "variation_sc_iters",
    "variation_sc_omega",
    "variation_sc_near_wr",
    "variation_sc_near_wz",
    "variation_sc_near_wphi",
    "variation_sc_near_kernel",
    "variation_sc_subdip_n",
    "variation_sc_gl_order",
    "variation_save_tag",
    "opt_mode",
    "opt_input_choice",
    "opt_custom_input",
    "opt_maxiter",
    "opt_gtol",
    "opt_roi_radius_m",
    "opt_roi_step_m",
    "opt_roi_samples",
    "angle_model",
    "angle_init",
    "grad_backend",
    "fourier_H",
    "lambda0",
    "lambda_theta",
    "lambda_z",
    "enable_beta_tilt_x",
    "beta_tilt_x_bound_deg",
    "mag_model",
    "opt_sc_chi",
    "opt_sc_Nd",
    "opt_sc_p0",
    "opt_sc_volume_mm3",
    "opt_sc_iters",
    "opt_sc_omega",
    "opt_sc_near_wr",
    "opt_sc_near_wz",
    "opt_sc_near_wphi",
    "sc_near_kernel",
    "opt_sc_subdip_n",
    "sc_gl_order",
    "fix_radius_layer_mode",
    "fix_center_radius_layers",
    "r_bounds_enabled",
    "r_bound_mode",
    "r_lower_delta_mm",
    "r_upper_delta_mm",
    "r_no_upper",
    "r_min_mm",
    "r_max_mm",
    "opt_tag",
    "gen_N",
    "gen_Lz",
    "gen_R",
    "gen_diameter_mm",
    "gen_K",
    "gen_ring_offset_step_mm",
    "gen_end_R",
    "gen_end_layers",
    "gen_tag",
    "gen_use_as_input",
    "gen_start_opt",
    "opt_auto_refresh",
    "opt_log_refresh_secs",
    "dc_N",
    "dc_K",
    "dc_R",
    "dc_radius_m",
    "dc_length_m",
    "dc_roi_radius_m",
    "dc_roi_grid_n",
    "dc_wx",
    "dc_wy",
    "dc_wz",
    "dc_phi0",
    "dc_delta_nom",
    "dc_step_enable",
    "dc_delta_step",
    "dc_tau0",
    "dc_tau_mult",
    "dc_tau_max",
    "dc_iters",
    "dc_tol",
    "dc_tol_f",
    "dc_init_select",
    "dc_init_manual",
    "dc_reg_x",
    "dc_reg_p",
    "dc_reg_z",
    "dc_sc_eq",
    "dc_p_fix",
    "dc_sc_chi",
    "dc_sc_Nd",
    "dc_sc_p0",
    "dc_sc_volume_mm3",
    "dc_sc_near_wr",
    "dc_sc_near_wz",
    "dc_sc_near_wphi",
    "dc_sc_near_kernel",
    "dc_sc_subdip_n",
    "dc_sc_gl_order",
    "dc_pmin",
    "dc_pmax",
    "dc_factor",
    "dc_solver",
    "dc_verbose",
    "dc_tag",
    "dc_auto_refresh",
    "dc_refresh_secs",
)

ChoiceT = TypeVar("ChoiceT")


def load_gui_config_values(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GUI config at {path} must be a JSON object")
    if payload.get("schema_version") != GUI_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"GUI config at {path} must have schema_version={GUI_CONFIG_SCHEMA_VERSION}"
        )
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ValueError(f"GUI config at {path} must contain object field 'values'")

    missing = [key for key in GUI_CONFIG_KEYS if key not in values]
    extras = sorted(str(key) for key in values if key not in GUI_CONFIG_KEYS)
    if missing or extras:
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if extras:
            parts.append(f"unexpected keys: {', '.join(extras)}")
        raise ValueError(f"GUI config key mismatch in {path}: {'; '.join(parts)}")

    return {key: values[key] for key in GUI_CONFIG_KEYS}


def merge_gui_config_values(
    default_values: Mapping[str, object], state: Mapping[str, object]
) -> dict[str, object]:
    merged = {key: default_values[key] for key in GUI_CONFIG_KEYS}
    for key in GUI_CONFIG_KEYS:
        if key in state:
            merged[key] = _json_ready_value(state[key])
    return merged


def write_gui_config(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GUI_CONFIG_SCHEMA_VERSION,
        "values": {key: _json_ready_value(values[key]) for key in GUI_CONFIG_KEYS},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def normalize_gui_choice(
    current: object,
    options: Sequence[ChoiceT],
    *,
    fallback: ChoiceT,
    warn_on_blank: bool = False,
) -> tuple[ChoiceT, bool]:
    if not options:
        raise ValueError("Choice options must not be empty")
    if current in options:
        return cast(ChoiceT, current), False

    resolved = fallback if fallback in options else options[0]
    should_warn = warn_on_blank or current not in (None, "")
    return resolved, should_warn


def default_gui_config_export_path(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return GUI_CONFIG_EXPORT_DIR / f"{stamp}_{sanitize_gui_config_tag(tag)}.json"


def sanitize_gui_config_tag(tag: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", tag.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "gui"


def _json_ready_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            return value
    return value


__all__ = [
    "GUI_CONFIG_EXPORT_DIR",
    "GUI_CONFIG_KEYS",
    "GUI_CONFIG_SCHEMA_VERSION",
    "GUI_DEFAULTS_PATH",
    "ROOT",
    "default_gui_config_export_path",
    "load_gui_config_values",
    "merge_gui_config_values",
    "normalize_gui_choice",
    "sanitize_gui_config_tag",
    "write_gui_config",
]
