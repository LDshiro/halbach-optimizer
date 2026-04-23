import json
from pathlib import Path

from app.gui_config import (
    GUI_CONFIG_EXPORT_DIR,
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


def test_gui_defaults_file_has_expected_schema_and_keys() -> None:
    values = load_gui_config_values(GUI_DEFAULTS_PATH)

    assert set(values) == set(GUI_CONFIG_KEYS)
    assert values["opt_mode"] == "L-BFGS-B"
    assert values["human_overlay_obj_path"] == "10688_GenericMale_v2.obj"


def test_merge_gui_config_values_filters_transient_state_and_keeps_utf8_text() -> None:
    defaults = load_gui_config_values(GUI_DEFAULTS_PATH)

    merged = merge_gui_config_values(
        defaults,
        {
            "map_roi_radius_m": 0.123,
            "human_overlay_obj_path": "日本語/体.obj",
            "opt_job": {"status": "running"},
        },
    )

    assert merged["map_roi_radius_m"] == 0.123
    assert merged["human_overlay_obj_path"] == "日本語/体.obj"
    assert "opt_job" not in merged


def test_gui_config_round_trip_preserves_exported_values(tmp_path: Path) -> None:
    defaults = load_gui_config_values(GUI_DEFAULTS_PATH)
    merged = merge_gui_config_values(
        defaults,
        {
            "opt_tag": "custom_opt",
            "variation_seed": 777,
            "view_target": "initial",
        },
    )
    path = tmp_path / "gui_config.json"

    write_gui_config(path, merged)
    reloaded = load_gui_config_values(path)

    assert reloaded == merged
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["values"]["opt_tag"] == "custom_opt"


def test_apply_gui_config_values_can_apply_selected_subset_only() -> None:
    defaults = load_gui_config_values(GUI_DEFAULTS_PATH)
    state: dict[str, object] = {
        "opt_tag": "before",
        "variation_seed": 1,
        "view_target": "optimized",
    }

    applied = apply_gui_config_values(
        state,
        defaults,
        selected_keys=["opt_tag", "variation_seed"],
    )

    assert applied == ["opt_tag", "variation_seed"]
    assert state["opt_tag"] == defaults["opt_tag"]
    assert state["variation_seed"] == defaults["variation_seed"]
    assert state["view_target"] == "optimized"


def test_apply_gui_config_values_rejects_unknown_key() -> None:
    defaults = load_gui_config_values(GUI_DEFAULTS_PATH)

    try:
        apply_gui_config_values({}, defaults, selected_keys=["unknown_key"])
    except ValueError as exc:
        assert "unknown_key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported GUI config key")


def test_normalize_gui_choice_falls_back_without_error() -> None:
    choice, should_warn = normalize_gui_choice("missing", ["alpha", "beta"], fallback="alpha")
    assert choice == "alpha"
    assert should_warn is True

    blank_choice, blank_warn = normalize_gui_choice("", ["alpha", "beta"], fallback="alpha")
    assert blank_choice == "alpha"
    assert blank_warn is False


def test_default_gui_config_export_path_uses_export_dir_and_json_suffix() -> None:
    path = default_gui_config_export_path("日本語 tag")

    assert path.parent == GUI_CONFIG_EXPORT_DIR
    assert path.suffix == ".json"
    assert path.name.endswith("_tag.json")


def test_list_gui_config_export_paths_returns_relative_json_paths(tmp_path: Path) -> None:
    original_dir = GUI_CONFIG_EXPORT_DIR
    try:
        from app import gui_config as gui_config_module

        gui_config_module.GUI_CONFIG_EXPORT_DIR = tmp_path
        (tmp_path / "20260101_120000_gui.json").write_text("{}", encoding="utf-8")
        (tmp_path / "note.txt").write_text("", encoding="utf-8")

        paths = list_gui_config_export_paths()
    finally:
        gui_config_module.GUI_CONFIG_EXPORT_DIR = original_dir

    assert paths == [str(tmp_path / "20260101_120000_gui.json")]
