from pathlib import Path

from app.streamlit_app import (
    DEFAULT_HUMAN_OVERLAY_OBJ,
    ROOT,
    _default_human_overlay_obj_path,
    _human_overlay_transform_key,
    _human_overlay_unit_scale_m,
    _resolve_human_overlay_obj_path,
)


def test_default_human_overlay_obj_path_points_to_repo_root() -> None:
    assert _default_human_overlay_obj_path() == ROOT / DEFAULT_HUMAN_OVERLAY_OBJ


def test_resolve_human_overlay_obj_path_resolves_relative_to_root() -> None:
    resolved = _resolve_human_overlay_obj_path(DEFAULT_HUMAN_OVERLAY_OBJ)
    assert resolved == ROOT / DEFAULT_HUMAN_OVERLAY_OBJ

    absolute = Path("C:/tmp/example.obj")
    assert _resolve_human_overlay_obj_path(str(absolute)) == absolute


def test_human_overlay_unit_scale_defaults_support_cm() -> None:
    assert _human_overlay_unit_scale_m("mm") == 1e-3
    assert _human_overlay_unit_scale_m("cm") == 1e-2
    assert _human_overlay_unit_scale_m("m") == 1.0


def test_human_overlay_transform_key_depends_on_path_and_mtime() -> None:
    key_a = _human_overlay_transform_key(
        "body.obj",
        1.0,
        "cm",
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    key_b = _human_overlay_transform_key(
        "body.obj",
        2.0,
        "cm",
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    key_c = _human_overlay_transform_key(
        "other.obj",
        1.0,
        "cm",
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    assert key_a != key_b
    assert key_a != key_c
