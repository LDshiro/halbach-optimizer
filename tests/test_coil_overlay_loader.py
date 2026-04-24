from pathlib import Path

import numpy as np
import pytest

from halbach.coil_overlay import load_coil_polyline_npz


def _write_coil_npz(
    path: Path,
    *,
    format_name: str = "gradientcoil_3d_coil",
    coordinate_unit: str = "m",
    counts: np.ndarray | None = None,
    include_color: bool = True,
) -> Path:
    points_xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    polyline_count = (
        np.asarray([3], dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
    )
    payload: dict[str, object] = {
        "format_name": np.array(format_name),
        "format_version": np.array("1.0"),
        "points_xyz": points_xyz,
        "polyline_start": np.asarray([0], dtype=np.int64),
        "polyline_count": polyline_count,
        "polyline_surface": np.asarray([0], dtype=np.int64),
        "polyline_sign": np.asarray([1.0], dtype=np.float64),
        "polyline_closed": np.asarray([False], dtype=np.bool_),
        "polyline_periodic_closed": np.asarray([False], dtype=np.bool_),
        "source_contour_npz": np.array("src.npz"),
        "source_config_json": np.array("{}"),
        "input_kind": np.array("test"),
        "coordinate_unit": np.array(coordinate_unit),
        "color_space": np.array("srgba_u8"),
    }
    if include_color:
        payload["polyline_color_rgba_u8"] = np.asarray([[255, 0, 0, 255]], dtype=np.uint8)
    np.savez_compressed(path, **payload)
    return path


def test_load_coil_polyline_npz_ok(tmp_path: Path) -> None:
    path = _write_coil_npz(tmp_path / "coil.npz")

    coil = load_coil_polyline_npz(path)

    assert coil.points_xyz.shape == (3, 3)
    assert coil.polyline_start.shape == (1,)
    assert coil.polyline_count.shape == (1,)
    assert coil.polyline_color_rgba_u8.shape == (1, 4)
    assert np.allclose(coil.bbox_min, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(coil.bbox_max, np.array([1.0, 1.0, 0.0]))


def test_load_coil_polyline_npz_rejects_missing_required_key(tmp_path: Path) -> None:
    path = _write_coil_npz(tmp_path / "missing_color.npz", include_color=False)

    with pytest.raises(ValueError, match="missing required keys"):
        load_coil_polyline_npz(path)


def test_load_coil_polyline_npz_rejects_bad_format_name(tmp_path: Path) -> None:
    path = _write_coil_npz(tmp_path / "bad_format.npz", format_name="wrong")

    with pytest.raises(ValueError, match="format_name"):
        load_coil_polyline_npz(path)


def test_load_coil_polyline_npz_rejects_non_meter_units(tmp_path: Path) -> None:
    path = _write_coil_npz(tmp_path / "bad_units.npz", coordinate_unit="cm")

    with pytest.raises(ValueError, match="coordinate_unit"):
        load_coil_polyline_npz(path)


def test_load_coil_polyline_npz_rejects_bad_count_sum(tmp_path: Path) -> None:
    path = _write_coil_npz(tmp_path / "bad_count.npz", counts=np.asarray([2], dtype=np.int64))

    with pytest.raises(ValueError, match="sum\\(polyline_count\\)"):
        load_coil_polyline_npz(path)
