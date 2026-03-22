from pathlib import Path

import numpy as np

from app.streamlit_app import (
    _coil_x_rotation_quarter_turns,
    _resolve_coil_overlay_path,
    _scan_coil_npz_files,
)


def test_scan_coil_npz_files_lists_relative_npz_paths(tmp_path: Path) -> None:
    coil_dir = tmp_path / "coil"
    sub_dir = coil_dir / "nested"
    sub_dir.mkdir(parents=True)
    np.savez_compressed(coil_dir / "a.npz", dummy=np.asarray([1], dtype=np.int64))
    np.savez_compressed(sub_dir / "b.npz", dummy=np.asarray([1], dtype=np.int64))

    files = _scan_coil_npz_files(coil_dir)

    assert len(files) == 2
    assert all(path.endswith(".npz") for path in files)


def test_resolve_coil_overlay_path_handles_absolute_and_relative_paths(tmp_path: Path) -> None:
    relative = "coil/example.npz"
    resolved_relative = _resolve_coil_overlay_path(relative)
    assert resolved_relative.name == "example.npz"

    absolute = tmp_path / "coil_abs.npz"
    assert _resolve_coil_overlay_path(str(absolute)) == absolute


def test_coil_x_rotation_quarter_turns_accepts_only_90_degree_steps() -> None:
    assert _coil_x_rotation_quarter_turns(0) == 0
    assert _coil_x_rotation_quarter_turns(90) == 1
    assert _coil_x_rotation_quarter_turns(180) == 2
    assert _coil_x_rotation_quarter_turns(270) == 3
