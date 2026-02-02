import json
from pathlib import Path

import numpy as np

from halbach.run_io import load_run
from halbach.viz2d import compute_error_map_ppm_plane


def _write_min_dc_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_dc"
    run_dir.mkdir()

    meta = {
        "framework": "dc",
        "geom": {"R": 1, "K": 1, "N": 1, "radius_m": 0.05, "length_m": 0.0},
        "factor": 1e-7,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    r0_flat = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)
    m_flat = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    pts = np.array(
        [
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [-0.01, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    center_idx = np.array([0], dtype=np.int64)

    np.savez(
        run_dir / "results.npz", r0_flat=r0_flat, m_flat=m_flat, pts=pts, center_idx=center_idx
    )
    return run_dir


def test_compute_error_map_dc_xy(tmp_path: Path) -> None:
    run = load_run(_write_min_dc_run(tmp_path))

    m = compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=0.05, step=0.05)

    assert m.ppm.shape == (len(m.ys), len(m.xs))
    assert m.mask.shape == m.ppm.shape
    assert int(np.sum(m.mask)) > 0
    assert np.isfinite(m.B0_T)
    assert m.B0_T > 0.0
    assert np.isfinite(m.ppm[m.mask]).all()
