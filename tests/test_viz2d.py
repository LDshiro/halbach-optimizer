import json
from pathlib import Path

import numpy as np

from halbach.run_io import load_run
from halbach.viz2d import compute_error_map_ppm_plane, extract_cross_section_y0


def _write_min_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    R = 1
    K = 3
    N = 8
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    ring_offsets = np.array([0.0], dtype=float)
    alphas = 1e-3 * np.arange(R * K, dtype=float).reshape(R, K)
    r_bases = 0.2 + 1e-3 * np.arange(K, dtype=float)

    np.savez(
        run_dir / "results.npz",
        alphas_opt=alphas,
        r_bases_opt=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )
    (run_dir / "meta.json").write_text(json.dumps({"label": "viz2d"}), encoding="utf-8")
    return run_dir


def test_compute_error_map_xy_and_cross_section(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path)
    run = load_run(run_dir)

    m = compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=0.05, step=0.05)

    assert m.ppm.shape == (len(m.ys), len(m.xs))
    assert m.mask.shape == m.ppm.shape
    assert int(np.sum(m.mask)) > 0
    assert np.isfinite(m.B0_T)
    assert m.B0_T > 0.0
    assert np.isnan(m.ppm[~m.mask]).any()

    line = extract_cross_section_y0(m)
    assert line.x.shape == m.xs.shape
    assert line.ppm.shape == m.xs.shape
