import json
from pathlib import Path

import numpy as np

from halbach.run_io import load_run
from halbach.viz2d import compute_error_map_ppm_plane


def _write_beta_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    R = 1
    K = 4
    N = 8
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    ring_offsets = np.array([0.0], dtype=float)
    alphas = np.zeros((R, K), dtype=float)
    beta = np.zeros((R, K), dtype=float)
    beta[:, 1:3] = np.deg2rad(10.0)
    r_bases = 0.2 + 1e-3 * np.arange(K, dtype=float)

    np.savez(
        run_dir / "results.npz",
        alphas_opt=alphas,
        beta_tilt_x_opt=beta,
        r_bases_opt=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"angle_model": "legacy-alpha"}), encoding="utf-8"
    )
    return run_dir


def test_viz2d_beta_reflection_smoke(tmp_path: Path) -> None:
    run = load_run(_write_beta_run(tmp_path))
    m = compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=0.05, step=0.05)
    vals = np.asarray(m.ppm[m.mask], dtype=np.float64)
    assert vals.size > 0
    assert np.isfinite(vals).all()
