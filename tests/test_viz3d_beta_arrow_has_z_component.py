import json
from pathlib import Path

import numpy as np
import pytest

from halbach.run_io import load_run
from halbach.viz3d import build_magnet_figure


def _write_beta_run(tmp_path: Path) -> Path:
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
    alphas = np.zeros((R, K), dtype=float)
    beta = np.zeros((R, K), dtype=float)
    beta[:, :] = np.deg2rad(12.0)
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


def test_viz3d_beta_arrow_has_z_component(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    run = load_run(_write_beta_run(tmp_path))
    fig = build_magnet_figure(
        run,
        stride=2,
        mode="cubes_arrows",
        magnet_size_m=0.02,
        arrow_length_m=0.03,
    )
    arrow_trace = None
    for trace in fig.data:
        if str(getattr(trace, "name", "")).endswith("arrows"):
            arrow_trace = trace
            break
    assert arrow_trace is not None
    z = np.asarray(arrow_trace.z, dtype=np.float64)
    z0 = z[0::3]
    z1 = z[1::3]
    valid = np.isfinite(z0) & np.isfinite(z1)
    assert valid.any()
    assert np.any(np.abs(z1[valid] - z0[valid]) > 1e-9)
