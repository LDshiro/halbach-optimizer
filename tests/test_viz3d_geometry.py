import json
from pathlib import Path

import numpy as np
import pytest

from halbach.run_io import load_run
from halbach.viz3d import build_magnet_figure, enumerate_magnets


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
    (run_dir / "meta.json").write_text(json.dumps({"label": "viz3d"}), encoding="utf-8")
    return run_dir


def test_enumerate_magnets_shapes_and_filtering(tmp_path: Path) -> None:
    run = load_run(_write_min_run(tmp_path))

    centers, phi, ring_id, layer_id = enumerate_magnets(run, stride=2, hide_x_negative=False)
    expected = 1 * 3 * 4
    assert centers.shape == (expected, 3)
    assert phi.shape == (expected,)
    assert ring_id.shape == (expected,)
    assert layer_id.shape == (expected,)

    centers_f, phi_f, ring_f, layer_f = enumerate_magnets(run, stride=2, hide_x_negative=True)
    assert centers_f.shape[0] < centers.shape[0]
    assert phi_f.shape[0] == centers_f.shape[0]
    assert ring_f.shape[0] == centers_f.shape[0]
    assert layer_f.shape[0] == centers_f.shape[0]


def test_build_magnet_figure_runs(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    run = load_run(_write_min_run(tmp_path))

    fig = build_magnet_figure(run, stride=2, mode="fast")
    assert fig is not None
