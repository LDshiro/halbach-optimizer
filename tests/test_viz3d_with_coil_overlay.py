import json
from pathlib import Path

import numpy as np
import pytest

from halbach.coil_overlay import CoilPolylineSet
from halbach.run_io import load_run
from halbach.viz3d import build_magnet_figure


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
    alphas = np.zeros((R, K), dtype=float)
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
    (run_dir / "meta.json").write_text(json.dumps({"label": "coil-overlay"}), encoding="utf-8")
    return run_dir


def test_build_magnet_figure_accepts_coil_overlay(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    run = load_run(_write_min_run(tmp_path))
    coil_overlay = CoilPolylineSet(
        points_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        ),
        polyline_start=np.asarray([0, 2], dtype=np.int64),
        polyline_count=np.asarray([2, 1], dtype=np.int64),
        polyline_color_rgba_u8=np.asarray(
            [
                [255, 0, 0, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        ),
        bbox_min=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        bbox_max=np.asarray([0.1, 0.1, 0.0], dtype=np.float64),
    )

    fig_plain = build_magnet_figure(run, stride=2, mode="fast")
    fig_coil = build_magnet_figure(
        run,
        stride=2,
        mode="fast",
        coil_overlay=coil_overlay,
        coil_line_width=5.0,
    )

    assert fig_coil is not None
    assert len(fig_coil.data) == len(fig_plain.data) + 2
