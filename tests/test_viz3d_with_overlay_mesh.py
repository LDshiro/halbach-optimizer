import json
from pathlib import Path

import numpy as np
import pytest

from halbach.obj_mesh import ObjMesh
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
    (run_dir / "meta.json").write_text(json.dumps({"label": "overlay"}), encoding="utf-8")
    return run_dir


def test_build_magnet_figure_accepts_overlay_mesh(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    run = load_run(_write_min_run(tmp_path))
    overlay_mesh = ObjMesh(
        vertices=np.array(
            [
                [-0.1, -0.1, 0.0],
                [0.1, -0.1, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        bbox_min=np.array([-0.1, -0.1, 0.0], dtype=np.float64),
        bbox_max=np.array([0.1, 0.1, 0.0], dtype=np.float64),
    )

    fig_plain = build_magnet_figure(run, stride=2, mode="fast")
    fig_overlay = build_magnet_figure(
        run,
        stride=2,
        mode="fast",
        overlay_mesh=overlay_mesh,
        overlay_opacity=0.4,
    )

    assert fig_overlay is not None
    assert len(fig_overlay.data) == len(fig_plain.data) + 1
    assert str(fig_overlay.data[0].name) == "Human model"


def test_build_magnet_figure_uses_custom_magnet_surface_color(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")
    run = load_run(_write_min_run(tmp_path))

    fig = build_magnet_figure(
        run,
        stride=2,
        mode="cubes",
        magnet_surface_color="rgb(255,246,0)",
    )

    mesh_traces = [trace for trace in fig.data if trace.type == "mesh3d"]
    assert mesh_traces
    assert str(mesh_traces[0].color) == "rgb(255,246,0)"
