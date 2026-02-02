import json
from pathlib import Path

import numpy as np

from halbach.run_io import load_run
from halbach.viz3d import enumerate_magnets


def _write_min_dc_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_dc"
    run_dir.mkdir()

    meta = {
        "framework": "dc",
        "geom": {"R": 1, "K": 1, "N": 4, "radius_m": 0.05, "length_m": 0.0},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    r0_flat = np.column_stack([0.05 * np.cos(theta), 0.05 * np.sin(theta), np.zeros_like(theta)])
    phi_opt = (2.0 * theta).astype(np.float64)

    np.savez(run_dir / "results.npz", r0_flat=r0_flat, phi_opt=phi_opt)
    return run_dir


def test_enumerate_magnets_dc_stride(tmp_path: Path) -> None:
    run = load_run(_write_min_dc_run(tmp_path))

    centers, phi, ring_id, layer_id = enumerate_magnets(run, stride=2, hide_x_negative=False)
    assert centers.shape == (2, 3)
    assert phi.shape == (2,)
    assert ring_id.shape == (2,)
    assert layer_id.shape == (2,)
