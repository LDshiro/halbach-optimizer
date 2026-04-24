from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from halbach.perturbation_eval import (
    PerturbationConfig,
    run_perturbation_case,
    save_perturbation_result,
)
from halbach.run_io import load_run


def _write_min_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    R = 1
    K = 2
    N = 6
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.01, 0.01, K)
    ring_offsets = np.array([0.0], dtype=float)
    alphas = np.zeros((R, K), dtype=float)
    r_bases = np.array([0.10, 0.11], dtype=float)

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
    (run_dir / "meta.json").write_text(
        json.dumps({"magnetization": {"model_effective": "fixed"}}),
        encoding="utf-8",
    )
    return run_dir


def test_save_perturbation_result_outputs(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("matplotlib")

    run = load_run(_write_min_run(tmp_path))
    cfg = PerturbationConfig(
        sigma_rel_pct=1.0,
        sigma_phi_deg=0.5,
        seed=7,
        roi_radius_m=0.03,
        roi_samples=10,
        map_radius_m=0.02,
        map_step_m=0.02,
        sc_cfg={
            "chi": 0.0,
            "Nd": 1.0 / 3.0,
            "p0": 1.0,
            "volume_mm3": 1000.0,
            "iters": 5,
            "omega": 0.6,
            "near_window": {"wr": 0, "wz": 1, "wphi": 1},
            "near_kernel": "dipole",
            "subdip_n": 2,
        },
    )
    result = run_perturbation_case(run, cfg)

    out_dir = tmp_path / "variation"
    paths = save_perturbation_result(result, out_dir, cfg, run)

    assert paths["roi_vectors_npz"].is_file()
    assert paths["roi_vectors_csv"].is_file()
    assert paths["map_xy_npz"].is_file()
    assert paths["map_xy_png"].is_file()
    assert paths["run_info_json"].is_file()

    with np.load(paths["roi_vectors_npz"], allow_pickle=False) as data:
        assert "pts_xyz_m" in data
        assert "B_xyz_T" in data
        assert "Bnorm_T" in data
        assert "B0_xyz_T" in data
        assert "ppm" in data
        assert int(data["seed"]) == cfg.seed
