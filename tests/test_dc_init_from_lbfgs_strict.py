import json
from pathlib import Path

import numpy as np
import pytest

from halbach.dc.init_from_lbfgs import load_phi_flat_from_lbfgs_run
from halbach.symmetry import build_mirror_x0


def _write_lbfgs_run(tmp_path: Path, angle_model: str) -> Path:
    run_dir = tmp_path / f"run_{angle_model}"
    run_dir.mkdir()

    R = 1
    K = 2
    N = 4
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.01, 0.01, K)
    ring_offsets = np.array([0.0], dtype=float)
    alphas = np.zeros((R, K), dtype=float)
    r_bases = 0.2 + 1e-3 * np.arange(K, dtype=float)

    npz_kwargs: dict[str, np.ndarray] = {
        "alphas_opt": alphas,
        "r_bases_opt": r_bases,
        "theta": theta,
        "sin2th": sin2th,
        "cth": cth,
        "sth": sth,
        "z_layers": z_layers,
        "ring_offsets": ring_offsets,
    }

    meta: dict[str, object] = {"angle_model": angle_model}
    if angle_model == "delta-rep-x0":
        mirror = build_mirror_x0(N)
        delta_rep = np.zeros((K, mirror.rep_idx.size), dtype=float)
        npz_kwargs["delta_rep_opt"] = delta_rep
    elif angle_model == "fourier-x0":
        H = 1
        coeffs = np.zeros((K, 2 * H), dtype=float)
        npz_kwargs["fourier_coeffs_opt"] = coeffs
        meta["fourier_H"] = H

    np.savez(run_dir / "results.npz", **npz_kwargs)
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def test_init_from_lbfgs_rejects_legacy_alpha(tmp_path: Path) -> None:
    run_dir = _write_lbfgs_run(tmp_path, "legacy-alpha")
    with pytest.raises(ValueError, match="angle_model must be one of"):
        load_phi_flat_from_lbfgs_run(run_dir, expected_R=1, expected_K=2, expected_N=4)


def test_init_from_lbfgs_accepts_delta_rep(tmp_path: Path) -> None:
    run_dir = _write_lbfgs_run(tmp_path, "delta-rep-x0")
    phi_flat, model = load_phi_flat_from_lbfgs_run(
        run_dir, expected_R=1, expected_K=2, expected_N=4
    )
    assert model == "delta-rep-x0"
    assert phi_flat.shape == (1 * 2 * 4,)
