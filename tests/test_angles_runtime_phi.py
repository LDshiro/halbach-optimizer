import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import phi_rkn_from_run
from halbach.constants import phi0
from halbach.run_io import load_run
from halbach.symmetry import build_mirror_x0
from halbach.symmetry_fourier import build_fourier_x0_features, delta_full_from_fourier


def _write_run(
    run_dir: Path,
    *,
    N: int,
    K: int,
    R: int,
    alphas: NDArray[np.float64],
    r_bases: NDArray[np.float64],
    ring_offsets: NDArray[np.float64],
    extras: dict[str, NDArray[np.float64]] | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    payload = dict(
        alphas_opt=alphas,
        r_bases_opt=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )
    if extras:
        payload.update(extras)
    np.savez(run_dir / "results.npz", **payload)
    if meta is not None:
        (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_phi_rkn_legacy_matches_formula(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    N = 12
    K = 4
    R = 2
    alphas = 1e-3 * np.arange(R * K, dtype=np.float64).reshape(R, K)
    r_bases = np.linspace(0.2, 0.203, K, dtype=np.float64)
    ring_offsets = np.array([0.0, 0.01], dtype=np.float64)
    _write_run(run_dir, N=N, K=K, R=R, alphas=alphas, r_bases=r_bases, ring_offsets=ring_offsets)

    run = load_run(run_dir)
    phi_rkn = phi_rkn_from_run(run)
    theta = run.geometry.theta
    sin2 = run.geometry.sin2
    expected = 2.0 * theta[None, None, :] + phi0 + alphas[:, :, None] * sin2[None, None, :]
    np.testing.assert_allclose(phi_rkn, expected, atol=1e-12)


def test_phi_rkn_delta_rep_mirror(tmp_path: Path) -> None:
    run_dir = tmp_path / "delta"
    run_dir.mkdir()
    N = 12
    K = 4
    R = 1
    mirror = build_mirror_x0(N)
    delta_rep = np.zeros((K, mirror.rep_idx.size), dtype=np.float64)
    delta_rep[0, 0] = 0.1
    alphas = np.zeros((R, K), dtype=np.float64)
    r_bases = np.linspace(0.2, 0.203, K, dtype=np.float64)
    ring_offsets = np.zeros(R, dtype=np.float64)
    _write_run(
        run_dir,
        N=N,
        K=K,
        R=R,
        alphas=alphas,
        r_bases=r_bases,
        ring_offsets=ring_offsets,
        extras={"delta_rep_opt": delta_rep},
        meta={"angle_model": "delta-rep-x0"},
    )

    run = load_run(run_dir)
    phi_rkn = phi_rkn_from_run(run)
    base = 2.0 * run.geometry.theta + phi0
    delta_full = phi_rkn[0] - base[None, :]
    np.testing.assert_allclose(delta_full[:, mirror.mirror_idx], -delta_full, atol=1e-12)
    np.testing.assert_allclose(delta_full[:, mirror.fixed_idx], 0.0, atol=1e-12)


def test_phi_rkn_fourier_mirror(tmp_path: Path) -> None:
    run_dir = tmp_path / "fourier"
    run_dir.mkdir()
    N = 12
    K = 4
    R = 1
    H = 2
    alphas = np.zeros((R, K), dtype=np.float64)
    r_bases = np.linspace(0.2, 0.203, K, dtype=np.float64)
    ring_offsets = np.zeros(R, dtype=np.float64)
    coeffs = np.zeros((K, 2 * H), dtype=np.float64)
    coeffs[1, 0] = 0.05
    _write_run(
        run_dir,
        N=N,
        K=K,
        R=R,
        alphas=alphas,
        r_bases=r_bases,
        ring_offsets=ring_offsets,
        extras={"fourier_coeffs_opt": coeffs},
        meta={"angle_model": "fourier-x0", "fourier_H": H},
    )

    run = load_run(run_dir)
    phi_rkn = phi_rkn_from_run(run)
    cos_odd, sin_even = build_fourier_x0_features(run.geometry.theta, H)
    delta_full = delta_full_from_fourier(coeffs, cos_odd, sin_even)
    base = 2.0 * run.geometry.theta + phi0
    expected = base[None, :] + delta_full
    np.testing.assert_allclose(phi_rkn[0], expected, atol=1e-12)
