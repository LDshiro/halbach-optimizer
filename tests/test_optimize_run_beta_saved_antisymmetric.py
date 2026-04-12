from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import halbach.cli.optimize_run as opt
from halbach.geom import build_param_map
from halbach.solvers.types import SolveResult, SolveTrace


def _write_min_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    R = 1
    K = 6
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
    return run_dir


def test_optimize_run_saves_beta_antisymmetric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("jax")
    import halbach.autodiff.jax_objective_legacy_beta_tilt as beta_mod

    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    def fake_beta_objective(
        alphas: np.ndarray,
        beta_tilt_x: np.ndarray,
        r_bases: np.ndarray,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            1.0,
            np.zeros_like(alphas),
            np.zeros_like(beta_tilt_x),
            np.zeros_like(r_bases),
            0.1,
        )

    def fake_solve(
        fun: Callable[[np.ndarray], tuple[float, np.ndarray, dict[str, Any]]],
        x0: np.ndarray,
        _bounds: object,
        _opt: object,
        iter_callback: object | None = None,
    ) -> SolveResult:
        x = np.array(x0, copy=True)
        param_map = build_param_map(R=1, K=6, n_fix_radius=2)
        legacy_alpha_dim = int(param_map.free_alpha_idx.size)
        beta_vals = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        x[legacy_alpha_dim : legacy_alpha_dim + beta_vals.size] = beta_vals
        f, g, extras = fun(x)
        trace = SolveTrace(
            iters=[0],
            f=[float(f)],
            gnorm=[float(np.linalg.norm(g))],
            extras=[extras],
        )
        return SolveResult(
            x=x,
            fun=float(f),
            success=True,
            message="ok",
            nit=1,
            nfev=1,
            njev=1,
            trace=trace,
        )

    monkeypatch.setattr(beta_mod, "objective_with_grads_fixed_beta_tilt_jax", fake_beta_objective)
    monkeypatch.setattr(opt, "solve_lbfgsb", fake_solve)
    monkeypatch.setattr(opt, "_git_hash", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        in_path=str(run_dir),
        out_dir=str(out_dir),
        maxiter=1,
        gtol=1e-12,
        log_every=10,
        log_precision=3,
        roi_r=0.05,
        roi_step=0.05,
        roi_mode="volume-grid",
        roi_samples=10,
        roi_seed=0,
        roi_half_x=False,
        roi_max_points=0,
        field_scale=1e6,
        angle_model="legacy-alpha",
        grad_backend="jax",
        fourier_H=4,
        lambda0=0.0,
        lambda_theta=0.0,
        lambda_z=0.0,
        angle_init="from-run",
        enable_beta_tilt_x=True,
        beta_tilt_x_bound_deg=20.0,
        mag_model="fixed",
        sc_chi=0.0,
        sc_Nd=1.0 / 3.0,
        sc_p0=1.0,
        sc_volume_mm3=1000.0,
        sc_iters=30,
        sc_omega=0.6,
        sc_near_wr=0,
        sc_near_wz=1,
        sc_near_wphi=2,
        sc_near_kernel="dipole",
        sc_gl_order=None,
        sc_subdip_n=2,
        r_bound_mode="relative",
        r_lower_delta_mm=5.0,
        r_upper_delta_mm=5.0,
        r_no_upper=False,
        r_min_mm=0.0,
        r_max_mm=1e9,
        min_radius_drop_mm=None,
        fix_center_radius_layers=2,
        fix_radius_layer_mode="center",
        log_level="INFO",
        debug_stacks_secs=0,
        sc_debug=False,
        sc_debug_scale_check=True,
        dry_run=False,
    )

    code = opt.run_optimize(args)
    assert code == 0

    with np.load(out_dir / "results.npz", allow_pickle=False) as data:
        beta = np.asarray(data["beta_tilt_x_opt"], dtype=np.float64)

    np.testing.assert_allclose(beta[:, 5], -beta[:, 0])
    np.testing.assert_allclose(beta[:, 4], -beta[:, 1])
    np.testing.assert_allclose(beta[:, 3], -beta[:, 2])
