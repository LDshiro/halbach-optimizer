import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import halbach.cli.optimize_run as opt
from halbach.magnetization_runtime import sc_cfg_fingerprint
from halbach.solvers.types import SolveResult, SolveTrace


def _write_min_run(tmp_path: Path) -> Path:
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
    return run_dir


def test_optimize_run_saves_sc_p_flat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("jax")
    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    def fake_sc_objective(
        alphas: np.ndarray,
        r_bases: np.ndarray,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, float]]:
        return 1.0, np.zeros_like(alphas), np.zeros_like(r_bases), 0.1, {}

    def fake_solve(
        fun: Callable[[np.ndarray], tuple[float, np.ndarray, dict[str, Any]]],
        x0: np.ndarray,
        _bounds: object,
        _opt: object,
        iter_callback: object | None = None,
    ) -> SolveResult:
        f, g, extras = fun(x0)
        trace = SolveTrace(
            iters=[0], f=[float(f)], gnorm=[float(np.linalg.norm(g))], extras=[extras]
        )
        return SolveResult(
            x=x0,
            fun=float(f),
            success=True,
            message="ok",
            nit=1,
            nfev=1,
            njev=1,
            trace=trace,
        )

    sc_cfg: dict[str, Any] = dict(
        chi=0.05,
        Nd=1.0 / 3.0,
        p0=1.0,
        volume_mm3=1000.0,
        iters=3,
        omega=0.6,
        near_window=dict(wr=0, wz=1, wphi=1),
        near_kernel="dipole",
        subdip_n=2,
    )
    expected_fp = sc_cfg_fingerprint(sc_cfg)

    def fake_p_flat(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.full((1 * 4 * 8,), 1.1, dtype=np.float64)

    import halbach.autodiff.jax_objective_self_consistent_legacy as sc_mod

    monkeypatch.setattr(
        sc_mod, "objective_with_grads_self_consistent_legacy_jax", fake_sc_objective
    )
    monkeypatch.setattr(opt, "solve_lbfgsb", fake_solve)
    monkeypatch.setattr(opt, "compute_p_flat_self_consistent_jax", fake_p_flat)
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
        mag_model="self-consistent-easy-axis",
        sc_chi=sc_cfg["chi"],
        sc_Nd=sc_cfg["Nd"],
        sc_p0=sc_cfg["p0"],
        sc_volume_mm3=sc_cfg["volume_mm3"],
        sc_iters=sc_cfg["iters"],
        sc_omega=sc_cfg["omega"],
        sc_near_wr=sc_cfg["near_window"]["wr"],
        sc_near_wz=sc_cfg["near_window"]["wz"],
        sc_near_wphi=sc_cfg["near_window"]["wphi"],
        sc_near_kernel=sc_cfg["near_kernel"],
        sc_subdip_n=sc_cfg["subdip_n"],
        r_bound_mode="relative",
        r_lower_delta_mm=5.0,
        r_upper_delta_mm=5.0,
        r_no_upper=False,
        r_min_mm=0.0,
        r_max_mm=1e9,
        min_radius_drop_mm=None,
        fix_center_radius_layers=2,
        log_level="INFO",
        debug_stacks_secs=0,
        dry_run=False,
    )

    code = opt.run_optimize(args)
    assert code == 0

    with np.load(out_dir / "results.npz", allow_pickle=False) as data:
        assert "sc_p_flat" in data
        assert "sc_cfg_fingerprint" in data
        assert data["sc_p_flat"].shape == (1 * 4 * 8,)
        fp_arr = np.asarray(data["sc_cfg_fingerprint"])
        fp_val = str(fp_arr.reshape(-1)[0])
        assert fp_val == expected_fp
