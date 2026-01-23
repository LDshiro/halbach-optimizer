import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import halbach.cli.optimize_run as opt


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


def test_roi_warning_and_downsample_helper() -> None:
    assert opt._roi_point_warning(6, 5)
    assert not opt._roi_point_warning(5, 5)

    pts = np.arange(30, dtype=float).reshape(10, 3)
    result = opt._downsample_roi_points(pts, max_points=5, seed=0)
    assert result.downsampled
    assert result.npts_before == 10
    assert result.pts.shape == (5, 3)


def test_opt_log_path_helper(tmp_path: Path) -> None:
    path = opt._opt_log_path(tmp_path)
    assert path == tmp_path / "opt.log"


def test_dry_run_calls_objective_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    calls = {"count": 0}

    def fake_fun_grad(
        *args: object, **kwargs: object
    ) -> tuple[float, npt.NDArray[np.floating[Any]], float, float, float]:
        x = args[0]
        calls["count"] += 1
        return 0.0, np.zeros_like(x), 1.0, 0.0, 0.0

    def fail_solve(*args: object, **kwargs: object) -> object:
        raise AssertionError("solve_lbfgsb should not be called in dry-run")

    monkeypatch.setattr(opt, "fun_grad_gradnorm_fixed", fake_fun_grad)
    monkeypatch.setattr(opt, "solve_lbfgsb", fail_solve)
    monkeypatch.setattr(opt, "_git_hash", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        in_path=str(run_dir),
        out_dir=str(out_dir),
        maxiter=50,
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
        rho_gn=0.0,
        field_scale=1e6,
        sigma_alpha_deg=0.5,
        sigma_r_mm=0.2,
        eps_hvp=1e-6,
        min_radius_drop_mm=20.0,
        fix_center_radius_layers=2,
        mc_samples=10,
        run_mc=False,
        log_level="INFO",
        debug_stacks_secs=0,
        dry_run=True,
    )

    code = opt.run_optimize(args)
    assert code == 0
    assert calls["count"] == 1


def test_format_iter_log() -> None:
    line = opt._format_iter_log(
        10,
        1.2345,
        2.3456,
        3.4,
        0.0123,
        precision=3,
        iter_width=4,
    )
    assert line.startswith("[iter 0010]")
    assert "J=1.234e+00" in line
    assert "gnorm=2.346e+00" in line
    assert "|B0|=3.400 mT" in line
    assert "dt_eval=0.012s" in line
