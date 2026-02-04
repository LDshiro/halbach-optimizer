import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import halbach.cli.optimize_run as opt


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


def test_dryrun_uses_self_consistent_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("jax")
    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    calls: dict[str, int] = {"count": 0}

    def fake_sc_objective(
        alphas: np.ndarray,
        r_bases: np.ndarray,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
        calls["count"] += 1
        return (
            0.0,
            np.zeros_like(alphas),
            np.zeros_like(r_bases),
            0.0,
            {
                "sc_p_min": 1.0,
                "sc_p_max": 1.0,
                "sc_p_mean": 1.0,
                "sc_p_std": 0.0,
                "sc_p_rel_std": 0.0,
                "sc_near_kernel": "dipole",
                "sc_subdip_n": 2,
                "sc_near_deg_max": 1,
            },
        )

    def fail_fixed(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("fixed objective should not be called")

    import halbach.autodiff.jax_objective as fixed_mod
    import halbach.autodiff.jax_objective_self_consistent_legacy as sc_mod

    monkeypatch.setattr(
        sc_mod, "objective_with_grads_self_consistent_legacy_jax", fake_sc_objective
    )
    monkeypatch.setattr(fixed_mod, "objective_with_grads_fixed_jax", fail_fixed)
    monkeypatch.setattr(opt, "_git_hash", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        in_path=str(run_dir),
        out_dir=str(out_dir),
        maxiter=10,
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
        sc_chi=0.05,
        sc_Nd=1.0 / 3.0,
        sc_p0=1.0,
        sc_volume_mm3=1000.0,
        sc_iters=5,
        sc_omega=0.6,
        sc_near_wr=0,
        sc_near_wz=1,
        sc_near_wphi=1,
        sc_near_kernel="dipole",
        sc_subdip_n=2,
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
        sc_debug=False,
        sc_debug_scale_check=True,
        dry_run=True,
    )

    code = opt.run_optimize(args)
    assert code == 0
    assert calls["count"] == 1

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["magnetization"]["model_effective"] == "self-consistent-easy-axis"
