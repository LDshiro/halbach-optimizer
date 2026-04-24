from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import halbach.cli.optimize_run as opt
from halbach.constants import phi0
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
    (run_dir / "meta.json").write_text(
        json.dumps({"angle_model": "legacy-alpha"}), encoding="utf-8"
    )
    return run_dir


def test_optimize_run_saves_magnet_export_for_self_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("jax")
    import halbach.autodiff.jax_objective_self_consistent_legacy as sc_mod

    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    def fake_sc_objective(
        alphas: np.ndarray,
        r_bases: np.ndarray,
        *_args: object,
        **kwargs: object,
    ) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, Any]]:
        return (
            1.25,
            np.zeros_like(alphas),
            np.zeros_like(r_bases),
            0.2,
            {
                "sc_near_deg_max": 1,
                "sc_subdip_n": int(kwargs.get("subdip_n", 1)),
            },
        )

    def fake_solve(
        fun: Any,
        x0: np.ndarray,
        _bounds: object,
        _opt: object,
        iter_callback: object | None = None,
    ) -> SolveResult:
        f, g, extras = fun(np.array(x0, copy=True))
        if iter_callback is not None:
            iter_callback(0, np.array(x0, copy=True), float(f), np.array(g, copy=True), extras)
        trace = SolveTrace(
            iters=[0],
            f=[float(f)],
            gnorm=[float(np.linalg.norm(g))],
            extras=[extras],
        )
        return SolveResult(
            x=np.array(x0, copy=True),
            fun=float(f),
            success=True,
            message="ok",
            nit=1,
            nfev=1,
            njev=1,
            trace=trace,
        )

    monkeypatch.setattr(
        sc_mod, "objective_with_grads_self_consistent_legacy_jax", fake_sc_objective
    )
    monkeypatch.setattr(
        opt,
        "compute_p_flat_self_consistent_jax",
        lambda phi_rkn, *_args, **_kwargs: np.ones(phi_rkn.size, dtype=np.float64),
    )
    monkeypatch.setattr(opt, "sc_cfg_fingerprint", lambda *_args, **_kwargs: "sc-config-fp")
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
        enable_beta_tilt_x=False,
        beta_tilt_x_bound_deg=20.0,
        mag_model="self-consistent-easy-axis",
        sc_chi=0.1,
        sc_Nd=1.0 / 3.0,
        sc_p0=1.0,
        sc_volume_mm3=1000.0,
        sc_iters=4,
        sc_omega=0.6,
        sc_near_wr=0,
        sc_near_wz=1,
        sc_near_wphi=2,
        sc_near_kernel="multi-dipole",
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

    with np.load(out_dir / "results.npz", allow_pickle=True) as data:
        centers = np.asarray(data["magnet_centers_m"], dtype=np.float64)
        phi = np.asarray(data["magnet_phi_rad"], dtype=np.float64)
        dims_mm = np.asarray(data["magnet_dimensions_mm"], dtype=np.float64)
        dims_m = np.asarray(data["magnet_dimensions_m"], dtype=np.float64)
        ring_id = np.asarray(data["magnet_ring_id"], dtype=np.int32)
        layer_id = np.asarray(data["magnet_layer_id"], dtype=np.int32)
        theta_id = np.asarray(data["magnet_theta_id"], dtype=np.int32)

    assert centers.shape == (48, 3)
    assert phi.shape == (48,)
    assert ring_id.shape == (48,)
    assert layer_id.shape == (48,)
    assert theta_id.shape == (48,)
    np.testing.assert_allclose(dims_mm, np.array([10.0, 10.0, 10.0], dtype=np.float64))
    np.testing.assert_allclose(dims_m, np.array([0.01, 0.01, 0.01], dtype=np.float64))
    np.testing.assert_allclose(centers[0], np.array([0.2, 0.0, -0.02], dtype=np.float64))
    np.testing.assert_allclose(phi[0], np.array(phi0, dtype=np.float64))

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    fusion = meta["fusion360_export"]
    assert fusion["coordinates_unit"] == "m"
    assert fusion["angles_unit"] == "rad"
    assert fusion["active_magnets_only"] is True
    assert fusion["magnet_dimensions_source"] == "self-consistent-volume-equivalent-cube"
    np.testing.assert_allclose(
        np.asarray(fusion["magnet_dimensions_mm"], dtype=np.float64),
        np.array([10.0, 10.0, 10.0], dtype=np.float64),
    )
