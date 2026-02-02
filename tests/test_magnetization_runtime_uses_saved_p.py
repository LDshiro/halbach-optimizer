import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from halbach.constants import m0
from halbach.magnetization_runtime import (
    build_m_flat_from_phi_and_p,
    compute_m_flat_from_run,
    sc_cfg_fingerprint,
)
from halbach.types import Geometry


def _build_geom() -> Geometry:
    N = 6
    K = 2
    R = 1
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.01, 0.01, K)
    ring_offsets = np.array([0.0], dtype=float)
    dz = float(np.median(np.abs(np.diff(z_layers)))) if K > 1 else 0.0
    Lz = float(np.max(z_layers) - np.min(z_layers))
    return Geometry(
        theta=theta,
        sin2=sin2,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )


def _write_meta(path: Path, meta: Mapping[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json_dumps(meta),
        encoding="utf-8",
    )


def json_dumps(data: Mapping[str, object]) -> str:
    import json

    return json.dumps(data, indent=2)


def test_magnetization_runtime_uses_saved_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geom = _build_geom()
    R = geom.R
    K = geom.K
    N = geom.N
    phi_rkn = np.zeros((R, K, N), dtype=np.float64)
    r_bases = np.full(K, 0.2, dtype=np.float64)
    rho = r_bases[None, :] + geom.ring_offsets[:, None]
    px = rho[:, :, None] * geom.cth[None, None, :]
    py = rho[:, :, None] * geom.sth[None, None, :]
    pz = np.broadcast_to(geom.z_layers[None, :, None], px.shape)
    r0_rkn = np.stack([px, py, pz], axis=-1)

    sc_cfg = {
        "chi": 0.05,
        "Nd": 1.0 / 3.0,
        "p0": float(m0),
        "volume_mm3": 1000.0,
        "iters": 3,
        "omega": 0.6,
        "near_window": {"wr": 0, "wz": 1, "wphi": 1},
        "near_kernel": "dipole",
        "subdip_n": 2,
    }
    meta = {
        "magnetization": {"model_effective": "self-consistent-easy-axis", "self_consistent": sc_cfg}
    }
    run_dir = tmp_path / "run"
    _write_meta(run_dir, meta)

    phi_flat = phi_rkn.reshape(-1)
    p_flat = np.linspace(0.9, 1.1, phi_flat.size, dtype=np.float64)
    fp = sc_cfg_fingerprint(sc_cfg)
    np.savez(run_dir / "results.npz", sc_p_flat=p_flat, sc_cfg_fingerprint=fp)

    if importlib.util.find_spec("jax") is not None:
        import halbach.autodiff.jax_self_consistent as sc_mod

        def fail_solver(*_args: object, **_kwargs: object) -> object:
            raise AssertionError(
                "self-consistent solver should not be called when sc_p_flat is saved"
            )

        monkeypatch.setattr(sc_mod, "solve_p_easy_axis_near", fail_solver)
        monkeypatch.setattr(sc_mod, "solve_p_easy_axis_near_multi_dipole", fail_solver)

    m_flat, debug = compute_m_flat_from_run(run_dir, geom, phi_rkn, r0_rkn)
    expected = build_m_flat_from_phi_and_p(phi_flat, p_flat)
    assert debug.get("sc_p_source") == "saved"
    assert debug.get("sc_cfg_fingerprint") == fp
    assert np.allclose(m_flat, expected)
