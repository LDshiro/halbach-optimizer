from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from halbach.constants import m0
from halbach.magnetization_runtime import (
    compute_m_flat_from_run,
    get_magnetization_config_from_meta,
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


def test_magnetization_config_fixed() -> None:
    meta: dict[str, object] = {}
    model, sc = get_magnetization_config_from_meta(meta)
    assert model == "fixed"
    assert sc == {}


def test_magnetization_runtime_calls_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from halbach.autodiff import jax_self_consistent as sc_mod

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

    calls = {"dipole": 0, "multi": 0}

    def fake_dipole(*_args: object, **_kwargs: object) -> jnp.ndarray:
        calls["dipole"] += 1
        return jnp.full((R * K * N,), float(m0), dtype=jnp.float64)

    def fake_multi(*_args: object, **_kwargs: object) -> jnp.ndarray:
        calls["multi"] += 1
        return jnp.full((R * K * N,), float(m0), dtype=jnp.float64)

    monkeypatch.setattr(sc_mod, "solve_p_easy_axis_near", fake_dipole)
    monkeypatch.setattr(sc_mod, "solve_p_easy_axis_near_multi_dipole", fake_multi)

    sc_meta = {
        "magnetization": {
            "model_effective": "self-consistent-easy-axis",
            "self_consistent": {
                "chi": 0.05,
                "Nd": 1.0 / 3.0,
                "p0": float(m0),
                "volume_mm3": 1000.0,
                "iters": 3,
                "omega": 0.6,
                "near_window": {"wr": 0, "wz": 1, "wphi": 1},
                "near_kernel": "dipole",
                "subdip_n": 2,
            },
        }
    }
    sc_dir = tmp_path / "sc_run"
    _write_meta(sc_dir, sc_meta)

    compute_m_flat_from_run(sc_dir, geom, phi_rkn, r0_rkn)
    assert calls["dipole"] == 1
    assert calls["multi"] == 0

    fixed_meta = {"magnetization": {"model_effective": "fixed"}}
    fixed_dir = tmp_path / "fixed_run"
    _write_meta(fixed_dir, fixed_meta)

    compute_m_flat_from_run(fixed_dir, geom, phi_rkn, r0_rkn)
    assert calls["dipole"] == 1
    assert calls["multi"] == 0
