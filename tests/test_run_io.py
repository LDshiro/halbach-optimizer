import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from halbach.run_io import load_run
from halbach.tools.normalize_run import normalize_run


def _write_minimal_run(run_dir: Path) -> None:
    """GUIが前提とする最小限の run 構造を tmp_path 配下に作る。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    # small but valid shapes
    N = 12
    K = 6
    R = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).astype(np.float64)
    sin2th = np.sin(2.0 * theta).astype(np.float64)
    cth = np.cos(theta).astype(np.float64)
    sth = np.sin(theta).astype(np.float64)

    z_layers = np.linspace(-0.05, 0.05, K).astype(np.float64)
    ring_offsets = np.array([0.0], dtype=np.float64)

    rng = np.random.default_rng(0)
    alphas_opt = (1e-3 * rng.standard_normal((R, K))).astype(np.float64)
    r_bases_opt = (0.2 + 1e-4 * rng.standard_normal(K)).astype(np.float64)

    # 互換性のため alphas / r_bases も同梱（実装がどちらを優先しても読めるように）
    np.savez_compressed(
        run_dir / "results.npz",
        alphas_opt=alphas_opt,
        r_bases_opt=r_bases_opt,
        alphas=alphas_opt,
        r_bases=r_bases_opt,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )

    (run_dir / "meta.json").write_text(
        json.dumps({"schema_version": 1, "name": "test-run"}, indent=2),
        encoding="utf-8",
    )


def test_load_run_builds_geometry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    _write_minimal_run(run_dir)

    rb = load_run(run_dir)

    # geometry が生成されていること（ここが今回の主目的）
    assert rb.geometry.N == rb.results.theta.size
    assert rb.geometry.K == rb.results.z_layers.size
    assert rb.geometry.R == rb.results.ring_offsets.size

    assert rb.geometry.theta.shape == (rb.geometry.N,)
    assert rb.geometry.cth.shape == (rb.geometry.N,)
    assert rb.geometry.sth.shape == (rb.geometry.N,)

    # Geometry 側のフィールド名が sin2 ならこのassert、sin2thなら適宜置換してください
    assert rb.geometry.sin2.shape == (rb.geometry.N,)


def _make_arrays(R: int = 1, K: int = 4, N: int = 6) -> dict[str, npt.NDArray[np.floating[Any]]]:
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.1, 0.1, K)
    ring_offsets = np.zeros(R, dtype=float)
    alphas = 1e-3 * np.arange(R * K, dtype=float).reshape(R, K)
    r_bases = 0.2 + 1e-3 * np.arange(K, dtype=float)

    return {
        "theta": theta,
        "sin2th": sin2th,
        "cth": cth,
        "sth": sth,
        "z_layers": z_layers,
        "ring_offsets": ring_offsets,
        "alphas": alphas,
        "r_bases": r_bases,
    }


def test_load_run_dir_with_meta(tmp_path: Path) -> None:
    arrays = _make_arrays()
    results_path = tmp_path / "results.npz"
    np.savez(
        results_path,
        alphas_opt=arrays["alphas"],
        r_bases_opt=arrays["r_bases"],
        theta=arrays["theta"],
        sin2th=arrays["sin2th"],
        cth=arrays["cth"],
        sth=arrays["sth"],
        z_layers=arrays["z_layers"],
        ring_offsets=arrays["ring_offsets"],
    )
    meta = {"label": "demo"}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    bundle = load_run(tmp_path)

    assert bundle.results_path == results_path
    assert bundle.meta_path == tmp_path / "meta.json"
    assert bundle.meta["label"] == "demo"
    assert bundle.results.alphas.shape == arrays["alphas"].shape
    assert bundle.geometry.N == arrays["theta"].size


def test_load_run_sin2_compat(tmp_path: Path) -> None:
    arrays = _make_arrays()
    results_path = tmp_path / "results.npz"
    np.savez(
        results_path,
        alphas_opt=arrays["alphas"],
        r_bases_opt=arrays["r_bases"],
        theta=arrays["theta"],
        sin2=arrays["sin2th"],
        z_layers=arrays["z_layers"],
        ring_offsets=arrays["ring_offsets"],
    )

    bundle = load_run(results_path)

    np.testing.assert_allclose(bundle.results.sin2th, arrays["sin2th"])


def test_load_run_from_results_file(tmp_path: Path) -> None:
    arrays = _make_arrays()
    results_path = tmp_path / "custom_results.npz"
    np.savez(
        results_path,
        alphas=arrays["alphas"],
        r_bases=arrays["r_bases"],
        theta=arrays["theta"],
        sin2th=arrays["sin2th"],
        z_layers=arrays["z_layers"],
        ring_offsets=arrays["ring_offsets"],
    )

    bundle = load_run(results_path)

    assert bundle.results_path == results_path
    assert bundle.run_dir == tmp_path


def test_load_run_prefers_matching_meta(tmp_path: Path) -> None:
    arrays = _make_arrays()
    results_path = tmp_path / "sample_results.npz"
    np.savez(
        results_path,
        alphas=arrays["alphas"],
        r_bases=arrays["r_bases"],
        theta=arrays["theta"],
        sin2th=arrays["sin2th"],
        z_layers=arrays["z_layers"],
        ring_offsets=arrays["ring_offsets"],
    )
    (tmp_path / "sample_meta.json").write_text(json.dumps({"picked": 1}), encoding="utf-8")
    (tmp_path / "sample_meta2.json").write_text(json.dumps({"picked": 2}), encoding="utf-8")

    bundle = load_run(results_path)

    assert bundle.meta_path == tmp_path / "sample_meta.json"
    assert bundle.meta["picked"] == 1


def test_normalize_run_writes_standard_files(tmp_path: Path) -> None:
    arrays = _make_arrays()
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    np.savez(
        in_dir / "results.npz",
        alphas_opt=arrays["alphas"],
        r_bases_opt=arrays["r_bases"],
        theta=arrays["theta"],
        sin2th=arrays["sin2th"],
        z_layers=arrays["z_layers"],
        ring_offsets=arrays["ring_offsets"],
    )

    out_dir = tmp_path / "normalized"
    normalize_run(in_dir, out_dir)

    results_out = out_dir / "results.npz"
    assert results_out.is_file()
    with np.load(results_out) as data:
        for key in (
            "alphas_opt",
            "r_bases_opt",
            "theta",
            "sin2th",
            "cth",
            "sth",
            "z_layers",
            "ring_offsets",
        ):
            assert key in data.files

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["normalized_from"]
    assert meta["normalized_at"]
