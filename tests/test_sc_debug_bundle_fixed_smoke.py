import json
from pathlib import Path

import numpy as np

from halbach.cli.debug_sc_run import run as debug_run


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
    (run_dir / "meta.json").write_text(
        json.dumps({"magnetization": {"model_effective": "fixed"}}, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_sc_debug_bundle_fixed_smoke(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path)
    out_dir = tmp_path / "out"

    code = debug_run(["--run", str(run_dir), "--out", str(out_dir), "--no-field-scale-check"])
    assert code == 0

    debug_dir = out_dir / "sc_debug"
    summary_path = debug_dir / "summary.json"
    report_path = debug_dir / "check_report.json"

    assert summary_path.is_file()
    assert report_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["model_effective"] == "fixed"
    assert report["pass"] is True
