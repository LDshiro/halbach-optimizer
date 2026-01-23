import sys
from pathlib import Path

from halbach.gui.opt_job import build_command, tail_log


def test_build_command_basic(tmp_path: Path) -> None:
    in_path = tmp_path / "input.npz"
    out_dir = tmp_path / "out"
    cmd = build_command(
        in_path,
        out_dir,
        maxiter=123,
        gtol=1e-12,
        roi_r=0.14,
        roi_step=0.02,
        rho_gn=0.0,
        sigma_alpha_deg=0.5,
        sigma_r_mm=0.2,
        run_mc=True,
        mc_samples=10,
    )
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-u", "-m", "halbach.cli.optimize_run"]
    assert "--in" in cmd
    assert "--out" in cmd
    assert "--run-mc" in cmd


def test_tail_log_handles_missing_and_existing(tmp_path: Path) -> None:
    log_path = tmp_path / "opt.log"
    assert tail_log(log_path, n_lines=5) == ""

    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    tail = tail_log(log_path, n_lines=2)
    assert tail.strip().splitlines() == ["line2", "line3"]
