import sys
from datetime import datetime
from pathlib import Path

from halbach.gui.opt_job import (
    build_command,
    build_generate_command,
    build_generate_out_dir,
    tail_log,
)


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
        r_bound_mode="relative",
        r_lower_delta_mm=25.0,
        r_upper_delta_mm=35.0,
        r_no_upper=True,
        r_min_mm=0.0,
        r_max_mm=1000.0,
        sigma_alpha_deg=0.5,
        sigma_r_mm=0.2,
        run_mc=True,
        mc_samples=10,
    )
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-u", "-m", "halbach.cli.optimize_run"]
    assert "--in" in cmd
    assert "--out" in cmd
    assert "--r-bound-mode" in cmd
    assert "--r-lower-delta-mm" in cmd
    assert "--r-upper-delta-mm" in cmd
    assert "--r-no-upper" in cmd
    assert "--r-min-mm" in cmd
    assert "--r-max-mm" in cmd
    assert "--fix-center-radius-layers" in cmd
    assert "--run-mc" in cmd


def test_build_generate_command_basic(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cmd = build_generate_command(
        out_dir,
        N=48,
        R=3,
        K=24,
        Lz=0.64,
        diameter_mm=400.0,
        ring_offset_step_mm=12.0,
        name="demo",
    )
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "halbach.cli.generate_run"]
    assert "--out" in cmd
    assert "--N" in cmd
    assert "--diameter-mm" in cmd
    assert "--name" in cmd


def test_build_generate_out_dir_naming(tmp_path: Path) -> None:
    stamp = datetime(2025, 1, 2, 3, 4, 5)
    out_dir = build_generate_out_dir(tmp_path, tag="init", N=48, R=3, K=24, timestamp=stamp)
    assert out_dir.name == "20250102_030405_init_N48_R3_K24"

    out_dir.mkdir()
    out_dir2 = build_generate_out_dir(
        tmp_path,
        tag="init",
        N=48,
        R=3,
        K=24,
        timestamp=stamp,
        suffix_provider=lambda: "abcd",
    )
    assert out_dir2.name == "20250102_030405_init_N48_R3_K24_abcd"


def test_tail_log_handles_missing_and_existing(tmp_path: Path) -> None:
    log_path = tmp_path / "opt.log"
    assert tail_log(log_path, n_lines=5) == ""

    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    tail = tail_log(log_path, n_lines=2)
    assert tail.strip().splitlines() == ["line2", "line3"]
