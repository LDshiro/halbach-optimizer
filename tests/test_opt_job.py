import sys
from datetime import datetime
from pathlib import Path

from halbach.gui.opt_job import (
    build_command,
    build_dc_ccp_sc_command,
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
        r_bound_mode="relative",
        r_lower_delta_mm=25.0,
        r_upper_delta_mm=35.0,
        r_no_upper=True,
        r_min_mm=0.0,
        r_max_mm=1000.0,
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
    assert "--mag-model" not in cmd
    assert "--sc-chi" not in cmd


def test_build_command_includes_self_consistent_flags(tmp_path: Path) -> None:
    in_path = tmp_path / "input.npz"
    out_dir = tmp_path / "out"
    cmd = build_command(
        in_path,
        out_dir,
        maxiter=50,
        gtol=1e-12,
        roi_r=0.14,
        roi_step=0.02,
        r_bound_mode="relative",
        r_lower_delta_mm=25.0,
        r_upper_delta_mm=35.0,
        r_no_upper=False,
        r_min_mm=0.0,
        r_max_mm=1000.0,
        mag_model="self-consistent-easy-axis",
        sc_chi=0.05,
        sc_Nd=1.0 / 3.0,
        sc_p0=1.1,
        sc_volume_mm3=999.0,
        sc_iters=20,
        sc_omega=0.5,
        sc_near_wr=0,
        sc_near_wz=1,
        sc_near_wphi=2,
        sc_near_kernel="gl-double-mixed",
        sc_subdip_n=2,
        sc_gl_order=3,
    )
    assert "--mag-model" in cmd
    assert "self-consistent-easy-axis" in cmd
    assert "--sc-chi" in cmd
    assert cmd[cmd.index("--sc-chi") + 1] == "0.05"
    assert "--sc-near-wphi" in cmd
    assert "--sc-near-kernel" in cmd
    assert cmd[cmd.index("--sc-near-kernel") + 1] == "gl-double-mixed"
    assert "--sc-gl-order" in cmd
    assert cmd[cmd.index("--sc-gl-order") + 1] == "3"


def test_build_command_includes_roi_sampling_flags(tmp_path: Path) -> None:
    in_path = tmp_path / "input.npz"
    out_dir = tmp_path / "out"
    cmd = build_command(
        in_path,
        out_dir,
        maxiter=50,
        gtol=1e-12,
        roi_r=0.14,
        roi_step=0.02,
        roi_mode="surface-fibonacci",
        roi_samples=500,
        r_bound_mode="relative",
        r_lower_delta_mm=25.0,
        r_upper_delta_mm=35.0,
        r_no_upper=False,
        r_min_mm=0.0,
        r_max_mm=1000.0,
    )
    assert "--roi-mode" in cmd
    assert cmd[cmd.index("--roi-mode") + 1] == "surface-fibonacci"
    assert "--roi-samples" in cmd
    assert cmd[cmd.index("--roi-samples") + 1] == "500"


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


def test_build_dc_ccp_sc_command_flags(tmp_path: Path) -> None:
    out_dir = tmp_path / "dc_out"
    cmd = build_dc_ccp_sc_command(
        out_dir,
        N=6,
        K=2,
        R=1,
        radius_m=0.1,
        length_m=0.02,
        roi_radius_m=0.05,
        roi_grid_n=11,
        wx=0.0,
        wy=1.0,
        wz=0.0,
        factor=1e-7,
        phi0=0.0,
        delta_nom_deg=5.0,
        delta_step_deg=None,
        tau0=1e-4,
        tau_mult=1.2,
        tau_max=1e-1,
        iters=5,
        tol=1e-6,
        tol_f=1e-9,
        reg_x=0.0,
        reg_p=0.0,
        reg_z=0.0,
        sc_eq=False,
        p_fix=1.0,
        sc_chi=0.05,
        sc_Nd=1.0 / 3.0,
        sc_p0=1.0,
        sc_volume_mm3=1000.0,
        sc_near_wr=0,
        sc_near_wz=1,
        sc_near_wphi=1,
        sc_near_kernel="dipole",
        sc_subdip_n=2,
        pmin=0.0,
        pmax=2.0,
        solver="ECOS",
        verbose=True,
    )
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-u", "-m", "halbach.cli.dc_ccp_sc_optimize_run"]
    assert "--no-sc-eq" in cmd
    assert "--p-fix" in cmd
    assert cmd[cmd.index("--p-fix") + 1] == "1.0"


def test_tail_log_handles_missing_and_existing(tmp_path: Path) -> None:
    log_path = tmp_path / "opt.log"
    assert tail_log(log_path, n_lines=5) == ""

    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    tail = tail_log(log_path, n_lines=2)
    assert tail.strip().splitlines() == ["line2", "line3"]
