from __future__ import annotations

import re
import secrets
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class OptJob:
    proc: subprocess.Popen[str]
    out_dir: Path
    log_path: Path
    command: list[str]


def build_command(
    in_path: str | Path,
    out_dir: str | Path,
    *,
    maxiter: int,
    gtol: float,
    roi_r: float,
    roi_step: float,
    roi_mode: str = "surface-fibonacci",
    roi_samples: int = 300,
    angle_model: str = "legacy-alpha",
    grad_backend: str | None = None,
    fourier_H: int = 4,
    lambda0: float = 0.0,
    lambda_theta: float = 0.0,
    lambda_z: float = 0.0,
    angle_init: str = "from-run",
    r_bound_mode: str = "relative",
    r_lower_delta_mm: float = 30.0,
    r_upper_delta_mm: float = 30.0,
    r_no_upper: bool = False,
    r_min_mm: float = 0.0,
    r_max_mm: float = 1e9,
    fix_center_radius_layers: int = 2,
    mag_model: str = "fixed",
    sc_chi: float = 0.0,
    sc_Nd: float = 1.0 / 3.0,
    sc_p0: float = 1.0,
    sc_volume_mm3: float = 1000.0,
    sc_iters: int = 30,
    sc_omega: float = 0.6,
    sc_near_wr: int = 0,
    sc_near_wz: int = 1,
    sc_near_wphi: int = 2,
    sc_near_kernel: str = "dipole",
    sc_subdip_n: int = 2,
    sc_gl_order: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "halbach.cli.optimize_run",
        "--in",
        str(in_path),
        "--out",
        str(out_dir),
        "--maxiter",
        str(maxiter),
        "--gtol",
        str(gtol),
        "--roi-r",
        str(roi_r),
        "--roi-step",
        str(roi_step),
        "--roi-mode",
        str(roi_mode),
        "--roi-samples",
        str(roi_samples),
        "--angle-model",
        str(angle_model),
        "--fourier-H",
        str(fourier_H),
        "--lambda0",
        str(lambda0),
        "--lambda-theta",
        str(lambda_theta),
        "--lambda-z",
        str(lambda_z),
        "--angle-init",
        str(angle_init),
        "--r-bound-mode",
        str(r_bound_mode),
        "--r-lower-delta-mm",
        str(r_lower_delta_mm),
        "--r-upper-delta-mm",
        str(r_upper_delta_mm),
        "--r-min-mm",
        str(r_min_mm),
        "--r-max-mm",
        str(r_max_mm),
        "--fix-center-radius-layers",
        str(fix_center_radius_layers),
    ]
    if r_no_upper:
        cmd.append("--r-no-upper")
    if grad_backend is not None:
        cmd.extend(["--grad-backend", str(grad_backend)])
    if mag_model != "fixed":
        cmd.extend(["--mag-model", str(mag_model)])
        cmd.extend(["--sc-chi", str(sc_chi)])
        cmd.extend(["--sc-Nd", str(sc_Nd)])
        cmd.extend(["--sc-p0", str(sc_p0)])
        cmd.extend(["--sc-volume-mm3", str(sc_volume_mm3)])
        cmd.extend(["--sc-iters", str(sc_iters)])
        cmd.extend(["--sc-omega", str(sc_omega)])
        cmd.extend(["--sc-near-wr", str(sc_near_wr)])
        cmd.extend(["--sc-near-wz", str(sc_near_wz)])
        cmd.extend(["--sc-near-wphi", str(sc_near_wphi)])
        cmd.extend(["--sc-near-kernel", str(sc_near_kernel)])
        if sc_near_kernel == "gl-double-mixed" and sc_gl_order is not None:
            cmd.extend(["--sc-gl-order", str(sc_gl_order)])
        cmd.extend(["--sc-subdip-n", str(sc_subdip_n)])
    return cmd


def build_dc_ccp_sc_command(
    out_dir: str | Path,
    *,
    N: int,
    K: int,
    R: int,
    radius_m: float,
    length_m: float,
    roi_radius_m: float,
    roi_grid_n: int,
    wx: float,
    wy: float,
    wz: float,
    factor: float,
    phi0: float,
    delta_nom_deg: float,
    delta_step_deg: float | None,
    tau0: float,
    tau_mult: float,
    tau_max: float,
    iters: int,
    tol: float,
    tol_f: float,
    reg_x: float,
    reg_p: float,
    reg_z: float,
    sc_eq: bool,
    p_fix: float | None,
    sc_chi: float,
    sc_Nd: float,
    sc_p0: float,
    sc_volume_mm3: float,
    sc_near_wr: int,
    sc_near_wz: int,
    sc_near_wphi: int,
    sc_near_kernel: str,
    sc_subdip_n: int,
    sc_gl_order: int | None = None,
    pmin: float,
    pmax: float,
    solver: str,
    init_run: str | Path | None = None,
    verbose: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "halbach.cli.dc_ccp_sc_optimize_run",
        "--out",
        str(out_dir),
        "--N",
        str(N),
        "--K",
        str(K),
        "--R",
        str(R),
        "--radius-m",
        str(radius_m),
        "--length-m",
        str(length_m),
        "--roi-radius-m",
        str(roi_radius_m),
        "--roi-grid-n",
        str(roi_grid_n),
        "--wx",
        str(wx),
        "--wy",
        str(wy),
        "--wz",
        str(wz),
        "--factor",
        str(factor),
        "--phi0",
        str(phi0),
        "--delta-nom-deg",
        str(delta_nom_deg),
        "--delta-step-deg",
        str(-1.0 if delta_step_deg is None else delta_step_deg),
        "--tau0",
        str(tau0),
        "--tau-mult",
        str(tau_mult),
        "--tau-max",
        str(tau_max),
        "--iters",
        str(iters),
        "--tol",
        str(tol),
        "--tol-f",
        str(tol_f),
        "--reg-x",
        str(reg_x),
        "--reg-p",
        str(reg_p),
        "--reg-z",
        str(reg_z),
        "--sc-chi",
        str(sc_chi),
        "--sc-Nd",
        str(sc_Nd),
        "--sc-p0",
        str(sc_p0),
        "--sc-volume-mm3",
        str(sc_volume_mm3),
        "--sc-near-wr",
        str(sc_near_wr),
        "--sc-near-wz",
        str(sc_near_wz),
        "--sc-near-wphi",
        str(sc_near_wphi),
        "--sc-near-kernel",
        str(sc_near_kernel),
        "--sc-subdip-n",
        str(sc_subdip_n),
        "--pmin",
        str(pmin),
        "--pmax",
        str(pmax),
        "--solver",
        str(solver),
    ]
    if sc_near_kernel == "gl-double-mixed" and sc_gl_order is not None:
        cmd.extend(["--sc-gl-order", str(sc_gl_order)])
    if init_run:
        cmd.extend(["--init-run", str(init_run)])
    if sc_eq:
        cmd.append("--sc-eq")
    else:
        cmd.append("--no-sc-eq")
        if p_fix is not None:
            cmd.extend(["--p-fix", str(p_fix)])
    if verbose:
        cmd.append("--verbose")
    return cmd


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", tag.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _random_suffix() -> str:
    return secrets.token_hex(2)


def build_generate_out_dir(
    runs_root: Path,
    *,
    tag: str,
    N: int,
    R: int,
    K: int,
    timestamp: datetime | None = None,
    suffix_provider: Callable[[], str] | None = None,
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{_sanitize_tag(tag)}_N{N}_R{R}_K{K}"
    candidate = runs_root / base
    if candidate.exists():
        provider = suffix_provider or _random_suffix
        while candidate.exists():
            candidate = runs_root / f"{base}_{provider()}"
    return candidate


def build_generate_command(
    out_dir: str | Path,
    *,
    N: int,
    R: int,
    K: int,
    Lz: float,
    diameter_mm: float,
    ring_offset_step_mm: float,
    name: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "halbach.cli.generate_run",
        "--out",
        str(out_dir),
        "--N",
        str(N),
        "--R",
        str(R),
        "--K",
        str(K),
        "--Lz",
        str(Lz),
        "--diameter-mm",
        str(diameter_mm),
        "--ring-offset-step-mm",
        str(ring_offset_step_mm),
    ]
    if name:
        cmd.extend(["--name", name])
    return cmd


def run_generate_command(
    cmd: list[str], *, cwd: Path, log_path: Path | None = None
) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if stdout and stderr and not stdout.endswith("\n"):
        stdout += "\n"
    combined = stdout + stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")
    return result.returncode, combined


def start_opt_job(
    in_path: str | Path,
    out_dir: str | Path,
    *,
    maxiter: int,
    gtol: float,
    roi_r: float,
    roi_step: float,
    roi_mode: str = "surface-fibonacci",
    roi_samples: int = 300,
    angle_model: str = "legacy-alpha",
    grad_backend: str | None = None,
    fourier_H: int = 4,
    lambda0: float = 0.0,
    lambda_theta: float = 0.0,
    lambda_z: float = 0.0,
    angle_init: str = "from-run",
    r_bound_mode: str = "relative",
    r_lower_delta_mm: float = 30.0,
    r_upper_delta_mm: float = 30.0,
    r_no_upper: bool = False,
    r_min_mm: float = 0.0,
    r_max_mm: float = 1e9,
    fix_center_radius_layers: int = 2,
    mag_model: str = "fixed",
    sc_chi: float = 0.0,
    sc_Nd: float = 1.0 / 3.0,
    sc_p0: float = 1.0,
    sc_volume_mm3: float = 1000.0,
    sc_iters: int = 30,
    sc_omega: float = 0.6,
    sc_near_wr: int = 0,
    sc_near_wz: int = 1,
    sc_near_wphi: int = 2,
    sc_near_kernel: str = "dipole",
    sc_subdip_n: int = 2,
    sc_gl_order: int | None = None,
    repo_root: Path | None = None,
) -> OptJob:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "opt.log"
    cmd = build_command(
        in_path,
        out_path,
        maxiter=maxiter,
        gtol=gtol,
        roi_r=roi_r,
        roi_step=roi_step,
        roi_mode=roi_mode,
        roi_samples=roi_samples,
        angle_model=angle_model,
        grad_backend=grad_backend,
        fourier_H=fourier_H,
        lambda0=lambda0,
        lambda_theta=lambda_theta,
        lambda_z=lambda_z,
        angle_init=angle_init,
        r_bound_mode=r_bound_mode,
        r_lower_delta_mm=r_lower_delta_mm,
        r_upper_delta_mm=r_upper_delta_mm,
        r_no_upper=r_no_upper,
        r_min_mm=r_min_mm,
        r_max_mm=r_max_mm,
        fix_center_radius_layers=fix_center_radius_layers,
        mag_model=mag_model,
        sc_chi=sc_chi,
        sc_Nd=sc_Nd,
        sc_p0=sc_p0,
        sc_volume_mm3=sc_volume_mm3,
        sc_iters=sc_iters,
        sc_omega=sc_omega,
        sc_near_wr=sc_near_wr,
        sc_near_wz=sc_near_wz,
        sc_near_wphi=sc_near_wphi,
        sc_near_kernel=sc_near_kernel,
        sc_subdip_n=sc_subdip_n,
        sc_gl_order=sc_gl_order,
    )
    cwd = repo_root or Path(__file__).resolve().parents[2]
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )
    return OptJob(proc=proc, out_dir=out_path, log_path=log_path, command=cmd)


def start_dc_ccp_sc_job(
    out_dir: str | Path,
    *,
    N: int,
    K: int,
    R: int,
    radius_m: float,
    length_m: float,
    roi_radius_m: float,
    roi_grid_n: int,
    wx: float,
    wy: float,
    wz: float,
    factor: float,
    phi0: float,
    delta_nom_deg: float,
    delta_step_deg: float | None,
    tau0: float,
    tau_mult: float,
    tau_max: float,
    iters: int,
    tol: float,
    tol_f: float,
    reg_x: float,
    reg_p: float,
    reg_z: float,
    sc_eq: bool,
    p_fix: float | None,
    sc_chi: float,
    sc_Nd: float,
    sc_p0: float,
    sc_volume_mm3: float,
    sc_near_wr: int,
    sc_near_wz: int,
    sc_near_wphi: int,
    sc_near_kernel: str,
    sc_subdip_n: int,
    sc_gl_order: int | None = None,
    pmin: float,
    pmax: float,
    solver: str,
    init_run: str | Path | None = None,
    verbose: bool = False,
    repo_root: Path | None = None,
) -> OptJob:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "dc_ccp_sc.log"
    cmd = build_dc_ccp_sc_command(
        out_path,
        N=N,
        K=K,
        R=R,
        radius_m=radius_m,
        length_m=length_m,
        roi_radius_m=roi_radius_m,
        roi_grid_n=roi_grid_n,
        wx=wx,
        wy=wy,
        wz=wz,
        factor=factor,
        phi0=phi0,
        delta_nom_deg=delta_nom_deg,
        delta_step_deg=delta_step_deg,
        tau0=tau0,
        tau_mult=tau_mult,
        tau_max=tau_max,
        iters=iters,
        tol=tol,
        tol_f=tol_f,
        reg_x=reg_x,
        reg_p=reg_p,
        reg_z=reg_z,
        sc_eq=sc_eq,
        p_fix=p_fix,
        sc_chi=sc_chi,
        sc_Nd=sc_Nd,
        sc_p0=sc_p0,
        sc_volume_mm3=sc_volume_mm3,
        sc_near_wr=sc_near_wr,
        sc_near_wz=sc_near_wz,
        sc_near_wphi=sc_near_wphi,
        sc_near_kernel=sc_near_kernel,
        sc_subdip_n=sc_subdip_n,
        sc_gl_order=sc_gl_order,
        pmin=pmin,
        pmax=pmax,
        solver=solver,
        init_run=init_run,
        verbose=verbose,
    )
    cwd = repo_root or Path(__file__).resolve().parents[2]
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )
    return OptJob(proc=proc, out_dir=out_path, log_path=log_path, command=cmd)


def poll_opt_job(job: OptJob) -> int | None:
    return job.proc.poll()


def stop_opt_job(job: OptJob) -> None:
    if job.proc.poll() is None:
        job.proc.terminate()


def tail_log(path: str | Path, n_lines: int = 200) -> str:
    log_path = Path(path)
    if not log_path.is_file():
        return ""
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = deque(handle, maxlen=n_lines)
    return "".join(lines)


__all__ = [
    "OptJob",
    "build_command",
    "build_dc_ccp_sc_command",
    "build_generate_command",
    "build_generate_out_dir",
    "run_generate_command",
    "start_opt_job",
    "start_dc_ccp_sc_job",
    "poll_opt_job",
    "stop_opt_job",
    "tail_log",
]
