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
    "build_generate_command",
    "build_generate_out_dir",
    "run_generate_command",
    "start_opt_job",
    "poll_opt_job",
    "stop_opt_job",
    "tail_log",
]
