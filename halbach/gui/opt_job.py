from __future__ import annotations

import subprocess
import sys
from collections import deque
from dataclasses import dataclass
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
    rho_gn: float,
    sigma_alpha_deg: float | None = None,
    sigma_r_mm: float | None = None,
    run_mc: bool = False,
    mc_samples: int | None = None,
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
        "--rho-gn",
        str(rho_gn),
    ]
    if sigma_alpha_deg is not None:
        cmd.extend(["--sigma-alpha-deg", str(sigma_alpha_deg)])
    if sigma_r_mm is not None:
        cmd.extend(["--sigma-r-mm", str(sigma_r_mm)])
    if mc_samples is not None:
        cmd.extend(["--mc-samples", str(mc_samples)])
    if run_mc:
        cmd.append("--run-mc")
    return cmd


def start_opt_job(
    in_path: str | Path,
    out_dir: str | Path,
    *,
    maxiter: int,
    gtol: float,
    roi_r: float,
    roi_step: float,
    rho_gn: float,
    sigma_alpha_deg: float | None = None,
    sigma_r_mm: float | None = None,
    run_mc: bool = False,
    mc_samples: int | None = None,
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
        rho_gn=rho_gn,
        sigma_alpha_deg=sigma_alpha_deg,
        sigma_r_mm=sigma_r_mm,
        run_mc=run_mc,
        mc_samples=mc_samples,
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


__all__ = ["OptJob", "build_command", "start_opt_job", "poll_opt_job", "stop_opt_job", "tail_log"]
