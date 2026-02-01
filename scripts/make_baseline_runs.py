from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
DOCS_DIR = REPO_ROOT / "docs"


@dataclass(frozen=True)
class BaselineCase:
    name: str
    angle_model: str
    grad_backend: str | None
    requires_jax: bool
    maxiter: int
    gtol: float
    roi_r: float
    roi_mode: str
    roi_samples: int
    roi_seed: int
    field_scale: float
    fix_center_radius_layers: int
    r_bound_mode: str
    r_lower_delta_mm: float
    r_upper_delta_mm: float


def _has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  exit: {proc.returncode}\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}"
        )


def _remove_dir(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(
        func: Callable[..., Any],
        p: str,
        _exc: tuple[type[BaseException], BaseException, TracebackType],
    ) -> None:
        try:
            os.chmod(p, 0o700)
        except OSError:
            pass
        func(p)

    shutil.rmtree(path, onerror=_onerror)


def _generate_run(run_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "halbach.cli.generate_run",
        "--out",
        str(run_dir),
        "--N",
        "12",
        "--R",
        "1",
        "--K",
        "6",
        "--Lz",
        "0.20",
        "--diameter-mm",
        "200",
        "--ring-offset-step-mm",
        "0",
    ]
    _run_cmd(cmd)


def _optimize_run(case: BaselineCase, in_dir: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "halbach.cli.optimize_run",
        "--in",
        str(in_dir),
        "--out",
        str(out_dir),
        "--maxiter",
        str(case.maxiter),
        "--gtol",
        f"{case.gtol:.1e}",
        "--roi-r",
        str(case.roi_r),
        "--roi-mode",
        case.roi_mode,
        "--roi-samples",
        str(case.roi_samples),
        "--roi-seed",
        str(case.roi_seed),
        "--field-scale",
        f"{case.field_scale:.0e}",
        "--angle-model",
        case.angle_model,
        "--fix-center-radius-layers",
        str(case.fix_center_radius_layers),
        "--r-bound-mode",
        case.r_bound_mode,
        "--r-lower-delta-mm",
        str(case.r_lower_delta_mm),
        "--r-upper-delta-mm",
        str(case.r_upper_delta_mm),
    ]
    if case.grad_backend is not None:
        cmd.extend(["--grad-backend", case.grad_backend])
    _run_cmd(cmd)
    return cmd


def _load_trace_metrics(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    extras = trace.get("extras", [])
    iters = trace.get("iters", [])
    if not extras or not iters:
        raise ValueError("trace.json missing extras/iters")
    last_extra = extras[-1]
    return {
        "J": float(last_extra.get("J")),
        "B0": float(last_extra.get("B0")),
        "iters": int(iters[-1]),
    }


def _render_markdown(
    cases: list[BaselineCase],
    summary: dict[str, Any],
    commands: dict[str, dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("# Baseline runs")
    lines.append("")
    lines.append("回帰比較用のベースライン run を再現可能に生成するための手順です。")
    lines.append("")
    lines.append("## 実行方法")
    lines.append("```powershell")
    lines.append("python scripts/make_baseline_runs.py --force")
    lines.append("```")
    lines.append("")
    lines.append("## ケースとコマンド")
    for case in cases:
        lines.append(f"### {case.name}")
        lines.append("")
        gen_cmd = commands[case.name]["generate"]
        opt_cmd = commands[case.name]["optimize"]
        lines.append("Generate:")
        lines.append("```powershell")
        lines.append(gen_cmd)
        lines.append("```")
        lines.append("Optimize:")
        lines.append("```powershell")
        lines.append(opt_cmd)
        lines.append("```")
        lines.append("")

    lines.append("## 取得メトリクス")
    lines.append("- J（trace.extras[-1].J）")
    lines.append("- |B0| [T]（trace.extras[-1].B0）")
    lines.append("- iters（trace.iters[-1]）")
    lines.append("- roi_seed（固定）")
    lines.append("")

    lines.append("## サマリ")
    lines.append("")
    lines.append(
        "| case | angle_model | grad_backend | status | J | |B0| [T] | iters | roi_seed | note |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in summary["cases"]:
        metrics = item.get("metrics", {})
        note = item.get("reason", "")
        lines.append(
            "| {case} | {angle} | {grad} | {status} | {J} | {B0} | {iters} | {seed} | {note} |".format(
                case=item["name"],
                angle=item["angle_model"],
                grad=item.get("grad_backend") or "-",
                status=item["status"],
                J=_fmt_num(metrics.get("J")),
                B0=_fmt_num(metrics.get("B0")),
                iters=metrics.get("iters", "-"),
                seed=item.get("roi_seed", "-"),
                note=note or "-",
            )
        )
    lines.append("")
    lines.append("## baseline_runs.json")
    lines.append("`docs/baseline_runs.json` はスクリプトが生成する機械比較用の出力です。")
    lines.append("主な構造:")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "generated_at": "ISO-8601 timestamp",
                "cases": [
                    {
                        "name": "baseline_tiny_legacy_analytic",
                        "angle_model": "legacy-alpha",
                        "grad_backend": "analytic",
                        "status": "ok|skipped|error",
                        "metrics": {"J": 0.0, "B0": 0.0, "iters": 0},
                        "roi_seed": 20250924,
                        "input_dir": "runs/...",
                        "opt_dir": "runs/..._opt",
                        "reason": "optional",
                    }
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines.append("```")
    return "\n".join(lines)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6e}"
    except Exception:
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline runs for regression testing.")
    parser.add_argument(
        "--force", action="store_true", help="delete existing runs before re-create"
    )
    args = parser.parse_args()

    cases = [
        BaselineCase(
            name="baseline_tiny_legacy_analytic",
            angle_model="legacy-alpha",
            grad_backend="analytic",
            requires_jax=False,
            maxiter=3,
            gtol=1e-12,
            roi_r=0.05,
            roi_mode="surface-fibonacci",
            roi_samples=50,
            roi_seed=20250924,
            field_scale=1e6,
            fix_center_radius_layers=2,
            r_bound_mode="relative",
            r_lower_delta_mm=5.0,
            r_upper_delta_mm=5.0,
        ),
        BaselineCase(
            name="baseline_tiny_legacy_jax",
            angle_model="legacy-alpha",
            grad_backend="jax",
            requires_jax=True,
            maxiter=3,
            gtol=1e-12,
            roi_r=0.05,
            roi_mode="surface-fibonacci",
            roi_samples=50,
            roi_seed=20250924,
            field_scale=1e6,
            fix_center_radius_layers=2,
            r_bound_mode="relative",
            r_lower_delta_mm=5.0,
            r_upper_delta_mm=5.0,
        ),
        BaselineCase(
            name="baseline_tiny_delta_rep",
            angle_model="delta-rep-x0",
            grad_backend=None,
            requires_jax=True,
            maxiter=3,
            gtol=1e-12,
            roi_r=0.05,
            roi_mode="surface-fibonacci",
            roi_samples=50,
            roi_seed=20250924,
            field_scale=1e6,
            fix_center_radius_layers=2,
            r_bound_mode="relative",
            r_lower_delta_mm=5.0,
            r_upper_delta_mm=5.0,
        ),
    ]

    RUNS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    jax_available = _has_jax()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": [],
    }
    commands: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for case in cases:
        input_dir = RUNS_DIR / case.name
        opt_dir = RUNS_DIR / f"{case.name}_opt"

        if args.force:
            _remove_dir(input_dir)
            _remove_dir(opt_dir)

        case_record: dict[str, Any] = {
            "name": case.name,
            "angle_model": case.angle_model,
            "grad_backend": case.grad_backend,
            "roi_seed": case.roi_seed,
            "input_dir": _rel_path(input_dir),
            "opt_dir": _rel_path(opt_dir),
            "status": "pending",
        }

        if case.requires_jax and not jax_available:
            case_record["status"] = "skipped"
            case_record["reason"] = "JAX not installed"
            summary["cases"].append(case_record)
            commands[case.name] = {
                "generate": _build_generate_cmd_str(input_dir),
                "optimize": _build_optimize_cmd_str(case, input_dir, opt_dir),
            }
            continue

        try:
            _generate_run(input_dir)
            _optimize_run(case, input_dir, opt_dir)
            trace_path = opt_dir / "trace.json"
            metrics = _load_trace_metrics(trace_path)
            case_record["metrics"] = metrics
            case_record["status"] = "ok"
            commands[case.name] = {
                "generate": _build_generate_cmd_str(input_dir),
                "optimize": _build_optimize_cmd_str(case, input_dir, opt_dir),
            }
        except Exception as exc:  # noqa: BLE001
            case_record["status"] = "error"
            case_record["reason"] = str(exc)
            errors.append(f"{case.name}: {exc}")
            commands[case.name] = {
                "generate": _build_generate_cmd_str(input_dir),
                "optimize": _build_optimize_cmd_str(case, input_dir, opt_dir),
            }
        summary["cases"].append(case_record)

    (DOCS_DIR / "baseline_runs.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_text = _render_markdown(cases, summary, commands)
    (DOCS_DIR / "baseline_runs.md").write_text(md_text, encoding="utf-8")

    if errors:
        print("Some cases failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    return 0


def _build_generate_cmd_str(run_dir: Path) -> str:
    rel_dir = _rel_path(run_dir)
    return " ".join(
        [
            "python",
            "-m",
            "halbach.cli.generate_run",
            "--out",
            rel_dir,
            "--N",
            "12",
            "--R",
            "1",
            "--K",
            "6",
            "--Lz",
            "0.20",
            "--diameter-mm",
            "200",
            "--ring-offset-step-mm",
            "0",
        ]
    )


def _build_optimize_cmd_str(case: BaselineCase, in_dir: Path, out_dir: Path) -> str:
    rel_in = _rel_path(in_dir)
    rel_out = _rel_path(out_dir)
    cmd = [
        "python",
        "-u",
        "-m",
        "halbach.cli.optimize_run",
        "--in",
        rel_in,
        "--out",
        rel_out,
        "--maxiter",
        str(case.maxiter),
        "--gtol",
        f"{case.gtol:.1e}",
        "--roi-r",
        str(case.roi_r),
        "--roi-mode",
        case.roi_mode,
        "--roi-samples",
        str(case.roi_samples),
        "--roi-seed",
        str(case.roi_seed),
        "--field-scale",
        f"{case.field_scale:.0e}",
        "--angle-model",
        case.angle_model,
        "--fix-center-radius-layers",
        str(case.fix_center_radius_layers),
        "--r-bound-mode",
        case.r_bound_mode,
        "--r-lower-delta-mm",
        str(case.r_lower_delta_mm),
        "--r-upper-delta-mm",
        str(case.r_upper_delta_mm),
    ]
    if case.grad_backend is not None:
        cmd.extend(["--grad-backend", case.grad_backend])
    return " ".join(cmd)


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
