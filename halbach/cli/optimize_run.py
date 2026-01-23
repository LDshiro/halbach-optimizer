from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, FIELD_SCALE_DEFAULT, m0, phi0
from halbach.geom import (
    RoiMode,
    build_r_bases_from_vars,
    build_roi_points,
    build_symmetry_indices,
    pack_x,
    unpack_x,
)
from halbach.physics import objective_only
from halbach.robust import Float1DArray, fun_grad_gradnorm_fixed
from halbach.run_io import load_run
from halbach.solvers.lbfgsb import bounds_from_arrays, solve_lbfgsb
from halbach.solvers.types import LBFGSBOptions, SolveResult

logger = logging.getLogger(__name__)

ROI_WARN_THRESHOLD = 200_000
ROI_DOWNSAMPLE_SEED = 0


class SummaryStats(TypedDict):
    name: str
    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize a Halbach run with L-BFGS-B")
    ap.add_argument("--in", dest="in_path", required=True, help="input run dir or results .npz")
    ap.add_argument("--out", dest="out_dir", required=True, help="output run directory")
    ap.add_argument("--maxiter", type=int, default=900, help="L-BFGS-B maxiter")
    ap.add_argument("--gtol", type=float, default=1e-12, help="L-BFGS-B gtol")
    ap.add_argument("--log-every", type=int, default=10, help="log every N iterations")
    ap.add_argument(
        "--log-precision",
        type=int,
        default=3,
        help="numeric precision for iteration logs",
    )
    ap.add_argument("--roi-r", type=float, default=0.14, help="ROI radius [m]")
    ap.add_argument("--roi-step", type=float, default=0.02, help="ROI grid step [m]")
    ap.add_argument(
        "--roi-mode",
        type=str,
        default="surface-fibonacci",
        choices=[
            "volume-grid",
            "volume-subsample",
            "surface-fibonacci",
            "surface-random",
        ],
        help="ROI sampling mode",
    )
    ap.add_argument("--roi-samples", type=int, default=300, help="ROI sample count")
    ap.add_argument("--roi-seed", type=int, default=20250924, help="ROI sampling seed")
    ap.add_argument("--roi-half-x", action="store_true", help="Reflect ROI points to x>=0")
    ap.add_argument(
        "--roi-max-points",
        type=int,
        default=0,
        help="downsample ROI points to at most M (0 disables)",
    )
    ap.add_argument("--rho-gn", type=float, default=1e-4, help="GN weight in y-space")
    ap.add_argument(
        "--field-scale",
        type=float,
        default=FIELD_SCALE_DEFAULT,
        help="field scaling factor for objective stability",
    )
    ap.add_argument("--sigma-alpha-deg", type=float, default=0.5, help="1-sigma alpha [deg]")
    ap.add_argument("--sigma-r-mm", type=float, default=0.2, help="1-sigma r [mm]")
    ap.add_argument("--eps-hvp", type=float, default=1e-6, help="HVP step base")
    ap.add_argument(
        "--min-radius-drop-mm",
        type=float,
        default=20.0,
        help="lower bound: r_vars >= r_init - this [mm]",
    )
    ap.add_argument("--mc-samples", type=int, default=600, help="MC samples")
    ap.add_argument("--run-mc", action="store_true", help="run Monte Carlo evaluation")
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="logging level",
    )
    ap.add_argument(
        "--debug-stacks-secs",
        type=int,
        default=0,
        help="dump stack traces every N seconds (0 disables)",
    )
    ap.add_argument("--dry-run", action="store_true", help="evaluate once and exit")
    return ap.parse_args(argv)


def _opt_log_path(out_dir: Path) -> Path:
    return out_dir / "opt.log"


def _configure_logging(level: str, out_dir: Path) -> logging.FileHandler:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _opt_log_path(out_dir)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, stream_handler],
        force=True,
    )
    return file_handler


def _git_hash(repo_root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return res.stdout.strip() or None


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _jsonify_extras(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _jsonify_value(v) for k, v in item.items()} for item in items]


def _write_trace(out_dir: Path, res: SolveResult) -> None:
    trace_path = out_dir / "trace.json"
    payload = {
        "iters": res.trace.iters,
        "f": res.trace.f,
        "gnorm": res.trace.gnorm,
        "extras": _jsonify_extras(res.trace.extras),
    }
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _summarize(arr: NDArray[np.float64], name: str) -> SummaryStats:
    q = np.quantile(arr, [0.50, 0.90, 0.95, 0.99])
    return dict(
        name=name,
        mean=float(arr.mean()),
        std=float(arr.std()),
        median=float(q[0]),
        p90=float(q[1]),
        p95=float(q[2]),
        p99=float(q[3]),
    )


def _format_iter_log(
    k: int,
    J: float,
    gnorm: float,
    B0_mT: float,
    dt_eval: float,
    *,
    precision: int,
    iter_width: int,
) -> str:
    prec = max(0, precision)
    return (
        f"[iter {k:0{iter_width}d}] "
        f"J={J:.{prec}e} "
        f"gnorm={gnorm:.{prec}e} "
        f"|B0|={B0_mT:.{prec}f} mT "
        f"dt_eval={dt_eval:.{prec}f}s"
    )


@dataclass(frozen=True)
class RoiSamplingResult:
    pts: NDArray[np.float64]
    npts_before: int
    downsampled: bool


def _roi_point_warning(npts: int, threshold: int) -> bool:
    return npts > threshold


def _downsample_roi_points(
    pts: NDArray[np.float64], max_points: int, seed: int
) -> RoiSamplingResult:
    npts = int(pts.shape[0])
    if max_points <= 0 or npts <= max_points:
        return RoiSamplingResult(pts=pts, npts_before=npts, downsampled=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(npts, size=max_points, replace=False)
    return RoiSamplingResult(pts=pts[idx], npts_before=npts, downsampled=True)


def _enable_stack_dumps(interval: int, file_handler: logging.FileHandler) -> None:
    if interval <= 0:
        return
    faulthandler.dump_traceback_later(
        interval,
        repeat=True,
        file=file_handler.stream,
    )
    logger.warning("Enabled stack dumps every %d seconds", interval)


def _mc_eval(
    alphas_base: NDArray[np.float64],
    r_bases_base: NDArray[np.float64],
    pts: NDArray[np.float64],
    factor: float,
    sigma_alpha: float,
    sigma_r: float,
    samples: int,
    seed: int,
    geom_theta: NDArray[np.float64],
    geom_sin2: NDArray[np.float64],
    geom_cth: NDArray[np.float64],
    geom_sth: NDArray[np.float64],
    geom_z_layers: NDArray[np.float64],
    geom_ring_offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    out = np.zeros(samples, dtype=np.float64)
    for s in range(samples):
        dA = rng.standard_normal(size=alphas_base.shape) * sigma_alpha
        dR = rng.standard_normal(size=r_bases_base.shape) * sigma_r
        out[s] = objective_only(
            alphas_base + dA,
            r_bases_base + dR,
            geom_theta,
            geom_sin2,
            geom_cth,
            geom_sth,
            geom_z_layers,
            geom_ring_offsets,
            pts,
            factor,
            phi0,
            m0,
        )
    return out


def run_optimize(args: argparse.Namespace) -> int:
    start_time = datetime.now(UTC)
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("load_run start: %s", in_path)
    t_load = perf_counter()
    run = load_run(in_path)
    logger.info("load_run done in %.3fs", perf_counter() - t_load)
    geom = run.geometry
    r0 = float(np.median(run.results.r_bases))
    lower_var, upper_var, _ = build_symmetry_indices(geom.K)

    roi_mode = cast(RoiMode, args.roi_mode)
    logger.info(
        "ROI build start (mode=%s, roi_r=%.4f, roi_step=%.4f, samples=%d, seed=%d, half_x=%s)",
        roi_mode,
        args.roi_r,
        args.roi_step,
        args.roi_samples,
        args.roi_seed,
        args.roi_half_x,
    )
    if roi_mode.startswith("surface"):
        logger.info("ROI mode %s ignores roi_step", roi_mode)
    t_roi = perf_counter()
    pts = build_roi_points(
        args.roi_r,
        args.roi_step,
        mode=roi_mode,
        n_samples=args.roi_samples,
        seed=args.roi_seed,
        half_x=args.roi_half_x,
    )
    roi_elapsed = perf_counter() - t_roi
    npts = int(pts.shape[0])
    logger.info(
        "ROI build done in %.3fs (npts=%d, mode=%s)",
        roi_elapsed,
        npts,
        roi_mode,
    )
    if _roi_point_warning(npts, ROI_WARN_THRESHOLD):
        logger.warning(
            "ROI points=%d exceed threshold (%d). Consider increasing roi_step.",
            npts,
            ROI_WARN_THRESHOLD,
        )
    downsampled = _downsample_roi_points(pts, args.roi_max_points, ROI_DOWNSAMPLE_SEED)
    if downsampled.downsampled:
        logger.warning(
            "Downsampled ROI points from %d to %d (seed=%d).",
            downsampled.npts_before,
            downsampled.pts.shape[0],
            ROI_DOWNSAMPLE_SEED,
        )
    pts = downsampled.pts

    rv0 = np.array([run.results.r_bases[k] for k in lower_var], dtype=np.float64)
    x0 = cast(Float1DArray, pack_x(run.results.alphas, rv0))

    delta_m = args.min_radius_drop_mm * 1e-3
    lb = np.full(x0.size, -np.inf, dtype=np.float64)
    ub = np.full(x0.size, np.inf, dtype=np.float64)
    P = geom.R * geom.K
    lb[P:] = rv0 - delta_m
    bounds = bounds_from_arrays(lb, ub)

    sigma_alpha = np.deg2rad(args.sigma_alpha_deg)
    sigma_r = args.sigma_r_mm * 1e-3
    field_scale = float(args.field_scale)
    if field_scale <= 0.0:
        raise ValueError("field_scale must be positive")
    factor = FACTOR * field_scale

    first_eval_done = False
    state: dict[str, float] = {}

    def fun_grad_solver(x: Float1DArray) -> tuple[float, Float1DArray, dict[str, Any]]:
        nonlocal first_eval_done
        if not first_eval_done:
            logger.info("first eval start")
        t_eval = perf_counter()
        Jgn, gx, B0, Jn, gn2 = fun_grad_gradnorm_fixed(
            x,
            geom,
            pts,
            sigma_alpha,
            sigma_r,
            args.rho_gn,
            args.eps_hvp,
            r0,
            lower_var,
            upper_var,
            factor,
        )
        dt_eval = perf_counter() - t_eval
        if not first_eval_done:
            logger.info("first eval done in %.3fs", dt_eval)
            first_eval_done = True
        state["J"] = float(Jn)
        state["gnorm"] = float(np.linalg.norm(gx))
        B0_T = float(B0) / field_scale
        state["B0"] = B0_T
        state["t_last_eval"] = float(dt_eval)
        extras: dict[str, Any] = {"J": float(Jn), "B0": B0_T, "gn2": float(gn2)}
        return float(Jgn), gx, extras

    log_every = max(1, int(args.log_every))
    if log_every != args.log_every:
        logger.warning("log-every adjusted to %d", log_every)
    log_precision = max(0, int(args.log_precision))
    if log_precision != args.log_precision:
        logger.warning("log-precision adjusted to %d", log_precision)
    iter_width = max(4, len(str(args.maxiter)))

    def iter_cb(
        k: int,
        xk: Float1DArray,
        fk: float,
        gk: Float1DArray,
        extras: dict[str, Any],
    ) -> None:
        if k % log_every == 0:
            J_val = float(state.get("J", fk))
            gnorm_val = state.get("gnorm")
            if gnorm_val is None:
                gnorm_val = float(np.linalg.norm(gk))
            B0_val = float(state.get("B0", np.nan))
            dt_eval = float(state.get("t_last_eval", np.nan))
            line = _format_iter_log(
                k,
                J_val,
                gnorm_val,
                B0_val * 1e3,
                dt_eval,
                precision=log_precision,
                iter_width=iter_width,
            )
            logger.info(line)

    if args.dry_run:
        Jgn, g0, extras_eval = fun_grad_solver(x0)
        gnorm = float(np.linalg.norm(g0))
        logger.info(
            "dry-run: Jgn=%.6e J=%.6e |B0|=%.3f mT gnorm=%.3e",
            Jgn,
            float(extras_eval.get("J", np.nan)),
            float(extras_eval.get("B0", np.nan)) * 1e3,
            gnorm,
        )
        meta_dry: dict[str, Any] = dict(
            input_run=str(in_path),
            out_dir=str(out_dir),
            start_time=start_time.isoformat(),
            end_time=datetime.now(UTC).isoformat(),
            git_hash=_git_hash(Path(__file__).resolve().parents[2]),
            roi=dict(
                roi_r=float(args.roi_r),
                roi_step=float(args.roi_step),
                roi_mode=roi_mode,
                roi_samples=int(args.roi_samples),
                roi_seed=int(args.roi_seed),
                roi_half_x=bool(args.roi_half_x),
            ),
            robust=dict(
                rho_gn=float(args.rho_gn),
                eps_hvp=float(args.eps_hvp),
                sigma_alpha_deg=float(args.sigma_alpha_deg),
                sigma_r_mm=float(args.sigma_r_mm),
            ),
            scaling=dict(
                field_scale=float(field_scale),
            ),
            optimizer=dict(
                method="L-BFGS-B",
                maxiter=int(args.maxiter),
                gtol=float(args.gtol),
                min_radius_drop_mm=float(args.min_radius_drop_mm),
            ),
            dry_run=True,
        )
        meta_path = out_dir / "meta.json"
        logger.info("save start (dry-run meta only)")
        t_save = perf_counter()
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_dry, handle, indent=2)
        logger.info("save done in %.3fs", perf_counter() - t_save)
        return 0

    logger.info("solver start (maxiter=%d, gtol=%.3e)", args.maxiter, args.gtol)
    opt = LBFGSBOptions(maxiter=args.maxiter, gtol=args.gtol, disp=True)
    res = solve_lbfgsb(fun_grad_solver, x0, bounds, opt, iter_callback=iter_cb)
    logger.info(
        "solver done: success=%s message=%s nit=%d nfev=%d njev=%s",
        res.success,
        res.message,
        res.nit,
        res.nfev,
        res.njev,
    )

    al_opt, rv_opt = unpack_x(res.x, geom.R, geom.K)
    rb_opt = build_r_bases_from_vars(rv_opt, geom.K, r0, lower_var, upper_var)

    Jgn_f, _, B0_f, Jn_f, gn2_f = fun_grad_gradnorm_fixed(
        res.x,
        geom,
        pts,
        sigma_alpha,
        sigma_r,
        args.rho_gn,
        args.eps_hvp,
        r0,
        lower_var,
        upper_var,
        factor,
    )
    B0_f_T = B0_f / field_scale
    logger.info(
        "[done] success=%s iters=%d Jgn=%.6e J=%.6e gn2=%.3e |B0|=%.3f mT",
        res.success,
        res.nit,
        Jgn_f,
        Jn_f,
        gn2_f,
        B0_f_T * 1e3,
    )

    extras = res.trace.extras
    J_hist = np.array(res.trace.f, dtype=float)
    Jn_hist = np.array([float(e.get("J", np.nan)) for e in extras], dtype=float)
    B0_hist = np.array([float(e.get("B0", np.nan)) for e in extras], dtype=float)
    gn2_hist = np.array([float(e.get("gn2", np.nan)) for e in extras], dtype=float)

    logger.info("save start")
    t_save = perf_counter()
    results_path = out_dir / "results.npz"
    np.savez_compressed(
        results_path,
        alphas_opt=al_opt,
        r_bases_opt=rb_opt,
        theta=geom.theta,
        sin2th=geom.sin2,
        cth=geom.cth,
        sth=geom.sth,
        z_layers=geom.z_layers,
        ring_offsets=geom.ring_offsets,
        J_hist=J_hist,
        Jn_hist=Jn_hist,
        B0_hist=B0_hist,
        gn2_hist=gn2_hist,
    )

    meta: dict[str, Any] = dict(
        input_run=str(in_path),
        out_dir=str(out_dir),
        start_time=start_time.isoformat(),
        end_time=datetime.now(UTC).isoformat(),
        git_hash=_git_hash(Path(__file__).resolve().parents[2]),
        roi=dict(
            roi_r=float(args.roi_r),
            roi_step=float(args.roi_step),
            roi_mode=roi_mode,
            roi_samples=int(args.roi_samples),
            roi_seed=int(args.roi_seed),
            roi_half_x=bool(args.roi_half_x),
        ),
        robust=dict(
            rho_gn=float(args.rho_gn),
            eps_hvp=float(args.eps_hvp),
            sigma_alpha_deg=float(args.sigma_alpha_deg),
            sigma_r_mm=float(args.sigma_r_mm),
        ),
        scaling=dict(
            field_scale=float(field_scale),
        ),
        optimizer=dict(
            method="L-BFGS-B",
            maxiter=int(args.maxiter),
            gtol=float(args.gtol),
            min_radius_drop_mm=float(args.min_radius_drop_mm),
        ),
        scipy_result=dict(
            success=bool(res.success),
            status=0 if res.success else 1,
            message=str(res.message),
            nit=int(res.nit),
            nfev=int(res.nfev),
            njev=-1 if res.njev is None else int(res.njev),
        ),
    )
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    _write_trace(out_dir, res)
    logger.info("save done in %.3fs", perf_counter() - t_save)

    if args.run_mc:
        sigma_alpha = np.deg2rad(args.sigma_alpha_deg)
        sigma_r = args.sigma_r_mm * 1e-3
        J_nom_mc = _mc_eval(
            run.results.alphas,
            run.results.r_bases,
            pts,
            factor,
            sigma_alpha,
            sigma_r,
            args.mc_samples,
            0,
            geom.theta,
            geom.sin2,
            geom.cth,
            geom.sth,
            geom.z_layers,
            geom.ring_offsets,
        )
        J_opt_mc = _mc_eval(
            al_opt,
            rb_opt,
            pts,
            factor,
            sigma_alpha,
            sigma_r,
            args.mc_samples,
            1,
            geom.theta,
            geom.sin2,
            geom.cth,
            geom.sth,
            geom.z_layers,
            geom.ring_offsets,
        )
        summary = dict(
            settings=dict(
                roi_r=float(args.roi_r),
                roi_step=float(args.roi_step),
                sigma_alpha_deg=float(args.sigma_alpha_deg),
                sigma_r_mm=float(args.sigma_r_mm),
                rho_gn=float(args.rho_gn),
                eps_hvp=float(args.eps_hvp),
                maxiter=int(args.maxiter),
                gtol=float(args.gtol),
                mc_samples=int(args.mc_samples),
            ),
            nominal=_summarize(J_nom_mc, "nominal"),
            optimized=_summarize(J_opt_mc, "optimized"),
        )
        mc_path = out_dir / "mc_summary.json"
        with mc_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return 0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    file_handler = _configure_logging(args.log_level, out_dir)
    _enable_stack_dumps(args.debug_stacks_secs, file_handler)
    try:
        code = run_optimize(args)
    except Exception as exc:
        logger.exception("Optimization failed: %s", exc)
        sys.exit(1)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
