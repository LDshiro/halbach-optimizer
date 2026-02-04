from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, FIELD_SCALE_DEFAULT, m0, phi0
from halbach.geom import (
    ParamMap,
    RoiMode,
    build_param_map,
    build_roi_points,
    pack_grad,
    pack_x,
    unpack_x,
)
from halbach.magnetization_runtime import compute_p_flat_self_consistent_jax, sc_cfg_fingerprint
from halbach.objective import objective_with_grads_fixed
from halbach.run_io import load_run
from halbach.sc_debug import make_sc_debug_bundle
from halbach.solvers.lbfgsb import solve_lbfgsb
from halbach.solvers.types import LBFGSBOptions, SolveResult
from halbach.symmetry import build_mirror_x0, expand_delta_phi
from halbach.symmetry_fourier import build_fourier_x0_features, delta_full_from_fourier
from halbach.types import Geometry

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
    ap.add_argument(
        "--field-scale",
        type=float,
        default=FIELD_SCALE_DEFAULT,
        help="field scaling factor for objective stability",
    )
    ap.add_argument(
        "--angle-model",
        type=str,
        choices=["legacy-alpha", "delta-rep-x0", "fourier-x0"],
        default="legacy-alpha",
        help="angle model to optimize",
    )
    ap.add_argument(
        "--grad-backend",
        type=str,
        choices=["analytic", "jax"],
        default=None,
        help="gradient backend (default depends on angle model)",
    )
    ap.add_argument(
        "--fourier-H",
        type=int,
        default=4,
        help="Fourier basis size for fourier-x0",
    )
    ap.add_argument("--lambda0", type=float, default=0.0, help="delta/phi L2 regularization")
    ap.add_argument(
        "--lambda-theta",
        type=float,
        default=0.0,
        help="delta/phi smoothness regularization in theta",
    )
    ap.add_argument(
        "--lambda-z",
        type=float,
        default=0.0,
        help="delta/phi smoothness regularization along z",
    )
    ap.add_argument(
        "--angle-init",
        type=str,
        choices=["from-run", "zeros"],
        default="from-run",
        help="initialize angle variables from run or zeros",
    )
    ap.add_argument(
        "--mag-model",
        type=str,
        choices=["fixed", "self-consistent-easy-axis"],
        default="fixed",
        help="magnetization model (self-consistent is not executed yet)",
    )
    ap.add_argument("--sc-chi", type=float, default=0.0, help="self-consistent chi")
    ap.add_argument("--sc-Nd", type=float, default=1.0 / 3.0, help="self-consistent Nd")
    ap.add_argument("--sc-p0", type=float, default=1.0, help="self-consistent p0")
    ap.add_argument(
        "--sc-volume-mm3",
        type=float,
        default=1000.0,
        help="self-consistent magnet volume [mm^3]",
    )
    ap.add_argument("--sc-iters", type=int, default=30, help="self-consistent iterations")
    ap.add_argument("--sc-omega", type=float, default=0.6, help="self-consistent mixing")
    ap.add_argument("--sc-near-wr", type=int, default=0, help="near window wr")
    ap.add_argument("--sc-near-wz", type=int, default=1, help="near window wz")
    ap.add_argument("--sc-near-wphi", type=int, default=2, help="near window wphi")
    ap.add_argument(
        "--sc-near-kernel",
        type=str,
        choices=["dipole", "multi-dipole", "cellavg", "cube-average", "gl-double-mixed"],
        default="dipole",
        help="near-field kernel model",
    )
    ap.add_argument(
        "--sc-gl-order",
        type=int,
        choices=[2, 3],
        default=None,
        help="gl-double-mixed order (2 or 3). When omitted, uses mixed 2/3.",
    )
    ap.add_argument("--sc-subdip-n", type=int, default=2, help="sub-dipole grid size")
    ap.add_argument(
        "--r-bound-mode",
        type=str,
        choices=["none", "relative", "absolute"],
        default="relative",
        help="radius bounds mode (none, relative, absolute)",
    )
    ap.add_argument(
        "--r-lower-delta-mm",
        type=float,
        default=30.0,
        help="relative lower delta for r_bases [mm]",
    )
    ap.add_argument(
        "--r-upper-delta-mm",
        type=float,
        default=30.0,
        help="relative upper delta for r_bases [mm]",
    )
    ap.add_argument(
        "--r-no-upper",
        action="store_true",
        help="disable upper bound in relative mode",
    )
    ap.add_argument("--r-min-mm", type=float, default=0.0, help="absolute min r [mm]")
    ap.add_argument("--r-max-mm", type=float, default=1e9, help="absolute max r [mm]")
    ap.add_argument(
        "--min-radius-drop-mm",
        type=float,
        default=None,
        help="deprecated (use --r-lower-delta-mm)",
    )
    ap.add_argument(
        "--fix-center-radius-layers",
        type=int,
        choices=[0, 2, 4],
        default=2,
        help="number of center z-layers to fix radius only (0, 2, or 4)",
    )
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
    ap.add_argument("--sc-debug", action="store_true", help="write self-consistent debug bundle")
    ap.add_argument(
        "--sc-debug-scale-check",
        dest="sc_debug_scale_check",
        action="store_true",
        default=True,
        help="enable field-scale debug checks",
    )
    ap.add_argument(
        "--no-sc-debug-scale-check",
        dest="sc_debug_scale_check",
        action="store_false",
        help="disable field-scale debug checks",
    )
    ap.add_argument("--dry-run", action="store_true", help="evaluate once and exit")
    return ap.parse_args(argv)


BoundMode = Literal["none", "relative", "absolute"]
Bounds = list[tuple[float | None, float | None]]
Float1DArray: TypeAlias = NDArray[np.float64]


def _require_jax() -> None:
    try:
        importlib.import_module("jax")
    except Exception as exc:
        raise ModuleNotFoundError(
            "JAX backend requested but `jax` is not installed. "
            "Install `jax` and `jaxlib`, or use --angle-model legacy-alpha "
            "and --grad-backend analytic."
        ) from exc


def build_bounds(
    param_map: ParamMap,
    r_bases0: NDArray[np.float64],
    *,
    n_angle: int,
    mode: BoundMode,
    dl_mm: float,
    du_mm: float,
    rmin_mm: float,
    rmax_mm: float,
    no_upper: bool,
) -> Bounds:
    n_r = int(param_map.free_r_idx.size)
    bounds: Bounds = [(None, None) for _ in range(n_angle)]

    if mode == "none":
        bounds.extend((None, None) for _ in range(n_r))
        return bounds

    if mode == "relative":
        if dl_mm < 0.0 or du_mm < 0.0:
            raise ValueError("relative radius deltas must be >= 0")
        dl_m = dl_mm * 1e-3
        du_m = du_mm * 1e-3
        for k_low in param_map.free_r_idx:
            r0 = float(r_bases0[k_low])
            lb = r0 - dl_m
            ub = None if no_upper else r0 + du_m
            bounds.append((lb, ub))
        return bounds

    if mode == "absolute":
        rmin_m = rmin_mm * 1e-3
        rmax_m = rmax_mm * 1e-3
        if rmax_m < rmin_m:
            raise ValueError("r_max_mm must be >= r_min_mm")
        for _k_low in param_map.free_r_idx:
            bounds.append((rmin_m, rmax_m))
        return bounds

    raise ValueError(f"Unsupported bound mode: {mode}")


def _pack_r_vars(r_bases: NDArray[np.float64], param_map: ParamMap) -> NDArray[np.float64]:
    return np.asarray(r_bases[param_map.free_r_idx], dtype=np.float64)


def _apply_r_vars(
    r_vars: NDArray[np.float64],
    r_bases0: NDArray[np.float64],
    param_map: ParamMap,
) -> NDArray[np.float64]:
    r_bases = np.array(r_bases0, dtype=np.float64, copy=True)
    if r_vars.size != param_map.free_r_idx.size:
        raise ValueError("r_vars size does not match parameter map")
    for idx, k_low in enumerate(param_map.free_r_idx):
        r_bases[k_low] = r_vars[idx]
        k_up = param_map.lower_to_upper[k_low]
        if k_up >= 0:
            r_bases[k_up] = r_vars[idx]
    return r_bases


def _pack_r_grad(grad_r_bases: NDArray[np.float64], param_map: ParamMap) -> NDArray[np.float64]:
    g_r = np.zeros(param_map.free_r_idx.size, dtype=np.float64)
    for idx, k_low in enumerate(param_map.free_r_idx):
        k_up = param_map.lower_to_upper[k_low]
        if k_up >= 0:
            g_r[idx] = grad_r_bases[k_low] + grad_r_bases[k_up]
        else:
            g_r[idx] = grad_r_bases[k_low]
    return g_r


def _build_r0_rkn_from_r_bases(r_bases: NDArray[np.float64], geom: Geometry) -> NDArray[np.float64]:
    r_bases_f = np.asarray(r_bases, dtype=np.float64)
    rho = r_bases_f[None, :] + np.asarray(geom.ring_offsets, dtype=np.float64)[:, None]
    px = rho[:, :, None] * np.asarray(geom.cth, dtype=np.float64)[None, None, :]
    py = rho[:, :, None] * np.asarray(geom.sth, dtype=np.float64)[None, None, :]
    pz = np.broadcast_to(
        np.asarray(geom.z_layers, dtype=np.float64)[None, :, None],
        px.shape,
    )
    return np.stack([px, py, pz], axis=-1)


def _phi_rkn_from_final(
    angle_model: str,
    geom: Geometry,
    alphas: NDArray[np.float64],
    delta_rep_opt: NDArray[np.float64] | None,
    coeffs_opt: NDArray[np.float64] | None,
    *,
    fourier_H: int,
    phi0_val: float,
) -> NDArray[np.float64]:
    theta = np.asarray(geom.theta, dtype=np.float64)
    base = 2.0 * theta + float(phi0_val)
    R = int(geom.R)
    K = int(geom.K)
    N = int(geom.N)

    if angle_model == "legacy-alpha":
        sin2 = np.asarray(geom.sin2, dtype=np.float64)
        phi_rkn = (
            base[None, None, :]
            + np.asarray(alphas, dtype=np.float64)[:, :, None] * sin2[None, None, :]
        )
        return np.asarray(phi_rkn, dtype=np.float64)

    if angle_model == "delta-rep-x0":
        if delta_rep_opt is None:
            raise ValueError("delta-rep-x0 requires delta_rep_opt to build phi")
        mirror = build_mirror_x0(N)
        if delta_rep_opt.shape[1] != mirror.rep_idx.size:
            raise ValueError("delta_rep_opt width does not match mirror rep size")
        delta_full = expand_delta_phi(delta_rep_opt, mirror.basis)
        phi_kn = base[None, :] + delta_full
        phi_rkn = np.broadcast_to(phi_kn[None, :, :], (R, K, N))
        return np.asarray(phi_rkn, dtype=np.float64)

    if angle_model == "fourier-x0":
        if coeffs_opt is None:
            raise ValueError("fourier-x0 requires fourier_coeffs_opt to build phi")
        cos_odd, sin_even = build_fourier_x0_features(theta, int(fourier_H))
        delta_full = delta_full_from_fourier(coeffs_opt, cos_odd, sin_even)
        phi_kn = base[None, :] + delta_full
        phi_rkn = np.broadcast_to(phi_kn[None, :, :], (R, K, N))
        return np.asarray(phi_rkn, dtype=np.float64)

    raise ValueError(f"Unsupported angle_model: {angle_model}")


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
    param_map = build_param_map(geom.R, geom.K, n_fix_radius=int(args.fix_center_radius_layers))
    alphas0 = np.array(run.results.alphas, dtype=np.float64, copy=True)
    r_bases0 = np.array(run.results.r_bases, dtype=np.float64, copy=True)

    angle_model = cast(str, args.angle_model)
    if angle_model not in ("legacy-alpha", "delta-rep-x0", "fourier-x0"):
        raise ValueError(f"Unsupported angle_model: {angle_model}")
    grad_backend = args.grad_backend
    if grad_backend is None:
        grad_backend = "analytic" if angle_model == "legacy-alpha" else "jax"
    if angle_model in ("delta-rep-x0", "fourier-x0") and grad_backend != "jax":
        raise ValueError("delta/fourier models require --grad-backend jax")
    if angle_model != "legacy-alpha" or grad_backend == "jax":
        _require_jax()

    angle_init = str(args.angle_init)
    if angle_init not in ("from-run", "zeros"):
        raise ValueError(f"Unsupported angle_init: {angle_init}")

    delta_rep0: NDArray[np.float64] | None = None
    coeffs0: NDArray[np.float64] | None = None
    fourier_H = int(args.fourier_H)
    if angle_model in ("delta-rep-x0", "fourier-x0") and geom.N % 2 != 0:
        raise ValueError("N must be even for delta/fourier angle models")

    mag_model_requested = str(args.mag_model)
    mag_model_effective = "fixed"
    if mag_model_requested == "self-consistent-easy-axis":
        if angle_model == "legacy-alpha":
            if grad_backend == "jax":
                mag_model_effective = "self-consistent-easy-axis"
            else:
                logger.warning(
                    "mag-model %s requested but only supported for legacy-alpha + jax; running fixed model",
                    mag_model_requested,
                )
        elif angle_model in ("delta-rep-x0", "fourier-x0"):
            mag_model_effective = "self-consistent-easy-axis"
        else:
            logger.warning(
                "mag-model %s requested but not supported; running fixed model",
                mag_model_requested,
            )
    elif mag_model_requested != "fixed":
        logger.warning(
            "mag-model %s requested but not supported; running fixed model",
            mag_model_requested,
        )

    if str(args.sc_near_kernel) == "cube-average":
        args.sc_near_kernel = "cellavg"

    if int(args.sc_near_wr) < 0 or int(args.sc_near_wz) < 0 or int(args.sc_near_wphi) < 0:
        raise ValueError("self-consistent near window must be >= 0")
    if int(args.sc_iters) < 1:
        raise ValueError("self-consistent iters must be >= 1")
    if float(args.sc_omega) <= 0.0 or float(args.sc_omega) > 1.0:
        raise ValueError("self-consistent omega must be in (0, 1]")
    if float(args.sc_volume_mm3) <= 0.0:
        raise ValueError("self-consistent volume_mm3 must be > 0")
    if float(args.sc_Nd) < 0.0 or float(args.sc_Nd) > 1.0:
        raise ValueError("self-consistent Nd must be in [0, 1]")
    if str(args.sc_near_kernel) == "multi-dipole" and int(args.sc_subdip_n) < 2:
        raise ValueError("self-consistent subdip_n must be >= 2 for multi-dipole")

    sc_cfg_payload: dict[str, Any] = dict(
        chi=float(args.sc_chi),
        Nd=float(args.sc_Nd),
        p0=float(args.sc_p0),
        volume_mm3=float(args.sc_volume_mm3),
        iters=int(args.sc_iters),
        omega=float(args.sc_omega),
        near_window=dict(
            wr=int(args.sc_near_wr),
            wz=int(args.sc_near_wz),
            wphi=int(args.sc_near_wphi),
        ),
        near_kernel=str(args.sc_near_kernel),
        subdip_n=int(args.sc_subdip_n),
    )
    sc_gl_order = 0
    if str(args.sc_near_kernel) == "gl-double-mixed" and args.sc_gl_order is not None:
        sc_gl_order = int(args.sc_gl_order)
        sc_cfg_payload["gl_order"] = sc_gl_order

    sc_final_extras: dict[str, Any] | None = None
    if angle_model == "legacy-alpha":
        if angle_init == "zeros":
            alphas0 = np.zeros_like(alphas0)
        angle_dim = int(alphas0.size)
        x0 = pack_x(alphas0, r_bases0, param_map)
    elif angle_model == "delta-rep-x0":
        mirror = build_mirror_x0(int(geom.N))
        n_rep = int(mirror.rep_idx.size)
        if angle_init == "from-run":
            raw = run.results.extras.get("delta_rep_opt")
            if raw is None:
                raw = run.results.extras.get("delta_rep")
            if raw is not None:
                delta_rep0 = np.asarray(raw, dtype=np.float64)
        if delta_rep0 is None:
            logger.warning("delta_rep not found; initializing delta_rep to zeros")
            delta_rep0 = np.zeros((geom.K, n_rep), dtype=np.float64)
        if delta_rep0.shape != (geom.K, n_rep):
            raise ValueError("delta_rep shape does not match K and mirror rep count")
        angle_dim = int(delta_rep0.size)
        r_vars0 = _pack_r_vars(r_bases0, param_map)
        x0 = cast(Float1DArray, np.concatenate([delta_rep0.ravel(), r_vars0]))
    else:
        if fourier_H < 0:
            raise ValueError("fourier_H must be >= 0")
        if angle_init == "from-run":
            raw = run.results.extras.get("fourier_coeffs_opt")
            if raw is None:
                raw = run.results.extras.get("fourier_coeffs")
            if raw is not None:
                coeffs0 = np.asarray(raw, dtype=np.float64)
                if coeffs0.shape[1] % 2 == 0:
                    inferred_H = int(coeffs0.shape[1] // 2)
                    if inferred_H != fourier_H:
                        logger.warning("fourier_H mismatch; using H=%d from coeffs", inferred_H)
                        fourier_H = inferred_H
        if coeffs0 is None:
            logger.warning("fourier_coeffs not found; initializing coeffs to zeros")
            coeffs0 = np.zeros((geom.K, 2 * fourier_H), dtype=np.float64)
        if coeffs0.shape != (geom.K, 2 * fourier_H):
            raise ValueError("fourier_coeffs shape does not match K and H")
        angle_dim = int(coeffs0.size)
        r_vars0 = _pack_r_vars(r_bases0, param_map)
        x0 = cast(Float1DArray, np.concatenate([coeffs0.ravel(), r_vars0]))

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

    r_bound_mode = cast(BoundMode, args.r_bound_mode)
    r_lower_delta_mm = float(args.r_lower_delta_mm)
    if args.min_radius_drop_mm is not None and r_bound_mode == "relative":
        logger.warning("min-radius-drop-mm is deprecated; use r-lower-delta-mm")
        r_lower_delta_mm = float(args.min_radius_drop_mm)
    bounds = build_bounds(
        param_map,
        r_bases0,
        n_angle=angle_dim,
        mode=r_bound_mode,
        dl_mm=r_lower_delta_mm,
        du_mm=float(args.r_upper_delta_mm),
        rmin_mm=float(args.r_min_mm),
        rmax_mm=float(args.r_max_mm),
        no_upper=bool(args.r_no_upper),
    )

    radius_dim_free = int(param_map.free_r_idx.size)
    logger.info(
        "x_dim total = %d, angle_dim = %d, radius_dim_free = %d, fixed_radius_layers = %s",
        x0.size,
        angle_dim,
        radius_dim_free,
        param_map.fixed_k_radius.tolist(),
    )
    logger.info(
        "radius bounds: mode=%s lower_delta_mm=%.3f upper_delta_mm=%.3f no_upper=%s min_mm=%.3f max_mm=%.3f",
        r_bound_mode,
        r_lower_delta_mm,
        float(args.r_upper_delta_mm),
        bool(args.r_no_upper),
        float(args.r_min_mm),
        float(args.r_max_mm),
    )
    if radius_dim_free > 0:
        sample_count = min(3, radius_dim_free)
        samples: list[str] = []
        for idx in range(sample_count):
            k_low = int(param_map.free_r_idx[idx])
            r0_mm = float(r_bases0[k_low] * 1e3)
            lb, ub = bounds[angle_dim + idx]
            lb_text = "None" if lb is None else f"{lb * 1e3:.3f}mm"
            ub_text = "None" if ub is None else f"{ub * 1e3:.3f}mm"
            samples.append(f"k={k_low} r0={r0_mm:.3f}mm lb={lb_text} ub={ub_text}")
        logger.info("radius bounds sample: %s", "; ".join(samples))

    field_scale = float(args.field_scale)
    if field_scale <= 0.0:
        raise ValueError("field_scale must be positive")
    factor = FACTOR * field_scale

    sc_volume_m3 = float(args.sc_volume_mm3) * 1e-9
    sc_nbr_idx: NDArray[np.int32] | None = None
    sc_nbr_mask: NDArray[np.bool_] | None = None
    sc_logged_kernel_info = False
    if mag_model_effective == "self-consistent-easy-axis":
        from halbach.near import NearWindow, build_near_graph

        window = NearWindow(
            wr=int(args.sc_near_wr),
            wz=int(args.sc_near_wz),
            wphi=int(args.sc_near_wphi),
        )
        near = build_near_graph(int(geom.R), int(geom.K), int(geom.N), window)
        sc_nbr_idx = near.nbr_idx
        sc_nbr_mask = near.nbr_mask

    first_eval_done = False
    state: dict[str, float] = {}
    n_rep = delta_rep0.shape[1] if delta_rep0 is not None else 0

    logger.info(
        "angle_model=%s grad_backend=%s angle_init=%s", angle_model, grad_backend, angle_init
    )

    def fun_grad_solver(x: Float1DArray) -> tuple[float, Float1DArray, dict[str, Any]]:
        nonlocal sc_logged_kernel_info
        nonlocal first_eval_done
        if not first_eval_done:
            logger.info("first eval start")
        t_eval = perf_counter()
        sc_extras: dict[str, Any] | None = None

        if angle_model == "legacy-alpha":
            if grad_backend == "analytic":
                alphas, r_bases = unpack_x(x, alphas0, r_bases0, param_map)
                J_data, gA_y, gRb_y, B0n = objective_with_grads_fixed(
                    alphas, r_bases, geom, pts, factor=factor
                )
                gx = pack_grad(gA_y, gRb_y, param_map)
                J_total = float(J_data)
            else:
                alphas, r_bases = unpack_x(x, alphas0, r_bases0, param_map)
                if mag_model_effective == "self-consistent-easy-axis":
                    from halbach.autodiff.jax_objective_self_consistent_legacy import (
                        objective_with_grads_self_consistent_legacy_jax,
                    )

                    if sc_nbr_idx is None or sc_nbr_mask is None:
                        raise ValueError("self-consistent near graph is missing")
                    J_data, gA_y, gRb_y, B0n, sc_extras = (
                        objective_with_grads_self_consistent_legacy_jax(
                            alphas,
                            r_bases,
                            geom,
                            pts,
                            sc_nbr_idx,
                            sc_nbr_mask,
                            chi=float(args.sc_chi),
                            Nd=float(args.sc_Nd),
                            p0=float(args.sc_p0),
                            volume_m3=sc_volume_m3,
                            near_kernel=str(args.sc_near_kernel),
                            subdip_n=int(args.sc_subdip_n),
                            gl_order=sc_gl_order,
                            iters=int(args.sc_iters),
                            omega=float(args.sc_omega),
                            factor=factor,
                            phi0_val=phi0,
                        )
                    )
                else:
                    from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax

                    J_data, gA_y, gRb_y, B0n = objective_with_grads_fixed_jax(
                        alphas, r_bases, geom, pts, factor=factor
                    )
                gx = pack_grad(gA_y, gRb_y, param_map)
                J_total = float(J_data)
        elif angle_model == "delta-rep-x0":
            from halbach.autodiff.jax_objective_delta_phi import (
                objective_with_grads_delta_phi_x0_jax,
            )

            delta_flat = np.asarray(x[:angle_dim], dtype=np.float64)
            r_vars = np.asarray(x[angle_dim:], dtype=np.float64)
            delta_rep = delta_flat.reshape((geom.K, n_rep))
            r_bases = _apply_r_vars(r_vars, r_bases0, param_map)
            if mag_model_effective == "self-consistent-easy-axis":
                from halbach.autodiff.jax_objective_self_consistent_delta_phi import (
                    objective_with_grads_self_consistent_delta_phi_x0_jax,
                )

                if sc_nbr_idx is None or sc_nbr_mask is None:
                    raise ValueError("self-consistent near graph is missing")
                J_total, g_delta, g_r_bases, B0n, sc_extras = (
                    objective_with_grads_self_consistent_delta_phi_x0_jax(
                        delta_rep,
                        r_bases,
                        geom,
                        pts,
                        sc_nbr_idx,
                        sc_nbr_mask,
                        chi=float(args.sc_chi),
                        Nd=float(args.sc_Nd),
                        p0=float(args.sc_p0),
                        volume_m3=sc_volume_m3,
                        iters=int(args.sc_iters),
                        omega=float(args.sc_omega),
                        near_kernel=str(args.sc_near_kernel),
                        subdip_n=int(args.sc_subdip_n),
                        gl_order=sc_gl_order,
                        lambda0=float(args.lambda0),
                        lambda_theta=float(args.lambda_theta),
                        lambda_z=float(args.lambda_z),
                        factor=factor,
                        phi0=phi0,
                    )
                )
            else:
                J_total, g_delta, g_r_bases, B0n = objective_with_grads_delta_phi_x0_jax(
                    delta_rep,
                    r_bases,
                    geom,
                    pts,
                    lambda0=float(args.lambda0),
                    lambda_theta=float(args.lambda_theta),
                    lambda_z=float(args.lambda_z),
                    factor=factor,
                    phi0=phi0,
                    m0=m0,
                )
            gx = cast(
                Float1DArray,
                np.concatenate([g_delta.ravel(), _pack_r_grad(g_r_bases, param_map)]),
            )
            J_data = float(J_total)
        else:
            from halbach.autodiff.jax_objective_delta_phi_fourier import (
                objective_with_grads_delta_phi_fourier_x0_jax,
            )

            coeffs_flat = np.asarray(x[:angle_dim], dtype=np.float64)
            r_vars = np.asarray(x[angle_dim:], dtype=np.float64)
            coeffs = coeffs_flat.reshape((geom.K, 2 * fourier_H))
            r_bases = _apply_r_vars(r_vars, r_bases0, param_map)
            if mag_model_effective == "self-consistent-easy-axis":
                from halbach.autodiff.jax_objective_self_consistent_delta_phi_fourier import (
                    objective_with_grads_self_consistent_delta_phi_fourier_x0_jax,
                )

                if sc_nbr_idx is None or sc_nbr_mask is None:
                    raise ValueError("self-consistent near graph is missing")
                J_total, g_coeffs, g_r_bases, B0n, sc_extras = (
                    objective_with_grads_self_consistent_delta_phi_fourier_x0_jax(
                        coeffs,
                        r_bases,
                        geom,
                        pts,
                        sc_nbr_idx,
                        sc_nbr_mask,
                        H=fourier_H,
                        chi=float(args.sc_chi),
                        Nd=float(args.sc_Nd),
                        p0=float(args.sc_p0),
                        volume_m3=sc_volume_m3,
                        iters=int(args.sc_iters),
                        omega=float(args.sc_omega),
                        near_kernel=str(args.sc_near_kernel),
                        subdip_n=int(args.sc_subdip_n),
                        gl_order=sc_gl_order,
                        lambda0=float(args.lambda0),
                        lambda_theta=float(args.lambda_theta),
                        lambda_z=float(args.lambda_z),
                        factor=factor,
                        phi0=phi0,
                    )
                )
            else:
                J_total, g_coeffs, g_r_bases, B0n = objective_with_grads_delta_phi_fourier_x0_jax(
                    coeffs,
                    r_bases,
                    geom,
                    pts,
                    H=fourier_H,
                    lambda0=float(args.lambda0),
                    lambda_theta=float(args.lambda_theta),
                    lambda_z=float(args.lambda_z),
                    factor=factor,
                    phi0=phi0,
                    m0=m0,
                )
            gx = cast(
                Float1DArray,
                np.concatenate([g_coeffs.ravel(), _pack_r_grad(g_r_bases, param_map)]),
            )
            J_data = float(J_total)

        dt_eval = perf_counter() - t_eval
        if not first_eval_done:
            logger.info("first eval done in %.3fs", dt_eval)
            first_eval_done = True
        state["J"] = float(J_data)
        state["gnorm"] = float(np.linalg.norm(gx))
        B0_T = float(B0n) / field_scale
        state["B0"] = B0_T
        state["t_last_eval"] = float(dt_eval)
        extras: dict[str, Any] = {"J": float(J_data), "B0": B0_T}
        if mag_model_effective == "self-consistent-easy-axis" and sc_extras is not None:
            extras.update(sc_extras)
            if not sc_logged_kernel_info:
                deg = int(sc_extras.get("sc_near_deg_max", 0))
                subdip_n = int(sc_extras.get("sc_subdip_n", 1))
                S = subdip_n**3 if str(args.sc_near_kernel) == "multi-dipole" else 1
                M = int(geom.R * geom.K * geom.N)
                approx_pairs = M * deg
                approx_evals = approx_pairs * S
                logger.info(
                    "self-consistent near kernel=%s deg=%d subdip=%d (S=%d) approx_evals_per_iter=%d",
                    args.sc_near_kernel,
                    deg,
                    subdip_n,
                    S,
                    approx_evals,
                )
                sc_logged_kernel_info = True
        return float(J_total), gx, extras

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
        Jn, g0, extras_eval = fun_grad_solver(x0)
        gnorm = float(np.linalg.norm(g0))
        logger.info(
            "dry-run: J=%.6e |B0|=%.3f mT gnorm=%.3e",
            Jn,
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
            scaling=dict(
                field_scale=float(field_scale),
            ),
            optimizer=dict(
                method="L-BFGS-B",
                maxiter=int(args.maxiter),
                gtol=float(args.gtol),
                r_bound_mode=str(r_bound_mode),
                r_lower_delta_mm=float(r_lower_delta_mm),
                r_upper_delta_mm=float(args.r_upper_delta_mm),
                r_no_upper=bool(args.r_no_upper),
                r_min_mm=float(args.r_min_mm),
                r_max_mm=float(args.r_max_mm),
            ),
            angle_model=str(angle_model),
            grad_backend=str(grad_backend),
            angle_init=str(angle_init),
            fourier_H=int(fourier_H),
            regularization=dict(
                lambda0=float(args.lambda0),
                lambda_theta=float(args.lambda_theta),
                lambda_z=float(args.lambda_z),
            ),
            magnetization=dict(
                model_requested=mag_model_requested,
                model_effective=mag_model_effective,
                self_consistent=sc_cfg_payload,
            ),
            fix_center_radius_layers=int(args.fix_center_radius_layers),
            fixed_k_radius=param_map.fixed_k_radius.tolist(),
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

    delta_rep_opt: NDArray[np.float64] | None = None
    coeffs_opt: NDArray[np.float64] | None = None
    if angle_model == "legacy-alpha":
        al_opt, rb_opt = unpack_x(res.x, alphas0, r_bases0, param_map)
        if grad_backend == "jax":
            if mag_model_effective == "self-consistent-easy-axis":
                from halbach.autodiff.jax_objective_self_consistent_legacy import (
                    objective_with_grads_self_consistent_legacy_jax,
                )

                if sc_nbr_idx is None or sc_nbr_mask is None:
                    raise ValueError("self-consistent near graph is missing")
                Jn_f, _gA, _gR, B0_f, sc_final_extras = (
                    objective_with_grads_self_consistent_legacy_jax(
                        al_opt,
                        rb_opt,
                        geom,
                        pts,
                        sc_nbr_idx,
                        sc_nbr_mask,
                        chi=float(args.sc_chi),
                        Nd=float(args.sc_Nd),
                        p0=float(args.sc_p0),
                        volume_m3=sc_volume_m3,
                        near_kernel=str(args.sc_near_kernel),
                        subdip_n=int(args.sc_subdip_n),
                        gl_order=sc_gl_order,
                        iters=int(args.sc_iters),
                        omega=float(args.sc_omega),
                        factor=factor,
                        phi0_val=phi0,
                    )
                )
            else:
                from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax

                Jn_f, _gA, _gR, B0_f = objective_with_grads_fixed_jax(
                    al_opt, rb_opt, geom, pts, factor=factor
                )
        else:
            Jn_f, _gA, _gR, B0_f = objective_with_grads_fixed(
                al_opt, rb_opt, geom, pts, factor=factor
            )
    elif angle_model == "delta-rep-x0":
        delta_flat = np.asarray(res.x[:angle_dim], dtype=np.float64)
        r_vars = np.asarray(res.x[angle_dim:], dtype=np.float64)
        delta_rep_opt = delta_flat.reshape((geom.K, n_rep))
        rb_opt = _apply_r_vars(r_vars, r_bases0, param_map)
        al_opt = np.zeros((geom.R, geom.K), dtype=np.float64)
        if mag_model_effective == "self-consistent-easy-axis":
            from halbach.autodiff.jax_objective_self_consistent_delta_phi import (
                objective_with_grads_self_consistent_delta_phi_x0_jax,
            )

            if sc_nbr_idx is None or sc_nbr_mask is None:
                raise ValueError("self-consistent near graph is missing")
            Jn_f, _gD, _gR, B0_f, sc_final_extras = (
                objective_with_grads_self_consistent_delta_phi_x0_jax(
                    delta_rep_opt,
                    rb_opt,
                    geom,
                    pts,
                    sc_nbr_idx,
                    sc_nbr_mask,
                    chi=float(args.sc_chi),
                    Nd=float(args.sc_Nd),
                    p0=float(args.sc_p0),
                    volume_m3=sc_volume_m3,
                    iters=int(args.sc_iters),
                    omega=float(args.sc_omega),
                    near_kernel=str(args.sc_near_kernel),
                    subdip_n=int(args.sc_subdip_n),
                    gl_order=sc_gl_order,
                    lambda0=float(args.lambda0),
                    lambda_theta=float(args.lambda_theta),
                    lambda_z=float(args.lambda_z),
                    factor=factor,
                    phi0=phi0,
                )
            )
        else:
            from halbach.autodiff.jax_objective_delta_phi import (
                objective_with_grads_delta_phi_x0_jax,
            )

            Jn_f, _gD, _gR, B0_f = objective_with_grads_delta_phi_x0_jax(
                delta_rep_opt,
                rb_opt,
                geom,
                pts,
                lambda0=float(args.lambda0),
                lambda_theta=float(args.lambda_theta),
                lambda_z=float(args.lambda_z),
                factor=factor,
                phi0=phi0,
                m0=m0,
            )
    else:
        coeffs_flat = np.asarray(res.x[:angle_dim], dtype=np.float64)
        r_vars = np.asarray(res.x[angle_dim:], dtype=np.float64)
        coeffs_opt = coeffs_flat.reshape((geom.K, 2 * fourier_H))
        rb_opt = _apply_r_vars(r_vars, r_bases0, param_map)
        al_opt = np.zeros((geom.R, geom.K), dtype=np.float64)
        if mag_model_effective == "self-consistent-easy-axis":
            from halbach.autodiff.jax_objective_self_consistent_delta_phi_fourier import (
                objective_with_grads_self_consistent_delta_phi_fourier_x0_jax,
            )

            if sc_nbr_idx is None or sc_nbr_mask is None:
                raise ValueError("self-consistent near graph is missing")
            Jn_f, _gC, _gR, B0_f, sc_final_extras = (
                objective_with_grads_self_consistent_delta_phi_fourier_x0_jax(
                    coeffs_opt,
                    rb_opt,
                    geom,
                    pts,
                    sc_nbr_idx,
                    sc_nbr_mask,
                    H=fourier_H,
                    chi=float(args.sc_chi),
                    Nd=float(args.sc_Nd),
                    p0=float(args.sc_p0),
                    volume_m3=sc_volume_m3,
                    iters=int(args.sc_iters),
                    omega=float(args.sc_omega),
                    near_kernel=str(args.sc_near_kernel),
                    subdip_n=int(args.sc_subdip_n),
                    gl_order=sc_gl_order,
                    lambda0=float(args.lambda0),
                    lambda_theta=float(args.lambda_theta),
                    lambda_z=float(args.lambda_z),
                    factor=factor,
                    phi0=phi0,
                )
            )
        else:
            from halbach.autodiff.jax_objective_delta_phi_fourier import (
                objective_with_grads_delta_phi_fourier_x0_jax,
            )

            Jn_f, _gC, _gR, B0_f = objective_with_grads_delta_phi_fourier_x0_jax(
                coeffs_opt,
                rb_opt,
                geom,
                pts,
                H=fourier_H,
                lambda0=float(args.lambda0),
                lambda_theta=float(args.lambda_theta),
                lambda_z=float(args.lambda_z),
                factor=factor,
                phi0=phi0,
                m0=m0,
            )

    B0_f_T = B0_f / field_scale
    logger.info(
        "[done] success=%s iters=%d J=%.6e |B0|=%.3f mT",
        res.success,
        res.nit,
        Jn_f,
        B0_f_T * 1e3,
    )

    extras = res.trace.extras
    J_hist = np.array(res.trace.f, dtype=float)
    Jn_hist = np.array([float(e.get("J", np.nan)) for e in extras], dtype=float)
    B0_hist = np.array([float(e.get("B0", np.nan)) for e in extras], dtype=float)

    sc_p_flat: NDArray[np.float64] | None = None
    sc_cfg_fp: str | None = None
    phi_rkn_final: NDArray[np.float64] | None = None
    r0_rkn_final: NDArray[np.float64] | None = None
    if mag_model_effective == "self-consistent-easy-axis":
        try:
            sc_cfg_fp = sc_cfg_fingerprint(sc_cfg_payload)
            phi_rkn = _phi_rkn_from_final(
                angle_model,
                geom,
                al_opt,
                delta_rep_opt,
                coeffs_opt,
                fourier_H=fourier_H,
                phi0_val=phi0,
            )
            r0_rkn = _build_r0_rkn_from_r_bases(rb_opt, geom)
            phi_rkn_final = phi_rkn
            r0_rkn_final = r0_rkn
            sc_p_flat = compute_p_flat_self_consistent_jax(
                phi_rkn,
                r0_rkn,
                geom,
                sc_cfg_payload,
            )
        except Exception as exc:
            logger.warning("sc_p_flat save skipped: %s", exc)
            sc_p_flat = None
            sc_cfg_fp = None

    logger.info("save start")
    t_save = perf_counter()
    results_path = out_dir / "results.npz"
    save_payload: dict[str, Any] = dict(
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
    )
    if sc_p_flat is not None and sc_cfg_fp is not None:
        save_payload["sc_p_flat"] = sc_p_flat
        save_payload["sc_cfg_fingerprint"] = sc_cfg_fp
    if mag_model_effective == "self-consistent-easy-axis" and sc_final_extras is not None:
        save_payload["extras_sc_stats"] = dict(sc_final_extras)
    if delta_rep_opt is not None:
        save_payload["delta_rep_opt"] = delta_rep_opt
    if coeffs_opt is not None:
        save_payload["fourier_coeffs_opt"] = coeffs_opt
    np.savez_compressed(results_path, **save_payload)

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
        scaling=dict(
            field_scale=float(field_scale),
        ),
        optimizer=dict(
            method="L-BFGS-B",
            maxiter=int(args.maxiter),
            gtol=float(args.gtol),
            r_bound_mode=str(r_bound_mode),
            r_lower_delta_mm=float(r_lower_delta_mm),
            r_upper_delta_mm=float(args.r_upper_delta_mm),
            r_no_upper=bool(args.r_no_upper),
            r_min_mm=float(args.r_min_mm),
            r_max_mm=float(args.r_max_mm),
        ),
        angle_model=str(angle_model),
        grad_backend=str(grad_backend),
        angle_init=str(angle_init),
        fourier_H=int(fourier_H),
        regularization=dict(
            lambda0=float(args.lambda0),
            lambda_theta=float(args.lambda_theta),
            lambda_z=float(args.lambda_z),
        ),
        magnetization=dict(
            model_requested=mag_model_requested,
            model_effective=mag_model_effective,
            self_consistent=sc_cfg_payload,
        ),
        fix_center_radius_layers=int(args.fix_center_radius_layers),
        fixed_k_radius=param_map.fixed_k_radius.tolist(),
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

    if bool(args.sc_debug) and mag_model_effective == "self-consistent-easy-axis":
        try:
            if phi_rkn_final is None or r0_rkn_final is None:
                phi_rkn_final = _phi_rkn_from_final(
                    angle_model,
                    geom,
                    al_opt,
                    delta_rep_opt,
                    coeffs_opt,
                    fourier_H=fourier_H,
                    phi0_val=phi0,
                )
                r0_rkn_final = _build_r0_rkn_from_r_bases(rb_opt, geom)
            pts_debug = build_roi_points(roi_r=0.05, roi_step=0.05)
            if pts_debug.shape[0] > 100:
                rng = np.random.default_rng(0)
                sample_idx = rng.choice(pts_debug.shape[0], size=100, replace=False)
                pts_debug = pts_debug[sample_idx]
            make_sc_debug_bundle(
                run_dir=out_dir,
                out_dir=out_dir,
                geom=geom,
                phi_rkn=phi_rkn_final,
                r0_rkn=r0_rkn_final,
                pts=pts_debug,
                factor=FACTOR,
                field_scale_check=bool(args.sc_debug_scale_check),
                scale_factor=10.0,
            )
        except Exception as exc:
            logger.warning("sc_debug bundle failed: %s", exc)

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
