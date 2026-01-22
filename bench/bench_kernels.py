from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from halbach.constants import FACTOR, m0, phi0
from halbach.geom import build_roi_points
from halbach.objective import objective_with_grads_fixed
from halbach.physics import objective_only
from halbach.robust import hvp_y
from halbach.types import Geometry


@dataclass(frozen=True)
class Preset:
    N: int
    K: int
    R: int
    roi_r: float
    roi_step: float
    Lz: float
    ring_offsets: list[float]


PRESETS: dict[str, Preset] = {
    "tiny": Preset(N=12, K=6, R=1, roi_r=0.03, roi_step=0.03, Lz=0.2, ring_offsets=[0.0]),
    "dev": Preset(N=24, K=12, R=1, roi_r=0.08, roi_step=0.02, Lz=0.4, ring_offsets=[0.0]),
    "prod": Preset(
        N=48,
        K=24,
        R=3,
        roi_r=0.14,
        roi_step=0.02,
        Lz=0.4,
        ring_offsets=[0.0, 0.012, 0.024],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-benchmark kernels (objective/grad/HVP)")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="dev")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def build_geometry(cfg: Preset) -> Geometry:
    theta = np.linspace(0.0, 2.0 * np.pi, cfg.N, endpoint=False, dtype=np.float64)
    sin2 = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-cfg.Lz / 2.0, cfg.Lz / 2.0, cfg.K, dtype=np.float64)
    ring_offsets = np.array(cfg.ring_offsets, dtype=np.float64)
    if ring_offsets.shape[0] != cfg.R:
        raise ValueError("ring_offsets length must match R")
    dz = float(cfg.Lz) / float(cfg.K - 1) if cfg.K > 1 else float(cfg.Lz)
    return Geometry(
        theta=theta,
        sin2=sin2,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=cfg.N,
        K=cfg.K,
        R=cfg.R,
        dz=dz,
        Lz=float(cfg.Lz),
    )


def build_inputs(
    cfg: Preset,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    Geometry,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    geom = build_geometry(cfg)
    alphas = 1e-3 * rng.standard_normal((cfg.R, cfg.K))
    r0 = 0.2
    r_bases = r0 + 1e-4 * rng.standard_normal(cfg.K)
    pts = build_roi_points(cfg.roi_r, cfg.roi_step)
    v_y = rng.standard_normal(cfg.R * cfg.K + cfg.K)
    return alphas, r_bases, geom, pts, v_y


def measure(fn: Callable[[], Any], repeats: int) -> dict[str, float]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    start = perf_counter()
    _ = fn()
    compile_ms = (perf_counter() - start) * 1000.0
    samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _ = fn()
        samples.append((perf_counter() - start) * 1000.0)
    return {
        "compile_ms": float(compile_ms),
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
    }


def run_bench(preset_name: str, repeats: int) -> dict[str, Any]:
    cfg = PRESETS[preset_name]
    rng = np.random.default_rng(0)
    alphas, r_bases, geom, pts, v_y = build_inputs(cfg, rng)
    eps_hvp = 1e-6

    def run_objective_only() -> float:
        return objective_only(
            alphas,
            r_bases,
            geom.theta,
            geom.sin2,
            geom.cth,
            geom.sth,
            geom.z_layers,
            geom.ring_offsets,
            pts,
            FACTOR,
            phi0,
            m0,
        )

    def run_objective_with_grads() -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
        return objective_with_grads_fixed(alphas, r_bases, geom, pts)

    def run_hvp() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return hvp_y(alphas, r_bases, geom, pts, v_y, eps_hvp)

    results: dict[str, Any] = {
        "preset": preset_name,
        "repeats": int(repeats),
        "config": {
            "N": cfg.N,
            "K": cfg.K,
            "R": cfg.R,
            "roi_r": float(cfg.roi_r),
            "roi_step": float(cfg.roi_step),
            "Lz": float(cfg.Lz),
            "ring_offsets": list(cfg.ring_offsets),
            "roi_points": int(pts.shape[0]),
        },
        "objective_only": measure(run_objective_only, repeats),
        "objective_with_grads_fixed": measure(run_objective_with_grads, repeats),
        "hvp_y": measure(run_hvp, repeats),
    }
    return results


def print_summary(results: dict[str, Any]) -> None:
    cfg = results["config"]
    print(
        "preset={preset} repeats={repeats} N={N} K={K} R={R} roi_r={roi_r} roi_step={roi_step} "
        "roi_points={roi_points}".format(
            preset=results["preset"],
            repeats=results["repeats"],
            N=cfg["N"],
            K=cfg["K"],
            R=cfg["R"],
            roi_r=cfg["roi_r"],
            roi_step=cfg["roi_step"],
            roi_points=cfg["roi_points"],
        )
    )
    print(f"{'name':<28} {'compile_ms':>11} {'mean_ms':>9} {'median_ms':>11} {'min_ms':>9}")
    rows = [
        ("objective_only", results["objective_only"]),
        ("objective_with_grads_fixed", results["objective_with_grads_fixed"]),
        ("hvp_y", results["hvp_y"]),
    ]
    for name, stats in rows:
        print(
            f"{name:<28} {stats['compile_ms']:>11.3f} {stats['mean_ms']:>9.3f} "
            f"{stats['median_ms']:>11.3f} {stats['min_ms']:>9.3f}"
        )


def write_json(path: str, results: dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    results = run_bench(args.preset, args.repeats)
    print_summary(results)
    if args.out:
        write_json(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
