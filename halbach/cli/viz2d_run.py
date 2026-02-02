from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from halbach.run_io import load_run
from halbach.viz2d import compute_error_map_ppm_plane_with_debug

Plane = Literal["xy", "xz", "yz"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate 2D ppm error map from a run.")
    ap.add_argument("--run", dest="run_path", required=True, help="run dir or results.npz path")
    ap.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    ap.add_argument("--coord0", type=float, default=0.0, help="fixed coordinate for the plane [m]")
    ap.add_argument("--roi-r", type=float, default=0.14, help="ROI radius [m]")
    ap.add_argument("--step", type=float, default=0.001, help="grid step [m]")
    ap.add_argument("--out", type=str, default=None, help="output directory (default: run dir)")
    ap.add_argument("--prefix", type=str, default="error_map", help="output filename prefix")
    ap.add_argument("--no-png", action="store_true", help="skip PNG output")
    ap.add_argument("--no-npz", action="store_true", help="skip NPZ output")
    ap.add_argument("--no-json", action="store_true", help="skip JSON output")
    return ap.parse_args(argv)


def _resolve_out_dir(run_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    if run_path.is_dir():
        return run_path
    return run_path.parent


def _save_npz(path: Path, payload: dict[str, Any]) -> None:
    np.savez_compressed(path, **payload)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_png(path: Path, xs: np.ndarray, ys: np.ndarray, ppm: np.ndarray, plane: Plane) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    im = ax.pcolormesh(xs * 1e3, ys * 1e3, ppm, shading="auto", cmap="RdBu_r")
    ax.set_xlabel("x [mm]" if plane != "yz" else "y [mm]")
    ax.set_ylabel("y [mm]" if plane == "xy" else "z [mm]")
    ax.set_title(f"ppm map ({plane})")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label="ppm")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run() -> int:
    args = parse_args()
    run_path = Path(args.run_path)
    out_dir = _resolve_out_dir(run_path, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = load_run(run_path)
    try:
        m, debug = compute_error_map_ppm_plane_with_debug(
            run,
            plane=cast(Plane, args.plane),
            coord0=float(args.coord0),
            roi_r=float(args.roi_r),
            step=float(args.step),
        )
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "JAX is required for self-consistent visualization" in msg
            or "self-consistent visualization requires JAX" in msg
        ):
            print(msg)
            return 1
        raise

    tag = f"{args.prefix}_{args.plane}"
    if not args.no_png:
        _save_png(out_dir / f"{tag}.png", m.xs, m.ys, m.ppm, cast(Plane, args.plane))
    if not args.no_npz:
        _save_npz(
            out_dir / f"{tag}.npz",
            dict(
                xs=m.xs,
                ys=m.ys,
                ppm=m.ppm,
                mask=m.mask,
                B0_T=m.B0_T,
                plane=str(m.plane),
                coord0=float(m.coord0),
                roi_r=float(args.roi_r),
                step=float(args.step),
            ),
        )
    if not args.no_json:
        _save_json(
            out_dir / f"{tag}.json",
            dict(
                run_path=str(run_path),
                plane=str(m.plane),
                coord0=float(m.coord0),
                roi_r=float(args.roi_r),
                step=float(args.step),
                debug=debug,
            ),
        )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
