from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from halbach.types import FloatArray


def _build_theta(N: int) -> FloatArray:
    return np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).astype(np.float64)


def _build_z_layers(K: int, Lz: float) -> FloatArray:
    if K == 1:
        return np.array([0.0], dtype=np.float64)
    half = 0.5 * Lz
    return np.linspace(-half, half, K).astype(np.float64)


def _build_ring_offsets(R: int, step_m: float) -> FloatArray:
    if R == 1:
        return np.array([0.0], dtype=np.float64)
    center = 0.5 * (R - 1)
    offsets = (np.arange(R, dtype=np.float64) - center) * step_m
    return offsets.astype(np.float64)


def generate_run(
    out_dir: str | Path,
    *,
    N: int,
    R: int,
    K: int,
    Lz: float,
    diameter_mm: float,
    ring_offset_step_mm: float,
    name: str | None = None,
) -> Path:
    if N <= 0 or R <= 0 or K <= 0:
        raise ValueError("N, R, K must be >= 1")
    if Lz < 0.0:
        raise ValueError("Lz must be >= 0")
    if diameter_mm <= 0.0:
        raise ValueError("diameter_mm must be > 0")
    if ring_offset_step_mm < 0.0:
        raise ValueError("ring_offset_step_mm must be >= 0")

    out_path = Path(out_dir).expanduser()
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    r0 = diameter_mm / 2000.0
    step_m = ring_offset_step_mm / 1000.0

    theta = _build_theta(N)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)

    z_layers = _build_z_layers(K, Lz)
    ring_offsets = _build_ring_offsets(R, step_m)

    alphas = np.zeros((R, K), dtype=np.float64)
    r_bases = np.full(K, r0, dtype=np.float64)

    results_path = out_path / "results.npz"
    np.savez_compressed(
        results_path,
        alphas_opt=alphas,
        r_bases_opt=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )

    dz = float(np.median(np.abs(np.diff(z_layers)))) if z_layers.size > 1 else 0.0
    meta: dict[str, Any] = {
        "schema_version": 1,
        "name": name or out_path.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "geometry": {
            "N": int(N),
            "R": int(R),
            "K": int(K),
            "Lz": float(Lz),
            "dz": float(dz),
            "r0": float(r0),
            "diameter_mm": float(diameter_mm),
            "ring_offset_step_mm": float(ring_offset_step_mm),
        },
    }
    meta_path = out_path / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate an initial Halbach run directory.")
    ap.add_argument("--out", dest="out_dir", required=True, help="output run directory")
    ap.add_argument("--N", type=int, required=True, help="magnets per ring")
    ap.add_argument("--R", type=int, required=True, help="number of rings")
    ap.add_argument("--K", type=int, required=True, help="number of layers")
    ap.add_argument("--Lz", type=float, required=True, help="stack length (m)")
    ap.add_argument("--diameter-mm", type=float, required=True, help="array diameter (mm)")
    ap.add_argument(
        "--ring-offset-step-mm",
        type=float,
        required=True,
        help="radial ring offset step (mm)",
    )
    ap.add_argument("--name", default=None, help="optional run name override")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    generate_run(
        args.out_dir,
        N=args.N,
        R=args.R,
        K=args.K,
        Lz=args.Lz,
        diameter_mm=args.diameter_mm,
        ring_offset_step_mm=args.ring_offset_step_mm,
        name=args.name,
    )


if __name__ == "__main__":
    main()
