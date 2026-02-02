from __future__ import annotations

import argparse
from pathlib import Path

from halbach.generate import generate_halbach_initial, write_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate an initial Halbach run directory")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--name", default=None, help="run name (default: out dir name)")
    ap.add_argument("--N", type=int, required=True, help="magnets per ring")
    ap.add_argument("--R", type=int, required=True, help="number of radial rings")
    ap.add_argument("--K", type=int, required=True, help="number of z layers")
    ap.add_argument("--Lz", type=float, required=True, help="total z-span [m]")
    ap.add_argument("--diameter-mm", type=float, default=400.0, help="diameter [mm]")
    ap.add_argument(
        "--ring-offset-step-mm",
        type=float,
        default=12.0,
        help="ring offset step [mm]",
    )
    ap.add_argument("--schema-version", type=int, default=1, help="schema version")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    name = args.name or out_dir.name

    diameter_m = float(args.diameter_mm) * 1e-3
    ring_offset_step_m = float(args.ring_offset_step_mm) * 1e-3

    results = generate_halbach_initial(
        N=int(args.N),
        R=int(args.R),
        K=int(args.K),
        Lz=float(args.Lz),
        diameter_m=diameter_m,
        ring_offset_step_m=ring_offset_step_m,
    )
    write_run(
        out_dir,
        results,
        name=name,
        schema_version=int(args.schema_version),
        generator_params=dict(
            N=int(args.N),
            R=int(args.R),
            K=int(args.K),
            Lz=float(args.Lz),
            diameter_mm=float(args.diameter_mm),
            ring_offset_step_mm=float(args.ring_offset_step_mm),
        ),
        description="generated initial pure Halbach",
    )


if __name__ == "__main__":
    main()
