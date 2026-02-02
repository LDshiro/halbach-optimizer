from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from halbach.run_io import load_run
from halbach.run_types import RunBundle
from halbach.solvers.types import SolveTrace

STANDARD_KEYS = (
    "alphas_opt",
    "r_bases_opt",
    "theta",
    "sin2th",
    "cth",
    "sth",
    "z_layers",
    "ring_offsets",
)


def _trace_to_dict(trace: SolveTrace) -> dict[str, Any]:
    return {
        "iters": trace.iters,
        "f": trace.f,
        "gnorm": trace.gnorm,
        "extras": trace.extras,
    }


def _ensure_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def normalize_run(
    in_path: str | Path, out_dir: str | Path, *, name: str | None = None
) -> RunBundle:
    bundle = load_run(in_path, name=name)

    out_path = Path(out_dir).expanduser()
    _ensure_output_dir(out_path)

    results_out = out_path / "results.npz"
    extra_npz: dict[str, npt.NDArray[Any]] = {}
    extras_skipped: list[str] = []

    for key, value in bundle.results.extras.items():
        if key in STANDARD_KEYS:
            extras_skipped.append(key)
            continue
        if isinstance(value, np.ndarray):
            extra_npz[key] = value
        else:
            extras_skipped.append(key)

    save_kwargs: dict[str, Any] = {
        "alphas_opt": bundle.results.alphas,
        "r_bases_opt": bundle.results.r_bases,
        "theta": bundle.results.theta,
        "sin2th": bundle.results.sin2th,
        "cth": bundle.results.cth,
        "sth": bundle.results.sth,
        "z_layers": bundle.results.z_layers,
        "ring_offsets": bundle.results.ring_offsets,
    }
    save_kwargs.update(extra_npz)
    np.savez_compressed(results_out, **save_kwargs)

    meta = dict(bundle.meta)
    meta["schema_version"] = 1
    meta["normalized_from"] = str(Path(in_path).expanduser().resolve())
    meta["normalized_at"] = datetime.now(UTC).isoformat()
    if extras_skipped:
        meta["extras_skipped"] = sorted(set(extras_skipped))

    meta_out = out_path / "meta.json"
    with meta_out.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    if bundle.trace is not None:
        trace_out = out_path / "trace.json"
        with trace_out.open("w", encoding="utf-8") as handle:
            json.dump(_trace_to_dict(bundle.trace), handle, indent=2, sort_keys=True)

    return bundle


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize run data into standard files.")
    parser.add_argument("--in", dest="in_path", required=True, help="Run directory or results .npz")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    parser.add_argument("--name", default=None, help="Optional run name override")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    normalize_run(args.in_path, args.out_dir, name=args.name)


if __name__ == "__main__":
    main()
