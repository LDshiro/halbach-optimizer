from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halbach.solvers.types import SolveTrace
from halbach.types import FloatArray, Geometry


@dataclass(frozen=True)
class RunResults:
    alphas: FloatArray
    r_bases: FloatArray
    theta: FloatArray
    sin2th: FloatArray
    cth: FloatArray
    sth: FloatArray
    z_layers: FloatArray
    ring_offsets: FloatArray
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunBundle:
    name: str
    run_dir: Path
    results_path: Path
    meta_path: Path | None
    trace_path: Path | None
    results: RunResults
    meta: dict[str, Any]
    geometry: Geometry
    trace: SolveTrace | None


__all__ = ["RunBundle", "RunResults"]
