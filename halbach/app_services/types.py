from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.types import FloatArray

Plane = Literal["xy", "xz", "yz"]
MagModelEval = Literal["auto", "fixed", "self-consistent-easy-axis"]
SceneRanges: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class RunCandidate:
    path: str
    label: str


@dataclass(frozen=True)
class RunSummary:
    name: str
    run_path: str
    results_path: str
    meta_path: str | None
    trace_path: str | None
    framework: str
    geometry_summary: dict[str, object]
    key_stats: dict[str, object]
    magnetization_debug: dict[str, object]


@dataclass(frozen=True)
class Map2DRequest:
    run_path: str
    roi_r: float = 0.14
    step: float = 0.001
    mag_model_eval: MagModelEval = "auto"
    sc_cfg_override: dict[str, Any] | None = None
    plane: Plane = "xy"
    coord0: float = 0.0


@dataclass(frozen=True)
class Map2DPayload:
    run_path: str
    xs: FloatArray
    ys: FloatArray
    ppm: FloatArray
    mask: NDArray[np.bool_]
    B0_T: float
    plane: Plane
    coord0: float
    summary_stats: dict[str, float]
    magnetization_debug: dict[str, object]


@dataclass(frozen=True)
class Scene3DRequest:
    primary_path: str
    secondary_path: str | None = None
    stride: int = 1
    hide_x_negative: bool = False


@dataclass(frozen=True)
class Scene3DRunPayload:
    run_path: str
    run_name: str
    centers: FloatArray
    direction_vectors: FloatArray
    ring_ids: NDArray[np.int_]
    layer_ids: NDArray[np.int_]


@dataclass(frozen=True)
class Scene3DPayload:
    primary: Scene3DRunPayload
    secondary: Scene3DRunPayload | None
    scene_ranges: SceneRanges
    view_metadata: dict[str, object]


__all__ = [
    "MagModelEval",
    "Map2DPayload",
    "Map2DRequest",
    "Plane",
    "RunCandidate",
    "RunSummary",
    "Scene3DPayload",
    "Scene3DRequest",
    "Scene3DRunPayload",
    "SceneRanges",
]
