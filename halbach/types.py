from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class Geometry:
    theta: FloatArray
    sin2: FloatArray
    cth: FloatArray
    sth: FloatArray
    z_layers: FloatArray
    ring_offsets: FloatArray
    N: int
    K: int
    R: int
    dz: float
    Lz: float


__all__ = ["Geometry", "FloatArray"]
