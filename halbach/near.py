"""
Near-graph utilities.

Example:
    >>> from halbach.near import NearWindow, build_near_graph
    >>> graph = build_near_graph(R=1, K=4, N=8, window=NearWindow(wr=0, wz=1, wphi=1))
    >>> graph.nbr_idx.shape
    (32, 8)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from halbach.types import Geometry

Int32Array: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class NearWindow:
    wr: int = 0
    wz: int = 1
    wphi: int = 2

    def __post_init__(self) -> None:
        if self.wr < 0 or self.wz < 0 or self.wphi < 0:
            raise ValueError("NearWindow values must be >= 0")


@dataclass(frozen=True)
class NearGraph:
    R: int
    K: int
    N: int
    window: NearWindow
    deg_max: int
    nbr_idx: Int32Array
    nbr_mask: BoolArray


def flatten_index(r: int, k: int, n: int, R: int, K: int, N: int) -> int:
    """Flatten (r,k,n) into a single index with order ((r*K+k)*N+n)."""
    return (r * K + k) * N + n


def unflatten_index(idx: int, R: int, K: int, N: int) -> tuple[int, int, int]:
    """Inverse of flatten_index."""
    if idx < 0 or idx >= R * K * N:
        raise ValueError("idx out of range")
    rk = idx // N
    n = idx % N
    r = rk // K
    k = rk % K
    return r, k, n


def build_near_graph(R: int, K: int, N: int, window: NearWindow) -> NearGraph:
    return _build_near_graph_cached(R, K, N, window.wr, window.wz, window.wphi)


def get_near_graph_from_geom(geom: Geometry, window: NearWindow) -> NearGraph:
    return build_near_graph(int(geom.R), int(geom.K), int(geom.N), window)


@cache
def _build_near_graph_cached(R: int, K: int, N: int, wr: int, wz: int, wphi: int) -> NearGraph:
    window = NearWindow(wr=wr, wz=wz, wphi=wphi)
    deg_max = (2 * wr + 1) * (2 * wz + 1) * (2 * wphi + 1) - 1
    M = R * K * N

    nbr_idx = np.zeros((M, deg_max), dtype=np.int32)
    nbr_mask = np.zeros((M, deg_max), dtype=bool)

    for r in range(R):
        for k in range(K):
            for n in range(N):
                src = flatten_index(r, k, n, R, K, N)
                count = 0
                for dr in range(-wr, wr + 1):
                    r2 = r + dr
                    if r2 < 0 or r2 >= R:
                        continue
                    for dk in range(-wz, wz + 1):
                        k2 = k + dk
                        if k2 < 0 or k2 >= K:
                            continue
                        for dn in range(-wphi, wphi + 1):
                            n2 = (n + dn) % N
                            dst = flatten_index(r2, k2, n2, R, K, N)
                            if dst == src:
                                continue
                            nbr_idx[src, count] = dst
                            nbr_mask[src, count] = True
                            count += 1

    return NearGraph(
        R=R,
        K=K,
        N=N,
        window=window,
        deg_max=deg_max,
        nbr_idx=nbr_idx,
        nbr_mask=nbr_mask,
    )


__all__ = [
    "NearWindow",
    "NearGraph",
    "flatten_index",
    "unflatten_index",
    "build_near_graph",
    "get_near_graph_from_geom",
]
