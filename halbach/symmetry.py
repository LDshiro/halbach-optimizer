from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["MirrorX0", "build_mirror_x0", "expand_delta_phi"]


@dataclass(frozen=True)
class MirrorX0:
    mirror_idx: NDArray[np.int_]
    rep_idx: NDArray[np.int_]
    fixed_idx: NDArray[np.int_]
    basis: NDArray[np.float64]


def build_mirror_x0(N: int) -> MirrorX0:
    if N % 2 != 0:
        raise ValueError("N must be even for x=0 mirror symmetry")

    idx = np.arange(N, dtype=np.int_)
    mirror_idx = (N // 2 - idx) % N
    if not np.array_equal(mirror_idx[mirror_idx], idx):
        raise ValueError("mirror mapping is not involutive")

    fixed_mask = mirror_idx == idx
    fixed_idx = idx[fixed_mask]
    rep_idx = idx[idx < mirror_idx]

    basis = np.zeros((rep_idx.size, N), dtype=np.float64)
    for t, i in enumerate(rep_idx):
        j = int(mirror_idx[i])
        basis[t, i] = 1.0
        basis[t, j] = -1.0

    return MirrorX0(
        mirror_idx=mirror_idx,
        rep_idx=rep_idx,
        fixed_idx=fixed_idx,
        basis=basis,
    )


def expand_delta_phi(
    delta_rep: NDArray[np.float64], basis: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(delta_rep @ basis, dtype=np.float64)
