from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from halbach.run_types import RunBundle, RunResults

RadialProfileMode = Literal["uniform", "end-only"]


@dataclass(frozen=True)
class RadialProfile:
    radial_count_per_layer: NDArray[np.int_]
    ring_active_mask: NDArray[np.bool_]
    base_r: int
    end_r: int
    end_layers_per_side: int
    r_max: int
    mode: RadialProfileMode


def build_radial_count_per_layer(
    K: int,
    R: int,
    end_R: int | None = None,
    end_layers_per_side: int = 0,
) -> NDArray[np.int_]:
    if K <= 0:
        raise ValueError("K must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    end_R_val = R if end_R is None else int(end_R)
    if end_R_val < R:
        raise ValueError("end_R must be >= R")
    if end_layers_per_side < 0 or end_layers_per_side > K // 2:
        raise ValueError("end_layers_per_side must be between 0 and K//2")

    counts = np.full(K, int(R), dtype=np.int_)
    if end_R_val == R or end_layers_per_side == 0:
        return counts
    counts[:end_layers_per_side] = int(end_R_val)
    counts[K - end_layers_per_side :] = int(end_R_val)
    return counts


def build_ring_active_mask(
    radial_count_per_layer: NDArray[np.int_],
    R_max: int,
) -> NDArray[np.bool_]:
    counts = np.asarray(radial_count_per_layer, dtype=np.int_).reshape(-1)
    if R_max <= 0:
        raise ValueError("R_max must be positive")
    if np.any(counts < 0):
        raise ValueError("radial_count_per_layer must be >= 0")
    if np.any(counts > R_max):
        raise ValueError("radial_count_per_layer must be <= R_max")

    ring_ids = np.arange(R_max, dtype=np.int_)[:, None]
    mask = ring_ids < counts[None, :]
    return np.asarray(mask, dtype=np.bool_)


def broadcast_ring_active_mask(
    ring_active_mask: NDArray[np.bool_],
    N: int,
) -> NDArray[np.bool_]:
    mask = np.asarray(ring_active_mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("ring_active_mask must be 2D")
    if N <= 0:
        raise ValueError("N must be positive")
    return np.broadcast_to(mask[:, :, None], (mask.shape[0], mask.shape[1], N))


def flatten_ring_active_mask(
    ring_active_mask: NDArray[np.bool_],
    N: int,
) -> NDArray[np.bool_]:
    return np.asarray(broadcast_ring_active_mask(ring_active_mask, N).reshape(-1), dtype=np.bool_)


def _profile_from_meta(
    meta: dict[str, Any], *, R: int, K: int
) -> tuple[int, int, int, RadialProfileMode]:
    raw = meta.get("radial_profile", {})
    if not isinstance(raw, dict):
        return R, R, 0, "uniform"
    base_R = int(raw.get("base_R", R))
    end_R = int(raw.get("end_R", base_R))
    end_layers = int(raw.get("end_layers_per_side", 0))
    mode_raw = raw.get("mode", "uniform")
    mode: RadialProfileMode = "end-only" if mode_raw == "end-only" else "uniform"
    counts = build_radial_count_per_layer(K, base_R, end_R, end_layers)
    if counts.shape != (K,):
        raise ValueError("radial profile metadata is inconsistent")
    return base_R, end_R, end_layers, mode


def radial_profile_from_results(
    results: RunResults,
    meta: dict[str, Any] | None = None,
) -> RadialProfile:
    R_max, K = results.alphas.shape
    extras = results.extras

    counts_arr: NDArray[np.int_] | None = None
    if "radial_count_per_layer" in extras:
        counts_arr = np.asarray(extras["radial_count_per_layer"], dtype=np.int_).reshape(-1)
        if counts_arr.shape != (K,):
            raise ValueError(
                "radial_count_per_layer shape " f"{counts_arr.shape} does not match expected {(K,)}"
            )

    mask_arr: NDArray[np.bool_] | None = None
    if "ring_active_mask" in extras:
        mask_arr = np.asarray(extras["ring_active_mask"], dtype=np.bool_)
        if mask_arr.shape != (R_max, K):
            raise ValueError(
                f"ring_active_mask shape {mask_arr.shape} does not match expected {(R_max, K)}"
            )

    meta_dict = meta if meta is not None else {}
    base_R_meta, end_R_meta, end_layers_meta, mode_meta = _profile_from_meta(
        meta_dict, R=R_max, K=K
    )

    if counts_arr is None and mask_arr is None:
        counts_arr = build_radial_count_per_layer(K, base_R_meta, end_R_meta, end_layers_meta)
        mode = "uniform" if np.all(counts_arr == counts_arr[0]) else mode_meta
        mask_arr = build_ring_active_mask(counts_arr, R_max)
        return RadialProfile(
            radial_count_per_layer=np.asarray(counts_arr, dtype=np.int_),
            ring_active_mask=np.asarray(mask_arr, dtype=np.bool_),
            base_r=int(base_R_meta),
            end_r=int(end_R_meta),
            end_layers_per_side=int(end_layers_meta),
            r_max=int(R_max),
            mode=mode,
        )

    if counts_arr is None:
        assert mask_arr is not None
        counts_arr = np.sum(mask_arr, axis=0, dtype=np.int_)
    if mask_arr is None:
        mask_arr = build_ring_active_mask(counts_arr, R_max)

    if np.any(np.sum(mask_arr, axis=0, dtype=np.int_) != counts_arr):
        raise ValueError("ring_active_mask does not match radial_count_per_layer")

    inferred_mode: RadialProfileMode = (
        "uniform" if np.all(counts_arr == counts_arr[0]) else "end-only"
    )
    base_R = int(np.min(counts_arr)) if counts_arr.size else int(R_max)
    end_R = int(np.max(counts_arr)) if counts_arr.size else int(R_max)
    end_layers = 0
    if inferred_mode == "end-only" and counts_arr.size > 0:
        k = 0
        while k < K // 2 and counts_arr[k] == end_R:
            k += 1
        end_layers = k
    return RadialProfile(
        radial_count_per_layer=np.asarray(counts_arr, dtype=np.int_),
        ring_active_mask=np.asarray(mask_arr, dtype=np.bool_),
        base_r=base_R,
        end_r=end_R,
        end_layers_per_side=end_layers,
        r_max=int(R_max),
        mode=inferred_mode,
    )


def radial_profile_from_run(run: RunBundle) -> RadialProfile:
    return radial_profile_from_results(run.results, run.meta)


__all__ = [
    "RadialProfile",
    "RadialProfileMode",
    "broadcast_ring_active_mask",
    "build_radial_count_per_layer",
    "build_ring_active_mask",
    "flatten_ring_active_mask",
    "radial_profile_from_results",
    "radial_profile_from_run",
]
