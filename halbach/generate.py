from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from halbach.radial_profile import build_radial_count_per_layer, build_ring_active_mask
from halbach.run_types import RunResults


def generate_halbach_initial(
    *,
    N: int,
    R: int,
    end_R: int | None,
    end_layers_per_side: int,
    K: int,
    Lz: float,
    diameter_m: float,
    ring_offset_step_m: float,
) -> RunResults:
    if N <= 0:
        raise ValueError("N must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if end_layers_per_side < 0 or end_layers_per_side > K // 2:
        raise ValueError("end_layers_per_side must be between 0 and K//2")
    if Lz < 0.0:
        raise ValueError("Lz must be non-negative")
    if diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive")
    if ring_offset_step_m < 0.0:
        raise ValueError("ring_offset_step_m must be non-negative")

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float64)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if K == 1:
        z_layers = np.array([0.0], dtype=np.float64)
    else:
        z_layers = np.linspace(-Lz / 2.0, Lz / 2.0, K, dtype=np.float64)

    radial_count_per_layer = build_radial_count_per_layer(
        K,
        R,
        end_R=end_R,
        end_layers_per_side=end_layers_per_side,
    )
    R_max = int(np.max(radial_count_per_layer))

    if R_max == 1:
        ring_offsets = np.array([0.0], dtype=np.float64)
    else:
        ring_offsets = np.arange(R_max, dtype=np.float64) * ring_offset_step_m

    alphas = np.zeros((R_max, K), dtype=np.float64)
    r_bases = np.full(K, diameter_m / 2.0, dtype=np.float64)
    ring_active_mask = build_ring_active_mask(radial_count_per_layer, R_max)

    return RunResults(
        alphas=alphas,
        r_bases=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        extras=dict(
            radial_count_per_layer=np.asarray(radial_count_per_layer, dtype=np.int_),
            ring_active_mask=np.asarray(ring_active_mask, dtype=np.bool_),
        ),
    )


def write_run(
    out_dir: str | Path,
    results: RunResults,
    *,
    name: str,
    schema_version: int,
    generator_params: dict[str, Any],
    description: str,
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_path = out_path / "results.npz"
    save_payload: dict[str, Any] = dict(
        alphas_opt=results.alphas,
        r_bases_opt=results.r_bases,
        theta=results.theta,
        sin2th=results.sin2th,
        cth=results.cth,
        sth=results.sth,
        z_layers=results.z_layers,
        ring_offsets=results.ring_offsets,
        alphas=results.alphas,
        r_bases=results.r_bases,
    )
    save_payload.update(results.extras)
    np.savez_compressed(results_path, **save_payload)

    meta = dict(
        schema_version=int(schema_version),
        name=name,
        created_at=datetime.now(UTC).isoformat(),
        description=description,
        generator=generator_params,
    )
    meta_path = out_path / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return results_path, meta_path


__all__ = ["generate_halbach_initial", "write_run"]
