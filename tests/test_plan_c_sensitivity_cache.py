from pathlib import Path

import numpy as np

from halbach.assembly.sensitivity_cache import (
    load_or_compute_sensitivity_table,
    sensitivity_cache_key,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path) -> RunBundle:
    run_dir = tmp_path / "sensitivity_cache_run"
    results = generate_halbach_initial(
        N=4,
        R=1,
        end_R=None,
        end_layers_per_side=0,
        K=3,
        Lz=0.16,
        diameter_m=0.3,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-sensitivity-cache-test",
        schema_version=1,
        generator_params={},
        description="plan c sensitivity cache test run",
    )
    return load_run(run_dir)


def test_load_or_compute_sensitivity_table_reuses_matching_cache(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(3, 0.02, seed=0)
    cache_dir = tmp_path / "cache"

    first = load_or_compute_sensitivity_table(
        cache_dir,
        slots,
        pts,
        finite_difference_step=1.0e-6,
        metadata={"case": "first"},
    )
    second = load_or_compute_sensitivity_table(
        cache_dir,
        slots,
        pts,
        finite_difference_step=1.0e-6,
        metadata={"case": "second"},
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_path == second.cache_path
    assert first.cache_path.exists()
    assert first.cache_key == second.cache_key
    np.testing.assert_allclose(first.table.C, second.table.C)
    assert second.table.metadata["sensitivity_cache_key"] == first.cache_key


def test_sensitivity_cache_key_changes_with_roi_points(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts_a = sample_sphere_surface_fibonacci(3, 0.02, seed=0)
    pts_b = sample_sphere_surface_fibonacci(4, 0.02, seed=0)

    assert sensitivity_cache_key(slots, pts_a) != sensitivity_cache_key(slots, pts_b)


def test_load_or_compute_sensitivity_table_recovers_from_corrupt_cache(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(3, 0.02, seed=0)
    cache_dir = tmp_path / "cache"
    first = load_or_compute_sensitivity_table(cache_dir, slots, pts)

    first.cache_path.write_bytes(b"not an npz")
    recovered = load_or_compute_sensitivity_table(cache_dir, slots, pts)

    assert not recovered.cache_hit
    assert recovered.cache_path == first.cache_path
    assert recovered.cache_path.stat().st_size > len(b"not an npz")
    np.testing.assert_allclose(first.table.C, recovered.table.C)
