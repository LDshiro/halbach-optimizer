from dataclasses import replace
from pathlib import Path

import numpy as np

from halbach.generate import generate_halbach_initial, write_run
from halbach.radial_profile import (
    build_radial_count_per_layer,
    build_ring_active_mask,
    radial_profile_from_results,
    radial_profile_from_run,
)
from halbach.run_io import load_run


def test_build_radial_count_per_layer_end_only_profile() -> None:
    counts = build_radial_count_per_layer(24, 2, 3, 3)
    expected = np.array([3, 3, 3] + [2] * 18 + [3, 3, 3], dtype=np.int_)
    np.testing.assert_array_equal(counts, expected)

    mask = build_ring_active_mask(counts, 3)
    assert mask.shape == (3, 24)
    np.testing.assert_array_equal(mask[:, 0], np.array([True, True, True]))
    np.testing.assert_array_equal(mask[:, 5], np.array([True, True, False]))


def test_radial_profile_uniform_case() -> None:
    counts = build_radial_count_per_layer(10, 2, 2, 0)
    np.testing.assert_array_equal(counts, np.full(10, 2, dtype=np.int_))
    mask = build_ring_active_mask(counts, 2)
    assert bool(np.all(mask))


def test_radial_profile_from_run_fallback_and_extras(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    results = generate_halbach_initial(
        N=8,
        R=2,
        K=6,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
        end_R=3,
        end_layers_per_side=2,
    )
    write_run(
        run_dir,
        results,
        name="profile-test",
        schema_version=1,
        generator_params={"N": 8, "R": 2, "K": 6, "end_R": 3, "end_layers_per_side": 2},
        description="test",
    )
    meta_path = run_dir / "meta.json"
    meta_path.write_text(
        '{"radial_profile": {"mode": "end-only", "base_R": 2, "end_R": 3, "end_layers_per_side": 2, "R_max": 3}}',
        encoding="utf-8",
    )
    run = load_run(run_dir)
    profile = radial_profile_from_run(run)
    np.testing.assert_array_equal(profile.radial_count_per_layer, np.array([3, 3, 2, 2, 3, 3]))
    assert profile.ring_active_mask.shape == (3, 6)
    assert profile.mode == "end-only"


def test_radial_profile_from_results_reconstructs_counts_from_mask_only() -> None:
    results = generate_halbach_initial(
        N=8,
        R=2,
        K=6,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
        end_R=3,
        end_layers_per_side=2,
    )
    mask_only_results = replace(
        results,
        extras={
            "ring_active_mask": np.asarray(results.extras["ring_active_mask"], dtype=np.bool_),
        },
    )

    profile = radial_profile_from_results(mask_only_results)

    np.testing.assert_array_equal(
        profile.radial_count_per_layer,
        np.asarray(results.extras["radial_count_per_layer"], dtype=np.int_),
    )
    np.testing.assert_array_equal(
        profile.ring_active_mask,
        np.asarray(results.extras["ring_active_mask"], dtype=np.bool_),
    )
    assert profile.mode == "end-only"
