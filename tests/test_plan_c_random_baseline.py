from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.simulation import build_random_placements, run_random_baseline
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"random_baseline_run_N{N}_R{R}_K{K}"
    results = generate_halbach_initial(
        N=N,
        R=R,
        end_R=None,
        end_layers_per_side=0,
        K=K,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-random-baseline-test",
        schema_version=1,
        generator_params={},
        description="plan c random baseline test run",
    )
    return load_run(run_dir)


def _magnets(count: int):
    return generate_virtual_magnets(
        count=count,
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0007,
        measurement_noise=None,
    )


def test_random_placements_are_seed_reproducible(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _magnets(len(slots))

    placements_a = build_random_placements(slots, magnets, seed=5)
    placements_b = build_random_placements(slots, magnets, seed=5)

    assert placements_a == placements_b
    assert {placement.slot_flat_id for placement in placements_a} == {
        slot.slot_flat_id for slot in slots
    }
    assert {placement.magnet_id for placement in placements_a} == {
        magnet.magnet_id for magnet in magnets
    }
    assert {placement.orientation_id for placement in placements_a} == {"O0"}


def test_random_discrete4_orientation_mode_is_reproducible(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _magnets(len(slots))

    placements_a = build_random_placements(
        slots,
        magnets,
        seed=8,
        orientation_mode="random_discrete4",
    )
    placements_b = build_random_placements(
        slots,
        magnets,
        seed=8,
        orientation_mode="random_discrete4",
    )

    assert placements_a == placements_b
    assert {placement.orientation_id for placement in placements_a} <= {
        "O0",
        "O90",
        "O180",
        "O270",
    }


def test_random_baseline_small_run_completes(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=8, R=1, K=4)
    slots = build_assembly_slots(run)
    magnets = _magnets(len(slots))
    pts = sample_sphere_surface_fibonacci(16, 0.02, seed=11)

    result = run_random_baseline(
        slots,
        magnets,
        pts,
        seed=10,
        orientation_mode="random_discrete4",
    )

    assert len(result.placements) == len(slots)
    assert result.evaluation.B.shape == (16, 3)
    assert result.evaluation.metrics.B0_norm > 0.0
    assert np.isfinite(result.evaluation.metrics.rms_homogeneity_ppm)


def test_random_baseline_rejects_slot_magnet_count_mismatch(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _magnets(len(slots) - 1)

    with pytest.raises(ValueError, match="same number"):
        build_random_placements(slots, magnets, seed=1)


def test_random_baseline_rejects_unknown_orientation_mode(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _magnets(len(slots))

    with pytest.raises(ValueError, match="orientation_mode"):
        build_random_placements(slots, magnets, seed=1, orientation_mode="bad-mode")
