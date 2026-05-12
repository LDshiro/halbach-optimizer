from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.field_eval import (
    build_fixed_model_arrays,
    evaluate_fixed_placement,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import MagnetError, Placement, VirtualMagnet
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"field_eval_run_N{N}_R{R}_K{K}"
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
        name="plan-c-field-eval-test",
        schema_version=1,
        generator_params={},
        description="plan c field eval test run",
    )
    return load_run(run_dir)


def _zero_magnets(count: int) -> list[VirtualMagnet]:
    zero = MagnetError(
        epsilon_parallel=0.0,
        delta_perp_1=0.0,
        delta_perp_2=0.0,
    )
    return [
        VirtualMagnet(
            magnet_id=idx,
            true_error=zero,
            measured_error=zero,
            quality=1.0,
        )
        for idx in range(count)
    ]


def _o0_placements(slots: list, magnets: list[VirtualMagnet]) -> list[Placement]:
    return [
        Placement(
            slot_flat_id=slot.slot_flat_id,
            magnet_id=magnet.magnet_id,
            orientation_id="O0",
            insert_order=idx,
            decision_engine="test",
        )
        for idx, (slot, magnet) in enumerate(zip(slots, magnets, strict=True))
    ]


def test_zero_variation_o0_builds_nominal_fixed_model(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _zero_magnets(len(slots))
    placements = _o0_placements(slots, magnets)

    r0_flat, m_flat = build_fixed_model_arrays(slots, magnets, placements)

    assert r0_flat.shape == (len(slots), 3)
    assert m_flat.shape == (len(slots), 3)
    for row, slot in enumerate(sorted(slots, key=lambda item: item.slot_flat_id)):
        np.testing.assert_allclose(r0_flat[row], slot.center_m)
        np.testing.assert_allclose(m_flat[row], slot.nominal_u, atol=1e-12)


def test_zero_variation_o0_evaluates_vector_homogeneity_metrics(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _zero_magnets(len(slots))
    placements = _o0_placements(slots, magnets)
    pts = sample_sphere_surface_fibonacci(24, 0.02, seed=0)

    result = evaluate_fixed_placement(slots, magnets, placements, pts)

    assert result.B.shape == (24, 3)
    assert result.B0.shape == (3,)
    assert result.metrics.B0_norm > 0.0
    assert result.metrics.J_vector >= 0.0
    assert np.isfinite(result.metrics.rms_homogeneity_ppm)
    assert result.metrics.max_homogeneity_ppm >= result.metrics.rms_homogeneity_ppm


def test_field_eval_rejects_duplicate_slot_placement(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _zero_magnets(len(slots))
    placements = _o0_placements(slots, magnets)
    placements[1] = Placement(
        slot_flat_id=placements[0].slot_flat_id,
        magnet_id=placements[1].magnet_id,
        orientation_id="O0",
    )

    with pytest.raises(ValueError, match="duplicate slot"):
        build_fixed_model_arrays(slots, magnets, placements)


def test_field_eval_rejects_incomplete_placement(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _zero_magnets(len(slots))
    placements = _o0_placements(slots, magnets)

    with pytest.raises(ValueError, match="every slot"):
        build_fixed_model_arrays(slots, magnets, placements[:-1])


def test_field_eval_rejects_too_small_B0_norm(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    magnets = _zero_magnets(len(slots))
    placements = _o0_placements(slots, magnets)
    pts = sample_sphere_surface_fibonacci(8, 0.02, seed=0)

    with pytest.raises(ValueError, match="B0_norm"):
        evaluate_fixed_placement(slots, magnets, placements, pts, factor=0.0)
