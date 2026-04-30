from pathlib import Path

import numpy as np

from halbach.assembly.field_eval import evaluate_fixed_placement
from halbach.assembly.sensitivity import (
    compute_sensitivity_table,
    flatten_field_residual,
    sensitivity_contribution,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import MagnetError, Placement, VirtualMagnet
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"sensitivity_run_N{N}_R{R}_K{K}"
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
        name="plan-c-sensitivity-test",
        schema_version=1,
        generator_params={},
        description="plan c sensitivity test run",
    )
    return load_run(run_dir)


def _zero_error() -> MagnetError:
    return MagnetError(
        epsilon_parallel=0.0,
        delta_perp_1=0.0,
        delta_perp_2=0.0,
    )


def _magnets_with_one_error(
    slots: list,
    *,
    slot_flat_id: int,
    error: MagnetError,
) -> list[VirtualMagnet]:
    magnets: list[VirtualMagnet] = []
    for idx, slot in enumerate(slots):
        slot_error = error if slot.slot_flat_id == slot_flat_id else _zero_error()
        magnets.append(
            VirtualMagnet(
                magnet_id=idx,
                true_error=slot_error,
                measured_error=slot_error,
                quality=1.0,
            )
        )
    return magnets


def _placements_for_one_orientation(
    slots: list,
    *,
    slot_flat_id: int,
    orientation_id: str,
) -> list[Placement]:
    placements: list[Placement] = []
    for idx, slot in enumerate(slots):
        placements.append(
            Placement(
                slot_flat_id=slot.slot_flat_id,
                magnet_id=idx,
                orientation_id=orientation_id if slot.slot_flat_id == slot_flat_id else "O0",
                insert_order=idx,
                decision_engine="test",
            )
        )
    return placements


def test_sensitivity_table_shape_and_zero_contribution(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=0)

    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)

    assert table.C.shape == (len(slots), 4, 15, 3)
    assert table.roi_points.shape == (5, 3)
    assert table.normalization_b0.shape == (3,)
    assert table.orientation_id == ("O0", "O90", "O180", "O270")
    contribution = sensitivity_contribution(
        table,
        slots[0].slot_flat_id,
        "O0",
        _zero_error(),
    )
    np.testing.assert_allclose(contribution, np.zeros(15, dtype=np.float64))


def test_sensitivity_matches_direct_field_difference_for_small_error(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(7, 0.02, seed=1)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    target_slot_id = slots[3].slot_flat_id
    orientation_id = "O90"
    error = MagnetError(
        epsilon_parallel=1.0e-5,
        delta_perp_1=-2.0e-5,
        delta_perp_2=1.5e-5,
    )

    baseline = evaluate_fixed_placement(
        slots,
        _magnets_with_one_error(slots, slot_flat_id=target_slot_id, error=_zero_error()),
        _placements_for_one_orientation(slots, slot_flat_id=target_slot_id, orientation_id="O0"),
        pts,
    )
    perturbed = evaluate_fixed_placement(
        slots,
        _magnets_with_one_error(slots, slot_flat_id=target_slot_id, error=error),
        _placements_for_one_orientation(
            slots,
            slot_flat_id=target_slot_id,
            orientation_id=orientation_id,
        ),
        pts,
    )

    direct = flatten_field_residual(perturbed.B, perturbed.B0) - flatten_field_residual(
        baseline.B,
        baseline.B0,
    )
    linear = sensitivity_contribution(table, target_slot_id, orientation_id, error)

    np.testing.assert_allclose(linear, direct, rtol=3.0e-5, atol=1.0e-13)


def test_o180_transverse_columns_are_sign_reversed(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(4, 0.02, seed=2)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    idx_o0 = table.orientation_id.index("O0")
    idx_o180 = table.orientation_id.index("O180")

    np.testing.assert_allclose(
        table.C[:, idx_o180, :, 0],
        table.C[:, idx_o0, :, 0],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        table.C[:, idx_o180, :, 1],
        -table.C[:, idx_o0, :, 1],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        table.C[:, idx_o180, :, 2],
        -table.C[:, idx_o0, :, 2],
        rtol=1e-8,
        atol=1e-12,
    )
