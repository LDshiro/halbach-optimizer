from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.slots import build_assembly_slots, flatten_slot_id
from halbach.generate import generate_halbach_initial, write_run
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(
    tmp_path: Path,
    *,
    N: int = 8,
    R: int = 1,
    K: int = 4,
    end_R: int | None = None,
    end_layers_per_side: int = 0,
) -> RunBundle:
    run_dir = tmp_path / "run"
    results = generate_halbach_initial(
        N=N,
        R=R,
        end_R=end_R,
        end_layers_per_side=end_layers_per_side,
        K=K,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-test",
        schema_version=1,
        generator_params={},
        description="plan c test run",
    )
    return load_run(run_dir)


def test_flatten_slot_id_uses_rkn_c_order() -> None:
    assert flatten_slot_id(0, 0, 0, K=4, N=8) == 0
    assert flatten_slot_id(0, 1, 0, K=4, N=8) == 8
    assert flatten_slot_id(1, 0, 0, K=4, N=8) == 32
    assert flatten_slot_id(1, 2, 3, K=4, N=8) == ((1 * 4 + 2) * 8 + 3)
    with pytest.raises(ValueError):
        flatten_slot_id(0, 4, 0, K=4, N=8)


def test_build_assembly_slots_uniform_run(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=8, R=2, K=4)
    slots = build_assembly_slots(run)

    assert len(slots) == 2 * 4 * 8
    first = slots[0]
    assert first.slot_flat_id == 0
    assert first.ring_id == 0
    assert first.layer_id == 0
    assert first.theta_id == 0
    assert first.physical_slot_number == 1
    assert first.center_m.shape == (3,)
    assert first.nominal_u.shape == (3,)

    sample = next(
        slot
        for slot in slots
        if slot.ring_id == 1 and slot.layer_id == 2 and slot.theta_id == 3
    )
    expected_flat = (1 * run.geometry.K + 2) * run.geometry.N + 3
    assert sample.slot_flat_id == expected_flat
    assert sample.physical_slot_number == 4

    rho = run.results.r_bases[2] + run.geometry.ring_offsets[1]
    expected_center = np.array(
        [
            rho * run.geometry.cth[3],
            rho * run.geometry.sth[3],
            run.geometry.z_layers[2],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(sample.center_m, expected_center)


def test_build_assembly_slots_skips_inactive_nonuniform_profile(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=8, R=1, end_R=2, end_layers_per_side=1, K=4)
    slots = build_assembly_slots(run)

    assert len(slots) == int((2 + 1 + 1 + 2) * 8)
    inactive = [
        slot
        for slot in slots
        if slot.ring_id == 1 and slot.layer_id in (1, 2)
    ]
    assert inactive == []
    assert any(slot.ring_id == 1 and slot.layer_id == 0 for slot in slots)
    assert any(slot.ring_id == 1 and slot.layer_id == 3 for slot in slots)


def test_mirror_pair_id_matches_mirror_layers_and_center_is_none(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=8, R=1, K=5)
    slots = build_assembly_slots(run)
    by_layer = {
        slot.layer_id: slot
        for slot in slots
        if slot.ring_id == 0 and slot.theta_id == 0
    }

    assert by_layer[0].mirror_pair_id == by_layer[4].mirror_pair_id
    assert by_layer[1].mirror_pair_id == by_layer[3].mirror_pair_id
    assert by_layer[2].mirror_pair_id is None
