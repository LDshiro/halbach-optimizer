from pathlib import Path

import pytest

from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.work_units import (
    assign_work_unit_ids,
    build_work_units,
    outer_to_inner_layer_order,
)
from halbach.generate import generate_halbach_initial, write_run
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(
    tmp_path: Path,
    *,
    N: int,
    R: int,
    K: int,
) -> RunBundle:
    run_dir = tmp_path / f"run_N{N}_R{R}_K{K}"
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
        name="plan-c-work-unit-test",
        schema_version=1,
        generator_params={},
        description="plan c work unit test run",
    )
    return load_run(run_dir)


def _assert_unit_coverage(slot_ids: set[int], unit_slot_ids: list[int]) -> None:
    assert len(unit_slot_ids) == len(set(unit_slot_ids))
    assert set(unit_slot_ids) == slot_ids


def test_all_slots_work_unit_covers_all_slots_once(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=8, R=1, K=4)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "all_slots")

    assert len(units) == 1
    assert units[0].work_unit_id == "W_ALL"
    assert units[0].mode == "all_slots"
    _assert_unit_coverage({slot.slot_flat_id for slot in slots}, list(units[0].slot_flat_ids))

    assigned = assign_work_unit_ids(slots, units)
    assert all(slot.work_unit_id == "W_ALL" for slot in assigned)
    assert all(slot.work_unit_id == "" for slot in slots)


def test_single_physical_ring_groups_by_layer_and_ring(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=6, R=2, K=4)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "single_physical_ring")

    assert len(units) == 2 * 4
    assert units[0].work_unit_id == "W_K000_R000"
    assert units[1].work_unit_id == "W_K000_R001"
    assert all(unit.mode == "single_physical_ring" for unit in units)
    assert all(len(unit.slot_flat_ids) == 6 for unit in units)
    _assert_unit_coverage(
        {slot.slot_flat_id for slot in slots},
        [slot_id for unit in units for slot_id in unit.slot_flat_ids],
    )


def test_outer_to_inner_layer_order_even_and_odd() -> None:
    assert outer_to_inner_layer_order(8) == (0, 7, 1, 6, 2, 5, 3, 4)
    assert outer_to_inner_layer_order(7) == (0, 6, 1, 5, 2, 4, 3)
    with pytest.raises(ValueError):
        outer_to_inner_layer_order(0)


def test_ring_by_ring_outer_to_inner_even_layers(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=8)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "ring_by_ring_outer_to_inner")

    assert [unit.mode for unit in units] == ["ring_by_ring_outer_to_inner"] * 8
    assert [unit.work_unit_id for unit in units] == [
        "W_K000_R000",
        "W_K007_R000",
        "W_K001_R000",
        "W_K006_R000",
        "W_K002_R000",
        "W_K005_R000",
        "W_K003_R000",
        "W_K004_R000",
    ]
    assert [unit.label for unit in units][0] == "outer-to-inner layer 0, ring 0"
    _assert_unit_coverage(
        {slot.slot_flat_id for slot in slots},
        [slot_id for unit in units for slot_id in unit.slot_flat_ids],
    )


def test_ring_by_ring_outer_to_inner_odd_layers(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=7)
    units = build_work_units(build_assembly_slots(run), "ring_by_ring_outer_to_inner")

    assert [unit.work_unit_id for unit in units] == [
        "W_K000_R000",
        "W_K006_R000",
        "W_K001_R000",
        "W_K005_R000",
        "W_K002_R000",
        "W_K004_R000",
        "W_K003_R000",
    ]


def test_ring_by_ring_outer_to_inner_orders_rings_within_layer(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=2, K=4)
    units = build_work_units(build_assembly_slots(run), "ring_by_ring_outer_to_inner")

    assert [unit.work_unit_id for unit in units] == [
        "W_K000_R000",
        "W_K000_R001",
        "W_K003_R000",
        "W_K003_R001",
        "W_K001_R000",
        "W_K001_R001",
        "W_K002_R000",
        "W_K002_R001",
    ]


def test_layer_by_layer_outer_to_inner_uses_one_unit_per_layer(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=2, K=4)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "layer_by_layer_outer_to_inner")

    assert [unit.mode for unit in units] == ["layer_by_layer_outer_to_inner"] * 4
    assert [unit.work_unit_id for unit in units] == [
        "W_LAYER000",
        "W_LAYER003",
        "W_LAYER001",
        "W_LAYER002",
    ]
    assert [len(unit.slot_flat_ids) for unit in units] == [8, 8, 8, 8]
    first_layer_slots = sorted(
        [slot for slot in slots if slot.layer_id == 0],
        key=lambda slot: (slot.ring_id, slot.theta_id),
    )
    first_unit_slots = [slot.slot_flat_id for slot in first_layer_slots]
    assert list(units[0].slot_flat_ids) == first_unit_slots
    _assert_unit_coverage(
        {slot.slot_flat_id for slot in slots},
        [slot_id for unit in units for slot_id in unit.slot_flat_ids],
    )


def test_stack_by_stack_outer_to_inner_aliases_layer_units(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=2, K=4)
    slots = build_assembly_slots(run)
    old_units = build_work_units(slots, "layer_by_layer_outer_to_inner")
    stack_units = build_work_units(slots, "stack_by_stack_outer_to_inner")

    assert [unit.mode for unit in stack_units] == ["stack_by_stack_outer_to_inner"] * 4
    assert [unit.work_unit_id for unit in stack_units] == [
        "W_LAYER000",
        "W_LAYER003",
        "W_LAYER001",
        "W_LAYER002",
    ]
    assert [unit.slot_flat_ids for unit in stack_units] == [
        unit.slot_flat_ids for unit in old_units
    ]
    assert "stack 0" in stack_units[0].label


def test_mirror_ring_pair_even_layers(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=6)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "mirror_ring_pair")

    assert [unit.mode for unit in units] == ["mirror_ring_pair"] * 3
    assert [unit.work_unit_id for unit in units] == [
        "W_PAIR000_R000",
        "W_PAIR001_R000",
        "W_PAIR002_R000",
    ]
    assert [len(unit.slot_flat_ids) for unit in units] == [8, 8, 8]
    unit_layer_sets = []
    for unit in units:
        layer_ids = sorted(
            {slot.layer_id for slot in slots if slot.slot_flat_id in set(unit.slot_flat_ids)}
        )
        unit_layer_sets.append(tuple(layer_ids))
    assert unit_layer_sets == [(0, 5), (1, 4), (2, 3)]
    _assert_unit_coverage(
        {slot.slot_flat_id for slot in slots},
        [slot_id for unit in units for slot_id in unit.slot_flat_ids],
    )


def test_mirror_ring_pair_odd_layers_has_center_unit(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=5)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "mirror_ring_pair")

    assert [unit.work_unit_id for unit in units] == [
        "W_PAIR000_R000",
        "W_PAIR001_R000",
        "W_PAIR002_R000",
    ]
    assert [len(unit.slot_flat_ids) for unit in units] == [8, 8, 4]
    assert "center layer 2" in units[-1].label
    unit_layer_sets = []
    for unit in units:
        layer_ids = sorted(
            {slot.layer_id for slot in slots if slot.slot_flat_id in set(unit.slot_flat_ids)}
        )
        unit_layer_sets.append(tuple(layer_ids))
    assert unit_layer_sets == [(0, 4), (1, 3), (2,)]


def test_assign_work_unit_ids_accepts_new_modes(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=5)
    slots = build_assembly_slots(run)

    for mode in (
        "stack_by_stack_outer_to_inner",
        "layer_by_layer_outer_to_inner",
        "ring_by_ring_outer_to_inner",
        "mirror_ring_pair",
    ):
        assigned = assign_work_unit_ids(slots, build_work_units(slots, mode))
        assert all(slot.work_unit_id for slot in assigned)
        assert all(slot.work_unit_id == "" for slot in slots)


def test_ring_group_chunks_physical_rings_in_layer_ring_order(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path, N=4, R=1, K=6)
    slots = build_assembly_slots(run)
    units = build_work_units(slots, "ring_group", ring_group_size=2)

    assert len(units) == 3
    assert [unit.work_unit_id for unit in units] == ["W_G000", "W_G001", "W_G002"]
    assert all(unit.mode == "ring_group" for unit in units)
    assert [len(unit.slot_flat_ids) for unit in units] == [8, 8, 8]

    expected_first = [
        slot.slot_flat_id for slot in slots if (slot.layer_id, slot.ring_id) in ((0, 0), (1, 0))
    ]
    assert list(units[0].slot_flat_ids) == expected_first
    _assert_unit_coverage(
        {slot.slot_flat_id for slot in slots},
        [slot_id for unit in units for slot_id in unit.slot_flat_ids],
    )


def test_auto_mode_selects_outer_to_inner_all_or_ring_group(tmp_path: Path) -> None:
    run_large_ring = _write_generated_run(tmp_path, N=60, R=1, K=4)
    units_large = build_work_units(build_assembly_slots(run_large_ring), "auto")
    assert {unit.mode for unit in units_large} == {"stack_by_stack_outer_to_inner"}
    assert [unit.work_unit_id for unit in units_large] == [
        "W_LAYER000",
        "W_LAYER003",
        "W_LAYER001",
        "W_LAYER002",
    ]

    run_small_total = _write_generated_run(tmp_path, N=8, R=1, K=4)
    units_small = build_work_units(build_assembly_slots(run_small_total), "auto")
    assert [unit.mode for unit in units_small] == ["all_slots"]

    run_group = _write_generated_run(tmp_path, N=20, R=1, K=10)
    units_group = build_work_units(build_assembly_slots(run_group), "auto", ring_group_size=3)
    assert {unit.mode for unit in units_group} == {"ring_group"}


def test_work_units_reject_empty_slots() -> None:
    with pytest.raises(ValueError):
        build_work_units([], "all_slots")
