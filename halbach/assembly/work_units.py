from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from halbach.assembly.types import AssemblySlot, BuildWorkUnitMode, WorkUnit, WorkUnitMode

PhysicalRingKey = tuple[int, int]  # layer_id, ring_id


def _slot_ids(slots: list[AssemblySlot]) -> tuple[int, ...]:
    return tuple(slot.slot_flat_id for slot in slots)


def _group_by_physical_ring(slots: list[AssemblySlot]) -> dict[PhysicalRingKey, list[AssemblySlot]]:
    groups: dict[PhysicalRingKey, list[AssemblySlot]] = defaultdict(list)
    for slot in slots:
        groups[(slot.layer_id, slot.ring_id)].append(slot)
    for group_slots in groups.values():
        group_slots.sort(key=lambda item: item.theta_id)
    return dict(groups)


def _ordered_physical_rings(slots: list[AssemblySlot]) -> list[PhysicalRingKey]:
    return sorted({(slot.layer_id, slot.ring_id) for slot in slots})


def _build_all_slots(slots: list[AssemblySlot]) -> list[WorkUnit]:
    return [
        WorkUnit(
            work_unit_id="W_ALL",
            mode="all_slots",
            slot_flat_ids=_slot_ids(slots),
            label="all slots",
        )
    ]


def _build_single_physical_ring(slots: list[AssemblySlot]) -> list[WorkUnit]:
    groups = _group_by_physical_ring(slots)
    units: list[WorkUnit] = []
    for layer_id, ring_id in _ordered_physical_rings(slots):
        unit_slots = groups[(layer_id, ring_id)]
        units.append(
            WorkUnit(
                work_unit_id=f"W_K{layer_id:03d}_R{ring_id:03d}",
                mode="single_physical_ring",
                slot_flat_ids=_slot_ids(unit_slots),
                label=f"layer {layer_id}, ring {ring_id}",
            )
        )
    return units


def _build_ring_group(slots: list[AssemblySlot], ring_group_size: int) -> list[WorkUnit]:
    if ring_group_size <= 0:
        raise ValueError("ring_group_size must be positive")

    groups = _group_by_physical_ring(slots)
    rings = _ordered_physical_rings(slots)
    units: list[WorkUnit] = []
    for group_idx, start in enumerate(range(0, len(rings), ring_group_size)):
        ring_chunk = rings[start : start + ring_group_size]
        chunk_slots: list[AssemblySlot] = []
        for ring_key in ring_chunk:
            chunk_slots.extend(groups[ring_key])
        first_layer, first_ring = ring_chunk[0]
        last_layer, last_ring = ring_chunk[-1]
        units.append(
            WorkUnit(
                work_unit_id=f"W_G{group_idx:03d}",
                mode="ring_group",
                slot_flat_ids=_slot_ids(chunk_slots),
                label=(
                    f"group {group_idx}: layer {first_layer}, ring {first_ring}"
                    f" to layer {last_layer}, ring {last_ring}"
                ),
            )
        )
    return units


def _resolve_auto_mode(
    slots: list[AssemblySlot],
    *,
    large_ring_threshold: int,
    small_total_threshold: int,
) -> WorkUnitMode:
    groups = _group_by_physical_ring(slots)
    magnets_per_physical_ring = max(len(group_slots) for group_slots in groups.values())
    total_magnets = len(slots)
    if magnets_per_physical_ring >= large_ring_threshold:
        return "single_physical_ring"
    if total_magnets <= small_total_threshold:
        return "all_slots"
    return "ring_group"


def build_work_units(
    slots: list[AssemblySlot],
    mode: BuildWorkUnitMode,
    *,
    large_ring_threshold: int = 60,
    small_total_threshold: int = 150,
    ring_group_size: int = 4,
) -> list[WorkUnit]:
    """Build Plan C work units from active assembly slots."""
    if not slots:
        raise ValueError("slots must not be empty")
    if large_ring_threshold <= 0:
        raise ValueError("large_ring_threshold must be positive")
    if small_total_threshold <= 0:
        raise ValueError("small_total_threshold must be positive")

    resolved_mode: WorkUnitMode
    if mode == "auto":
        resolved_mode = _resolve_auto_mode(
            slots,
            large_ring_threshold=large_ring_threshold,
            small_total_threshold=small_total_threshold,
        )
    else:
        resolved_mode = mode

    if resolved_mode == "all_slots":
        return _build_all_slots(slots)
    if resolved_mode == "single_physical_ring":
        return _build_single_physical_ring(slots)
    if resolved_mode == "ring_group":
        return _build_ring_group(slots, ring_group_size)
    raise ValueError(f"Unsupported work unit mode: {resolved_mode}")


def assign_work_unit_ids(slots: list[AssemblySlot], work_units: list[WorkUnit]) -> list[AssemblySlot]:
    """Return copies of slots with `work_unit_id` populated from work units."""
    slot_to_unit: dict[int, str] = {}
    for unit in work_units:
        for slot_id in unit.slot_flat_ids:
            if slot_id in slot_to_unit:
                raise ValueError(f"slot_flat_id {slot_id} appears in multiple work units")
            slot_to_unit[slot_id] = unit.work_unit_id

    slot_ids = {slot.slot_flat_id for slot in slots}
    if set(slot_to_unit) != slot_ids:
        missing = sorted(slot_ids - set(slot_to_unit))
        extra = sorted(set(slot_to_unit) - slot_ids)
        raise ValueError(f"work unit slot coverage mismatch; missing={missing}, extra={extra}")

    return [replace(slot, work_unit_id=slot_to_unit[slot.slot_flat_id]) for slot in slots]


__all__ = ["assign_work_unit_ids", "build_work_units"]
