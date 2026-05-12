from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from halbach.assembly.inventory import decrement_cluster
from halbach.assembly.measurement import (
    MeasurementProvider,
    MeasurementProviderError,
    MeasurementQualityError,
)
from halbach.assembly.online_assignment import choose_best_linear_candidate
from halbach.assembly.sensitivity import sensitivity_contribution
from halbach.assembly.types import (
    ClusterAssignment,
    ClusterInventory,
    LinearCandidate,
    MagnetError,
    Placement,
    SensitivityTable,
    VirtualMagnet,
)
from halbach.types import FloatArray

SessionState = Literal[
    "PREPARE_SESSION",
    "SELECT_WORK_UNIT",
    "BUILD_WORK_UNIT",
    "EVALUATE_MIRROR_PAIR_SWAP",
    "INSTALL_OR_CONFIRM_PAIR",
    "COMPLETE",
    "PAUSED",
    "ERROR",
]

SessionSubState = Literal[
    "IDLE",
    "CHOOSE_CLUSTER",
    "WAIT_FOR_MAGNET_MEASUREMENT",
    "VALIDATE_MEASUREMENT",
    "SOLVE_PLACEMENT",
    "WAIT_FOR_INSERT_CONFIRMATION",
    "UPDATE_STATE",
]


class SessionLogError(ValueError):
    """Session JSONL log is malformed or does not contain a valid snapshot."""


@dataclass(frozen=True)
class SessionSnapshot:
    """Serializable Plan C session state."""

    state: SessionState
    sub_state: SessionSubState
    residual: FloatArray
    remaining_slot_flat_ids: tuple[int, ...]
    placements: tuple[Placement, ...]
    measurement_index: int
    pending_magnet: VirtualMagnet | None = None
    pending_candidate: LinearCandidate | None = None


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("expected int value, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected int value, got {type(value).__name__}")


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("expected float value, got bool")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"expected float value, got {type(value).__name__}")


def _error_to_dict(error: MagnetError) -> dict[str, float]:
    return {
        "epsilon_parallel": error.epsilon_parallel,
        "delta_perp_1": error.delta_perp_1,
        "delta_perp_2": error.delta_perp_2,
    }


def _error_from_dict(data: dict[str, object]) -> MagnetError:
    return MagnetError(
        epsilon_parallel=_float_value(data["epsilon_parallel"]),
        delta_perp_1=_float_value(data["delta_perp_1"]),
        delta_perp_2=_float_value(data["delta_perp_2"]),
    )


def _magnet_to_dict(magnet: VirtualMagnet) -> dict[str, object]:
    return {
        "magnet_id": magnet.magnet_id,
        "true_error": _error_to_dict(magnet.true_error),
        "measured_error": _error_to_dict(magnet.measured_error),
        "quality": magnet.quality,
    }


def _magnet_from_dict(data: dict[str, object]) -> VirtualMagnet:
    true_error = cast(dict[str, object], data["true_error"])
    measured_error = cast(dict[str, object], data["measured_error"])
    return VirtualMagnet(
        magnet_id=_int_value(data["magnet_id"]),
        true_error=_error_from_dict(true_error),
        measured_error=_error_from_dict(measured_error),
        quality=None if data.get("quality") is None else _float_value(data["quality"]),
    )


def _candidate_to_dict(candidate: LinearCandidate) -> dict[str, object]:
    return {
        "slot_flat_id": candidate.slot_flat_id,
        "orientation_id": candidate.orientation_id,
        "score": candidate.score,
        "contribution": candidate.contribution.tolist(),
    }


def _candidate_from_dict(data: dict[str, object]) -> LinearCandidate:
    contribution = np.asarray(cast(list[float], data["contribution"]), dtype=np.float64)
    return LinearCandidate(
        slot_flat_id=_int_value(data["slot_flat_id"]),
        orientation_id=str(data["orientation_id"]),
        score=_float_value(data["score"]),
        contribution=np.ascontiguousarray(contribution, dtype=np.float64),
    )


def _placement_to_dict(placement: Placement) -> dict[str, object]:
    return {
        "slot_flat_id": placement.slot_flat_id,
        "magnet_id": placement.magnet_id,
        "orientation_id": placement.orientation_id,
        "cluster_requested": placement.cluster_requested,
        "insert_order": placement.insert_order,
        "decision_engine": placement.decision_engine,
    }


def _placement_from_dict(data: dict[str, object]) -> Placement:
    cluster_raw = data.get("cluster_requested")
    return Placement(
        slot_flat_id=_int_value(data["slot_flat_id"]),
        magnet_id=_int_value(data["magnet_id"]),
        orientation_id=str(data["orientation_id"]),
        cluster_requested=None if cluster_raw is None else str(cluster_raw),
        insert_order=_int_value(data["insert_order"]),
        decision_engine=str(data["decision_engine"]),
    )


def _snapshot_to_dict(snapshot: SessionSnapshot) -> dict[str, object]:
    return {
        "state": snapshot.state,
        "sub_state": snapshot.sub_state,
        "residual": snapshot.residual.tolist(),
        "remaining_slot_flat_ids": list(snapshot.remaining_slot_flat_ids),
        "placements": [_placement_to_dict(placement) for placement in snapshot.placements],
        "measurement_index": snapshot.measurement_index,
        "pending_magnet": (
            None if snapshot.pending_magnet is None else _magnet_to_dict(snapshot.pending_magnet)
        ),
        "pending_candidate": (
            None
            if snapshot.pending_candidate is None
            else _candidate_to_dict(snapshot.pending_candidate)
        ),
    }


def _snapshot_from_dict(data: dict[str, object]) -> SessionSnapshot:
    placements_raw = cast(list[dict[str, object]], data.get("placements", []))
    pending_magnet_raw = data.get("pending_magnet")
    pending_candidate_raw = data.get("pending_candidate")
    residual = np.asarray(cast(list[float], data["residual"]), dtype=np.float64)
    if residual.ndim != 1:
        raise ValueError("snapshot residual must be 1-D")
    remaining = tuple(
        int(item) for item in cast(list[int], data["remaining_slot_flat_ids"])
    )
    if len(remaining) != len(set(remaining)):
        raise ValueError("snapshot remaining_slot_flat_ids contains duplicates")
    placements = tuple(_placement_from_dict(item) for item in placements_raw)
    placement_slot_ids = [placement.slot_flat_id for placement in placements]
    if len(placement_slot_ids) != len(set(placement_slot_ids)):
        raise ValueError("snapshot placements contain duplicate slot_flat_id values")
    return SessionSnapshot(
        state=cast(SessionState, str(data["state"])),
        sub_state=cast(SessionSubState, str(data["sub_state"])),
        residual=np.ascontiguousarray(residual, dtype=np.float64),
        remaining_slot_flat_ids=remaining,
        placements=placements,
        measurement_index=_int_value(data["measurement_index"]),
        pending_magnet=(
            None
            if pending_magnet_raw is None
            else _magnet_from_dict(cast(dict[str, object], pending_magnet_raw))
        ),
        pending_candidate=(
            None
            if pending_candidate_raw is None
            else _candidate_from_dict(cast(dict[str, object], pending_candidate_raw))
        ),
    )


def _slot_ids_from_table(table: SensitivityTable) -> tuple[int, ...]:
    slot_ids = tuple(int(item) for item in table.slot_flat_id.tolist())
    if not slot_ids:
        raise ValueError("sensitivity table must contain at least one slot")
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("sensitivity table contains duplicate slot ids")
    if table.C.ndim != 4 or table.C.shape[2] <= 0:
        raise ValueError("sensitivity table C must have shape (S, O, residual_dim, 3)")
    if table.C.shape[0] != len(slot_ids):
        raise ValueError("sensitivity table slot ids and C shape mismatch")
    return slot_ids


def _cluster_map(
    assignments: Sequence[ClusterAssignment] | None,
    inventory: ClusterInventory | None,
) -> dict[int, str | None]:
    if assignments is None:
        return {}
    mapping: dict[int, str | None] = {}
    counts_by_cluster: dict[str, int] = {}
    for assignment in assignments:
        if assignment.magnet_id in mapping:
            raise ValueError(f"duplicate cluster assignment for magnet {assignment.magnet_id}")
        if assignment.quarantine_id is not None:
            raise ValueError("Plan C Step 7 session does not install quarantined magnets")
        if assignment.cluster_id is None:
            raise ValueError("normal cluster assignment must include cluster_id")
        mapping[assignment.magnet_id] = assignment.cluster_id
        counts_by_cluster[assignment.cluster_id] = counts_by_cluster.get(assignment.cluster_id, 0) + 1
    if inventory is not None:
        for cluster_id, requested_count in counts_by_cluster.items():
            if cluster_id not in inventory.clusters:
                raise ValueError(f"inventory is missing assigned cluster {cluster_id}")
            available_count = inventory.clusters[cluster_id].count
            if requested_count > available_count:
                raise ValueError(
                    f"inventory cluster {cluster_id} has {available_count} items for {requested_count} assignments"
                )
    return mapping


def _measurement_reject_event(exc: MeasurementProviderError) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "measurement_rejected",
        "error_type": exc.__class__.__name__,
        "reason": exc.reason,
        "message": str(exc),
    }
    if exc.quarantine_id is not None:
        payload["quarantine_id"] = exc.quarantine_id
    if isinstance(exc, MeasurementQualityError):
        payload["quality"] = exc.quality
        payload["quality_threshold"] = exc.quality_threshold
    return payload


def _inventory_after_placements(
    inventory: ClusterInventory | None,
    placements: Sequence[Placement],
) -> ClusterInventory | None:
    current = inventory
    if current is None:
        return None
    for placement in placements:
        if placement.cluster_requested is not None:
            current = decrement_cluster(current, placement.cluster_requested)
    return current


class PlanCSession:
    """Step-by-step Plan C session state machine MVP."""

    def __init__(
        self,
        *,
        sensitivity_table: SensitivityTable,
        provider: MeasurementProvider,
        assignments: Sequence[ClusterAssignment] | None = None,
        inventory: ClusterInventory | None = None,
        allowed_orientation_ids: Sequence[str] | None = None,
        log_path: str | Path | None = None,
        start_new: bool = True,
        reset_log: bool = True,
    ) -> None:
        self.sensitivity_table = sensitivity_table
        self._table_slot_ids = _slot_ids_from_table(sensitivity_table)
        self.provider = provider
        self.assignments = tuple(assignments or ())
        self._cluster_by_magnet = _cluster_map(assignments, inventory)
        self._initial_inventory = inventory
        self.current_inventory = inventory
        self.allowed_orientation_ids = (
            None if allowed_orientation_ids is None else tuple(allowed_orientation_ids)
        )
        self.log_path = None if log_path is None else Path(log_path)
        self.state: SessionState = "PREPARE_SESSION"
        self.sub_state: SessionSubState = "IDLE"
        self.residual = np.zeros(sensitivity_table.C.shape[2], dtype=np.float64)
        self.remaining_slot_flat_ids = self._table_slot_ids
        self.placements: tuple[Placement, ...] = ()
        self.pending_magnet: VirtualMagnet | None = None
        self.pending_candidate: LinearCandidate | None = None
        self._undo_snapshot: SessionSnapshot | None = None

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if start_new and reset_log:
                self.log_path.write_text("", encoding="utf-8")
        if start_new:
            self._append_event({"event": "session_started"})
            self._log_snapshot()

    @property
    def measurement_index(self) -> int:
        return self.provider.position

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            state=self.state,
            sub_state=self.sub_state,
            residual=np.array(self.residual, dtype=np.float64, copy=True),
            remaining_slot_flat_ids=tuple(self.remaining_slot_flat_ids),
            placements=tuple(self.placements),
            measurement_index=self.provider.position,
            pending_magnet=self.pending_magnet,
            pending_candidate=self.pending_candidate,
        )

    def _append_event(self, payload: dict[str, object]) -> None:
        if self.log_path is None:
            return
        event = {
            "schema_version": 1,
            "state": self.state,
            "sub_state": self.sub_state,
            **payload,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")

    def _log_snapshot(self) -> None:
        self._append_event(
            {
                "event": "state_snapshot",
                "snapshot": _snapshot_to_dict(self.snapshot()),
            }
        )

    def step(self) -> SessionSnapshot:
        """Advance one state-machine step. Insert confirmation is explicit."""
        if self.state == "COMPLETE":
            return self.snapshot()
        if self.state == "ERROR":
            raise RuntimeError("cannot step a session in ERROR state")
        if self.state == "PAUSED":
            self.state = "BUILD_WORK_UNIT"
            self._append_event({"event": "session_resumed"})

        if self.state == "PREPARE_SESSION":
            self.state = "SELECT_WORK_UNIT"
            self.sub_state = "CHOOSE_CLUSTER"
        elif self.state == "SELECT_WORK_UNIT":
            self.state = "BUILD_WORK_UNIT"
            self.sub_state = "WAIT_FOR_MAGNET_MEASUREMENT"
        elif self.state == "BUILD_WORK_UNIT":
            self._step_build_work_unit()
        elif self.state == "EVALUATE_MIRROR_PAIR_SWAP":
            self.state = "INSTALL_OR_CONFIRM_PAIR"
            self.sub_state = "WAIT_FOR_INSERT_CONFIRMATION"
        elif self.state == "INSTALL_OR_CONFIRM_PAIR":
            self._append_event({"event": "waiting_for_insert_confirmation"})
        self._log_snapshot()
        return self.snapshot()

    def _step_build_work_unit(self) -> None:
        if self.sub_state == "WAIT_FOR_MAGNET_MEASUREMENT":
            try:
                self.pending_magnet = self.provider.next_magnet()
            except MeasurementProviderError as exc:
                self.pending_magnet = None
                self.pending_candidate = None
                self.state = "BUILD_WORK_UNIT"
                self.sub_state = "WAIT_FOR_MAGNET_MEASUREMENT"
                self._append_event(_measurement_reject_event(exc))
                return
            self.sub_state = "VALIDATE_MEASUREMENT"
            self._append_event(
                {
                    "event": "measurement_received",
                    "magnet_id": self.pending_magnet.magnet_id,
                }
            )
            return
        if self.sub_state == "VALIDATE_MEASUREMENT":
            if self.pending_magnet is None:
                raise RuntimeError("missing pending magnet")
            if self.assignments and self.pending_magnet.magnet_id not in self._cluster_by_magnet:
                raise ValueError(f"missing cluster assignment for magnet {self.pending_magnet.magnet_id}")
            self.sub_state = "SOLVE_PLACEMENT"
            return
        if self.sub_state == "SOLVE_PLACEMENT":
            if self.pending_magnet is None:
                raise RuntimeError("missing pending magnet")
            self.pending_candidate = choose_best_linear_candidate(
                self.sensitivity_table,
                self.residual,
                self.remaining_slot_flat_ids,
                self.pending_magnet.measured_error,
                allowed_orientation_ids=self.allowed_orientation_ids,
            )
            self.state = "EVALUATE_MIRROR_PAIR_SWAP"
            self.sub_state = "WAIT_FOR_INSERT_CONFIRMATION"
            self._append_event(
                {
                    "event": "placement_solved",
                    "slot_flat_id": self.pending_candidate.slot_flat_id,
                    "orientation_id": self.pending_candidate.orientation_id,
                    "score": self.pending_candidate.score,
                }
            )
            return
        raise RuntimeError(f"unsupported BUILD_WORK_UNIT sub_state: {self.sub_state}")

    def _validate_slot_state(self) -> None:
        table_slots = set(self._table_slot_ids)
        remaining = list(self.remaining_slot_flat_ids)
        if len(remaining) != len(set(remaining)):
            raise ValueError("remaining slot state contains duplicate slot ids")
        remaining_set = set(remaining)
        placed_slot_ids = [placement.slot_flat_id for placement in self.placements]
        if len(placed_slot_ids) != len(set(placed_slot_ids)):
            raise ValueError("placements contain duplicate slot ids")
        placed_set = set(placed_slot_ids)
        unknown = sorted((remaining_set | placed_set) - table_slots)
        overlap = sorted(remaining_set & placed_set)
        missing = sorted(table_slots - (remaining_set | placed_set))
        if unknown or overlap or missing:
            raise ValueError(
                f"slot state mismatch; unknown={unknown}, overlap={overlap}, missing={missing}"
            )
        expected_residual_shape = (self.sensitivity_table.C.shape[2],)
        if self.residual.shape != expected_residual_shape:
            raise ValueError(f"residual must have shape {expected_residual_shape}")

    def _validate_confirm_ready(self) -> None:
        self._validate_slot_state()
        if self.pending_magnet is None or self.pending_candidate is None:
            raise RuntimeError("no pending placement to confirm")
        if self.pending_candidate.slot_flat_id not in self.remaining_slot_flat_ids:
            raise ValueError(
                f"pending slot {self.pending_candidate.slot_flat_id} is not available in remaining slots"
            )
        if self.pending_candidate.orientation_id not in self.sensitivity_table.orientation_id:
            raise ValueError(f"unknown orientation_id: {self.pending_candidate.orientation_id}")
        expected_residual_shape = (self.sensitivity_table.C.shape[2],)
        if self.pending_candidate.contribution.shape != expected_residual_shape:
            raise ValueError(f"pending candidate contribution must have shape {expected_residual_shape}")

    def override_pending_candidate(
        self,
        slot_flat_id: int,
        orientation_id: str,
        *,
        reason: str,
        operator: str | None = None,
    ) -> SessionSnapshot:
        """Replace the solved pending candidate and log a manual override event."""
        if self.pending_magnet is None or self.pending_candidate is None:
            raise RuntimeError("manual override requires a pending placement")
        override_reason = str(reason).strip()
        if not override_reason:
            raise ValueError("manual override reason must be non-empty")
        slot_id = int(slot_flat_id)
        if slot_id not in self.remaining_slot_flat_ids:
            raise ValueError(f"override slot {slot_id} is not available in remaining slots")
        if orientation_id not in self.sensitivity_table.orientation_id:
            raise ValueError(f"unknown orientation_id: {orientation_id}")
        old_candidate = self.pending_candidate
        contribution = sensitivity_contribution(
            self.sensitivity_table,
            slot_id,
            orientation_id,
            self.pending_magnet.measured_error,
        )
        updated = self.residual + contribution
        score = float(np.dot(updated, updated))
        self.pending_candidate = LinearCandidate(
            slot_flat_id=slot_id,
            orientation_id=orientation_id,
            score=score,
            contribution=contribution,
        )
        self._append_event(
            {
                "event": "manual_override",
                "old_slot_flat_id": old_candidate.slot_flat_id,
                "old_orientation_id": old_candidate.orientation_id,
                "slot_flat_id": slot_id,
                "orientation_id": orientation_id,
                "score": score,
                "reason": override_reason,
                "operator": operator,
            }
        )
        self._log_snapshot()
        return self.snapshot()

    def confirm_insert(self) -> SessionSnapshot:
        """Confirm insertion of the solved slot/orientation candidate."""
        if self.state != "INSTALL_OR_CONFIRM_PAIR":
            raise RuntimeError("insert can only be confirmed in INSTALL_OR_CONFIRM_PAIR")
        self._validate_confirm_ready()
        self._undo_snapshot = self.snapshot()

        assert self.pending_magnet is not None
        assert self.pending_candidate is not None
        cluster_id = self._cluster_by_magnet.get(self.pending_magnet.magnet_id)
        if self.current_inventory is not None and cluster_id is not None:
            self.current_inventory = decrement_cluster(self.current_inventory, cluster_id)
        placement = Placement(
            slot_flat_id=self.pending_candidate.slot_flat_id,
            magnet_id=self.pending_magnet.magnet_id,
            orientation_id=self.pending_candidate.orientation_id,
            cluster_requested=cluster_id,
            insert_order=len(self.placements),
            decision_engine="linear_sensitivity",
        )
        self.residual = np.ascontiguousarray(
            self.residual + self.pending_candidate.contribution,
            dtype=np.float64,
        )
        remaining = list(self.remaining_slot_flat_ids)
        remaining.remove(self.pending_candidate.slot_flat_id)
        self.remaining_slot_flat_ids = tuple(remaining)
        self.placements = (*self.placements, placement)
        self._append_event(
            {
                "event": "insert_confirmed",
                "slot_flat_id": placement.slot_flat_id,
                "magnet_id": placement.magnet_id,
                "orientation_id": placement.orientation_id,
                "insert_order": placement.insert_order,
            }
        )
        self.pending_magnet = None
        self.pending_candidate = None
        if len(self.placements) >= self.sensitivity_table.slot_flat_id.shape[0]:
            self.state = "COMPLETE"
            self.sub_state = "IDLE"
            self._append_event({"event": "session_completed", "placements": len(self.placements)})
        else:
            self.state = "BUILD_WORK_UNIT"
            self.sub_state = "WAIT_FOR_MAGNET_MEASUREMENT"
        self._log_snapshot()
        return self.snapshot()

    def undo_last_insert(self) -> SessionSnapshot:
        """Undo the most recent confirmed insert, restoring residual and slot occupancy."""
        if self._undo_snapshot is None:
            raise RuntimeError("no confirmed insert is available to undo")
        snapshot = self._undo_snapshot
        self._restore_snapshot(snapshot)
        self._undo_snapshot = None
        self._append_event({"event": "undo_requested"})
        self._log_snapshot()
        return self.snapshot()

    def pause(self) -> SessionSnapshot:
        self.state = "PAUSED"
        self._append_event({"event": "session_paused"})
        self._log_snapshot()
        return self.snapshot()

    def _restore_snapshot(self, snapshot: SessionSnapshot) -> None:
        self.state = snapshot.state
        self.sub_state = snapshot.sub_state
        self.residual = np.array(snapshot.residual, dtype=np.float64, copy=True)
        self.remaining_slot_flat_ids = tuple(snapshot.remaining_slot_flat_ids)
        self.placements = tuple(snapshot.placements)
        self.pending_magnet = snapshot.pending_magnet
        self.pending_candidate = snapshot.pending_candidate
        self.provider.set_position(snapshot.measurement_index)
        self.current_inventory = _inventory_after_placements(self._initial_inventory, self.placements)
        self._validate_slot_state()

    @classmethod
    def resume_from_log(
        cls,
        log_path: str | Path,
        *,
        sensitivity_table: SensitivityTable,
        provider: MeasurementProvider,
        assignments: Sequence[ClusterAssignment] | None = None,
        inventory: ClusterInventory | None = None,
        allowed_orientation_ids: Sequence[str] | None = None,
    ) -> PlanCSession:
        snapshot = load_latest_session_snapshot(log_path)
        session = cls(
            sensitivity_table=sensitivity_table,
            provider=provider,
            assignments=assignments,
            inventory=inventory,
            allowed_orientation_ids=allowed_orientation_ids,
            log_path=log_path,
            start_new=False,
            reset_log=False,
        )
        session._restore_snapshot(snapshot)
        session._append_event({"event": "session_resumed"})
        session._log_snapshot()
        return session


def load_latest_session_snapshot(path: str | Path) -> SessionSnapshot:
    """Load the latest state_snapshot event from a session JSONL log."""
    latest: SessionSnapshot | None = None
    log_path = Path(path)
    for line_no, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionLogError(f"{log_path}:{line_no}: invalid JSON line") from exc
        if not isinstance(raw, dict):
            raise SessionLogError(f"{log_path}:{line_no}: session log event must be an object")
        event = cast(dict[str, object], raw)
        if event.get("event") == "state_snapshot":
            snapshot_raw = event.get("snapshot")
            if not isinstance(snapshot_raw, dict):
                raise SessionLogError(f"{log_path}:{line_no}: state_snapshot missing object snapshot")
            try:
                latest = _snapshot_from_dict(cast(dict[str, object], snapshot_raw))
            except (KeyError, TypeError, ValueError) as exc:
                raise SessionLogError(f"{log_path}:{line_no}: invalid state_snapshot payload") from exc
    if latest is None:
        raise SessionLogError("session log does not contain a state_snapshot event")
    return latest


def run_session_to_completion(session: PlanCSession) -> SessionSnapshot:
    """Advance and auto-confirm inserts until the session is complete."""
    while session.state != "COMPLETE":
        if session.state == "INSTALL_OR_CONFIRM_PAIR":
            session.confirm_insert()
        else:
            session.step()
    return session.snapshot()


__all__ = [
    "PlanCSession",
    "SessionSnapshot",
    "SessionState",
    "SessionSubState",
    "SessionLogError",
    "load_latest_session_snapshot",
    "run_session_to_completion",
]
