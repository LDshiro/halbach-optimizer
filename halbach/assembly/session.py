from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from halbach.assembly.inventory import decrement_cluster
from halbach.assembly.measurement import MeasurementProvider
from halbach.assembly.online_assignment import choose_best_linear_candidate
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
    return SessionSnapshot(
        state=cast(SessionState, str(data["state"])),
        sub_state=cast(SessionSubState, str(data["sub_state"])),
        residual=np.ascontiguousarray(residual, dtype=np.float64),
        remaining_slot_flat_ids=tuple(
            int(item) for item in cast(list[int], data["remaining_slot_flat_ids"])
        ),
        placements=tuple(_placement_from_dict(item) for item in placements_raw),
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


def _cluster_map(assignments: Sequence[ClusterAssignment] | None) -> dict[int, str | None]:
    if assignments is None:
        return {}
    mapping: dict[int, str | None] = {}
    for assignment in assignments:
        if assignment.quarantine_id is not None:
            raise ValueError("Plan C Step 7 session does not install quarantined magnets")
        mapping[assignment.magnet_id] = assignment.cluster_id
    return mapping


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
        self.provider = provider
        self.assignments = tuple(assignments or ())
        self._cluster_by_magnet = _cluster_map(assignments)
        self._initial_inventory = inventory
        self.current_inventory = inventory
        self.allowed_orientation_ids = (
            None if allowed_orientation_ids is None else tuple(allowed_orientation_ids)
        )
        self.log_path = None if log_path is None else Path(log_path)
        self.state: SessionState = "PREPARE_SESSION"
        self.sub_state: SessionSubState = "IDLE"
        self.residual = np.zeros(sensitivity_table.C.shape[2], dtype=np.float64)
        self.remaining_slot_flat_ids = tuple(int(item) for item in sensitivity_table.slot_flat_id.tolist())
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
            self.pending_magnet = self.provider.next_magnet()
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

    def confirm_insert(self) -> SessionSnapshot:
        """Confirm insertion of the solved slot/orientation candidate."""
        if self.state != "INSTALL_OR_CONFIRM_PAIR":
            raise RuntimeError("insert can only be confirmed in INSTALL_OR_CONFIRM_PAIR")
        if self.pending_magnet is None or self.pending_candidate is None:
            raise RuntimeError("no pending placement to confirm")
        self._undo_snapshot = self.snapshot()

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
    latest: dict[str, object] | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            continue
        event = cast(dict[str, object], raw)
        if event.get("event") == "state_snapshot":
            latest = cast(dict[str, object], event["snapshot"])
    if latest is None:
        raise ValueError("session log does not contain a state_snapshot event")
    return _snapshot_from_dict(latest)


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
    "load_latest_session_snapshot",
    "run_session_to_completion",
]
