from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

_TRIAL_RE = re.compile(r"assembly_timeline_trial_(?P<trial>\d+)\.jsonl$")
_CLUSTER_RE = re.compile(r"^S(?P<strength>\d+)_A(?P<angle>\d+)$")
_STRENGTH_COLORSCALE = [
    (0.0, "#2166ac"),
    (0.5, "#f7f7f7"),
    (1.0, "#b2182b"),
]
_STRENGTH_COLOR_VALUES = [color for _position, color in _STRENGTH_COLORSCALE]
_ORIENTATION_PATTERN_LABELS = {
    "O0": "1",
    "O90": "2",
    "O180": "3",
    "O270": "4",
}

ClusterMatrixMode = Literal["initial", "used", "remaining", "planned"]


@dataclass(frozen=True)
class RingTrialBundle:
    out_dir: Path
    trial_id: int
    summary: dict[str, Any]
    ring_summary: list[dict[str, str]]
    ring_pair_summary: list[dict[str, str]]
    timeline: list[dict[str, Any]]
    quota_plan: list[dict[str, str]]
    pickup_log: list[dict[str, str]]


@dataclass(frozen=True)
class PlaybackState:
    step: int
    active_layer_id: int | None
    active_ring_id: int | None
    active_work_unit_id: str | None
    current_event: dict[str, Any] | None
    placed_slot_ids: tuple[int, ...]
    placed_count: int
    current_cluster: str | None
    current_slot_flat_id: int | None
    current_orientation_id: str | None
    residual_norm: float | None


@dataclass(frozen=True)
class MatrixPayload:
    x: tuple[int, ...]
    y: tuple[int, ...]
    z: np.ndarray
    label: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} line {line_number} must contain a JSON object")
        rows.append(raw)
    return rows


def _trial_file(out_dir: Path, stem: str, trial_id: int, suffix: str) -> Path:
    return out_dir / f"{stem}_trial_{trial_id:03d}.{suffix}"


def _to_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _to_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _finite_or_none(value: object) -> float | None:
    result = _to_float(value)
    if not math.isfinite(result):
        return None
    return result


def _float_tuple(value: object, expected_len: int) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple):
        return None
    if len(value) != expected_len:
        return None
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and item > 0.0 for item in result):
        return None
    return result


def _metadata(bundle: RingTrialBundle) -> dict[str, Any]:
    raw = bundle.summary.get("metadata")
    if not isinstance(raw, dict):
        return {}
    return raw


def _visualization_geometry(bundle: RingTrialBundle) -> dict[str, Any]:
    raw = _metadata(bundle).get("visualization_geometry")
    if not isinstance(raw, dict):
        return {}
    return raw


def _magnet_dimensions_m(bundle: RingTrialBundle) -> tuple[float, float, float] | None:
    geometry = _visualization_geometry(bundle)
    dimensions = _float_tuple(geometry.get("magnet_dimensions_m"), 3)
    if dimensions is not None:
        return dimensions

    volume_mm3 = geometry.get("sc_volume_mm3")
    if volume_mm3 is None:
        return None
    volume_m3 = float(volume_mm3) * 1.0e-9
    if not math.isfinite(volume_m3) or volume_m3 <= 0.0:
        return None
    edge_m = volume_m3 ** (1.0 / 3.0)
    return (edge_m, edge_m, edge_m)


def _orientation_pattern_label(orientation_id: object) -> str:
    return _ORIENTATION_PATTERN_LABELS.get(str(orientation_id or ""), "")


def _json_dict(value: str) -> dict[str, int]:
    if not value:
        return {}
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("expected JSON object")
    result: dict[str, int] = {}
    for key, count in raw.items():
        result[str(key)] = int(count)
    return result


def discover_trial_ids(out_dir: str | Path) -> tuple[int, ...]:
    """Return trial ids that have R9 assembly timeline files."""
    path = Path(out_dir)
    if not path.is_dir():
        return ()
    ids: set[int] = set()
    for file_path in path.glob("assembly_timeline_trial_*.jsonl"):
        match = _TRIAL_RE.fullmatch(file_path.name)
        if match is not None:
            ids.add(int(match.group("trial")))
    return tuple(sorted(ids))


def load_ring_trial_bundle(out_dir: str | Path, trial_id: int) -> RingTrialBundle:
    """Load one R9 visualization output bundle."""
    path = Path(out_dir)
    summary_path = path / "simulation_summary.json"
    required = {
        "ring_summary": _trial_file(path, "ring_summary", trial_id, "csv"),
        "ring_pair_summary": _trial_file(path, "ring_pair_summary", trial_id, "csv"),
        "timeline": _trial_file(path, "assembly_timeline", trial_id, "jsonl"),
        "quota_plan": _trial_file(path, "cluster_quota_plan", trial_id, "csv"),
        "pickup_log": _trial_file(path, "cluster_pickup_log", trial_id, "csv"),
    }
    missing = [
        str(file_path) for file_path in [summary_path, *required.values()] if not file_path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing Plan C R9 output file(s): {missing}")
    summary_raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_raw, dict):
        raise ValueError("simulation_summary.json must contain an object")
    return RingTrialBundle(
        out_dir=path,
        trial_id=int(trial_id),
        summary=summary_raw,
        ring_summary=_read_csv(required["ring_summary"]),
        ring_pair_summary=_read_csv(required["ring_pair_summary"]),
        timeline=_read_jsonl(required["timeline"]),
        quota_plan=_read_csv(required["quota_plan"]),
        pickup_log=_read_csv(required["pickup_log"]),
    )


def playback_state_at_step(bundle: RingTrialBundle, step: int) -> PlaybackState:
    """Return active ring and completed placement state at a timeline step."""
    if not bundle.timeline:
        return PlaybackState(
            step=0,
            active_layer_id=None,
            active_ring_id=None,
            active_work_unit_id=None,
            current_event=None,
            placed_slot_ids=(),
            placed_count=0,
            current_cluster=None,
            current_slot_flat_id=None,
            current_orientation_id=None,
            residual_norm=None,
        )
    clamped = max(0, min(int(step), len(bundle.timeline) - 1))
    completed = bundle.timeline[: clamped + 1]
    event = completed[-1]
    return PlaybackState(
        step=clamped,
        active_layer_id=_to_int(event.get("layer_id")),
        active_ring_id=_to_int(event.get("ring_id")),
        active_work_unit_id=str(event.get("work_unit_id") or ""),
        current_event=event,
        placed_slot_ids=tuple(_to_int(item.get("slot_flat_id")) for item in completed),
        placed_count=len(completed),
        current_cluster=str(event.get("cluster_requested") or "") or None,
        current_slot_flat_id=_to_int(event.get("slot_flat_id")),
        current_orientation_id=str(event.get("orientation_id") or "") or None,
        residual_norm=_finite_or_none(event.get("residual_norm")),
    )


def ring_completion_steps(bundle: RingTrialBundle) -> tuple[int, ...]:
    """Return timeline step indices where the active physical ring changes or ends."""
    if not bundle.timeline:
        return ()
    steps: list[int] = []
    previous_key: tuple[int, int] | None = None
    for index, event in enumerate(bundle.timeline):
        key = (_to_int(event.get("layer_id")), _to_int(event.get("ring_id")))
        if previous_key is not None and key != previous_key:
            steps.append(index - 1)
        previous_key = key
    steps.append(len(bundle.timeline) - 1)
    return tuple(steps)


def ring_metric_matrix(bundle: RingTrialBundle, metric: str) -> MatrixPayload:
    """Build a layer x ring metric matrix from ring_summary CSV rows."""
    if not bundle.ring_summary:
        return MatrixPayload(x=(), y=(), z=np.zeros((0, 0), dtype=np.float64), label=metric)
    layer_ids = tuple(sorted({_to_int(row.get("layer_id")) for row in bundle.ring_summary}))
    ring_ids = tuple(sorted({_to_int(row.get("ring_id")) for row in bundle.ring_summary}))
    layer_to_idx = {layer_id: idx for idx, layer_id in enumerate(layer_ids)}
    ring_to_idx = {ring_id: idx for idx, ring_id in enumerate(ring_ids)}
    z = np.full((len(layer_ids), len(ring_ids)), np.nan, dtype=np.float64)
    for row in bundle.ring_summary:
        if metric not in row:
            raise KeyError(f"ring summary metric not found: {metric}")
        z[layer_to_idx[_to_int(row.get("layer_id"))], ring_to_idx[_to_int(row.get("ring_id"))]] = (
            _to_float(row[metric])
        )
    return MatrixPayload(x=ring_ids, y=layer_ids, z=z, label=metric)


def _cluster_key(cluster_id: str) -> tuple[int, int]:
    match = _CLUSTER_RE.fullmatch(cluster_id)
    if match is None:
        raise ValueError(f"cluster id must use Sxx_Ayy format: {cluster_id}")
    return int(match.group("strength")), int(match.group("angle"))


def _matrix_from_cluster_counts(counts: dict[str, int], label: str) -> MatrixPayload:
    if not counts:
        return MatrixPayload(x=(0,), y=(0,), z=np.zeros((1, 1), dtype=np.float64), label=label)
    keys = {_cluster_key(cluster_id) for cluster_id in counts}
    strength_bins = tuple(range(max(strength for strength, _angle in keys) + 1))
    angle_bins = tuple(range(max(angle for _strength, angle in keys) + 1))
    z = np.zeros((len(angle_bins), len(strength_bins)), dtype=np.float64)
    for cluster_id, count in counts.items():
        strength, angle = _cluster_key(cluster_id)
        z[angle, strength] = float(count)
    return MatrixPayload(x=strength_bins, y=angle_bins, z=z, label=label)


def cluster_inventory_counts(bundle: RingTrialBundle, mode: ClusterMatrixMode) -> dict[str, int]:
    """Return cluster counts for initial, used, remaining, or planned quota modes."""
    initial: dict[str, int] = {}
    used: dict[str, int] = {}
    planned: dict[str, int] = {}
    for row in bundle.pickup_log:
        assignment_cluster = str(row.get("assignment_cluster_id") or "")
        requested_cluster = str(row.get("cluster_requested") or "")
        if assignment_cluster:
            initial[assignment_cluster] = initial.get(assignment_cluster, 0) + 1
        if requested_cluster:
            used[requested_cluster] = used.get(requested_cluster, 0) + 1
    for row in bundle.quota_plan:
        for cluster_id, count in _json_dict(str(row.get("quota_by_cluster_json") or "")).items():
            planned[cluster_id] = planned.get(cluster_id, 0) + count
    if mode == "initial":
        return dict(sorted(initial.items()))
    if mode == "used":
        return dict(sorted(used.items()))
    if mode == "planned":
        return dict(sorted(planned.items()))
    if mode == "remaining":
        cluster_ids = sorted(set(initial) | set(used))
        return {
            cluster_id: initial.get(cluster_id, 0) - used.get(cluster_id, 0)
            for cluster_id in cluster_ids
        }
    raise ValueError(f"unsupported cluster matrix mode: {mode}")


def cluster_inventory_matrix(bundle: RingTrialBundle, mode: ClusterMatrixMode) -> MatrixPayload:
    """Build a strength-bin x angle-bin matrix for inventory/usage visualization."""
    return _matrix_from_cluster_counts(cluster_inventory_counts(bundle, mode), mode)


def plot_side_stack_view(
    bundle: RingTrialBundle,
    state: PlaybackState,
) -> go.Figure:
    """Plot completed/active/pending physical rings as a layer x ring heatmap."""
    metric = ring_metric_matrix(bundle, "mean_epsilon")
    z = np.zeros_like(metric.z, dtype=np.float64)
    placed_keys = {
        (_to_int(event.get("layer_id")), _to_int(event.get("ring_id")))
        for event in bundle.timeline[: state.step + 1]
    }
    layer_to_idx = {layer_id: idx for idx, layer_id in enumerate(metric.y)}
    ring_to_idx = {ring_id: idx for idx, ring_id in enumerate(metric.x)}
    for layer_id, ring_id in placed_keys:
        if layer_id in layer_to_idx and ring_id in ring_to_idx:
            z[layer_to_idx[layer_id], ring_to_idx[ring_id]] = 1.0
    if state.active_layer_id in layer_to_idx:
        layer_idx = layer_to_idx[int(state.active_layer_id)]
        for ring_idx in range(len(metric.x)):
            z[layer_idx, ring_idx] = 2.0
    fig = go.Figure(
        go.Heatmap(
            x=metric.x,
            y=metric.y,
            z=z,
            zmin=0.0,
            zmax=2.0,
            colorscale=[(0.0, "#f3f4f6"), (0.5, "#60a5fa"), (1.0, "#ef4444")],
            colorbar={"tickvals": [0, 1, 2], "ticktext": ["pending", "completed", "active"]},
        )
    )
    fig.update_layout(
        title="Assembly Stack",
        xaxis_title="ring_id",
        yaxis_title="layer_id",
        height=360,
        margin={"l": 50, "r": 10, "b": 40, "t": 45},
    )
    return fig


def plot_active_ring_polar_view(
    bundle: RingTrialBundle,
    state: PlaybackState,
) -> go.Figure:
    """Plot active-layer magnet frames using run-derived coordinates when available."""
    if state.active_layer_id is None:
        fig = go.Figure()
        fig.update_layout(
            title="Active Stack R Layers",
            height=680,
            margin={"l": 30, "r": 30, "b": 30, "t": 45},
        )
        return fig

    layer_rows = [
        row for row in bundle.pickup_log if _to_int(row.get("layer_id")) == state.active_layer_id
    ]
    completed_rows = [row for row in layer_rows if _to_int(row.get("insert_order")) <= state.step]
    pending_rows = [row for row in layer_rows if _to_int(row.get("insert_order")) > state.step]
    ring_ids = tuple(sorted({_to_int(row.get("ring_id")) for row in layer_rows}))
    count_hint_by_ring = {
        ring_id: max(
            1,
            1
            + max(
                _to_int(row.get("theta_id"))
                for row in layer_rows
                if _to_int(row.get("ring_id")) == ring_id
            ),
        )
        for ring_id in ring_ids
    }

    def row_has_real_position(row: dict[str, str]) -> bool:
        return (
            _finite_or_none(row.get("center_x_m")) is not None
            and _finite_or_none(row.get("center_y_m")) is not None
        )

    use_real_positions = bool(layer_rows) and all(row_has_real_position(row) for row in layer_rows)
    fallback_ring_to_radius = {ring_id: index + 1 for index, ring_id in enumerate(ring_ids)}

    def row_position(row: dict[str, str]) -> tuple[float, float]:
        if use_real_positions:
            x_m = _finite_or_none(row.get("center_x_m"))
            y_m = _finite_or_none(row.get("center_y_m"))
            if x_m is not None and y_m is not None:
                return x_m, y_m
        ring_id = _to_int(row.get("ring_id"))
        theta_rad = 2.0 * math.pi * _to_int(row.get("theta_id")) / count_hint_by_ring[ring_id]
        radius = float(fallback_ring_to_radius[ring_id])
        return radius * math.cos(theta_rad), radius * math.sin(theta_rad)

    def row_theta_rad(row: dict[str, str]) -> float:
        x0, y0 = row_position(row)
        if use_real_positions and (x0 != 0.0 or y0 != 0.0):
            return math.atan2(y0, x0)
        ring_id = _to_int(row.get("ring_id"))
        return 2.0 * math.pi * _to_int(row.get("theta_id")) / count_hint_by_ring[ring_id]

    def row_phi(row: dict[str, str]) -> float:
        raw = row.get("nominal_phi_rad")
        if raw not in (None, ""):
            return _to_float(raw)
        return row_theta_rad(row) + 0.5 * math.pi

    ring_to_radius = {
        ring_id: float(
            np.mean(
                [
                    math.hypot(*row_position(row))
                    for row in layer_rows
                    if _to_int(row.get("ring_id")) == ring_id
                ]
            )
        )
        for ring_id in ring_ids
    }

    def ring_pitch(row: dict[str, str]) -> float:
        ring_id = _to_int(row.get("ring_id"))
        radius = max(ring_to_radius[ring_id], 1.0e-12)
        return 2.0 * math.pi * radius / count_hint_by_ring[ring_id]

    pitch_values = [ring_pitch(row) for row in layer_rows] or [1.0]
    min_pitch = max(1.0e-12, min(pitch_values))
    radius_values = sorted(set(ring_to_radius.values()))
    radial_gaps = [
        abs(upper - lower)
        for lower, upper in zip(radius_values, radius_values[1:], strict=False)
        if abs(upper - lower) > 1.0e-12
    ]
    min_radial_gap = min(radial_gaps) if radial_gaps else min_pitch
    dimensions_m = _magnet_dimensions_m(bundle)

    if use_real_positions and dimensions_m is not None:
        magnet_long = float(dimensions_m[0])
        magnet_short = float(dimensions_m[1])
    else:
        magnet_long = min(0.72 * min_pitch, 0.82 * min_radial_gap)
        magnet_short = min(0.42 * min_pitch, 0.46 * min_radial_gap)
        if magnet_short >= magnet_long:
            magnet_short = 0.58 * magnet_long
    magnet_long = max(magnet_long, 1.0e-9)
    magnet_short = max(magnet_short, 1.0e-9)
    label_offset = max(0.60 * magnet_short, 0.035 * max(radius_values or [1.0]))

    max_abs_epsilon = max(
        [abs(_to_float(row.get("epsilon_parallel"))) for row in completed_rows] + [1.0e-12]
    )

    def strength_color(row: dict[str, str]) -> str:
        epsilon = _to_float(row.get("epsilon_parallel"))
        normalized = 0.5 + 0.5 * max(-1.0, min(1.0, epsilon / max_abs_epsilon))
        return str(sample_colorscale(_STRENGTH_COLOR_VALUES, [normalized])[0])

    def local_axes(row: dict[str, str]) -> tuple[float, float, float, float]:
        phi = row_phi(row)
        ux, uy = math.cos(phi), math.sin(phi)
        vx, vy = -uy, ux
        return ux, uy, vx, vy

    def rectangle_xy(row: dict[str, str]) -> tuple[list[float], list[float]]:
        x0, y0 = row_position(row)
        ux, uy, vx, vy = local_axes(row)
        half_long = 0.5 * magnet_long
        half_short = 0.5 * magnet_short
        corners = [
            (half_long, half_short),
            (-half_long, half_short),
            (-half_long, -half_short),
            (half_long, -half_short),
            (half_long, half_short),
        ]
        xs = [x0 + a * ux + b * vx for a, b in corners]
        ys = [y0 + a * uy + b * vy for a, b in corners]
        return xs, ys

    rectangle_by_slot = {_to_int(row.get("slot_flat_id")): rectangle_xy(row) for row in layer_rows}

    def frame_label_position(row: dict[str, str]) -> tuple[float, float]:
        x0, y0 = row_position(row)
        _ux, _uy, vx, vy = local_axes(row)
        return x0 - label_offset * vx, y0 - label_offset * vy

    def orientation_label_position(row: dict[str, str]) -> tuple[float, float]:
        x0, y0 = row_position(row)
        _ux, _uy, vx, vy = local_axes(row)
        return x0 + label_offset * vx, y0 + label_offset * vy

    def hover_text(row: dict[str, str], *, pending: bool) -> str:
        x0, y0 = row_position(row)
        phi_deg = math.degrees(row_phi(row))
        radius = math.hypot(x0, y0)
        if pending:
            return (
                f"pending<br>ring {row.get('ring_id')}<br>"
                f"frame {row.get('physical_slot_number')}<br>"
                f"nominal phi {phi_deg:.3f} deg<br>"
                f"center radius {radius:.6g}"
            )
        return (
            f"magnet {row.get('magnet_id')}<br>"
            f"ring {row.get('ring_id')}<br>"
            f"cluster {row.get('cluster_requested')}<br>"
            f"frame {row.get('physical_slot_number')}<br>"
            f"orientation {row.get('orientation_id')} "
            f"({_orientation_pattern_label(row.get('orientation_id'))})<br>"
            f"epsilon {row.get('epsilon_parallel')}<br>"
            f"measured epsilon {row.get('measured_epsilon_parallel', '')}<br>"
            f"nominal phi {phi_deg:.3f} deg<br>"
            f"center radius {radius:.6g}"
        )

    fig = go.Figure()

    for ring_id in ring_ids:
        radius = float(ring_to_radius[ring_id])
        theta = np.linspace(0.0, 2.0 * math.pi, 160)
        fig.add_trace(
            go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode="lines",
                name=f"R{ring_id}",
                line={"color": "rgba(100, 116, 139, 0.24)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for idx, row in enumerate(pending_rows):
        xs, ys = rectangle_by_slot[_to_int(row.get("slot_flat_id"))]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                name="pending",
                text=hover_text(row, pending=True),
                hoverinfo="text",
                fillcolor="rgba(148, 163, 184, 0.24)",
                line={"color": "rgba(148, 163, 184, 0.65)", "width": 1},
                showlegend=idx == 0,
            )
        )

    for idx, row in enumerate(completed_rows):
        xs, ys = rectangle_by_slot[_to_int(row.get("slot_flat_id"))]
        is_current = state.current_slot_flat_id == _to_int(row.get("slot_flat_id"))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                name="completed",
                text=hover_text(row, pending=False),
                hoverinfo="text",
                fillcolor=strength_color(row),
                line={"color": "black", "width": 3 if is_current else 1},
                showlegend=idx == 0,
            )
        )

    if completed_rows:
        positions = [row_position(row) for row in completed_rows]
        fig.add_trace(
            go.Scatter(
                x=[position[0] for position in positions],
                y=[position[1] for position in positions],
                mode="markers",
                marker={
                    "color": [_to_float(row.get("epsilon_parallel")) for row in completed_rows],
                    "colorscale": _STRENGTH_COLORSCALE,
                    "cmin": -max_abs_epsilon,
                    "cmax": max_abs_epsilon,
                    "size": 0.1,
                    "opacity": 0.0,
                    "colorbar": {"title": "epsilon"},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    label_rows = layer_rows
    frame_label_positions = [frame_label_position(row) for row in label_rows]
    fig.add_trace(
        go.Scatter(
            x=[position[0] for position in frame_label_positions],
            y=[position[1] for position in frame_label_positions],
            mode="text",
            text=[str(row.get("physical_slot_number") or "") for row in label_rows],
            textfont={"size": 10, "color": "#111827"},
            hoverinfo="skip",
            showlegend=False,
            name="frame_numbers",
        )
    )
    orientation_label_rows = [
        row for row in completed_rows if _orientation_pattern_label(row.get("orientation_id"))
    ]
    orientation_label_positions = [
        orientation_label_position(row) for row in orientation_label_rows
    ]
    if orientation_label_rows:
        fig.add_trace(
            go.Scatter(
                x=[position[0] for position in orientation_label_positions],
                y=[position[1] for position in orientation_label_positions],
                mode="text",
                text=[
                    _orientation_pattern_label(row.get("orientation_id"))
                    for row in orientation_label_rows
                ],
                textfont={"size": 11, "color": "#0f172a"},
                hoverinfo="skip",
                showlegend=False,
                name="orientation_patterns",
            )
        )

    bound_x: list[float] = []
    bound_y: list[float] = []
    for xs, ys in rectangle_by_slot.values():
        bound_x.extend(xs)
        bound_y.extend(ys)
    bound_x.extend(position[0] for position in frame_label_positions + orientation_label_positions)
    bound_y.extend(position[1] for position in frame_label_positions + orientation_label_positions)
    if bound_x and bound_y:
        x_min = min(bound_x)
        x_max = max(bound_x)
        y_min = min(bound_y)
        y_max = max(bound_y)
        span = max(x_max - x_min, y_max - y_min, 1.0e-12)
        pad = 0.05 * span
        x_range = [x_min - pad, x_max + pad]
        y_range = [y_min - pad, y_max + pad]
    else:
        x_range = None
        y_range = None

    dimension_source = _visualization_geometry(bundle).get("magnet_dimensions_source", "")
    subtitle = "real center geometry" if use_real_positions else "fallback geometry"
    if dimension_source:
        subtitle = f"{subtitle}, dimensions: {dimension_source}"
    fig.update_layout(
        title=f"Active Stack {state.active_layer_id} R Layers ({subtitle})",
        xaxis={
            "visible": False,
            "scaleanchor": "y",
            "scaleratio": 1,
            **({} if x_range is None else {"range": x_range}),
        },
        yaxis={"visible": False, **({} if y_range is None else {"range": y_range})},
        showlegend=True,
        height=680,
        margin={"l": 12, "r": 12, "b": 12, "t": 58},
    )
    return fig


def plot_ring_metric_heatmap(bundle: RingTrialBundle, metric: str) -> go.Figure:
    """Plot any scalar metric from ring_summary_trial_XXX.csv."""
    matrix = ring_metric_matrix(bundle, metric)
    fig = go.Figure(
        go.Heatmap(
            x=matrix.x,
            y=matrix.y,
            z=matrix.z,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": metric},
        )
    )
    fig.update_layout(
        title=f"Ring {metric}",
        xaxis_title="ring_id",
        yaxis_title="layer_id",
        height=420,
        margin={"l": 50, "r": 10, "b": 40, "t": 45},
    )
    return fig


def plot_mirror_pair_imbalance(
    bundle: RingTrialBundle,
    metric: Literal["mean_epsilon_difference", "mean_angle_error_difference"],
) -> go.Figure:
    """Plot mirror-pair imbalance rows."""
    labels = [str(row.get("pair_id") or "") for row in bundle.ring_pair_summary]
    values = [_to_float(row.get(metric)) for row in bundle.ring_pair_summary]
    fig = go.Figure(go.Bar(x=labels, y=values, marker={"color": "#2563eb"}))
    fig.update_layout(
        title=f"Mirror Pair {metric}",
        xaxis_title="pair_id",
        yaxis_title=metric,
        height=320,
        margin={"l": 50, "r": 10, "b": 80, "t": 45},
    )
    return fig


def plot_cluster_inventory_heatmap(
    bundle: RingTrialBundle,
    mode: ClusterMatrixMode,
) -> go.Figure:
    """Plot cluster inventory/usage as angle-bin x strength-bin heatmap."""
    matrix = cluster_inventory_matrix(bundle, mode)
    fig = go.Figure(
        go.Heatmap(
            x=matrix.x,
            y=matrix.y,
            z=matrix.z,
            colorscale="Viridis",
            colorbar={"title": mode},
        )
    )
    fig.update_layout(
        title=f"Cluster {mode}",
        xaxis_title="strength bin",
        yaxis_title="angle bin",
        height=420,
        margin={"l": 50, "r": 10, "b": 40, "t": 45},
    )
    return fig


__all__ = [
    "ClusterMatrixMode",
    "MatrixPayload",
    "PlaybackState",
    "RingTrialBundle",
    "cluster_inventory_counts",
    "cluster_inventory_matrix",
    "discover_trial_ids",
    "load_ring_trial_bundle",
    "playback_state_at_step",
    "plot_active_ring_polar_view",
    "plot_cluster_inventory_heatmap",
    "plot_mirror_pair_imbalance",
    "plot_ring_metric_heatmap",
    "plot_side_stack_view",
    "ring_completion_steps",
    "ring_metric_matrix",
]
