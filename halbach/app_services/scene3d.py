from __future__ import annotations

import numpy as np

from halbach.run_types import RunBundle
from halbach.viz3d import compute_scene_ranges, enumerate_magnets_with_directions

from .runs import load_run_bundle, path_to_display
from .types import Scene3DPayload, Scene3DRequest, Scene3DRunPayload


def _scene_run_payload(
    run: RunBundle,
    *,
    stride: int,
    hide_x_negative: bool,
) -> Scene3DRunPayload:
    centers, _phi, directions, ring_ids, layer_ids = enumerate_magnets_with_directions(
        run,
        stride=stride,
        hide_x_negative=hide_x_negative,
    )
    return Scene3DRunPayload(
        run_path=path_to_display(run.run_dir),
        run_name=run.name,
        centers=np.asarray(centers, dtype=np.float64),
        direction_vectors=np.asarray(directions, dtype=np.float64),
        ring_ids=np.asarray(ring_ids, dtype=np.int_),
        layer_ids=np.asarray(layer_ids, dtype=np.int_),
    )


def load_scene3d_payload(
    request: Scene3DRequest,
    *,
    primary_run: RunBundle | None = None,
    secondary_run: RunBundle | None = None,
) -> Scene3DPayload:
    resolved_primary = load_run_bundle(request.primary_path) if primary_run is None else primary_run
    resolved_secondary = (
        None
        if request.secondary_path is None
        else load_run_bundle(request.secondary_path) if secondary_run is None else secondary_run
    )

    primary_payload = _scene_run_payload(
        resolved_primary,
        stride=request.stride,
        hide_x_negative=request.hide_x_negative,
    )
    secondary_payload = (
        None
        if resolved_secondary is None
        else _scene_run_payload(
            resolved_secondary,
            stride=request.stride,
            hide_x_negative=request.hide_x_negative,
        )
    )
    range_sources = [primary_payload.centers]
    if secondary_payload is not None:
        range_sources.append(secondary_payload.centers)
    scene_ranges = compute_scene_ranges(range_sources)
    view_metadata: dict[str, object] = {
        "stride": int(request.stride),
        "hide_x_negative": bool(request.hide_x_negative),
        "compare": secondary_payload is not None,
    }
    return Scene3DPayload(
        primary=primary_payload,
        secondary=secondary_payload,
        scene_ranges=scene_ranges,
        view_metadata=view_metadata,
    )


__all__ = ["load_scene3d_payload"]
