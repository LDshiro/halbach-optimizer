from __future__ import annotations

import numpy as np

from halbach.run_types import RunBundle
from halbach.viz2d import compute_error_map_ppm_plane_with_debug

from .runs import load_run_bundle, path_to_display
from .types import Map2DPayload, Map2DRequest


def load_map2d_payload(
    request: Map2DRequest,
    *,
    run: RunBundle | None = None,
) -> Map2DPayload:
    run_bundle = load_run_bundle(request.run_path) if run is None else run
    error_map, debug = compute_error_map_ppm_plane_with_debug(
        run_bundle,
        plane=request.plane,
        coord0=request.coord0,
        roi_r=request.roi_r,
        step=request.step,
        mag_model_eval=request.mag_model_eval,
        sc_cfg_override=request.sc_cfg_override,
    )
    ppm_values = np.asarray(error_map.ppm, dtype=np.float64)
    valid_ppm = ppm_values[np.asarray(error_map.mask, dtype=np.bool_)]
    summary_stats = {
        "ppm_mean": float(np.nanmean(ppm_values)),
        "ppm_max_abs": float(np.nanmax(np.abs(ppm_values))),
        "ppm_valid_min": float(np.min(valid_ppm)),
        "ppm_valid_max": float(np.max(valid_ppm)),
    }
    return Map2DPayload(
        run_path=path_to_display(run_bundle.run_dir),
        xs=np.asarray(error_map.xs, dtype=np.float64),
        ys=np.asarray(error_map.ys, dtype=np.float64),
        ppm=ppm_values,
        mask=np.asarray(error_map.mask, dtype=np.bool_),
        B0_T=float(error_map.B0_T),
        plane=error_map.plane,
        coord0=float(error_map.coord0),
        summary_stats=summary_stats,
        magnetization_debug=debug,
    )


__all__ = ["load_map2d_payload"]
