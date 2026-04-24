from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from halbach.app_services import (
    build_run_summary,
    list_run_candidates,
    load_map2d_payload,
    load_run_bundle,
    load_scene3d_payload,
    run_file_fingerprint,
)
from halbach.app_services.types import Map2DRequest, Scene3DRequest


class OverviewRequestModel(BaseModel):
    run_path: str


class Map2DRequestModel(BaseModel):
    run_path: str
    roi_r: float = Field(default=0.14, gt=0.0)
    step: float = Field(default=0.001, gt=0.0)
    mag_model_eval: Literal["auto", "fixed", "self-consistent-easy-axis"] = "auto"
    sc_cfg_override: dict[str, object] | None = None
    plane: Literal["xy", "xz", "yz"] = "xy"
    coord0: float = 0.0


class Scene3DRequestModel(BaseModel):
    primary_path: str
    secondary_path: str | None = None
    stride: int = Field(default=1, ge=1)
    hide_x_negative: bool = False


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _serialize_run_summary(run_path: str) -> dict[str, object]:
    summary = build_run_summary(load_run_bundle(run_path))
    return {
        "name": summary.name,
        "run_path": summary.run_path,
        "results_path": summary.results_path,
        "meta_path": summary.meta_path,
        "trace_path": summary.trace_path,
        "framework": summary.framework,
        "geometry_summary": _jsonable(summary.geometry_summary),
        "key_stats": _jsonable(summary.key_stats),
        "magnetization_debug": _jsonable(summary.magnetization_debug),
    }


def _serialize_map2d_payload(request: Map2DRequest) -> dict[str, object]:
    payload = load_map2d_payload(request)
    return {
        "run_path": payload.run_path,
        "xs": _jsonable(payload.xs),
        "ys": _jsonable(payload.ys),
        "ppm": _jsonable(payload.ppm),
        "mask": _jsonable(payload.mask),
        "B0_T": payload.B0_T,
        "plane": payload.plane,
        "coord0": payload.coord0,
        "summary_stats": _jsonable(payload.summary_stats),
        "magnetization_debug": _jsonable(payload.magnetization_debug),
    }


def _serialize_scene3d_payload(request: Scene3DRequest) -> dict[str, object]:
    payload = load_scene3d_payload(request)
    secondary = None
    if payload.secondary is not None:
        secondary = {
            "run_path": payload.secondary.run_path,
            "run_name": payload.secondary.run_name,
            "centers": _jsonable(payload.secondary.centers),
            "direction_vectors": _jsonable(payload.secondary.direction_vectors),
            "ring_ids": _jsonable(payload.secondary.ring_ids),
            "layer_ids": _jsonable(payload.secondary.layer_ids),
        }
    return {
        "primary": {
            "run_path": payload.primary.run_path,
            "run_name": payload.primary.run_name,
            "centers": _jsonable(payload.primary.centers),
            "direction_vectors": _jsonable(payload.primary.direction_vectors),
            "ring_ids": _jsonable(payload.primary.ring_ids),
            "layer_ids": _jsonable(payload.primary.layer_ids),
        },
        "secondary": secondary,
        "scene_ranges": _jsonable(payload.scene_ranges),
        "view_metadata": _jsonable(payload.view_metadata),
    }


@lru_cache(maxsize=128)
def _cached_overview(run_path: str, results_stamp: float, meta_stamp: float) -> dict[str, object]:
    _ = (results_stamp, meta_stamp)
    return _serialize_run_summary(run_path)


@lru_cache(maxsize=128)
def _cached_map2d(
    run_path: str,
    results_stamp: float,
    meta_stamp: float,
    roi_r: float,
    step: float,
    mag_model_eval: str,
    sc_cfg_key: str,
    plane: str,
    coord0: float,
) -> dict[str, object]:
    _ = (results_stamp, meta_stamp)
    sc_cfg_override = None if not sc_cfg_key else json.loads(sc_cfg_key)
    return _serialize_map2d_payload(
        Map2DRequest(
            run_path=run_path,
            roi_r=roi_r,
            step=step,
            mag_model_eval=cast_mag_model_eval(mag_model_eval),
            sc_cfg_override=sc_cfg_override,
            plane=cast_plane(plane),
            coord0=coord0,
        )
    )


@lru_cache(maxsize=128)
def _cached_scene3d(
    primary_path: str,
    primary_results_stamp: float,
    primary_meta_stamp: float,
    secondary_path: str,
    secondary_results_stamp: float,
    secondary_meta_stamp: float,
    stride: int,
    hide_x_negative: bool,
) -> dict[str, object]:
    _ = (
        primary_results_stamp,
        primary_meta_stamp,
        secondary_results_stamp,
        secondary_meta_stamp,
    )
    return _serialize_scene3d_payload(
        Scene3DRequest(
            primary_path=primary_path,
            secondary_path=secondary_path or None,
            stride=stride,
            hide_x_negative=hide_x_negative,
        )
    )


def cast_mag_model_eval(value: str) -> Literal["auto", "fixed", "self-consistent-easy-axis"]:
    if value not in ("auto", "fixed", "self-consistent-easy-axis"):
        raise ValueError(f"Unsupported mag_model_eval: {value}")
    return cast(Literal["auto", "fixed", "self-consistent-easy-axis"], value)


def cast_plane(value: str) -> Literal["xy", "xz", "yz"]:
    if value not in ("xy", "xz", "yz"):
        raise ValueError(f"Unsupported plane: {value}")
    return cast(Literal["xy", "xz", "yz"], value)


def create_app(*, runs_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Halbach GUI Backend", version="0.1.0")
    app.state.runs_dir = runs_dir

    @app.get("/api/runs")
    def get_runs() -> dict[str, list[dict[str, str]]]:
        candidates = list_run_candidates(app.state.runs_dir)
        return {
            "runs": [
                {
                    "path": candidate.path,
                    "label": candidate.label,
                }
                for candidate in candidates
            ]
        }

    @app.post("/api/run/overview")
    def get_run_overview(request: OverviewRequestModel) -> dict[str, object]:
        try:
            results_stamp, meta_stamp = run_file_fingerprint(request.run_path)
            return _cached_overview(request.run_path, results_stamp, meta_stamp)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/run/map2d")
    def get_run_map2d(request: Map2DRequestModel) -> dict[str, object]:
        try:
            results_stamp, meta_stamp = run_file_fingerprint(request.run_path)
            sc_cfg_key = (
                ""
                if request.sc_cfg_override is None
                else json.dumps(request.sc_cfg_override, sort_keys=True, default=str)
            )
            return _cached_map2d(
                request.run_path,
                results_stamp,
                meta_stamp,
                request.roi_r,
                request.step,
                request.mag_model_eval,
                sc_cfg_key,
                request.plane,
                request.coord0,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/run/scene3d")
    def get_run_scene3d(request: Scene3DRequestModel) -> dict[str, object]:
        try:
            primary_results_stamp, primary_meta_stamp = run_file_fingerprint(request.primary_path)
            if request.secondary_path:
                secondary_results_stamp, secondary_meta_stamp = run_file_fingerprint(
                    request.secondary_path
                )
            else:
                secondary_results_stamp, secondary_meta_stamp = 0.0, 0.0
            return _cached_scene3d(
                request.primary_path,
                primary_results_stamp,
                primary_meta_stamp,
                request.secondary_path or "",
                secondary_results_stamp,
                secondary_meta_stamp,
                request.stride,
                request.hide_x_negative,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()

__all__ = ["app", "create_app"]
