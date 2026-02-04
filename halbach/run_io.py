from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from halbach.run_types import RunBundle, RunResults
from halbach.solvers.types import SolveTrace
from halbach.types import FloatArray, Geometry

_RESULTS_PRIMARY = "results.npz"
_RESULTS_GLOB = "*results*.npz"
_META_PRIMARY = "meta.json"
_META_GLOB = "*meta*.json"
_TRACE_NAME = "trace.json"

_KNOWN_RESULT_KEYS = {
    "alphas_opt",
    "alphas",
    "r_bases_opt",
    "r_bases",
    "theta",
    "sin2th",
    "sin2",
    "cth",
    "sth",
    "z_layers",
    "ring_offsets",
}


def _to_float_array(value: Any) -> FloatArray:
    return np.array(value, dtype=np.float64, copy=True)


def _to_float_array_1d(value: Any) -> FloatArray:
    return np.array(value, dtype=np.float64, copy=True).reshape(-1)


def _pick_first_key(data: np.lib.npyio.NpzFile, keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in data.files:
            return key
    return None


def _load_required_1d(data: np.lib.npyio.NpzFile, keys: Sequence[str], label: str) -> FloatArray:
    key = _pick_first_key(data, keys)
    if key is None:
        raise KeyError(f"Missing required key for {label}: {', '.join(keys)}")
    return _to_float_array_1d(data[key])


def _load_required_2d(data: np.lib.npyio.NpzFile, keys: Sequence[str], label: str) -> FloatArray:
    key = _pick_first_key(data, keys)
    if key is None:
        raise KeyError(f"Missing required key for {label}: {', '.join(keys)}")
    arr = _to_float_array(data[key])
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {arr.shape}")
    return arr


def _resolve_required_npz(run_dir: Path) -> Path:
    primary = run_dir / _RESULTS_PRIMARY
    if primary.is_file():
        return primary
    matches = sorted(run_dir.glob(_RESULTS_GLOB))
    if not matches:
        raise FileNotFoundError(f"No results .npz found in {run_dir}")
    if len(matches) > 1:
        items = ", ".join(str(p) for p in matches)
        raise ValueError(f"Multiple results files found in {run_dir}: {items}")
    return matches[0]


def _resolve_optional_json(
    run_dir: Path,
    primary: str,
    pattern: str,
    label: str,
    *,
    preferred: Path | None = None,
) -> Path | None:
    primary_path = run_dir / primary
    if primary_path.is_file():
        return primary_path
    if preferred is not None and preferred.is_file():
        return preferred
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        items = ", ".join(str(p) for p in matches)
        raise ValueError(f"Multiple {label} files found in {run_dir}: {items}")
    return matches[0]


def _replace_tail(token: str, old: str, new: str) -> str:
    head, sep, tail = token.rpartition(old)
    if not sep:
        return token
    return f"{head}{new}{tail}"


def _guess_meta_path(results_path: Path) -> Path | None:
    stem = results_path.stem
    if "results" in stem:
        candidate_stem = _replace_tail(stem, "results", "meta")
    elif "result" in stem:
        candidate_stem = _replace_tail(stem, "result", "meta")
    else:
        candidate_stem = f"{stem}_meta"
    return results_path.with_name(f"{candidate_stem}.json")


def _load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], data)


def _load_trace(path: Path) -> SolveTrace:
    raw = _load_json_dict(path)
    iters_raw = raw.get("iters")
    f_raw = raw.get("f")
    gnorm_raw = raw.get("gnorm")
    extras_raw = raw.get("extras")

    if not isinstance(iters_raw, list):
        raise ValueError(f"trace.json missing 'iters' list in {path}")
    if not isinstance(f_raw, list):
        raise ValueError(f"trace.json missing 'f' list in {path}")
    if not isinstance(gnorm_raw, list):
        raise ValueError(f"trace.json missing 'gnorm' list in {path}")
    if not isinstance(extras_raw, list):
        raise ValueError(f"trace.json missing 'extras' list in {path}")

    iters = [int(v) for v in iters_raw]
    f_vals = [float(v) for v in f_raw]
    gnorm = [float(v) for v in gnorm_raw]

    extras: list[dict[str, Any]] = []
    for item in extras_raw:
        if not isinstance(item, dict):
            raise ValueError(f"trace.json extras must be list[dict] in {path}")
        extras.append(cast(dict[str, Any], item))

    return SolveTrace(iters=iters, f=f_vals, gnorm=gnorm, extras=extras)


def _load_results(results_path: Path) -> RunResults:
    with np.load(results_path, allow_pickle=True) as data:
        alphas = _load_required_2d(data, ["alphas_opt", "alphas"], "alphas")
        r_bases = _load_required_1d(data, ["r_bases_opt", "r_bases"], "r_bases")
        theta = _load_required_1d(data, ["theta"], "theta")
        z_layers = _load_required_1d(data, ["z_layers"], "z_layers")

        R, K = alphas.shape
        if r_bases.size != K:
            raise ValueError(f"r_bases length {r_bases.size} does not match K={K}")
        if z_layers.size != K:
            raise ValueError(f"z_layers length {z_layers.size} does not match K={K}")

        if "sin2th" in data.files:
            sin2th = _to_float_array_1d(data["sin2th"])
        elif "sin2" in data.files:
            sin2th = _to_float_array_1d(data["sin2"])
        else:
            sin2th = np.sin(2.0 * theta)
        if sin2th.size != theta.size:
            raise ValueError(f"sin2th length {sin2th.size} does not match N={theta.size}")

        if "cth" in data.files:
            cth = _to_float_array_1d(data["cth"])
        else:
            cth = np.cos(theta)
        if cth.size != theta.size:
            raise ValueError(f"cth length {cth.size} does not match N={theta.size}")

        if "sth" in data.files:
            sth = _to_float_array_1d(data["sth"])
        else:
            sth = np.sin(theta)
        if sth.size != theta.size:
            raise ValueError(f"sth length {sth.size} does not match N={theta.size}")

        if "ring_offsets" in data.files:
            ring_offsets = _to_float_array_1d(data["ring_offsets"])
        else:
            ring_offsets = np.zeros(R, dtype=np.float64)
        if ring_offsets.size == 1 and R > 1:
            ring_offsets = np.full(R, float(ring_offsets[0]), dtype=np.float64)
        if ring_offsets.size != R:
            raise ValueError(f"ring_offsets length {ring_offsets.size} does not match R={R}")

        extras: dict[str, Any] = {}
        for key in data.files:
            if key in _KNOWN_RESULT_KEYS:
                continue
            extras[key] = np.array(data[key], copy=True)

    return RunResults(
        alphas=alphas,
        r_bases=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        extras=extras,
    )


def _load_results_dc(results_path: Path, meta: dict[str, Any]) -> RunResults:
    geom = meta.get("geom", {})
    R = int(geom.get("R", 1))
    K = int(geom.get("K", 1))
    N = int(geom.get("N", 1))
    radius_m = float(geom.get("radius_m", 0.0))
    length_m = float(geom.get("length_m", 0.0))

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float64)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if K > 1:
        half = 0.5 * length_m
        z_layers = np.linspace(-half, half, K, dtype=np.float64)
    else:
        z_layers = np.array([0.0], dtype=np.float64)

    r_bases = np.full(K, radius_m, dtype=np.float64)
    alphas = np.zeros((R, K), dtype=np.float64)
    ring_offsets = np.zeros(R, dtype=np.float64)

    extras: dict[str, Any] = {}
    with np.load(results_path, allow_pickle=True) as data:
        for key in data.files:
            extras[key] = np.array(data[key], copy=True)

    return RunResults(
        alphas=alphas,
        r_bases=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        extras=extras,
    )


def _build_geometry(results: RunResults) -> Geometry:
    N = int(results.theta.size)
    R, K = results.alphas.shape
    z_layers = results.z_layers
    if z_layers.size > 1:
        dz = float(np.median(np.abs(np.diff(z_layers))))
        Lz = float(np.max(z_layers) - np.min(z_layers))
    else:
        dz = 0.0
        Lz = 0.0

    return Geometry(
        theta=results.theta,
        sin2=results.sin2th,
        cth=results.cth,
        sth=results.sth,
        z_layers=results.z_layers,
        ring_offsets=results.ring_offsets,
        N=N,
        K=K,
        R=R,
        dz=dz,
        Lz=Lz,
    )


def load_run(path: str | Path, *, name: str | None = None) -> RunBundle:
    run_path = Path(path).expanduser()

    if run_path.is_dir():
        run_dir = run_path
        results_path = _resolve_required_npz(run_dir)
    elif run_path.is_file():
        if run_path.suffix != ".npz":
            raise ValueError(f"Expected .npz file or run directory, got {run_path}")
        results_path = run_path
        run_dir = run_path.parent
    else:
        raise FileNotFoundError(f"Run path not found: {run_path}")

    meta_hint = _guess_meta_path(results_path)
    meta_path = _resolve_optional_json(
        run_dir,
        _META_PRIMARY,
        _META_GLOB,
        "meta",
        preferred=meta_hint,
    )
    trace_candidate = run_dir / _TRACE_NAME
    trace_path = trace_candidate if trace_candidate.is_file() else None
    meta = _load_json_dict(meta_path) if meta_path is not None else {}
    if meta.get("framework") == "dc":
        results = _load_results_dc(results_path, meta)
        geometry = _build_geometry(results)
        trace = None
    else:
        results = _load_results(results_path)
        geometry = _build_geometry(results)
        trace = _load_trace(trace_path) if trace_path is not None else None

    resolved_name = name
    if resolved_name is None:
        resolved_name = run_dir.name if run_path.is_dir() else results_path.stem

    return RunBundle(
        name=resolved_name,
        run_dir=run_dir,
        results_path=results_path,
        meta_path=meta_path,
        trace_path=trace_path,
        results=results,
        meta=meta,
        geometry=geometry,
        trace=trace,
    )


__all__ = ["load_run"]
