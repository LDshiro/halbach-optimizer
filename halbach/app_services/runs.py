from __future__ import annotations

import json
from pathlib import Path

from halbach.run_io import load_run
from halbach.run_types import RunBundle

from .types import RunCandidate

ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return ROOT


def default_runs_dir() -> Path:
    return ROOT / "runs"


def list_run_candidates(runs_dir: Path | None = None) -> list[RunCandidate]:
    target_dir = default_runs_dir() if runs_dir is None else runs_dir
    if not target_dir.is_dir():
        return []

    candidates: list[RunCandidate] = []
    for entry in sorted(target_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not ((entry / "results.npz").is_file() or any(entry.glob("*results*.npz"))):
            continue
        label = path_to_display(entry)
        candidates.append(RunCandidate(path=label, label=label))
    return candidates


def resolve_selected_path(selected: str, manual: str) -> str:
    manual_clean = manual.strip()
    if manual_clean:
        return manual_clean
    return selected.strip()


def resolve_run_path(path_text: str | Path) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def resolve_results_path(path: Path) -> Path:
    if path.is_dir():
        primary = path / "results.npz"
        if primary.is_file():
            return primary
        matches = sorted(path.glob("*results*.npz"))
        if not matches:
            raise FileNotFoundError(f"No results .npz found in {path}")
        if len(matches) > 1:
            items = ", ".join(str(item) for item in matches)
            raise ValueError(f"Multiple results files found in {path}: {items}")
        return matches[0]
    if path.is_file() and path.suffix == ".npz":
        return path
    raise FileNotFoundError(f"Run path not found: {path}")


def results_mtime(path: Path) -> float:
    return float(resolve_results_path(path).stat().st_mtime)


def meta_mtime(path: Path) -> float:
    if path.is_dir():
        primary = path / "meta.json"
        if primary.is_file():
            return float(primary.stat().st_mtime)
        matches = sorted(path.glob("*meta*.json"))
        if len(matches) == 1:
            return float(matches[0].stat().st_mtime)
        return 0.0
    if path.is_file() and path.suffix == ".npz":
        candidate = path.with_name("meta.json")
        if candidate.is_file():
            return float(candidate.stat().st_mtime)
        matches = sorted(path.parent.glob("*meta*.json"))
        if len(matches) == 1:
            return float(matches[0].stat().st_mtime)
    return 0.0


def run_file_fingerprint(path_text: str | Path) -> tuple[float, float]:
    resolved = resolve_run_path(path_text)
    return results_mtime(resolved), meta_mtime(resolved)


def load_run_bundle(path_text: str | Path) -> RunBundle:
    return load_run(resolve_run_path(path_text))


def try_load_run_bundle(
    path_text: str,
) -> tuple[RunBundle | None, float | None, float | None, str | None]:
    if not path_text:
        return None, None, None, None
    try:
        resolved = resolve_run_path(path_text)
        result_mtime, meta_time = run_file_fingerprint(resolved)
        run = load_run_bundle(resolved)
        return run, result_mtime, meta_time, None
    except Exception as exc:
        return None, None, None, str(exc)


def path_to_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def magnetization_cache_key(meta: dict[str, object]) -> str:
    magnetization = meta.get("magnetization", {})
    try:
        return json.dumps(magnetization, sort_keys=True, default=str)
    except Exception:
        return ""


__all__ = [
    "ROOT",
    "default_runs_dir",
    "list_run_candidates",
    "load_run_bundle",
    "magnetization_cache_key",
    "meta_mtime",
    "path_to_display",
    "repo_root",
    "resolve_results_path",
    "resolve_run_path",
    "resolve_selected_path",
    "results_mtime",
    "run_file_fingerprint",
    "try_load_run_bundle",
]
