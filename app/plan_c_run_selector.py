from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL_RUN_CHOICE = "(manual path)"


def _relative_or_absolute(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def list_plan_c_run_result_choices(runs_dir: str | Path | None = None) -> list[str]:
    """Return selectable result files under runs/, relative to the repo root when possible."""
    base = ROOT / "runs" if runs_dir is None else Path(runs_dir)
    if not base.is_absolute():
        base = ROOT / base
    if not base.is_dir():
        return []

    results: set[Path] = set()
    results.update(path for path in base.rglob("results.npz") if path.is_file())
    results.update(path for path in base.rglob("*results*.npz") if path.is_file())
    return sorted(_relative_or_absolute(path, root=ROOT) for path in results)


def resolve_selected_run_path(selected: str, manual_path: str) -> str:
    """Resolve a selectbox choice and manual fallback into the path passed to Plan C loaders."""
    choice = selected.strip()
    manual = manual_path.strip()
    if choice and choice != MANUAL_RUN_CHOICE:
        return choice
    return manual


def default_plan_c_child_output_dir(run_path: str, child_name: str) -> str:
    """Return a stable default output directory next to the selected run result."""
    raw = run_path.strip()
    if not raw:
        return str(Path("runs") / child_name)
    path = Path(raw)
    base = path.parent if path.suffix.lower() == ".npz" else path
    return str(base / child_name)


__all__ = [
    "MANUAL_RUN_CHOICE",
    "default_plan_c_child_output_dir",
    "list_plan_c_run_result_choices",
    "resolve_selected_run_path",
]
