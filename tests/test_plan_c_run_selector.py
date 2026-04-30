from pathlib import Path

from app.plan_c_run_selector import (
    MANUAL_RUN_CHOICE,
    default_plan_c_child_output_dir,
    list_plan_c_run_result_choices,
    resolve_selected_run_path,
)


def test_list_plan_c_run_result_choices_finds_results_files(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "opt_a").mkdir(parents=True)
    (runs_dir / "opt_b" / "nested").mkdir(parents=True)
    (runs_dir / "ignored").mkdir(parents=True)
    (runs_dir / "opt_a" / "results.npz").write_bytes(b"npz")
    (runs_dir / "opt_b" / "nested" / "custom_results_final.npz").write_bytes(b"npz")
    (runs_dir / "ignored" / "notes.txt").write_text("nope", encoding="utf-8")

    choices = list_plan_c_run_result_choices(runs_dir)

    assert str(runs_dir / "opt_a" / "results.npz") in choices
    assert str(runs_dir / "opt_b" / "nested" / "custom_results_final.npz") in choices
    assert all("notes.txt" not in choice for choice in choices)


def test_resolve_selected_run_path_prefers_selection_over_manual() -> None:
    assert resolve_selected_run_path("runs/opt/results.npz", "manual/path") == "runs/opt/results.npz"
    assert resolve_selected_run_path(MANUAL_RUN_CHOICE, "manual/path") == "manual/path"


def test_default_plan_c_child_output_dir_uses_result_parent() -> None:
    assert default_plan_c_child_output_dir("runs/opt/results.npz", "plan_c_sim") == str(
        Path("runs") / "opt" / "plan_c_sim"
    )
    assert default_plan_c_child_output_dir("runs/opt", "plan_c_session") == str(
        Path("runs") / "opt" / "plan_c_session"
    )
