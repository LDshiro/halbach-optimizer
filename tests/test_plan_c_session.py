from pathlib import Path

from halbach.assembly.clustering import assign_quantile_clusters
from halbach.assembly.inventory import build_cluster_inventory, inventory_total_count
from halbach.assembly.measurement import SyntheticMeasurementProvider
from halbach.assembly.online_assignment import run_linear_sensitivity_assignment
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.session import PlanCSession, run_session_to_completion
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.variation import generate_virtual_magnets
from halbach.cli.plan_c_session import main as session_main
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"session_run_N{N}_R{R}_K{K}"
    results = generate_halbach_initial(
        N=N,
        R=R,
        end_R=None,
        end_layers_per_side=0,
        K=K,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-session-test",
        schema_version=1,
        generator_params={},
        description="plan c session test run",
    )
    return load_run(run_dir)


def _case(tmp_path: Path):
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0005,
        measurement_noise=None,
    )
    assignments = assign_quantile_clusters(magnets, strength_count=2, angle_count=2)
    inventory = build_cluster_inventory(magnets, assignments)
    return table, magnets, assignments, inventory


def test_session_state_transitions_to_insert_confirmation(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )

    assert session.state == "PREPARE_SESSION"
    session.step()
    assert (session.state, session.sub_state) == ("SELECT_WORK_UNIT", "CHOOSE_CLUSTER")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "WAIT_FOR_MAGNET_MEASUREMENT")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "VALIDATE_MEASUREMENT")
    session.step()
    assert (session.state, session.sub_state) == ("BUILD_WORK_UNIT", "SOLVE_PLACEMENT")
    session.step()
    assert session.state == "EVALUATE_MIRROR_PAIR_SWAP"
    session.step()
    assert session.state == "INSTALL_OR_CONFIRM_PAIR"


def test_step_by_step_session_matches_auto_linear_assignment(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    auto = run_linear_sensitivity_assignment(
        table,
        magnets,
        assignments=assignments,
        inventory=inventory,
    )
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )

    final = run_session_to_completion(session)

    assert final.state == "COMPLETE"
    assert final.placements == auto.placements
    assert inventory_total_count(session.current_inventory) == 0


def test_session_undo_restores_last_pending_insert(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )
    while session.state != "INSTALL_OR_CONFIRM_PAIR":
        session.step()
    pending_slot = session.pending_candidate.slot_flat_id
    session.confirm_insert()

    assert len(session.placements) == 1
    session.undo_last_insert()

    assert len(session.placements) == 0
    assert pending_slot in session.remaining_slot_flat_ids
    assert session.pending_candidate is not None
    assert session.state == "INSTALL_OR_CONFIRM_PAIR"


def test_session_resume_from_log_completes_without_duplicate_slots(tmp_path: Path) -> None:
    table, magnets, assignments, inventory = _case(tmp_path)
    log_path = tmp_path / "session.jsonl"
    session = PlanCSession(
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
        log_path=log_path,
    )
    while len(session.placements) < 1:
        if session.state == "INSTALL_OR_CONFIRM_PAIR":
            session.confirm_insert()
        else:
            session.step()

    resumed = PlanCSession.resume_from_log(
        log_path,
        sensitivity_table=table,
        provider=SyntheticMeasurementProvider(magnets),
        assignments=assignments,
        inventory=inventory,
    )
    final = run_session_to_completion(resumed)
    slot_ids = [placement.slot_flat_id for placement in final.placements]

    assert len(slot_ids) == len(set(slot_ids))
    assert len(slot_ids) == len(magnets)
    assert final.state == "COMPLETE"


def test_plan_c_session_cli_writes_log_and_final_placement(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    out_dir = tmp_path / "session_cli"

    session_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_dir),
            "--seed",
            "5",
            "--strength-sigma",
            "0.001",
            "--direction-sigma",
            "0.0005",
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "5",
        ]
    )

    assert (out_dir / "session_log.jsonl").exists()
    assert (out_dir / "placement_final.csv").exists()
    assert (out_dir / "session_summary.json").exists()
