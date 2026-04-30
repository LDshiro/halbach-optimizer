from pathlib import Path

import numpy as np
import pytest

from halbach.assembly.field_eval import evaluate_fixed_placement
from halbach.assembly.self_consistent_assignment import (
    SelfConsistentConfig,
    choose_self_consistent_candidate,
    evaluate_self_consistent_placement,
    run_self_consistent_assignment,
    self_consistent_config_from_run,
)
from halbach.assembly.sensitivity import compute_sensitivity_table
from halbach.assembly.simulation import run_simulation_trial, summarize_comparison_results
from halbach.assembly.slots import build_assembly_slots
from halbach.assembly.types import (
    AssemblySlot,
    MagnetError,
    Placement,
    SensitivityTable,
    VirtualMagnet,
)
from halbach.assembly.variation import generate_virtual_magnets
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _error(eps: float = 0.0, d1: float = 0.0, d2: float = 0.0) -> MagnetError:
    return MagnetError(
        epsilon_parallel=eps,
        delta_perp_1=d1,
        delta_perp_2=d2,
    )


def _manual_slots(count: int) -> list[AssemblySlot]:
    return [
        AssemblySlot(
            slot_flat_id=idx,
            ring_id=0,
            layer_id=idx,
            theta_id=0,
            center_m=np.array([0.045 + 0.01 * idx, 0.006 * (idx % 2), 0.002], dtype=np.float64),
            nominal_u=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            nominal_phi_rad=np.pi / 2.0,
            physical_slot_number=1,
        )
        for idx in range(count)
    ]


def _manual_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.008, 0.0],
            [0.0, 0.0, 0.008],
        ],
        dtype=np.float64,
    )


def _manual_magnets(count: int) -> list[VirtualMagnet]:
    return [
        VirtualMagnet(
            magnet_id=idx,
            true_error=_error(
                eps=0.0004 * (idx - count / 2.0),
                d1=0.0001 * ((idx % 3) - 1),
                d2=-0.00008 * ((idx % 2) * 2 - 1),
            ),
            measured_error=_error(
                eps=0.0004 * (idx - count / 2.0),
                d1=0.0001 * ((idx % 3) - 1),
                d2=-0.00008 * ((idx % 2) * 2 - 1),
            ),
            quality=1.0,
        )
        for idx in range(count)
    ]


def _o0_placements(slots: list[AssemblySlot], magnets: list[VirtualMagnet]) -> list[Placement]:
    return [
        Placement(
            slot_flat_id=slot.slot_flat_id,
            magnet_id=magnet.magnet_id,
            orientation_id="O0",
            insert_order=idx,
            decision_engine="test",
        )
        for idx, (slot, magnet) in enumerate(zip(slots, magnets, strict=True))
    ]


def _manual_table(slots: list[AssemblySlot], residual_dim: int = 3) -> SensitivityTable:
    C = np.zeros((len(slots), 4, residual_dim, 3), dtype=np.float64)
    for slot_idx in range(len(slots)):
        for orientation_idx in range(4):
            C[slot_idx, orientation_idx, 0, 0] = float(slot_idx + 1)
            if residual_dim > 1:
                C[slot_idx, orientation_idx, 1, 1] = float(orientation_idx + 1)
            if residual_dim > 2:
                C[slot_idx, orientation_idx, 2, 2] = float(slot_idx - orientation_idx)
    return SensitivityTable(
        slot_flat_id=np.array([slot.slot_flat_id for slot in slots], dtype=np.int_),
        ring_id=np.array([slot.ring_id for slot in slots], dtype=np.int_),
        layer_id=np.array([slot.layer_id for slot in slots], dtype=np.int_),
        theta_id=np.array([slot.theta_id for slot in slots], dtype=np.int_),
        centers_m=np.array([slot.center_m for slot in slots], dtype=np.float64),
        nominal_u=np.array([slot.nominal_u for slot in slots], dtype=np.float64),
        orientation_id=("O0", "O90", "O180", "O270"),
        C=C,
        roi_points=_manual_points(),
        normalization_b0=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata={"case": "self-consistent-assignment-test"},
    )


def _write_generated_run(tmp_path: Path) -> RunBundle:
    run_dir = tmp_path / "self_consistent_assignment_run"
    results = generate_halbach_initial(
        N=4,
        R=1,
        end_R=None,
        end_layers_per_side=0,
        K=2,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.01,
    )
    write_run(
        run_dir,
        results,
        name="plan-c-self-consistent-test",
        schema_version=1,
        generator_params={},
        description="plan c self-consistent test run",
    )
    return load_run(run_dir)


def test_chi_zero_matches_fixed_placement() -> None:
    slots = _manual_slots(5)
    magnets = _manual_magnets(len(slots))
    placements = _o0_placements(slots, magnets)
    pts = _manual_points()

    fixed = evaluate_fixed_placement(slots, magnets, placements, pts)
    self_consistent = evaluate_self_consistent_placement(
        slots,
        magnets,
        placements,
        pts,
        SelfConsistentConfig(chi=0.0),
    )

    np.testing.assert_allclose(self_consistent.B, fixed.B)
    np.testing.assert_allclose(self_consistent.B0, fixed.B0)
    assert self_consistent.metrics.J_vector == pytest.approx(fixed.metrics.J_vector)


def test_self_consistent_config_from_run_uses_saved_optimization_params(
    tmp_path: Path,
) -> None:
    run = _write_generated_run(tmp_path)
    run.meta["magnetization"] = {
        "model_effective": "self-consistent-easy-axis",
        "self_consistent": {
            "chi": 0.025,
            "Nd": 0.21,
            "p0": 1.4,
            "volume_mm3": 250.0,
            "iters": 7,
            "omega": 0.45,
        },
    }

    config = self_consistent_config_from_run(
        run,
        factor=2.0,
        max_linear_candidates=3,
        require=True,
    )

    assert config.chi == pytest.approx(0.025)
    assert config.Nd == pytest.approx(0.21)
    assert config.p0 == pytest.approx(1.4)
    assert config.volume_m3 == pytest.approx(250.0e-9)
    assert config.iters == 7
    assert config.omega == pytest.approx(0.45)
    assert config.factor == pytest.approx(2.0)
    assert config.max_linear_candidates == 3


def test_self_consistent_config_from_run_requires_metadata(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)

    with pytest.raises(ValueError, match="magnetization.self_consistent"):
        self_consistent_config_from_run(run, require=True)


def test_top_k_limit_caps_self_consistent_evaluations() -> None:
    slots = _manual_slots(6)
    table = _manual_table(slots)
    magnet = VirtualMagnet(
        magnet_id=99,
        true_error=_error(eps=0.002, d1=0.001, d2=-0.0005),
        measured_error=_error(eps=0.002, d1=0.001, d2=-0.0005),
        quality=1.0,
    )

    decision = choose_self_consistent_candidate(
        table,
        slots,
        (),
        {},
        magnet,
        _manual_points(),
        [slot.slot_flat_id for slot in slots],
        SelfConsistentConfig(chi=0.0, max_linear_candidates=3),
    )

    assert decision.evaluated_count == 3
    assert len(decision.evaluations) == 3
    assert np.isfinite(decision.selected.score)


def test_linear_residual_is_used_for_top_k_pruning() -> None:
    slots = _manual_slots(2)
    table = _manual_table(slots, residual_dim=2)
    C = np.zeros_like(table.C)
    C[0, :, 0, 0] = 1.0
    C[1, :, 0, 0] = -1.0
    table = SensitivityTable(
        slot_flat_id=table.slot_flat_id,
        ring_id=table.ring_id,
        layer_id=table.layer_id,
        theta_id=table.theta_id,
        centers_m=table.centers_m,
        nominal_u=table.nominal_u,
        orientation_id=table.orientation_id,
        C=C,
        roi_points=table.roi_points,
        normalization_b0=table.normalization_b0,
        metadata=table.metadata,
    )
    magnet = VirtualMagnet(
        magnet_id=1,
        true_error=_error(eps=1.0),
        measured_error=_error(eps=1.0),
        quality=1.0,
    )

    decision = choose_self_consistent_candidate(
        table,
        slots,
        (),
        {},
        magnet,
        _manual_points(),
        [slot.slot_flat_id for slot in slots],
        SelfConsistentConfig(chi=0.0, max_linear_candidates=1),
        residual=np.array([1.0, 0.0], dtype=np.float64),
    )

    assert decision.linear_best_slot_flat_id == slots[1].slot_flat_id
    assert decision.selected.candidate.slot_flat_id == slots[1].slot_flat_id


def test_small_generated_run_self_consistent_assignment_completes(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(4, 0.02, seed=0)
    table = compute_sensitivity_table(slots, pts, finite_difference_step=1e-6)
    magnets = generate_virtual_magnets(
        count=len(slots),
        seed=12,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0005,
        direction_sigma_2=0.0005,
        measurement_noise=None,
    )

    result = run_self_consistent_assignment(
        table,
        slots,
        magnets,
        pts,
        SelfConsistentConfig(chi=0.01, iters=2, max_linear_candidates=2),
    )

    assert len(result.placements) == len(slots)
    assert len({placement.slot_flat_id for placement in result.placements}) == len(slots)
    assert result.evaluated_count <= len(slots) * 2
    assert all(decision.evaluated_count <= 2 for decision in result.decisions)
    assert {placement.decision_engine for placement in result.placements} == {
        "sequential_self_consistent"
    }
    assert result.final_evaluation.B.shape == (4, 3)
    assert np.isfinite(result.final_evaluation.metrics.J_vector)


def test_seed_fixed_magnets_make_assignment_reproducible() -> None:
    slots = _manual_slots(5)
    table = _manual_table(slots)
    magnets_a = generate_virtual_magnets(
        count=len(slots),
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0004,
        direction_sigma_2=0.0004,
        measurement_noise=None,
    )
    magnets_b = generate_virtual_magnets(
        count=len(slots),
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.001},
        direction_sigma_1=0.0004,
        direction_sigma_2=0.0004,
        measurement_noise=None,
    )

    result_a = run_self_consistent_assignment(
        table,
        slots,
        magnets_a,
        _manual_points(),
        SelfConsistentConfig(chi=0.0, max_linear_candidates=2),
    )
    result_b = run_self_consistent_assignment(
        table,
        slots,
        magnets_b,
        _manual_points(),
        SelfConsistentConfig(chi=0.0, max_linear_candidates=2),
    )

    assert result_a.placements == result_b.placements
    assert result_a.evaluated_count == result_b.evaluated_count


def test_simulation_summary_can_include_self_consistent_comparison() -> None:
    slots = _manual_slots(5)
    table = _manual_table(slots)
    magnets = _manual_magnets(len(slots))

    result = run_simulation_trial(
        slots,
        magnets,
        table,
        _manual_points(),
        trial_id=0,
        seed=8,
        include_self_consistent=True,
        self_consistent_config=SelfConsistentConfig(chi=0.0, max_linear_candidates=2),
    )
    summary = summarize_comparison_results([result])

    assert result.self_consistent is not None
    assert result.rms_ratio_self_consistent_over_linear is not None
    assert summary["self_consistent_trials"] == 1
    assert "self_consistent_rms_ppm_mean" in summary
    assert "rms_ratio_self_consistent_over_linear_mean" in summary


def test_p0_flat_shape_mismatch_is_rejected() -> None:
    slots = _manual_slots(4)
    magnets = _manual_magnets(len(slots))
    placements = _o0_placements(slots, magnets)

    with pytest.raises(ValueError, match="p0_flat_override"):
        evaluate_self_consistent_placement(
            slots,
            magnets,
            placements,
            _manual_points(),
            SelfConsistentConfig(chi=0.0),
            p0_flat_override=np.ones(len(slots) - 1, dtype=np.float64),
        )
