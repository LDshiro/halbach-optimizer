from pathlib import Path

import numpy as np

from halbach.assembly.sensitivity import (
    compute_sensitivity_table,
    load_sensitivity_table,
    save_sensitivity_table,
)
from halbach.assembly.slots import build_assembly_slots
from halbach.cli.plan_c_compute_sensitivity import main as compute_sensitivity_main
from halbach.generate import generate_halbach_initial, write_run
from halbach.geom import sample_sphere_surface_fibonacci
from halbach.run_io import load_run
from halbach.run_types import RunBundle


def _write_generated_run(tmp_path: Path, *, N: int = 8, R: int = 1, K: int = 4) -> RunBundle:
    run_dir = tmp_path / f"sensitivity_io_run_N{N}_R{R}_K{K}"
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
        name="plan-c-sensitivity-io-test",
        schema_version=1,
        generator_params={},
        description="plan c sensitivity io test run",
    )
    return load_run(run_dir)


def test_sensitivity_npz_roundtrip_preserves_values(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    slots = build_assembly_slots(run)
    pts = sample_sphere_surface_fibonacci(5, 0.02, seed=0)
    table = compute_sensitivity_table(
        slots,
        pts,
        finite_difference_step=1e-6,
        metadata={"case": "roundtrip"},
    )
    path = tmp_path / "plan_c_sensitivity.npz"

    save_sensitivity_table(path, table)
    loaded = load_sensitivity_table(path)

    np.testing.assert_array_equal(loaded.slot_flat_id, table.slot_flat_id)
    np.testing.assert_array_equal(loaded.ring_id, table.ring_id)
    np.testing.assert_array_equal(loaded.layer_id, table.layer_id)
    np.testing.assert_array_equal(loaded.theta_id, table.theta_id)
    np.testing.assert_allclose(loaded.centers_m, table.centers_m)
    np.testing.assert_allclose(loaded.nominal_u, table.nominal_u)
    np.testing.assert_allclose(loaded.C, table.C)
    np.testing.assert_allclose(loaded.roi_points, table.roi_points)
    np.testing.assert_allclose(loaded.normalization_b0, table.normalization_b0)
    assert loaded.orientation_id == table.orientation_id
    assert loaded.metadata["case"] == "roundtrip"
    assert loaded.projection_basis is None

    with np.load(path, allow_pickle=False) as data:
        assert "metadata_json" in data.files
        assert "projection_basis" in data.files


def test_plan_c_compute_sensitivity_cli_writes_loadable_npz(tmp_path: Path) -> None:
    run = _write_generated_run(tmp_path)
    out_path = tmp_path / "cli_sensitivity.npz"

    compute_sensitivity_main(
        [
            "--run",
            str(run.run_dir),
            "--out",
            str(out_path),
            "--roi-r",
            "0.02",
            "--roi-mode",
            "surface-fibonacci",
            "--roi-samples",
            "5",
            "--finite-difference-step",
            "1e-6",
        ]
    )

    loaded = load_sensitivity_table(out_path)
    assert loaded.C.shape[0] == len(build_assembly_slots(run))
    assert loaded.roi_points.shape == (5, 3)
    assert loaded.metadata["roi_mode"] == "surface-fibonacci"
