from pathlib import Path

import numpy as np

from halbach.generate import generate_halbach_initial, write_run
from halbach.run_io import load_run


def test_generate_run_loadable(tmp_path: Path) -> None:
    N = 12
    R = 3
    K = 4
    Lz = 0.2
    diameter_m = 0.4
    ring_step_m = 0.012

    results = generate_halbach_initial(
        N=N,
        R=R,
        K=K,
        Lz=Lz,
        diameter_m=diameter_m,
        ring_offset_step_m=ring_step_m,
        end_R=None,
        end_layers_per_side=0,
    )
    out_dir = tmp_path / "run"
    write_run(
        out_dir,
        results,
        name="test",
        schema_version=1,
        generator_params=dict(
            N=N,
            R=R,
            K=K,
            Lz=Lz,
            diameter_mm=diameter_m * 1e3,
            ring_offset_step_mm=ring_step_m * 1e3,
        ),
        description="generated initial pure Halbach",
    )

    run = load_run(out_dir)
    assert run.geometry.N == N
    assert run.geometry.R == R
    assert run.geometry.K == K
    assert run.results.alphas.shape == (R, K)
    assert run.results.r_bases.shape == (K,)

    if K > 1:
        np.testing.assert_allclose(run.results.z_layers[0], -Lz / 2.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(run.results.z_layers[-1], Lz / 2.0, rtol=0.0, atol=1e-12)

    expected_offsets = np.arange(R, dtype=np.float64) * ring_step_m
    np.testing.assert_allclose(run.results.ring_offsets, expected_offsets, rtol=0.0, atol=1e-12)


def test_generate_run_with_end_layers_stores_profile_extras(tmp_path: Path) -> None:
    results = generate_halbach_initial(
        N=12,
        R=2,
        K=24,
        Lz=0.2,
        diameter_m=0.4,
        ring_offset_step_m=0.012,
        end_R=3,
        end_layers_per_side=3,
    )
    out_dir = tmp_path / "run_profile"
    write_run(
        out_dir,
        results,
        name="test-profile",
        schema_version=1,
        generator_params=dict(N=12, R=2, K=24, end_R=3, end_layers_per_side=3),
        description="generated initial pure Halbach",
    )

    run = load_run(out_dir)
    assert run.results.alphas.shape == (3, 24)
    assert run.results.ring_offsets.shape == (3,)
    assert "radial_count_per_layer" in run.results.extras
    assert "ring_active_mask" in run.results.extras
    np.testing.assert_array_equal(
        np.asarray(run.results.extras["radial_count_per_layer"]),
        np.array([3, 3, 3] + [2] * 18 + [3, 3, 3]),
    )
    ring_active_mask = np.asarray(run.results.extras["ring_active_mask"], dtype=bool)
    assert ring_active_mask.shape == (3, 24)
    assert bool(np.all(ring_active_mask[:, 0]))
    np.testing.assert_array_equal(ring_active_mask[:, 10], np.array([True, True, False]))
