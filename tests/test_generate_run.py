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
