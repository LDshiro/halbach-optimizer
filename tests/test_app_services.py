import json
from pathlib import Path

import numpy as np

from halbach.app_services import (
    Map2DRequest,
    Scene3DRequest,
    build_run_delta_summary,
    build_run_summary,
    list_run_candidates,
    load_map2d_payload,
    load_run_bundle,
    load_scene3d_payload,
    resolve_run_path,
    resolve_selected_path,
)
from halbach.viz2d import compute_error_map_ppm_plane
from halbach.viz3d import compute_scene_ranges, enumerate_magnets_with_directions


def _write_min_run(tmp_path: Path, name: str = "run") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()

    R = 1
    K = 3
    N = 8
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    sin2th = np.sin(2.0 * theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    z_layers = np.linspace(-0.02, 0.02, K)
    ring_offsets = np.array([0.0], dtype=float)
    alphas = 1e-3 * np.arange(R * K, dtype=float).reshape(R, K)
    r_bases = 0.2 + 1e-3 * np.arange(K, dtype=float)

    np.savez(
        run_dir / "results.npz",
        alphas_opt=alphas,
        r_bases_opt=r_bases,
        theta=theta,
        sin2th=sin2th,
        cth=cth,
        sth=sth,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
    )
    (run_dir / "meta.json").write_text(json.dumps({"label": name}), encoding="utf-8")
    return run_dir


def test_list_run_candidates_and_path_helpers(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path, name="candidate")

    candidates = list_run_candidates(tmp_path)

    assert [candidate.label for candidate in candidates] == [str(run_dir)]
    assert resolve_selected_path("runs/demo", "") == "runs/demo"
    assert resolve_selected_path("runs/demo", " custom/path ") == "custom/path"
    assert resolve_run_path(str(run_dir)) == run_dir


def test_build_run_summary_and_delta_summary(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path, name="summary")
    run = load_run_bundle(run_dir)

    summary = build_run_summary(run)

    assert summary.name == "summary"
    assert summary.framework == "legacy"
    assert summary.geometry_summary["N"] == 8
    assert float(summary.key_stats["B0_T"]) > 0.0

    optimized_dir = _write_min_run(tmp_path, name="optimized")
    optimized = load_run_bundle(optimized_dir)
    delta = build_run_delta_summary(run, optimized)
    assert delta["available"] is True


def test_map2d_payload_matches_existing_viz2d_result(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path, name="map")
    run = load_run_bundle(run_dir)

    payload = load_map2d_payload(
        Map2DRequest(run_path=str(run_dir), roi_r=0.05, step=0.05), run=run
    )
    reference = compute_error_map_ppm_plane(run, plane="xy", coord0=0.0, roi_r=0.05, step=0.05)

    assert np.allclose(payload.xs, reference.xs)
    assert np.allclose(payload.ys, reference.ys)
    assert np.allclose(payload.ppm, reference.ppm, equal_nan=True)
    assert np.array_equal(payload.mask, reference.mask)
    assert payload.summary_stats["ppm_max_abs"] >= 0.0


def test_scene3d_payload_matches_existing_viz3d_result(tmp_path: Path) -> None:
    run_a_dir = _write_min_run(tmp_path, name="scene_a")
    run_b_dir = _write_min_run(tmp_path, name="scene_b")
    run_a = load_run_bundle(run_a_dir)
    run_b = load_run_bundle(run_b_dir)

    payload = load_scene3d_payload(
        Scene3DRequest(
            primary_path=str(run_a_dir),
            secondary_path=str(run_b_dir),
            stride=2,
            hide_x_negative=False,
        ),
        primary_run=run_a,
        secondary_run=run_b,
    )
    centers_a, _phi_a, dir_a, ring_a, layer_a = enumerate_magnets_with_directions(
        run_a,
        stride=2,
        hide_x_negative=False,
    )
    centers_b, _phi_b, _dir_b, _ring_b, _layer_b = enumerate_magnets_with_directions(
        run_b,
        stride=2,
        hide_x_negative=False,
    )
    reference_ranges = compute_scene_ranges([centers_a, centers_b])

    assert np.allclose(payload.primary.centers, centers_a)
    assert np.allclose(payload.primary.direction_vectors, dir_a)
    assert np.array_equal(payload.primary.ring_ids, ring_a)
    assert np.array_equal(payload.primary.layer_ids, layer_a)
    assert payload.secondary is not None
    assert np.allclose(payload.secondary.centers, centers_b)
    assert payload.scene_ranges == reference_ranges
