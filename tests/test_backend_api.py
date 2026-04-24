import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app import create_app


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


def test_backend_runs_and_readonly_endpoints(tmp_path: Path) -> None:
    run_dir = _write_min_run(tmp_path, name="api_run")
    client = TestClient(create_app(runs_dir=tmp_path))

    runs_response = client.get("/api/runs")
    assert runs_response.status_code == 200
    runs_payload = runs_response.json()
    assert runs_payload["runs"][0]["label"] == str(run_dir)

    overview_response = client.post("/api/run/overview", json={"run_path": str(run_dir)})
    assert overview_response.status_code == 200
    overview_payload = overview_response.json()
    assert overview_payload["name"] == "api_run"
    assert overview_payload["framework"] == "legacy"

    map_response = client.post(
        "/api/run/map2d",
        json={"run_path": str(run_dir), "roi_r": 0.05, "step": 0.05},
    )
    assert map_response.status_code == 200
    map_payload = map_response.json()
    assert map_payload["plane"] == "xy"
    assert len(map_payload["xs"]) == len(map_payload["ys"])
    assert map_payload["B0_T"] > 0.0

    scene_response = client.post(
        "/api/run/scene3d",
        json={"primary_path": str(run_dir), "stride": 2, "hide_x_negative": False},
    )
    assert scene_response.status_code == 200
    scene_payload = scene_response.json()
    assert scene_payload["primary"]["run_name"] == "api_run"
    assert scene_payload["secondary"] is None
    assert len(scene_payload["scene_ranges"]) == 3
