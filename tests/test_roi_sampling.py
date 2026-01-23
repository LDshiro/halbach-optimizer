import numpy as np

from halbach.geom import (
    build_roi_points,
    sample_sphere_surface_fibonacci,
    sample_sphere_surface_random,
)


def test_surface_fibonacci_shape_dtype_and_radius() -> None:
    roi_r = 0.14
    pts = sample_sphere_surface_fibonacci(300, roi_r, seed=1)
    assert pts.shape == (300, 3)
    assert pts.dtype == np.float64
    assert pts.flags["C_CONTIGUOUS"]
    norms = np.linalg.norm(pts, axis=1)
    assert np.max(np.abs(norms - roi_r)) < 1e-10


def test_surface_fibonacci_half_x() -> None:
    pts = sample_sphere_surface_fibonacci(100, 0.1, seed=2, half_x=True)
    assert np.all(pts[:, 0] >= 0.0)


def test_surface_random_half_x() -> None:
    pts = sample_sphere_surface_random(50, 0.2, seed=3, half_x=True)
    assert np.all(pts[:, 0] >= 0.0)


def test_volume_subsample_returns_at_most_n() -> None:
    pts = build_roi_points(0.05, 0.01, mode="volume-subsample", n_samples=20, seed=0)
    assert pts.shape[0] <= 20
