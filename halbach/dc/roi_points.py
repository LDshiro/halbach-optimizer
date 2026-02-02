from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_roi_points_xy_plane(
    *, radius_m: float, grid_n: int, include_center: bool = True
) -> NDArray[np.float64]:
    if grid_n < 1:
        raise ValueError("grid_n must be >= 1")
    radius = float(radius_m)
    xs = np.linspace(-radius, radius, grid_n, dtype=np.float64)
    ys = np.linspace(-radius, radius, grid_n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    mask = x_flat * x_flat + y_flat * y_flat <= radius * radius
    count = int(np.count_nonzero(mask))
    pts = np.stack([x_flat[mask], y_flat[mask], np.zeros(count, dtype=np.float64)], axis=1)

    if include_center:
        has_center = np.any(np.isclose(pts[:, 0], 0.0) & np.isclose(pts[:, 1], 0.0))
        if not has_center:
            center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
            pts = np.vstack([center, pts])

    return np.asarray(pts, dtype=np.float64)


def find_center_index(pts: NDArray[np.float64]) -> int:
    if pts.size == 0:
        raise ValueError("pts must be non-empty")
    norms = np.linalg.norm(pts, axis=1)
    return int(np.argmin(norms))


__all__ = ["make_roi_points_xy_plane", "find_center_index"]
