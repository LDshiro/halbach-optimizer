import numpy as np
import pytest

from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.qp_relaxed_mxy import solve_qp_relaxed_mxy
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane


def test_dc_qp_relaxed_mxy_smoke() -> None:
    cp = pytest.importorskip("cvxpy")
    if "OSQP" not in cp.installed_solvers():
        pytest.skip("OSQP not available")

    N, K, R = 6, 2, 1
    radius = 0.1
    length = 0.02
    z_positions = make_z_positions_uniform(K, length)
    r0_rkn, _info = build_ring_stack_positions(
        N=N,
        K=K,
        R=R,
        radii_m=radius,
        z_positions_m=z_positions,
    )
    r0_flat = r0_rkn.reshape(-1, 3)

    pts = make_roi_points_xy_plane(radius_m=0.05, grid_n=11, include_center=True)
    center_idx = find_center_index(pts)

    Ax, Ay, Az = build_B_mxy_matrices(pts, r0_flat, factor=1e-7)
    x_opt, info = solve_qp_relaxed_mxy(
        Ax,
        Ay,
        Az,
        center_idx=center_idx,
        wx=0.0,
        wy=1.0,
        wz=0.0,
        pmax=1.0,
        reg=0.0,
        verbose=False,
    )

    assert str(info.get("status")) in {"optimal", "optimal_inaccurate"}
    assert np.isfinite(x_opt).all()
    assert float(np.max(np.abs(x_opt))) <= 1.0 + 1e-6
