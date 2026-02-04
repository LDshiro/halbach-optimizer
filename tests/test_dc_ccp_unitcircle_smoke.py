import numpy as np
import pytest

from halbach.dc.ccp_unitcircle import make_phi_nominal_halbach, phi_to_zvec, run_ccp_unitcircle
from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane


def test_dc_ccp_unitcircle_smoke() -> None:
    cp = pytest.importorskip("cvxpy")
    installed = cp.installed_solvers()
    if not installed:
        pytest.skip("No cvxpy solvers available")
    solver = "ECOS" if "ECOS" in installed else "SCS"

    N, K, R = 6, 2, 1
    radius = 0.10
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
    p_mag = 1.0
    Axd = p_mag * Ax
    Ayd = p_mag * Ay
    Azd = p_mag * Az

    phi_nom = make_phi_nominal_halbach(N=N, K=K, R=R, phi0=0.0)
    phi_nom_flat = phi_nom.reshape(-1)

    z_opt, trace = run_ccp_unitcircle(
        Axd=Axd,
        Ayd=Ayd,
        Azd=Azd,
        center_idx=center_idx,
        wx=0.0,
        wy=1.0,
        wz=0.0,
        reg=0.0,
        tau0=1e-4,
        tau_mult=1.0,
        tau_max=1e-4,
        iters=3,
        tol=1e-6,
        tol_f=1e-9,
        phi_nom_flat=phi_nom_flat,
        delta_nom_deg=20.0,
        delta_step_deg=10.0,
        solver=solver,
        verbose=False,
    )

    assert np.isfinite(z_opt).all()
    norms = np.sqrt(z_opt[0::2] ** 2 + z_opt[1::2] ** 2)
    assert float(np.max(norms)) <= 1.0 + 1e-6

    z_nom = phi_to_zvec(phi_nom_flat)
    z_nom_blocks = z_nom.reshape(-1, 2)
    z_blocks = z_opt.reshape(-1, 2)
    dots = np.sum(z_nom_blocks * z_blocks, axis=1)
    cos_delta_nom = np.cos(np.deg2rad(20.0))
    assert np.all(dots >= cos_delta_nom - 1e-6)

    assert "iters" in trace
    assert len(trace["iters"]) > 0
    last = trace["iters"][-1]
    assert "norm_min" in last
    assert "norm_max" in last
