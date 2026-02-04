import numpy as np
import pytest

from halbach.dc.ccp_sc_linearized import run_ccp_sc
from halbach.dc.ccp_unitcircle import make_phi_nominal_halbach, phi_to_zvec
from halbach.dc.dipole_linmap import build_B_mxy_matrices
from halbach.dc.geom_build import build_ring_stack_positions, make_z_positions_uniform
from halbach.dc.roi_points import find_center_index, make_roi_points_xy_plane
from halbach.near import NearWindow, build_near_graph
from halbach.sc_linear_system import (
    build_T2_edges,
    edges_from_near,
    solve_p_easy_axis_linear_system,
)


def _setup_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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

    phi_nom = make_phi_nominal_halbach(N=N, K=K, R=R, phi0=0.0)
    phi_nom_flat = phi_nom.reshape(-1)
    z_nom = phi_to_zvec(phi_nom_flat)

    near = build_near_graph(R, K, N, NearWindow(wr=0, wz=1, wphi=1))
    i_edge, j_edge = edges_from_near(near.nbr_idx, near.nbr_mask)
    return (
        Ax,
        Ay,
        Az,
        center_idx,
        phi_nom_flat,
        z_nom,
        r0_flat,
        i_edge,
        j_edge,
    )


def test_dc_ccp_sc_linearized_smoke() -> None:
    cp = pytest.importorskip("cvxpy")
    installed = cp.installed_solvers()
    if not installed:
        pytest.skip("No cvxpy solvers available")
    solver = "ECOS" if "ECOS" in installed else "SCS"

    (
        Ax,
        Ay,
        Az,
        center_idx,
        phi_nom_flat,
        z_nom,
        r0_flat,
        i_edge,
        j_edge,
    ) = _setup_problem()

    M = int(phi_nom_flat.shape[0])
    near = build_near_graph(1, 2, 6, NearWindow(wr=0, wz=1, wphi=1))
    volume_m3 = 1000.0 * 1e-9

    p0 = 1.0
    chi = 0.05
    Nd = 1.0 / 3.0
    sc_params = {"chi": chi, "Nd": Nd, "p0": p0}

    T2_edge = build_T2_edges(
        r0_flat,
        i_edge,
        j_edge,
        near_kernel="dipole",
        volume_m3=volume_m3,
        subdip_n=2,
    )

    p_init_on, _stats = solve_p_easy_axis_linear_system(
        phi_nom_flat,
        r0_flat,
        near.nbr_idx,
        near.nbr_mask,
        near_kernel="dipole",
        volume_m3=volume_m3,
        p0=p0,
        chi=chi,
        Nd=Nd,
        subdip_n=2,
    )

    z_opt_on, p_opt_on, x_opt_on, trace_on = run_ccp_sc(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        center_idx=center_idx,
        wx=0.0,
        wy=1.0,
        wz=0.0,
        reg_x=0.0,
        reg_p=0.0,
        reg_z=0.0,
        tau0=1e-4,
        tau_mult=1.0,
        tau_max=1e-4,
        iters=2,
        tol=1e-6,
        tol_f=1e-9,
        z_nom=z_nom,
        delta_nom_deg=20.0,
        delta_step_deg=10.0,
        p_init=p_init_on,
        sc_eq_enabled=True,
        sc_params=sc_params,
        p_bounds=(0.0, 2.0),
        p_fix_value=None,
        i_edge=i_edge,
        j_edge=j_edge,
        T2_edge=T2_edge,
        solver=solver,
        verbose=False,
    )

    assert np.isfinite(z_opt_on).all()
    assert np.isfinite(p_opt_on).all()
    assert np.isfinite(x_opt_on).all()

    norms_on = np.sqrt(z_opt_on[0::2] ** 2 + z_opt_on[1::2] ** 2)
    assert float(np.max(norms_on)) <= 1.0 + 1e-6

    z_nom_blocks = z_nom.reshape(-1, 2)
    z_on_blocks = z_opt_on.reshape(-1, 2)
    dots_on = np.sum(z_nom_blocks * z_on_blocks, axis=1)
    cos_delta_nom = np.cos(np.deg2rad(20.0))
    assert np.all(dots_on >= cos_delta_nom - 1e-6)

    assert "iters" in trace_on
    assert len(trace_on["iters"]) > 0
    last_on = trace_on["iters"][-1]
    assert "r_prod" in last_on
    assert "r_sc" in last_on

    p_fix = 1.0
    p_init_off = np.full(M, p_fix, dtype=np.float64)
    z_opt_off, p_opt_off, x_opt_off, trace_off = run_ccp_sc(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        center_idx=center_idx,
        wx=0.0,
        wy=1.0,
        wz=0.0,
        reg_x=0.0,
        reg_p=0.0,
        reg_z=0.0,
        tau0=1e-4,
        tau_mult=1.0,
        tau_max=1e-4,
        iters=2,
        tol=1e-6,
        tol_f=1e-9,
        z_nom=z_nom,
        delta_nom_deg=20.0,
        delta_step_deg=10.0,
        p_init=p_init_off,
        sc_eq_enabled=False,
        sc_params=sc_params,
        p_bounds=(0.0, 2.0),
        p_fix_value=p_fix,
        i_edge=i_edge,
        j_edge=j_edge,
        T2_edge=T2_edge,
        solver=solver,
        verbose=False,
    )

    assert np.isfinite(z_opt_off).all()
    assert np.isfinite(p_opt_off).all()
    assert np.isfinite(x_opt_off).all()
    np.testing.assert_allclose(p_opt_off, p_fix, rtol=0.0, atol=1e-6)

    norms_off = np.sqrt(z_opt_off[0::2] ** 2 + z_opt_off[1::2] ** 2)
    assert float(np.max(norms_off)) <= 1.0 + 1e-6

    z_off_blocks = z_opt_off.reshape(-1, 2)
    dots_off = np.sum(z_nom_blocks * z_off_blocks, axis=1)
    assert np.all(dots_off >= cos_delta_nom - 1e-6)

    assert "iters" in trace_off
    assert len(trace_off["iters"]) > 0
    last_off = trace_off["iters"][-1]
    assert "r_prod" in last_off
    assert "r_sc" in last_off
