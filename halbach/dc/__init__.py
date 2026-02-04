"""DC/CCP baseline tools (separate optimization framework)."""

from .ccp_sc_linearized import eval_g_true, linearize_sc_terms, run_ccp_sc
from .ccp_unitcircle import (
    make_phi_nominal_halbach,
    phi_to_zvec,
    run_ccp_unitcircle,
    solve_ccp_subproblem,
    zvec_to_phi_pnorm,
)
from .dc_io import write_dc_run
from .dipole_linmap import build_Ay_diff, build_B_mxy_matrices
from .geom_build import build_ring_stack_positions, make_z_positions_uniform
from .qp_relaxed_mxy import derive_phi_p_from_mxy, solve_qp_relaxed_mxy
from .roi_points import find_center_index, make_roi_points_xy_plane

__all__ = [
    "build_Ay_diff",
    "build_B_mxy_matrices",
    "build_ring_stack_positions",
    "make_z_positions_uniform",
    "derive_phi_p_from_mxy",
    "solve_qp_relaxed_mxy",
    "find_center_index",
    "make_roi_points_xy_plane",
    "make_phi_nominal_halbach",
    "phi_to_zvec",
    "run_ccp_unitcircle",
    "solve_ccp_subproblem",
    "zvec_to_phi_pnorm",
    "eval_g_true",
    "linearize_sc_terms",
    "run_ccp_sc",
    "write_dc_run",
]
