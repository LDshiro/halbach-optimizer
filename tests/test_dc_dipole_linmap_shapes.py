import numpy as np

from halbach.dc.dipole_linmap import build_B_mxy_matrices


def test_dc_dipole_linmap_shapes() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=np.float64)
    r0_flat = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]], dtype=np.float64)

    Ax, Ay, Az = build_B_mxy_matrices(pts, r0_flat, factor=1e-7)

    assert Ax.shape == (3, 4)
    assert Ay.shape == (3, 4)
    assert Az.shape == (3, 4)
    assert np.isfinite(Ax).all()
    assert np.isfinite(Ay).all()
    assert np.isfinite(Az).all()
