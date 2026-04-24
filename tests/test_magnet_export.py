from __future__ import annotations

import numpy as np

from halbach.magnet_export import (
    build_magnet_export_payload,
    equivalent_cube_dimensions_from_volume_mm3,
)


def test_equivalent_cube_dimensions_from_volume_mm3() -> None:
    dims_m, dims_mm = equivalent_cube_dimensions_from_volume_mm3(1000.0)

    np.testing.assert_allclose(dims_mm, np.array([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(dims_m, np.array([0.01, 0.01, 0.01]))


def test_build_magnet_export_payload_filters_inactive_magnets() -> None:
    phi_rkn = np.array(
        [
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]],
            [[2.0, 2.1, 2.2], [3.0, 3.1, 3.2]],
        ],
        dtype=np.float64,
    )
    r0_rkn = np.array(
        [
            [
                [[1.0, 0.0, -1.0], [1.0, 1.0, -1.0], [1.0, 2.0, -1.0]],
                [[9.0, 0.0, -1.0], [9.0, 1.0, -1.0], [9.0, 2.0, -1.0]],
            ],
            [
                [[8.0, 0.0, 1.0], [8.0, 1.0, 1.0], [8.0, 2.0, 1.0]],
                [[2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
            ],
        ],
        dtype=np.float64,
    )
    ring_active_mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    beta_rk = np.array([[0.0, 0.0], [0.0, 0.3]], dtype=np.float64)

    payload = build_magnet_export_payload(
        phi_rkn,
        r0_rkn,
        ring_active_mask=ring_active_mask,
        beta_rk=beta_rk,
        dimensions_m=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        dimensions_mm=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )

    np.testing.assert_allclose(
        payload["magnet_centers_m"],
        np.array(
            [
                [1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 2.0, -1.0],
                [2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        payload["magnet_phi_rad"],
        np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=np.float64),
    )
    np.testing.assert_allclose(
        payload["magnet_beta_rad"],
        np.array([0.0, 0.0, 0.0, 0.3, 0.3, 0.3], dtype=np.float64),
    )
    np.testing.assert_array_equal(payload["magnet_ring_id"], np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(payload["magnet_layer_id"], np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(payload["magnet_theta_id"], np.array([0, 1, 2, 0, 1, 2]))
    np.testing.assert_allclose(
        payload["magnet_dimensions_m"],
        np.array([0.01, 0.02, 0.03], dtype=np.float64),
    )
    np.testing.assert_allclose(
        payload["magnet_dimensions_mm"],
        np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
