import numpy as np

from halbach.geom import build_param_map, fixed_radius_layer_indices, pack_grad, pack_x, unpack_x


def test_pack_unpack_keeps_fixed_layers() -> None:
    R = 2
    K = 8
    param_map = build_param_map(R, K, n_fix_radius=2)
    alphas0 = np.arange(R * K, dtype=float).reshape(R, K)
    r_bases0 = np.linspace(0.1, 0.2, K)

    x0 = pack_x(alphas0, r_bases0, param_map)
    alphas1, r_bases1 = unpack_x(x0 + 1.0, alphas0, r_bases0, param_map)

    fixed_k = param_map.fixed_k_radius
    assert np.all(alphas1[:, fixed_k] == alphas0[:, fixed_k] + 1.0)
    assert np.all(r_bases1[fixed_k] == r_bases0[fixed_k])
    assert not np.allclose(alphas1[:, 0], alphas0[:, 0])


def test_pack_x_dimension_reduction() -> None:
    R = 3
    K = 10
    n_fix = 4
    param_map = build_param_map(R, K, n_fix_radius=n_fix)
    alphas0 = np.zeros((R, K), dtype=float)
    r_bases0 = np.zeros(K, dtype=float)
    x = pack_x(alphas0, r_bases0, param_map)
    expected = R * K + (K // 2 - n_fix // 2)
    assert x.size == expected


def test_param_map_angle_dim_unchanged_radius_dim_changes() -> None:
    R = 3
    K = 10
    map0 = build_param_map(R, K, n_fix_radius=0)
    map4 = build_param_map(R, K, n_fix_radius=4)
    assert map0.free_alpha_idx.size == map4.free_alpha_idx.size == R * K
    assert map4.free_r_idx.size == map0.free_r_idx.size - 2


def test_pack_grad_sums_symmetric_pairs() -> None:
    R = 1
    K = 6
    param_map = build_param_map(R, K, n_fix_radius=2)
    grad_alphas = np.zeros((R, K), dtype=float)
    grad_r = np.zeros(K, dtype=float)

    k_low = int(param_map.free_r_idx[0])
    k_up = int(param_map.lower_to_upper[k_low])
    grad_r[k_low] = 1.0
    grad_r[k_up] = 2.0

    grad_x = pack_grad(grad_alphas, grad_r, param_map)
    n_alpha = int(param_map.free_alpha_idx.size)
    assert grad_x[n_alpha] == 3.0


def test_fixed_radius_layer_indices_for_center_and_ends() -> None:
    np.testing.assert_array_equal(fixed_radius_layer_indices(10, 2, "center"), np.array([4, 5]))
    np.testing.assert_array_equal(
        fixed_radius_layer_indices(10, 4, "center"), np.array([3, 4, 5, 6])
    )
    np.testing.assert_array_equal(fixed_radius_layer_indices(10, 2, "ends"), np.array([0, 9]))
    np.testing.assert_array_equal(fixed_radius_layer_indices(10, 4, "ends"), np.array([0, 1, 8, 9]))


def test_build_param_map_supports_end_fixed_layers() -> None:
    param_map = build_param_map(R=2, K=10, n_fix_radius=4, fix_radius_mode="ends")
    np.testing.assert_array_equal(param_map.fixed_k_radius, np.array([0, 1, 8, 9]))
    assert param_map.free_r_idx.size == 3


def test_unpack_x_keeps_end_fixed_layers_at_initial_values() -> None:
    param_map = build_param_map(R=1, K=10, n_fix_radius=4, fix_radius_mode="ends")
    alphas0 = np.zeros((1, 10), dtype=float)
    r_bases0 = np.linspace(0.1, 0.19, 10)
    x0 = pack_x(alphas0, r_bases0, param_map)
    _, r_bases1 = unpack_x(x0 + 1.0, alphas0, r_bases0, param_map)
    np.testing.assert_allclose(
        r_bases1[param_map.fixed_k_radius], r_bases0[param_map.fixed_k_radius]
    )
