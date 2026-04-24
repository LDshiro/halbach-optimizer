import numpy as np

from halbach.cli.optimize_run import (
    _apply_beta_vars,
    _build_beta_active_mask,
    _build_beta_symmetry_indices,
    _pack_beta_grad,
    _pack_beta_vars,
)


def test_beta_symmetry_pack_unpack_and_grad() -> None:
    R = 2
    K = 6
    reps, mates = _build_beta_symmetry_indices(K)

    assert reps.tolist() == [0, 1, 2]
    assert mates.tolist() == [5, 4, 3]

    beta0 = np.zeros((R, K), dtype=np.float64)
    ring_active_mask = np.ones((R, K), dtype=bool)
    beta_active_mask = _build_beta_active_mask(ring_active_mask, reps, mates)
    beta_vars = np.arange(R * reps.size, dtype=np.float64) * 0.01
    beta = _apply_beta_vars(beta_vars, beta0, reps, mates, beta_active_mask)

    np.testing.assert_allclose(beta[:, 5], -beta[:, 0])
    np.testing.assert_allclose(beta[:, 4], -beta[:, 1])
    np.testing.assert_allclose(beta[:, 3], -beta[:, 2])

    packed = _pack_beta_vars(beta, reps, beta_active_mask)
    np.testing.assert_allclose(packed, beta_vars)

    grad_beta = np.arange(R * K, dtype=np.float64).reshape(R, K)
    grad_packed = _pack_beta_grad(grad_beta, reps, mates, beta_active_mask).reshape(R, reps.size)

    np.testing.assert_allclose(grad_packed[:, 0], grad_beta[:, 0] - grad_beta[:, 5])
    np.testing.assert_allclose(grad_packed[:, 1], grad_beta[:, 1] - grad_beta[:, 4])
    np.testing.assert_allclose(grad_packed[:, 2], grad_beta[:, 2] - grad_beta[:, 3])


def test_beta_symmetry_odd_k_center_fixed_zero() -> None:
    R = 1
    K = 5
    reps, mates = _build_beta_symmetry_indices(K)

    assert reps.tolist() == [0, 1]
    assert mates.tolist() == [4, 3]

    beta0 = np.full((R, K), 0.25, dtype=np.float64)
    ring_active_mask = np.ones((R, K), dtype=bool)
    beta_active_mask = _build_beta_active_mask(ring_active_mask, reps, mates)
    beta_vars = np.array([0.1, -0.2], dtype=np.float64)
    beta = _apply_beta_vars(beta_vars, beta0, reps, mates, beta_active_mask)

    np.testing.assert_allclose(beta[:, 0], 0.1)
    np.testing.assert_allclose(beta[:, 4], -0.1)
    np.testing.assert_allclose(beta[:, 1], -0.2)
    np.testing.assert_allclose(beta[:, 3], 0.2)
    np.testing.assert_allclose(beta[:, 2], 0.0)


def test_beta_symmetry_respects_active_mask() -> None:
    R = 2
    K = 6
    reps, mates = _build_beta_symmetry_indices(K)
    ring_active_mask = np.ones((R, K), dtype=bool)
    ring_active_mask[1, 2] = False
    ring_active_mask[1, 3] = False
    beta_active_mask = _build_beta_active_mask(ring_active_mask, reps, mates)

    beta0 = np.full((R, K), 7.0, dtype=np.float64)
    beta_vars = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    beta = _apply_beta_vars(beta_vars, beta0, reps, mates, beta_active_mask)

    assert beta_active_mask.shape == (R, reps.size)
    assert beta_active_mask[1, 2] == 0
    np.testing.assert_allclose(beta[1, 2], 0.0)
    np.testing.assert_allclose(beta[1, 3], 0.0)
    packed = _pack_beta_vars(beta, reps, beta_active_mask)
    np.testing.assert_allclose(packed, beta_vars)
