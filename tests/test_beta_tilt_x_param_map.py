import numpy as np

from halbach.cli.optimize_run import (
    _apply_beta_vars,
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
    beta_vars = np.arange(R * reps.size, dtype=np.float64) * 0.01
    beta = _apply_beta_vars(beta_vars, beta0, reps, mates)

    np.testing.assert_allclose(beta[:, 5], -beta[:, 0])
    np.testing.assert_allclose(beta[:, 4], -beta[:, 1])
    np.testing.assert_allclose(beta[:, 3], -beta[:, 2])

    packed = _pack_beta_vars(beta, reps)
    np.testing.assert_allclose(packed, beta_vars)

    grad_beta = np.arange(R * K, dtype=np.float64).reshape(R, K)
    grad_packed = _pack_beta_grad(grad_beta, reps, mates).reshape(R, reps.size)

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
    beta_vars = np.array([0.1, -0.2], dtype=np.float64)
    beta = _apply_beta_vars(beta_vars, beta0, reps, mates)

    np.testing.assert_allclose(beta[:, 0], 0.1)
    np.testing.assert_allclose(beta[:, 4], -0.1)
    np.testing.assert_allclose(beta[:, 1], -0.2)
    np.testing.assert_allclose(beta[:, 3], 0.2)
    np.testing.assert_allclose(beta[:, 2], 0.0)
