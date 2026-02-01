import numpy as np
import pytest

from halbach.symmetry import build_mirror_x0


@pytest.mark.parametrize("N", [6, 8, 12])
def test_build_mirror_x0_properties(N: int) -> None:
    mirror = build_mirror_x0(N)
    idx = np.arange(N, dtype=int)

    assert np.array_equal(mirror.mirror_idx[mirror.mirror_idx], idx)
    assert np.intersect1d(mirror.rep_idx, mirror.fixed_idx).size == 0
    assert mirror.rep_idx.size == (N - mirror.fixed_idx.size) // 2

    covered = np.concatenate([mirror.rep_idx, mirror.mirror_idx[mirror.rep_idx], mirror.fixed_idx])
    assert np.array_equal(np.sort(covered), idx)

    if N % 4 == 0:
        assert mirror.fixed_idx.size == 2
        assert np.array_equal(mirror.fixed_idx, np.array([N // 4, 3 * N // 4]))
    else:
        assert mirror.fixed_idx.size == 0


def test_delta_phi_basis_mirror_signs() -> None:
    N = 8
    mirror = build_mirror_x0(N)
    rng = np.random.default_rng(0)
    K = 3
    delta_rep = rng.standard_normal(size=(K, mirror.rep_idx.size))
    delta_full = delta_rep @ mirror.basis

    np.testing.assert_allclose(delta_full[:, mirror.mirror_idx], -delta_full, atol=1e-12)
    if mirror.fixed_idx.size:
        np.testing.assert_allclose(delta_full[:, mirror.fixed_idx], 0.0, atol=1e-12)


def test_build_mirror_x0_requires_even() -> None:
    with pytest.raises(ValueError):
        build_mirror_x0(5)
