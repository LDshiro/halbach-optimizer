import numpy as np
import pytest

from halbach.assembly.orientations import default_orientations, rotate_error_for_orientation
from halbach.assembly.types import MagnetError


def test_default_orientations_are_discrete4() -> None:
    orientations = default_orientations()
    assert [item.id for item in orientations] == ["O0", "O90", "O180", "O270"]
    np.testing.assert_allclose([item.angle_deg for item in orientations], [0.0, 90.0, 180.0, 270.0])


def test_rotate_error_for_orientation_discrete4() -> None:
    error = MagnetError(epsilon_parallel=0.123, delta_perp_1=0.2, delta_perp_2=-0.3)

    o0 = rotate_error_for_orientation(error, "O0")
    np.testing.assert_allclose(
        [o0.epsilon_parallel, o0.delta_perp_1, o0.delta_perp_2],
        [0.123, 0.2, -0.3],
        atol=1e-15,
    )

    o90 = rotate_error_for_orientation(error, "O90")
    np.testing.assert_allclose(
        [o90.epsilon_parallel, o90.delta_perp_1, o90.delta_perp_2],
        [0.123, 0.3, 0.2],
        atol=1e-15,
    )

    o180 = rotate_error_for_orientation(error, "O180")
    np.testing.assert_allclose(
        [o180.epsilon_parallel, o180.delta_perp_1, o180.delta_perp_2],
        [0.123, -0.2, 0.3],
        atol=1e-15,
    )

    o270 = rotate_error_for_orientation(error, "O270")
    np.testing.assert_allclose(
        [o270.epsilon_parallel, o270.delta_perp_1, o270.delta_perp_2],
        [0.123, -0.3, -0.2],
        atol=1e-15,
    )


def test_rotate_error_rejects_unknown_orientation() -> None:
    with pytest.raises(ValueError):
        rotate_error_for_orientation(
            MagnetError(epsilon_parallel=0.0, delta_perp_1=0.0, delta_perp_2=0.0),
            "O45",
        )
