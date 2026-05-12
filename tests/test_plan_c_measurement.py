import numpy as np
import pytest

from halbach.assembly.measurement import measured_magnet_from_direction


def test_measured_magnet_from_nominal_direction_has_zero_transverse_error() -> None:
    magnet = measured_magnet_from_direction(1.0, [0.0, 0.0, 1.0])

    assert magnet.error.epsilon_parallel == 0.0
    assert magnet.error.delta_perp_1 == 0.0
    assert magnet.error.delta_perp_2 == 0.0
    np.testing.assert_allclose(magnet.direction, [0.0, 0.0, 1.0])


def test_measured_magnet_strength_error_and_unnormalized_direction() -> None:
    magnet = measured_magnet_from_direction(
        1.002,
        [0.0, 0.0, 10.0],
        nominal_magnitude=1.0,
        quality=0.98,
        cluster_id="S00_A00",
    )

    assert magnet.error.epsilon_parallel == pytest.approx(0.002)
    assert magnet.error.delta_perp_1 == 0.0
    assert magnet.error.delta_perp_2 == 0.0
    assert magnet.quality == 0.98
    assert magnet.cluster_id == "S00_A00"
    np.testing.assert_allclose(magnet.direction, [0.0, 0.0, 1.0])


def test_measured_magnet_small_angle_transverse_components_use_ratios() -> None:
    magnet = measured_magnet_from_direction(1.0, [0.002, -0.001, 1.0])

    assert magnet.error.delta_perp_1 == pytest.approx(0.002)
    assert magnet.error.delta_perp_2 == pytest.approx(-0.001)


@pytest.mark.parametrize(
    ("moment", "direction", "nominal"),
    [
        (1.0, [0.0, 0.0, 1.0], 0.0),
        (1.0, [0.0, 0.0, 0.0], 1.0),
        (1.0, [1.0, 0.0, 0.0], 1.0),
    ],
)
def test_measured_magnet_rejects_invalid_inputs(
    moment: float,
    direction: list[float],
    nominal: float,
) -> None:
    with pytest.raises(ValueError):
        measured_magnet_from_direction(moment, direction, nominal_magnitude=nominal)
