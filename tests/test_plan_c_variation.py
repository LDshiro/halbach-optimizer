import pytest

from halbach.assembly.variation import generate_virtual_magnets


def _error_rows(magnets):
    return [
        (
            magnet.true_error.epsilon_parallel,
            magnet.true_error.delta_perp_1,
            magnet.true_error.delta_perp_2,
            magnet.measured_error.epsilon_parallel,
            magnet.measured_error.delta_perp_1,
            magnet.measured_error.delta_perp_2,
            magnet.quality,
        )
        for magnet in magnets
    ]


def test_generate_virtual_magnets_is_seed_reproducible() -> None:
    kwargs = dict(
        count=5,
        seed=123,
        strength_model={"mode": "iid_normal", "mu": 0.01, "sigma": 0.002},
        direction_sigma_1=0.001,
        direction_sigma_2=0.002,
        measurement_noise={
            "strength_sigma": 0.0001,
            "transverse_component_1_sigma": 0.0002,
            "transverse_component_2_sigma": 0.0003,
        },
    )
    magnets_a = generate_virtual_magnets(**kwargs)
    magnets_b = generate_virtual_magnets(**kwargs)

    assert len(magnets_a) == 5
    assert _error_rows(magnets_a) == _error_rows(magnets_b)
    assert [magnet.magnet_id for magnet in magnets_a] == list(range(5))


def test_iid_zero_strength_and_zero_noise_keeps_true_and_measured_equal() -> None:
    magnets = generate_virtual_magnets(
        count=4,
        seed=1,
        strength_model={"mode": "iid_normal", "mu": 0.0, "sigma": 0.0},
        direction_sigma_1=0.0,
        direction_sigma_2=0.0,
        measurement_noise={
            "strength_sigma": 0.0,
            "transverse_component_1_sigma": 0.0,
            "transverse_component_2_sigma": 0.0,
        },
    )

    for magnet in magnets:
        assert magnet.true_error.epsilon_parallel == 0.0
        assert magnet.true_error.delta_perp_1 == 0.0
        assert magnet.true_error.delta_perp_2 == 0.0
        assert magnet.measured_error == magnet.true_error


def test_linear_drift_increases_strength_by_index_without_noise() -> None:
    magnets = generate_virtual_magnets(
        count=4,
        seed=1,
        strength_model={
            "mode": "linear_drift",
            "mu0": 0.001,
            "drift_per_index": 0.01,
            "sigma": 0.0,
        },
        direction_sigma_1=0.0,
        direction_sigma_2=0.0,
        measurement_noise=None,
    )

    assert [magnet.true_error.epsilon_parallel for magnet in magnets] == pytest.approx(
        [0.001, 0.011, 0.021, 0.031]
    )


def test_two_lot_mixture_validates_weight_and_sigma() -> None:
    magnets = generate_virtual_magnets(
        count=3,
        seed=1,
        strength_model={
            "mode": "two_lot_mixture",
            "weight": 1.0,
            "mu1": 0.02,
            "sigma1": 0.0,
            "mu2": -0.02,
            "sigma2": 0.0,
        },
        direction_sigma_1=0.0,
        direction_sigma_2=0.0,
        measurement_noise=None,
    )
    assert [magnet.true_error.epsilon_parallel for magnet in magnets] == pytest.approx(
        [0.02, 0.02, 0.02]
    )

    with pytest.raises(ValueError):
        generate_virtual_magnets(
            count=3,
            seed=1,
            strength_model={"mode": "two_lot_mixture", "weight": 1.1},
            direction_sigma_1=0.0,
            direction_sigma_2=0.0,
            measurement_noise=None,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"direction_sigma_1": -1.0},
        {"direction_sigma_2": -1.0},
        {"strength_model": {"mode": "iid_normal", "sigma": -1.0}},
        {"measurement_noise": {"strength_sigma": -1.0}},
    ],
)
def test_generate_virtual_magnets_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "count": 3,
        "seed": 1,
        "strength_model": {"mode": "iid_normal", "mu": 0.0, "sigma": 0.0},
        "direction_sigma_1": 0.0,
        "direction_sigma_2": 0.0,
        "measurement_noise": None,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        generate_virtual_magnets(**base)
