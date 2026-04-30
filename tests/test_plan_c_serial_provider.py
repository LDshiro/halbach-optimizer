import json

import pytest

import halbach.assembly.measurement as measurement_module
from halbach.assembly.measurement import (
    FakeSerialMeasurementProvider,
    MeasurementParseError,
    MeasurementQualityError,
    MeasurementTimeoutError,
    SerialDependencyError,
    SerialMeasurementProvider,
    parse_serial_measurement_line,
)


def test_fake_serial_provider_reads_standard_json_line() -> None:
    provider = FakeSerialMeasurementProvider(
        [
            json.dumps(
                {
                    "magnet_id": 7,
                    "moment_magnitude": 1.002,
                    "direction": [0.0, 0.0, 1.0],
                    "quality": 0.99,
                }
            )
        ]
    )

    magnet = provider.next_magnet()

    assert provider.position == 1
    assert magnet.magnet_id == 7
    assert magnet.measured_error.epsilon_parallel == pytest.approx(0.002)
    assert magnet.measured_error.delta_perp_1 == pytest.approx(0.0)
    assert magnet.measured_error.delta_perp_2 == pytest.approx(0.0)
    assert magnet.quality == pytest.approx(0.99)


def test_serial_parser_normalizes_direction_before_conversion() -> None:
    normalized = parse_serial_measurement_line(
        json.dumps({"moment_magnitude": 1.002, "direction": [0.0, 0.0, 1.0]}),
        default_magnet_id=0,
    )
    unnormalized = parse_serial_measurement_line(
        json.dumps({"moment_magnitude": 1.002, "direction": [0.0, 0.0, 2.0]}),
        default_magnet_id=0,
    )

    assert unnormalized.measured_error == normalized.measured_error


def test_fake_serial_provider_keeps_position_on_malformed_json() -> None:
    provider = FakeSerialMeasurementProvider(["{not-json"])

    with pytest.raises(MeasurementParseError):
        provider.next_magnet()

    assert provider.position == 0


def test_serial_parser_rejects_missing_direction() -> None:
    with pytest.raises(MeasurementParseError, match="direction"):
        parse_serial_measurement_line(
            json.dumps({"moment_magnitude": 1.0}),
            default_magnet_id=0,
        )


def test_fake_serial_provider_treats_empty_line_as_timeout() -> None:
    provider = FakeSerialMeasurementProvider([""])

    with pytest.raises(MeasurementTimeoutError):
        provider.next_magnet()

    assert provider.position == 0


def test_quality_below_threshold_is_quarantined_without_consuming_line() -> None:
    provider = FakeSerialMeasurementProvider(
        [
            json.dumps(
                {
                    "moment_magnitude": 1.0,
                    "direction": [0.0, 0.0, 1.0],
                    "quality": 0.5,
                }
            )
        ],
        quality_threshold=0.9,
    )

    with pytest.raises(MeasurementQualityError) as exc_info:
        provider.next_magnet()

    assert exc_info.value.quarantine_id == "Q_MEASUREMENT_UNSTABLE"
    assert provider.position == 0


def test_fake_serial_provider_keeps_legacy_epsilon_delta_json_support() -> None:
    provider = FakeSerialMeasurementProvider(
        [
            json.dumps(
                {
                    "magnet_id": 4,
                    "epsilon_parallel": 0.04,
                    "delta_perp_1": 0.001,
                    "delta_perp_2": -0.001,
                    "quality": 0.99,
                }
            )
        ]
    )

    magnet = provider.next_magnet()

    assert magnet.magnet_id == 4
    assert magnet.measured_error.epsilon_parallel == pytest.approx(0.04)
    assert magnet.quality == pytest.approx(0.99)


def test_serial_provider_reports_missing_pyserial(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_missing_serial(_name: str) -> object:
        raise ModuleNotFoundError("serial")

    monkeypatch.setattr(measurement_module.importlib, "import_module", _raise_missing_serial)

    with pytest.raises(SerialDependencyError):
        SerialMeasurementProvider("COM_TEST")
