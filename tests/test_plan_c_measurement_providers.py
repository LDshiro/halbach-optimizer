import json

import pytest

from halbach.assembly.measurement import (
    CsvMeasurementProvider,
    FakeSerialMeasurementProvider,
    ManualMeasurementProvider,
    SyntheticMeasurementProvider,
)
from halbach.assembly.types import MagnetError, VirtualMagnet


def _magnet(magnet_id: int, eps: float) -> VirtualMagnet:
    error = MagnetError(epsilon_parallel=eps, delta_perp_1=0.0, delta_perp_2=0.0)
    return VirtualMagnet(
        magnet_id=magnet_id,
        true_error=error,
        measured_error=error,
        quality=1.0,
    )


def test_synthetic_provider_supports_resume_position() -> None:
    provider = SyntheticMeasurementProvider([_magnet(0, 0.0), _magnet(1, 0.1)])

    assert provider.next_magnet().magnet_id == 0
    assert provider.position == 1
    provider.set_position(0)
    assert provider.next_magnet().magnet_id == 0


def test_csv_provider_returns_magnets_in_file_order(tmp_path) -> None:
    path = tmp_path / "measurements.csv"
    path.write_text(
        "\n".join(
            [
                "magnet_id,true_epsilon_parallel,true_delta_perp_1,true_delta_perp_2,measured_epsilon_parallel,measured_delta_perp_1,measured_delta_perp_2,quality",
                "10,0.01,0.001,0.002,0.011,0.0011,0.0021,0.95",
                "11,-0.02,-0.001,-0.002,-0.021,-0.0011,-0.0021,0.90",
            ]
        ),
        encoding="utf-8",
    )
    provider = CsvMeasurementProvider(path)

    first = provider.next_magnet()
    second = provider.next_magnet()

    assert first.magnet_id == 10
    assert first.true_error.epsilon_parallel == pytest.approx(0.01)
    assert first.measured_error.epsilon_parallel == pytest.approx(0.011)
    assert first.quality == pytest.approx(0.95)
    assert second.magnet_id == 11
    with pytest.raises(StopIteration):
        provider.next_magnet()


def test_manual_provider_accepts_late_measurements() -> None:
    provider = ManualMeasurementProvider()
    provider.submit_magnet(_magnet(3, 0.03))

    assert provider.next_magnet().magnet_id == 3


def test_fake_serial_provider_reads_json_lines() -> None:
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
