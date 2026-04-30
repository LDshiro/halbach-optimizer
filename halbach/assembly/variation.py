from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import SupportsFloat, cast

import numpy as np

from halbach.assembly.types import MagnetError, VirtualMagnet

StrengthModel = Mapping[str, object]
MeasurementNoise = Mapping[str, float]


def _float_value(model: Mapping[str, object], key: str, default: float) -> float:
    raw = model.get(key, default)
    value = float(cast(SupportsFloat, raw))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _validate_sigma(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return value


def _strength_errors(
    count: int,
    seed_rng: np.random.Generator,
    strength_model: StrengthModel,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    mode_raw = strength_model.get("mode", "iid_normal")
    mode = str(mode_raw)

    if mode == "iid_normal":
        mu = _float_value(strength_model, "mu", 0.0)
        sigma = _validate_sigma(_float_value(strength_model, "sigma", 0.0), "sigma")
        return np.asarray(seed_rng.normal(loc=mu, scale=sigma, size=count), dtype=np.float64)

    if mode == "two_lot_mixture":
        weight = _float_value(strength_model, "weight", 0.5)
        if weight < 0.0 or weight > 1.0:
            raise ValueError("weight must be in [0, 1]")
        mu1 = _float_value(strength_model, "mu1", 0.0)
        sigma1 = _validate_sigma(_float_value(strength_model, "sigma1", 0.0), "sigma1")
        mu2 = _float_value(strength_model, "mu2", 0.0)
        sigma2 = _validate_sigma(_float_value(strength_model, "sigma2", 0.0), "sigma2")
        choose_first = seed_rng.random(count) < weight
        lot1 = seed_rng.normal(loc=mu1, scale=sigma1, size=count)
        lot2 = seed_rng.normal(loc=mu2, scale=sigma2, size=count)
        return np.asarray(np.where(choose_first, lot1, lot2), dtype=np.float64)

    if mode == "linear_drift":
        mu0 = _float_value(strength_model, "mu0", 0.0)
        drift = _float_value(strength_model, "drift_per_index", 0.0)
        sigma = _validate_sigma(_float_value(strength_model, "sigma", 0.0), "sigma")
        idx = np.arange(count, dtype=np.float64)
        noise = seed_rng.normal(loc=0.0, scale=sigma, size=count)
        return np.asarray(mu0 + drift * idx + noise, dtype=np.float64)

    raise ValueError(f"Unsupported strength model mode: {mode}")


def _measurement_noise_sigmas(
    measurement_noise: MeasurementNoise | None,
) -> tuple[float, float, float]:
    if measurement_noise is None:
        return 0.0, 0.0, 0.0
    strength_sigma = _validate_sigma(
        float(measurement_noise.get("strength_sigma", 0.0)),
        "measurement_noise.strength_sigma",
    )
    d1_sigma = _validate_sigma(
        float(measurement_noise.get("transverse_component_1_sigma", 0.0)),
        "measurement_noise.transverse_component_1_sigma",
    )
    d2_sigma = _validate_sigma(
        float(measurement_noise.get("transverse_component_2_sigma", 0.0)),
        "measurement_noise.transverse_component_2_sigma",
    )
    return strength_sigma, d1_sigma, d2_sigma


def _quality_values(count: int, quality: float | Sequence[float] | None) -> list[float | None]:
    if quality is None:
        return [None] * count
    arr = np.asarray(quality, dtype=np.float64)
    if arr.ndim == 0:
        value = float(arr)
        if not math.isfinite(value):
            raise ValueError("quality must be finite")
        return [value] * count
    arr = arr.reshape(-1)
    if arr.size != count:
        raise ValueError(f"quality length {arr.size} does not match count={count}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("quality values must be finite")
    return [float(item) for item in arr]


def generate_virtual_magnets(
    count: int,
    seed: int,
    strength_model: StrengthModel,
    direction_sigma_1: float,
    direction_sigma_2: float,
    measurement_noise: MeasurementNoise | None,
    *,
    quality: float | Sequence[float] | None = 1.0,
) -> list[VirtualMagnet]:
    """
    Generate synthetic Plan C magnets.

    true_error and measured_error use [epsilon_parallel, delta_perp_1, delta_perp_2].
    Direction sigmas and transverse measurement noise are rad-equivalent small angles.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    sigma_d1 = _validate_sigma(float(direction_sigma_1), "direction_sigma_1")
    sigma_d2 = _validate_sigma(float(direction_sigma_2), "direction_sigma_2")

    rng = np.random.default_rng(int(seed))
    eps = _strength_errors(count, rng, strength_model)
    d1 = np.asarray(rng.normal(loc=0.0, scale=sigma_d1, size=count), dtype=np.float64)
    d2 = np.asarray(rng.normal(loc=0.0, scale=sigma_d2, size=count), dtype=np.float64)

    noise_eps_sigma, noise_d1_sigma, noise_d2_sigma = _measurement_noise_sigmas(
        measurement_noise
    )
    measured_eps = eps + rng.normal(loc=0.0, scale=noise_eps_sigma, size=count)
    measured_d1 = d1 + rng.normal(loc=0.0, scale=noise_d1_sigma, size=count)
    measured_d2 = d2 + rng.normal(loc=0.0, scale=noise_d2_sigma, size=count)

    qualities = _quality_values(count, quality)
    magnets: list[VirtualMagnet] = []
    for idx in range(count):
        magnets.append(
            VirtualMagnet(
                magnet_id=idx,
                true_error=MagnetError(
                    epsilon_parallel=float(eps[idx]),
                    delta_perp_1=float(d1[idx]),
                    delta_perp_2=float(d2[idx]),
                ),
                measured_error=MagnetError(
                    epsilon_parallel=float(measured_eps[idx]),
                    delta_perp_1=float(measured_d1[idx]),
                    delta_perp_2=float(measured_d2[idx]),
                ),
                quality=qualities[idx],
            )
        )
    return magnets


__all__ = ["MeasurementNoise", "StrengthModel", "generate_virtual_magnets"]
