from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from halbach.geom import RoiMode

WorkUnitConfigMode = Literal[
    "all_slots",
    "single_physical_ring",
    "ring_group",
    "ring_by_ring_outer_to_inner",
    "mirror_ring_pair",
    "auto",
]


@dataclass(frozen=True)
class PlanCWorkUnitsConfig:
    mode: WorkUnitConfigMode = "auto"
    large_ring_threshold: int = 60
    small_total_threshold: int = 150
    allow_mirror_pair_swap: bool = True


@dataclass(frozen=True)
class PlanCRoiConfig:
    mode: RoiMode = "surface-fibonacci"
    samples: int = 300
    radius_m: float = 0.14
    step_m: float = 0.01
    seed: int = 0


@dataclass(frozen=True)
class PlanCOnlineAssignmentConfig:
    decision_engine: str = "linear_sensitivity"
    alternative_engine: str = "sequential_self_consistent"
    future_virtual_mode: str = "quantile"
    local_search_swaps: int = 20000
    restarts: int = 8
    seed: int = 1234


@dataclass(frozen=True)
class PlanCSensitivityConfig:
    dimension: int = 30
    remove_center_strength_mode: bool = True
    finite_difference_step: float = 0.001


@dataclass(frozen=True)
class PlanCClustersConfig:
    strength_count: int = 10
    angle_count: int = 3
    transverse_2_weight: float = 1.0


@dataclass(frozen=True)
class PlanCRejectConfig:
    policy: str = "isolate_up_to_fraction"
    max_fraction: float = 0.10
    prefer_direction_outliers: bool = True


@dataclass(frozen=True)
class PlanCMeasurementNoiseConfig:
    strength_sigma: float = 0.001
    transverse_component_1_sigma: float = 0.001
    transverse_component_2_sigma: float = 0.001


@dataclass(frozen=True)
class PlanCConfig:
    schema_version: int = 1
    run_dir: str = ""
    work_units: PlanCWorkUnitsConfig = PlanCWorkUnitsConfig()
    roi: PlanCRoiConfig = PlanCRoiConfig()
    online_assignment: PlanCOnlineAssignmentConfig = PlanCOnlineAssignmentConfig()
    sensitivity: PlanCSensitivityConfig = PlanCSensitivityConfig()
    clusters: PlanCClustersConfig = PlanCClustersConfig()
    reject: PlanCRejectConfig = PlanCRejectConfig()
    measurement_noise: PlanCMeasurementNoiseConfig = PlanCMeasurementNoiseConfig()


def default_plan_c_config() -> PlanCConfig:
    """Return the JSON-friendly Step 6 Plan C default configuration."""
    return PlanCConfig()


def plan_c_config_to_dict(config: PlanCConfig) -> dict[str, object]:
    """Serialize PlanCConfig to a JSON-compatible dict."""
    return {
        "schema_version": config.schema_version,
        "run_dir": config.run_dir,
        "work_units": {
            "mode": config.work_units.mode,
            "large_ring_threshold": config.work_units.large_ring_threshold,
            "small_total_threshold": config.work_units.small_total_threshold,
            "allow_mirror_pair_swap": config.work_units.allow_mirror_pair_swap,
        },
        "roi": {
            "mode": config.roi.mode,
            "samples": config.roi.samples,
            "radius_m": config.roi.radius_m,
            "step_m": config.roi.step_m,
            "seed": config.roi.seed,
        },
        "online_assignment": {
            "decision_engine": config.online_assignment.decision_engine,
            "alternative_engine": config.online_assignment.alternative_engine,
            "future_virtual_mode": config.online_assignment.future_virtual_mode,
            "local_search_swaps": config.online_assignment.local_search_swaps,
            "restarts": config.online_assignment.restarts,
            "seed": config.online_assignment.seed,
        },
        "sensitivity": {
            "dimension": config.sensitivity.dimension,
            "remove_center_strength_mode": config.sensitivity.remove_center_strength_mode,
            "finite_difference_step": config.sensitivity.finite_difference_step,
        },
        "clusters": {
            "strength_count": config.clusters.strength_count,
            "angle_count": config.clusters.angle_count,
            "transverse_2_weight": config.clusters.transverse_2_weight,
        },
        "reject": {
            "policy": config.reject.policy,
            "max_fraction": config.reject.max_fraction,
            "prefer_direction_outliers": config.reject.prefer_direction_outliers,
        },
        "measurement_noise": {
            "strength_sigma": config.measurement_noise.strength_sigma,
            "transverse_component_1_sigma": (
                config.measurement_noise.transverse_component_1_sigma
            ),
            "transverse_component_2_sigma": (
                config.measurement_noise.transverse_component_2_sigma
            ),
        },
    }


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    raw = data.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], raw)


def _str_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    raise ValueError(f"expected string-compatible value, got {type(value).__name__}")


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("expected int value, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected int value, got {type(value).__name__}")


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("expected float value, got bool")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"expected float value, got {type(value).__name__}")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"expected bool value, got {type(value).__name__}")


def plan_c_config_from_dict(data: dict[str, object]) -> PlanCConfig:
    """Deserialize PlanCConfig, merging missing fields with defaults."""
    base = default_plan_c_config()
    work_units = _section(data, "work_units")
    roi = _section(data, "roi")
    online = _section(data, "online_assignment")
    sensitivity = _section(data, "sensitivity")
    clusters = _section(data, "clusters")
    reject = _section(data, "reject")
    measurement_noise = _section(data, "measurement_noise")

    return PlanCConfig(
        schema_version=_int_value(data.get("schema_version", base.schema_version)),
        run_dir=_str_value(data.get("run_dir", base.run_dir)),
        work_units=PlanCWorkUnitsConfig(
            mode=cast(
                WorkUnitConfigMode,
                _str_value(work_units.get("mode", base.work_units.mode)),
            ),
            large_ring_threshold=_int_value(
                work_units.get("large_ring_threshold", base.work_units.large_ring_threshold)
            ),
            small_total_threshold=_int_value(
                work_units.get("small_total_threshold", base.work_units.small_total_threshold)
            ),
            allow_mirror_pair_swap=_bool_value(
                work_units.get("allow_mirror_pair_swap", base.work_units.allow_mirror_pair_swap)
            ),
        ),
        roi=PlanCRoiConfig(
            mode=cast(RoiMode, _str_value(roi.get("mode", base.roi.mode))),
            samples=_int_value(roi.get("samples", base.roi.samples)),
            radius_m=_float_value(roi.get("radius_m", base.roi.radius_m)),
            step_m=_float_value(roi.get("step_m", base.roi.step_m)),
            seed=_int_value(roi.get("seed", base.roi.seed)),
        ),
        online_assignment=PlanCOnlineAssignmentConfig(
            decision_engine=_str_value(
                online.get("decision_engine", base.online_assignment.decision_engine)
            ),
            alternative_engine=_str_value(
                online.get("alternative_engine", base.online_assignment.alternative_engine)
            ),
            future_virtual_mode=_str_value(
                online.get("future_virtual_mode", base.online_assignment.future_virtual_mode)
            ),
            local_search_swaps=_int_value(
                online.get("local_search_swaps", base.online_assignment.local_search_swaps)
            ),
            restarts=_int_value(online.get("restarts", base.online_assignment.restarts)),
            seed=_int_value(online.get("seed", base.online_assignment.seed)),
        ),
        sensitivity=PlanCSensitivityConfig(
            dimension=_int_value(sensitivity.get("dimension", base.sensitivity.dimension)),
            remove_center_strength_mode=_bool_value(
                sensitivity.get(
                    "remove_center_strength_mode",
                    base.sensitivity.remove_center_strength_mode,
                )
            ),
            finite_difference_step=_float_value(
                sensitivity.get(
                    "finite_difference_step",
                    base.sensitivity.finite_difference_step,
                )
            ),
        ),
        clusters=PlanCClustersConfig(
            strength_count=_int_value(
                clusters.get("strength_count", base.clusters.strength_count)
            ),
            angle_count=_int_value(clusters.get("angle_count", base.clusters.angle_count)),
            transverse_2_weight=_float_value(
                clusters.get("transverse_2_weight", base.clusters.transverse_2_weight)
            ),
        ),
        reject=PlanCRejectConfig(
            policy=_str_value(reject.get("policy", base.reject.policy)),
            max_fraction=_float_value(reject.get("max_fraction", base.reject.max_fraction)),
            prefer_direction_outliers=_bool_value(
                reject.get("prefer_direction_outliers", base.reject.prefer_direction_outliers)
            ),
        ),
        measurement_noise=PlanCMeasurementNoiseConfig(
            strength_sigma=_float_value(
                measurement_noise.get(
                    "strength_sigma",
                    base.measurement_noise.strength_sigma,
                )
            ),
            transverse_component_1_sigma=_float_value(
                measurement_noise.get(
                    "transverse_component_1_sigma",
                    base.measurement_noise.transverse_component_1_sigma,
                )
            ),
            transverse_component_2_sigma=_float_value(
                measurement_noise.get(
                    "transverse_component_2_sigma",
                    base.measurement_noise.transverse_component_2_sigma,
                )
            ),
        ),
    )


def save_plan_c_config_json(path: str | Path, config: PlanCConfig) -> None:
    """Write Plan C config as JSON without adding YAML dependencies."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan_c_config_to_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_plan_c_config_json(path: str | Path) -> PlanCConfig:
    """Read Plan C JSON config."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Plan C config JSON must contain an object")
    return plan_c_config_from_dict(cast(dict[str, object], raw))


def load_plan_c_config(path: str | Path) -> PlanCConfig:
    """Read a Plan C config file. Step 6 supports JSON only."""
    in_path = Path(path)
    if in_path.suffix.lower() not in (".json", ""):
        raise ValueError("Step 6 supports JSON config only; YAML support is intentionally deferred")
    return load_plan_c_config_json(in_path)


__all__ = [
    "PlanCClustersConfig",
    "PlanCConfig",
    "PlanCMeasurementNoiseConfig",
    "PlanCOnlineAssignmentConfig",
    "PlanCRejectConfig",
    "PlanCRoiConfig",
    "PlanCSensitivityConfig",
    "PlanCWorkUnitsConfig",
    "default_plan_c_config",
    "load_plan_c_config",
    "load_plan_c_config_json",
    "plan_c_config_from_dict",
    "plan_c_config_to_dict",
    "save_plan_c_config_json",
]
