from halbach.assembly.config import (
    default_plan_c_config,
    load_plan_c_config,
    plan_c_config_from_dict,
    plan_c_config_to_dict,
    save_plan_c_config_json,
)


def test_default_plan_c_config_matches_step6_baseline() -> None:
    config = default_plan_c_config()

    assert config.schema_version == 1
    assert config.work_units.mode == "auto"
    assert config.work_units.large_ring_threshold == 60
    assert config.roi.mode == "surface-fibonacci"
    assert config.roi.samples == 300
    assert config.roi.radius_m == 0.14
    assert config.online_assignment.decision_engine == "linear_sensitivity"
    assert config.sensitivity.dimension == 30
    assert config.sensitivity.finite_difference_step == 0.001
    assert config.clusters.binning == "quantile"
    assert config.clusters.strength_count == 10
    assert config.clusters.angle_count == 5
    assert config.clusters.strength_sigma_step == 0.5
    assert config.clusters.angle_sigma_step == 0.5
    assert config.reject.max_fraction == 0.10
    assert config.measurement_noise.strength_sigma == 0.001


def test_plan_c_config_partial_dict_merges_with_defaults() -> None:
    config = plan_c_config_from_dict(
        {
            "run_dir": "runs/demo_opt",
            "roi": {"samples": 12, "radius_m": 0.03},
            "online_assignment": {"seed": 99},
        }
    )

    assert config.run_dir == "runs/demo_opt"
    assert config.roi.samples == 12
    assert config.roi.radius_m == 0.03
    assert config.roi.mode == "surface-fibonacci"
    assert config.online_assignment.seed == 99
    assert config.work_units.mode == "auto"


def test_plan_c_config_json_roundtrip(tmp_path) -> None:
    config = plan_c_config_from_dict(
        {
            "run_dir": "runs/demo_opt",
            "clusters": {
                "binning": "sigma_band",
                "strength_count": 4,
                "angle_count": 2,
                "strength_sigma_step": 0.25,
                "angle_sigma_step": 0.75,
            },
        }
    )
    path = tmp_path / "plan_c_config.json"

    save_plan_c_config_json(path, config)
    loaded = load_plan_c_config(path)

    assert loaded == config
    data = plan_c_config_to_dict(loaded)
    assert data["schema_version"] == 1
    assert data["run_dir"] == "runs/demo_opt"


def test_plan_c_config_accepts_ring_by_ring_work_unit_modes() -> None:
    for mode in (
        "stack_by_stack_outer_to_inner",
        "layer_by_layer_outer_to_inner",
        "ring_by_ring_outer_to_inner",
        "mirror_ring_pair",
    ):
        config = plan_c_config_from_dict({"work_units": {"mode": mode}})
        assert config.work_units.mode == mode
        data = plan_c_config_to_dict(config)
        assert data["work_units"]["mode"] == mode
