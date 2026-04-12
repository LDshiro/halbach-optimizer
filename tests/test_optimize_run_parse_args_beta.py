from halbach.cli.optimize_run import parse_args


def test_parse_args_accepts_beta_flags() -> None:
    args = parse_args(
        [
            "--in",
            "dummy",
            "--out",
            "dummy_out",
            "--enable-beta-tilt-x",
            "--beta-tilt-x-bound-deg",
            "15.5",
        ]
    )
    assert args.enable_beta_tilt_x is True
    assert args.beta_tilt_x_bound_deg == 15.5


def test_parse_args_accepts_fix_radius_layer_mode() -> None:
    args = parse_args(
        [
            "--in",
            "dummy",
            "--out",
            "dummy_out",
            "--fix-radius-layer-mode",
            "ends",
            "--fix-center-radius-layers",
            "4",
        ]
    )
    assert args.fix_radius_layer_mode == "ends"
    assert args.fix_center_radius_layers == 4
