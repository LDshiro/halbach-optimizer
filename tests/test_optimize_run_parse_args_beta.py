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
