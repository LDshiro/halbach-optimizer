from halbach.cli.optimize_run import parse_args


def test_parse_args_accepts_sc_flags() -> None:
    args = parse_args(
        [
            "--in",
            "dummy",
            "--out",
            "dummy_out",
            "--mag-model",
            "self-consistent-easy-axis",
            "--sc-chi",
            "0.05",
            "--sc-Nd",
            "0.3333333333",
            "--sc-iters",
            "10",
            "--sc-omega",
            "0.5",
            "--sc-near-wz",
            "1",
            "--sc-near-wphi",
            "2",
        ]
    )
    assert args.mag_model == "self-consistent-easy-axis"
    assert args.sc_chi == 0.05
    assert args.sc_Nd == 0.3333333333
    assert args.sc_iters == 10
    assert args.sc_omega == 0.5
    assert args.sc_near_wz == 1
    assert args.sc_near_wphi == 2
