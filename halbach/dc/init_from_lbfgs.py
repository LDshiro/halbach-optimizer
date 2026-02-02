from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from halbach.angles_runtime import angle_model_from_run, phi_rkn_from_run
from halbach.constants import phi0
from halbach.run_io import load_run

ALLOWED_INIT_ANGLE_MODELS: Final[set[str]] = {"delta-rep-x0", "fourier-x0"}


def load_phi_flat_from_lbfgs_run(
    run_dir: Path,
    *,
    expected_R: int | None = None,
    expected_K: int | None = None,
    expected_N: int | None = None,
) -> tuple[NDArray[np.float64], str]:
    run = load_run(run_dir)
    if run.meta.get("framework") == "dc":
        raise ValueError("init-run must be an L-BFGS run, not a DC/CCP run")

    model = angle_model_from_run(run)
    if model not in ALLOWED_INIT_ANGLE_MODELS:
        allowed = ", ".join(sorted(ALLOWED_INIT_ANGLE_MODELS))
        raise ValueError(f"init-run angle_model must be one of [{allowed}] (got {model})")

    if expected_R is not None and int(run.geometry.R) != int(expected_R):
        raise ValueError(f"init-run R={int(run.geometry.R)} does not match expected R={expected_R}")
    if expected_K is not None and int(run.geometry.K) != int(expected_K):
        raise ValueError(f"init-run K={int(run.geometry.K)} does not match expected K={expected_K}")
    if expected_N is not None and int(run.geometry.N) != int(expected_N):
        raise ValueError(f"init-run N={int(run.geometry.N)} does not match expected N={expected_N}")

    phi_rkn = phi_rkn_from_run(run, phi0=phi0)
    return np.asarray(phi_rkn, dtype=np.float64).reshape(-1), model


__all__ = ["ALLOWED_INIT_ANGLE_MODELS", "load_phi_flat_from_lbfgs_run"]
