from __future__ import annotations

import numpy as np

MU0_SI = 4 * np.pi * 1e-7
DIPOLE_FACTOR_SI = MU0_SI / (4 * np.pi)
FIELD_SCALE_DEFAULT = 1e6

mu0 = MU0_SI
FACTOR = DIPOLE_FACTOR_SI
phi0 = -np.pi / 2
m0 = 1.0

__all__ = [
    "MU0_SI",
    "DIPOLE_FACTOR_SI",
    "FIELD_SCALE_DEFAULT",
    "mu0",
    "FACTOR",
    "phi0",
    "m0",
]
