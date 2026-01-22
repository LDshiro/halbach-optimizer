from __future__ import annotations

import numpy as np

mu0 = 4 * np.pi * 1e-7
FACTOR = mu0 / (4 * np.pi)
phi0 = -np.pi / 2
m0 = 1.0

__all__ = ["mu0", "FACTOR", "phi0", "m0"]
