from __future__ import annotations

import os
from typing import Any

import numpy as np


def load_nominal_npz(path: str) -> dict[str, Any]:
    """
    Load nominal design .npz and return geometry-related fields in a dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nominal NPZ not found: {path}")

    dat = np.load(path, allow_pickle=True)
    alphas = np.array(dat["alphas_opt"], dtype=float)
    r_bases = np.array(dat["r_bases_opt"], dtype=float).reshape(-1)
    theta = np.array(dat["theta"], dtype=float).reshape(-1)

    if "sin2" in dat:
        sin2 = np.array(dat["sin2"], dtype=float).reshape(-1)
    elif "sin2th" in dat:
        sin2 = np.array(dat["sin2th"], dtype=float).reshape(-1)
    else:
        sin2 = np.sin(2.0 * theta)

    z_layers = np.array(dat["z_layers"], dtype=float).reshape(-1)
    ring_offsets = np.array(dat["ring_offsets"], dtype=float).reshape(-1)

    N = theta.size
    R, K = alphas.shape
    cth, sth = np.cos(theta), np.sin(theta)
    dzs = np.diff(z_layers)
    dz = float(np.median(np.abs(dzs))) if dzs.size > 0 else 0.01
    Lz = dz * K

    return dict(
        alphas=alphas,
        r_bases=r_bases,
        theta=theta,
        sin2=sin2,
        z_layers=z_layers,
        ring_offsets=ring_offsets,
        N=N,
        R=R,
        K=K,
        cth=cth,
        sth=sth,
        dz=dz,
        Lz=Lz,
    )


__all__ = ["load_nominal_npz"]
