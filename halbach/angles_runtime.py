from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from halbach.constants import phi0
from halbach.run_types import RunBundle
from halbach.symmetry import build_mirror_x0, expand_delta_phi
from halbach.symmetry_fourier import build_fourier_x0_features, delta_full_from_fourier

AngleModelKind = Literal["legacy-alpha", "delta-rep-x0", "fourier-x0"]
PHI0_DEFAULT = phi0


def angle_model_from_run(run: RunBundle) -> AngleModelKind:
    raw = run.meta.get("angle_model")
    if raw is None:
        return "legacy-alpha"
    if not isinstance(raw, str):
        raise ValueError("meta['angle_model'] must be a string")
    if raw not in ("legacy-alpha", "delta-rep-x0", "fourier-x0"):
        raise ValueError(f"Unsupported angle_model: {raw}")
    return cast(AngleModelKind, raw)


def _pick_extras_array(run: RunBundle, keys: list[str]) -> NDArray[np.float64] | None:
    for key in keys:
        if key in run.results.extras:
            return np.asarray(run.results.extras[key], dtype=np.float64)
    return None


def phi_rkn_from_run(run: RunBundle, *, phi0: float = PHI0_DEFAULT) -> NDArray[np.float64]:
    geom = run.geometry
    N = int(geom.N)
    if N % 2 != 0:
        raise ValueError("N must be even to build x=0 mirror phi")

    theta = np.asarray(geom.theta, dtype=np.float64)
    base = 2.0 * theta + float(phi0)
    R = int(geom.R)
    K = int(geom.K)

    model = angle_model_from_run(run)
    if model == "legacy-alpha":
        alphas = np.asarray(run.results.alphas, dtype=np.float64)
        sin2 = np.asarray(geom.sin2, dtype=np.float64)
        phi_rkn = base[None, None, :] + alphas[:, :, None] * sin2[None, None, :]
        return np.asarray(phi_rkn, dtype=np.float64)

    if model == "delta-rep-x0":
        delta_rep = _pick_extras_array(run, ["delta_rep_opt", "delta_rep"])
        if delta_rep is None:
            raise KeyError("delta-rep-x0 requires delta_rep[_opt] in results extras")
        if delta_rep.shape[0] != K:
            raise ValueError(f"delta_rep shape {delta_rep.shape} does not match K={K}")
        mirror = build_mirror_x0(N)
        if delta_rep.shape[1] != mirror.rep_idx.size:
            raise ValueError("delta_rep width does not match mirror rep size")
        delta_full = expand_delta_phi(delta_rep, mirror.basis)
        phi_kn = base[None, :] + delta_full
        phi_rkn = np.broadcast_to(phi_kn[None, :, :], (R, K, N))
        return np.asarray(phi_rkn, dtype=np.float64)

    if model == "fourier-x0":
        coeffs = _pick_extras_array(run, ["fourier_coeffs_opt", "fourier_coeffs"])
        if coeffs is None:
            raise KeyError("fourier-x0 requires fourier_coeffs[_opt] in results extras")
        if coeffs.shape[0] != K:
            raise ValueError(f"fourier_coeffs shape {coeffs.shape} does not match K={K}")
        meta_H = run.meta.get("fourier_H")
        if meta_H is None:
            H = int(coeffs.shape[1] // 2)
        else:
            H = int(meta_H)
        cos_odd, sin_even = build_fourier_x0_features(theta, H)
        delta_full = delta_full_from_fourier(coeffs, cos_odd, sin_even)
        phi_kn = base[None, :] + delta_full
        phi_rkn = np.broadcast_to(phi_kn[None, :, :], (R, K, N))
        return np.asarray(phi_rkn, dtype=np.float64)

    raise ValueError(f"Unsupported angle_model: {model}")


__all__ = ["AngleModelKind", "angle_model_from_run", "phi_rkn_from_run"]
