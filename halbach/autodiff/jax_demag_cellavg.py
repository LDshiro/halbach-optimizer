from __future__ import annotations

import math
from typing import Any, cast

import jax
import jax.numpy as jnp

cast(Any, jax.config).update("jax_enable_x64", True)

EPS = 1e-30
FOUR_PI = 4.0 * math.pi

_EPS_LIST = jnp.array(
    [
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=jnp.int32,
)


def _safe_log(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(x, EPS))


def _atan2_ratio(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    mask = (num == 0.0) & (den == 0.0)
    num_safe = jnp.where(mask, 0.0, num)
    den_safe = jnp.where(mask, EPS, den)
    angle = jnp.arctan2(num_safe, den_safe)
    angle = jnp.where(num_safe == 0.0, 0.0, angle)
    adjust = jnp.where(den_safe < 0.0, jnp.sign(num_safe) * math.pi, 0.0)
    return angle - adjust


def _f(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    R = jnp.sqrt(x * x + y * y + z * z) + EPS
    term1 = (1.0 / 6.0) * (2.0 * x * x - y * y - z * z) * R
    term2 = 0.5 * y * (z * z - x * x) * _safe_log(y + R)
    term3 = 0.5 * z * (y * y - x * x) * _safe_log(z + R)
    term4 = -x * y * z * _atan2_ratio(y * z, x * R)
    return term1 + term2 + term3 + term4


def _g(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    R = jnp.sqrt(x * x + y * y + z * z) + EPS
    term1 = -(1.0 / 3.0) * x * y * R
    term2 = x * y * z * _safe_log(z + R)
    term3 = (1.0 / 6.0) * y * (3.0 * z * z - y * y) * _safe_log(x + R)
    term4 = (1.0 / 6.0) * x * (3.0 * z * z - x * x) * _safe_log(y + R)
    term5 = -(1.0 / 6.0) * (z**3) * _atan2_ratio(x * y, z * R)
    term6 = -(1.0 / 2.0) * (y**2) * z * _atan2_ratio(x * z, y * R)
    term7 = -(1.0 / 2.0) * (x**2) * z * _atan2_ratio(y * z, x * R)
    return term1 + term2 + term3 + term4 + term5 + term6 + term7


def _l_operator(
    phi: Any,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    hx: jnp.ndarray,
    hy: jnp.ndarray,
    hz: jnp.ndarray,
) -> jnp.ndarray:
    acc = jnp.zeros_like(x, dtype=jnp.float64)
    for k in range(27):
        e = _EPS_LIST[k]
        abs_sum = jnp.abs(e[0]) + jnp.abs(e[1]) + jnp.abs(e[2])
        w = 8.0 / ((-2.0) ** abs_sum)
        acc = acc + w * phi(x + e[0] * hx, y + e[1] * hy, z + e[2] * hz)
    return acc / (FOUR_PI * hx * hy * hz)


def demag_N_cellavg_jax(s: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    s_f = jnp.asarray(s, dtype=jnp.float64)
    if s_f.shape[-1] != 3:
        raise ValueError("s must have shape (..., 3)")
    h_f = jnp.asarray(h, dtype=jnp.float64)
    if h_f.shape != (3,):
        raise ValueError("h must have shape (3,)")

    x = s_f[..., 0]
    y = s_f[..., 1]
    z = s_f[..., 2]
    hx, hy, hz = h_f[0], h_f[1], h_f[2]

    Nxx = _l_operator(_f, x, y, z, hx, hy, hz)
    Nxy = _l_operator(_g, x, y, z, hx, hy, hz)
    Nyy = _l_operator(_f, y, x, z, hy, hx, hz)
    Nzz = _l_operator(_f, z, y, x, hz, hy, hx)
    Nxz = _l_operator(_g, x, z, y, hx, hz, hy)
    Nyz = _l_operator(_g, y, z, x, hy, hz, hx)

    row0 = jnp.stack([Nxx, Nxy, Nxz], axis=-1)
    row1 = jnp.stack([Nxy, Nyy, Nyz], axis=-1)
    row2 = jnp.stack([Nxz, Nyz, Nzz], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def coupling_scalar_easy_axis_jax(ui: jnp.ndarray, uj: jnp.ndarray, N: jnp.ndarray) -> jnp.ndarray:
    Nu = jnp.einsum("...ab,...b->...a", N, uj)
    return -jnp.einsum("...a,...a->...", ui, Nu)


__all__ = ["demag_N_cellavg_jax", "coupling_scalar_easy_axis_jax"]
