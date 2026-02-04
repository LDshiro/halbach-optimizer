from __future__ import annotations

import itertools

import numpy as np
import pytest


def _jnp() -> object:
    _ = pytest.importorskip("jax")
    import jax.numpy as jnp

    return jnp


def _gl_nodes_weights_n2(a: float) -> tuple[np.ndarray, np.ndarray]:
    x = a / (2.0 * np.sqrt(3.0))
    nodes = np.array([-x, x], dtype=np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)
    return nodes, weights


def _gl_nodes_weights_n3(a: float) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * a * np.sqrt(3.0 / 5.0)
    nodes = np.array([-x, 0.0, x], dtype=np.float64)
    weights = np.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0], dtype=np.float64)
    return nodes, weights


def _pair_sum_gl_double(
    s_base: np.ndarray,
    ui: np.ndarray,
    uj: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    pts = np.array(list(itertools.product(nodes, nodes, nodes)), dtype=np.float64)
    w3 = np.array(
        [
            weights[i] * weights[j] * weights[k]
            for i, j, k in itertools.product(range(len(weights)), repeat=3)
        ],
        dtype=np.float64,
    )

    acc = 0.0
    for pt_t, wt in zip(pts, w3, strict=True):
        for pt_s, ws in zip(pts, w3, strict=True):
            r = s_base + (pt_t - pt_s)
            r2 = float(np.dot(r, r))
            r2 = max(r2, 1e-30)
            rmag = np.sqrt(r2)
            invr3 = 1.0 / (rmag * r2)
            invr5 = invr3 / r2
            ui_dot_r = ui[0] * r[0] + ui[1] * r[1]
            uj_dot_r = uj[0] * r[0] + uj[1] * r[1]
            ui_dot_uj = ui[0] * uj[0] + ui[1] * uj[1]
            term = 3.0 * ui_dot_r * uj_dot_r * invr5 - ui_dot_uj * invr3
            acc += wt * ws * term
    return acc


@pytest.mark.parametrize("order", [2, 3])
def test_gl_double_delta_equivalence_to_pair_sum(order: int) -> None:
    jnp = _jnp()
    from halbach.autodiff import jax_self_consistent as jsc

    cube_edge = 0.01
    s_base = np.array([0.0, 0.0, 0.02], dtype=np.float64)
    phi_i = 0.7
    phi_j = 1.1
    ui = np.array([np.cos(phi_i), np.sin(phi_i), 0.0], dtype=np.float64)
    uj = np.array([np.cos(phi_j), np.sin(phi_j), 0.0], dtype=np.float64)

    if order == 2:
        nodes, weights = _gl_nodes_weights_n2(cube_edge)
        delta_offsets, delta_w = jsc._gl_double_delta_table_n2(cube_edge)
    else:
        nodes, weights = _gl_nodes_weights_n3(cube_edge)
        delta_offsets, delta_w = jsc._gl_double_delta_table_n3(cube_edge)

    acc_pair = _pair_sum_gl_double(s_base, ui, uj, nodes, weights)
    k_pair = float(jsc.H_FACTOR) * acc_pair

    phi_flat = jnp.asarray([phi_i, phi_j], dtype=jnp.float64)
    r0_flat = jnp.asarray([s_base, np.zeros(3)], dtype=jnp.float64)
    i_edge = jnp.asarray([0], dtype=jnp.int32)
    j_edge = jnp.asarray([1], dtype=jnp.int32)
    k_delta = jsc._k_edge_gl_double_from_table(
        phi_flat, r0_flat, i_edge, j_edge, delta_offsets, delta_w
    )

    np.testing.assert_allclose(np.asarray(k_delta)[0], k_pair, rtol=1e-14, atol=1e-14)
