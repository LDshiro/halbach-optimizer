import numpy as np
import pytest


def _jnp() -> object:
    _ = pytest.importorskip("jax")
    import jax.numpy as jnp

    return jnp


def test_gl_double_delta_tables() -> None:
    _ = _jnp()
    from halbach.autodiff.jax_self_consistent import (
        _gl_double_delta_table_n2,
        _gl_double_delta_table_n3,
    )

    cube_edge = 0.01
    offsets2, w2 = _gl_double_delta_table_n2(cube_edge)
    offsets3, w3 = _gl_double_delta_table_n3(cube_edge)

    assert offsets2.shape == (27, 3)
    assert w2.shape == (27,)
    assert offsets3.shape == (125, 3)
    assert w3.shape == (125,)

    w2_np = np.asarray(w2)
    w3_np = np.asarray(w3)
    np.testing.assert_allclose(w2_np.sum(), 1.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(w3_np.sum(), 1.0, atol=1e-14, rtol=0.0)
    assert np.all(w2_np >= 0.0)
    assert np.all(w3_np >= 0.0)
