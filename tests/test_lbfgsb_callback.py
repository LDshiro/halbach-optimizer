from typing import Any

import numpy as np

from halbach.solvers import lbfgsb
from halbach.types import FloatArray


def test_callback_payload_does_not_call_fun_grad() -> None:
    calls: dict[str, int] = {"count": 0}

    def fun_grad(x: FloatArray) -> tuple[float, FloatArray, dict[str, Any]]:
        calls["count"] += 1
        return 0.0, np.zeros_like(x, dtype=np.float64), {}

    cache = lbfgsb._Cache(fun_grad=fun_grad)
    x = np.array([1.0, 2.0], dtype=np.float64)

    cache.eval(x)
    assert calls["count"] == 1

    _ = lbfgsb._callback_payload(cache, x)
    assert calls["count"] == 1
