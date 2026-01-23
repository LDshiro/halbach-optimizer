from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from halbach.solvers.types import FunGrad, LBFGSBOptions, SolveResult, SolveTrace
from halbach.types import FloatArray

if TYPE_CHECKING:
    import optype.numpy as onp
    from scipy.optimize._minimize import _MinimizeOptions as MinimizeOptions

    Float1DArray: TypeAlias = onp.Array1D[np.float64]
else:
    Float1DArray: TypeAlias = NDArray[np.float64]
    MinimizeOptions: TypeAlias = dict[str, object]


@dataclass
class _Cache:
    fun_grad: FunGrad
    x_last: FloatArray | None = None
    f_last: float | None = None
    g_last: FloatArray | None = None
    extra_last: dict[str, Any] | None = None

    def eval(self, x: FloatArray) -> tuple[float, FloatArray, dict[str, Any]]:
        if self.x_last is not None and np.array_equal(x, self.x_last):
            # cache hit
            assert (
                self.f_last is not None and self.g_last is not None and self.extra_last is not None
            )
            return self.f_last, self.g_last, self.extra_last

        f, g, extra = self.fun_grad(x)
        # store copies to avoid accidental mutation
        self.x_last = np.array(x, dtype=float, copy=True)
        self.f_last = float(f)
        self.g_last = np.array(g, dtype=float, copy=True)
        self.extra_last = dict(extra)
        return self.f_last, self.g_last, self.extra_last

    def fun(self, x: FloatArray) -> float:
        f, _, _ = self.eval(x)
        return f

    def jac(self, x: FloatArray) -> FloatArray:
        _, g, _ = self.eval(x)
        return g


def _callback_payload(
    cache: _Cache, xk: Float1DArray
) -> tuple[float, Float1DArray, dict[str, Any]]:
    if cache.f_last is None or cache.g_last is None or cache.extra_last is None:
        fk = float("nan")
        gk = cast(Float1DArray, np.zeros_like(xk, dtype=np.float64))
        return fk, gk, {}
    return float(cache.f_last), cast(Float1DArray, cache.g_last), dict(cache.extra_last)


def bounds_from_arrays(lb: FloatArray, ub: FloatArray) -> list[tuple[float, float]]:
    if lb.shape != ub.shape:
        raise ValueError("lb and ub must have the same shape")
    return [(float(a), float(b)) for a, b in zip(lb, ub, strict=True)]


def solve_lbfgsb(
    fun_grad: FunGrad,
    x0: FloatArray,
    bounds: list[tuple[float, float]] | None,
    options: LBFGSBOptions,
    *,
    iter_callback: (
        Callable[[int, FloatArray, float, FloatArray, dict[str, Any]], None] | None
    ) = None,
    record_extras: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> SolveResult:
    cache = _Cache(fun_grad=fun_grad)

    trace = SolveTrace(iters=[], f=[], gnorm=[], extras=[])

    # record iteration 0
    x0_arr = cast(Float1DArray, np.array(x0, dtype=np.float64, copy=True))
    f0, g0, e0 = cache.eval(x0_arr)
    trace.iters.append(0)
    trace.f.append(float(f0))
    trace.gnorm.append(float(np.linalg.norm(g0)))
    trace.extras.append(record_extras(e0) if record_extras is not None else dict(e0))

    def cb(xk: Float1DArray) -> None:
        k = len(trace.iters)  # iteration count (approx)
        xk_f = cast(Float1DArray, np.asarray(xk, dtype=np.float64))
        fk, gk, ek = _callback_payload(cache, xk_f)

        trace.iters.append(k)
        trace.f.append(float(fk))
        trace.gnorm.append(float(np.linalg.norm(gk)))
        trace.extras.append(record_extras(ek) if record_extras is not None else dict(ek))

        if iter_callback is not None:
            iter_callback(k, xk_f, float(fk), gk, dict(ek))

    scipy_options: MinimizeOptions = {
        "maxiter": options.maxiter,
        "gtol": options.gtol,
        "disp": options.disp,
    }
    if options.ftol is not None:
        scipy_options["ftol"] = options.ftol
    if options.maxls is not None:
        scipy_options["maxls"] = options.maxls

    def _fun(x: Float1DArray, *fargs: Any, **fkwargs: Any) -> float:
        return cache.fun(x)

    def _jac(x: Float1DArray, *fargs: Any, **fkwargs: Any) -> Float1DArray:
        return cache.jac(x)

    method: Literal["L-BFGS-B"] = "L-BFGS-B"
    res = minimize(
        fun=_fun,
        x0=x0_arr,
        jac=_jac,
        method=method,
        bounds=bounds,
        callback=cb,
        options=scipy_options,
    )

    njev: int | None = getattr(res, "njev", None)
    return SolveResult(
        x=np.array(res.x, dtype=float, copy=True),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        nit=int(getattr(res, "nit", len(trace.iters) - 1)),
        nfev=int(getattr(res, "nfev", 0)),
        njev=None if njev is None else int(njev),
        trace=trace,
    )
