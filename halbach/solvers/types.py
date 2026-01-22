from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from halbach.types import FloatArray


class FunGrad(Protocol):
    """Return (f, grad, extras) for a given x."""

    def __call__(self, x: FloatArray) -> tuple[float, FloatArray, dict[str, Any]]: ...


@dataclass(frozen=True)
class LBFGSBOptions:
    maxiter: int = 500
    gtol: float = 1e-12
    ftol: float | None = None
    maxls: int | None = None
    disp: bool = False


@dataclass
class SolveTrace:
    iters: list[int]
    f: list[float]
    gnorm: list[float]
    extras: list[dict[str, Any]]


@dataclass
class SolveResult:
    x: FloatArray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int | None
    trace: SolveTrace
