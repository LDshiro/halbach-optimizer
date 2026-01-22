from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:

    def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...

else:
    from numba import njit as _njit  # type: ignore[import-untyped]

    njit = _njit
