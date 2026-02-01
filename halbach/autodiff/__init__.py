"""Automatic differentiation backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from halbach.autodiff.jax_objective import (
        objective_with_grads_fixed_jax as objective_with_grads_fixed_jax,
    )
else:
    try:
        from halbach.autodiff.jax_objective import (
            objective_with_grads_fixed_jax as _objective_with_grads_fixed_jax,
        )
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None)
        if missing_name not in ("jax", "jaxlib"):
            raise
        _exc = exc

        def objective_with_grads_fixed_jax(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                "JAX is required for objective_with_grads_fixed_jax. "
                "Install `jax` and `jaxlib` or use the analytic backend."
            ) from _exc

    else:
        objective_with_grads_fixed_jax = _objective_with_grads_fixed_jax

__all__ = ["objective_with_grads_fixed_jax"]
