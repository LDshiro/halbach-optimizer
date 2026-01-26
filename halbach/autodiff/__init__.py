"""Automatic differentiation backends."""

from halbach.autodiff.jax_objective import objective_with_grads_fixed_jax

__all__ = ["objective_with_grads_fixed_jax"]
