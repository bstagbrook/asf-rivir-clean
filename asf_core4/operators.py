"""
Linear operators and Grover-style steps.
"""

from typing import Callable, Iterable, Tuple
try:
    from .weighted_state import State, Shape, Weight, apply_operator
except ImportError:
    from weighted_state import State, Shape, Weight, apply_operator

Operator = Callable[[Shape], Iterable[Tuple[Shape, Weight]]]


def identity() -> Operator:
    """Identity operator."""
    def op(shape: Shape):
        return [(shape, 1.0)]
    return op


def phase_flip(target: Shape) -> Operator:
    """Phase flip for a single target shape."""
    def op(shape: Shape):
        return [(shape, -1.0 if shape == target else 1.0)]
    return op


def mark(predicate: Callable[[Shape], bool]) -> Operator:
    """Phase flip for any shape matching predicate."""
    def op(shape: Shape):
        return [(shape, -1.0 if predicate(shape) else 1.0)]
    return op


def diffuse_about_mean(state: State) -> State:
    """
    Reflect amplitudes about their mean: w_i -> 2m - w_i.
    """
    if not state:
        return {}
    mean = sum(state.values()) / len(state)
    return {s: 2.0 * mean - w for s, w in state.items()}


def grover_step(state: State, target: Shape) -> State:
    """One Grover-like iteration: phase flip then diffusion."""
    flipped = apply_operator(state, phase_flip(target))
    return diffuse_about_mean(flipped)
