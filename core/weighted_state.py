"""
Weighted state utilities.

A state is a dict[Shape, complex]. Shapes are any hashable objects.
"""

from typing import Dict, Iterable, Tuple, Hashable
import math

Shape = Hashable
Weight = complex
State = Dict[Shape, Weight]


def add_state(a: State, b: State) -> State:
    """Add two states, combining amplitudes."""
    out: State = dict(a)
    for shape, weight in b.items():
        out[shape] = out.get(shape, 0.0) + weight
    return out


def scale_state(state: State, scalar: Weight) -> State:
    """Scale all amplitudes by a constant."""
    if scalar == 1:
        return dict(state)
    return {s: w * scalar for s, w in state.items()}


def normalize_l1(state: State) -> State:
    """Normalize by L1 norm (sum of magnitudes)."""
    total = sum(abs(w) for w in state.values())
    if total == 0:
        return {}
    return {s: w / total for s, w in state.items() if w != 0.0}


def normalize_l2(state: State) -> State:
    """Normalize by L2 norm (sqrt of sum of squared magnitudes)."""
    total = math.sqrt(sum(abs(w) ** 2 for w in state.values()))
    if total == 0:
        return {}
    return {s: w / total for s, w in state.items() if w != 0.0}


def prune(state: State, eps: float = 1e-12) -> State:
    """Drop near-zero amplitudes to model interference cancellation."""
    return {s: w for s, w in state.items() if abs(w) > eps}


def from_shapes(shapes: Iterable[Shape], weight: Weight = 1.0) -> State:
    """Create a uniform-weight state over shapes."""
    shapes = list(shapes)
    if not shapes:
        return {}
    w = weight / len(shapes)
    return {s: w for s in shapes}


Rewrite = Tuple[Shape, Weight]


def apply_operator(state: State, operator) -> State:
    """Apply a linear operator: operator(shape) -> iterable[(shape, coeff)]."""
    out: State = {}
    for shape, weight in state.items():
        for new_shape, coeff in operator(shape):
            out[new_shape] = out.get(new_shape, 0.0) + weight * coeff
    return out
