"""
Sampling and measurement utilities.
"""

from typing import Dict, Hashable, Optional
import random

Shape = Hashable
State = Dict[Shape, complex]


def probabilities(state: State) -> Dict[Shape, float]:
    """Return |amp|^2 probabilities (not normalized if state isn't)."""
    return {s: float(abs(w) ** 2) for s, w in state.items()}


def measure(state: State, rng: random.Random) -> Optional[Shape]:
    """Sample a shape proportional to |amp|^2."""
    if not state:
        return None
    items = list(state.items())
    weights = [abs(w) ** 2 for _, w in items]
    total = sum(weights)
    if total <= 0.0:
        return None
    r = rng.random() * total
    acc = 0.0
    for (shape, _), w in zip(items, weights):
        acc += w
        if r <= acc:
            return shape
    return items[-1][0]


def collapse(state: State, outcome: Shape) -> State:
    """Collapse the state onto a single outcome with amplitude 1."""
    if outcome not in state:
        return {}
    return {outcome: 1.0}


def sample_counts(state: State, n: int, seed: int = 0) -> Dict[Shape, int]:
    """Sample n times and return a histogram of outcomes."""
    rng = random.Random(seed)
    counts: Dict[Shape, int] = {}
    for _ in range(n):
        outcome = measure(state, rng)
        if outcome is not None:
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts
