"""
Transform operators (Hadamard and Walsh-Hadamard) over a fixed basis.
"""

from typing import Iterable, List, Dict
import math
try:
    from .weighted_state import State, Shape
except ImportError:
    from weighted_state import State, Shape


def hadamard_pair(a: Shape, b: Shape):
    """
    Return an operator implementing a 2x2 Hadamard on basis [a, b].
    Maps:
      a -> (a + b) / sqrt(2)
      b -> (a - b) / sqrt(2)
    """
    scale = 1.0 / math.sqrt(2.0)

    def op(shape: Shape):
        if shape == a:
            return [(a, scale), (b, scale)]
        if shape == b:
            return [(a, scale), (b, -scale)]
        return [(shape, 1.0)]

    return op


def walsh_hadamard(state: State, basis: Iterable[Shape]) -> State:
    """
    Apply Walsh-Hadamard transform to the subspace spanned by basis.
    The basis length must be a power of two.
    """
    basis_list = list(basis)
    n = len(basis_list)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Basis length must be a power of two")

    vec: List[complex] = [state.get(s, 0.0) for s in basis_list]

    h = 1
    while h < n:
        step = h * 2
        for i in range(0, n, step):
            for j in range(i, i + h):
                x = vec[j]
                y = vec[j + h]
                vec[j] = x + y
                vec[j + h] = x - y
        h = step

    scale = 1.0 / math.sqrt(n)
    out: Dict[Shape, complex] = dict(state)
    for s in basis_list:
        out.pop(s, None)
    for s, amp in zip(basis_list, vec):
        if amp != 0.0:
            out[s] = amp * scale
    return out
