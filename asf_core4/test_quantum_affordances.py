#!/usr/bin/env python3
"""
Proof tests for quantum affordances in this system.

These tests verify superposition, interference, phase marking, amplitude
amplification, transforms, and measurement/collapse.
"""

import math
import random

from weighted_state import from_shapes, normalize_l2, prune, apply_operator
from operators import mark, grover_step
from sampling import probabilities, measure, collapse
from transforms import walsh_hadamard


def _assert_close(a: float, b: float, eps: float = 1e-9):
    if abs(a - b) > eps:
        raise AssertionError(f"expected {a} ~= {b}")


def test_interference_cancellation():
    # Two distinct paths map into one target with opposite phase.
    def op(shape):
        if shape == "x":
            return [("t", 1.0)]
        if shape == "y":
            return [("t", -1.0)]
        return [(shape, 1.0)]

    state = {"x": 1.0, "y": 1.0}
    state = apply_operator(state, op)
    state = prune(state)
    assert state == {}, f"expected cancellation, got {state}"


def test_phase_kickback_parity():
    basis = ["00", "01", "10", "11"]
    state = from_shapes(basis)
    parity_is_odd = lambda s: (s.count("1") % 2) == 1

    state = apply_operator(state, mark(parity_is_odd))
    state = walsh_hadamard(state, basis)
    state = normalize_l2(prune(state))

    # Phase information becomes a measurable spike.
    _assert_close(state.get("11", 0.0), 1.0)
    assert all(state.get(b, 0.0) == 0.0 for b in ["00", "01", "10"])


def test_grover_amplification():
    items = [f"s{i}" for i in range(16)]
    target = "s7"
    state = from_shapes(items)

    steps = int(math.floor(math.pi / 4 * math.sqrt(len(items))))
    for _ in range(steps):
        state = grover_step(state, target)
        state = normalize_l2(prune(state))

    probs = probabilities(state)
    target_p = probs.get(target, 0.0)
    assert target_p > 0.6, f"expected amplification, got p={target_p:.3f}"


def test_measurement_collapse():
    state = {"u": 0.6, "v": 0.8}
    state = normalize_l2(prune(state))
    rng = random.Random(7)
    outcome = measure(state, rng)
    collapsed = collapse(state, outcome)
    assert list(collapsed.keys()) == [outcome]
    _assert_close(list(collapsed.values())[0], 1.0)


def run_all():
    tests = [
        test_interference_cancellation,
        test_phase_kickback_parity,
        test_grover_amplification,
        test_measurement_collapse,
    ]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")


if __name__ == "__main__":
    run_all()
