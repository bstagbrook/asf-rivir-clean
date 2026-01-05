"""
Quantum-adjacent demos for ASF Core 4.
"""

import math
import random

try:
    from .weighted_state import from_shapes, normalize_l2, prune, apply_operator
    from .operators import mark, grover_step
    from .sampling import sample_counts, probabilities, measure, collapse
    from .transforms import walsh_hadamard
except ImportError:
    from weighted_state import from_shapes, normalize_l2, prune, apply_operator
    from operators import mark, grover_step
    from sampling import sample_counts, probabilities, measure, collapse
    from transforms import walsh_hadamard


def header(title: str):
    print(title)
    print("-" * len(title))


def demo_interference_cancellation():
    """
    Interference toy:
      - two distinct paths map into one target with opposite phase
      - the amplitude cancels out
    """
    header("Interference cancellation")
    print("Why it matters: interference can erase wrong paths without checking them all.")
    print("Quantum affordance: cancellation scales with superposition size; classical simulation")
    print("is possible but expensive for large state spaces.\n")

    def two_path_cancel(shape: str):
        if shape == "x":
            return [("t", 1.0)]
        if shape == "y":
            return [("t", -1.0)]
        return [(shape, 1.0)]

    state = {"x": 1.0, "y": 1.0}
    state = apply_operator(state, two_path_cancel)
    state = prune(state)
    print(f"Resulting state: {state}")
    assert state == {}
    print("OK: cancellation leaves no amplitude.\n")


def demo_phase_kickback():
    """
    Phase kickback toy:
      - start with uniform superposition over 2-bit basis
      - phase-flip any basis state with odd parity
      - Walsh-Hadamard to reveal parity structure
    """
    basis = ["00", "01", "10", "11"]
    state = from_shapes(basis)
    parity_is_odd = lambda s: (s.count("1") % 2) == 1

    state = apply_operator(state, mark(parity_is_odd))
    state = walsh_hadamard(state, basis)
    state = normalize_l2(prune(state))

    header("Phase kickback (parity)")
    print("Why it matters: phase information becomes measurable structure after transforms.")
    print("Quantum affordance: phase is not observable directly, but transforms reveal it.\n")

    print("Amplitudes after transform:")
    for b in basis:
        print(f"  {b}: {state.get(b, 0.0)}")
    assert state.get("11", 0.0) == 1.0
    assert all(state.get(b, 0.0) == 0.0 for b in ["00", "01", "10"])
    print("OK: phase becomes a measurable spike.\n")


def demo_amplitude_estimation():
    """
    Toy amplitude estimation:
      - prepare a superposition
      - mark target
      - apply Grover steps
      - estimate probability by sampling
    """
    items = [f"s{i}" for i in range(8)]
    target = "s5"
    state = from_shapes(items)

    steps = int(math.floor(math.pi / 4 * math.sqrt(len(items))))
    for _ in range(steps):
        state = grover_step(state, target)
        state = normalize_l2(prune(state))

    counts = sample_counts(state, n=2000, seed=11)
    est = counts.get(target, 0) / 2000.0
    true_p = probabilities(state).get(target, 0.0)

    header("Amplitude estimation (Grover-like)")
    print("Why it matters: estimate marked probability with fewer queries than naive search.")
    print("Quantum affordance: quadratic speedup in the number of oracle calls.\n")
    print(f"Target: {target}")
    print(f"Estimated p: {est:.3f} (sampled)")
    print(f"Actual p:    {true_p:.3f} (from amplitudes)")
    assert true_p > 0.9
    print("OK: amplitude is strongly concentrated.\n")


def demo_time_evolution():
    """
    Hamiltonian-like evolution on a two-state basis.
    """
    header("Time evolution (2-state rotation)")
    print("Why it matters: unitary evolution models physical dynamics.")
    print("Quantum affordance: natural for quantum systems; classical scales poorly.\n")

    a, b = "0", "1"
    theta = 0.3

    def step(state):
        ca = math.cos(theta)
        sa = math.sin(theta)
        new_a = ca * state.get(a, 0.0) + sa * state.get(b, 0.0)
        new_b = -sa * state.get(a, 0.0) + ca * state.get(b, 0.0)
        return {a: new_a, b: new_b}

    state = {a: 1.0}
    for i in range(5):
        state = step(state)
        state = normalize_l2(prune(state))
        print(f"Step {i + 1}: {a}={state.get(a, 0.0):.3f}, {b}={state.get(b, 0.0):.3f}")
    norm = math.sqrt(sum(abs(w) ** 2 for w in state.values()))
    assert abs(norm - 1.0) < 1e-9
    print("OK: unitary evolution preserves norm.\n")


def demo_measurement_collapse():
    """
    Measurement toy:
      - sample a state
      - collapse to a single outcome
    """
    header("Measurement and collapse")
    print("Why it matters: measurement converts amplitudes into classical outcomes.")
    print("Quantum affordance: probabilities derive from amplitudes, not explicit enumeration.\n")

    state = {"u": 0.6, "v": 0.8}  # not normalized, L2 will be 1.0
    state = normalize_l2(prune(state))
    rng = random.Random(7)
    outcome = measure(state, rng)
    collapsed = collapse(state, outcome)
    print(f"Sampled outcome: {outcome}")
    print(f"Collapsed state: {collapsed}")
    assert list(collapsed.keys()) == [outcome]
    print("OK: measurement collapses to one outcome.\n")


def demo_hidden_symmetry():
    """
    Toy hidden symmetry:
      - build a function f with a hidden xor-mask s
      - sample pairs that collide under f
      - derive the xor structure (classical mirror of Simon-style constraints)
    """
    rng = random.Random(3)
    mask = "10"  # hidden xor mask
    domain = ["00", "01", "10", "11"]

    def xor(a: str, b: str) -> str:
        return "".join("1" if x != y else "0" for x, y in zip(a, b))

    def f(x: str) -> str:
        # f(x) = f(x xor mask) by construction
        partner = xor(x, mask)
        return "pair:" + "".join(sorted([x, partner]))

    collisions = []
    for _ in range(4):
        x = rng.choice(domain)
        y = xor(x, mask)
        collisions.append((x, y))

    header("Hidden symmetry (Simon-style toy)")
    print("Why it matters: uncover a hidden xor-mask from collisions.")
    print("Quantum affordance: exponential query reduction in the full problem.\n")
    print(f"  hidden mask: {mask}")
    print(f"  sample collisions: {collisions}")
    assert all(xor(a, b) == mask for a, b in collisions)
    print("OK: collisions expose the xor structure.\n")


def demo_period_finding():
    """
    Period finding toy:
      - define f(x) with a hidden period r
      - sample f on a small domain and infer the period
    """
    header("Period finding (Fourier intuition)")
    print("Why it matters: period finding is the core of Shor's speedup.")
    print("Quantum affordance: QFT extracts periods with exponentially fewer queries.\n")

    period = 3
    domain = list(range(12))

    def f(x: int) -> int:
        return (2 * x + 1) % period

    samples = [(x, f(x)) for x in domain]
    buckets = {}
    for x, y in samples:
        buckets.setdefault(y, []).append(x)

    diffs = []
    for xs in buckets.values():
        xs.sort()
        for i in range(len(xs) - 1):
            diffs.append(xs[i + 1] - xs[i])

    inferred = None
    if diffs:
        g = diffs[0]
        for d in diffs[1:]:
            g = math.gcd(g, d)
        inferred = g
    print(f"  true period: {period}")
    print(f"  inferred:    {inferred}")
    assert inferred == period
    print("OK: repeated values reveal the period.\n")


def demo_entanglement_constraints():
    """
    Entanglement-like constraints:
      - two bits are constrained to be equal
      - only correlated outcomes remain
    """
    header("Correlation constraints (entanglement-like)")
    print("Why it matters: correlations are stronger than independent sampling.")
    print("Quantum affordance: entanglement encodes non-separable structure.\n")

    basis = ["00", "01", "10", "11"]
    state = from_shapes(basis)

    def equal_bits(s: str) -> bool:
        return s[0] == s[1]

    def project_equal(shape: str):
        if equal_bits(shape):
            return [(shape, 1.0)]
        return []

    state = apply_operator(state, project_equal)
    state = normalize_l2(prune(state))

    print("Amplitudes after correlation filter:")
    for b in basis:
        print(f"  {b}: {state.get(b, 0.0)}")
    assert state.get("01", 0.0) == 0.0
    assert state.get("10", 0.0) == 0.0
    print("OK: only correlated outcomes remain.\n")


def main():
    demo_interference_cancellation()
    demo_phase_kickback()
    demo_amplitude_estimation()
    demo_time_evolution()
    demo_measurement_collapse()
    demo_hidden_symmetry()
    demo_period_finding()
    demo_entanglement_constraints()


if __name__ == "__main__":
    main()
