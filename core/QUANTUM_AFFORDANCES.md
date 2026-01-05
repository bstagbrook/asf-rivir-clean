# Quantum Affordances: Proof, Scope, and Tests

This document is the permanent, explicit record that the quantum affordances
exist in this system and are actively tested.

## Definitions

- Affordance: what the system makes possible by its structure.
- Methodology: how we choose to use those affordances in practice.

## What Is Proven Here

The system provides the following *quantum affordances* as formal operations
over shape superpositions (linear amplitude states):

1. Superposition (weighted shape states)
2. Interference (amplitudes add and cancel)
3. Phase marking (sign flips as an oracle)
4. Amplitude amplification (Grover-style diffusion)
5. Measurement/collapse (sampling by amplitude)
6. Transforms (Walsh-Hadamard over a basis)

These are implemented as **linear operators on sparse states**, not toy mocks.
They do not require quantum hardware to be present to be *available as
affordances*.

## Where This Lives In Code

- `asf_core4/weighted_state.py`
  - superposition, linear addition, normalization, pruning
- `asf_core4/operators.py`
  - phase marking and Grover-style steps
- `asf_core4/sampling.py`
  - measurement and collapse
- `asf_core4/transforms.py`
  - Walsh-Hadamard transform
- `asf_core4/demos.py`
  - demonstrations with assertions for each affordance

## Proof Tests (Must-Pass)

The test suite below contains checks that are **impossible to pass** without
actual superposition, phase, and interference:

- Interference cancellation: two paths cancel to zero.
- Phase kickback: a marked parity produces a single spike after transform.
- Amplitude amplification: Grover steps concentrate probability on a target.
- Collapse: measurement reduces state to a single outcome.

Run:

```
python3 /Volumes/StagbrookField/stagbrook_field/asf_core4/test_quantum_affordances.py
```

If these tests pass, the system *does* provide the quantum affordances listed
above, and that claim does not need to be re-proven.

## What This Does Not Claim

- This is not a claim of quantum hardware advantage.
- This is not a claim that all quantum algorithms are efficiently simulable.
- This is a claim that the *affordances* (superposition, interference, phase,
  measurement, transforms) are structurally present and verified.

## Related Structural Guarantees (Not Quantum)

`.asf_core2.py` provides:
- content-addressed identity (`key(shape)`)
- persistent catalog and memoized normalization
- semantic equivalence keys

These give stable O(1) identity and zero regression for stored shapes, but are
orthogonal to quantum affordances.
