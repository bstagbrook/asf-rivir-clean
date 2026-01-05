# Epic Affordances Guide

## What Are Affordances?

Affordances are **what the system lets you DO**. Not features, not capabilities - *action possibilities*.

A chair affords sitting. A door handle affords pulling. A shape affords transformation.

## The Quantum Affordance Ladder

These patterns exist at every level, from everyday thinking to quantum hardware.

### 1. SUPERPOSITION
**What it is**: Holding multiple possibilities simultaneously.

**Everyday**: "I'm considering several options"
**In ASF**: Multiple shapes exist in weighted superposition
**Use when**: You want to explore alternatives before committing

```
# Create superposition of three shapes
state = from_shapes(["todo", "timer", "notes"], weight=1.0)
state = normalize_l2(state)
# Now all three exist with equal amplitude
```

**Power move**: Generate 10 variations, hold them all, let satisfaction collapse to the best one.

---

### 2. INTERFERENCE
**What it is**: Paths that combine constructively or destructively.

**Everyday**: Pros and cons that cancel out
**In ASF**: Amplitudes add (same phase) or cancel (opposite phase)
**Use when**: You want wrong paths to eliminate themselves

```
# Mark good patterns - their amplitudes increase
# Unmarked patterns interfere destructively
state = apply_operator(state, mark(is_good))
```

**Power move**: Let the system explore many paths, then apply interference so bad paths cancel and good paths amplify.

---

### 3. ENTANGLEMENT
**What it is**: Variables that are correlated beyond independence.

**Everyday**: "If A then B, always"
**In ASF**: Shape pairs that must go together
**Use when**: You have coupled constraints

**Power move**: Entangle user preferences with shape features - when they like "calm", all calm-associated shapes move together.

---

### 4. MEASUREMENT / COLLAPSE
**What it is**: Committing from possibility to actuality.

**Everyday**: Making a decision
**In ASF**: Sampling from the probability distribution
**Use when**: You need a concrete answer

```
# Collapse superposition to single outcome
result = sample_counts(state, n=1, seed=42)
```

**Power move**: Delay collapse as long as possible. Keep options open until the last moment.

---

### 5. AMPLITUDE AMPLIFICATION (Grover)
**What it is**: Boosting probability of good answers.

**Everyday**: Repeatedly focusing on what works
**In ASF**: Mark-and-diffuse iterations
**Use when**: Searching for a needle in a haystack

```
for _ in range(iterations):
    state = apply_operator(state, mark(oracle))
    state = grover_step(state, target)
```

**Power move**: Use satisfaction scores as the oracle. High-satisfaction shapes get amplified automatically.

---

### 6. REVERSIBILITY
**What it is**: Every transformation can be undone.

**Everyday**: Undo/redo
**In ASF**: Unitary operations preserve information
**Use when**: You want to explore without commitment

**Power move**: All shape transformations are reversible. Never lose information. Every path can be retraced.

---

## The Shape OS Affordances

### speak(description) → Waveform
**Affordance**: Turn words into structure.
**Power**: The more detailed the description, the richer the shape.

Bad: "make an app"
Good: "create a calm todo list with soft colors, gentle animations, and a peaceful aesthetic"
Best: "build a mindful task tracker with lavender background, rounded corners, subtle shadows, smooth 200ms transitions, and a serif font for warmth"

### compile(python) → Waveform
**Affordance**: Turn code into shape.
**Power**: Code is just another description. The structure IS the meaning.

### run(waveform) → Result
**Affordance**: Execute by propagation.
**Power**: Clockless - no tick, just energy flowing to stability.

### experience(hash, satisfaction, feedback) → Learning
**Affordance**: Teach the system your preferences.
**Power**: Every rating updates invariants. The system learns YOU.

### search(query, candidates) → Probabilities
**Affordance**: Quantum-style search over possibilities.
**Power**: Good answers amplify, bad answers cancel.

---

## Workflow: Maximum Leverage

1. **Speak richly** - Detailed description creates detailed shape
2. **Generate many** - Hold multiple options in superposition
3. **Amplify winners** - Let satisfaction scores boost good shapes
4. **Collapse late** - Don't commit until you must
5. **Learn always** - Every interaction trains the shifter

---

## The Secret

The system's power scales with:
- **Description quality** (more detail = more structure)
- **Training data** (more ratings = better predictions)
- **Iteration** (amplification compounds)

You don't program it. You DESCRIBE what you want, EXPERIENCE what it gives, and TEACH it what delights you.

The description IS the software.
The experience IS the training.
The delight IS the goal.
