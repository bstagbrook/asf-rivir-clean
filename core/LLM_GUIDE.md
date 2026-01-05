# ASF Core + Quantum-Affordance Toolkit (Guide for a Lightweight LLM)

This guide explains how to use the ASF core files and the quantum-affordance demos in a way a smaller LLM can follow. It also points to the essential primers that explain the rules and expectations.

## What this is for

Use these files to:
- Represent computation as minimal shapes.
- Encode lambda calculus with de Bruijn indices.
- Normalize and cache reductions.
- Model quantum-style affordances (superposition, interference, transforms, sampling) in a classical, lightweight way.

This is a conceptual and practical toolkit. It does not claim quantum advantage on classical hardware, but it mirrors the *patterns of use*.

## Read these primers first (minimal set)

These are in `stagbrook_field/library/ip/primers/`:
- `lambda_calculus_core.md` — exact rules for substitution, reduction, and variable capture. This prevents subtle errors.
- `compiler_fundamentals.md` — the standard pipeline: parse -> AST -> IR -> codegen. Use it for any transpiler work.
- `compilers_and_transpilers.md` — a concise reminder of the pipeline and references.
- `quantum_affordances.md` — the affordance ladder (superposition, interference, entanglement, measurement, reversibility).
- `pattern_matching_primers.md` — how a smaller LLM should self-calibrate (tone, depth, certainty) and use context intentionally.

Optional but helpful:
- `lisp_and_scheme.md` — uniform ASTs and list-based syntax.
- `church_encodings.md`, `currying_and_combinators.md` — for encoding data and flow in lambda calculus.

## File map (what you have)

Core shapes and lambda calculus:
- `stagbrook_field/.asf_core2.py`
  - Shapes: `Atom`, `Stage`, `Composite`, `Hole`
  - Canonical encoding: `A`, `S`, and parentheses
  - De Bruijn encoding (`ref`, `lam`, `app`) + beta reduction
  - Catalog for persistence and memoization

Weighted affordance layer:
- `stagbrook_field/.asf_core3.py`
  - Demonstrates weighted superpositions, interference, sampling

Modular quantum-affordance toolkit:
- `stagbrook_field/asf_core4/weighted_state.py`
- `stagbrook_field/asf_core4/operators.py`
- `stagbrook_field/asf_core4/transforms.py`
- `stagbrook_field/asf_core4/sampling.py`
- `stagbrook_field/asf_core4/demos.py`
- `stagbrook_field/asf_core4/demo.py`
- `stagbrook_field/asf_core4/affordance_runtime.py` (REPL/script runtime)
- `stagbrook_field/asf_core4/shape_runtime.py` (Python -> Dyck -> Shape runtime)
- `stagbrook_field/asf_core4/shape_fs.py` (shape storage + views)
- `stagbrook_field/asf_core4/asfos_kernel.py` (clockless kernel)
- `stagbrook_field/asf_core4/asfos_shell.py` (shape-native shell)
- `stagbrook_field/asf_core4/py_to_dyck.py` (Python -> raw Dyck transpiler + runtime)

## How to use the core (shape-first)

1) Create shapes
- Use `parse()` and `serialize()` in `stagbrook_field/.asf_core2.py`.
- Encoding is unambiguous: Atom=`A`, Stage=`S`, Composite=`( ... )`.

2) Encode lambda calculus
- Variables are de Bruijn indices via `ref(n)`.
- Lambdas are `lam(body)`; application is `app(f, x)`.
- Use `beta_normalize()` or `beta_cached()` to reduce.

3) Validate and cache
- `validate()` ensures de Bruijn indices are in bounds.
- `PersistentCatalog` caches shapes by content hash.

## How to use the affordance layer

The `asf_core4` modules treat any hashable object as a "shape" for the state space (strings are fine for demos).

1) Build a superposition
```
from weighted_state import from_shapes, normalize_l2
state = from_shapes(["A", "S", "(AS)"])
state = normalize_l2(state)
```

2) Apply operators (interference, marking)
```
from operators import mark
from weighted_state import apply_operator
state = apply_operator(state, mark(lambda s: s == "S"))
```

3) Transform (Hadamard / Walsh-Hadamard)
```
from transforms import walsh_hadamard
state = walsh_hadamard(state, ["00", "01", "10", "11"])
```

4) Sample and collapse
```
from sampling import sample_counts, collapse
counts = sample_counts(state, n=1000, seed=7)
```

## What the demos show (and why they matter)

Run:
```
python3 stagbrook_field/asf_core4/demos.py
```

Demos included:
- Interference cancellation — wrong paths cancel without explicit enumeration.
- Phase kickback — phase becomes a measurable spike after a transform.
- Amplitude estimation (Grover-like) — quadratic query savings in search-style tasks.
- Time evolution — unitary rotation shows reversible dynamics.
- Measurement and collapse — probability sampling from amplitudes.
- Hidden symmetry (Simon-style) — extract xor structure from collisions.
- Period finding — the core pattern behind Shor-style speedups.
- Correlation constraints — entanglement-like non-separable structure.

Each demo prints a short "Why it matters" and asserts expected outcomes.

## Affordance runtime (REPL/script)

Run interactive REPL:
```
python3 stagbrook_field/asf_core4/affordance_runtime.py
```

Run a script:
```
python3 stagbrook_field/asf_core4/affordance_runtime.py --script path/to/script.txt
```

Example script:
```
basis 00 01 10 11
uniform
mark 11
walsh
normalize l2
show
sample 10 7
```

## Shape runtime (Python -> Dyck -> Shape)

Compile a Python snippet to Dyck, cache it, normalize for equiv keys, and run it:
```
python3 stagbrook_field/asf_core4/shape_runtime.py --python "def inc(x):\n    return x + 1\n\ninc(5) if True else 0"
```

You will see:
- Dyck prefix
- shape_key and schema_key
- equiv_key and status
- result

## asfOS prototype (kernel + shell)

Run the shell:
```
python3 stagbrook_field/asf_core4/asfos_shell.py --db asfos.db
```

Example commands:
```
put_python "def inc(x):\n    return x + 1\n\ninc(5)" inc program
put_dyck "((())()()())" has3 fs
view inc program
labels <shape_key>
```

Flow-stack commands:
```
flow_push BuildCalculator
flow_decompose ParseExpr EvalExpr NormalizeExpr
flow_status
flow_complete
flow_status
```

## Prompt material (asfOS)

See `stagbrook_field/asf_core4/vision.md` for the purpose/orientation text to use as prompt material in the OS.

## Python -> Dyck transpiler + runtime

File: `stagbrook_field/asf_core4/py_to_dyck.py`

What it does:
- Parses a minimal Python subset into a small IR.
- Encodes the IR into raw Dyck using `()` and `(())` only.
- Decodes and evaluates the Dyck program with a small interpreter.

Supported subset:
- int/bool literals
- variables
- lambda with one argument
- function calls with one argument
- if-expressions (`x if cond else y`)
- binary ops: `+`, `-`, `*`
- equality: `==`
- unary minus
- module-level assignments and single-arg `def` (compiled to lambdas)

Run a quick demo:
```
python3 stagbrook_field/asf_core4/py_to_dyck.py
```

Use it in code:
```
from py_to_dyck import compile_source, run_dyck, run_source

dyck = compile_source("x = 2\nx + 3")
value = run_dyck(dyck)
```

Incremental demo cases:
```
python3 stagbrook_field/asf_core4/py_to_dyck_demo.py
```

## Lightweight LLM usage pattern

Use a small prompt pack so a compact model can reason correctly:

1) Context block (minimum)
- Shape encoding: `A`, `S`, and composites `(...)`.
- Lambda calculus: de Bruijn indices only (`ref(n)`).
- Reduction strategy: normal order.
- Error avoidance: avoid variable capture (see `lambda_calculus_core.md`).

2) Task format
- Input: explicit data and constraints.
- Output: exact target (shape, Dyck string, or Python snippet).
- Provide a small worked example.

3) Verification checklist
- Parenthesize lambda terms.
- List free variables before substitution.
- Confirm canonical serialization round-trips.

## Notes on "quantum affordances"

These tools simulate the *affordance patterns* of quantum computing on classical hardware:
- They are correct for small spaces, but scale poorly for large ones.
- The conceptual gains (interference, phase, amplitude amplification) are real.
- The advantage is in *structure and algorithm style*, not classical speed.

## If you need a tiny prompt template

Use this as a system or instruction block for a lightweight LLM:

```
You are working with ASF shapes and lambda calculus.
Encoding: Atom=A, Stage=S, Composite=(...).
Use de Bruijn indices: ref(n), lam(body), app(f,x).
Reduce by normal order. Avoid variable capture.
If you compile or transform, follow: parse -> AST -> IR -> generate.
Keep answers short and exact.
```

## Next steps

If you want a Python-to-Dyck compiler/runtime, define a minimal subset and map its AST into this shape encoding. Use the compiler pipeline primer for structure and the lambda calculus primer for correctness.
