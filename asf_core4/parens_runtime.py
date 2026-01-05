#!/usr/bin/env python3
"""
Nuttn-But-Parens Runtime

A pure parentheses runtime. Everything is shapes.
No keywords, no operators, no syntax - just () and (()).

This is the minimal computational substrate:
  ()      = Atom (unit, false, 0)
  (())    = Stage (marker, true, 1)
  (...)   = Composite (structure, application, data)

De Bruijn lambda calculus built on this:
  ref(n)  = variable reference (index n)
  lam(b)  = lambda abstraction
  app(f,x) = application
"""

from dataclasses import dataclass
from typing import Union, Tuple, List, Optional, Any
import hashlib


# =============================================================================
# SHAPES: THE ONLY PRIMITIVES
# =============================================================================

@dataclass(frozen=True)
class Atom:
    """The empty parentheses: ()"""
    def __repr__(self): return "()"
    def to_dyck(self): return "()"

@dataclass(frozen=True)
class Stage:
    """The nested parentheses: (())"""
    def __repr__(self): return "(())"
    def to_dyck(self): return "(())"

@dataclass(frozen=True)
class Composite:
    """Nested shapes: (s1 s2 s3 ...)"""
    children: Tuple['Shape', ...]
    def __repr__(self):
        return "(" + "".join(repr(c) for c in self.children) + ")"
    def to_dyck(self):
        return "(" + "".join(c.to_dyck() for c in self.children) + ")"

Shape = Union[Atom, Stage, Composite]

def A() -> Atom:
    """Create an Atom: ()"""
    return Atom()

def S() -> Stage:
    """Create a Stage: (())"""
    return Stage()

def C(*children: Shape) -> Composite:
    """Create a Composite from children."""
    return Composite(tuple(children))


# =============================================================================
# PARSING: DYCK STRING → SHAPE
# =============================================================================

def parse(dyck: str) -> Shape:
    """Parse a Dyck string into a Shape."""
    dyck = dyck.strip()
    if dyck == "()":
        return Atom()
    if dyck == "(())":
        return Stage()
    if not (dyck.startswith("(") and dyck.endswith(")")):
        raise ValueError(f"Invalid Dyck: {dyck}")

    inner = dyck[1:-1]
    children = []
    depth = 0
    start = 0

    for i, c in enumerate(inner):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                children.append(parse(inner[start:i+1]))
                start = i + 1

    if depth != 0:
        raise ValueError(f"Unbalanced: {dyck}")

    return Composite(tuple(children))


def serialize(shape: Shape) -> str:
    """Serialize a Shape to Dyck string."""
    return shape.to_dyck()


# =============================================================================
# CONTENT ADDRESSING: SHAPE → KEY
# =============================================================================

def key(shape: Shape) -> str:
    """Content-addressed key for a shape."""
    dyck = serialize(shape)
    return hashlib.sha256(dyck.encode()).hexdigest()


# =============================================================================
# DE BRUIJN ENCODING
# =============================================================================

# Tags for lambda calculus terms
TAG_VAR = 0  # Variable reference
TAG_LAM = 1  # Lambda abstraction
TAG_APP = 2  # Application

def nat(n: int) -> Shape:
    """Encode a natural number as a shape: n+2 atoms."""
    return Composite(tuple(Atom() for _ in range(n + 2)))

def decode_nat(shape: Shape) -> int:
    """Decode a natural number from a shape."""
    if not isinstance(shape, Composite):
        raise ValueError("Expected Composite for nat")
    return len(shape.children) - 2

def tag(t: int) -> Shape:
    """Create a tag shape."""
    return Composite((Stage(), nat(t)))

def is_tag(shape: Shape, t: int) -> bool:
    """Check if shape is tagged with t."""
    if not isinstance(shape, Composite) or len(shape.children) != 2:
        return False
    if not isinstance(shape.children[0], Stage):
        return False
    if not isinstance(shape.children[1], Composite):
        return False
    return decode_nat(shape.children[1]) == t

def ref(n: int) -> Shape:
    """Variable reference (de Bruijn index n)."""
    return Composite((tag(TAG_VAR), nat(n)))

def lam(body: Shape) -> Shape:
    """Lambda abstraction."""
    return Composite((tag(TAG_LAM), body))

def app(fn: Shape, arg: Shape) -> Shape:
    """Application."""
    return Composite((tag(TAG_APP), fn, arg))


# =============================================================================
# BETA REDUCTION
# =============================================================================

def shift(shape: Shape, d: int, c: int = 0) -> Shape:
    """Shift free variables by d above cutoff c."""
    if isinstance(shape, (Atom, Stage)):
        return shape
    if not isinstance(shape, Composite) or len(shape.children) < 2:
        return shape

    if is_tag(shape.children[0], TAG_VAR):
        n = decode_nat(shape.children[1])
        if n >= c:
            return ref(n + d)
        return shape

    if is_tag(shape.children[0], TAG_LAM):
        return lam(shift(shape.children[1], d, c + 1))

    if is_tag(shape.children[0], TAG_APP):
        return app(shift(shape.children[1], d, c),
                   shift(shape.children[2], d, c))

    # Non-lambda composite: shift children
    return Composite(tuple(shift(ch, d, c) for ch in shape.children))


def subst(shape: Shape, j: int, s: Shape) -> Shape:
    """Substitute s for variable j in shape."""
    if isinstance(shape, (Atom, Stage)):
        return shape
    if not isinstance(shape, Composite) or len(shape.children) < 2:
        return shape

    if is_tag(shape.children[0], TAG_VAR):
        n = decode_nat(shape.children[1])
        if n == j:
            return s
        return shape

    if is_tag(shape.children[0], TAG_LAM):
        return lam(subst(shape.children[1], j + 1, shift(s, 1, 0)))

    if is_tag(shape.children[0], TAG_APP):
        return app(subst(shape.children[1], j, s),
                   subst(shape.children[2], j, s))

    return Composite(tuple(subst(ch, j, s) for ch in shape.children))


def beta_step(shape: Shape) -> Tuple[Shape, bool]:
    """One step of beta reduction. Returns (result, changed)."""
    if isinstance(shape, (Atom, Stage)):
        return shape, False
    if not isinstance(shape, Composite):
        return shape, False

    # Application of lambda: (λ.body) arg → body[0 := arg]
    if is_tag(shape.children[0], TAG_APP) and len(shape.children) >= 3:
        fn = shape.children[1]
        arg = shape.children[2]

        if isinstance(fn, Composite) and is_tag(fn.children[0], TAG_LAM):
            body = fn.children[1]
            result = shift(subst(body, 0, shift(arg, 1, 0)), -1, 0)
            return result, True

        # Reduce function
        new_fn, changed = beta_step(fn)
        if changed:
            return app(new_fn, arg), True

        # Reduce argument
        new_arg, changed = beta_step(arg)
        if changed:
            return app(fn, new_arg), True

    # Lambda body
    if is_tag(shape.children[0], TAG_LAM):
        new_body, changed = beta_step(shape.children[1])
        if changed:
            return lam(new_body), True

    return shape, False


def normalize(shape: Shape, max_steps: int = 1000) -> Shape:
    """Normalize to beta-normal form."""
    for _ in range(max_steps):
        result, changed = beta_step(shape)
        if not changed:
            return result
        shape = result
    return shape  # Didn't normalize in time


# =============================================================================
# CHURCH ENCODINGS
# =============================================================================

# Booleans
TRUE = lam(lam(ref(1)))   # λx.λy.x
FALSE = lam(lam(ref(0)))  # λx.λy.y

# Church numerals
def church(n: int) -> Shape:
    """Church numeral for n."""
    # λf.λx. f^n x
    body = ref(0)  # x
    for _ in range(n):
        body = app(ref(1), body)  # f body
    return lam(lam(body))

ZERO = church(0)
ONE = church(1)
TWO = church(2)

# Successor: λn.λf.λx. f (n f x)
SUCC = lam(lam(lam(app(ref(1), app(app(ref(2), ref(1)), ref(0))))))

# Add: λm.λn.λf.λx. m f (n f x)
ADD = lam(lam(lam(lam(app(app(ref(3), ref(1)), app(app(ref(2), ref(1)), ref(0)))))))


def decode_church(shape: Shape) -> int:
    """Decode a Church numeral."""
    # Apply to successor and zero, then count
    shape = normalize(shape)
    if not isinstance(shape, Composite) or len(shape.children) < 2:
        return 0

    # λf.λx.body - count applications of f
    if is_tag(shape.children[0], TAG_LAM):
        body = shape.children[1]
        if isinstance(body, Composite) and is_tag(body.children[0], TAG_LAM):
            inner = body.children[1]
            count = 0
            while isinstance(inner, Composite) and is_tag(inner.children[0], TAG_APP):
                count += 1
                inner = inner.children[2] if len(inner.children) > 2 else inner
            return count
    return 0


# =============================================================================
# COMBINATORS
# =============================================================================

# I = λx.x
I = lam(ref(0))

# K = λx.λy.x
K = lam(lam(ref(1)))

# S = λx.λy.λz. x z (y z)
S_COMB = lam(lam(lam(app(app(ref(2), ref(0)), app(ref(1), ref(0))))))

# Y combinator (for recursion)
Y = lam(app(
    lam(app(ref(1), app(ref(0), ref(0)))),
    lam(app(ref(1), app(ref(0), ref(0))))
))


# =============================================================================
# RUNTIME (REPL)
# =============================================================================

class ParensRuntime:
    """Interactive runtime for pure parentheses computation."""

    def __init__(self):
        self.env = {
            'I': I,
            'K': K,
            'S': S_COMB,
            'TRUE': TRUE,
            'FALSE': FALSE,
            'ZERO': ZERO,
            'ONE': ONE,
            'TWO': TWO,
            'SUCC': SUCC,
            'ADD': ADD,
        }
        self.history = []

    def eval(self, code: str) -> Shape:
        """Evaluate code."""
        code = code.strip()

        # Named binding: name = expr
        if '=' in code and not code.startswith('('):
            name, expr = code.split('=', 1)
            name = name.strip()
            shape = self.eval(expr.strip())
            self.env[name] = shape
            return shape

        # Variable lookup
        if code in self.env:
            return self.env[code]

        # Lambda: \x.body or λx.body
        if code.startswith('\\') or code.startswith('λ'):
            body = code[2:].strip() if code[1] == '.' else code[1:].strip()
            if body.startswith('.'):
                body = body[1:]
            return lam(self.eval(body))

        # Application: (f x)
        if code.startswith('(') and code.endswith(')'):
            # Check if it's pure Dyck
            if all(c in '()' for c in code):
                return parse(code)

            # Parse as application
            inner = code[1:-1].strip()
            parts = self._split_app(inner)
            if len(parts) == 1:
                return self.eval(parts[0])
            result = self.eval(parts[0])
            for part in parts[1:]:
                result = app(result, self.eval(part))
            return result

        # Church numeral shorthand: #n
        if code.startswith('#'):
            try:
                n = int(code[1:])
                return church(n)
            except ValueError:
                pass

        # Pure Dyck
        if all(c in '()' for c in code):
            return parse(code)

        # Atom/Stage shorthand
        if code == 'A':
            return Atom()
        if code == 'S':
            return Stage()

        raise ValueError(f"Cannot parse: {code}")

    def _split_app(self, s: str) -> List[str]:
        """Split application into parts."""
        parts = []
        depth = 0
        current = ""

        for c in s:
            if c == '(' or c == '[':
                depth += 1
                current += c
            elif c == ')' or c == ']':
                depth -= 1
                current += c
            elif c == ' ' and depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += c

        if current.strip():
            parts.append(current.strip())

        return parts

    def run_repl(self):
        """Interactive REPL."""
        print("\n" + "=" * 60)
        print("  NUTTN-BUT-PARENS RUNTIME")
        print("  Everything is shapes. ()=(()) is all you need.")
        print("=" * 60)
        print("""
  Primitives:
    ()        Atom
    (())      Stage
    (s1 s2)   Composite

  Lambda calculus:
    \\x.body   Lambda (or λx.body)
    (f x)     Application
    I K S     Combinators

  Church:
    #n        Church numeral n
    ZERO ONE TWO  Presets
    SUCC ADD  Operations

  Commands:
    :norm     Normalize to β-normal form
    :church   Decode as Church numeral
    :key      Show content hash
    :quit     Exit
""")

        while True:
            try:
                line = input("\nparens> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue

            if line == ':quit' or line == ':q':
                break

            if line.startswith(':norm '):
                expr = line[6:].strip()
                try:
                    shape = self.eval(expr)
                    result = normalize(shape)
                    print(f"  β-normal: {serialize(result)}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            if line.startswith(':church '):
                expr = line[8:].strip()
                try:
                    shape = self.eval(expr)
                    n = decode_church(shape)
                    print(f"  Church numeral: {n}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            if line.startswith(':key '):
                expr = line[5:].strip()
                try:
                    shape = self.eval(expr)
                    k = key(shape)
                    print(f"  Key: {k}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            try:
                result = self.eval(line)
                self.history.append(result)
                print(f"  = {serialize(result)}")
            except Exception as e:
                print(f"  Error: {e}")

        print("\n  Goodbye from parens-land!")


def main():
    runtime = ParensRuntime()
    runtime.run_repl()


if __name__ == "__main__":
    main()
