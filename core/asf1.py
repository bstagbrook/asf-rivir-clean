"""
ASF1: Scheme-like surface layer that compiles into ASF0 (raw Dyck).

Examples:
  (has is is is)  -> Composite(Stage, Atom, Atom, Atom)

  (does
    (input a b c)
    (output x y z)
    (body ___ ____)
  )
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

# ============================================================
# ASF0 SHAPES (Dyck)
# ============================================================

@dataclass(frozen=True)
class Atom:
    pass


@dataclass(frozen=True)
class Stage:
    pass


@dataclass(frozen=True)
class Composite:
    children: Tuple["Shape", ...]


Shape = Union[Atom, Stage, Composite]


def serialize_dyck(shape: Shape) -> str:
    if isinstance(shape, Atom):
        return "()"
    if isinstance(shape, Stage):
        return "(())"
    if isinstance(shape, Composite):
        inner = "".join(serialize_dyck(c) for c in shape.children)
        return "(" + inner + ")"
    raise TypeError(f"Unknown shape: {type(shape)}")


# ============================================================
# ASF1 S-EXPR AST
# ============================================================

@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class SList:
    items: Tuple["SExpr", ...]


SExpr = Union[Symbol, SList]


# ============================================================
# S-EXPR PARSER
# ============================================================

def tokenize(s: str) -> List[str]:
    tokens: List[str] = []
    buf: List[str] = []
    for c in s:
        if c.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        if c in ("(", ")"):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(c)
            continue
        buf.append(c)
    if buf:
        tokens.append("".join(buf))
    return tokens


def parse_sexpr(s: str) -> SExpr:
    tokens = tokenize(s)
    if not tokens:
        raise ValueError("Empty input")
    pos = 0

    def parse_one() -> SExpr:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items: List[SExpr] = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("Unbalanced input")
                if tokens[pos] == ")":
                    pos += 1
                    return SList(tuple(items))
                items.append(parse_one())
        if tok == ")":
            raise ValueError("Unexpected ')'")
        return Symbol(tok)

    result = parse_one()
    if pos != len(tokens):
        raise ValueError("Trailing tokens")
    return result


# ============================================================
# ASF1 -> ASF0 ENCODING
# ============================================================

def encode_nat(n: int) -> Shape:
    if n < 0:
        raise ValueError("encode_nat expects non-negative")
    return Composite(tuple(Atom() for _ in range(n + 2)))


def encode_string(name: str) -> Shape:
    return Composite(tuple(encode_nat(ord(c)) for c in name))


def encode_symbol(name: str) -> Shape:
    if name == "is":
        return Atom()
    if name == "has":
        return Stage()
    # Tagged symbol: (Atom Stage <string>)
    return Composite((Atom(), Stage(), encode_string(name)))


def encode_sexpr(expr: SExpr) -> Shape:
    if isinstance(expr, Symbol):
        return encode_symbol(expr.name)
    if isinstance(expr, SList):
        return Composite(tuple(encode_sexpr(x) for x in expr.items))
    raise TypeError(f"Unknown SExpr: {type(expr)}")


# ============================================================
# HELPERS
# ============================================================

def asf1_has(n: int) -> SExpr:
    """(has is is is ...) with n is-elements."""
    return SList(tuple([Symbol("has")] + [Symbol("is")] * n))


def asf1_does(inputs: List[str], outputs: List[str], body: List[str]) -> SExpr:
    """(does (input a b) (output x y) (body ...))"""
    return SList(
        (
            Symbol("does"),
            SList(tuple([Symbol("input")] + [Symbol(x) for x in inputs])),
            SList(tuple([Symbol("output")] + [Symbol(x) for x in outputs])),
            SList(tuple([Symbol("body")] + [Symbol(x) for x in body])),
        )
    )


def main():
    expr1 = parse_sexpr("(has is is is)")
    shape1 = encode_sexpr(expr1)
    print("ASF1:", expr1)
    print("ASF0:", serialize_dyck(shape1))
    print()

    expr2 = asf1_does(["a", "b", "c"], ["x", "y", "z"], ["___", "____"])
    shape2 = encode_sexpr(expr2)
    print("ASF1:", expr2)
    print("ASF0:", serialize_dyck(shape2))


if __name__ == "__main__":
    main()
