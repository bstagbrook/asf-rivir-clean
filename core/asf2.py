"""
ASF2 Phase 1: Scheme-like surface with Python parity (expression subset).

Supported expression subset:
  - int/bool literals
  - names
  - lambda (single arg)
  - call (single arg)
  - if-expr
  - binop: +, -, *
  - compare: ==
  - unary negation

ASF2 S-expr examples:
  (if (== 1 1) 2 0)
  (lambda (x) (+ x 1))
  (call f 3)
  (+ 1 (* 2 3))
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import ast

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
# ASF2 S-EXPR AST
# ============================================================

@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class SList:
    items: Tuple["SExpr", ...]


SExpr = Union[Symbol, SList]


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


def sexpr_to_string(expr: SExpr) -> str:
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, SList):
        inner = " ".join(sexpr_to_string(x) for x in expr.items)
        return "(" + inner + ")"
    raise TypeError(f"Unknown SExpr: {type(expr)}")


# ============================================================
# ASF2 CORE AST
# ============================================================

@dataclass(frozen=True)
class IntLit:
    value: int


@dataclass(frozen=True)
class BoolLit:
    value: bool


@dataclass(frozen=True)
class Name:
    ident: str


@dataclass(frozen=True)
class Lambda:
    param: str
    body: "Expr"


@dataclass(frozen=True)
class Call:
    fn: "Expr"
    arg: "Expr"


@dataclass(frozen=True)
class IfExpr:
    cond: "Expr"
    then: "Expr"
    other: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Cmp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Neg:
    expr: "Expr"


Expr = Union[IntLit, BoolLit, Name, Lambda, Call, IfExpr, BinOp, Cmp, Neg]


def sexpr_to_ast(expr: SExpr) -> Expr:
    if isinstance(expr, Symbol):
        if expr.name == "true":
            return BoolLit(True)
        if expr.name == "false":
            return BoolLit(False)
        if expr.name.isdigit() or (expr.name.startswith("-") and expr.name[1:].isdigit()):
            return IntLit(int(expr.name))
        return Name(expr.name)

    if isinstance(expr, SList):
        if not expr.items:
            raise ValueError("Empty list is not a valid expression")
        head = expr.items[0]
        tail = expr.items[1:]

        if isinstance(head, Symbol):
            if head.name == "if":
                if len(tail) != 3:
                    raise ValueError("if expects 3 arguments")
                return IfExpr(sexpr_to_ast(tail[0]), sexpr_to_ast(tail[1]), sexpr_to_ast(tail[2]))
            if head.name == "lambda":
                if len(tail) != 2 or not isinstance(tail[0], SList):
                    raise ValueError("lambda expects (lambda (x) body)")
                params = tail[0].items
                if len(params) != 1 or not isinstance(params[0], Symbol):
                    raise ValueError("lambda expects a single symbol parameter")
                return Lambda(params[0].name, sexpr_to_ast(tail[1]))
            if head.name == "call":
                if len(tail) != 2:
                    raise ValueError("call expects 2 arguments")
                return Call(sexpr_to_ast(tail[0]), sexpr_to_ast(tail[1]))
            if head.name in ("+", "-", "*"):
                if len(tail) == 1 and head.name == "-":
                    return Neg(sexpr_to_ast(tail[0]))
                if len(tail) != 2:
                    raise ValueError("binop expects 2 arguments")
                return BinOp(head.name, sexpr_to_ast(tail[0]), sexpr_to_ast(tail[1]))
            if head.name == "==":
                if len(tail) != 2:
                    raise ValueError("== expects 2 arguments")
                return Cmp("==", sexpr_to_ast(tail[0]), sexpr_to_ast(tail[1]))

        if len(expr.items) != 2:
            raise ValueError("Only single-arg application supported")
        return Call(sexpr_to_ast(expr.items[0]), sexpr_to_ast(expr.items[1]))

    raise TypeError(f"Unknown SExpr: {type(expr)}")


def ast_to_sexpr(expr: Expr) -> SExpr:
    if isinstance(expr, IntLit):
        return Symbol(str(expr.value))
    if isinstance(expr, BoolLit):
        return Symbol("true" if expr.value else "false")
    if isinstance(expr, Name):
        return Symbol(expr.ident)
    if isinstance(expr, Lambda):
        return SList((Symbol("lambda"), SList((Symbol(expr.param),)), ast_to_sexpr(expr.body)))
    if isinstance(expr, Call):
        return SList((Symbol("call"), ast_to_sexpr(expr.fn), ast_to_sexpr(expr.arg)))
    if isinstance(expr, IfExpr):
        return SList((Symbol("if"), ast_to_sexpr(expr.cond), ast_to_sexpr(expr.then), ast_to_sexpr(expr.other)))
    if isinstance(expr, BinOp):
        return SList((Symbol(expr.op), ast_to_sexpr(expr.left), ast_to_sexpr(expr.right)))
    if isinstance(expr, Cmp):
        return SList((Symbol(expr.op), ast_to_sexpr(expr.left), ast_to_sexpr(expr.right)))
    if isinstance(expr, Neg):
        return SList((Symbol("-"), ast_to_sexpr(expr.expr)))
    raise TypeError(f"Unknown Expr: {type(expr)}")


# ============================================================
# ASF2 <-> ASF0 ENCODING
# ============================================================

TAG_INT = 0
TAG_BOOL = 1
TAG_NAME = 2
TAG_LAM = 3
TAG_CALL = 4
TAG_IF = 5
TAG_BINOP = 6
TAG_CMP = 7
TAG_NEG = 8

OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_EQ = 3


def encode_nat(n: int) -> Shape:
    if n < 0:
        raise ValueError("encode_nat expects non-negative")
    return Composite(tuple(Atom() for _ in range(n + 2)))


def decode_nat(shape: Shape) -> int:
    if not isinstance(shape, Composite):
        raise ValueError("decode_nat expects Composite")
    if len(shape.children) < 2:
        raise ValueError("decode_nat expects >=2 children")
    if not all(isinstance(c, Atom) for c in shape.children):
        raise ValueError("decode_nat expects only Atoms")
    return len(shape.children) - 2


def encode_string(name: str) -> Shape:
    return Composite(tuple(encode_nat(ord(c)) for c in name))


def decode_string(shape: Shape) -> str:
    if not isinstance(shape, Composite):
        raise ValueError("decode_string expects Composite")
    chars: List[str] = []
    for child in shape.children:
        chars.append(chr(decode_nat(child)))
    return "".join(chars)


def tag_shape(tag_id: int) -> Shape:
    return Composite((Stage(), encode_nat(tag_id)))


def tag_id(shape: Shape) -> int:
    if not isinstance(shape, Composite) or len(shape.children) != 2:
        raise ValueError("Invalid tag shape")
    if not isinstance(shape.children[0], Stage):
        raise ValueError("Invalid tag shape")
    return decode_nat(shape.children[1])


def encode_op(op: str) -> Shape:
    if op == "+":
        return encode_nat(OP_ADD)
    if op == "-":
        return encode_nat(OP_SUB)
    if op == "*":
        return encode_nat(OP_MUL)
    if op == "==":
        return encode_nat(OP_EQ)
    raise ValueError(f"Unknown op: {op}")


def decode_op(shape: Shape) -> str:
    op_id = decode_nat(shape)
    if op_id == OP_ADD:
        return "+"
    if op_id == OP_SUB:
        return "-"
    if op_id == OP_MUL:
        return "*"
    if op_id == OP_EQ:
        return "=="
    raise ValueError(f"Unknown op id: {op_id}")


def encode_ast(expr: Expr) -> Shape:
    if isinstance(expr, IntLit):
        return Composite((tag_shape(TAG_INT), encode_nat(expr.value)))
    if isinstance(expr, BoolLit):
        return Composite((tag_shape(TAG_BOOL), encode_nat(1 if expr.value else 0)))
    if isinstance(expr, Name):
        return Composite((tag_shape(TAG_NAME), encode_string(expr.ident)))
    if isinstance(expr, Lambda):
        return Composite((tag_shape(TAG_LAM), encode_string(expr.param), encode_ast(expr.body)))
    if isinstance(expr, Call):
        return Composite((tag_shape(TAG_CALL), encode_ast(expr.fn), encode_ast(expr.arg)))
    if isinstance(expr, IfExpr):
        return Composite(
            (tag_shape(TAG_IF), encode_ast(expr.cond), encode_ast(expr.then), encode_ast(expr.other))
        )
    if isinstance(expr, BinOp):
        return Composite((tag_shape(TAG_BINOP), encode_op(expr.op), encode_ast(expr.left), encode_ast(expr.right)))
    if isinstance(expr, Cmp):
        return Composite((tag_shape(TAG_CMP), encode_op(expr.op), encode_ast(expr.left), encode_ast(expr.right)))
    if isinstance(expr, Neg):
        return Composite((tag_shape(TAG_NEG), encode_ast(expr.expr)))
    raise TypeError(f"Unknown Expr: {type(expr)}")


def decode_ast(shape: Shape) -> Expr:
    if not isinstance(shape, Composite) or len(shape.children) < 2:
        raise ValueError("Invalid encoded AST")
    t = tag_id(shape.children[0])
    args = shape.children[1:]

    if t == TAG_INT:
        return IntLit(decode_nat(args[0]))
    if t == TAG_BOOL:
        return BoolLit(decode_nat(args[0]) == 1)
    if t == TAG_NAME:
        return Name(decode_string(args[0]))
    if t == TAG_LAM:
        return Lambda(decode_string(args[0]), decode_ast(args[1]))
    if t == TAG_CALL:
        return Call(decode_ast(args[0]), decode_ast(args[1]))
    if t == TAG_IF:
        return IfExpr(decode_ast(args[0]), decode_ast(args[1]), decode_ast(args[2]))
    if t == TAG_BINOP:
        return BinOp(decode_op(args[0]), decode_ast(args[1]), decode_ast(args[2]))
    if t == TAG_CMP:
        return Cmp(decode_op(args[0]), decode_ast(args[1]), decode_ast(args[2]))
    if t == TAG_NEG:
        return Neg(decode_ast(args[0]))
    raise ValueError(f"Unknown tag id: {t}")


# ============================================================
# PYTHON <-> ASF2 AST
# ============================================================

def python_expr_to_ast(src: str) -> Expr:
    node = ast.parse(src, mode="eval").body
    return python_ast_to_asf2(node)


def python_ast_to_asf2(node: ast.AST) -> Expr:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return BoolLit(node.value)
        if isinstance(node.value, int):
            return IntLit(node.value)
        raise ValueError("Only int/bool literals are supported")

    if isinstance(node, ast.Name):
        return Name(node.id)

    if isinstance(node, ast.Lambda):
        if len(node.args.args) != 1:
            raise ValueError("Only single-arg lambdas supported")
        param = node.args.args[0].arg
        return Lambda(param, python_ast_to_asf2(node.body))

    if isinstance(node, ast.Call):
        if len(node.args) != 1:
            raise ValueError("Only single-arg calls supported")
        return Call(python_ast_to_asf2(node.func), python_ast_to_asf2(node.args[0]))

    if isinstance(node, ast.IfExp):
        return IfExpr(
            python_ast_to_asf2(node.test),
            python_ast_to_asf2(node.body),
            python_ast_to_asf2(node.orelse),
        )

    if isinstance(node, ast.BinOp):
        left = python_ast_to_asf2(node.left)
        right = python_ast_to_asf2(node.right)
        if isinstance(node.op, ast.Add):
            return BinOp("+", left, right)
        if isinstance(node.op, ast.Sub):
            return BinOp("-", left, right)
        if isinstance(node.op, ast.Mult):
            return BinOp("*", left, right)
        raise ValueError("Only +, -, * supported")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single comparisons supported")
        left = python_ast_to_asf2(node.left)
        right = python_ast_to_asf2(node.comparators[0])
        if isinstance(node.ops[0], ast.Eq):
            return Cmp("==", left, right)
        raise ValueError("Only == supported")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Neg(python_ast_to_asf2(node.operand))

    raise ValueError(f"Unsupported Python AST: {type(node).__name__}")


def _needs_parens_in_call(expr: Expr) -> bool:
    return isinstance(expr, (Lambda, IfExpr, BinOp, Cmp, Neg))


def asf2_ast_to_python_src(expr: Expr) -> str:
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, BoolLit):
        return "True" if expr.value else "False"
    if isinstance(expr, Name):
        return expr.ident
    if isinstance(expr, Lambda):
        return "lambda " + expr.param + ": " + asf2_ast_to_python_src(expr.body)
    if isinstance(expr, Call):
        fn_src = asf2_ast_to_python_src(expr.fn)
        if _needs_parens_in_call(expr.fn):
            fn_src = "(" + fn_src + ")"
        return fn_src + "(" + asf2_ast_to_python_src(expr.arg) + ")"
    if isinstance(expr, IfExpr):
        return (
            asf2_ast_to_python_src(expr.then)
            + " if "
            + asf2_ast_to_python_src(expr.cond)
            + " else "
            + asf2_ast_to_python_src(expr.other)
        )
    if isinstance(expr, BinOp):
        return "(" + asf2_ast_to_python_src(expr.left) + " " + expr.op + " " + asf2_ast_to_python_src(expr.right) + ")"
    if isinstance(expr, Cmp):
        return "(" + asf2_ast_to_python_src(expr.left) + " " + expr.op + " " + asf2_ast_to_python_src(expr.right) + ")"
    if isinstance(expr, Neg):
        return "(-" + asf2_ast_to_python_src(expr.expr) + ")"
    raise TypeError(f"Unknown Expr: {type(expr)}")


# ============================================================
# PIPELINE HELPERS
# ============================================================

def asf2_from_sexpr(source: str) -> Expr:
    return sexpr_to_ast(parse_sexpr(source))


def asf2_to_sexpr(expr: Expr) -> str:
    return sexpr_to_string(ast_to_sexpr(expr))


def asf2_to_dyck(expr: Expr) -> str:
    return serialize_dyck(encode_ast(expr))


def asf2_from_dyck(dyck: str) -> Expr:
    return decode_ast(parse_dyck(dyck))

def explain_pipeline(python_expr: str) -> dict:
    """Return a dict with each desugaring/resugaring step for a Python expression."""
    asf2_ast = python_expr_to_ast(python_expr)
    sexpr = asf2_to_sexpr(asf2_ast)
    dyck = asf2_to_dyck(asf2_ast)
    asf2_back = asf2_from_dyck(dyck)
    python_back = asf2_ast_to_python_src(asf2_back)
    return {
        "python_in": python_expr,
        "asf2_sexpr": sexpr,
        "asf0_dyck": dyck,
        "python_out": python_back,
    }


def parse_dyck(s: str) -> Shape:
    s = s.strip()
    if not s:
        raise ValueError("Empty Dyck string")
    if s == "()":
        return Atom()
    if s == "(())":
        return Stage()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Invalid Dyck: {s}")

    inner = s[1:-1]
    children: List[Shape] = []
    depth = 0
    start = 0
    for i, c in enumerate(inner):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                children.append(parse_dyck(inner[start : i + 1]))
                start = i + 1
    if depth != 0:
        raise ValueError(f"Unbalanced Dyck: {s}")
    return Composite(tuple(children))
