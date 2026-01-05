"""
Python -> raw Dyck transpiler and runtime (minimal subset).

Supported Python subset:
  - int and bool literals
  - variables
  - lambda with one argument
  - function calls with one argument
  - if-expressions: x if cond else y
  - binary ops: +, -, *
  - equality: ==
  - unary minus
  - module-level assignments and def (compiled as lambdas)

Target encoding uses only () and (()).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import ast

# ============================================================
# DYCK SHAPES
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


def parse_dyck(s: str) -> Shape:
    """
    Parse a Dyck string into a Shape with strict validation.

    Raises ValueError for:
    - Empty input
    - Invalid characters (anything other than '(' and ')')
    - Unbalanced parentheses
    - Negative depth (more ')' than '(' at any point)
    - Trailing characters after valid expression
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty Dyck string")

    # Validate characters
    invalid_chars = set(s) - {'(', ')'}
    if invalid_chars:
        raise ValueError(f"Invalid characters in Dyck string: {invalid_chars!r}")

    # Quick balance check
    if s.count('(') != s.count(')'):
        raise ValueError(f"Unbalanced Dyck: {s.count('(')} open vs {s.count(')')} close")

    # Check for negative depth (prefix validation)
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                raise ValueError(f"Negative depth at position {i}: too many ')' before '('")

    # Primitives
    if s == "()":
        return Atom()
    if s == "(())":
        return Stage()

    # Must be wrapped in parens
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Invalid Dyck (must be wrapped in parens): {s}")

    # Parse children
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
                children.append(parse_dyck(inner[start:i + 1]))
                start = i + 1

    # Check for trailing characters
    if start < len(inner):
        trailing = inner[start:]
        raise ValueError(f"Trailing characters after valid expression: {trailing!r}")

    if depth != 0:
        raise ValueError(f"Unbalanced Dyck (inner): {s}")

    return Composite(tuple(children))


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
# ENCODING
# ============================================================

TAG_VAR = 0
TAG_LAM = 1
TAG_APP = 2
TAG_INT = 3
TAG_BOOL = 4
TAG_IF = 5
TAG_PRIM = 6

OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_EQ = 3


def encode_nat(n: int) -> Shape:
    if n < 0:
        raise ValueError("encode_nat expects non-negative")
    # Composite of (n + 2) Atoms to avoid ambiguity.
    return Composite(tuple(Atom() for _ in range(n + 2)))


def decode_nat(shape: Shape) -> int:
    if not isinstance(shape, Composite):
        raise ValueError("decode_nat expects Composite")
    if len(shape.children) < 2:
        raise ValueError("decode_nat expects >=2 children")
    if not all(isinstance(c, Atom) for c in shape.children):
        raise ValueError("decode_nat expects only Atoms")
    return len(shape.children) - 2


def tag_shape(tag_id: int) -> Shape:
    return Composite((Stage(), encode_nat(tag_id)))


def is_tag(shape: Shape) -> bool:
    return (
        isinstance(shape, Composite)
        and len(shape.children) == 2
        and isinstance(shape.children[0], Stage)
        and isinstance(shape.children[1], Composite)
    )


def tag_id(shape: Shape) -> int:
    if not is_tag(shape):
        raise ValueError("Not a tag shape")
    return decode_nat(shape.children[1])


# ============================================================
# CORE IR
# ============================================================

@dataclass(frozen=True)
class Var:
    index: int


@dataclass(frozen=True)
class Lam:
    body: "Expr"


@dataclass(frozen=True)
class App:
    fn: "Expr"
    arg: "Expr"


@dataclass(frozen=True)
class Int:
    value: int


@dataclass(frozen=True)
class Bool:
    value: bool


@dataclass(frozen=True)
class If:
    cond: "Expr"
    then: "Expr"
    other: "Expr"


@dataclass(frozen=True)
class Prim:
    op: int
    left: "Expr"
    right: "Expr"


Expr = Union[Var, Lam, App, Int, Bool, If, Prim]


def encode_expr(expr: Expr) -> Shape:
    if isinstance(expr, Var):
        return Composite((tag_shape(TAG_VAR), encode_nat(expr.index)))
    if isinstance(expr, Lam):
        return Composite((tag_shape(TAG_LAM), encode_expr(expr.body)))
    if isinstance(expr, App):
        return Composite((tag_shape(TAG_APP), encode_expr(expr.fn), encode_expr(expr.arg)))
    if isinstance(expr, Int):
        if expr.value < 0:
            raise ValueError("Negative ints must be lowered before encoding")
        return Composite((tag_shape(TAG_INT), encode_nat(expr.value)))
    if isinstance(expr, Bool):
        return Composite((tag_shape(TAG_BOOL), encode_nat(1 if expr.value else 0)))
    if isinstance(expr, If):
        return Composite(
            (tag_shape(TAG_IF), encode_expr(expr.cond), encode_expr(expr.then), encode_expr(expr.other))
        )
    if isinstance(expr, Prim):
        return Composite(
            (tag_shape(TAG_PRIM), encode_nat(expr.op), encode_expr(expr.left), encode_expr(expr.right))
        )
    raise TypeError(f"Unknown expr: {type(expr)}")


def decode_expr(shape: Shape) -> Expr:
    if not isinstance(shape, Composite) or len(shape.children) < 2:
        raise ValueError("Invalid encoded expr")

    t = tag_id(shape.children[0])
    args = shape.children[1:]

    if t == TAG_VAR:
        return Var(decode_nat(args[0]))
    if t == TAG_LAM:
        return Lam(decode_expr(args[0]))
    if t == TAG_APP:
        return App(decode_expr(args[0]), decode_expr(args[1]))
    if t == TAG_INT:
        return Int(decode_nat(args[0]))
    if t == TAG_BOOL:
        return Bool(decode_nat(args[0]) == 1)
    if t == TAG_IF:
        return If(decode_expr(args[0]), decode_expr(args[1]), decode_expr(args[2]))
    if t == TAG_PRIM:
        return Prim(decode_nat(args[0]), decode_expr(args[1]), decode_expr(args[2]))

    raise ValueError(f"Unknown tag id: {t}")


# ============================================================
# COMPILER (PYTHON AST -> IR)
# ============================================================

def compile_source(source: str) -> str:
    module = ast.parse(source)
    expr = compile_module(module)
    shape = encode_expr(expr)
    return serialize_dyck(shape)


def compile_module(module: ast.Module) -> Expr:
    env: List[str] = []
    bindings: List[Tuple[str, Expr]] = []
    final_expr: Optional[Expr] = None

    for stmt in module.body:
        if isinstance(stmt, ast.Expr):
            final_expr = compile_expr(stmt.value, env)
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments are supported")
            name = stmt.targets[0].id
            value = compile_expr(stmt.value, env)
            bindings.append((name, value))
            env.insert(0, name)
        elif isinstance(stmt, ast.FunctionDef):
            if len(stmt.args.args) != 1:
                raise ValueError("Only single-arg functions are supported")
            arg_name = stmt.args.args[0].arg
            if len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Return):
                raise ValueError("Function body must be a single return")
            fn_body = compile_expr(stmt.body[0].value, [arg_name] + env)
            fn_expr = Lam(fn_body)
            bindings.append((stmt.name, fn_expr))
            env.insert(0, stmt.name)
        else:
            raise ValueError(f"Unsupported statement: {type(stmt).__name__}")

    if final_expr is None:
        raise ValueError("Module has no expression to evaluate")

    for _, value in reversed(bindings):
        final_expr = App(Lam(final_expr), value)

    return final_expr


def compile_expr(node: ast.AST, env: List[str]) -> Expr:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return Bool(node.value)
        if isinstance(node.value, int):
            if node.value < 0:
                return Prim(OP_SUB, Int(0), Int(-node.value))
            return Int(node.value)
        raise ValueError("Only int/bool literals are supported")

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unbound name: {node.id}")
        return Var(env.index(node.id))

    if isinstance(node, ast.Lambda):
        if len(node.args.args) != 1:
            raise ValueError("Only single-arg lambdas are supported")
        arg = node.args.args[0].arg
        body = compile_expr(node.body, [arg] + env)
        return Lam(body)

    if isinstance(node, ast.Call):
        if len(node.args) != 1:
            raise ValueError("Only single-arg calls are supported")
        fn = compile_expr(node.func, env)
        arg = compile_expr(node.args[0], env)
        return App(fn, arg)

    if isinstance(node, ast.IfExp):
        cond = compile_expr(node.test, env)
        then = compile_expr(node.body, env)
        other = compile_expr(node.orelse, env)
        return If(cond, then, other)

    if isinstance(node, ast.BinOp):
        left = compile_expr(node.left, env)
        right = compile_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return Prim(OP_ADD, left, right)
        if isinstance(node.op, ast.Sub):
            return Prim(OP_SUB, left, right)
        if isinstance(node.op, ast.Mult):
            return Prim(OP_MUL, left, right)
        raise ValueError("Only +, -, * are supported")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single comparisons are supported")
        left = compile_expr(node.left, env)
        right = compile_expr(node.comparators[0], env)
        if isinstance(node.ops[0], ast.Eq):
            return Prim(OP_EQ, left, right)
        raise ValueError("Only == is supported")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = compile_expr(node.operand, env)
        return Prim(OP_SUB, Int(0), value)

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


# ============================================================
# RUNTIME
# ============================================================

@dataclass
class Closure:
    body: Expr
    env: List[Any]


def eval_expr(expr: Expr, env: List[Any]) -> Any:
    if isinstance(expr, Var):
        return env[expr.index]
    if isinstance(expr, Lam):
        return Closure(expr.body, env.copy())
    if isinstance(expr, App):
        fn = eval_expr(expr.fn, env)
        arg = eval_expr(expr.arg, env)
        if not isinstance(fn, Closure):
            raise ValueError("Attempted to call a non-function")
        return eval_expr(fn.body, [arg] + fn.env)
    if isinstance(expr, Int):
        return expr.value
    if isinstance(expr, Bool):
        return expr.value
    if isinstance(expr, If):
        cond = eval_expr(expr.cond, env)
        return eval_expr(expr.then if cond else expr.other, env)
    if isinstance(expr, Prim):
        left = eval_expr(expr.left, env)
        right = eval_expr(expr.right, env)
        if expr.op == OP_ADD:
            return left + right
        if expr.op == OP_SUB:
            return left - right
        if expr.op == OP_MUL:
            return left * right
        if expr.op == OP_EQ:
            return left == right
    raise ValueError(f"Unknown expression: {expr}")


def run_dyck(dyck: str) -> Any:
    shape = parse_dyck(dyck)
    expr = decode_expr(shape)
    return eval_expr(expr, [])


def run_source(source: str) -> Any:
    dyck = compile_source(source)
    return run_dyck(dyck)


def main():
    program = """
def inc(x):
    return x + 1

inc(5) if True else 0
"""
    dyck = compile_source(program)
    print("Dyck:", dyck[:80] + "...")
    print("Result:", run_dyck(dyck))


if __name__ == "__main__":
    main()
