"""
Sanity checks for canonicalization and equivalence keys.
"""

import ast as _ast
from pathlib import Path
import importlib.util

from py_to_dyck import compile_source, run_dyck
from python_norm import python_equiv_key


def load_asf_core2():
    path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def run_python_direct(source: str):
    env = {}
    parsed = _ast.parse(source)
    if parsed.body and isinstance(parsed.body[-1], _ast.Expr):
        prefix = _ast.Module(body=parsed.body[:-1], type_ignores=[])
        if prefix.body:
            exec(compile(prefix, "<asf_runtime>", "exec"), env, env)
        expr = _ast.Expression(parsed.body[-1].value)
        return eval(compile(expr, "<asf_runtime>", "eval"), env, env)
    exec(compile(parsed, "<asf_runtime>", "exec"), env, env)
    return None


def check_python_norm_keys():
    eq_pairs = [
        ("1 + 2", "3"),
        ("x if True else y", "x"),
        ("lambda x: x + 1", "lambda y: y + 1"),
    ]
    neq_pairs = [
        ("1 + 2", "4"),
        ("x if True else y", "y"),
    ]

    for a, b in eq_pairs:
        ka = python_equiv_key(a, mode="python_norm")
        kb = python_equiv_key(b, mode="python_norm")
        assert ka == kb, f"expected equiv keys to match: {a} vs {b}"

    for a, b in neq_pairs:
        ka = python_equiv_key(a, mode="python_norm")
        kb = python_equiv_key(b, mode="python_norm")
        assert ka != kb, f"expected equiv keys to differ: {a} vs {b}"


def check_python_parity():
    programs = [
        "1 + 2 * 3",
        "def inc(x):\n    return x + 1\n\ninc(5)",
        "def add(x):\n    return (lambda y: x + y)\n\nadd(2)(3)",
        "y = 0\nx = 9\nx if y == 0 else 3",
    ]
    for src in programs:
        dyck = compile_source(src)
        asf_result = run_dyck(dyck)
        py_result = run_python_direct(src)
        assert asf_result == py_result, f"parity mismatch for: {src}"


def check_asf_semantics():
    core2 = load_asf_core2()
    expr = core2.app(core2.I, core2.Stage())
    norm, status = core2.semantic_normalize(expr)
    assert status == "normal"
    assert norm == core2.Stage()


def main():
    check_python_norm_keys()
    check_python_parity()
    check_asf_semantics()
    print("Sanity checks passed")


if __name__ == "__main__":
    main()
