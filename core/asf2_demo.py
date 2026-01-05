"""
ASF2 Phase 1 demo: Python <-> ASF2 <-> ASF0 round-trip (expression subset).
"""

import argparse

from asf2 import (
    python_expr_to_ast,
    asf2_ast_to_python_src,
    asf2_to_sexpr,
    asf2_to_dyck,
    asf2_from_dyck,
    explain_pipeline,
)


CASES = [
    "1",
    "True",
    "x",
    "1 + 2 * 3",
    "x if y == 0 else 3",
    "lambda x: x + 1",
    "f(3)",
    "(lambda x: x * x)(4)",
    "-(1 + 2)",
]


def run_case(src: str, explain: bool):
    if explain:
        steps = explain_pipeline(src)
        print("Python input:")
        print(steps["python_in"])
        print("ASF2 S-expr:")
        print(steps["asf2_sexpr"])
        print("ASF0 Dyck prefix:")
        dyck = steps["asf0_dyck"]
        print(dyck[:80] + "..." if len(dyck) > 80 else dyck)
        print("Python output:")
        print(steps["python_out"])
        print()
        return

    asf2_ast = python_expr_to_ast(src)
    sexpr = asf2_to_sexpr(asf2_ast)
    dyck = asf2_to_dyck(asf2_ast)
    asf2_back = asf2_from_dyck(dyck)
    src_back = asf2_ast_to_python_src(asf2_back)

    print("Source:")
    print(src)
    print("ASF2 S-expr:")
    print(sexpr)
    print("Dyck prefix:")
    print(dyck[:80] + "..." if len(dyck) > 80 else dyck)
    print("Back to Python:")
    print(src_back)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true", help="Show desugaring/resugaring steps")
    args = parser.parse_args()

    for src in CASES:
        run_case(src, args.explain)


if __name__ == "__main__":
    main()
