"""
Incremental demo cases for the Python -> Dyck transpiler/runtime.
"""

from py_to_dyck import compile_source, run_dyck


CASES = [
    (
        "Literal",
        "42",
    ),
    (
        "Arithmetic",
        "1 + 2 * 3",
    ),
    (
        "Let bindings",
        "x = 5\ny = 2\nx * y + 1",
    ),
    (
        "If expression",
        "3 if (2 + 2) == 4 else 0",
    ),
    (
        "Lambda application",
        "(lambda x: x + 1)(10)",
    ),
    (
        "Def + call",
        "def inc(x):\n    return x + 1\n\ninc(7)",
    ),
    (
        "Higher-order function",
        "def add(x):\n    return (lambda y: x + y)\n\nadd(2)(3)",
    ),
    (
        "Closure + nested lets",
        "def mul(x):\n    return (lambda y: x * y)\n\nfactor = 3\nmul(factor)(4) + 1",
    ),
]


def run_case(name: str, source: str):
    dyck = compile_source(source)
    result = run_dyck(dyck)
    prefix = dyck[:80] + "..." if len(dyck) > 80 else dyck
    print(name)
    print("-" * len(name))
    print("Source:")
    print(source)
    print("Dyck prefix:")
    print(prefix)
    print("Result:")
    print(result)
    print()


def main():
    for name, source in CASES:
        run_case(name, source)


if __name__ == "__main__":
    main()
