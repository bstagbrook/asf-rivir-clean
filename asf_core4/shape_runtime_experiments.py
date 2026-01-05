"""
Experiments for cache hits, schema/spine matching, and parity checks.
"""

import ast as _ast
from datetime import datetime
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


def main():
    core2 = load_asf_core2()
    db_path = Path("/Volumes/StagbrookField/stagbrook_field/asf_core4") / (
        "experiments_cache_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".db"
    )
    catalog = core2.PersistentCatalog(str(db_path))

    programs = [
        ("arith", "1 + 2 * 3"),
        ("fold", "1 + 2"),
        ("fold_equiv", "3"),
        ("let", "x = 5\nx + 1"),
        ("def", "def inc(x):\n    return x + 1\n\ninc(5)"),
        ("def_alpha", "def inc(y):\n    return y + 1\n\ninc(5)"),
        ("if", "y = 0\nx = 9\nx if y == 0 else 3"),
        ("if_fold", "x = 9\nx if True else 3"),
        ("hof", "def add(x):\n    return (lambda y: x + y)\n\nadd(2)(3)"),
        ("closure", "def mul(x):\n    return (lambda y: x * y)\n\nfactor = 3\nmul(factor)(4) + 1"),
        ("repeat_def", "def inc(x):\n    return x + 1\n\ninc(5)"),
    ]

    print("Experiment DB:", db_path)
    print("=" * 60)

    for name, source in programs:
        dyck = compile_source(source)
        shape = core2.parse_dyck(dyck)

        content_hit = catalog.get(shape) is not None
        schema_candidates = catalog.find_by_schema(core2.schema_key(shape))
        spine1 = catalog.find_by_spine(shape, depth=1)
        spine2 = catalog.find_by_spine(shape, depth=2)
        spine3 = catalog.find_by_spine(shape, depth=3)

        catalog.put(shape)
        eq_key = python_equiv_key(source, mode="python_norm")
        eq_status = "python_norm"
        catalog.set_equiv_key(shape, eq_key, status=eq_status)
        equiv_candidates = catalog.find_by_equiv(eq_key)

        cached = catalog.get_result_by_equiv(eq_key)
        if cached is None:
            asf_result = run_dyck(dyck)
            catalog.set_result_by_equiv(eq_key, asf_result)
            equiv_cache_hit = False
        else:
            asf_result = cached
            equiv_cache_hit = True
        py_result = run_python_direct(source)

        catalog.set_label(shape, name, namespace="program", confidence=1.0, notes="experiment label")
        label_keys = set(catalog.find_by_label(name, namespace="program"))
        candidate_keys = {core2.key(e.shape) for e in equiv_candidates}
        label_hits = len(label_keys & candidate_keys)

        print(f"[{name}]")
        print(f"  cache_hit={content_hit} schema={len(schema_candidates)} spine1={len(spine1)} spine2={len(spine2)} spine3={len(spine3)} equiv={len(equiv_candidates)}")
        print(f"  equiv_cache_hit={equiv_cache_hit} label_hits={label_hits}")
        assert py_result == asf_result, f"parity mismatch for {name}"
        print(f"  py_result={py_result} asf_result={asf_result} equiv_status={eq_status}")

    catalog.close()
    print("=" * 60)
    print("Done")


if __name__ == "__main__":
    main()
