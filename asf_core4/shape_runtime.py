"""
Clockless shape runtime: compile Python -> Dyck -> Shape, cache, normalize, run.

Portable loading:
1. Checks for _asf_core2_shim module
2. Checks ASF_CORE2_PATH environment variable
3. Falls back to .asf_core2.py or asf_core2.py alongside this file
4. Falls back to known location (for backwards compatibility)
"""

import argparse
import ast as _ast
import os
from pathlib import Path
import importlib.util

try:
    from .py_to_dyck import compile_source, run_dyck
except ImportError:
    from py_to_dyck import compile_source, run_dyck

try:
    from . import _asf_core2_shim  # type: ignore
except Exception:
    _asf_core2_shim = None


def load_asf_core2():
    """
    Load ASF Core 2 module using portable search strategy.

    Search order:
    1. _asf_core2_shim if present
    2. ASF_CORE2_PATH environment variable
    3. .asf_core2.py alongside this file
    4. asf_core2.py alongside this file
    5. Known legacy path (for backwards compatibility)
    """
    if _asf_core2_shim is not None:
        return _asf_core2_shim

    tried_paths = []

    # Check environment variable
    env_path = os.environ.get("ASF_CORE2_PATH")
    if env_path:
        path = Path(env_path)
        tried_paths.append(str(path))
        if path.exists():
            return _load_module_from_path(path)

    # Check alongside this file
    this_dir = Path(__file__).parent
    for name in [".asf_core2.py", "asf_core2.py"]:
        path = this_dir / name
        tried_paths.append(str(path))
        if path.exists():
            return _load_module_from_path(path)

    # Check parent directory
    parent_dir = this_dir.parent
    for name in [".asf_core2.py", "asf_core2.py"]:
        path = parent_dir / name
        tried_paths.append(str(path))
        if path.exists():
            return _load_module_from_path(path)

    # Legacy fallback
    legacy_path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    tried_paths.append(str(legacy_path))
    if legacy_path.exists():
        return _load_module_from_path(legacy_path)

    raise FileNotFoundError(
        f"Could not find ASF Core 2 module. Tried paths:\n" +
        "\n".join(f"  - {p}" for p in tried_paths) +
        "\n\nSet ASF_CORE2_PATH environment variable or place asf_core2.py alongside this module."
    )


def _load_module_from_path(path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# Graceful python_norm import
try:
    from python_norm import python_equiv_key
    _HAS_PYTHON_NORM = True
except ImportError:
    _HAS_PYTHON_NORM = False

    def python_equiv_key(source: str, mode: str = "python_ast") -> str:
        """Fallback equivalence key using AST hash."""
        import hashlib
        _ = mode  # Unused in fallback, but kept for API compatibility
        try:
            tree = _ast.parse(source)
            # Simple AST dump hash
            return hashlib.sha256(_ast.dump(tree).encode()).hexdigest()[:32]
        except SyntaxError:
            # Fall back to source hash
            return hashlib.sha256(source.encode()).hexdigest()[:32]


def run_python_source(
    source: str,
    db_path: str,
    label: str = None,
    namespace: str = "program",
    equiv_mode: str = "python_ast"
):
    """
    Compile Python source to Dyck, store in catalog, normalize, and run.

    Args:
        source: Python source code
        db_path: Path to SQLite catalog database
        label: Optional label to attach
        namespace: Label namespace
        equiv_mode: Equivalence key mode (python_ast, python_norm, beta, beta_eta)
    """
    core2 = load_asf_core2()
    catalog = core2.PersistentCatalog(db_path)

    dyck = compile_source(source)
    shape = core2.parse_dyck(dyck)

    content_hit = catalog.get(shape) is not None
    schema_candidates = catalog.find_by_schema(core2.schema_key(shape))
    spine1 = catalog.find_by_spine(shape, depth=1)
    spine2 = catalog.find_by_spine(shape, depth=2)
    spine3 = catalog.find_by_spine(shape, depth=3)

    entry = catalog.put(shape)

    # Determine equivalence key based on mode and available modules
    if equiv_mode in ("python_ast", "python_norm"):
        if equiv_mode == "python_norm" and not _HAS_PYTHON_NORM:
            print(f"Warning: python_norm not available, falling back to python_ast")
            equiv_mode = "python_ast"
        eq_key = python_equiv_key(source, mode=equiv_mode)
        eq_status = equiv_mode
        catalog.set_equiv_key(shape, eq_key, status=eq_status)
    else:
        eq_key, eq_status = catalog.set_equiv(shape, mode=equiv_mode)

    equiv_candidates = catalog.find_by_equiv(eq_key)

    equiv_cached = catalog.get_result_by_equiv(eq_key)
    if equiv_cached is not None:
        asf_result = equiv_cached
        equiv_cache_hit = True
    else:
        asf_result = run_dyck(dyck)
        catalog.set_result_by_equiv(eq_key, asf_result)
        equiv_cache_hit = False

    py_result = run_python_direct(source)

    print("Dyck prefix:", dyck[:80] + "..." if len(dyck) > 80 else dyck)
    print("shape_key:", core2.key(shape))
    print("schema_key:", core2.schema_key(shape))
    print("spine1:", len(spine1))
    print("spine2:", len(spine2))
    print("spine3:", len(spine3))
    print("equiv_key:", eq_key)
    print("equiv_status:", eq_status)
    print("cache_hit:", content_hit)
    print("schema_candidates:", len(schema_candidates))
    print("equiv_candidates:", len(equiv_candidates))
    print("equiv_cache_hit:", equiv_cache_hit)

    if label:
        catalog.set_label(shape, label, namespace=namespace, confidence=1.0, notes="runtime label")
        label_keys = set(catalog.find_by_label(label, namespace=namespace))
        candidate_keys = {core2.key(e.shape) for e in equiv_candidates}
        print("label_hits:", len(label_keys & candidate_keys))

    print("python_result:", py_result)
    print("asf_result:", asf_result)

    catalog.close()


def run_python_direct(source: str):
    """Execute Python and return the value of the last expression if present."""
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
    parser = argparse.ArgumentParser(
        description="Clockless shape runtime: Python -> Dyck -> Shape"
    )
    parser.add_argument("--python", help="Python source string to compile and run")
    parser.add_argument("--file", help="Path to a Python source file")
    parser.add_argument("--db", default="asf_shape_runtime.db", help="SQLite catalog DB path")
    parser.add_argument("--label", help="Optional label to attach to the shape")
    parser.add_argument("--namespace", default="program", help="Label namespace")
    parser.add_argument("--equiv", default="python_ast",
                        help="Equivalence mode: python_ast, python_norm, beta, beta_eta")
    args = parser.parse_args()

    if not args.python and not args.file:
        raise SystemExit("Provide --python or --file")

    if args.python:
        source = args.python.replace("\\n", "\n")
    else:
        source = Path(args.file).read_text(encoding="utf-8")

    run_python_source(
        source,
        args.db,
        label=args.label,
        namespace=args.namespace,
        equiv_mode=args.equiv
    )


if __name__ == "__main__":
    main()
