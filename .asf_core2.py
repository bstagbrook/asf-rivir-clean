"""
ASF Core 2 - Atomic State Flow
==============================

A shape-first computing foundation with an unambiguous encoding.

Primitives:
  - Atom (A)  : declaration from nothing ("is")
  - Stage (S) : container of distinction ("has")

Compound:
  - Composite(children) : ordered collection

Serialization format (canonical, unambiguous):
  - Atom  -> "A"
  - Stage -> "S"
  - Composite(children...) -> "(" + children + ")"

Lambda calculus via de Bruijn indices:
  - ref(n)     : reference to variable bound n levels up
  - lam(body)  : lambda binder
  - app(f, x)  : application

Content-addressed catalog with persistence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any
import ast
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime

# ============================================================
# SHAPES
# ============================================================

def dataclass_slots(**kwargs):
    """Use dataclass slots when supported; fallback for older Python."""
    try:
        return dataclass(**kwargs, slots=True)
    except TypeError:
        return dataclass(**kwargs)

@dataclass_slots(frozen=True)
class Atom:
    """The A primitive - 'is' - declaration from nothing."""
    pass

@dataclass_slots(frozen=True)
class Stage:
    """The S primitive - 'has' - container of distinction."""
    pass

@dataclass_slots(frozen=True)
class Composite:
    """Ordered children - the only compound structure."""
    children: Tuple['Shape', ...]

@dataclass_slots(frozen=True)
class Hole:
    """Pattern variable for matching."""
    index: int

Shape = Union[Atom, Stage, Composite, Hole]

# Singletons for convenience
IS = Atom()
HAS = Stage()

def composite(*children: Shape) -> Composite:
    return Composite(tuple(children))

# ============================================================
# PARSING
# ============================================================

def _tokenize(s: str) -> List[str]:
    tokens: List[str] = []
    for c in s:
        if c.isspace():
            continue
        if c in ("(", ")", "A", "S"):
            tokens.append(c)
        else:
            raise ValueError(f"Invalid character in input: {c!r}")
    return tokens

def parse(s: str) -> Shape:
    """
    Parse a canonical shape string.
    Grammar:
      shape := 'A' | 'S' | '(' shape* ')'
    """
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("Empty input")

    pos = 0

    def parse_shape() -> Shape:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")
        tok = tokens[pos]
        pos += 1
        if tok == "A":
            return Atom()
        if tok == "S":
            return Stage()
        if tok == "(":
            children: List[Shape] = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("Unbalanced: missing ')'")
                if tokens[pos] == ")":
                    pos += 1
                    return Composite(tuple(children))
                children.append(parse_shape())
        raise ValueError(f"Unexpected token: {tok!r}")

    result = parse_shape()
    if pos != len(tokens):
        raise ValueError("Trailing tokens after valid shape")
    return result

def parse_dyck(s: str) -> Shape:
    """Parse raw Dyck input using only () and (())."""
    s = s.strip()
    if not s:
        raise ValueError("Empty input")
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
                children.append(parse_dyck(inner[start:i+1]))
                start = i + 1

    if depth != 0:
        raise ValueError(f"Unbalanced Dyck: {s}")
    return Composite(tuple(children))

def serialize_dyck(shape: Shape) -> bytes:
    """Serialize to raw Dyck using only () and (())."""
    if isinstance(shape, Atom):
        return b"()"
    if isinstance(shape, Stage):
        return b"(())"
    if isinstance(shape, Composite):
        inner = b"".join(serialize_dyck(c) for c in shape.children)
        return b"(" + inner + b")"
    if isinstance(shape, Hole):
        raise ValueError("Cannot serialize Hole")
    raise TypeError(f"Unknown: {type(shape)}")

# ============================================================
# SERIALIZATION & HASHING
# ============================================================

def serialize(shape: Shape) -> bytes:
    """Convert shape to canonical bytes."""
    if isinstance(shape, Atom):
        return b"A"
    if isinstance(shape, Stage):
        return b"S"
    if isinstance(shape, Composite):
        inner = b"".join(serialize(c) for c in shape.children)
        return b"(" + inner + b")"
    if isinstance(shape, Hole):
        raise ValueError("Cannot serialize Hole")
    raise TypeError(f"Unknown: {type(shape)}")

def schema_serialize(shape: Shape) -> bytes:
    """
    Serialize only structure (ignore Atom vs Stage distinction).
    Useful for structural similarity bucketing.
    """
    if isinstance(shape, (Atom, Stage)):
        return b"L"
    if isinstance(shape, Composite):
        inner = b"".join(schema_serialize(c) for c in shape.children)
        return b"(" + inner + b")"
    if isinstance(shape, Hole):
        raise ValueError("Cannot serialize Hole")
    raise TypeError(f"Unknown: {type(shape)}")

def spine_serialize(shape: Shape, depth: int) -> bytes:
    """
    Serialize a shape with holes below a depth.
    depth == 0 collapses to a hole marker.
    """
    if depth <= 0:
        return b"H"
    if isinstance(shape, (Atom, Stage)):
        return b"L"
    if isinstance(shape, Composite):
        inner = b"".join(spine_serialize(c, depth - 1) for c in shape.children)
        return b"(" + inner + b")"
    if isinstance(shape, Hole):
        raise ValueError("Cannot serialize Hole")
    raise TypeError(f"Unknown: {type(shape)}")

def key(shape: Shape) -> str:
    """Content-addressed key (SHA256)."""
    return hashlib.sha256(serialize(shape)).hexdigest()

def schema_key(shape: Shape) -> str:
    """Structure-only key (SHA256)."""
    return hashlib.sha256(schema_serialize(shape)).hexdigest()

def spine_key(shape: Shape, depth: int) -> str:
    """Structure-with-holes key (SHA256) for a given depth."""
    return hashlib.sha256(spine_serialize(shape, depth)).hexdigest()

def pretty(shape: Shape, indent: int = 0) -> str:
    """Pretty-print a shape for debugging."""
    prefix = "  " * indent
    if isinstance(shape, Atom):
        return f"{prefix}Atom"
    if isinstance(shape, Stage):
        return f"{prefix}Stage"
    if isinstance(shape, Composite):
        if not shape.children:
            return f"{prefix}Composite()"
        children_str = "\n".join(pretty(c, indent + 1) for c in shape.children)
        return f"{prefix}Composite(\n{children_str}\n{prefix})"
    if isinstance(shape, Hole):
        return f"{prefix}Hole({shape.index})"
    return f"{prefix}???"

# ============================================================
# LAMBDA CALCULUS (DE BRUIJN ENCODING)
# ============================================================

def ref(n: int) -> Shape:
    """
    Reference to variable bound n levels up.
    Encoding: ⟨Atom ⟨n Atoms⟩⟩
    """
    index_shape = Composite(tuple(Atom() for _ in range(n)))
    return Composite((Atom(), index_shape))

def lam(body: Shape) -> Shape:
    """
    Lambda binder.
    Encoding: ⟨Stage body⟩
    """
    return Composite((Stage(), body))

def app(func: Shape, arg: Shape) -> Shape:
    """
    Application.
    Encoding: ⟨func arg⟩ (distinguished by not being ref or lam)
    """
    return Composite((func, arg))

def is_ref(shape: Shape) -> bool:
    """Check if shape is a de Bruijn reference."""
    if not isinstance(shape, Composite) or len(shape.children) != 2:
        return False
    if not isinstance(shape.children[0], Atom):
        return False
    if not isinstance(shape.children[1], Composite):
        return False
    return all(isinstance(c, Atom) for c in shape.children[1].children)

def is_lam(shape: Shape) -> bool:
    """Check if shape is a lambda."""
    return (isinstance(shape, Composite) and
            len(shape.children) == 2 and
            isinstance(shape.children[0], Stage))

def is_app(shape: Shape) -> bool:
    """Check if shape is an application."""
    return (isinstance(shape, Composite) and
            len(shape.children) == 2 and
            not is_ref(shape) and
            not is_lam(shape))

def ref_index(shape: Shape) -> int:
    """Extract de Bruijn index from reference."""
    assert is_ref(shape)
    return len(shape.children[1].children)

def lam_body(shape: Shape) -> Shape:
    """Extract body from lambda."""
    assert is_lam(shape)
    return shape.children[1]

def app_func(shape: Shape) -> Shape:
    """Extract function from application."""
    return shape.children[0]

def app_arg(shape: Shape) -> Shape:
    """Extract argument from application."""
    return shape.children[1]

# ============================================================
# DE BRUIJN OPERATIONS
# ============================================================

def shift(shape: Shape, delta: int, cutoff: int = 0) -> Shape:
    """
    Shift free variables by delta.
    Variables with index >= cutoff are shifted.
    """
    if isinstance(shape, (Atom, Stage)):
        return shape

    if is_ref(shape):
        idx = ref_index(shape)
        if idx >= cutoff:
            new_idx = idx + delta
            if new_idx < 0:
                raise ValueError(f"Negative index after shift: {new_idx}")
            return ref(new_idx)
        return shape

    if is_lam(shape):
        return lam(shift(lam_body(shape), delta, cutoff + 1))

    if is_app(shape):
        return app(shift(app_func(shape), delta, cutoff),
                   shift(app_arg(shape), delta, cutoff))

    if isinstance(shape, Composite):
        return Composite(tuple(shift(c, delta, cutoff) for c in shape.children))

    return shape

def subst(body: Shape, arg: Shape, depth: int = 0) -> Shape:
    """
    Substitute arg for ref(depth) in body.
    Standard de Bruijn substitution.
    """
    if isinstance(body, (Atom, Stage)):
        return body

    if is_ref(body):
        idx = ref_index(body)
        if idx == depth:
            # Substitute: shift arg to account for binders crossed
            return shift(arg, depth)
        if idx > depth:
            # Free variable above: decrement (binder removed)
            return ref(idx - 1)
        # Bound below: unchanged
        return ref(idx)

    if is_lam(body):
        return lam(subst(lam_body(body), arg, depth + 1))

    if is_app(body):
        return app(subst(app_func(body), arg, depth),
                   subst(app_arg(body), arg, depth))

    if isinstance(body, Composite):
        return Composite(tuple(subst(c, arg, depth) for c in body.children))

    return body

# ============================================================
# ETA REDUCTION
# ============================================================

def _uses_ref_at_depth(shape: Shape, depth: int) -> bool:
    if isinstance(shape, (Atom, Stage)):
        return False
    if is_ref(shape):
        return ref_index(shape) == depth
    if is_lam(shape):
        return _uses_ref_at_depth(lam_body(shape), depth + 1)
    if is_app(shape):
        return _uses_ref_at_depth(app_func(shape), depth) or _uses_ref_at_depth(app_arg(shape), depth)
    if isinstance(shape, Composite):
        return any(_uses_ref_at_depth(c, depth) for c in shape.children)
    return False


def eta_step(shape: Shape) -> Optional[Shape]:
    """
    One eta-reduction step (leftmost-outermost).
    λ. (f 0) -> f  if 0 not free in f
    """
    if is_lam(shape):
        body = lam_body(shape)
        if is_app(body):
            f = app_func(body)
            arg = app_arg(body)
            if is_ref(arg) and ref_index(arg) == 0 and not _uses_ref_at_depth(f, 0):
                return shift(f, -1)
        reduced = eta_step(body)
        if reduced is not None:
            return lam(reduced)

    if is_app(shape):
        reduced = eta_step(app_func(shape))
        if reduced is not None:
            return app(reduced, app_arg(shape))
        reduced = eta_step(app_arg(shape))
        if reduced is not None:
            return app(app_func(shape), reduced)

    if isinstance(shape, Composite):
        for i, child in enumerate(shape.children):
            reduced = eta_step(child)
            if reduced is not None:
                new_children = list(shape.children)
                new_children[i] = reduced
                return Composite(tuple(new_children))

    return None


def eta_normalize(shape: Shape, max_steps: int = 10000) -> Tuple[Shape, str]:
    """
    Reduce to eta normal form.
    Returns (result, status) where status is 'normal' or 'max_steps'.
    """
    for _ in range(max_steps):
        reduced = eta_step(shape)
        if reduced is None:
            return (shape, "normal")
        shape = reduced
    return (shape, "max_steps")


def semantic_normalize(shape: Shape, max_steps: int = 10000) -> Tuple[Shape, str]:
    """
    Normalize with beta then eta reductions.
    """
    result, status = beta_normalize(shape, max_steps)
    if status != "normal":
        return (result, status)
    return eta_normalize(result, max_steps)

# ============================================================
# BETA REDUCTION
# ============================================================

def beta_step(shape: Shape) -> Optional[Shape]:
    """
    One beta reduction step (leftmost-outermost).
    Returns None if no redex found.
    """
    # Check for redex at top level
    if is_app(shape):
        func = app_func(shape)
        arg = app_arg(shape)
        if is_lam(func):
            # (λ. body) arg → body[0 := arg]
            body = lam_body(func)
            return subst(body, arg, 0)

    # Try inside lambda body
    if is_lam(shape):
        reduced = beta_step(lam_body(shape))
        if reduced is not None:
            return lam(reduced)

    # Try inside application
    if is_app(shape):
        reduced = beta_step(app_func(shape))
        if reduced is not None:
            return app(reduced, app_arg(shape))
        reduced = beta_step(app_arg(shape))
        if reduced is not None:
            return app(app_func(shape), reduced)

    # Try inside generic composite
    if isinstance(shape, Composite):
        for i, child in enumerate(shape.children):
            reduced = beta_step(child)
            if reduced is not None:
                new_children = list(shape.children)
                new_children[i] = reduced
                return Composite(tuple(new_children))

    return None

def beta_normalize(shape: Shape, max_steps: int = 10000) -> Tuple[Shape, str]:
    """
    Reduce to beta normal form.
    Returns (result, status) where status is 'normal' or 'max_steps'.
    """
    for _ in range(max_steps):
        reduced = beta_step(shape)
        if reduced is None:
            return (shape, "normal")
        shape = reduced
    return (shape, "max_steps")

# ============================================================
# VALIDATION / NORMALIZATION
# ============================================================

class NormalizationError(Exception):
    pass

def validate(shape: Shape, depth: int = 0) -> Shape:
    """
    Validate a shape (check de Bruijn indices are in bounds).
    Returns the shape unchanged if valid.
    """
    if isinstance(shape, (Atom, Stage)):
        return shape

    if isinstance(shape, Hole):
        raise NormalizationError("Cannot validate shape with Holes")

    if is_ref(shape):
        idx = ref_index(shape)
        if idx >= depth:
            raise NormalizationError(f"Unbound ref: index {idx} at depth {depth}")
        return shape

    if is_lam(shape):
        validate(lam_body(shape), depth + 1)
        return shape

    if isinstance(shape, Composite):
        for c in shape.children:
            validate(c, depth)
        return shape

    raise TypeError(f"Unknown shape: {type(shape)}")

def is_closed(shape: Shape) -> bool:
    """Check if shape has no free variables."""
    try:
        validate(shape)
        return True
    except NormalizationError:
        return False

# ============================================================
# PATTERN MATCHING (for rewrite rules)
# ============================================================

def match(pattern: Shape, target: Shape) -> Optional[Dict[int, Shape]]:
    """Match pattern against target, return captures or None."""
    captures: Dict[int, Shape] = {}

    def go(p: Shape, t: Shape) -> bool:
        if isinstance(p, Hole):
            if p.index in captures:
                return captures[p.index] == t
            captures[p.index] = t
            return True
        if type(p) != type(t):
            return False
        if isinstance(p, (Atom, Stage)):
            return True
        if isinstance(p, Composite):
            if len(p.children) != len(t.children):
                return False
            return all(go(pc, tc) for pc, tc in zip(p.children, t.children))
        return False

    return captures if go(pattern, target) else None

def substitute_holes(shape: Shape, captures: Dict[int, Shape]) -> Shape:
    """Replace Holes with captured values."""
    if isinstance(shape, Hole):
        return captures[shape.index]
    if isinstance(shape, (Atom, Stage)):
        return shape
    if isinstance(shape, Composite):
        return Composite(tuple(substitute_holes(c, captures) for c in shape.children))
    raise TypeError(f"Unknown: {type(shape)}")

def anti_unify(a: Shape, b: Shape, next_index: int = 0) -> Tuple[Shape, int]:
    """
    Compute a generalization with Holes where shapes differ.
    Returns (pattern, next_index).
    """
    if type(a) != type(b):
        return (Hole(next_index), next_index + 1)
    if isinstance(a, (Atom, Stage)):
        return (a, next_index)
    if isinstance(a, Hole):
        return (a, next_index)
    if isinstance(a, Composite):
        if len(a.children) != len(b.children):
            return (Hole(next_index), next_index + 1)
        children = []
        idx = next_index
        for ca, cb in zip(a.children, b.children):
            child, idx = anti_unify(ca, cb, idx)
            children.append(child)
        return (Composite(tuple(children)), idx)
    return (Hole(next_index), next_index + 1)

def pattern_serialize(shape: Shape) -> bytes:
    """Serialize a shape with holes (stable)."""
    if isinstance(shape, Atom):
        return b"()"
    if isinstance(shape, Stage):
        return b"(())"
    if isinstance(shape, Hole):
        return b"H" + str(shape.index).encode("ascii") + b";"
    if isinstance(shape, Composite):
        inner = b"".join(pattern_serialize(c) for c in shape.children)
        return b"(" + inner + b")"
    raise TypeError(f"Unknown: {type(shape)}")

def pattern_key(shape: Shape) -> str:
    """Hash key for a holey pattern."""
    return hashlib.sha256(pattern_serialize(shape)).hexdigest()

# ============================================================
# CATALOG ENTRY
# ============================================================

@dataclass
class CatalogEntry:
    canonical_bytes: bytes
    shape: Shape
    impl: Optional[Shape] = None
    witness: Optional[bytes] = None
    verilog: Optional[str] = None
    access_count: int = 0
    schema_key: Optional[str] = None
    equiv_key: Optional[str] = None
    equiv_status: Optional[str] = None

# ============================================================
# PERSISTENT CATALOG
# ============================================================

class PersistentCatalog:
    """Content-addressed shape storage with SQLite persistence."""

    def __init__(self, db_path: str = "asf_catalog.db"):
        self.db_path = Path(db_path)
        self._cache: Dict[str, CatalogEntry] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS shapes (
                key TEXT PRIMARY KEY,
                canonical_bytes BLOB NOT NULL,
                impl_key TEXT,
                witness BLOB,
                verilog TEXT,
                access_count INTEGER DEFAULT 0,
                schema_key TEXT,
                equiv_key TEXT,
                equiv_status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_access ON shapes(access_count DESC);
            CREATE INDEX IF NOT EXISTS idx_impl ON shapes(impl_key);
            CREATE INDEX IF NOT EXISTS idx_schema ON shapes(schema_key);
            CREATE INDEX IF NOT EXISTS idx_equiv ON shapes(equiv_key);
            CREATE TABLE IF NOT EXISTS spines (
                depth INTEGER NOT NULL,
                spine_key TEXT NOT NULL,
                shape_key TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (depth, spine_key, shape_key)
            );
            CREATE INDEX IF NOT EXISTS idx_spine_key ON spines(depth, spine_key);
            CREATE TABLE IF NOT EXISTS results (
                equiv_key TEXT PRIMARY KEY,
                value_text TEXT,
                value_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS labels (
                shape_key TEXT NOT NULL,
                namespace TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (shape_key, namespace, label)
            );
            CREATE INDEX IF NOT EXISTS idx_label ON labels(label);
        """)
        self._ensure_columns()
        self._conn.commit()

    def _ensure_columns(self):
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(shapes)")}
        if "schema_key" not in cols:
            self._conn.execute("ALTER TABLE shapes ADD COLUMN schema_key TEXT")
        if "equiv_key" not in cols:
            self._conn.execute("ALTER TABLE shapes ADD COLUMN equiv_key TEXT")
        if "equiv_status" not in cols:
            self._conn.execute("ALTER TABLE shapes ADD COLUMN equiv_status TEXT")

    def get(self, shape: Shape) -> Optional[CatalogEntry]:
        """Look up a shape by content."""
        k = key(shape)

        if k in self._cache:
            self._cache[k].access_count += 1
            self._touch(k)
            return self._cache[k]

        row = self._conn.execute(
            "SELECT * FROM shapes WHERE key = ?", (k,)
        ).fetchone()

        if not row:
            return None

        entry = self._row_to_entry(row)
        entry.access_count += 1
        self._touch(k)
        self._cache[k] = entry
        return entry

    def _row_to_entry(self, row) -> CatalogEntry:
        canonical_bytes = row["canonical_bytes"]
        shape = parse(canonical_bytes.decode())

        impl = None
        if row["impl_key"]:
            impl_row = self._conn.execute(
                "SELECT canonical_bytes FROM shapes WHERE key = ?",
                (row["impl_key"],)
            ).fetchone()
            if impl_row:
                impl = parse(impl_row["canonical_bytes"].decode())

        return CatalogEntry(
            canonical_bytes=canonical_bytes,
            shape=shape,
            impl=impl,
            witness=row["witness"],
            verilog=row["verilog"],
            access_count=row["access_count"],
            schema_key=row["schema_key"],
            equiv_key=row["equiv_key"],
            equiv_status=row["equiv_status"]
        )

    def _touch(self, k: str):
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE shapes SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
            (now, k)
        )
        self._conn.commit()

    def put(self, shape: Shape) -> CatalogEntry:
        """Store a shape (or return existing entry)."""
        k = key(shape)
        s_key = schema_key(shape)

        if k in self._cache:
            self._cache[k].access_count += 1
            self._touch(k)
            return self._cache[k]

        existing = self._conn.execute(
            "SELECT key FROM shapes WHERE key = ?", (k,)
        ).fetchone()

        if existing:
            return self.get(shape)

        canonical_bytes = serialize(shape)
        now = datetime.utcnow().isoformat()

        self._conn.execute(
            "INSERT INTO shapes (key, canonical_bytes, access_count, schema_key, created_at, last_accessed) VALUES (?, ?, 1, ?, ?, ?)",
            (k, canonical_bytes, s_key, now, now)
        )
        self._conn.commit()
        self._index_spines(k, shape)

        entry = CatalogEntry(canonical_bytes=canonical_bytes, shape=shape, access_count=1, schema_key=s_key)
        self._cache[k] = entry
        return entry

    def set_impl(self, contract: Shape, impl: Shape) -> CatalogEntry:
        """Record that impl is the normal form of contract."""
        self.put(contract)
        self.put(impl)

        ck, ik = key(contract), key(impl)
        self._conn.execute("UPDATE shapes SET impl_key = ? WHERE key = ?", (ik, ck))
        self._conn.commit()

        if ck in self._cache:
            self._cache[ck].impl = impl
        return self._cache.get(ck) or self.get(contract)

    def set_witness(self, shape: Shape, witness: bytes):
        """Store proof/transcript for a shape."""
        k = key(shape)
        self._conn.execute("UPDATE shapes SET witness = ? WHERE key = ?", (witness, k))
        self._conn.commit()
        if k in self._cache:
            self._cache[k].witness = witness

    def _index_spines(self, k: str, shape: Shape, depths: Tuple[int, ...] = (1, 2, 3, 4)):
        for depth in depths:
            s_key = spine_key(shape, depth)
            self._conn.execute(
                "INSERT OR IGNORE INTO spines (depth, spine_key, shape_key) VALUES (?, ?, ?)",
                (depth, s_key, k)
            )
        self._conn.commit()

    def set_equiv(self, shape: Shape, max_steps: int = 10000, mode: str = "beta") -> Tuple[str, str]:
        """Compute and store semantic equivalence key."""
        if mode == "beta":
            result, status = beta_normalize(shape, max_steps)
        elif mode == "beta_eta":
            result, status = semantic_normalize(shape, max_steps)
        else:
            raise ValueError(f"Unknown equiv mode: {mode}")
        eq_key = key(result) if status == "normal" else key(shape)
        k = key(shape)
        self._conn.execute(
            "UPDATE shapes SET equiv_key = ?, equiv_status = ? WHERE key = ?",
            (eq_key, status, k)
        )
        self._conn.commit()
        if k in self._cache:
            self._cache[k].equiv_key = eq_key
            self._cache[k].equiv_status = status
        return (eq_key, status)

    def set_equiv_key(self, shape: Shape, eq_key: str, status: str = "manual"):
        """Set semantic equivalence key explicitly."""
        k = key(shape)
        self._conn.execute(
            "UPDATE shapes SET equiv_key = ?, equiv_status = ? WHERE key = ?",
            (eq_key, status, k)
        )
        self._conn.commit()
        if k in self._cache:
            self._cache[k].equiv_key = eq_key
            self._cache[k].equiv_status = status

    def set_result_by_equiv(self, eq_key: str, value: Any):
        value_text = repr(value)
        value_type = type(value).__name__
        self._conn.execute(
            "INSERT OR REPLACE INTO results (equiv_key, value_text, value_type) VALUES (?, ?, ?)",
            (eq_key, value_text, value_type)
        )
        self._conn.commit()

    def get_result_by_equiv(self, eq_key: str) -> Optional[Any]:
        row = self._conn.execute(
            "SELECT value_text FROM results WHERE equiv_key = ?",
            (eq_key,)
        ).fetchone()
        if not row:
            return None
        try:
            return ast.literal_eval(row["value_text"])
        except Exception:
            return row["value_text"]

    def find_by_schema(self, s_key: str) -> List[CatalogEntry]:
        rows = self._conn.execute(
            "SELECT * FROM shapes WHERE schema_key = ?",
            (s_key,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def find_by_equiv(self, eq_key: str) -> List[CatalogEntry]:
        rows = self._conn.execute(
            "SELECT * FROM shapes WHERE equiv_key = ?",
            (eq_key,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def find_by_spine(self, shape: Shape, depth: int) -> List[CatalogEntry]:
        s_key = spine_key(shape, depth)
        rows = self._conn.execute(
            "SELECT shapes.* FROM shapes JOIN spines ON shapes.key = spines.shape_key WHERE spines.depth = ? AND spines.spine_key = ?",
            (depth, s_key)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def reindex_spines(self, depths: Tuple[int, ...] = (1, 2, 3, 4)):
        rows = self._conn.execute("SELECT key, canonical_bytes FROM shapes").fetchall()
        for row in rows:
            shape = parse(row["canonical_bytes"].decode())
            self._index_spines(row["key"], shape, depths=depths)

    def set_label(self, shape: Shape, label: str, namespace: str = "default", confidence: float = 1.0, notes: Optional[str] = None):
        """Attach a semantic label to a shape."""
        k = key(shape)
        self._conn.execute(
            "INSERT OR REPLACE INTO labels (shape_key, namespace, label, confidence, notes) VALUES (?, ?, ?, ?, ?)",
            (k, namespace, label, confidence, notes)
        )
        self._conn.commit()

    def get_labels(self, shape: Shape) -> List[dict]:
        k = key(shape)
        rows = self._conn.execute(
            "SELECT namespace, label, confidence, notes, created_at FROM labels WHERE shape_key = ?",
            (k,)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_label(self, label: str, namespace: Optional[str] = None) -> List[str]:
        if namespace is None:
            rows = self._conn.execute(
                "SELECT shape_key FROM labels WHERE label = ?",
                (label,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT shape_key FROM labels WHERE label = ? AND namespace = ?",
                (label, namespace)
            ).fetchall()
        return [r["shape_key"] for r in rows]

    def stats(self) -> dict:
        row = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(access_count) as accesses,
                   SUM(CASE WHEN impl_key IS NOT NULL THEN 1 ELSE 0 END) as impls
            FROM shapes
        """).fetchone()
        return {
            "total_entries": row["total"],
            "total_accesses": row["accesses"] or 0,
            "with_impl": row["impls"],
            "cache_size": len(self._cache)
        }

    def hottest(self, n: int = 10) -> list:
        rows = self._conn.execute(
            "SELECT key, canonical_bytes, access_count FROM shapes ORDER BY access_count DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [
            {"key": r["key"][:12] + "...", "shape": r["canonical_bytes"].decode()[:40], "accesses": r["access_count"]}
            for r in rows
        ]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __len__(self):
        return self._conn.execute("SELECT COUNT(*) FROM shapes").fetchone()[0]

# ============================================================
# MEMOIZED REDUCTION
# ============================================================

def beta_cached(shape: Shape, catalog: PersistentCatalog, max_steps: int = 10000) -> Tuple[Shape, str]:
    """Beta normalize with catalog memoization."""
    entry = catalog.get(shape)
    if entry and entry.impl is not None:
        return (entry.impl, "cached")

    result, status = beta_normalize(shape, max_steps)
    if status == "normal":
        catalog.set_impl(shape, result)
    return (result, status)

# ============================================================
# STANDARD COMBINATORS
# ============================================================

# I = λx. x
I = lam(ref(0))

# K = λx. λy. x
K = lam(lam(ref(1)))

# S = λx. λy. λz. x z (y z)
S = lam(lam(lam(app(app(ref(2), ref(0)), app(ref(1), ref(0))))))

# omega = λx. x x
omega = lam(app(ref(0), ref(0)))

# Omega = omega omega (non-terminating)
Omega = app(omega, omega)

# TRUE = K
TRUE = K

# FALSE = λx. λy. y
FALSE = lam(lam(ref(0)))

# ============================================================
# DEMO / TESTS
# ============================================================

def run_tests():
    print("ASF Core 2 Tests")
    print("=" * 60)

    # Test parsing round-trip
    print("\n1. Parsing round-trip:")
    test_strings = ["A", "S", "(AS)", "(A)", "(())", "((A)S)"]
    for s in test_strings:
        shape = parse(s)
        back = serialize(shape).decode()
        status = "ok" if s.replace(" ", "") == back else "FAIL"
        print(f"   [{status}] {s}")

    # Test lambda encoding
    print("\n2. Lambda encoding:")
    print(f"   I = {serialize(I).decode()}")
    print(f"   K = {serialize(K).decode()}")
    print(f"   S = {serialize(S).decode()}")

    # Test beta reduction
    print("\n3. Beta reduction:")

    # I a -> a
    test1 = app(I, Stage())
    r1, s1 = beta_normalize(test1)
    print(f"   I S -> {serialize(r1).decode()} [{s1}]")
    assert r1 == Stage()

    # I I -> I
    test2 = app(I, I)
    r2, s2 = beta_normalize(test2)
    print(f"   I I -> {serialize(r2).decode()} [{s2}]")
    assert r2 == I

    # K a b -> a
    test3 = app(app(K, Stage()), Atom())
    r3, s3 = beta_normalize(test3)
    print(f"   K S A -> {serialize(r3).decode()} [{s3}]")
    assert r3 == Stage()

    # S K K a -> a (S K K is identity)
    test4 = app(app(app(S, K), K), Stage())
    r4, s4 = beta_normalize(test4)
    print(f"   S K K S -> {serialize(r4).decode()} [{s4}]")
    assert r4 == Stage()

    # Test catalog
    print("\n4. Catalog persistence:")
    import os
    if os.path.exists("asf_test.db"):
        os.remove("asf_test.db")

    catalog = PersistentCatalog("asf_test.db")

    # Store some shapes
    for shape in [I, K, S, TRUE, FALSE]:
        catalog.put(shape)

    print("   Stored 5 combinators")
    print(f"   Stats: {catalog.stats()}")

    # Test cached reduction
    print("\n5. Cached reduction:")

    # First time: compute
    expr1 = app(app(K, Stage()), Atom())
    r1, status1 = beta_cached(expr1, catalog)
    print(f"   K S A -> {serialize(r1).decode()} [{status1}]")

    # Second time: cache hit
    r2, status2 = beta_cached(expr1, catalog)
    print(f"   K S A -> {serialize(r2).decode()} [{status2}]")

    # Test schema_key (structure-only)
    print("\n6. Schema key:")
    shape_a = Composite((Atom(), Stage(), Composite((Atom(), Atom()))))
    shape_b = Composite((Stage(), Atom(), Composite((Stage(), Stage()))))
    sk_a = schema_key(shape_a)
    sk_b = schema_key(shape_b)
    print(f"   schema_key(a): {sk_a[:16]}...")
    print(f"   schema_key(b): {sk_b[:16]}...")
    assert sk_a == sk_b

    # Test semantic equivalence key
    print("\n7. Semantic equivalence:")
    expr_sem = app(I, Stage())
    eq_key, eq_status = catalog.set_equiv(expr_sem, mode="beta_eta")
    print(f"   equiv status: {eq_status}")
    assert eq_key == key(Stage())

    # Test labels
    print("\n8. Labels:")
    catalog.set_label(I, "identity", namespace="lambda", confidence=0.9, notes="standard I combinator")
    labels = catalog.get_labels(I)
    print(f"   labels: {labels}")
    assert any(l["label"] == "identity" for l in labels)

    # Test eta-reduction
    print("\n9. Eta reduction:")
    eta_shape = lam(app(lam(ref(0)), ref(0)))
    eta_result, eta_status = eta_normalize(eta_shape)
    print(f"   eta status: {eta_status}")
    assert eta_result == lam(ref(0))

    # Test result cache
    print("\n10. Result cache:")
    catalog.set_result_by_equiv(eq_key, 123)
    cached = catalog.get_result_by_equiv(eq_key)
    print(f"   cached result: {cached}")
    assert cached == 123

    # Test anti-unify
    print("\n11. Anti-unify:")
    au_a = Composite((Atom(), Stage()))
    au_b = Composite((Stage(), Stage()))
    pattern, _ = anti_unify(au_a, au_b)
    print(f"   pattern: {pretty(pattern)}")
    assert isinstance(pattern, Composite)
    assert isinstance(pattern.children[0], Hole)

    # Test spine matching
    print("\n12. Spine matching:")
    shape_c = Composite((Atom(), Composite((Stage(), Atom()))))
    shape_d = Composite((Stage(), Composite((Atom(), Stage()))))
    catalog.put(shape_c)
    catalog.put(shape_d)
    matches = catalog.find_by_spine(shape_c, depth=2)
    print(f"   spine matches: {len(matches)}")
    assert len(matches) >= 2

    # Test alpha-equivalence (same shape = same key)
    print("\n13. Alpha equivalence:")
    id1 = lam(ref(0))  # "lambda x. x"
    id2 = lam(ref(0))  # "lambda y. y" (already canonical)
    print(f"   lam(ref(0)) key: {key(id1)[:16]}...")
    print(f"   lam(ref(0)) key: {key(id2)[:16]}...")
    print(f"   Same key: {key(id1) == key(id2)} [ok]")

    print(f"\n   Final catalog stats: {catalog.stats()}")
    print("   Hottest shapes:")
    for h in catalog.hottest(5):
        print(f"      {h['accesses']:3} hits: {h['shape']}")

    catalog.close()
    print("\n" + "=" * 60)
    print("All tests passed")

if __name__ == "__main__":
    run_tests()
