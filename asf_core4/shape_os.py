#!/usr/bin/env python3
"""
Shape OS - The Unified Runtime

Core Concepts:
1. The description IS the software (O(1) content-addressed)
2. Waveform propagation (clockless reduction to normal form)
3. Double-membrane (recognize → continue, else → outlier)
4. Structural intelligence (learns Bruce's satisfaction invariants)

This consolidates: py_to_dyck, structural_intelligence, waveform model,
and the speak-see-experience demo into one coherent runtime.
"""

import hashlib
import json
import time
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# Core shape runtime
from py_to_dyck import (
    compile_source, run_dyck, run_source,
    parse_dyck, serialize_dyck, Shape, Atom, Stage, Composite
)

# Quantum-style affordances
from weighted_state import from_shapes, normalize_l2, apply_operator
from operators import grover_step, mark
from sampling import probabilities, sample_counts


# =============================================================================
# WAVEFORM MODEL
# =============================================================================

@dataclass
class Waveform:
    """A waveform is a shape with propagation energy."""
    shape: Shape
    dyck: str
    energy: float
    phase: str  # 'ready', 'propagating', 'stable'
    source_hash: str
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dyck(cls, dyck: str) -> 'Waveform':
        shape = parse_dyck(dyck)
        energy = cls._compute_energy(shape)
        source_hash = hashlib.sha256(dyck.encode()).hexdigest()[:16]
        return cls(
            shape=shape,
            dyck=dyck,
            energy=energy,
            phase='ready',
            source_hash=source_hash
        )

    @classmethod
    def from_python(cls, source: str) -> 'Waveform':
        dyck = compile_source(source)
        return cls.from_dyck(dyck)

    @classmethod
    def from_description(cls, description: str) -> 'Waveform':
        """The description IS the software. Direct O(1) addressing."""
        desc_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
        # Simple structure from description length/complexity
        n_parens = max(1, len(description) // 10)
        dyck = "(" * n_parens + "()" * min(n_parens, 3) + ")" * n_parens
        shape = parse_dyck(dyck)
        return cls(
            shape=shape,
            dyck=dyck,
            energy=cls._compute_energy(shape),
            phase='ready',
            source_hash=desc_hash
        )

    @staticmethod
    def _compute_energy(shape: Shape) -> float:
        """Energy from structure depth/breadth."""
        if isinstance(shape, Atom):
            return 0.0
        if isinstance(shape, Stage):
            return 0.1
        if isinstance(shape, Composite):
            return 0.1 + sum(Waveform._compute_energy(c) for c in shape.children)
        return 0.0

    def propagate(self) -> bool:
        """Propagate waveform one step. Returns True if still active."""
        if self.phase == 'stable' or self.energy <= 0:
            self.phase = 'stable'
            return False
        self.phase = 'propagating'
        self.energy *= 0.9  # Decay
        if self.energy < 0.01:
            self.phase = 'stable'
            return False
        self.phase = 'ready'
        return True


# =============================================================================
# DOUBLE MEMBRANE
# =============================================================================

class DoubleMembrane:
    """
    Recognition filter:
    - Recognized shapes → continuation (pass through)
    - Unrecognized shapes → outlier artifact only
    """

    def __init__(self, db_path: str = "membrane.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recognized_shapes (
                shape_hash TEXT PRIMARY KEY,
                dyck TEXT,
                label TEXT,
                first_seen REAL,
                hit_count INTEGER DEFAULT 1
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS outliers (
                id INTEGER PRIMARY KEY,
                shape_hash TEXT,
                dyck TEXT,
                reason TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def recognize(self, waveform: Waveform) -> Optional[str]:
        """Attempt to recognize a waveform. Returns label if recognized."""
        row = self.conn.execute(
            "SELECT label, hit_count FROM recognized_shapes WHERE shape_hash = ?",
            (waveform.source_hash,)
        ).fetchone()
        if row:
            # Update hit count
            self.conn.execute(
                "UPDATE recognized_shapes SET hit_count = hit_count + 1 WHERE shape_hash = ?",
                (waveform.source_hash,)
            )
            self.conn.commit()
            return row['label']
        return None

    def engulf(self, waveform: Waveform, label: Optional[str] = None) -> dict:
        """
        Engulf a waveform through the membrane.
        Returns receipt with recognition status.
        """
        recognized = self.recognize(waveform)
        if recognized:
            return {
                "status": "continuation",
                "label": recognized,
                "shape_hash": waveform.source_hash
            }
        else:
            # Record as outlier
            self.conn.execute(
                "INSERT INTO outliers (shape_hash, dyck, reason, timestamp) VALUES (?, ?, ?, ?)",
                (waveform.source_hash, waveform.dyck, "unrecognized", time.time())
            )
            self.conn.commit()
            return {
                "status": "outlier",
                "shape_hash": waveform.source_hash,
                "reason": "unrecognized"
            }

    def teach(self, waveform: Waveform, label: str):
        """Teach the membrane to recognize a shape."""
        self.conn.execute("""
            INSERT OR REPLACE INTO recognized_shapes (shape_hash, dyck, label, first_seen, hit_count)
            VALUES (?, ?, ?, ?, 1)
        """, (waveform.source_hash, waveform.dyck, label, time.time()))
        self.conn.commit()

    def close(self):
        self.conn.close()


# =============================================================================
# STRUCTURAL INTELLIGENCE (SIMPLIFIED)
# =============================================================================

@dataclass
class SatisfactionSignal:
    """A satisfaction signal from Bruce."""
    utterance: str
    shape_hash: str
    satisfaction: float
    feedback: str
    timestamp: float = field(default_factory=time.time)


class StructuralIntelligenceLite:
    """
    Learns satisfaction-shape invariants for Bruce.
    Simplified from full SI for clean integration.
    """

    def __init__(self, db_path: str = "si_lite.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.invariants: Dict[str, List[float]] = {}
        self._load_invariants()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                utterance TEXT,
                shape_hash TEXT,
                satisfaction REAL,
                feedback TEXT,
                timestamp REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS invariants (
                key TEXT PRIMARY KEY,
                samples TEXT,
                mean REAL
            )
        """)
        self.conn.commit()

    def _load_invariants(self):
        for row in self.conn.execute("SELECT key, samples FROM invariants"):
            self.invariants[row['key']] = json.loads(row['samples'])

    def _save_invariant(self, key: str, samples: List[float]):
        mean = sum(samples) / len(samples) if samples else 0.5
        self.conn.execute(
            "INSERT OR REPLACE INTO invariants (key, samples, mean) VALUES (?, ?, ?)",
            (key, json.dumps(samples[-100:]), mean)  # Keep last 100
        )
        self.conn.commit()

    def record(self, signal: SatisfactionSignal):
        """Record a satisfaction signal and update invariants."""
        self.conn.execute(
            "INSERT INTO signals (utterance, shape_hash, satisfaction, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (signal.utterance, signal.shape_hash, signal.satisfaction, signal.feedback, signal.timestamp)
        )
        self.conn.commit()

        # Update shape invariant
        key = f"shape:{signal.shape_hash}"
        if key not in self.invariants:
            self.invariants[key] = []
        self.invariants[key].append(signal.satisfaction)
        self._save_invariant(key, self.invariants[key])

        # Update keyword invariants
        for word in self._extract_keywords(signal.utterance):
            key = f"kw:{word}"
            if key not in self.invariants:
                self.invariants[key] = []
            self.invariants[key].append(signal.satisfaction)
            self._save_invariant(key, self.invariants[key])

    def predict(self, shape_hash: str, utterance: str = "") -> Tuple[float, float]:
        """Predict satisfaction. Returns (prediction, confidence)."""
        samples = []

        # Shape-based
        key = f"shape:{shape_hash}"
        if key in self.invariants and self.invariants[key]:
            samples.extend(self.invariants[key])

        # Keyword-based
        for word in self._extract_keywords(utterance):
            key = f"kw:{word}"
            if key in self.invariants and self.invariants[key]:
                samples.extend(self.invariants[key])

        if not samples:
            return 0.5, 0.0

        mean = sum(samples) / len(samples)
        conf = min(0.95, len(samples) / 20.0)
        return mean, conf

    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {'a', 'an', 'the', 'is', 'are', 'i', 'want', 'need', 'to', 'that', 'with', 'for'}
        return [w for w in text.lower().split() if w not in stopwords and len(w) > 2]

    def close(self):
        self.conn.close()


# =============================================================================
# SHAPE OS RUNTIME
# =============================================================================

class ShapeOS:
    """
    The unified Shape OS runtime.

    Combines:
    - Waveform propagation (clockless reduction)
    - Double membrane (recognition filter)
    - Structural intelligence (satisfaction learning)
    - NL → Shape direct path
    """

    def __init__(self, db_prefix: str = "shape_os"):
        self.membrane = DoubleMembrane(f"{db_prefix}_membrane.db")
        self.si = StructuralIntelligenceLite(f"{db_prefix}_si.db")
        self.active_waves: Dict[str, Waveform] = {}

    def speak(self, description: str) -> Tuple[Waveform, dict]:
        """
        Speak it → See it.
        Creates a waveform from description.
        """
        wave = Waveform.from_description(description)
        self.active_waves[wave.source_hash] = wave
        receipt = self.membrane.engulf(wave)
        return wave, receipt

    def compile(self, python_source: str) -> Tuple[Waveform, dict]:
        """Compile Python to waveform."""
        wave = Waveform.from_python(python_source)
        self.active_waves[wave.source_hash] = wave
        receipt = self.membrane.engulf(wave)
        return wave, receipt

    def run(self, waveform: Waveform) -> Any:
        """Run a waveform to completion."""
        while waveform.propagate():
            pass
        return run_dyck(waveform.dyck)

    def experience(self, wave_hash: str, satisfaction: float, feedback: str = ""):
        """
        Experience it → Learn.
        Record satisfaction for a waveform.
        """
        wave = self.active_waves.get(wave_hash)
        if not wave:
            return

        signal = SatisfactionSignal(
            utterance=wave_hash,  # We could store original utterance
            shape_hash=wave.source_hash,
            satisfaction=satisfaction,
            feedback=feedback
        )
        self.si.record(signal)

        # High satisfaction → teach membrane
        if satisfaction >= 0.8:
            self.membrane.teach(wave, feedback or wave_hash)

    def search(self, query: str, candidates: List[str]) -> Dict[str, float]:
        """Quantum-style search over candidates."""
        state = from_shapes(candidates, weight=1.0)
        state = normalize_l2(state)

        # Mark high-satisfaction candidates
        def oracle(shape_key):
            pred, conf = self.si.predict(shape_key, query)
            return pred > 0.6 and conf > 0.2

        state = apply_operator(state, mark(oracle))

        # Grover step
        predictions = {k: self.si.predict(k, query)[0] for k in state}
        if predictions:
            best = max(predictions, key=predictions.get)
            state = grover_step(state, best)

        return probabilities(state)

    def status(self) -> dict:
        """Get system status."""
        trend = []
        for row in self.si.conn.execute(
            "SELECT satisfaction FROM signals ORDER BY timestamp DESC LIMIT 20"
        ):
            trend.append(row['satisfaction'])

        return {
            "active_waves": len(self.active_waves),
            "invariants": len(self.si.invariants),
            "satisfaction_trend": trend[::-1],
            "mean_satisfaction": sum(trend) / len(trend) if trend else 0.5
        }

    def close(self):
        self.membrane.close()
        self.si.close()


# =============================================================================
# DEMO: SPEAK-SEE-EXPERIENCE
# =============================================================================

def demo():
    """Run the speak-see-experience demo."""
    os = ShapeOS(db_prefix=":memory:")

    print("\n" + "=" * 60)
    print("  SHAPE OS: SPEAK IT, SEE IT, EXPERIENCE IT")
    print("=" * 60)
    print("""
  The description IS the software.
  Waveforms propagate without a clock.
  The membrane recognizes or flags outliers.
  You teach me what delights you.

  Commands:
    speak <description>   - Create a waveform from words
    compile <code>        - Compile Python to waveform
    rate <hash> <0-10>    - Rate your experience
    search <query>        - Quantum search
    status                - System status
    quit                  - Exit
""")

    while True:
        try:
            line = input("\nshape> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ('quit', 'exit', 'q'):
            break

        elif cmd == 'speak':
            if not arg:
                print("  Usage: speak <description>")
                continue
            wave, receipt = os.speak(arg)
            print(f"\n  Waveform created:")
            print(f"    Hash:   {wave.source_hash}")
            print(f"    Dyck:   {wave.dyck}")
            print(f"    Energy: {wave.energy:.2f}")
            print(f"    Status: {receipt['status']}")
            if receipt['status'] == 'continuation':
                print(f"    Label:  {receipt['label']}")

        elif cmd == 'compile':
            if not arg:
                print("  Usage: compile <python_code>")
                continue
            try:
                wave, receipt = os.compile(arg.replace('\\n', '\n'))
                print(f"\n  Compiled:")
                print(f"    Hash:   {wave.source_hash}")
                print(f"    Dyck:   {wave.dyck[:60]}...")
                print(f"    Status: {receipt['status']}")
            except Exception as e:
                print(f"  Error: {e}")

        elif cmd == 'run':
            if not arg or arg not in os.active_waves:
                print(f"  Usage: run <hash>")
                print(f"  Active: {list(os.active_waves.keys())}")
                continue
            wave = os.active_waves[arg]
            try:
                result = os.run(wave)
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  Error: {e}")

        elif cmd == 'rate':
            parts = arg.split()
            if len(parts) < 2:
                print("  Usage: rate <hash> <0-10> [feedback]")
                continue
            wave_hash, score_str = parts[0], parts[1]
            feedback = " ".join(parts[2:]) if len(parts) > 2 else ""
            try:
                score = float(score_str) / 10.0
                os.experience(wave_hash, score, feedback)
                print(f"  Recorded: {score:.0%} satisfaction")
            except ValueError:
                print("  Invalid score")

        elif cmd == 'search':
            if not arg:
                print("  Usage: search <query>")
                continue
            # Use active waves as candidates
            candidates = list(os.active_waves.keys())
            if not candidates:
                print("  No active waveforms to search")
                continue
            results = os.search(arg, candidates)
            print("\n  Search results:")
            for k, p in sorted(results.items(), key=lambda x: -x[1])[:5]:
                print(f"    {p:.1%} | {k}")

        elif cmd == 'status':
            status = os.status()
            print(f"\n  Active waves:  {status['active_waves']}")
            print(f"  Invariants:    {status['invariants']}")
            print(f"  Mean sat:      {status['mean_satisfaction']:.0%}")
            if status['satisfaction_trend']:
                trend = status['satisfaction_trend']
                spark = ''.join(['▁▂▃▄▅▆▇█'[int(s*7)] for s in trend])
                print(f"  Trend:         {spark}")

        else:
            print(f"  Unknown command: {cmd}")
            print("  Try: speak, compile, run, rate, search, status, quit")

    print("\n  Goodbye.")
    os.close()


if __name__ == "__main__":
    demo()
