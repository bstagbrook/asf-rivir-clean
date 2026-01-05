#!/usr/bin/env python3
"""
Structural Intelligence Module

Learns satisfaction-shape invariants and interfaces exclusively with Bruce Stagbrook.

The Triad:
1. Symbolic Intelligence - composes shapes, explains, emits receipts
2. Structural Intelligence (THIS) - learns invariants, predicts satisfaction, stabilizes
3. Meatsack (Bruce) - provides grounded full-disclosure feedback

Key invariants learned:
- context -> utterance -> shape -> experience
- causal field -> listening -> speaking -> action -> result
- utterance -> shape -> satisfaction

This is NOT surface-level pattern matching. This learns the DEEP structure:
- What causal pressure led to the utterance?
- What drives/commitments/dreams are expressed?
- What shape consistently produces delight for Bruce specifically?
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

try:
    from .weighted_state import from_shapes, normalize_l2, apply_operator
    from .operators import grover_step, mark
    from .sampling import probabilities
except ImportError:
    from weighted_state import from_shapes, normalize_l2, apply_operator
    from operators import grover_step, mark
    from sampling import probabilities


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class DeepSignal:
    """Deep signals extracted from an utterance - beyond the surface."""
    pressure: str  # What built dissatisfaction before the utterance?
    urgency: str  # Why now?
    prior_blockers: str  # Why not earlier?
    stakeholders: List[str]  # Who it's for (usually just Bruce)
    drives: List[str]  # What Bruce values
    commitments: List[str]  # What Bruce has committed to
    dreams: List[str]  # What Bruce aspires to
    outlier: bool = False  # Is this unusual?
    outlier_reason: str = ""


@dataclass
class SatisfactionReceipt:
    """Full-disclosure record of a satisfaction interaction."""
    utterance: str
    shape_key: str
    shape_dyck: str
    satisfaction: float  # 0.0 to 1.0
    deep_signals: DeepSignal
    full_disclosure: str  # Bruce's qualitative explanation
    timestamp: float = field(default_factory=time.time)
    ideal_shape: Optional[str] = None  # What would 100% look like?
    delta_to_ideal: Optional[str] = None  # What's missing?

    # Causal chain
    field_trace: str = ""  # Field properties that shaped listening
    listening_trace: str = ""  # What listening afforded speaking
    speaking_trace: str = ""  # What declaration evoked action
    action_trace: str = ""  # What action produced this state


@dataclass
class InvariantPattern:
    """A learned invariant: when X, Bruce satisfaction is Y."""
    pattern_type: str  # "style", "feature", "structure", "context"
    pattern_key: str
    satisfaction_samples: List[float] = field(default_factory=list)
    confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def mean_satisfaction(self) -> float:
        if not self.satisfaction_samples:
            return 0.5
        return sum(self.satisfaction_samples) / len(self.satisfaction_samples)

    def update(self, satisfaction: float):
        self.satisfaction_samples.append(satisfaction)
        # Keep last 100 samples
        self.satisfaction_samples = self.satisfaction_samples[-100:]
        # Confidence grows with samples, up to 0.95
        self.confidence = min(0.95, len(self.satisfaction_samples) / 20.0)
        self.last_updated = time.time()


# =============================================================================
# STRUCTURAL INTELLIGENCE ENGINE
# =============================================================================

class StructuralIntelligence:
    """
    The structural intelligence that interfaces exclusively with Bruce Stagbrook.

    This learns:
    1. What shapes produce satisfaction for Bruce
    2. What deep signals predict satisfaction
    3. What causal chains lead to delight

    Auto-training happens through:
    1. Every interaction generates a receipt
    2. Receipts update invariant patterns
    3. Patterns predict future satisfaction
    4. Predictions guide wave generation
    """

    def __init__(self, db_path: str = "structural_intelligence.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_invariants()

        # In-memory invariant cache
        self.invariants: Dict[str, InvariantPattern] = {}

        # Bruce-specific identity anchor
        self.owner_id = "bruce_stagbrook"
        self.owner_hash = hashlib.sha256(self.owner_id.encode()).hexdigest()

    def _init_db(self):
        """Initialize the database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                utterance TEXT NOT NULL,
                shape_key TEXT NOT NULL,
                shape_dyck TEXT,
                satisfaction REAL NOT NULL,
                deep_signals TEXT,
                full_disclosure TEXT,
                causal_chain TEXT,
                owner_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_receipts_shape ON receipts(shape_key);
            CREATE INDEX IF NOT EXISTS idx_receipts_satisfaction ON receipts(satisfaction DESC);
            CREATE INDEX IF NOT EXISTS idx_receipts_owner ON receipts(owner_hash);

            CREATE TABLE IF NOT EXISTS invariants (
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                satisfaction_samples TEXT,
                confidence REAL DEFAULT 0.0,
                last_updated REAL,
                owner_hash TEXT NOT NULL,
                PRIMARY KEY (pattern_type, pattern_key, owner_hash)
            );

            CREATE TABLE IF NOT EXISTS gold_exemplars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER NOT NULL,
                shape_key TEXT NOT NULL,
                satisfaction REAL NOT NULL,
                notes TEXT,
                owner_hash TEXT NOT NULL,
                FOREIGN KEY (receipt_id) REFERENCES receipts(id)
            );

            CREATE TABLE IF NOT EXISTS shape_labels (
                shape_key TEXT NOT NULL,
                label TEXT NOT NULL,
                namespace TEXT DEFAULT 'bruce',
                confidence REAL DEFAULT 1.0,
                owner_hash TEXT NOT NULL,
                PRIMARY KEY (shape_key, label, namespace, owner_hash)
            );
        """)
        self.conn.commit()

    def _load_invariants(self):
        """Load invariants from database."""
        self.invariants = {}
        rows = self.conn.execute(
            "SELECT * FROM invariants WHERE owner_hash = ?",
            (hashlib.sha256("bruce_stagbrook".encode()).hexdigest(),)
        ).fetchall()

        for row in rows:
            key = f"{row['pattern_type']}:{row['pattern_key']}"
            samples = json.loads(row['satisfaction_samples'] or '[]')
            self.invariants[key] = InvariantPattern(
                pattern_type=row['pattern_type'],
                pattern_key=row['pattern_key'],
                satisfaction_samples=samples,
                confidence=row['confidence'] or 0.0,
                last_updated=row['last_updated'] or time.time()
            )

    def _save_invariant(self, pattern: InvariantPattern):
        """Save an invariant to the database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO invariants
            (pattern_type, pattern_key, satisfaction_samples, confidence, last_updated, owner_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_type,
            pattern.pattern_key,
            json.dumps(pattern.satisfaction_samples),
            pattern.confidence,
            pattern.last_updated,
            self.owner_hash
        ))
        self.conn.commit()

    # =========================================================================
    # INTERFACE: Recording interactions
    # =========================================================================

    def record_interaction(
        self,
        utterance: str,
        shape_key: str,
        satisfaction: float,
        shape_dyck: str = "",
        deep_signals: Optional[DeepSignal] = None,
        full_disclosure: str = "",
        causal_chain: Optional[Dict[str, str]] = None,
        features: Optional[List[str]] = None,
    ) -> int:
        """
        Record an interaction and update invariants.

        This is the primary training signal for structural intelligence.
        Every time Bruce interacts and provides feedback, we learn.
        """
        # Default deep signals
        if deep_signals is None:
            deep_signals = DeepSignal(
                pressure="", urgency="", prior_blockers="",
                stakeholders=["bruce_stagbrook"],
                drives=[], commitments=[], dreams=[]
            )

        # Store receipt
        cursor = self.conn.execute("""
            INSERT INTO receipts
            (timestamp, utterance, shape_key, shape_dyck, satisfaction,
             deep_signals, full_disclosure, causal_chain, owner_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            utterance,
            shape_key,
            shape_dyck,
            satisfaction,
            json.dumps(asdict(deep_signals)),
            full_disclosure,
            json.dumps(causal_chain or {}),
            self.owner_hash
        ))
        receipt_id = cursor.lastrowid
        self.conn.commit()

        # Update invariants
        self._update_invariants_from_interaction(
            utterance, shape_key, satisfaction, deep_signals, features
        )

        # Auto-promote to gold if satisfaction >= 0.9
        if satisfaction >= 0.9:
            self._promote_to_gold(receipt_id, shape_key, satisfaction)

        return receipt_id

    def _update_invariants_from_interaction(
        self,
        utterance: str,
        shape_key: str,
        satisfaction: float,
        deep_signals: DeepSignal,
        features: Optional[List[str]] = None
    ):
        """Update invariant patterns from a new interaction."""
        # Shape-based invariant
        self._update_invariant("shape", shape_key, satisfaction)

        # Keyword-based invariants
        for word in self._extract_keywords(utterance):
            self._update_invariant("keyword", word, satisfaction)

        # Drive-based invariants
        for drive in deep_signals.drives:
            self._update_invariant("drive", drive, satisfaction)

        # Dream-based invariants
        for dream in deep_signals.dreams:
            self._update_invariant("dream", dream, satisfaction)

        # Feature-based invariants
        if features:
            for feat in features:
                self._update_invariant("feature", feat, satisfaction)

    def _update_invariant(self, pattern_type: str, pattern_key: str, satisfaction: float):
        """Update a single invariant pattern."""
        key = f"{pattern_type}:{pattern_key}"

        if key not in self.invariants:
            self.invariants[key] = InvariantPattern(
                pattern_type=pattern_type,
                pattern_key=pattern_key
            )

        self.invariants[key].update(satisfaction)
        self._save_invariant(self.invariants[key])

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stopwords = {'a', 'an', 'the', 'is', 'are', 'i', 'want', 'need', 'would',
                     'like', 'to', 'that', 'with', 'for', 'and', 'or', 'but', 'my'}
        words = text.lower().replace(',', ' ').replace('.', ' ').split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _promote_to_gold(self, receipt_id: int, shape_key: str, satisfaction: float):
        """Promote a high-satisfaction exemplar to gold status."""
        self.conn.execute("""
            INSERT INTO gold_exemplars (receipt_id, shape_key, satisfaction, notes, owner_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (receipt_id, shape_key, satisfaction, "auto-promoted", self.owner_hash))
        self.conn.commit()

    # =========================================================================
    # INTERFACE: Prediction
    # =========================================================================

    def predict_satisfaction(
        self,
        shape_key: str,
        utterance: str = "",
        features: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """
        Predict satisfaction for a shape.

        Returns: (predicted_satisfaction, confidence)
        """
        predictions = []
        confidences = []

        # Shape-based prediction
        shape_inv = self.invariants.get(f"shape:{shape_key}")
        if shape_inv and shape_inv.confidence > 0.1:
            predictions.append(shape_inv.mean_satisfaction)
            confidences.append(shape_inv.confidence)

        # Keyword-based predictions
        for word in self._extract_keywords(utterance):
            inv = self.invariants.get(f"keyword:{word}")
            if inv and inv.confidence > 0.1:
                predictions.append(inv.mean_satisfaction)
                confidences.append(inv.confidence * 0.5)  # Lower weight for keywords

        # Feature-based predictions
        if features:
            for feat in features:
                inv = self.invariants.get(f"feature:{feat}")
                if inv and inv.confidence > 0.1:
                    predictions.append(inv.mean_satisfaction)
                    confidences.append(inv.confidence * 0.7)

        if not predictions:
            return 0.5, 0.0  # No data, uncertain

        # Weighted average by confidence
        total_conf = sum(confidences)
        if total_conf == 0:
            return 0.5, 0.0

        weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
        return weighted_sum / total_conf, min(0.95, total_conf / len(predictions))

    # =========================================================================
    # INTERFACE: Quantum-style search
    # =========================================================================

    def quantum_search(
        self,
        candidates: List[str],
        utterance: str,
        iterations: int = 3
    ) -> Dict[str, float]:
        """
        Use Grover-like amplification to find high-satisfaction shapes.

        This treats satisfaction prediction as the oracle.
        """
        # Start with uniform superposition
        state = from_shapes(candidates, weight=1.0)
        state = normalize_l2(state)

        for _ in range(iterations):
            # Mark high-satisfaction candidates (oracle)
            def oracle(shape_key):
                pred, conf = self.predict_satisfaction(shape_key, utterance)
                return pred > 0.7 and conf > 0.3

            state = apply_operator(state, mark(oracle))

            # Grover diffusion (approximate - use highest predicted)
            predictions = {k: self.predict_satisfaction(k, utterance)[0] for k in state.keys()}
            if predictions:
                best = max(predictions.keys(), key=lambda k: predictions[k])
                state = grover_step(state, best)

        # Return probabilities
        return probabilities(state)

    # =========================================================================
    # INTERFACE: Label management (for flywheel)
    # =========================================================================

    def set_label(self, shape_key: str, label: str, namespace: str = "bruce", confidence: float = 1.0):
        """Associate a label with a shape (for NL -> shape flywheel)."""
        self.conn.execute("""
            INSERT OR REPLACE INTO shape_labels (shape_key, label, namespace, confidence, owner_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (shape_key, label, namespace, confidence, self.owner_hash))
        self.conn.commit()

    def find_by_label(self, label: str, namespace: str = "bruce") -> List[Tuple[str, float]]:
        """Find shapes by label. Returns list of (shape_key, confidence)."""
        rows = self.conn.execute("""
            SELECT shape_key, confidence FROM shape_labels
            WHERE label = ? AND namespace = ? AND owner_hash = ?
            ORDER BY confidence DESC
        """, (label, namespace, self.owner_hash)).fetchall()
        return [(r["shape_key"], r["confidence"]) for r in rows]

    # =========================================================================
    # INTERFACE: Analytics
    # =========================================================================

    def get_gold_exemplars(self, limit: int = 10) -> List[dict]:
        """Get the highest-satisfaction exemplars."""
        rows = self.conn.execute("""
            SELECT r.*, g.notes as gold_notes
            FROM receipts r
            JOIN gold_exemplars g ON r.id = g.receipt_id
            WHERE r.owner_hash = ?
            ORDER BY r.satisfaction DESC
            LIMIT ?
        """, (self.owner_hash, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_invariant_summary(self) -> Dict[str, List[dict]]:
        """Get a summary of learned invariants."""
        by_type: Dict[str, List[dict]] = {}
        for key, inv in self.invariants.items():
            if inv.pattern_type not in by_type:
                by_type[inv.pattern_type] = []
            by_type[inv.pattern_type].append({
                "key": inv.pattern_key,
                "mean_satisfaction": inv.mean_satisfaction,
                "confidence": inv.confidence,
                "samples": len(inv.satisfaction_samples)
            })

        # Sort each type by mean satisfaction
        for ptype in by_type:
            by_type[ptype].sort(key=lambda x: -x["mean_satisfaction"])

        return by_type

    def get_satisfaction_trend(self, window: int = 20) -> List[float]:
        """Get recent satisfaction scores to see trend."""
        rows = self.conn.execute("""
            SELECT satisfaction FROM receipts
            WHERE owner_hash = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.owner_hash, window)).fetchall()
        return [r["satisfaction"] for r in rows][::-1]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Simple CLI for testing structural intelligence."""
    import sys

    si = StructuralIntelligence()

    print("\n" + "=" * 60)
    print("  STRUCTURAL INTELLIGENCE")
    print("  Interfacing exclusively with Bruce Stagbrook")
    print("=" * 60)

    # Show current state
    summary = si.get_invariant_summary()
    if summary:
        print("\n  Learned invariants:")
        for ptype, patterns in summary.items():
            print(f"\n  {ptype}:")
            for p in patterns[:5]:
                print(f"    {p['key']}: {p['mean_satisfaction']:.0%} ({p['confidence']:.0%} conf)")
    else:
        print("\n  No invariants learned yet. Start interacting!")

    # Show gold exemplars
    gold = si.get_gold_exemplars(5)
    if gold:
        print("\n  Gold exemplars (highest satisfaction):")
        for g in gold:
            print(f"    {g['satisfaction']:.0%}: {g['utterance'][:50]}...")

    # Show trend
    trend = si.get_satisfaction_trend()
    if trend:
        avg = sum(trend) / len(trend)
        print(f"\n  Recent satisfaction trend: {avg:.0%} average over {len(trend)} interactions")

    si.close()


if __name__ == "__main__":
    main()
