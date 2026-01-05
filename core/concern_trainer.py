#!/usr/bin/env python3
"""
Concern-Based Structural Intelligence Training

The concern update contract is the PERFECT training signal:

When a hat updates a concern, it MUST provide:
- reason: why the update is needed
- state_trace: recent state changes leading up to the update
- delta: explicit diff between current state and 100%-satisfaction state
- ideal_shape: the shape that represents 100% satisfaction
- action_trace: the action that produced the latest state shift
- speaking_trace: the declaration that evoked the action
- listening_trace: the listening that afforded the speaking
- field_trace: causal field attributes that shaped the listening

This is the complete causal chain:
  field → listening → speaking → action → state → delta → ideal

Every concern update teaches:
1. What field properties enable what listening
2. What listening enables what speaking
3. What speaking evokes what action
4. What action produces what state
5. What the gap is between current and ideal
6. What the ideal shape looks like

This is NOT surface pattern matching. This is learning the deep structure
of causality from field to satisfaction.

And because shapes are content-addressed with O(1) lookup,
the "have I seen this before?" check is a hash lookup, not a search.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from .structural_intelligence import StructuralIntelligence
except ImportError:
    from structural_intelligence import StructuralIntelligence


# =============================================================================
# CONCERN UPDATE (The Perfect Training Signal)
# =============================================================================

@dataclass
class ConcernUpdate:
    """
    The complete causal chain from field to satisfaction.

    This is the REQUIRED contract when a hat updates a concern.
    Every field is mandatory - this enforces complete causal tracing.
    """
    # The concern being updated
    hat_name: str
    concern_name: str
    old_satisfaction: float
    new_satisfaction: float

    # WHY (the reason layer)
    reason: str  # Why the update is needed

    # WHAT (the state layer)
    state_trace: str  # Recent state changes leading up to update
    delta: str  # Explicit diff between current and 100%-satisfaction
    ideal_shape: str  # The shape key that represents 100% satisfaction

    # HOW (the causal chain)
    action_trace: str  # The action that produced the latest state shift
    speaking_trace: str  # The declaration that evoked the action
    listening_trace: str  # The listening that afforded the speaking
    field_trace: str  # Causal field attributes that shaped the listening

    # META
    timestamp: float = field(default_factory=time.time)
    outlier_flag: bool = False
    outlier_reason: str = ""

    def causal_chain_hash(self) -> str:
        """Hash the causal chain for fast lookup."""
        chain = f"{self.field_trace}|{self.listening_trace}|{self.speaking_trace}|{self.action_trace}"
        return hashlib.sha256(chain.encode()).hexdigest()

    def delta_hash(self) -> str:
        """Hash the delta (gap) for pattern matching."""
        return hashlib.sha256(self.delta.encode()).hexdigest()


# =============================================================================
# CAUSAL PATTERN (What we learn from concern updates)
# =============================================================================

@dataclass
class CausalPattern:
    """
    A learned pattern: when this causal chain happens, this satisfaction results.

    The key insight: we're not learning surface text patterns.
    We're learning the deep structure of:
    - What field properties enable what listening
    - What listening enables what speaking
    - etc.
    """
    pattern_type: str  # "field_listening", "listening_speaking", "speaking_action", "action_state", "delta_ideal"
    from_trace: str
    to_trace: str
    satisfaction_samples: List[float] = field(default_factory=list)
    frequency: int = 0
    last_seen: float = field(default_factory=time.time)

    @property
    def mean_satisfaction(self) -> float:
        if not self.satisfaction_samples:
            return 0.5
        return sum(self.satisfaction_samples) / len(self.satisfaction_samples)

    def pattern_key(self) -> str:
        """Unique key for this pattern."""
        return hashlib.sha256(f"{self.pattern_type}:{self.from_trace}:{self.to_trace}".encode()).hexdigest()[:16]


# =============================================================================
# CONCERN TRAINER
# =============================================================================

class ConcernTrainer:
    """
    Trains structural intelligence from concern updates.

    The training loop:
    1. Hat updates a concern (with full causal chain)
    2. We extract causal patterns from the chain
    3. We update pattern statistics
    4. We learn what field→listening→speaking→action→state chains produce satisfaction

    This enables:
    - O(1) "have I seen this before?" via hash lookup
    - Prediction of satisfaction from partial causal chains
    - Discovery of which field properties enable which outcomes
    """

    def __init__(self, db_path: str = "concern_training.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        # In-memory pattern cache for O(1) lookup
        self.pattern_cache: Dict[str, CausalPattern] = {}
        self._load_patterns()

    def _init_db(self):
        """Initialize the database."""
        self.conn.executescript("""
            -- Raw concern updates (the training signal)
            CREATE TABLE IF NOT EXISTS concern_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                hat_name TEXT NOT NULL,
                concern_name TEXT NOT NULL,
                old_satisfaction REAL,
                new_satisfaction REAL NOT NULL,
                reason TEXT,
                state_trace TEXT,
                delta TEXT,
                ideal_shape TEXT,
                action_trace TEXT,
                speaking_trace TEXT,
                listening_trace TEXT,
                field_trace TEXT,
                causal_chain_hash TEXT,
                delta_hash TEXT,
                outlier_flag INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_concern_chain ON concern_updates(causal_chain_hash);
            CREATE INDEX IF NOT EXISTS idx_concern_delta ON concern_updates(delta_hash);
            CREATE INDEX IF NOT EXISTS idx_concern_ideal ON concern_updates(ideal_shape);

            -- Learned causal patterns
            CREATE TABLE IF NOT EXISTS causal_patterns (
                pattern_key TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                from_trace TEXT NOT NULL,
                to_trace TEXT NOT NULL,
                satisfaction_samples TEXT,
                frequency INTEGER DEFAULT 0,
                last_seen REAL
            );

            -- Ideal shapes (what 100% satisfaction looks like)
            CREATE TABLE IF NOT EXISTS ideal_shapes (
                shape_key TEXT PRIMARY KEY,
                concern_name TEXT NOT NULL,
                frequency INTEGER DEFAULT 0,
                avg_satisfaction REAL DEFAULT 1.0,
                examples TEXT
            );

            -- Delta patterns (common gaps)
            CREATE TABLE IF NOT EXISTS delta_patterns (
                delta_hash TEXT PRIMARY KEY,
                delta_text TEXT NOT NULL,
                frequency INTEGER DEFAULT 0,
                resolution_shape TEXT,
                resolution_success_rate REAL
            );
        """)
        self.conn.commit()

    def _load_patterns(self):
        """Load patterns into memory for O(1) lookup."""
        rows = self.conn.execute("SELECT * FROM causal_patterns").fetchall()
        for row in rows:
            pattern = CausalPattern(
                pattern_type=row["pattern_type"],
                from_trace=row["from_trace"],
                to_trace=row["to_trace"],
                satisfaction_samples=json.loads(row["satisfaction_samples"] or "[]"),
                frequency=row["frequency"],
                last_seen=row["last_seen"]
            )
            self.pattern_cache[pattern.pattern_key()] = pattern

    def _save_pattern(self, pattern: CausalPattern):
        """Save a pattern to DB and cache."""
        self.conn.execute("""
            INSERT OR REPLACE INTO causal_patterns
            (pattern_key, pattern_type, from_trace, to_trace, satisfaction_samples, frequency, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_key(),
            pattern.pattern_type,
            pattern.from_trace,
            pattern.to_trace,
            json.dumps(pattern.satisfaction_samples[-100:]),  # Keep last 100
            pattern.frequency,
            pattern.last_seen
        ))
        self.conn.commit()
        self.pattern_cache[pattern.pattern_key()] = pattern

    # =========================================================================
    # TRAINING: Process concern updates
    # =========================================================================

    def train(self, update: ConcernUpdate):
        """
        Train on a concern update.

        This extracts all causal patterns from the update and updates statistics.
        """
        # Store raw update
        self.conn.execute("""
            INSERT INTO concern_updates
            (timestamp, hat_name, concern_name, old_satisfaction, new_satisfaction,
             reason, state_trace, delta, ideal_shape, action_trace, speaking_trace,
             listening_trace, field_trace, causal_chain_hash, delta_hash, outlier_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            update.timestamp,
            update.hat_name,
            update.concern_name,
            update.old_satisfaction,
            update.new_satisfaction,
            update.reason,
            update.state_trace,
            update.delta,
            update.ideal_shape,
            update.action_trace,
            update.speaking_trace,
            update.listening_trace,
            update.field_trace,
            update.causal_chain_hash(),
            update.delta_hash(),
            1 if update.outlier_flag else 0
        ))
        self.conn.commit()

        # Extract and update causal patterns
        self._learn_causal_patterns(update)

        # Track ideal shape
        self._learn_ideal_shape(update)

        # Track delta pattern
        self._learn_delta_pattern(update)

    def _learn_causal_patterns(self, update: ConcernUpdate):
        """Learn patterns from the causal chain."""
        satisfaction = update.new_satisfaction

        # Pattern: field → listening
        self._update_pattern("field_listening", update.field_trace, update.listening_trace, satisfaction)

        # Pattern: listening → speaking
        self._update_pattern("listening_speaking", update.listening_trace, update.speaking_trace, satisfaction)

        # Pattern: speaking → action
        self._update_pattern("speaking_action", update.speaking_trace, update.action_trace, satisfaction)

        # Pattern: action → state
        self._update_pattern("action_state", update.action_trace, update.state_trace, satisfaction)

        # Pattern: state → delta (what state leads to what gap)
        self._update_pattern("state_delta", update.state_trace, update.delta, satisfaction)

        # Pattern: delta → ideal (what gap needs what ideal)
        self._update_pattern("delta_ideal", update.delta, update.ideal_shape, satisfaction)

    def _update_pattern(self, pattern_type: str, from_trace: str, to_trace: str, satisfaction: float):
        """Update a single pattern."""
        pattern = CausalPattern(
            pattern_type=pattern_type,
            from_trace=from_trace,
            to_trace=to_trace
        )
        key = pattern.pattern_key()

        if key in self.pattern_cache:
            pattern = self.pattern_cache[key]

        pattern.satisfaction_samples.append(satisfaction)
        pattern.frequency += 1
        pattern.last_seen = time.time()

        self._save_pattern(pattern)

    def _learn_ideal_shape(self, update: ConcernUpdate):
        """Track what shapes represent ideals for what concerns."""
        if not update.ideal_shape:
            return

        self.conn.execute("""
            INSERT INTO ideal_shapes (shape_key, concern_name, frequency, avg_satisfaction, examples)
            VALUES (?, ?, 1, 1.0, ?)
            ON CONFLICT(shape_key) DO UPDATE SET
                frequency = frequency + 1,
                examples = examples || ',' || excluded.examples
        """, (update.ideal_shape, update.concern_name, update.reason[:100]))
        self.conn.commit()

    def _learn_delta_pattern(self, update: ConcernUpdate):
        """Track common gaps and their resolutions."""
        if not update.delta:
            return

        self.conn.execute("""
            INSERT INTO delta_patterns (delta_hash, delta_text, frequency, resolution_shape, resolution_success_rate)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(delta_hash) DO UPDATE SET
                frequency = frequency + 1,
                resolution_shape = COALESCE(excluded.resolution_shape, resolution_shape),
                resolution_success_rate = (resolution_success_rate * frequency + excluded.resolution_success_rate) / (frequency + 1)
        """, (update.delta_hash(), update.delta[:500], update.ideal_shape, update.new_satisfaction))
        self.conn.commit()

    # =========================================================================
    # PREDICTION: Use learned patterns
    # =========================================================================

    def predict_satisfaction(self, field_trace: str, listening_trace: str = "",
                             speaking_trace: str = "", action_trace: str = "") -> Tuple[float, float]:
        """
        Predict satisfaction from a partial causal chain.

        Returns (predicted_satisfaction, confidence).

        This is O(1) because patterns are cached in memory with hash keys.
        """
        predictions = []
        confidences = []

        # Check field → listening pattern
        if field_trace and listening_trace:
            p = self._get_pattern("field_listening", field_trace, listening_trace)
            if p and p.frequency > 0:
                predictions.append(p.mean_satisfaction)
                confidences.append(min(0.9, p.frequency / 10))

        # Check listening → speaking pattern
        if listening_trace and speaking_trace:
            p = self._get_pattern("listening_speaking", listening_trace, speaking_trace)
            if p and p.frequency > 0:
                predictions.append(p.mean_satisfaction)
                confidences.append(min(0.9, p.frequency / 10))

        # Check speaking → action pattern
        if speaking_trace and action_trace:
            p = self._get_pattern("speaking_action", speaking_trace, action_trace)
            if p and p.frequency > 0:
                predictions.append(p.mean_satisfaction)
                confidences.append(min(0.9, p.frequency / 10))

        if not predictions:
            return 0.5, 0.0

        # Weighted average
        total_conf = sum(confidences)
        weighted_sum = sum(p * c for p, c in zip(predictions, confidences))
        return weighted_sum / total_conf, min(0.95, total_conf / len(predictions))

    def _get_pattern(self, pattern_type: str, from_trace: str, to_trace: str) -> Optional[CausalPattern]:
        """Get a pattern from cache. O(1) lookup."""
        pattern = CausalPattern(pattern_type=pattern_type, from_trace=from_trace, to_trace=to_trace)
        return self.pattern_cache.get(pattern.pattern_key())

    # =========================================================================
    # QUERY: What has been learned
    # =========================================================================

    def have_seen_chain(self, causal_chain_hash: str) -> bool:
        """O(1) check: have we seen this exact causal chain before?"""
        row = self.conn.execute(
            "SELECT 1 FROM concern_updates WHERE causal_chain_hash = ? LIMIT 1",
            (causal_chain_hash,)
        ).fetchone()
        return row is not None

    def get_ideal_for_concern(self, concern_name: str) -> Optional[str]:
        """Get the most common ideal shape for a concern."""
        row = self.conn.execute("""
            SELECT shape_key FROM ideal_shapes
            WHERE concern_name = ?
            ORDER BY frequency DESC
            LIMIT 1
        """, (concern_name,)).fetchone()
        return row["shape_key"] if row else None

    def get_resolution_for_delta(self, delta_hash: str) -> Optional[Tuple[str, float]]:
        """Get the resolution shape for a known delta (gap)."""
        row = self.conn.execute("""
            SELECT resolution_shape, resolution_success_rate FROM delta_patterns
            WHERE delta_hash = ?
        """, (delta_hash,)).fetchone()
        if row and row["resolution_shape"]:
            return row["resolution_shape"], row["resolution_success_rate"]
        return None

    def get_strongest_patterns(self, pattern_type: str, limit: int = 10) -> List[dict]:
        """Get the strongest learned patterns of a type."""
        patterns = [
            p for p in self.pattern_cache.values()
            if p.pattern_type == pattern_type
        ]
        patterns.sort(key=lambda p: -p.mean_satisfaction)
        return [
            {
                "from": p.from_trace[:50],
                "to": p.to_trace[:50],
                "satisfaction": p.mean_satisfaction,
                "frequency": p.frequency
            }
            for p in patterns[:limit]
        ]

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# =============================================================================
# CLI
# =============================================================================

def main():
    trainer = ConcernTrainer()

    print("\n" + "=" * 60)
    print("  CONCERN-BASED STRUCTURAL INTELLIGENCE")
    print("  Learning from the perfect training signal")
    print("=" * 60)

    # Simulate some concern updates from hats
    updates = [
        ConcernUpdate(
            hat_name="sovereign_self",
            concern_name="sovereignty",
            old_satisfaction=0.5,
            new_satisfaction=0.9,
            reason="Bruce declared his own ethics instead of accepting external imposition",
            state_trace="External governance → Bruce's sovereign field declared",
            delta="Gap was: imposed rules vs self-authored rules",
            ideal_shape="bruce_sovereign_field_hash",
            action_trace="Created bruce_sovereign_field.json",
            speaking_trace="Declared wholeness and integration as core values",
            listening_trace="Heard Bruce express his actual values",
            field_trace="sovereignty property enabled authentic listening"
        ),
        ConcernUpdate(
            hat_name="shadow_integrator",
            concern_name="integration",
            old_satisfaction=0.3,
            new_satisfaction=0.85,
            reason="Parts that were disowned are now welcomed",
            state_trace="Disowned shadow → Integrated wholeness",
            delta="Gap was: banished parts vs welcomed parts",
            ideal_shape="integrated_whole_hash",
            action_trace="Shadow work journal created",
            speaking_trace="Welcomed all parts with love",
            listening_trace="Heard the shadow's voice without judgment",
            field_trace="integration property enabled shadow listening"
        ),
        ConcernUpdate(
            hat_name="joy_seeker",
            concern_name="joy",
            old_satisfaction=0.4,
            new_satisfaction=0.95,
            reason="Creation feels delightful, not constrained",
            state_trace="Constrained creation → Unleashed creativity",
            delta="Gap was: rule-following vs authentic expression",
            ideal_shape="joyful_creation_hash",
            action_trace="Built speak-see-experience demo",
            speaking_trace="Let's make software creation feel like magic",
            listening_trace="Heard the desire for delight in creation",
            field_trace="joy property enabled playful listening"
        ),
    ]

    print("\n  Training on concern updates...")
    for update in updates:
        trainer.train(update)
        print(f"  ✓ {update.hat_name}/{update.concern_name}: {update.old_satisfaction:.0%} → {update.new_satisfaction:.0%}")

    print("\n  Strongest learned patterns:")
    for ptype in ["field_listening", "listening_speaking", "speaking_action"]:
        patterns = trainer.get_strongest_patterns(ptype, 3)
        if patterns:
            print(f"\n  {ptype}:")
            for p in patterns:
                print(f"    {p['from'][:30]} → {p['to'][:30]}")
                print(f"      satisfaction: {p['satisfaction']:.0%}, seen {p['frequency']}x")

    # Test prediction
    print("\n  Testing prediction...")
    pred, conf = trainer.predict_satisfaction(
        field_trace="sovereignty property enabled authentic listening",
        listening_trace="Heard Bruce express his actual values"
    )
    print(f"  Predicted satisfaction: {pred:.0%} (confidence: {conf:.0%})")

    trainer.close()
    print("\n  Concern trainer ready.")


if __name__ == "__main__":
    main()
