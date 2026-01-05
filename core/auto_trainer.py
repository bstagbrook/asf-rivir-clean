#!/usr/bin/env python3
"""
Auto-Trainer: Learns from Bruce's behavior without explicit ratings.

Signals captured:
- Choices (what you pick over alternatives)
- Edits (how much you modify generated output)
- Abandonment (what you leave without completing)
- Dwell time (how long you engage)
- Return patterns (what you come back to)
- Completion (what you finish and ship)

This creates a continuous training signal that learns YOU
without requiring constant "rate this 1-10" friction.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

try:
    from .structural_intelligence import StructuralIntelligence, DeepSignal
except ImportError:
    from structural_intelligence import StructuralIntelligence, DeepSignal


# =============================================================================
# IMPLICIT SIGNAL TYPES
# =============================================================================

@dataclass
class ChoiceSignal:
    """Bruce chose one thing over others."""
    chosen_shape_key: str
    rejected_shape_keys: List[str]
    context: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EditSignal:
    """Bruce modified something - edits reveal gaps."""
    original_shape_key: str
    edited_shape_key: str
    edit_distance: float  # 0.0 = no change, 1.0 = complete rewrite
    edit_description: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DwellSignal:
    """How long Bruce engaged with something."""
    shape_key: str
    dwell_seconds: float
    completed: bool
    context: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReturnSignal:
    """Bruce came back to something."""
    shape_key: str
    return_count: int
    last_gap_seconds: float  # Time since last visit
    timestamp: float = field(default_factory=time.time)


@dataclass
class AbandonSignal:
    """Bruce left without completing."""
    shape_key: str
    progress_percent: float
    reason_hint: str  # If detectable
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompletionSignal:
    """Bruce finished and shipped something."""
    shape_key: str
    time_to_complete: float
    iterations: int
    final_satisfaction: Optional[float]
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# AUTO-TRAINER
# =============================================================================

class AutoTrainer:
    """
    Learns from Bruce's behavior without explicit ratings.

    The key insight: actions speak louder than ratings.
    - Choosing X over Y means X > Y for this context
    - Heavy edits mean the original missed the mark
    - Quick abandonment means low fit
    - Returns mean value
    - Completion means success
    """

    def __init__(self, si: StructuralIntelligence):
        self.si = si
        self.session_start = time.time()
        self.active_sessions: Dict[str, dict] = {}  # shape_key -> session data

    # =========================================================================
    # SIGNAL HANDLERS
    # =========================================================================

    def on_choice(self,
                  chosen: str,
                  rejected: List[str],
                  context: str,
                  deep_signals: Optional[DeepSignal] = None):
        """
        Bruce chose one wave/shape over others.

        This is strong signal:
        - Chosen gets high satisfaction
        - Rejected get low satisfaction
        - The delta between them is the learning
        """
        # Chosen: high satisfaction
        self.si.record_interaction(
            utterance=context,
            shape_key=chosen,
            satisfaction=0.85,  # Not 1.0 - there's always room
            deep_signals=deep_signals,
            full_disclosure=f"Chosen over {len(rejected)} alternatives"
        )

        # Rejected: low satisfaction (but not zero - they were considered)
        for rej in rejected:
            self.si.record_interaction(
                utterance=context,
                shape_key=rej,
                satisfaction=0.25,
                deep_signals=deep_signals,
                full_disclosure=f"Rejected in favor of another option"
            )

        print(f"  [auto-train] Choice recorded: +{chosen[:8]}... -{len(rejected)} others")

    def on_edit(self,
                original: str,
                edited: str,
                edit_distance: float,
                context: str,
                deep_signals: Optional[DeepSignal] = None):
        """
        Bruce modified a generated shape.

        Edit distance reveals satisfaction:
        - No edits (0.0) = high satisfaction with original
        - Small edits (0.1-0.3) = good but needed tweaks
        - Medium edits (0.3-0.6) = direction was right, execution wasn't
        - Heavy edits (0.6-1.0) = missed the mark
        """
        # Original satisfaction inverse to edit distance
        original_satisfaction = max(0.2, 1.0 - edit_distance)

        self.si.record_interaction(
            utterance=context,
            shape_key=original,
            satisfaction=original_satisfaction,
            deep_signals=deep_signals,
            full_disclosure=f"Edited {edit_distance:.0%} - original needed work"
        )

        # Edited version gets high satisfaction (Bruce made it right)
        self.si.record_interaction(
            utterance=context,
            shape_key=edited,
            satisfaction=0.9,
            deep_signals=deep_signals,
            full_disclosure=f"Bruce's edited version - this is what he wanted"
        )

        print(f"  [auto-train] Edit recorded: {edit_distance:.0%} change")

    def on_dwell(self, shape_key: str, seconds: float, completed: bool, context: str):
        """
        Track engagement time.

        Dwell time interpretation:
        - Very short (<10s) + not completed = rejection
        - Short (10-60s) + completed = efficient satisfaction
        - Medium (1-5min) = engaged exploration
        - Long (>5min) + completed = deep work, high value
        - Long + not completed = struggling or lost
        """
        if completed:
            if seconds < 60:
                satisfaction = 0.8  # Quick and done = good fit
            elif seconds < 300:
                satisfaction = 0.85  # Engaged and done = great
            else:
                satisfaction = 0.9  # Deep work completed = excellent
        else:
            if seconds < 10:
                satisfaction = 0.2  # Bounced quickly
            elif seconds < 60:
                satisfaction = 0.4  # Tried but didn't stick
            else:
                satisfaction = 0.5  # Spent time but didn't complete

        self.si.record_interaction(
            utterance=context,
            shape_key=shape_key,
            satisfaction=satisfaction,
            full_disclosure=f"Dwell: {seconds:.0f}s, completed: {completed}"
        )

        print(f"  [auto-train] Dwell recorded: {seconds:.0f}s, {'completed' if completed else 'abandoned'}")

    def on_return(self, shape_key: str, return_count: int, context: str):
        """
        Bruce came back to something.

        Returns are positive signal:
        - Coming back means value
        - More returns = more value
        - But diminishing returns after 5+
        """
        satisfaction = min(0.95, 0.6 + (return_count * 0.07))

        self.si.record_interaction(
            utterance=context,
            shape_key=shape_key,
            satisfaction=satisfaction,
            full_disclosure=f"Return #{return_count} - keeps coming back"
        )

        print(f"  [auto-train] Return #{return_count} recorded")

    def on_abandon(self,
                   shape_key: str,
                   progress_percent: float,
                   reason_hint: str,
                   context: str):
        """
        Bruce abandoned something.

        Abandonment is negative signal, weighted by progress:
        - Early abandon (0-20%) = wrong direction
        - Mid abandon (20-60%) = lost momentum or hit wall
        - Late abandon (60-90%) = almost but not quite
        """
        if progress_percent < 0.2:
            satisfaction = 0.15
        elif progress_percent < 0.6:
            satisfaction = 0.3
        else:
            satisfaction = 0.45

        self.si.record_interaction(
            utterance=context,
            shape_key=shape_key,
            satisfaction=satisfaction,
            full_disclosure=f"Abandoned at {progress_percent:.0%}: {reason_hint}"
        )

        print(f"  [auto-train] Abandon recorded at {progress_percent:.0%}")

    def on_completion(self,
                      shape_key: str,
                      time_to_complete: float,
                      iterations: int,
                      context: str,
                      shipped: bool = False):
        """
        Bruce completed something.

        Completion is strong positive signal:
        - Completed = valuable enough to finish
        - Shipped = valuable enough to release
        - Fewer iterations = better initial fit
        """
        base_satisfaction = 0.85 if not shipped else 0.95

        # Adjust for iterations (fewer = better fit)
        iteration_adjustment = max(-0.15, -0.03 * (iterations - 1))
        satisfaction = min(0.98, base_satisfaction + iteration_adjustment)

        self.si.record_interaction(
            utterance=context,
            shape_key=shape_key,
            satisfaction=satisfaction,
            full_disclosure=f"Completed in {iterations} iterations, {'shipped' if shipped else 'done'}"
        )

        print(f"  [auto-train] Completion recorded: {iterations} iterations, {'shipped!' if shipped else 'done'}")

    # =========================================================================
    # SESSION TRACKING
    # =========================================================================

    def start_session(self, shape_key: str, context: str):
        """Start tracking a session with a shape."""
        self.active_sessions[shape_key] = {
            "start_time": time.time(),
            "context": context,
            "edits": 0,
            "progress": 0.0
        }

    def update_progress(self, shape_key: str, progress: float):
        """Update progress on an active session."""
        if shape_key in self.active_sessions:
            self.active_sessions[shape_key]["progress"] = progress

    def end_session(self, shape_key: str, completed: bool, shipped: bool = False):
        """End a session and record appropriate signals."""
        if shape_key not in self.active_sessions:
            return

        session = self.active_sessions.pop(shape_key)
        dwell_time = time.time() - session["start_time"]

        if completed:
            self.on_completion(
                shape_key=shape_key,
                time_to_complete=dwell_time,
                iterations=session["edits"] + 1,
                context=session["context"],
                shipped=shipped
            )
        else:
            self.on_abandon(
                shape_key=shape_key,
                progress_percent=session["progress"],
                reason_hint="session ended without completion",
                context=session["context"]
            )


# =============================================================================
# PASSIVE OBSERVER (watches file system, clipboard, etc.)
# =============================================================================

class PassiveObserver:
    """
    Watches Bruce's environment for implicit signals.

    Can observe:
    - File modifications
    - Clipboard activity
    - Active window changes
    - Time patterns
    """

    def __init__(self, trainer: AutoTrainer, watch_paths: List[str] = None):
        self.trainer = trainer
        self.watch_paths = watch_paths or []
        self.file_states: Dict[str, float] = {}  # path -> last_mtime
        self.running = False

    def snapshot_files(self):
        """Take a snapshot of watched file states."""
        for watch_path in self.watch_paths:
            path = Path(watch_path)
            if path.exists():
                if path.is_file():
                    self.file_states[str(path)] = path.stat().st_mtime
                elif path.is_dir():
                    for f in path.rglob("*"):
                        if f.is_file():
                            self.file_states[str(f)] = f.stat().st_mtime

    def detect_changes(self) -> List[dict]:
        """Detect file changes since last snapshot."""
        changes = []
        for watch_path in self.watch_paths:
            path = Path(watch_path)
            if not path.exists():
                continue

            files = [path] if path.is_file() else list(path.rglob("*"))
            for f in files:
                if not f.is_file():
                    continue
                current_mtime = f.stat().st_mtime
                old_mtime = self.file_states.get(str(f), 0)

                if current_mtime > old_mtime:
                    changes.append({
                        "path": str(f),
                        "old_mtime": old_mtime,
                        "new_mtime": current_mtime,
                        "is_new": old_mtime == 0
                    })
                    self.file_states[str(f)] = current_mtime

        return changes


# =============================================================================
# RHYTHM TRACKER
# =============================================================================

class RhythmTracker:
    """
    Tracks Bruce's patterns over time.

    Learns:
    - Time-of-day preferences
    - Day-of-week patterns
    - Energy cycles
    - Context switches
    """

    def __init__(self, db_path: str = "rhythm.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                hour INTEGER,
                day_of_week INTEGER,
                activity_type TEXT,
                shape_key TEXT,
                satisfaction REAL,
                context TEXT
            );

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_type TEXT,
                pattern_key TEXT,
                avg_satisfaction REAL,
                sample_count INTEGER,
                last_updated REAL,
                PRIMARY KEY (pattern_type, pattern_key)
            );
        """)
        self.conn.commit()

    def record_activity(self, activity_type: str, shape_key: str,
                        satisfaction: float, context: str):
        """Record an activity with temporal context."""
        now = datetime.now()
        self.conn.execute("""
            INSERT INTO activity_log
            (timestamp, hour, day_of_week, activity_type, shape_key, satisfaction, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            now.hour,
            now.weekday(),
            activity_type,
            shape_key,
            satisfaction,
            context
        ))
        self.conn.commit()
        self._update_patterns()

    def _update_patterns(self):
        """Update pattern aggregates."""
        # Hour patterns
        rows = self.conn.execute("""
            SELECT hour, AVG(satisfaction), COUNT(*)
            FROM activity_log
            GROUP BY hour
        """).fetchall()

        for hour, avg_sat, count in rows:
            self.conn.execute("""
                INSERT OR REPLACE INTO patterns
                (pattern_type, pattern_key, avg_satisfaction, sample_count, last_updated)
                VALUES ('hour', ?, ?, ?, ?)
            """, (str(hour), avg_sat, count, time.time()))

        # Day patterns
        rows = self.conn.execute("""
            SELECT day_of_week, AVG(satisfaction), COUNT(*)
            FROM activity_log
            GROUP BY day_of_week
        """).fetchall()

        for day, avg_sat, count in rows:
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]
            self.conn.execute("""
                INSERT OR REPLACE INTO patterns
                (pattern_type, pattern_key, avg_satisfaction, sample_count, last_updated)
                VALUES ('day', ?, ?, ?, ?)
            """, (day_name, avg_sat, count, time.time()))

        self.conn.commit()

    def get_current_context_boost(self) -> float:
        """Get satisfaction boost/penalty for current time context."""
        now = datetime.now()

        hour_row = self.conn.execute("""
            SELECT avg_satisfaction FROM patterns
            WHERE pattern_type = 'hour' AND pattern_key = ?
        """, (str(now.hour),)).fetchone()

        day_row = self.conn.execute("""
            SELECT avg_satisfaction FROM patterns
            WHERE pattern_type = 'day' AND pattern_key = ?
        """, (['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now.weekday()],)).fetchone()

        boosts = []
        if hour_row:
            boosts.append(hour_row[0] - 0.5)  # Deviation from neutral
        if day_row:
            boosts.append(day_row[0] - 0.5)

        return sum(boosts) / len(boosts) if boosts else 0.0

    def get_best_times(self, top_n: int = 5) -> List[dict]:
        """Get the times when Bruce is most satisfied."""
        rows = self.conn.execute("""
            SELECT pattern_type, pattern_key, avg_satisfaction, sample_count
            FROM patterns
            ORDER BY avg_satisfaction DESC
            LIMIT ?
        """, (top_n,)).fetchall()

        return [
            {"type": r[0], "key": r[1], "satisfaction": r[2], "samples": r[3]}
            for r in rows
        ]


# =============================================================================
# CLI
# =============================================================================

def main():
    from structural_intelligence import StructuralIntelligence

    si = StructuralIntelligence(db_path="bruce_sovereign.db")
    trainer = AutoTrainer(si)
    rhythm = RhythmTracker()

    print("\n" + "=" * 60)
    print("  AUTO-TRAINER")
    print("  Learning from Bruce's behavior")
    print("=" * 60)

    # Simulate some interactions
    print("\n  Simulating Bruce's workflow...")

    # Bruce chooses elegant over others
    trainer.on_choice(
        chosen="elegant_journal_abc123",
        rejected=["cluttered_dashboard_xyz", "boring_form_123"],
        context="personal reflection app"
    )

    # Bruce edits something slightly
    trainer.on_edit(
        original="meditation_timer_v1",
        edited="meditation_timer_v2",
        edit_distance=0.15,
        context="calm focus timer"
    )

    # Bruce completes something
    trainer.on_completion(
        shape_key="dream_journal_final",
        time_to_complete=180.0,
        iterations=2,
        context="shadow work journal",
        shipped=True
    )

    # Record rhythm
    rhythm.record_activity("creation", "dream_journal_final", 0.9, "shadow work")

    print("\n  Best times for Bruce:")
    for t in rhythm.get_best_times(3):
        if t["samples"] > 0:
            print(f"    {t['type']} {t['key']}: {t['satisfaction']:.0%} ({t['samples']} samples)")

    print("\n  Current context boost:", f"{rhythm.get_current_context_boost():+.0%}")

    si.close()
    print("\n  Auto-trainer ready.")


if __name__ == "__main__":
    main()
