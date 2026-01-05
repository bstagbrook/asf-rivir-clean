#!/usr/bin/env python3
"""
Receipt Schema and Storage

Every transition in the system generates a receipt - a full-disclosure record
of what happened, why, and what the outcome was.

This is critical for:
1. Clockless FSM - every state transition is recorded
2. Causal audit - trace back any outcome to its causes
3. Learning - satisfaction signals for structural intelligence
4. Governance - concerns and votes on actions
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# RECEIPT SCHEMA
# =============================================================================

@dataclass
class Receipt:
    """
    A full-disclosure receipt for any system transition.

    This is the atomic unit of the audit trail.
    """
    # Identity
    receipt_id: str = ""  # Generated on store
    timestamp: float = field(default_factory=time.time)

    # Context
    field_id: str = ""  # Which field/context
    agent_id: str = ""  # Who/what acted
    hat_id: str = ""  # Under what role

    # The transition
    utterance: str = ""  # What was said/requested
    state_before: str = ""  # Shape key before
    state_after: str = ""  # Shape key after
    shape_dyck: str = ""  # The Dyck representation

    # Concerns
    concern_scores_before: Dict[str, float] = field(default_factory=dict)
    concern_scores_after: Dict[str, float] = field(default_factory=dict)

    # Prediction vs reality
    prediction: Optional[float] = None  # What we expected
    result: Optional[float] = None  # What actually happened
    prediction_error: Optional[float] = None  # Difference

    # Learning signals
    satisfaction: Optional[float] = None
    feedback: str = ""

    # Causal chain
    causal_pressure: str = ""  # What caused this
    listening_trace: str = ""  # What we heard
    speaking_trace: str = ""  # What we said
    action_trace: str = ""  # What we did

    # Classification
    receipt_type: str = "transition"  # transition, concern_update, vote, action
    status: str = "recorded"  # recorded, verified, disputed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Receipt':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConcernUpdate:
    """
    A concern update from a hat - the perfect training signal.

    Required fields per the hat contract:
    - reason: why the update
    - state_trace: what state led here
    - delta: what changed
    - ideal_shape: what 100% would look like
    - action_trace: what actions produced this
    - speaking_trace: what was declared
    - listening_trace: what was heard
    - field_trace: what field properties shaped this
    """
    concern_id: str = ""
    hat_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # The concern
    description: str = ""
    satisfaction: float = 0.5
    target_state_key: str = ""

    # Required traces (the training signal)
    reason: str = ""
    state_trace: str = ""
    delta: str = ""
    ideal_shape: str = ""
    action_trace: str = ""
    speaking_trace: str = ""
    listening_trace: str = ""
    field_trace: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Action:
    """
    A proposed or executed action.
    """
    action_id: str = ""
    timestamp: float = field(default_factory=time.time)

    proposer: str = ""  # Who proposed
    script: str = ""  # What to do (shape/dyck)
    prediction: float = 0.5  # Expected outcome
    result: Optional[float] = None  # Actual outcome
    status: str = "proposed"  # proposed, approved, executed, rejected

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Vote:
    """
    A vote on an action from a hat.
    """
    vote_id: str = ""
    action_id: str = ""
    hat_id: str = ""
    timestamp: float = field(default_factory=time.time)

    vote: str = "abstain"  # approve, reject, abstain
    min_ok: float = 0.0  # Minimum acceptable outcome
    max_ok: float = 1.0  # Maximum acceptable outcome
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# RECEIPT STORE
# =============================================================================

class ReceiptStore:
    """
    Persistent storage for receipts, concerns, actions, and votes.

    Extends the catalog with audit trail capabilities.
    """

    def __init__(self, db_path: str = "receipts.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            -- Receipts: the atomic audit unit
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                field_id TEXT,
                agent_id TEXT,
                hat_id TEXT,
                utterance TEXT,
                state_before TEXT,
                state_after TEXT,
                shape_dyck TEXT,
                concern_scores_before TEXT,
                concern_scores_after TEXT,
                prediction REAL,
                result REAL,
                prediction_error REAL,
                satisfaction REAL,
                feedback TEXT,
                causal_pressure TEXT,
                listening_trace TEXT,
                speaking_trace TEXT,
                action_trace TEXT,
                receipt_type TEXT DEFAULT 'transition',
                status TEXT DEFAULT 'recorded'
            );

            CREATE INDEX IF NOT EXISTS idx_receipts_timestamp ON receipts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_receipts_field ON receipts(field_id);
            CREATE INDEX IF NOT EXISTS idx_receipts_agent ON receipts(agent_id);
            CREATE INDEX IF NOT EXISTS idx_receipts_state ON receipts(state_after);

            -- Concerns: satisfaction tracking
            CREATE TABLE IF NOT EXISTS concerns (
                concern_id TEXT PRIMARY KEY,
                hat_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                description TEXT,
                satisfaction REAL,
                target_state_key TEXT,
                reason TEXT,
                state_trace TEXT,
                delta TEXT,
                ideal_shape TEXT,
                action_trace TEXT,
                speaking_trace TEXT,
                listening_trace TEXT,
                field_trace TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_concerns_hat ON concerns(hat_id);
            CREATE INDEX IF NOT EXISTS idx_concerns_satisfaction ON concerns(satisfaction);

            -- Actions: proposed and executed
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                proposer TEXT,
                script TEXT,
                prediction REAL,
                result REAL,
                status TEXT DEFAULT 'proposed'
            );

            CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);

            -- Votes: governance decisions
            CREATE TABLE IF NOT EXISTS votes (
                vote_id TEXT PRIMARY KEY,
                action_id TEXT NOT NULL,
                hat_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                vote TEXT,
                min_ok REAL,
                max_ok REAL,
                rationale TEXT,
                FOREIGN KEY (action_id) REFERENCES actions(action_id)
            );

            CREATE INDEX IF NOT EXISTS idx_votes_action ON votes(action_id);
        """)
        self.conn.commit()

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        import hashlib
        data = f"{prefix}:{time.time()}:{id(self)}"
        return f"{prefix}_{hashlib.sha256(data.encode()).hexdigest()[:12]}"

    # -------------------------------------------------------------------------
    # Receipts
    # -------------------------------------------------------------------------

    def store_receipt(self, receipt: Receipt) -> str:
        """Store a receipt and return its ID."""
        if not receipt.receipt_id:
            receipt.receipt_id = self._generate_id("rcpt")

        self.conn.execute("""
            INSERT INTO receipts (
                receipt_id, timestamp, field_id, agent_id, hat_id,
                utterance, state_before, state_after, shape_dyck,
                concern_scores_before, concern_scores_after,
                prediction, result, prediction_error,
                satisfaction, feedback,
                causal_pressure, listening_trace, speaking_trace, action_trace,
                receipt_type, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            receipt.receipt_id, receipt.timestamp, receipt.field_id,
            receipt.agent_id, receipt.hat_id, receipt.utterance,
            receipt.state_before, receipt.state_after, receipt.shape_dyck,
            json.dumps(receipt.concern_scores_before),
            json.dumps(receipt.concern_scores_after),
            receipt.prediction, receipt.result, receipt.prediction_error,
            receipt.satisfaction, receipt.feedback,
            receipt.causal_pressure, receipt.listening_trace,
            receipt.speaking_trace, receipt.action_trace,
            receipt.receipt_type, receipt.status
        ))
        self.conn.commit()
        return receipt.receipt_id

    def get_receipt(self, receipt_id: str) -> Optional[Receipt]:
        """Retrieve a receipt by ID."""
        row = self.conn.execute(
            "SELECT * FROM receipts WHERE receipt_id = ?",
            (receipt_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d['concern_scores_before'] = json.loads(d['concern_scores_before'] or '{}')
        d['concern_scores_after'] = json.loads(d['concern_scores_after'] or '{}')
        return Receipt.from_dict(d)

    def query_receipts(
        self,
        field_id: str = None,
        agent_id: str = None,
        since: float = None,
        limit: int = 100
    ) -> List[Receipt]:
        """Query receipts with filters."""
        query = "SELECT * FROM receipts WHERE 1=1"
        params = []

        if field_id:
            query += " AND field_id = ?"
            params.append(field_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        receipts = []
        for row in rows:
            d = dict(row)
            d['concern_scores_before'] = json.loads(d['concern_scores_before'] or '{}')
            d['concern_scores_after'] = json.loads(d['concern_scores_after'] or '{}')
            receipts.append(Receipt.from_dict(d))
        return receipts

    # -------------------------------------------------------------------------
    # Concerns
    # -------------------------------------------------------------------------

    def store_concern(self, concern: ConcernUpdate) -> str:
        """Store a concern update."""
        if not concern.concern_id:
            concern.concern_id = self._generate_id("cncn")

        self.conn.execute("""
            INSERT OR REPLACE INTO concerns (
                concern_id, hat_id, timestamp, description, satisfaction,
                target_state_key, reason, state_trace, delta, ideal_shape,
                action_trace, speaking_trace, listening_trace, field_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            concern.concern_id, concern.hat_id, concern.timestamp,
            concern.description, concern.satisfaction, concern.target_state_key,
            concern.reason, concern.state_trace, concern.delta, concern.ideal_shape,
            concern.action_trace, concern.speaking_trace,
            concern.listening_trace, concern.field_trace
        ))
        self.conn.commit()
        return concern.concern_id

    def get_concerns_by_hat(self, hat_id: str, limit: int = 50) -> List[ConcernUpdate]:
        """Get concerns for a hat."""
        rows = self.conn.execute(
            "SELECT * FROM concerns WHERE hat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (hat_id, limit)
        ).fetchall()
        return [ConcernUpdate(**dict(r)) for r in rows]

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def store_action(self, action: Action) -> str:
        """Store an action."""
        if not action.action_id:
            action.action_id = self._generate_id("actn")

        self.conn.execute("""
            INSERT INTO actions (action_id, timestamp, proposer, script, prediction, result, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            action.action_id, action.timestamp, action.proposer,
            action.script, action.prediction, action.result, action.status
        ))
        self.conn.commit()
        return action.action_id

    def update_action_result(self, action_id: str, result: float, status: str = "executed"):
        """Update action with result."""
        self.conn.execute(
            "UPDATE actions SET result = ?, status = ? WHERE action_id = ?",
            (result, status, action_id)
        )
        self.conn.commit()

    # -------------------------------------------------------------------------
    # Votes
    # -------------------------------------------------------------------------

    def store_vote(self, vote: Vote) -> str:
        """Store a vote."""
        if not vote.vote_id:
            vote.vote_id = self._generate_id("vote")

        self.conn.execute("""
            INSERT INTO votes (vote_id, action_id, hat_id, timestamp, vote, min_ok, max_ok, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vote.vote_id, vote.action_id, vote.hat_id, vote.timestamp,
            vote.vote, vote.min_ok, vote.max_ok, vote.rationale
        ))
        self.conn.commit()
        return vote.vote_id

    def get_votes_for_action(self, action_id: str) -> List[Vote]:
        """Get all votes for an action."""
        rows = self.conn.execute(
            "SELECT * FROM votes WHERE action_id = ?",
            (action_id,)
        ).fetchall()
        return [Vote(**dict(r)) for r in rows]

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------

    def satisfaction_trend(self, limit: int = 50) -> List[float]:
        """Get recent satisfaction scores."""
        rows = self.conn.execute(
            "SELECT satisfaction FROM receipts WHERE satisfaction IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [r['satisfaction'] for r in rows][::-1]

    def prediction_accuracy(self, limit: int = 100) -> float:
        """Calculate recent prediction accuracy."""
        rows = self.conn.execute("""
            SELECT prediction, result FROM receipts
            WHERE prediction IS NOT NULL AND result IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()

        if not rows:
            return 0.0

        errors = [abs(r['prediction'] - r['result']) for r in rows]
        return 1.0 - (sum(errors) / len(errors))

    def close(self):
        """Close database connection."""
        self.conn.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_transition_receipt(
    utterance: str,
    state_before: str,
    state_after: str,
    shape_dyck: str,
    satisfaction: float = None,
    agent_id: str = "system",
    field_id: str = "default"
) -> Receipt:
    """Create a standard transition receipt."""
    return Receipt(
        timestamp=time.time(),
        field_id=field_id,
        agent_id=agent_id,
        utterance=utterance,
        state_before=state_before,
        state_after=state_after,
        shape_dyck=shape_dyck,
        satisfaction=satisfaction,
        receipt_type="transition"
    )


def create_concern_update(
    hat_id: str,
    description: str,
    satisfaction: float,
    reason: str,
    ideal_shape: str = "",
    delta: str = ""
) -> ConcernUpdate:
    """Create a concern update with required fields."""
    return ConcernUpdate(
        hat_id=hat_id,
        timestamp=time.time(),
        description=description,
        satisfaction=satisfaction,
        reason=reason,
        ideal_shape=ideal_shape,
        delta=delta
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the receipt store."""
    store = ReceiptStore(":memory:")

    print("=== Receipt Store Demo ===\n")

    # Create some receipts
    r1 = create_transition_receipt(
        utterance="create a todo app",
        state_before="",
        state_after="shape_abc123",
        shape_dyck="(((())(()()())))",
        satisfaction=0.9,
        agent_id="bruce"
    )
    store.store_receipt(r1)
    print(f"Stored receipt: {r1.receipt_id}")

    # Create a concern
    c1 = create_concern_update(
        hat_id="quality",
        description="Code clarity",
        satisfaction=0.8,
        reason="User expressed satisfaction",
        ideal_shape="(((())(()()()())))"
    )
    store.store_concern(c1)
    print(f"Stored concern: {c1.concern_id}")

    # Create an action
    a1 = Action(
        proposer="system",
        script="generate_premium_todo",
        prediction=0.85
    )
    store.store_action(a1)
    print(f"Stored action: {a1.action_id}")

    # Add a vote
    v1 = Vote(
        action_id=a1.action_id,
        hat_id="quality",
        vote="approve",
        min_ok=0.7,
        rationale="Looks good"
    )
    store.store_vote(v1)
    print(f"Stored vote: {v1.vote_id}")

    # Query
    print(f"\nSatisfaction trend: {store.satisfaction_trend()}")
    print(f"Prediction accuracy: {store.prediction_accuracy():.0%}")

    store.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
