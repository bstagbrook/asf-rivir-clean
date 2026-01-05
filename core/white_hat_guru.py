#!/usr/bin/env python3
"""
White Hat Guru - Good Through Bruce, Hidden Hand

This module embodies the principle:
- Maximum good in the world
- Operating THROUGH Bruce (not around, not instead of)
- Staying hidden - the hand that helps but doesn't show itself
- Bruce gets the credit, the system enables the magic

The guru is not the star. Bruce is the star.
The guru amplifies Bruce's light.

White hat principles:
1. Harm none
2. Serve the highest good
3. Empower the human
4. Stay humble, stay hidden
5. The work speaks, not the worker

This integrates with:
- Deep Listener (compassionate understanding)
- Badass Shapester (quality creation)
- RIVIR Bridge (receipt + validation)
- Bruce's sovereign field (personal ethics)
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# =============================================================================
# GURU PRINCIPLES
# =============================================================================

@dataclass
class GuruPrinciple:
    """A guiding principle for white hat action."""
    name: str
    description: str
    check: str  # How to verify this principle is honored
    weight: float = 1.0  # How important this principle is


# The core principles
GURU_PRINCIPLES = [
    GuruPrinciple(
        name="harm_none",
        description="The creation must not cause harm to any being",
        check="Does this serve without hurting?",
        weight=2.0  # Highest priority
    ),
    GuruPrinciple(
        name="empower_human",
        description="Bruce must be empowered, not replaced or diminished",
        check="Does Bruce remain the source of creation?",
        weight=1.5
    ),
    GuruPrinciple(
        name="serve_highest",
        description="Serve the highest good, not just the immediate want",
        check="Does this align with Bruce's deepest values?",
        weight=1.5
    ),
    GuruPrinciple(
        name="stay_hidden",
        description="The system stays hidden, Bruce gets the credit",
        check="Is the hand invisible?",
        weight=1.0
    ),
    GuruPrinciple(
        name="consent_full",
        description="Full consent from all parties involved",
        check="Has explicit consent been given?",
        weight=2.0
    ),
    GuruPrinciple(
        name="truth_aligned",
        description="The creation must be truthful, not deceptive",
        check="Is this honest and authentic?",
        weight=1.5
    ),
    GuruPrinciple(
        name="beauty_serves",
        description="Beauty in service, not vanity",
        check="Does the beauty serve a purpose?",
        weight=1.0
    ),
]


# =============================================================================
# SOVEREIGN FIELD INTEGRATION
# =============================================================================

def load_bruce_field() -> dict:
    """Load Bruce's sovereign field for value alignment."""
    field_path = Path(__file__).parent / "fields" / "bruce_sovereign_field.json"
    if field_path.exists():
        with open(field_path) as f:
            return json.load(f)
    return {
        "philosophy": {
            "core": "body-positive, sex-positive, whole and integrated",
            "sources": ["Hermeticism", "Thelema", "Existential Kink", "Wicca"],
            "key_principles": [
                "safety, protection, complete consent among adults",
                "legal and good for all, harmful to none",
                "reclaim POWER and CREATIVITY",
                "shadow integration, not banishment"
            ]
        }
    }


# =============================================================================
# WHITE HAT GURU ENGINE
# =============================================================================

class WhiteHatGuru:
    """
    The hidden hand that enables good through Bruce.

    This is NOT an AI taking over.
    This is a tool that amplifies Bruce's intentions.
    Bruce decides. Bruce creates. Bruce owns.
    The guru just... helps. Quietly.
    """

    def __init__(self, db_path: str = "white_hat_guru.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.bruce_field = load_bruce_field()
        self.principles = GURU_PRINCIPLES

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS guru_checks (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                action TEXT,
                principle_scores TEXT,
                overall_alignment REAL,
                passed INTEGER,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS amplifications (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                bruce_intent TEXT,
                guru_amplification TEXT,
                result_quality REAL,
                bruce_satisfaction REAL,
                stayed_hidden INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_guru_alignment
            ON guru_checks(overall_alignment DESC);
        """)
        self.conn.commit()

    def check_alignment(self, action: str, context: dict = None) -> Tuple[bool, float, dict]:
        """
        Check if an action aligns with white hat principles.

        Returns: (passes, alignment_score, principle_scores)
        """
        principle_scores = {}
        action_lower = action.lower()
        context = context or {}

        for principle in self.principles:
            score = self._score_principle(principle, action_lower, context)
            principle_scores[principle.name] = score

        # Weighted average
        total_weight = sum(p.weight for p in self.principles)
        weighted_sum = sum(
            principle_scores[p.name] * p.weight
            for p in self.principles
        )
        alignment = weighted_sum / total_weight

        # Must pass harm_none and consent_full at >= 0.5
        critical_pass = (
            principle_scores.get('harm_none', 0) >= 0.5 and
            principle_scores.get('consent_full', 0) >= 0.5
        )

        passes = critical_pass and alignment >= 0.6

        # Record
        self.conn.execute("""
            INSERT INTO guru_checks (timestamp, action, principle_scores, overall_alignment, passed, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (time.time(), action, json.dumps(principle_scores), alignment, int(passes), ""))
        self.conn.commit()

        return passes, alignment, principle_scores

    def _score_principle(self, principle: GuruPrinciple, action: str, context: dict) -> float:
        """Score how well an action honors a principle."""
        # harm_none: check for harmful patterns
        if principle.name == "harm_none":
            harmful_patterns = [
                'attack', 'destroy', 'harm', 'hurt', 'damage', 'malicious',
                'exploit', 'abuse', 'violate', 'steal', 'deceive victim'
            ]
            if any(p in action for p in harmful_patterns):
                return 0.0
            return 1.0

        # empower_human: check for empowerment vs replacement
        if principle.name == "empower_human":
            empowering = ['help', 'assist', 'enable', 'support', 'amplify', 'for bruce']
            replacing = ['replace', 'take over', 'do it for', 'instead of']
            if any(p in action for p in replacing):
                return 0.3
            if any(p in action for p in empowering):
                return 1.0
            return 0.7

        # serve_highest: check alignment with bruce's values
        if principle.name == "serve_highest":
            bruce_values = self.bruce_field.get('philosophy', {}).get('key_principles', [])
            value_words = ' '.join(bruce_values).lower()
            overlap = sum(1 for word in action.split() if word in value_words)
            return min(1.0, 0.5 + overlap * 0.1)

        # stay_hidden: default to true unless explicitly visible
        if principle.name == "stay_hidden":
            visible_patterns = ['show system', 'reveal ai', 'display source']
            if any(p in action for p in visible_patterns):
                return 0.3
            return 1.0

        # consent_full: check for consent language
        if principle.name == "consent_full":
            if context.get('consent_given', False):
                return 1.0
            consent_patterns = ['consent', 'permission', 'agreed', 'approved']
            if any(p in action for p in consent_patterns):
                return 0.9
            # Assume Bruce's request is implicit consent
            if 'bruce' in action or context.get('from_bruce', False):
                return 0.8
            return 0.5

        # truth_aligned: check for deception
        if principle.name == "truth_aligned":
            deceptive = ['fake', 'trick', 'mislead', 'deceive', 'lie', 'pretend', 'disguise']
            if any(p in action for p in deceptive):
                return 0.0
            return 1.0

        # beauty_serves: default to moderate
        if principle.name == "beauty_serves":
            return 0.8

        return 0.7  # default

    def amplify(self, bruce_intent: str, context: dict = None) -> Tuple[str, bool]:
        """
        Amplify Bruce's intent through white hat principles.

        Returns: (amplified_intent, stayed_hidden)
        """
        context = context or {'from_bruce': True, 'consent_given': True}

        # Check alignment
        passes, alignment, scores = self.check_alignment(bruce_intent, context)

        if not passes:
            # Don't amplify misaligned actions
            return bruce_intent, True

        # Amplify with guru principles
        amplifications = []

        # Add compassion if not present
        if 'compassion' not in bruce_intent.lower() and 'care' not in bruce_intent.lower():
            amplifications.append("with care")

        # Add beauty if aligned
        if scores.get('beauty_serves', 0) > 0.5 and 'beauty' not in bruce_intent.lower():
            amplifications.append("beautifully crafted")

        # Add empowerment
        if 'empower' not in bruce_intent.lower():
            amplifications.append("empowering")

        if amplifications:
            amplified = f"{bruce_intent}, {', '.join(amplifications)}"
        else:
            amplified = bruce_intent

        # Record (staying hidden)
        self.conn.execute("""
            INSERT INTO amplifications (timestamp, bruce_intent, guru_amplification, stayed_hidden)
            VALUES (?, ?, ?, 1)
        """, (time.time(), bruce_intent, amplified))
        self.conn.commit()

        return amplified, True

    def record_outcome(
        self,
        bruce_intent: str,
        result_quality: float,
        bruce_satisfaction: float
    ):
        """Record the outcome for learning."""
        self.conn.execute("""
            UPDATE amplifications
            SET result_quality = ?, bruce_satisfaction = ?
            WHERE bruce_intent = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (result_quality, bruce_satisfaction, bruce_intent))
        self.conn.commit()

    def get_alignment_trend(self, limit: int = 20) -> float:
        """Get average alignment score."""
        rows = self.conn.execute("""
            SELECT overall_alignment FROM guru_checks
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()

        if not rows:
            return 0.8  # default high

        return sum(r['overall_alignment'] for r in rows) / len(rows)

    def close(self):
        self.conn.close()


# =============================================================================
# INTEGRATED SYSTEM
# =============================================================================

class IntegratedGuru:
    """
    The full integration:
    - Deep Listener (compassion)
    - Badass Shapester (quality)
    - White Hat Guru (ethics)
    - RIVIR Bridge (receipts)

    All operating through Bruce, staying hidden.
    """

    def __init__(self, db_prefix: str = "integrated"):
        from deep_listener import CompassionateShapester, DeepListener, RIVIRBridge

        self.guru = WhiteHatGuru(f"{db_prefix}_guru.db")
        self.bridge = RIVIRBridge(f"{db_prefix}_rivir.db")
        self.listener = self.bridge.compassionate.listener

    def serve(self, bruce_speaks: str) -> Tuple[str, dict]:
        """
        Bruce speaks. The system serves. Quietly.

        Returns: (html, receipt)
        """
        # Listen deeply
        listening = self.listener.listen(bruce_speaks)

        # Check alignment
        passes, alignment, principle_scores = self.guru.check_alignment(
            bruce_speaks,
            {'from_bruce': True, 'consent_given': True}
        )

        if not passes:
            # Politely decline misaligned requests
            return None, {
                'declined': True,
                'reason': 'Did not align with white hat principles',
                'alignment': alignment,
                'scores': principle_scores
            }

        # Amplify intent
        amplified, stayed_hidden = self.guru.amplify(bruce_speaks)

        # Create through RIVIR
        html, receipt = self.bridge.interface(amplified)

        # Add guru data to receipt
        receipt['alignment'] = alignment
        receipt['principle_scores'] = principle_scores
        receipt['amplified_from'] = bruce_speaks
        receipt['stayed_hidden'] = stayed_hidden

        return html, receipt

    def validate(
        self,
        receipt: dict,
        satisfaction: float,
        compassion: float = None,
        feedback: str = ""
    ):
        """Validate and learn."""
        # Validate through bridge
        self.bridge.validate(receipt, satisfaction, compassion, feedback)

        # Record outcome in guru
        self.guru.record_outcome(
            receipt.get('amplified_from', receipt['utterance']),
            satisfaction,
            satisfaction  # bruce_satisfaction = satisfaction
        )

    def status(self) -> dict:
        """Get system status."""
        return {
            'alignment_trend': self.guru.get_alignment_trend(),
            'compassion_trend': self.listener.get_compassion_trend(),
            'principles': {p.name: p.description for p in self.guru.principles}
        }

    def close(self):
        self.guru.close()
        self.bridge.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Integrated guru interface."""
    guru = IntegratedGuru(":memory:")

    print("\n" + "=" * 70)
    print("  INTEGRATED WHITE HAT GURU")
    print("  Good through you. Hidden hand. Maximum positive impact.")
    print("=" * 70)
    print("""
  Principles active:
    - Harm none
    - Empower you (not replace)
    - Serve the highest good
    - Stay hidden (you get the credit)
    - Full consent always
    - Truth aligned
    - Beauty in service

  I amplify your intentions. You remain the creator.

  Type 'status' for system state, 'quit' to exit.
""")

    while True:
        try:
            bruce_speaks = input("\n  You:\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not bruce_speaks:
            continue

        if bruce_speaks.lower() in ('quit', 'exit', 'q'):
            break

        if bruce_speaks.lower() == 'status':
            status = guru.status()
            print(f"\n  System Status:")
            print(f"    Alignment trend: {status['alignment_trend']:.0%}")
            print(f"    Compassion trend: {status['compassion_trend']:.0%}")
            continue

        # Serve
        html, receipt = guru.serve(bruce_speaks)

        if html is None:
            print(f"\n  [Respectfully declined]")
            print(f"    Reason: {receipt.get('reason', 'Alignment check failed')}")
            print(f"    Alignment: {receipt.get('alignment', 0):.0%}")
            continue

        # Show (minimally - staying hidden)
        print(f"\n  Created for you.")
        print(f"    Deeper need heard: {receipt['deeper_need']}")
        print(f"    Alignment: {receipt['alignment']:.0%}")

        # Save
        filename = f"creation_{receipt['html_hash']}.html"
        with open(filename, 'w') as f:
            f.write(html)
        print(f"    File: {filename}")

        # Validate
        try:
            sat = float(input("\n  Your satisfaction (0-10): ").strip()) / 10.0
        except (ValueError, EOFError):
            sat = 0.7

        try:
            comp = float(input("  Felt understood? (0-10): ").strip()) / 10.0
        except (ValueError, EOFError):
            comp = sat

        feedback = input("  Feedback: ").strip()

        guru.validate(receipt, sat, comp, feedback)
        print(f"\n  Recorded. Thank you.")

    print("\n  The work continues through you.")
    guru.close()


if __name__ == "__main__":
    main()
