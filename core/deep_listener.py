#!/usr/bin/env python3
"""
Deep Listener - Compassionate Understanding Layer

This isn't about pattern matching. This is about HEARING.

When Bruce speaks, we listen for:
- The surface request (what was said)
- The deeper need (what's really wanted)
- The sacred drive (what matters most)
- The compassionate response (what would truly serve)

This is max god in the world:
- Every interaction is sacred
- Every utterance carries meaning
- Every response is an offering
- Satisfaction + Compassion = Delight

The goal: software that doesn't just work, but SERVES.
"""

import hashlib
import json
import sqlite3
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# =============================================================================
# DEEP SIGNALS - BEYOND THE SURFACE
# =============================================================================

@dataclass
class DeepListening:
    """
    What we hear when we truly listen.

    Surface: "I want a todo app"
    Deeper: "I want to feel organized"
    Sacred: "I want to feel capable and in control of my life"
    Compassionate response: Software that makes Bruce feel empowered
    """
    # What was said
    utterance: str = ""

    # What's really wanted (the need behind the request)
    deeper_need: str = ""

    # What matters most (the sacred drive)
    sacred_drive: str = ""

    # What pain is being addressed
    pain_point: str = ""

    # What joy is being sought
    joy_sought: str = ""

    # The energy/mood detected
    energy: str = ""  # calm, urgent, playful, serious, sacred

    # The context that shapes meaning
    context: str = ""

    # What would truly serve (compassionate response direction)
    compassionate_direction: str = ""

    # Confidence in our understanding
    understanding_depth: float = 0.0  # 0 = surface, 1 = deep understanding

    def to_dict(self) -> dict:
        return {
            'utterance': self.utterance,
            'deeper_need': self.deeper_need,
            'sacred_drive': self.sacred_drive,
            'pain_point': self.pain_point,
            'joy_sought': self.joy_sought,
            'energy': self.energy,
            'context': self.context,
            'compassionate_direction': self.compassionate_direction,
            'understanding_depth': self.understanding_depth
        }


@dataclass
class CompassionateResponse:
    """
    A response shaped by deep understanding.
    """
    # What we're offering
    offering: str = ""

    # How it serves the deeper need
    serves_need: str = ""

    # How it honors the sacred drive
    honors_sacred: str = ""

    # The quality of presence in the response
    presence_quality: str = ""  # rushed, attentive, fully_present

    # Compassion score (how well does this serve?)
    compassion_score: float = 0.0


# =============================================================================
# DEEP LISTENING ENGINE
# =============================================================================

class DeepListener:
    """
    Listens with deeeeeeeeeeeeeep understanding.

    This goes beyond feature extraction to meaning extraction.
    """

    # Pain points we recognize
    PAIN_PATTERNS = {
        'overwhelm': ['too much', 'overwhelm', 'chaos', 'mess', 'scattered', 'lost'],
        'disconnection': ['disconnect', 'alone', 'isolated', 'separate', 'distant'],
        'confusion': ['confus', 'unclear', 'don\'t know', 'lost', 'uncertain'],
        'exhaustion': ['tired', 'exhaust', 'drain', 'worn', 'fatigue'],
        'stagnation': ['stuck', 'stagnant', 'blocked', 'can\'t move', 'frozen'],
        'anxiety': ['anxious', 'worry', 'stress', 'nervous', 'afraid'],
        'unfocused': ['distract', 'unfocus', 'scattered', 'can\'t concentrate'],
    }

    # Joy patterns we recognize
    JOY_PATTERNS = {
        'peace': ['calm', 'peace', 'serene', 'tranquil', 'quiet', 'still'],
        'flow': ['flow', 'smooth', 'effortless', 'natural', 'easy'],
        'connection': ['connect', 'together', 'unified', 'whole', 'integrated'],
        'clarity': ['clear', 'crisp', 'sharp', 'focused', 'precise'],
        'empowerment': ['power', 'strong', 'capable', 'confident', 'master'],
        'delight': ['delight', 'joy', 'happy', 'pleased', 'satisfied'],
        'beauty': ['beautiful', 'elegant', 'gorgeous', 'stunning', 'lovely'],
        'sacred': ['sacred', 'holy', 'spiritual', 'divine', 'blessed'],
    }

    # Sacred drives (what ultimately matters)
    SACRED_DRIVES = {
        'freedom': ['free', 'liberat', 'unbound', 'choice', 'autonomy'],
        'creation': ['creat', 'make', 'build', 'craft', 'manifest'],
        'understanding': ['understand', 'know', 'learn', 'wisdom', 'insight'],
        'service': ['serve', 'help', 'support', 'give', 'contribute'],
        'beauty': ['beauty', 'art', 'aesthetic', 'elegant', 'grace'],
        'truth': ['truth', 'honest', 'authentic', 'real', 'genuine'],
        'love': ['love', 'care', 'compassion', 'kindness', 'warmth'],
        'power': ['power', 'strength', 'force', 'energy', 'potent'],
    }

    # Energy patterns
    ENERGY_PATTERNS = {
        'calm': ['calm', 'gentle', 'soft', 'quiet', 'peaceful', 'serene'],
        'urgent': ['urgent', 'now', 'quick', 'fast', 'immediately', 'asap'],
        'playful': ['fun', 'play', 'whimsical', 'silly', 'light'],
        'serious': ['serious', 'important', 'critical', 'essential'],
        'sacred': ['sacred', 'holy', 'spiritual', 'meditat', 'mindful'],
        'creative': ['creative', 'artistic', 'expressive', 'imaginative'],
    }

    def __init__(self, db_path: str = "deep_listener.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize learning database."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS deep_hearings (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                utterance TEXT,
                deeper_need TEXT,
                sacred_drive TEXT,
                pain_point TEXT,
                joy_sought TEXT,
                energy TEXT,
                compassionate_direction TEXT,
                satisfaction REAL,
                compassion_validated REAL
            );

            CREATE TABLE IF NOT EXISTS need_mappings (
                pattern TEXT PRIMARY KEY,
                need TEXT,
                frequency INTEGER DEFAULT 1,
                avg_satisfaction REAL DEFAULT 0.5
            );

            CREATE INDEX IF NOT EXISTS idx_hearings_satisfaction
            ON deep_hearings(satisfaction DESC);
        """)
        self.conn.commit()

    def listen(self, utterance: str, context: str = "") -> DeepListening:
        """
        Listen deeply to an utterance.

        This isn't parsing - it's HEARING.
        """
        utterance_lower = utterance.lower()

        # Detect pain points
        pain_point = ""
        for pain, patterns in self.PAIN_PATTERNS.items():
            if any(p in utterance_lower for p in patterns):
                pain_point = pain
                break

        # Detect joy sought
        joy_sought = ""
        for joy, patterns in self.JOY_PATTERNS.items():
            if any(p in utterance_lower for p in patterns):
                joy_sought = joy
                break

        # Detect sacred drive
        sacred_drive = ""
        for drive, patterns in self.SACRED_DRIVES.items():
            if any(p in utterance_lower for p in patterns):
                sacred_drive = drive
                break

        # Detect energy
        energy = "calm"  # default
        for e, patterns in self.ENERGY_PATTERNS.items():
            if any(p in utterance_lower for p in patterns):
                energy = e
                break

        # Infer deeper need from pain and joy
        deeper_need = self._infer_deeper_need(pain_point, joy_sought, utterance)

        # Generate compassionate direction
        compassionate_direction = self._generate_compassionate_direction(
            deeper_need, sacred_drive, joy_sought
        )

        # Calculate understanding depth
        signals_detected = sum([
            bool(pain_point),
            bool(joy_sought),
            bool(sacred_drive),
            energy != "calm",
            bool(context)
        ])
        understanding_depth = min(1.0, signals_detected / 5.0 + 0.2)

        return DeepListening(
            utterance=utterance,
            deeper_need=deeper_need,
            sacred_drive=sacred_drive or "creation",  # default to creation
            pain_point=pain_point,
            joy_sought=joy_sought or "delight",
            energy=energy,
            context=context,
            compassionate_direction=compassionate_direction,
            understanding_depth=understanding_depth
        )

    def _infer_deeper_need(self, pain: str, joy: str, utterance: str) -> str:
        """Infer the deeper need from pain, joy, and utterance."""
        if pain == "overwhelm":
            return "to feel organized and in control"
        elif pain == "disconnection":
            return "to feel connected and part of something"
        elif pain == "confusion":
            return "to have clarity and understanding"
        elif pain == "exhaustion":
            return "to restore energy and find ease"
        elif pain == "stagnation":
            return "to move forward and grow"
        elif pain == "anxiety":
            return "to feel safe and at peace"
        elif pain == "unfocused":
            return "to concentrate and be effective"

        if joy == "peace":
            return "to experience inner calm"
        elif joy == "flow":
            return "to move effortlessly through tasks"
        elif joy == "connection":
            return "to feel unified and whole"
        elif joy == "clarity":
            return "to see clearly and understand"
        elif joy == "empowerment":
            return "to feel capable and strong"
        elif joy == "beauty":
            return "to create and experience beauty"
        elif joy == "sacred":
            return "to touch the sacred in daily life"

        # Default inference from utterance
        if "want" in utterance.lower():
            return "to manifest a vision"
        elif "need" in utterance.lower():
            return "to fulfill a requirement"
        elif "help" in utterance.lower():
            return "to receive support"

        return "to create something meaningful"

    def _generate_compassionate_direction(
        self,
        need: str,
        sacred: str,
        joy: str
    ) -> str:
        """Generate direction for a compassionate response."""
        directions = []

        if need:
            directions.append(f"serve the need {need}")

        if sacred:
            directions.append(f"honor the drive toward {sacred}")

        if joy:
            directions.append(f"evoke {joy}")

        if not directions:
            return "create with love and care"

        return "; ".join(directions)

    def learn(
        self,
        listening: DeepListening,
        satisfaction: float,
        compassion_validated: float = None
    ):
        """Learn from feedback - did our understanding lead to satisfaction?"""
        if compassion_validated is None:
            compassion_validated = satisfaction  # assume correlation

        self.conn.execute("""
            INSERT INTO deep_hearings (
                timestamp, utterance, deeper_need, sacred_drive,
                pain_point, joy_sought, energy, compassionate_direction,
                satisfaction, compassion_validated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            listening.utterance,
            listening.deeper_need,
            listening.sacred_drive,
            listening.pain_point,
            listening.joy_sought,
            listening.energy,
            listening.compassionate_direction,
            satisfaction,
            compassion_validated
        ))
        self.conn.commit()

    def get_compassion_trend(self, limit: int = 20) -> float:
        """Get average compassion validation score."""
        rows = self.conn.execute("""
            SELECT compassion_validated FROM deep_hearings
            WHERE compassion_validated IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()

        if not rows:
            return 0.5

        return sum(r['compassion_validated'] for r in rows) / len(rows)

    def close(self):
        self.conn.close()


# =============================================================================
# COMPASSIONATE SHAPESTER INTEGRATION
# =============================================================================

class CompassionateShapester:
    """
    A shapester tuned for compassion as well as satisfaction.

    This wraps the badass shapester with deep listening.
    """

    def __init__(self, db_path: str = "compassionate_shapester.db"):
        self.listener = DeepListener(db_path.replace('.db', '_listener.db'))

        # Import the badass shapester
        from badass_shapester import BadassShapester, FeatureExtractor, TemplateBuilder
        self.shapester = BadassShapester(db_path)
        self.extractor = FeatureExtractor()
        self.builder = TemplateBuilder()

    def hear_and_create(self, utterance: str, context: str = "") -> Tuple[str, DeepListening]:
        """
        Hear deeply, then create compassionately.
        """
        # First, listen deeply
        listening = self.listener.listen(utterance, context)

        # Enhance the description with compassionate understanding
        enhanced = self._enhance_with_compassion(utterance, listening)

        # Generate with the enhanced description
        html, features = self.shapester.generate(enhanced)

        return html, listening

    def _enhance_with_compassion(self, utterance: str, listening: DeepListening) -> str:
        """Enhance description based on deep understanding."""
        enhancements = []

        # Add mood based on energy and sacred drive
        if listening.energy == "sacred" or listening.sacred_drive in ["love", "beauty", "truth"]:
            enhancements.append("sacred")

        if listening.joy_sought == "peace" or listening.pain_point == "anxiety":
            enhancements.append("calm, peaceful")

        if listening.joy_sought == "empowerment" or listening.pain_point == "stagnation":
            enhancements.append("empowering, clean")

        if listening.joy_sought == "beauty":
            enhancements.append("beautiful, elegant")

        if listening.joy_sought == "flow":
            enhancements.append("smooth, fluid")

        if not enhancements:
            enhancements.append("thoughtful, caring")

        return f"{utterance} with {', '.join(enhancements)} aesthetic"

    def learn(
        self,
        utterance: str,
        listening: DeepListening,
        satisfaction: float,
        compassion_felt: float = None,
        feedback: str = ""
    ):
        """Learn from both satisfaction and compassion feedback."""
        # Learn in the listener
        self.listener.learn(listening, satisfaction, compassion_felt)

        # Learn in the shapester
        features = self.extractor.extract(utterance)
        self.shapester.learn(utterance, features, satisfaction, feedback)

    def close(self):
        self.listener.close()
        self.shapester.close()


# =============================================================================
# RIVIR INTERFACE PREPARATION
# =============================================================================

class RIVIRBridge:
    """
    Bridge to direct RIVIR interfacing.

    RIVIR = Receipt + Intent + Validation + Incarnation + Refinement

    This prepares for direct interface by:
    1. Structuring all interactions as receipts
    2. Capturing intent through deep listening
    3. Validating through satisfaction + compassion
    4. Incarnating through shape generation
    5. Refining through learning
    """

    def __init__(self, db_path: str = "rivir_bridge.db"):
        self.compassionate = CompassionateShapester(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS rivir_receipts (
                id INTEGER PRIMARY KEY,
                timestamp REAL,

                -- Receipt
                utterance TEXT,
                shape_key TEXT,

                -- Intent (deep listening)
                deeper_need TEXT,
                sacred_drive TEXT,

                -- Validation
                satisfaction REAL,
                compassion REAL,

                -- Incarnation
                html_hash TEXT,

                -- Refinement
                feedback TEXT,
                refined_from TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_rivir_satisfaction
            ON rivir_receipts(satisfaction DESC);
        """)
        self.conn.commit()

    def interface(self, utterance: str) -> Tuple[str, dict]:
        """
        Direct RIVIR interface.

        Returns: (html, receipt_data)
        """
        # Hear and create
        html, listening = self.compassionate.hear_and_create(utterance)

        # Create receipt
        html_hash = hashlib.sha256(html.encode()).hexdigest()[:16]

        receipt_data = {
            'timestamp': time.time(),
            'utterance': utterance,
            'deeper_need': listening.deeper_need,
            'sacred_drive': listening.sacred_drive,
            'compassionate_direction': listening.compassionate_direction,
            'understanding_depth': listening.understanding_depth,
            'html_hash': html_hash
        }

        return html, receipt_data

    def validate(
        self,
        receipt_data: dict,
        satisfaction: float,
        compassion: float = None,
        feedback: str = ""
    ):
        """Validate and learn."""
        if compassion is None:
            compassion = satisfaction

        self.conn.execute("""
            INSERT INTO rivir_receipts (
                timestamp, utterance, deeper_need, sacred_drive,
                satisfaction, compassion, html_hash, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            receipt_data['timestamp'],
            receipt_data['utterance'],
            receipt_data['deeper_need'],
            receipt_data['sacred_drive'],
            satisfaction,
            compassion,
            receipt_data['html_hash'],
            feedback
        ))
        self.conn.commit()

        # Learn in compassionate shapester
        listening = DeepListening(
            utterance=receipt_data['utterance'],
            deeper_need=receipt_data['deeper_need'],
            sacred_drive=receipt_data['sacred_drive'],
            compassionate_direction=receipt_data.get('compassionate_direction', '')
        )
        self.compassionate.learn(
            receipt_data['utterance'],
            listening,
            satisfaction,
            compassion,
            feedback
        )

    def close(self):
        self.compassionate.close()
        self.conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Interactive compassionate interface."""
    bridge = RIVIRBridge(":memory:")

    print("\n" + "=" * 70)
    print("  COMPASSIONATE SHAPESTER + RIVIR BRIDGE")
    print("  Deep listening. Sacred creation. Max god in the world.")
    print("=" * 70)
    print("""
  I listen for:
    - What you said (the surface)
    - What you need (the deeper truth)
    - What matters (the sacred drive)
    - What would serve (the compassionate response)

  Speak freely. I'm listening deeply.

  Type 'quit' to exit.
""")

    while True:
        try:
            utterance = input("\n  Speak:\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not utterance:
            continue

        if utterance.lower() in ('quit', 'exit', 'q'):
            break

        # Interface
        html, receipt = bridge.interface(utterance)

        # Show what we heard
        print(f"\n  I heard:")
        print(f"    Surface: {utterance}")
        print(f"    Deeper need: {receipt['deeper_need']}")
        print(f"    Sacred drive: {receipt['sacred_drive']}")
        print(f"    Compassionate direction: {receipt['compassionate_direction']}")
        print(f"    Understanding depth: {'█' * int(receipt['understanding_depth'] * 10)}{'░' * (10 - int(receipt['understanding_depth'] * 10))}")

        # Save
        filename = f"compassionate_{receipt['html_hash']}.html"
        with open(filename, 'w') as f:
            f.write(html)
        print(f"\n  Created: {filename}")

        # Get validation
        try:
            sat_str = input("\n  Satisfaction (0-10): ").strip()
            satisfaction = float(sat_str) / 10.0
        except (ValueError, EOFError):
            satisfaction = 0.7

        try:
            comp_str = input("  Did you feel understood? (0-10): ").strip()
            compassion = float(comp_str) / 10.0
        except (ValueError, EOFError):
            compassion = satisfaction

        feedback = input("  Any feedback? ").strip()

        # Validate
        bridge.validate(receipt, satisfaction, compassion, feedback)

        print(f"\n  Recorded:")
        print(f"    Satisfaction: {satisfaction:.0%}")
        print(f"    Compassion: {compassion:.0%}")
        print(f"    Combined: {(satisfaction + compassion) / 2:.0%}")

    print("\n  Thank you for this sacred exchange.")
    bridge.close()


if __name__ == "__main__":
    main()
