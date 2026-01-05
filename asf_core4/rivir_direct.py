#!/usr/bin/env python3
"""
RIVIR Direct Interface - The Full Integration

RIVIR = Receipt + Intent + Validation + Incarnation + Refinement

This is the direct interface where Bruce speaks and the system LISTENS:

1. **QUANTUM AFFORDANCES** - Superposition of possibilities, interference of paths
2. **DEEP LISTENER** - Hearing beyond surface to sacred drive
3. **WHITE HAT GURU** - Maximum good, through Bruce, staying hidden
4. **BADASS SHAPESTER** - Quality software from quality description
5. **RECEIPT CHAIN** - Every transition is recorded, every outcome is learned

The system doesn't take over. The system AMPLIFIES.
Bruce speaks. The system listens. Quality emerges.

This IS the direct RIVIR interface.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Local imports with graceful fallbacks
try:
    from deep_listener import DeepListener, DeepListening, CompassionateShapester, RIVIRBridge
except ImportError:
    from .deep_listener import DeepListener, DeepListening, CompassionateShapester, RIVIRBridge

try:
    from white_hat_guru import WhiteHatGuru, GuruPrinciple, GURU_PRINCIPLES, load_bruce_field
except ImportError:
    from .white_hat_guru import WhiteHatGuru, GuruPrinciple, GURU_PRINCIPLES, load_bruce_field

try:
    from receipts import ReceiptStore, Receipt, create_transition_receipt
except ImportError:
    from .receipts import ReceiptStore, Receipt, create_transition_receipt

try:
    from weighted_state import from_shapes, normalize_l2, apply_operator, prune
    from operators import mark, grover_step
    from sampling import sample_counts, probabilities
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False


# =============================================================================
# WAVEFORM INTERFACE - QUANTUM AFFORDANCES FOR EXPLORATION
# =============================================================================

@dataclass
class WaveformState:
    """A superposition of possibilities."""
    amplitudes: Dict[str, complex] = field(default_factory=dict)
    collapsed: bool = False
    measured_outcome: str = ""

    def add_possibility(self, key: str, amplitude: complex = 1.0):
        """Add a possibility to the superposition."""
        self.amplitudes[key] = amplitude

    def normalize(self):
        """Normalize amplitudes."""
        if HAS_QUANTUM:
            self.amplitudes = normalize_l2(self.amplitudes)
        else:
            # Simple normalization
            total = sum(abs(a)**2 for a in self.amplitudes.values()) ** 0.5
            if total > 0:
                self.amplitudes = {k: v/total for k, v in self.amplitudes.items()}

    def amplify(self, oracle_fn):
        """Amplify good outcomes using Grover-style search."""
        if HAS_QUANTUM:
            self.amplitudes = apply_operator(self.amplitudes, mark(oracle_fn))
        else:
            # Simple amplification: double amplitude for good outcomes
            for k in list(self.amplitudes.keys()):
                if oracle_fn(k):
                    self.amplitudes[k] *= 2.0
            self.normalize()

    def collapse(self, seed: int = None) -> str:
        """Collapse to a definite outcome."""
        if HAS_QUANTUM:
            counts = sample_counts(self.amplitudes, n=1, seed=seed)
            self.measured_outcome = max(counts, key=counts.get)
        else:
            import random
            if seed is not None:
                random.seed(seed)
            # Sample proportional to |amplitude|^2
            probs = {k: abs(v)**2 for k, v in self.amplitudes.items()}
            total = sum(probs.values())
            if total == 0:
                self.measured_outcome = list(self.amplitudes.keys())[0] if self.amplitudes else ""
            else:
                r = random.random() * total
                cumulative = 0
                for k, p in probs.items():
                    cumulative += p
                    if r <= cumulative:
                        self.measured_outcome = k
                        break

        self.collapsed = True
        return self.measured_outcome


# =============================================================================
# INTEGRATED RIVIR ENGINE
# =============================================================================

class RIVIRDirect:
    """
    Direct RIVIR interface integrating all systems.

    This is the full integration:
    - Quantum affordances (superposition, amplification, collapse)
    - Deep listening (surface → deeper → sacred)
    - White hat guru (harm_none, stay_hidden, empower_human)
    - Badass shapester (quality through detail)
    - Receipt chain (full audit, learning)

    Bruce speaks. The system serves. Quietly.
    """

    def __init__(self, db_prefix: str = "rivir_direct"):
        # Core components
        self.guru = WhiteHatGuru(f"{db_prefix}_guru.db")
        self.listener = DeepListener(f"{db_prefix}_listener.db")
        self.receipts = ReceiptStore(f"{db_prefix}_receipts.db")
        self.bruce_field = load_bruce_field()

        # State tracking
        self.session_start = time.time()
        self.session_receipts = []
        self.alignment_scores = []
        self.compassion_scores = []

        # Connection tracking for DB
        self.conn = sqlite3.connect(f"{db_prefix}_session.db")
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize session tracking."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                started REAL,
                ended REAL,
                receipts_count INTEGER,
                avg_alignment REAL,
                avg_compassion REAL,
                avg_satisfaction REAL
            );

            CREATE TABLE IF NOT EXISTS intentions (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                timestamp REAL,
                utterance TEXT,
                deeper_need TEXT,
                sacred_drive TEXT,
                alignment REAL,
                passed INTEGER,
                outcome TEXT
            );
        """)
        self.conn.commit()

    def speak(self, utterance: str, context: dict = None) -> Tuple[Dict, dict]:
        """
        Bruce speaks. The system listens.

        Returns: (waveform_possibilities, listening_data)
        """
        context = context or {'from_bruce': True, 'consent_given': True}

        # 1. LISTEN DEEPLY
        listening = self.listener.listen(utterance)

        # 2. CHECK ALIGNMENT (white hat)
        passes, alignment, principle_scores = self.guru.check_alignment(
            utterance, context
        )
        self.alignment_scores.append(alignment)

        if not passes:
            return {}, {
                'declined': True,
                'reason': 'Did not pass white hat alignment check',
                'alignment': alignment,
                'principle_scores': principle_scores,
                'listening': listening.to_dict()
            }

        # 3. CREATE WAVEFORM (superposition of possibilities)
        waveform = self._create_waveform(utterance, listening)

        # 4. RECORD INTENTION
        self.conn.execute("""
            INSERT INTO intentions (timestamp, utterance, deeper_need, sacred_drive, alignment, passed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (time.time(), utterance, listening.deeper_need, listening.sacred_drive, alignment, 1))
        self.conn.commit()

        return waveform.amplitudes, {
            'listening': listening.to_dict(),
            'alignment': alignment,
            'principle_scores': principle_scores,
            'waveform_count': len(waveform.amplitudes),
            'energy': listening.energy,
            'compassionate_direction': listening.compassionate_direction
        }

    def _create_waveform(self, utterance: str, listening: DeepListening) -> WaveformState:
        """Create a waveform of possibilities from the utterance."""
        waveform = WaveformState()

        # Generate variations based on listening
        base_description = utterance

        # Create possibilities based on mood/energy
        moods = {
            'calm': ['serene', 'peaceful', 'gentle'],
            'urgent': ['clean', 'focused', 'minimal'],
            'playful': ['whimsical', 'colorful', 'fun'],
            'serious': ['professional', 'refined', 'elegant'],
            'sacred': ['transcendent', 'beautiful', 'profound'],
            'creative': ['artistic', 'expressive', 'bold']
        }

        energy_moods = moods.get(listening.energy, ['balanced'])
        for mood in energy_moods:
            key = f"{base_description} | {mood}"
            waveform.add_possibility(key, 1.0)

        # Add variations based on sacred drive
        drives = {
            'freedom': ['liberated', 'open', 'flowing'],
            'creation': ['innovative', 'generative', 'original'],
            'love': ['warm', 'connected', 'harmonious'],
            'truth': ['authentic', 'clear', 'honest'],
            'beauty': ['aesthetic', 'elegant', 'refined'],
            'power': ['empowered', 'strong', 'confident']
        }

        drive_words = drives.get(listening.sacred_drive, ['intentional'])
        for drive in drive_words:
            key = f"{base_description} | {drive}"
            waveform.add_possibility(key, 1.0)

        # Normalize the waveform
        waveform.normalize()
        return waveform

    def amplify(self, waveform_key: str, oracle_criteria: str = "quality") -> Tuple[str, dict]:
        """
        Amplify a chosen possibility through the full system.
        
        This is where the magic happens:
        1. Take the waveform possibility
        2. Run it through the compassionate shapester
        3. Apply white hat amplification
        4. Generate the actual creation
        """
        # Check alignment again
        passes, alignment, scores = self.guru.check_alignment(waveform_key)
        if not passes:
            return "", {'declined': True, 'reason': 'Amplification failed alignment'}

        # Amplify through guru
        amplified, stayed_hidden = self.guru.amplify(waveform_key, self.bruce_field)
        
        # Create through shapester
        try:
            from badass_shapester import BadassShapester
            shapester = BadassShapester(":memory:")
            creation, features = shapester.generate(amplified)
            shapester.close()
        except ImportError:
            # Fallback: use RIVIR's simple generation
            creation = self._simple_html_generation(amplified)
            features = {'description': amplified}
        
        # Record receipt
        receipt = create_transition_receipt(
            utterance=waveform_key,
            state_before="waveform_superposition",
            state_after="creation_manifest",
            shape_dyck="((()))",  # Placeholder
            satisfaction=0.85,  # Will be updated with feedback
            agent_id="rivir_direct",
            field_id="bruce_sovereign"
        )
        receipt.feedback = "Creation amplified through RIVIR"
        
        self.receipts.store_receipt(receipt)
        self.session_receipts.append(receipt)
        
        return creation, {
            'amplified_intent': amplified,
            'stayed_hidden': stayed_hidden,
            'features': features,
            'alignment': alignment,
            'receipt_id': receipt.receipt_id
        }

    def collapse_and_create(self, utterance: str, seed: int = None) -> Tuple[str, dict]:
        """
        Full flow: speak → waveform → collapse → amplify → create
        
        This is the main interface for Bruce.
        """
        # 1. Create waveform from utterance
        possibilities, listening_data = self.speak(utterance)
        
        if listening_data.get('declined'):
            return "", listening_data
        
        # 2. Create waveform object and collapse
        waveform = WaveformState()
        waveform.amplitudes = possibilities
        
        # 3. Amplify good outcomes (oracle: quality + alignment)
        def quality_oracle(key: str) -> bool:
            return any(word in key.lower() for word in 
                      ['elegant', 'beautiful', 'refined', 'quality', 'authentic'])
        
        waveform.amplify(quality_oracle)
        
        # 4. Collapse to definite outcome
        chosen = waveform.collapse(seed)
        
        # 5. Amplify and create
        creation, creation_data = self.amplify(chosen)
        
        # Combine all data
        full_data = {
            **listening_data,
            **creation_data,
            'chosen_possibility': chosen,
            'total_possibilities': len(possibilities)
        }
        
        return creation, full_data

    def learn_from_feedback(self, receipt_id: str, satisfaction: float, notes: str = ""):
        """
        Learn from Bruce's feedback on creations.
        """
        # Update receipt with satisfaction
        receipt = self.receipts.get_receipt(receipt_id)
        if receipt:
            receipt.satisfaction = satisfaction
            receipt.feedback = notes
            self.receipts.update_receipt(receipt)
        
        # Track satisfaction
        self.compassion_scores.append(satisfaction)
        
        # Update session record
        self.conn.execute("""
            UPDATE intentions SET outcome = ? 
            WHERE id = (SELECT MAX(id) FROM intentions)
        """, (f"satisfaction: {satisfaction}, notes: {notes}",))
        self.conn.commit()

    def status(self) -> dict:
        """
        Get current session status.
        """
        return {
            'session_duration': time.time() - self.session_start,
            'receipts_created': len(self.session_receipts),
            'avg_alignment': sum(self.alignment_scores) / len(self.alignment_scores) if self.alignment_scores else 0,
            'avg_satisfaction': sum(self.compassion_scores) / len(self.compassion_scores) if self.compassion_scores else 0,
            'has_quantum': HAS_QUANTUM
        }

    def _simple_html_generation(self, description: str) -> str:
        """Fallback HTML generation."""
        return f"""<!DOCTYPE html>
<html><head><title>RIVIR Creation</title>
<style>body{{font-family:system-ui;padding:2rem;background:#f5f5f5;}}</style>
</head><body><h1>Created with RIVIR</h1><p>{description}</p></body></html>"""

    def close(self):
        """Clean shutdown."""
        # Record session
        avg_alignment = sum(self.alignment_scores) / len(self.alignment_scores) if self.alignment_scores else 0
        avg_satisfaction = sum(self.compassion_scores) / len(self.compassion_scores) if self.compassion_scores else 0
        
        self.conn.execute("""
            INSERT INTO sessions (started, ended, receipts_count, avg_alignment, avg_satisfaction)
            VALUES (?, ?, ?, ?, ?)
        """, (self.session_start, time.time(), len(self.session_receipts), avg_alignment, avg_satisfaction))
        self.conn.commit()
        self.conn.close()
        
        self.guru.close()
        self.receipts.close()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Interactive CLI for RIVIR Direct."""
    print("🌊 RIVIR Direct Interface")
    print("Quantum affordances + Deep listening + White hat guru + Badass shapester")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    rivir = RIVIRDirect()
    
    try:
        while True:
            try:
                user_input = input("Bruce> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if user_input.lower() == 'help':
                    print("""
Commands:
  <utterance>           - Speak and create (full RIVIR flow)
  status               - Show session status
  feedback <rating>    - Rate last creation (0.0-1.0)
  quantum              - Show quantum affordances status
  help                 - Show this help
  quit                 - Exit

Example:
  Bruce> create a calm todo list with soft colors
  Bruce> feedback 0.9
                    """)
                    continue
                    
                if user_input.lower() == 'status':
                    status = rivir.status()
                    print(f"\n📊 Session Status:")
                    print(f"  Duration: {status['session_duration']:.1f}s")
                    print(f"  Receipts: {status['receipts_created']}")
                    print(f"  Avg Alignment: {status['avg_alignment']:.2f}")
                    print(f"  Avg Satisfaction: {status['avg_satisfaction']:.2f}")
                    print(f"  Quantum: {'✓' if status['has_quantum'] else '✗'}")
                    continue
                    
                if user_input.lower() == 'quantum':
                    print(f"\n⚛️  Quantum Affordances: {'Available' if HAS_QUANTUM else 'Simulated'}")
                    if HAS_QUANTUM:
                        print("  - Superposition of possibilities")
                        print("  - Grover amplification of quality")
                        print("  - Quantum collapse to creation")
                    else:
                        print("  - Classical simulation active")
                        print("  - Install quantum modules for full affordances")
                    continue
                    
                if user_input.startswith('feedback '):
                    try:
                        rating = float(user_input.split()[1])
                        if rivir.session_receipts:
                            last_receipt = rivir.session_receipts[-1]
                            rivir.learn_from_feedback(last_receipt.receipt_id, rating)
                            print(f"✓ Feedback recorded: {rating}")
                        else:
                            print("No recent creation to rate")
                    except (ValueError, IndexError):
                        print("Usage: feedback <rating> (0.0-1.0)")
                    continue
                
                # Main creation flow
                print("\n🎯 Processing...")
                creation, data = rivir.collapse_and_create(user_input)
                
                if data.get('declined'):
                    print(f"❌ {data['reason']}")
                    print(f"   Alignment: {data.get('alignment', 0):.2f}")
                    continue
                
                print(f"\n✨ Created ({len(creation)} chars)")
                print(f"   Listening: {data['listening']['deeper_need']} → {data['listening']['sacred_drive']}")
                print(f"   Alignment: {data['alignment']:.2f}")
                print(f"   Possibilities: {data['total_possibilities']}")
                print(f"   Chosen: {data['chosen_possibility']}")
                print(f"   Receipt: {data['receipt_id'][:8]}...")
                
                # Show creation (truncated)
                if len(creation) > 200:
                    print(f"\n📄 Creation (first 200 chars):")
                    print(creation[:200] + "...")
                else:
                    print(f"\n📄 Creation:")
                    print(creation)
                    
                print("\n💡 Rate this creation with: feedback <0.0-1.0>")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        print("\n🌊 Closing RIVIR Direct...")
        rivir.close()
        print("Session saved. Goodbye!")


if __name__ == "__main__":
    main()