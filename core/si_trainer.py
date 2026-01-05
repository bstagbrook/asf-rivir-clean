#!/usr/bin/env python3
"""
Structural Intelligence Training Harness

Minimal, focused trainer that learns Bruce's satisfaction patterns rapidly.
Auto-captures interactions and updates invariants in real-time.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from .structural_intelligence import StructuralIntelligence, DeepSignal
    from .shape_runtime import run_python_source
except ImportError:
    from structural_intelligence import StructuralIntelligence, DeepSignal
    from shape_runtime import run_python_source

@dataclass
class TrainingSession:
    utterance: str
    satisfaction: float
    features: List[str]
    context: str = ""

class SITrainer:
    def __init__(self, db_path: str = "bruce_si_training.db"):
        self.si = StructuralIntelligence(db_path)
        self.session_count = 0

    def train_from_choice(self, utterance: str, chosen_code: str, rejected_codes: List[str]):
        """Train from Bruce choosing one option over others."""
        self.session_count += 1
        
        # High satisfaction for chosen
        self._record_code(utterance, chosen_code, satisfaction=0.9, 
                         context=f"chosen from {len(rejected_codes)+1} options")
        
        # Low satisfaction for rejected
        for code in rejected_codes:
            self._record_code(utterance, code, satisfaction=0.2, 
                             context="rejected option")

    def train_from_edit(self, utterance: str, original_code: str, edited_code: str, edit_distance: int):
        """Train from Bruce editing generated code."""
        self.session_count += 1
        
        # Satisfaction inversely related to edit distance
        original_satisfaction = max(0.1, 1.0 - edit_distance * 0.1)
        final_satisfaction = 0.95  # Assume edited version is nearly perfect
        
        self._record_code(utterance, original_code, satisfaction=original_satisfaction,
                         context=f"edited {edit_distance} changes")
        self._record_code(utterance, edited_code, satisfaction=final_satisfaction,
                         context="final edited version")

    def train_from_rating(self, utterance: str, code: str, rating: float):
        """Train from explicit 0-10 rating."""
        self.session_count += 1
        satisfaction = rating / 10.0
        self._record_code(utterance, code, satisfaction=satisfaction,
                         context=f"explicit rating {rating}/10")

    def _record_code(self, utterance: str, code: str, satisfaction: float, context: str):
        """Record a code sample with satisfaction."""
        # Extract features from code
        features = self._extract_code_features(code)
        
        # Generate shape key via runtime
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
                temp_db = f.name
            
            # Capture shape_key from runtime output
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, 'shape_runtime.py', 
                '--python', code, 
                '--db', temp_db
            ], capture_output=True, text=True, cwd='/Volumes/StagbrookField/stagbrook_field/asf_core4')
            
            shape_key = None
            for line in result.stdout.split('\n'):
                if line.startswith('shape_key:'):
                    shape_key = line.split(':', 1)[1].strip()
                    break
            
            os.unlink(temp_db)
            
            if not shape_key:
                shape_key = f"fallback_{hash(code) % 1000000:06d}"
                
        except Exception:
            shape_key = f"fallback_{hash(code) % 1000000:06d}"

        # Record interaction
        deep_signals = DeepSignal(
            pressure="live coding",
            urgency="now", 
            prior_blockers="",
            stakeholders=["bruce_stagbrook"],
            drives=["efficiency", "elegance"],
            commitments=["working software"],
            dreams=["effortless creation"]
        )

        self.si.record_interaction(
            utterance=utterance,
            shape_key=shape_key,
            satisfaction=satisfaction,
            shape_dyck="",  # Runtime will fill this
            deep_signals=deep_signals,
            full_disclosure=context,
            features=features
        )

    def _extract_code_features(self, code: str) -> List[str]:
        """Extract features from code for learning."""
        features = []
        
        # Language constructs
        if 'def ' in code: features.append('function_definition')
        if 'class ' in code: features.append('class_definition')
        if 'lambda' in code: features.append('lambda_expression')
        if 'for ' in code: features.append('for_loop')
        if 'while ' in code: features.append('while_loop')
        if 'if ' in code: features.append('conditional')
        if 'try:' in code: features.append('exception_handling')
        
        # Style indicators
        if len(code.split('\n')) <= 5: features.append('concise')
        if len(code.split('\n')) > 20: features.append('verbose')
        if code.count('#') > 2: features.append('well_commented')
        if '"""' in code or "'''" in code: features.append('documented')
        
        # Complexity
        nesting = max(line.count('    ') for line in code.split('\n') if line.strip())
        if nesting <= 1: features.append('flat_structure')
        if nesting >= 3: features.append('nested_structure')
        
        return features

    def predict_satisfaction(self, utterance: str, code: str) -> float:
        """Predict satisfaction for new code."""
        features = self._extract_code_features(code)
        shape_key = f"predict_{hash(code) % 1000000:06d}"
        
        prediction, confidence = self.si.predict_satisfaction(
            shape_key=shape_key,
            utterance=utterance,
            features=features
        )
        
        return prediction

    def get_learning_summary(self) -> Dict:
        """Get summary of what's been learned."""
        invariants = self.si.get_invariant_summary()
        trend = self.si.get_satisfaction_trend()
        
        return {
            "sessions": self.session_count,
            "invariants_learned": sum(len(patterns) for patterns in invariants.values()),
            "recent_satisfaction": sum(trend[-10:]) / len(trend[-10:]) if trend else 0.5,
            "top_features": [
                p for patterns in invariants.get('feature', [])[:5] 
                for p in [patterns] if p['mean_satisfaction'] > 0.7
            ]
        }

    def close(self):
        self.si.close()

def main():
    """Demo the trainer."""
    trainer = SITrainer()
    
    # Simulate training from choices
    trainer.train_from_choice(
        utterance="simple calculator",
        chosen_code="def calc(a, b, op):\n    return eval(f'{a} {op} {b}')",
        rejected_codes=[
            "class Calculator:\n    def add(self, a, b): return a + b\n    def sub(self, a, b): return a - b",
            "import operator\nops = {'+': operator.add, '-': operator.sub}\ndef calc(a, b, op): return ops[op](a, b)"
        ]
    )
    
    # Simulate training from edit
    trainer.train_from_edit(
        utterance="file reader",
        original_code="def read_file(path):\n    with open(path) as f:\n        return f.read()",
        edited_code="def read_file(path):\n    try:\n        with open(path) as f:\n            return f.read()\n    except FileNotFoundError:\n        return None",
        edit_distance=3
    )
    
    # Show learning
    summary = trainer.get_learning_summary()
    print(f"Training Summary: {summary}")
    
    # Test prediction
    prediction = trainer.predict_satisfaction(
        "simple calculator", 
        "lambda a, b, op: eval(f'{a} {op} {b}')"
    )
    print(f"Predicted satisfaction: {prediction:.1%}")
    
    trainer.close()

if __name__ == "__main__":
    main()