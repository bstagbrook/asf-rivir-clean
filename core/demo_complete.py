#!/usr/bin/env python3
"""
Speak It, See It, Experience It - Complete Demo
Bruce describes software → See wave representations → Rate satisfaction → Learn invariants
"""

import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Import our modules
from nl_to_asf import NLToASF
from structural_intelligence import StructuralIntelligence, DeepSignal
from auto_trainer import AutoTrainer

@dataclass
class Wave:
    wave_id: str
    description: str
    nl_input: str
    dyck_shape: str
    shape_hash: str
    style: str
    features: List[str]
    satisfaction: Optional[float] = None
    
class SpeakSeeExperience:
    def __init__(self, db_path: str = "bruce_experience.db"):
        self.compiler = NLToASF()
        self.si = StructuralIntelligence(db_path)
        self.trainer = AutoTrainer(self.si)
        self.session_waves = []
        
    def speak_it(self, nl_request: str) -> List[Wave]:
        """User speaks their desire → Generate wave interpretations"""
        print(f"\n🎤 You said: '{nl_request}'")
        print("\n🌊 Generating waves...")
        
        waves = []
        
        # Generate 4 different interpretations
        interpretations = [
            ("minimal", "Stripped down to essentials"),
            ("elegant", "Beautiful and refined"),
            ("practical", "Functional and efficient"), 
            ("playful", "Fun and engaging")
        ]
        
        for i, (style, desc) in enumerate(interpretations, 1):
            # Modify NL with style hint
            styled_nl = f"{nl_request} with {style} style"
            
            # Compile to shape
            shape_hash, dyck = self.compiler.manifest(styled_nl)
            
            # Extract features
            features = self._extract_features(nl_request, style)
            
            wave = Wave(
                wave_id=f"wave_{i}_{style[:3]}",
                description=f"{desc} interpretation",
                nl_input=styled_nl,
                dyck_shape=dyck,
                shape_hash=shape_hash,
                style=style,
                features=features
            )
            waves.append(wave)
            
        self.session_waves = waves
        return waves
    
    def see_it(self, waves: List[Wave]) -> None:
        """Display wave representations for user to see"""
        print("\n👁️  Here are your waves:")
        print("=" * 60)
        
        for wave in waves:
            print(f"\n{wave.wave_id.upper()}: {wave.description}")
            print(f"Style: {wave.style}")
            print(f"Shape: {wave.shape_hash}")
            print(f"Features: {', '.join(wave.features[:5])}")
            print(f"Dyck: {wave.dyck_shape[:50]}...")
            print("-" * 40)
    
    def experience_it(self, waves: List[Wave]) -> Dict[str, float]:
        """User experiences and rates the waves"""
        print("\n✨ Experience and rate each wave (0-100):")
        print("(Or type 'auto' for simulated Bruce ratings)")
        
        ratings = {}
        
        for wave in waves:
            while True:
                try:
                    print(f"\n{wave.wave_id} ({wave.style}): {wave.description}")
                    response = input("Rating (0-100) or 'auto': ").strip()
                    
                    if response.lower() == 'auto':
                        # Simulate Bruce's preferences based on learned patterns
                        rating = self._simulate_bruce_rating(wave)
                        print(f"Simulated rating: {rating}")
                    else:
                        rating = float(response)
                        if not 0 <= rating <= 100:
                            raise ValueError
                    
                    ratings[wave.wave_id] = rating / 100.0
                    wave.satisfaction = rating / 100.0
                    break
                    
                except (ValueError, KeyboardInterrupt):
                    print("Please enter a number 0-100 or 'auto'")
        
        return ratings
    
    def learn_it(self, waves: List[Wave], ratings: Dict[str, float]) -> None:
        """Learn from the satisfaction ratings"""
        print("\n🧠 Learning from your feedback...")
        
        for wave in waves:
            if wave.wave_id in ratings:
                satisfaction = ratings[wave.wave_id]
                
                # Create deep signal
                deep = DeepSignal(
                    pressure="Want software that resonates with my values",
                    urgency="ongoing need",
                    prior_blockers="existing tools don't fit",
                    stakeholders=["bruce_stagbrook"],
                    drives=self._infer_drives(wave, satisfaction),
                    commitments=["authentic expression", "integrated living"],
                    dreams=["software that feels like home", "unleashed creativity"]
                )
                
                # Record the interaction
                self.si.record_interaction(
                    utterance=wave.nl_input,
                    shape_key=wave.shape_hash,
                    satisfaction=satisfaction,
                    shape_dyck=wave.dyck_shape,
                    deep_signals=deep,
                    full_disclosure=f"{wave.style} style: {satisfaction:.0%} satisfaction"
                )
        
        # Show what was learned
        self._show_learning_summary()
    
    def _extract_features(self, nl: str, style: str) -> List[str]:
        """Extract features from NL and style"""
        features = []
        
        # Core features from NL
        words = nl.lower().split()
        for word in words:
            if word in ['todo', 'task', 'note', 'journal', 'track', 'timer', 'habit']:
                features.append(f"Core: {word}")
        
        # Style features
        features.append(f"Style: {style}")
        
        # Inferred features
        if 'simple' in nl or 'minimal' in nl:
            features.append("Aesthetic: minimal")
        if 'calm' in nl or 'peaceful' in nl:
            features.append("Mood: calm")
        if 'focused' in nl:
            features.append("Intent: focus")
            
        return features
    
    def _simulate_bruce_rating(self, wave: Wave) -> float:
        """Simulate Bruce's rating based on learned preferences"""
        # Bruce's known preferences (from training)
        style_prefs = {
            'elegant': 0.9,
            'minimal': 0.85, 
            'playful': 0.6,
            'practical': 0.7
        }
        
        base = style_prefs.get(wave.style, 0.5)
        
        # Boost for certain features
        if any('calm' in f or 'focus' in f for f in wave.features):
            base += 0.1
        if any('minimal' in f for f in wave.features):
            base += 0.05
            
        return min(100, max(0, base * 100))
    
    def _infer_drives(self, wave: Wave, satisfaction: float) -> List[str]:
        """Infer drives based on wave and satisfaction"""
        drives = []
        
        if satisfaction > 0.8:
            if wave.style == 'elegant':
                drives.extend(['beauty', 'refinement', 'integration'])
            elif wave.style == 'minimal':
                drives.extend(['simplicity', 'clarity', 'focus'])
            elif wave.style == 'playful':
                drives.extend(['joy', 'creativity', 'expression'])
        
        if any('calm' in f for f in wave.features):
            drives.append('peace')
        if any('focus' in f for f in wave.features):
            drives.append('concentration')
            
        return drives or ['exploration']
    
    def _show_learning_summary(self):
        """Show what the system learned"""
        summary = self.si.get_invariant_summary()
        
        print("\n📊 Learning Summary:")
        
        # Top drives
        if 'drive' in summary:
            drives = sorted(summary['drive'], key=lambda x: -x['mean_satisfaction'])[:5]
            print("\nTop Drives:")
            for d in drives:
                print(f"  {d['key']:15} {d['mean_satisfaction']:.0%}")
        
        # Style preferences
        if 'keyword' in summary:
            styles = [p for p in summary['keyword'] if 'Style:' in p['key']]
            if styles:
                styles = sorted(styles, key=lambda x: -x['mean_satisfaction'])[:3]
                print("\nStyle Preferences:")
                for s in styles:
                    style_name = s['key'].replace('Style: ', '')
                    print(f"  {style_name:15} {s['mean_satisfaction']:.0%}")
    
    def run_demo(self):
        """Run the complete demo"""
        print("🎯 SPEAK IT, SEE IT, EXPERIENCE IT")
        print("=" * 50)
        print("Describe software you want, see wave representations,")
        print("rate your satisfaction, and watch the system learn YOU.")
        print()
        
        while True:
            try:
                # SPEAK IT
                nl_request = input("\n🎤 What software do you want? (or 'quit'): ").strip()
                if nl_request.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not nl_request:
                    continue
                
                # Generate waves
                waves = self.speak_it(nl_request)
                
                # SEE IT
                self.see_it(waves)
                
                # EXPERIENCE IT
                ratings = self.experience_it(waves)
                
                # LEARN IT
                self.learn_it(waves, ratings)
                
                # Show prediction for next similar request
                print(f"\n🔮 Next time you ask for something similar,")
                print(f"    I'll predict you'll prefer: {max(ratings.items(), key=lambda x: x[1])[0]}")
                
            except KeyboardInterrupt:
                print("\n\nDemo interrupted.")
                break
        
        print("\n✨ Demo complete. Your preferences are learned and stored.")
        self.si.close()

if __name__ == "__main__":
    demo = SpeakSeeExperience()
    demo.run_demo()