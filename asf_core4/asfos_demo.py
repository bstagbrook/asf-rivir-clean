#!/usr/bin/env python3
"""
Speak It, See It, Experience It - asfOS Edition
Bruce describes software → Compile to shapes → Execute in asfOS → Experience results → Rate satisfaction
"""

import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Import ASF runtime components
from nl_to_asf import NLToASF
from structural_intelligence import StructuralIntelligence, DeepSignal
from asfos_kernel import Kernel
from shape_runtime import run_python_source, run_python_direct
from py_to_dyck import compile_source, run_dyck

@dataclass
class ExecutableWave:
    wave_id: str
    description: str
    nl_input: str
    python_code: str
    dyck_shape: str
    shape_key: str
    style: str
    execution_result: Optional[str] = None
    satisfaction: Optional[float] = None
    
class ASFOSExperience:
    def __init__(self, db_path: str = "asfos_experience.db"):
        self.compiler = NLToASF()
        self.si = StructuralIntelligence(db_path)
        self.kernel = Kernel(db_path)
        self.kernel.install_default_handlers()
        self.session_waves = []
        
    def speak_it(self, nl_request: str) -> List[ExecutableWave]:
        """User speaks → Generate executable wave interpretations"""
        print(f"\n🎤 You said: '{nl_request}'")
        print("\n🌊 Compiling to executable shapes...")
        
        waves = []
        
        # Generate different style interpretations with actual Python code
        interpretations = [
            ("minimal", self._generate_minimal_code(nl_request)),
            ("elegant", self._generate_elegant_code(nl_request)),
            ("practical", self._generate_practical_code(nl_request)),
            ("playful", self._generate_playful_code(nl_request))
        ]
        
        for i, (style, python_code) in enumerate(interpretations, 1):
            # Compile Python → Dyck → Shape
            try:
                dyck = compile_source(python_code)
                shape_key = hashlib.sha256(dyck.encode()).hexdigest()[:16]
                
                wave = ExecutableWave(
                    wave_id=f"wave_{i}_{style[:3]}",
                    description=f"{style.title()} interpretation",
                    nl_input=nl_request,
                    python_code=python_code,
                    dyck_shape=dyck,
                    shape_key=shape_key,
                    style=style
                )
                waves.append(wave)
                
            except Exception as e:
                print(f"Failed to compile {style} wave: {e}")
                continue
                
        self.session_waves = waves
        return waves
    
    def see_it(self, waves: List[ExecutableWave]) -> None:
        """Display wave representations - code, shapes, and structure"""
        print("\n👁️  Here are your executable waves:")
        print("=" * 70)
        
        for wave in waves:
            print(f"\n{wave.wave_id.upper()}: {wave.description}")
            print(f"Style: {wave.style}")
            print(f"Shape Key: {wave.shape_key}")
            print(f"\nPython Code:")
            print("```python")
            print(wave.python_code)
            print("```")
            print(f"\nDyck Shape: {wave.dyck_shape[:60]}...")
            print("-" * 50)
    
    def experience_it(self, waves: List[ExecutableWave]) -> Dict[str, float]:
        """Execute waves in asfOS and let user experience the results"""
        print("\n✨ Executing waves in asfOS...")
        
        for wave in waves:
            print(f"\n🔄 Executing {wave.wave_id} ({wave.style})...")
            
            try:
                # Execute in ASF runtime
                asf_result = run_dyck(wave.dyck_shape)
                
                # Also run Python directly for comparison
                py_result = run_python_direct(wave.python_code)
                
                # Store execution results
                wave.execution_result = {
                    'asf_result': asf_result,
                    'py_result': py_result,
                    'dyck_length': len(wave.dyck_shape)
                }
                
                print(f"  ASF Result: {asf_result}")
                print(f"  Python Result: {py_result}")
                print(f"  Shape Length: {len(wave.dyck_shape)} chars")
                
            except Exception as e:
                wave.execution_result = f"Error: {e}"
                print(f"  Execution failed: {e}")
        
        # Now get satisfaction ratings
        print("\n🎯 Rate your satisfaction with each wave (0-100):")
        print("(Consider: Does it do what you wanted? Do you like the style? Does it feel right?)")
        
        ratings = {}
        for wave in waves:
            while True:
                try:
                    print(f"\n{wave.wave_id} ({wave.style}):")
                    print(f"  Result: {wave.execution_result}")
                    response = input("Satisfaction (0-100) or 'auto': ").strip()
                    
                    if response.lower() == 'auto':
                        rating = self._simulate_bruce_rating(wave)
                        print(f"Simulated Bruce rating: {rating}")
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
    
    def learn_it(self, waves: List[ExecutableWave], ratings: Dict[str, float]) -> None:
        """Learn from execution results and satisfaction"""
        print("\n🧠 Learning from execution and satisfaction...")
        
        for wave in waves:
            if wave.wave_id in ratings:
                satisfaction = ratings[wave.wave_id]
                
                # Store in asfOS kernel
                self.kernel.emit("store_python", {
                    "source": wave.python_code,
                    "label": f"{wave.style}_interpretation",
                    "namespace": "user_waves"
                })
                
                # Create deep signal based on execution
                deep = DeepSignal(
                    pressure="Want software that works AND feels right",
                    urgency="immediate feedback",
                    prior_blockers="tools that work but don't resonate",
                    stakeholders=["bruce_stagbrook"],
                    drives=self._infer_drives_from_execution(wave, satisfaction),
                    commitments=["functional AND beautiful", "shape-first computing"],
                    dreams=["software that executes my intentions perfectly"]
                )
                
                # Record in structural intelligence
                self.si.record_interaction(
                    utterance=wave.nl_input,
                    shape_key=wave.shape_key,
                    satisfaction=satisfaction,
                    shape_dyck=wave.dyck_shape,
                    deep_signals=deep,
                    full_disclosure=f"{wave.style}: {satisfaction:.0%} - executed: {wave.execution_result}"
                )
        
        # Process kernel events
        self.kernel.run()
        
        # Show learning
        self._show_execution_learning()
    
    def _generate_minimal_code(self, nl: str) -> str:
        """Generate minimal Python implementation"""
        if 'todo' in nl.lower():
            return 'lambda x: []'
        elif 'note' in nl.lower():
            return 'lambda x: ""'
        elif 'timer' in nl.lower():
            return 'lambda x: 0'
        else:
            return 'lambda x: "simple"'
    
    def _generate_elegant_code(self, nl: str) -> str:
        """Generate elegant Python implementation"""
        if 'todo' in nl.lower():
            return 'lambda x: {"tasks": [], "completed": [], "beauty": True}'
        elif 'note' in nl.lower():
            return 'lambda x: {"content": "", "created": "now", "essence": "captured"}'
        else:
            return 'lambda x: {"form": "beautiful", "function": "perfect"}'
    
    def _generate_practical_code(self, nl: str) -> str:
        """Generate practical Python implementation"""
        if 'todo' in nl.lower():
            return 'lambda x: {"add": "function", "done": "function"}'
        else:
            return 'lambda x: {"status": "working", "efficiency": 100}'
    
    def _generate_playful_code(self, nl: str) -> str:
        """Generate playful Python implementation"""
        if 'todo' in nl.lower():
            return 'lambda x: {"tasks": ["🎯", "✨", "🎉"], "mood": "joyful", "energy": "high"}'
        else:
            return 'lambda x: {"fun": True, "creativity": "unleashed", "joy": "∞"}'
    
    def _simulate_bruce_rating(self, wave: ExecutableWave) -> float:
        """Simulate Bruce's rating based on style and execution"""
        # Base preferences
        style_prefs = {
            'elegant': 90,
            'minimal': 85,
            'practical': 70,
            'playful': 60
        }
        
        base = style_prefs.get(wave.style, 50)
        
        # Boost if execution succeeded
        if wave.execution_result and 'Error' not in str(wave.execution_result):
            base += 10
        
        # Boost for certain patterns Bruce likes
        if isinstance(wave.execution_result, dict):
            result = wave.execution_result
            if result.get('asf_result') == result.get('py_result'):
                base += 5  # Consistency bonus
        
        return min(100, max(0, base))
    
    def _infer_drives_from_execution(self, wave: ExecutableWave, satisfaction: float) -> List[str]:
        """Infer drives based on execution results and satisfaction"""
        drives = []
        
        if satisfaction > 0.8:
            if wave.style == 'elegant':
                drives.extend(['beauty', 'integration', 'refinement'])
            elif wave.style == 'minimal':
                drives.extend(['simplicity', 'clarity', 'essence'])
            
            # Execution-based drives
            if wave.execution_result and 'Error' not in str(wave.execution_result):
                drives.append('reliability')
                
        if satisfaction < 0.5:
            drives.append('improvement_needed')
            
        return drives or ['exploration']
    
    def _show_execution_learning(self):
        """Show what was learned from execution"""
        summary = self.si.get_invariant_summary()
        
        print("\n📊 Execution Learning Summary:")
        
        # Style preferences
        if 'keyword' in summary:
            styles = [p for p in summary['keyword'] if any(s in p['key'] for s in ['minimal', 'elegant', 'practical', 'playful'])]
            if styles:
                styles = sorted(styles, key=lambda x: -x['mean_satisfaction'])[:3]
                print("\nPreferred Styles (from execution):")
                for s in styles:
                    print(f"  {s['key']:15} {s['mean_satisfaction']:.0%}")
        
        # Execution drives
        if 'drive' in summary:
            drives = sorted(summary['drive'], key=lambda x: -x['mean_satisfaction'])[:5]
            print("\nKey Drives (from satisfaction):")
            for d in drives:
                print(f"  {d['key']:15} {d['mean_satisfaction']:.0%}")
    
    def run_demo(self):
        """Run the complete asfOS-based demo"""
        print("🎯 SPEAK IT, SEE IT, EXPERIENCE IT - asfOS Edition")
        print("=" * 60)
        print("Describe software → Compile to shapes → Execute in asfOS → Rate satisfaction")
        print()
        
        while True:
            try:
                # SPEAK IT
                nl_request = input("\n🎤 What software do you want? (or 'quit'): ").strip()
                if nl_request.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not nl_request:
                    continue
                
                # Generate executable waves
                waves = self.speak_it(nl_request)
                if not waves:
                    print("Failed to generate waves. Try a different request.")
                    continue
                
                # SEE IT
                self.see_it(waves)
                
                # EXPERIENCE IT (execute and rate)
                ratings = self.experience_it(waves)
                
                # LEARN IT
                self.learn_it(waves, ratings)
                
                # Show best wave
                best_wave = max(ratings.items(), key=lambda x: x[1])
                print(f"\n🏆 Best wave: {best_wave[0]} ({best_wave[1]:.0%} satisfaction)")
                
            except KeyboardInterrupt:
                print("\n\nDemo interrupted.")
                break
        
        print("\n✨ asfOS demo complete. Shapes executed, satisfaction learned.")
        self.kernel.close()
        self.si.close()

if __name__ == "__main__":
    demo = ASFOSExperience()
    demo.run_demo()