#!/usr/bin/env python3
"""
Speak It, See It, Experience It - Shape Wave Demo

The user describes what software they want.
The system generates multiple "wave" representations simultaneously.
User provides satisfaction feedback.
System converges on what delights.

This is the Triad in action:
- Symbolic Intelligence: composes shapes + receipts
- Structural Intelligence: learns satisfaction-shape invariants
- Meatsack (Bruce): provides full-disclosure feedback

The goal: software creation that feels like magic.
"""

import json
import random
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from .weighted_state import from_shapes, normalize_l2, prune, apply_operator
    from .operators import grover_step
    from .sampling import probabilities, sample_counts
    from .structural_intelligence import StructuralIntelligence, DeepSignal
    from .interactive_demo import InteractiveDemo, rate_experience
except ImportError:
    from weighted_state import from_shapes, normalize_l2, prune, apply_operator
    from operators import grover_step
    from sampling import probabilities, sample_counts
    from structural_intelligence import StructuralIntelligence, DeepSignal
    from interactive_demo import InteractiveDemo, rate_experience


# =============================================================================
# SHAPE REPRESENTATIONS
# =============================================================================

@dataclass
class ShapeWave:
    """A single wave/representation of the software."""
    wave_id: str
    description: str
    shape_key: str
    features: List[str]
    implementation_style: str
    visual_sketch: str
    dyck_skeleton: str
    amplitude: float = 1.0
    satisfaction: Optional[float] = None
    feedback: Optional[str] = None


@dataclass
class Receipt:
    """Full-disclosure record of an interaction."""
    utterance: str
    context: str
    waves: List[str]  # wave_ids
    satisfaction_scores: Dict[str, float]
    full_disclosure: str
    causal_pressure: str
    timestamp: float = field(default_factory=time.time)
    outlier_flag: bool = False

    def to_dict(self) -> dict:
        return {
            "utterance": self.utterance,
            "context": self.context,
            "waves": self.waves,
            "satisfaction_scores": self.satisfaction_scores,
            "full_disclosure": self.full_disclosure,
            "causal_pressure": self.causal_pressure,
            "timestamp": self.timestamp,
            "outlier_flag": self.outlier_flag
        }


# =============================================================================
# WAVE GENERATORS (These create different "views" of the software)
# =============================================================================

class WaveGenerator:
    """Generate multiple wave representations from a description."""

    STYLES = [
        ("minimal", "Bare essentials only, nothing extra"),
        ("feature_rich", "All the bells and whistles"),
        ("elegant", "Simple but sophisticated"),
        ("practical", "Gets the job done, pragmatic"),
        ("playful", "Fun, with personality"),
        ("professional", "Corporate-ready, polished"),
    ]

    FRAMEWORKS = [
        "vanilla_html_css_js",
        "react_tailwind",
        "vue_bootstrap",
        "svelte_minimal",
        "htmx_hyperscript",
        "pure_python_cli",
    ]

    def generate_waves(self, description: str, n: int = 4) -> List[ShapeWave]:
        """Generate n different wave representations."""
        waves = []

        # Extract key concepts from description
        keywords = self._extract_keywords(description)

        for i in range(n):
            style_name, style_desc = self.STYLES[i % len(self.STYLES)]
            framework = self.FRAMEWORKS[i % len(self.FRAMEWORKS)]

            wave = ShapeWave(
                wave_id=f"wave_{i+1}_{style_name[:3]}",
                description=f"{style_name.replace('_', ' ').title()} interpretation",
                shape_key=self._compute_shape_key(description, style_name, i),
                features=self._generate_features(keywords, style_name),
                implementation_style=f"{style_name} with {framework}",
                visual_sketch=self._generate_visual_sketch(keywords, style_name),
                dyck_skeleton=self._generate_dyck(keywords, style_name),
                amplitude=1.0 / n,  # Start with equal amplitudes
            )
            waves.append(wave)

        return waves

    def _extract_keywords(self, description: str) -> List[str]:
        """Extract key concepts from the description."""
        # Simple keyword extraction (in production, use NLP)
        stopwords = {'a', 'an', 'the', 'is', 'are', 'i', 'want', 'need', 'would', 'like', 'to', 'that', 'with', 'for', 'and', 'or', 'but'}
        words = description.lower().replace(',', ' ').replace('.', ' ').split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _compute_shape_key(self, description: str, style: str, seed: int) -> str:
        """Compute a deterministic shape key."""
        data = f"{description}:{style}:{seed}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _generate_features(self, keywords: List[str], style: str) -> List[str]:
        """Generate feature list based on style."""
        base_features = [f"Core: {kw}" for kw in keywords[:3]]

        if style == "minimal":
            return base_features
        elif style == "feature_rich":
            extras = ["Dark mode", "Export to PDF", "Keyboard shortcuts", "Undo/redo", "Auto-save"]
            return base_features + extras
        elif style == "elegant":
            return base_features + ["Smooth animations", "Clean typography"]
        elif style == "practical":
            return base_features + ["Error handling", "Loading states"]
        elif style == "playful":
            return base_features + ["Confetti on success", "Fun sounds", "Easter eggs"]
        else:  # professional
            return base_features + ["Accessibility (WCAG)", "Analytics", "Multi-language"]

    def _generate_visual_sketch(self, keywords: List[str], style: str) -> str:
        """Generate an ASCII visual sketch."""
        width = 50

        if style == "minimal":
            return self._sketch_minimal(keywords, width)
        elif style == "feature_rich":
            return self._sketch_feature_rich(keywords, width)
        elif style == "elegant":
            return self._sketch_elegant(keywords, width)
        elif style == "playful":
            return self._sketch_playful(keywords, width)
        else:
            return self._sketch_generic(keywords, width)

    def _sketch_minimal(self, keywords: List[str], w: int) -> str:
        title = " ".join(keywords[:2]).title() if keywords else "App"
        return f"""
+{'-' * (w-2)}+
|{title:^{w-2}}|
+{'-' * (w-2)}+
|{' ' * (w-2)}|
|  [ Main Content Area ]{' ' * (w-26)}|
|{' ' * (w-2)}|
|  [Action]{' ' * (w-12)}|
+{'-' * (w-2)}+"""

    def _sketch_feature_rich(self, keywords: List[str], w: int) -> str:
        title = " ".join(keywords[:2]).title() if keywords else "App"
        return f"""
+{'-' * (w-2)}+
|[=] {title:<{w-8}} [?][X]|
+{'-' * (w-2)}+
|[Nav] | [Search...        ] [+]|
|------+{'-' * (w-9)}|
| Home |                        |
| List |   Main Content Area    |
| Add  |                        |
| Cfg  |  [Card 1] [Card 2]    |
|------+{'-' * (w-9)}|
|[<] Page 1 of 5 [>]  [Export] |
+{'-' * (w-2)}+"""

    def _sketch_elegant(self, keywords: List[str], w: int) -> str:
        title = " ".join(keywords[:2]).title() if keywords else "App"
        return f"""

    {title}

    ~~~~~~~~~~~~~~~~~~~~~~~~

       Your content here,
       elegantly presented
       with breathing room.

    ~~~~~~~~~~~~~~~~~~~~~~~~

            [ Begin ]

"""

    def _sketch_playful(self, keywords: List[str], w: int) -> str:
        title = " ".join(keywords[:2]).title() if keywords else "App"
        return f"""
  *  .  *     {title}     *  .  *
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|   Welcome!                    |
|                               |
|      ,---.                    |
|     /     \\    Let's make    |
|    | ^   ^ |   something      |
|    |  ___  |   awesome!       |
|     \\_____/                   |
|                               |
|   [  Let's Go!  ]             |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
      *    .    *    .    *"""

    def _sketch_generic(self, keywords: List[str], w: int) -> str:
        title = " ".join(keywords[:2]).title() if keywords else "Application"
        return f"""
+{'-' * (w-2)}+
| {title:<{w-4}} |
+{'-' * (w-2)}+
|{' ' * (w-2)}|
|   [Main Content Area]{' ' * (w-24)}|
|{' ' * (w-2)}|
|   Feature 1 | Feature 2{' ' * (w-27)}|
|{' ' * (w-2)}|
+{'-' * (w-2)}+
| [Cancel]              [OK]   |
+{'-' * (w-2)}+"""

    def _generate_dyck(self, keywords: List[str], style: str) -> str:
        """Generate a Dyck skeleton representing the structure."""
        # App structure in Dyck: (root (header) (content (items)) (footer))
        n_items = 1 if style == "minimal" else (5 if style == "feature_rich" else 3)
        items = "()" * n_items  # Each () is a content item
        return f"(()({items})())"


# =============================================================================
# SATISFACTION TRACKER
# =============================================================================

@dataclass
class SatisfactionMap:
    """Tracks satisfaction invariants over time."""
    shape_scores: Dict[str, List[float]] = field(default_factory=dict)
    style_scores: Dict[str, List[float]] = field(default_factory=dict)
    feature_scores: Dict[str, List[float]] = field(default_factory=dict)
    receipts: List[Receipt] = field(default_factory=list)

    def record(self, wave: ShapeWave, score: float, feedback: str):
        """Record a satisfaction data point."""
        # By shape
        if wave.shape_key not in self.shape_scores:
            self.shape_scores[wave.shape_key] = []
        self.shape_scores[wave.shape_key].append(score)

        # By style
        style = wave.implementation_style.split()[0]
        if style not in self.style_scores:
            self.style_scores[style] = []
        self.style_scores[style].append(score)

        # By features
        for feature in wave.features:
            if feature not in self.feature_scores:
                self.feature_scores[feature] = []
            self.feature_scores[feature].append(score)

    def get_style_preference(self) -> Dict[str, float]:
        """Return average satisfaction per style."""
        return {
            style: sum(scores) / len(scores)
            for style, scores in self.style_scores.items()
            if scores
        }

    def get_feature_preference(self) -> Dict[str, float]:
        """Return average satisfaction per feature."""
        return {
            feat: sum(scores) / len(scores)
            for feat, scores in self.feature_scores.items()
            if scores
        }

    def predict_satisfaction(self, wave: ShapeWave) -> float:
        """Predict satisfaction for a new wave based on learned invariants."""
        predictions = []

        # Style-based prediction
        style = wave.implementation_style.split()[0]
        if style in self.style_scores and self.style_scores[style]:
            predictions.append(sum(self.style_scores[style]) / len(self.style_scores[style]))

        # Feature-based prediction
        for feat in wave.features:
            if feat in self.feature_scores and self.feature_scores[feat]:
                predictions.append(sum(self.feature_scores[feat]) / len(self.feature_scores[feat]))

        return sum(predictions) / len(predictions) if predictions else 0.5


# =============================================================================
# THE MAIN DEMO
# =============================================================================

class SpeakSeeExperienceDemo:
    """The main speak-it-see-it-experience-it demo."""

    def __init__(self, db_path: Optional[str] = None, si_db: str = "bruce_sovereign.db"):
        self.generator = WaveGenerator()
        self.satisfaction_map = SatisfactionMap()
        self.db_path = Path(db_path) if db_path else Path("speak_see_experience.json")
        self.si = StructuralIntelligence(db_path=si_db)
        self.demo = InteractiveDemo()
        self.load_state()

    def load_state(self):
        """Load previous satisfaction data if available."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.satisfaction_map.style_scores = data.get("style_scores", {})
                self.satisfaction_map.feature_scores = data.get("feature_scores", {})
                print(f"  (Loaded {len(self.satisfaction_map.style_scores)} style preferences)")
            except Exception:
                pass

    def save_state(self):
        """Save satisfaction data."""
        data = {
            "style_scores": self.satisfaction_map.style_scores,
            "feature_scores": self.satisfaction_map.feature_scores,
        }
        self.db_path.write_text(json.dumps(data, indent=2))

    def run(self):
        """Run the interactive demo."""
        print("\n" + "=" * 60)
        print("  SPEAK IT, SEE IT, EXPERIENCE IT")
        print("  Shape-Wave Software Creation")
        print("=" * 60)
        print("""
  Tell me what software you want.
  I'll show you multiple "waves" - different interpretations.
  Rate each one, and I'll learn what delights you.

  The goal: make software creation feel like magic.

  Type 'quit' to exit.
""")

        iteration = 0
        while True:
            iteration += 1

            # Get the user's desire
            print("-" * 60)
            try:
                desire = input("\n  What software do you want?\n  > ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not desire or desire.lower() in ('quit', 'exit', 'q'):
                break

            # Generate waves
            print("\n  Generating wave representations...")
            n_waves = 4 if iteration == 1 else 3
            waves = self.generator.generate_waves(desire, n=n_waves)

            # Weight by predicted satisfaction (structural intelligence)
            for wave in waves:
                predicted, _ = self.si.predict_satisfaction(
                    shape_key=wave.shape_key,
                    utterance=desire,
                    features=wave.features
                )
                if predicted == 0.5:
                    predicted = self.satisfaction_map.predict_satisfaction(wave)
                wave.amplitude *= (0.5 + predicted)  # Boost high-predicted waves

            # Normalize amplitudes
            total = sum(w.amplitude for w in waves)
            for wave in waves:
                wave.amplitude /= total

            # Show all waves
            print("\n" + "=" * 60)
            print("  WAVE SUPERPOSITION: Multiple interpretations exist simultaneously")
            print("=" * 60)

            for i, wave in enumerate(waves):
                self.display_wave(wave, i + 1)

            # Get satisfaction feedback for each
            print("\n  SATISFACTION FEEDBACK")
            print("  Rate each wave 0-10, or 'd' to demo it first")
            print()

            scored_waves = []
            for wave in waves:
                while True:
                    try:
                        response = input(f"  {wave.wave_id} (0-10 or 'd' to demo): ").strip().lower()
                        if response == 's':
                            break
                        elif response == 'd':
                            # Run interactive demo
                            print(f"\n  Starting demo for: {wave.description}")
                            demo_results = self.demo.run_demo(desire)
                            rating = rate_experience(demo_results, wave.description)
                            wave.satisfaction = rating
                            wave.feedback = f"Demo: {demo_results['interactions']} interactions, {demo_results['duration']:.1f}s"
                            self._record_feedback(desire, wave, wave.feedback)
                            scored_waves.append(wave)
                            break
                        elif response == 'f':
                            score = float(input(f"    Score (0-10): "))
                            feedback = input(f"    Full feedback: ")
                            wave.satisfaction = score / 10.0
                            wave.feedback = feedback
                            self._record_feedback(desire, wave, feedback)
                            scored_waves.append(wave)
                            break
                        else:
                            score = float(response)
                            if 0 <= score <= 10:
                                wave.satisfaction = score / 10.0
                                self._record_feedback(desire, wave, "")
                                scored_waves.append(wave)
                                break
                            else:
                                print("    Please enter 0-10 or 'd' for demo")
                    except ValueError:
                        print("    Please enter a number 0-10 or 'd' for demo")
                    except (KeyboardInterrupt, EOFError):
                        break

            # Find the best wave
            if scored_waves:
                best = max(scored_waves, key=lambda w: w.satisfaction or 0)
                print(f"\n  Best match: {best.wave_id} ({best.satisfaction:.0%} satisfaction)")

                if best.satisfaction >= 0.9:
                    print("\n  YOU'RE DELIGHTED!")
                    print("  This is the shape that brings you joy.")
                    self.display_wave(best, 0, detailed=True)

                    # Ask about convergence
                    proceed = input("\n  Ready to refine further? (y/n): ").strip().lower()
                    if proceed != 'y':
                        print("\n  Wave collapsed. Your software is manifest.")
                        self.save_state()
                        continue

                # Show learned preferences
                print("\n  STRUCTURAL INTELLIGENCE LEARNING:")
                prefs = self.satisfaction_map.get_style_preference()
                if prefs:
                    print("  Style preferences learned:")
                    for style, score in sorted(prefs.items(), key=lambda x: -x[1])[:3]:
                        print(f"    {style}: {score:.0%}")

            self.save_state()

        print("\n  Thank you for exploring with me!")
        print("  Your preferences have been saved for next time.")
        self.save_state()
        self.si.close()

    def run_demo(self):
        """Run a scripted demo without user input."""
        desire = "calm daily journal with gentle prompts and a sacred tone"
        print("\n" + "=" * 60)
        print("  SPEAK IT, SEE IT, EXPERIENCE IT (DEMO)")
        print("=" * 60)
        print(f"\n  Desire: {desire}")

        waves = self.generator.generate_waves(desire, n=3)
        for wave in waves:
            predicted, _ = self.si.predict_satisfaction(
                shape_key=wave.shape_key,
                utterance=desire,
                features=wave.features
            )
            wave.amplitude *= (0.5 + predicted)

        total = sum(w.amplitude for w in waves)
        for wave in waves:
            wave.amplitude /= total

        for i, wave in enumerate(waves, 1):
            self.display_wave(wave, i)

        # Simulated ratings
        demo_scores = [8, 6, 9]
        for wave, score in zip(waves, demo_scores):
            wave.satisfaction = score / 10.0
            self._record_feedback(desire, wave, "demo feedback")

        best = max(waves, key=lambda w: w.satisfaction or 0)
        print(f"\n  Best match: {best.wave_id} ({best.satisfaction:.0%} satisfaction)")
        self.save_state()
        self.si.close()

    def _record_feedback(self, desire: str, wave: ShapeWave, feedback: str):
        """Record feedback into both satisfaction map and structural intelligence."""
        self.satisfaction_map.record(wave, wave.satisfaction or 0.0, feedback)

        deep = DeepSignal(
            pressure="live creation",
            urgency="now",
            prior_blockers="",
            stakeholders=["bruce_stagbrook"],
            drives=[],
            commitments=[],
            dreams=[]
        )

        self.si.record_interaction(
            utterance=desire,
            shape_key=wave.shape_key,
            satisfaction=wave.satisfaction or 0.0,
            shape_dyck=wave.dyck_skeleton,
            deep_signals=deep,
            full_disclosure=feedback,
            features=wave.features
        )

        if wave.satisfaction and wave.satisfaction >= 0.9:
            self.si.set_label(wave.shape_key, desire, namespace="nl")

    def display_wave(self, wave: ShapeWave, index: int, detailed: bool = False):
        """Display a single wave representation."""
        print(f"\n  WAVE {index}: {wave.description}")
        print(f"  Style: {wave.implementation_style}")
        print(f"  Amplitude: {wave.amplitude:.1%}")

        if detailed or True:  # Always show visual
            print(f"\n  VISUAL SKETCH:")
            for line in wave.visual_sketch.split('\n'):
                print(f"    {line}")

        print(f"\n  FEATURES:")
        for feat in wave.features:
            print(f"    - {feat}")

        print(f"\n  SHAPE (Dyck): {wave.dyck_skeleton}")

        if wave.satisfaction is not None:
            print(f"  SATISFACTION: {wave.satisfaction:.0%}")
        if wave.feedback:
            print(f"  FEEDBACK: {wave.feedback}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Speak It, See It, Experience It demo")
    parser.add_argument("--demo", action="store_true", help="Run a scripted demo")
    parser.add_argument("--db", default="bruce_sovereign.db", help="Structural intelligence DB")
    args = parser.parse_args()

    demo = SpeakSeeExperienceDemo(si_db=args.db)
    if args.demo:
        demo.run_demo()
    else:
        demo.run()


if __name__ == "__main__":
    main()
