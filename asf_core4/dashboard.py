#!/usr/bin/env python3
"""
Visualization Dashboard for Bruce's Structural Intelligence

Shows:
- Learned invariants and their strengths
- Satisfaction trends over time
- Gold exemplars (what delights Bruce)
- Emerging patterns
- Rhythm/time patterns
- The shape of Bruce's preferences

This makes the learning VISIBLE so Bruce can see himself reflected back.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .structural_intelligence import StructuralIntelligence
    from .auto_trainer import RhythmTracker
except ImportError:
    from structural_intelligence import StructuralIntelligence
    from auto_trainer import RhythmTracker


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    """Create a horizontal bar visualization."""
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def sparkline(values: List[float], width: int = 20) -> str:
    """Create a sparkline from values."""
    if not values:
        return "─" * width

    # Normalize to 0-1
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        normalized = [0.5] * len(values)
    else:
        normalized = [(v - min_v) / (max_v - min_v) for v in values]

    # Resample to width
    step = len(normalized) / width
    resampled = []
    for i in range(width):
        idx = int(i * step)
        resampled.append(normalized[min(idx, len(normalized) - 1)])

    # Convert to sparkline characters
    chars = "▁▂▃▄▅▆▇█"
    return "".join(chars[int(v * 7)] for v in resampled)


def box(title: str, content: str, width: int = 60) -> str:
    """Create a boxed section."""
    lines = content.split("\n")
    top = f"┌─ {title} " + "─" * (width - len(title) - 4) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"

    middle = []
    for line in lines:
        if len(line) > width - 4:
            line = line[:width - 7] + "..."
        middle.append(f"│ {line:<{width - 4}} │")

    return "\n".join([top] + middle + [bottom])


# =============================================================================
# DASHBOARD
# =============================================================================

class Dashboard:
    """Interactive visualization dashboard."""

    def __init__(self,
                 si_db: str = "bruce_sovereign.db",
                 rhythm_db: str = "rhythm.db"):
        self.si = StructuralIntelligence(db_path=si_db)
        self.rhythm = RhythmTracker(db_path=rhythm_db)

    def show_full_dashboard(self):
        """Display the complete dashboard."""
        print("\n" + "=" * 70)
        print("  BRUCE STAGBROOK - STRUCTURAL INTELLIGENCE DASHBOARD")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("=" * 70)

        self.show_satisfaction_trend()
        self.show_top_invariants()
        self.show_gold_exemplars()
        self.show_emerging_patterns()
        self.show_rhythm_patterns()
        self.show_drives_and_dreams()

        print("\n" + "=" * 70)

    def show_satisfaction_trend(self):
        """Show recent satisfaction trend."""
        trend = self.si.get_satisfaction_trend(50)

        if not trend:
            print("\n  No satisfaction data yet. Start interacting!")
            return

        avg = sum(trend) / len(trend)
        recent_avg = sum(trend[-10:]) / len(trend[-10:]) if len(trend) >= 10 else avg

        direction = "↑" if recent_avg > avg else ("↓" if recent_avg < avg else "→")

        content = f"""
Overall: {avg:.0%}  Recent: {recent_avg:.0%} {direction}

{sparkline(trend, 50)}
{'oldest':<25}{'newest':>25}

Last {len(trend)} interactions
"""
        print(box("SATISFACTION TREND", content.strip()))

    def show_top_invariants(self):
        """Show the strongest learned invariants."""
        summary = self.si.get_invariant_summary()

        content_lines = []

        # Styles
        if "keyword" in summary:
            content_lines.append("KEYWORDS (what resonates):")
            for p in sorted(summary["keyword"], key=lambda x: -x["mean_satisfaction"])[:5]:
                content_lines.append(f"  {p['key']:<20} {bar(p['mean_satisfaction'], 15)} {p['mean_satisfaction']:.0%}")

        # Drives
        if "drive" in summary:
            content_lines.append("\nDRIVES (what matters):")
            for p in sorted(summary["drive"], key=lambda x: -x["mean_satisfaction"])[:5]:
                content_lines.append(f"  {p['key']:<20} {bar(p['mean_satisfaction'], 15)} {p['mean_satisfaction']:.0%}")

        if not content_lines:
            content_lines.append("No invariants learned yet.")

        print(box("TOP INVARIANTS", "\n".join(content_lines)))

    def show_gold_exemplars(self):
        """Show highest satisfaction examples."""
        gold = self.si.get_gold_exemplars(5)

        if not gold:
            content = "No gold exemplars yet. Keep creating!"
        else:
            lines = []
            for g in gold:
                utterance = g["utterance"][:45] + "..." if len(g["utterance"]) > 45 else g["utterance"]
                lines.append(f"  {g['satisfaction']:.0%} │ {utterance}")
            content = "\n".join(lines)

        print(box("GOLD EXEMPLARS (Bruce's Delights)", content))

    def show_emerging_patterns(self):
        """Show patterns that are starting to form."""
        summary = self.si.get_invariant_summary()

        # Find patterns with moderate confidence (emerging, not established)
        emerging = []
        for ptype, patterns in summary.items():
            for p in patterns:
                if 0.2 < p["confidence"] < 0.6 and p["samples"] >= 3:
                    emerging.append({
                        "type": ptype,
                        "key": p["key"],
                        "satisfaction": p["mean_satisfaction"],
                        "confidence": p["confidence"],
                        "samples": p["samples"]
                    })

        emerging.sort(key=lambda x: -x["satisfaction"])

        if not emerging:
            content = "No emerging patterns yet. More data needed."
        else:
            lines = []
            for e in emerging[:5]:
                conf_bar = "●" * int(e["confidence"] * 5) + "○" * (5 - int(e["confidence"] * 5))
                lines.append(f"  {e['type']:<8} {e['key']:<15} {e['satisfaction']:.0%} [{conf_bar}]")
            content = "\n".join(lines)

        print(box("EMERGING PATTERNS (forming, not yet solid)", content))

    def show_rhythm_patterns(self):
        """Show time-based patterns."""
        best_times = self.rhythm.get_best_times(6)

        if not best_times or all(t["samples"] == 0 for t in best_times):
            content = "No rhythm data yet. Keep working at different times!"
        else:
            lines = []
            for t in best_times:
                if t["samples"] > 0:
                    lines.append(f"  {t['type']:<5} {t['key']:<10} {bar(t['satisfaction'], 15)} ({t['samples']} samples)")
            content = "\n".join(lines) if lines else "Gathering rhythm data..."

        current_boost = self.rhythm.get_current_context_boost()
        content += f"\n\nCurrent time context: {current_boost:+.0%} satisfaction boost"

        print(box("RHYTHM PATTERNS (when Bruce thrives)", content))

    def show_drives_and_dreams(self):
        """Show Bruce's core drives and dreams from the learning."""
        summary = self.si.get_invariant_summary()

        lines = []

        if "drive" in summary:
            lines.append("CORE DRIVES:")
            top_drives = sorted(summary["drive"], key=lambda x: -x["mean_satisfaction"])[:5]
            for d in top_drives:
                lines.append(f"  ◆ {d['key']}")

        if "dream" in summary:
            lines.append("\nASPIRATIONS:")
            top_dreams = sorted(summary["dream"], key=lambda x: -x["mean_satisfaction"])[:3]
            for d in top_dreams:
                lines.append(f"  ★ {d['key']}")

        if "commitment" in summary:
            lines.append("\nCOMMITMENTS:")
            top_commits = sorted(summary["commitment"], key=lambda x: -x["mean_satisfaction"])[:3]
            for c in top_commits:
                lines.append(f"  ✓ {c['key']}")

        if not lines:
            lines.append("Drives and dreams will emerge from your interactions.")

        print(box("BRUCE'S SOUL MAP", "\n".join(lines)))

    def close(self):
        """Clean up resources."""
        self.si.close()


# =============================================================================
# LIVE MONITOR
# =============================================================================

class LiveMonitor:
    """Continuously updating dashboard."""

    def __init__(self, dashboard: Dashboard, refresh_seconds: int = 5):
        self.dashboard = dashboard
        self.refresh_seconds = refresh_seconds
        self.running = False

    def run(self):
        """Run the live monitor."""
        import os
        self.running = True

        print("\n  Live monitor starting. Press Ctrl+C to stop.\n")

        try:
            while self.running:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')

                # Show dashboard
                self.dashboard.show_full_dashboard()

                print(f"\n  Refreshing in {self.refresh_seconds}s... (Ctrl+C to stop)")
                time.sleep(self.refresh_seconds)

        except KeyboardInterrupt:
            self.running = False
            print("\n\n  Live monitor stopped.")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Structural Intelligence Dashboard")
    parser.add_argument("--live", action="store_true", help="Run live updating dashboard")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval for live mode")
    parser.add_argument("--db", default="bruce_sovereign.db", help="Path to SI database")
    args = parser.parse_args()

    dashboard = Dashboard(si_db=args.db)

    if args.live:
        monitor = LiveMonitor(dashboard, refresh_seconds=args.refresh)
        monitor.run()
    else:
        dashboard.show_full_dashboard()

    dashboard.close()


if __name__ == "__main__":
    main()
