#!/usr/bin/env python3
"""
Bruce's Unified CLI - One entry point for everything.

Commands:
  bruce speak      - Speak-it-see-it-experience-it demo
  bruce quantum    - Quantum affordances for noobs
  bruce dashboard  - View your learned patterns
  bruce train      - Explicit training session
  bruce auto       - Start auto-trainer daemon
  bruce status     - Quick status check
  bruce wave       - Generate waves for a request
  bruce search     - Quantum search the catalog
  bruce field      - Show/edit your sovereign field

This is YOUR interface, Bruce. It learns YOU.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))


def cmd_speak(args):
    """Run the speak-it-see-it-experience-it demo."""
    from speak_see_experience import SpeakSeeExperienceDemo
    demo = SpeakSeeExperienceDemo()
    demo.run()


def cmd_quantum(args):
    """Run the quantum affordances CLI."""
    from quantum_noob_cli import QuantumNoobRuntime

    runtime = QuantumNoobRuntime()

    if args.ladder:
        from quantum_noob_cli import LADDER
        print(LADDER)
    elif args.command:
        runtime.dispatch(" ".join(args.command))
    else:
        runtime.run_interactive()


def cmd_dashboard(args):
    """Show the visualization dashboard."""
    from dashboard import Dashboard, LiveMonitor

    dashboard = Dashboard(si_db=args.db)

    if args.live:
        monitor = LiveMonitor(dashboard, refresh_seconds=args.refresh)
        monitor.run()
    else:
        dashboard.show_full_dashboard()

    dashboard.close()


def cmd_train(args):
    """Run an explicit training session."""
    from structural_intelligence import StructuralIntelligence, DeepSignal
    from speak_see_experience import WaveGenerator

    si = StructuralIntelligence(db_path=args.db)
    gen = WaveGenerator()

    print("\n" + "=" * 60)
    print("  EXPLICIT TRAINING SESSION")
    print("  Tell me what you want, rate the results, I learn YOU.")
    print("=" * 60)
    print("\n  Type 'done' when finished.\n")

    while True:
        try:
            request = input("  What do you want? > ").strip()
            if not request or request.lower() == 'done':
                break

            # Generate waves
            waves = gen.generate_waves(request, n=3)

            print("\n  Options:")
            for i, w in enumerate(waves, 1):
                print(f"\n  [{i}] {w.description}")
                print(f"      Style: {w.implementation_style}")
                print(f"      Features: {', '.join(w.features[:3])}")

            # Get choice
            choice = input("\n  Which one? (1-3, or 0 for none): ").strip()
            if choice == '0':
                print("  None selected. Let's try another request.")
                continue

            try:
                idx = int(choice) - 1
                chosen = waves[idx]
                rejected = [w for i, w in enumerate(waves) if i != idx]
            except (ValueError, IndexError):
                print("  Invalid choice.")
                continue

            # Get satisfaction
            sat_str = input(f"  Satisfaction with {chosen.wave_id} (0-10): ").strip()
            try:
                satisfaction = float(sat_str) / 10.0
            except ValueError:
                satisfaction = 0.7

            # Get optional disclosure
            disclosure = input("  Any feedback? (optional): ").strip()

            # Get drives
            drives_str = input("  What drives this? (comma-separated, optional): ").strip()
            drives = [d.strip() for d in drives_str.split(",") if d.strip()] if drives_str else []

            # Record
            deep = DeepSignal(
                pressure="training session",
                urgency="learning",
                prior_blockers="",
                stakeholders=["bruce_stagbrook"],
                drives=drives,
                commitments=[],
                dreams=[]
            )

            si.record_interaction(
                utterance=request,
                shape_key=chosen.shape_key,
                satisfaction=satisfaction,
                deep_signals=deep,
                full_disclosure=disclosure
            )

            # Record rejections
            for rej in rejected:
                si.record_interaction(
                    utterance=request,
                    shape_key=rej.shape_key,
                    satisfaction=0.3,
                    deep_signals=deep,
                    full_disclosure="rejected"
                )

            print(f"\n  Recorded! {chosen.wave_id} at {satisfaction:.0%}")
            print()

        except (KeyboardInterrupt, EOFError):
            break

    print("\n  Training session complete.")
    si.close()


def cmd_auto(args):
    """Start the auto-trainer daemon."""
    from structural_intelligence import StructuralIntelligence
    from auto_trainer import AutoTrainer, PassiveObserver

    si = StructuralIntelligence(db_path=args.db)
    trainer = AutoTrainer(si)

    print("\n  Auto-trainer started.")
    print("  Watching for your actions...")
    print("  Press Ctrl+C to stop.\n")

    # In a real implementation, this would hook into file watchers,
    # clipboard monitors, etc. For now, just keep alive.
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Auto-trainer stopped.")

    si.close()


def cmd_status(args):
    """Quick status check."""
    from structural_intelligence import StructuralIntelligence

    si = StructuralIntelligence(db_path=args.db)

    trend = si.get_satisfaction_trend(20)
    gold = si.get_gold_exemplars(3)
    summary = si.get_invariant_summary()

    print("\n  QUICK STATUS")
    print("  " + "-" * 40)

    if trend:
        avg = sum(trend) / len(trend)
        print(f"  Satisfaction: {avg:.0%} (last {len(trend)} interactions)")
    else:
        print("  Satisfaction: No data yet")

    total_invariants = sum(len(patterns) for patterns in summary.values())
    print(f"  Invariants learned: {total_invariants}")

    print(f"  Gold exemplars: {len(gold)}")

    if gold:
        print(f"\n  Latest delight: {gold[0]['utterance'][:40]}...")

    si.close()


def cmd_wave(args):
    """Generate waves for a request."""
    from speak_see_experience import WaveGenerator

    gen = WaveGenerator()
    request = " ".join(args.request) if args.request else input("  Request: ").strip()

    if not request:
        print("  No request provided.")
        return

    waves = gen.generate_waves(request, n=args.count)

    print(f"\n  Waves for: \"{request}\"\n")
    for i, w in enumerate(waves, 1):
        print(f"  [{i}] {w.description}")
        print(f"      {w.implementation_style}")
        for line in w.visual_sketch.split('\n')[:8]:
            if line.strip():
                print(f"      {line}")
        print()


def cmd_search(args):
    """Quantum search the catalog."""
    from structural_intelligence import StructuralIntelligence
    from speak_see_experience import WaveGenerator

    si = StructuralIntelligence(db_path=args.db)
    gen = WaveGenerator()

    query = " ".join(args.query) if args.query else input("  Search: ").strip()

    if not query:
        print("  No query provided.")
        si.close()
        return

    # Generate candidate shapes
    waves = gen.generate_waves(query, n=8)
    candidates = [w.shape_key for w in waves]

    # Quantum search
    results = si.quantum_search(candidates, query, iterations=3)

    print(f"\n  Quantum search results for: \"{query}\"\n")

    # Sort by probability
    sorted_results = sorted(results.items(), key=lambda x: -x[1])

    for shape_key, prob in sorted_results[:5]:
        # Find matching wave
        wave = next((w for w in waves if w.shape_key == shape_key), None)
        if wave:
            print(f"  {prob:.1%} │ {wave.description}")
            print(f"        │ {wave.implementation_style}")
        else:
            print(f"  {prob:.1%} │ {shape_key[:16]}...")

    si.close()


def cmd_field(args):
    """Show or edit the sovereign field."""
    field_path = Path(__file__).parent / "fields" / "bruce_sovereign_field.json"

    if not field_path.exists():
        print("  Sovereign field not found. Creating default...")
        # Would create default here
        return

    with open(field_path) as f:
        field = json.load(f)

    print("\n  BRUCE STAGBROOK SOVEREIGN FIELD")
    print("  " + "=" * 50)

    print(f"\n  Philosophy: {field['philosophy']['core']}")

    print("\n  Sources:")
    for src in field['philosophy']['sources']:
        print(f"    • {src}")

    print("\n  Key Principles:")
    for p in field['philosophy']['key_principles']:
        print(f"    ◆ {p}")

    print("\n  Properties:")
    for prop in field['properties']:
        print(f"    {prop['name']:<15} threshold: {prop['threshold']}")

    print("\n  Anti-Patterns (REJECTED):")
    for anti in field['anti_patterns']['rejected']:
        print(f"    ✗ {anti}")


def main():
    parser = argparse.ArgumentParser(
        prog="bruce",
        description="Bruce's unified interface - learns YOU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bruce speak                  # Interactive speak-it-see-it demo
  bruce quantum                # Quantum affordances REPL
  bruce quantum --ladder       # Show the quantum ladder
  bruce dashboard              # View your patterns
  bruce dashboard --live       # Live updating dashboard
  bruce train                  # Explicit training session
  bruce status                 # Quick status check
  bruce wave "calm todo app"   # Generate waves for a request
  bruce search "peaceful"      # Quantum search the catalog
  bruce field                  # Show your sovereign field
        """
    )

    parser.add_argument("--db", default="bruce_sovereign.db",
                        help="Path to structural intelligence database")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # speak
    sp_speak = subparsers.add_parser("speak", help="Speak-it-see-it-experience-it demo")
    sp_speak.set_defaults(func=cmd_speak)

    # quantum
    sp_quantum = subparsers.add_parser("quantum", help="Quantum affordances CLI")
    sp_quantum.add_argument("--ladder", action="store_true", help="Show the affordance ladder")
    sp_quantum.add_argument("command", nargs="*", help="Command to run")
    sp_quantum.set_defaults(func=cmd_quantum)

    # dashboard
    sp_dash = subparsers.add_parser("dashboard", help="View your learned patterns")
    sp_dash.add_argument("--live", action="store_true", help="Live updating mode")
    sp_dash.add_argument("--refresh", type=int, default=10, help="Refresh interval")
    sp_dash.set_defaults(func=cmd_dashboard)

    # train
    sp_train = subparsers.add_parser("train", help="Explicit training session")
    sp_train.set_defaults(func=cmd_train)

    # auto
    sp_auto = subparsers.add_parser("auto", help="Start auto-trainer daemon")
    sp_auto.set_defaults(func=cmd_auto)

    # status
    sp_status = subparsers.add_parser("status", help="Quick status check")
    sp_status.set_defaults(func=cmd_status)

    # wave
    sp_wave = subparsers.add_parser("wave", help="Generate waves for a request")
    sp_wave.add_argument("request", nargs="*", help="The request")
    sp_wave.add_argument("--count", type=int, default=4, help="Number of waves")
    sp_wave.set_defaults(func=cmd_wave)

    # search
    sp_search = subparsers.add_parser("search", help="Quantum search the catalog")
    sp_search.add_argument("query", nargs="*", help="Search query")
    sp_search.set_defaults(func=cmd_search)

    # field
    sp_field = subparsers.add_parser("field", help="Show your sovereign field")
    sp_field.set_defaults(func=cmd_field)

    args = parser.parse_args()

    if args.command is None:
        # Default to status
        args.func = cmd_status
        cmd_status(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
