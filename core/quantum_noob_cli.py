#!/usr/bin/env python3
"""
Quantum Affordances CLI for Noobs (and Bruce)

A friendly, approachable command-line interface for exploring quantum-style
affordances without needing a physics degree.

Usage:
    python quantum_noob_cli.py          # Interactive mode
    python quantum_noob_cli.py --help   # Show this help
    python quantum_noob_cli.py ladder   # Show the affordance ladder
    python quantum_noob_cli.py demo     # Run the speak-it-see-it demo
"""

import argparse
import random
import sys
from typing import Dict, List, Optional, Tuple

try:
    from .weighted_state import from_shapes, normalize_l1, normalize_l2, prune, apply_operator
    from .operators import mark, grover_step
    from .transforms import hadamard_pair, walsh_hadamard
    from .sampling import sample_counts, measure, probabilities
except ImportError:
    from weighted_state import from_shapes, normalize_l1, normalize_l2, prune, apply_operator
    from operators import mark, grover_step
    from transforms import hadamard_pair, walsh_hadamard
    from sampling import sample_counts, measure, probabilities


# =============================================================================
# HELP TEXT FOR QUANTUM NOOBS
# =============================================================================

LADDER = """
QUANTUM AFFORDANCES - THE PROGRESSIVE LADDER
=============================================

Think of this as climbing from "normal thinking" to "quantum thinking."
Each rung builds on the one below.

1. SUPERPOSITION - "Holding multiple possibilities at once"

   Everyday:     You're deciding between pizza or sushi for dinner.
                 Both exist in your mind simultaneously.

   Quantum:      Instead of picking one, we WEIGHT each option.
                 Pizza might be 70%, sushi 30%.

   Try it:       basis pizza sushi tacos
                 uniform
                 show

2. INTERFERENCE - "Some possibilities cancel out"

   Everyday:     Pros and cons canceling in a decision.
                 "I want pizza, but I had it yesterday" = weaker pizza signal.

   Quantum:      Amplitudes can have PHASE (positive/negative).
                 Negative pizza + positive pizza = zero pizza.

   Try it:       set pizza 0.5
                 set anti_pizza -0.5
                 show
                 prune

3. AMPLITUDE AMPLIFICATION - "Making good options louder"

   Everyday:     Your friend says "DEFINITELY get the sushi" and
                 suddenly sushi feels more prominent.

   Quantum:      The Grover algorithm marks desired items and
                 amplifies them through repeated iterations.

   Try it:       basis a b c d e f g h
                 uniform
                 grover d 3
                 show

4. TRANSFORMS - "Spreading possibilities around"

   Everyday:     Brainstorming: one idea spawns many related ideas.

   Quantum:      The Hadamard transform takes one definite state
                 and spreads it into a superposition.

   Try it:       basis 00 01 10 11
                 set 00 1.0
                 walsh
                 show

5. MEASUREMENT - "Collapsing to one outcome"

   Everyday:     You finally pick sushi. The decision is made.
                 Other possibilities disappear.

   Quantum:      Sampling from the probability distribution
                 gives you ONE outcome.

   Try it:       basis heads tails
                 uniform
                 normalize l2
                 sample 20

6. THE WHOLE DANCE

   Real quantum algorithms combine all of these:
   - Start with superposition (all possibilities)
   - Transform to spread/mix
   - Mark desired outcomes (interference)
   - Amplify the marked ones (Grover)
   - Measure to get your answer

   The magic: you didn't have to check every possibility one by one!
"""

COMMANDS_HELP = """
COMMANDS (type any of these at the quantum> prompt)
===================================================

SETTING UP YOUR POSSIBILITY SPACE:

  basis <items...>       Define what's possible
                         Example: basis red green blue

  uniform [weight]       Give equal weight to all basis items
                         Example: uniform

  set <item> <weight>    Set a specific item's amplitude
                         Example: set red 0.7

  add <item> <weight>    Add to an item's amplitude
                         Example: add blue 0.3

LOOKING AT YOUR STATE:

  show                   Display all items with their amplitudes and probabilities

  probs                  Just show probabilities (simpler view)

TRANSFORMATIONS:

  normalize l1|l2        Normalize amplitudes (make them sum/square-sum to 1)
                         l1 = sum of magnitudes = 1
                         l2 = sum of squared magnitudes = 1 (quantum-correct)

  prune [epsilon]        Remove near-zero amplitudes
                         Example: prune 0.01

  hadamard <a> <b>       Apply 2x2 Hadamard to two items
                         Mixes them together quantum-style

  walsh                  Apply Walsh-Hadamard to ALL basis items
                         Spreads everything around

QUANTUM SEARCH:

  mark <item>            Phase-flip the target (makes it "different")
                         Example: mark blue

  grover <item> <steps>  Run Grover search for target
                         Example: grover blue 3

SAMPLING & MEASUREMENT:

  measure [seed]         Take ONE sample from the distribution
                         Example: measure

  sample <n> [seed]      Take n samples and show counts
                         Example: sample 100

OTHER:

  reset                  Start over with empty state
  help                   Show this help
  ladder                 Show the progressive affordance ladder
  demo                   Run the speak-it-see-it experience demo
  quit / exit            Leave the program

TIPS FOR NOOBS:

  1. Always start with 'basis' to define your possibility space
  2. Use 'uniform' to give everything equal weight
  3. Use 'normalize l2' before sampling (quantum-correct)
  4. Use 'show' often to see what's happening
  5. Grover works best with sqrt(N) iterations (N = basis size)
"""


# =============================================================================
# VISUAL REPRESENTATIONS
# =============================================================================

def visualize_state_bars(state: Dict[str, complex], width: int = 40) -> str:
    """Show state as horizontal bar chart."""
    if not state:
        return "(empty state)"

    probs = probabilities(state)
    max_prob = max(probs.values()) if probs else 1.0

    lines = []
    for k in sorted(state.keys()):
        amp = state[k]
        prob = probs.get(k, 0.0)
        bar_len = int((prob / max_prob) * width) if max_prob > 0 else 0
        bar = "" * bar_len
        phase_indicator = "+" if amp.real >= 0 else "-"
        lines.append(f"  {k:12} {phase_indicator} |{bar}| {prob:.3f}")

    return "\n".join(lines)


def visualize_state_wave(state: Dict[str, complex], width: int = 60) -> str:
    """Show state as a waveform visualization."""
    if not state:
        return "(empty state)"

    keys = sorted(state.keys())
    lines = []

    # Header
    lines.append("  WAVEFORM VIEW (amplitude oscillation)")
    lines.append("  " + "-" * width)

    center = width // 2
    for k in keys:
        amp = state[k]
        # Map amplitude to position: negative left, positive right
        real_part = amp.real if isinstance(amp, complex) else amp
        pos = int(center + real_part * (center - 2))
        pos = max(1, min(width - 2, pos))

        line = list(" " * width)
        line[center] = "|"  # center line
        if pos < center:
            for i in range(pos, center):
                line[i] = "~"
            line[pos] = "<"
        elif pos > center:
            for i in range(center + 1, pos + 1):
                line[i] = "~"
            line[pos] = ">"
        else:
            line[center] = "O"

        lines.append(f"  {k:10} {''.join(line)}")

    lines.append("  " + "-" * width)
    lines.append(f"  {'negative':^{center}}|{'positive':^{center}}")

    return "\n".join(lines)


def visualize_state_circle(state: Dict[str, complex]) -> str:
    """Show state as probability pie (text-based)."""
    if not state:
        return "(empty state)"

    probs = probabilities(state)
    total_chars = 50

    lines = ["  PROBABILITY PIE"]
    lines.append("  [", )

    chars = []
    for k in sorted(state.keys()):
        prob = probs.get(k, 0.0)
        n_chars = max(1, int(prob * total_chars))
        char = k[0].upper() if k else "?"
        chars.extend([char] * n_chars)

    lines[-1] += "".join(chars[:total_chars])
    lines[-1] += "]"

    # Legend
    lines.append("  Legend:")
    for k in sorted(state.keys()):
        prob = probs.get(k, 0.0)
        char = k[0].upper() if k else "?"
        lines.append(f"    {char} = {k} ({prob:.1%})")

    return "\n".join(lines)


# =============================================================================
# RUNTIME
# =============================================================================

class QuantumNoobRuntime:
    def __init__(self):
        self.state: Dict[str, complex] = {}
        self.basis: List[str] = []
        self.history: List[Tuple[str, Dict[str, complex]]] = []

    def save_history(self, action: str):
        """Save state to history for undo."""
        self.history.append((action, dict(self.state)))

    def run_interactive(self):
        """Interactive REPL mode."""
        print("\n" + "=" * 60)
        print("  QUANTUM AFFORDANCES CLI")
        print("  For noobs, dreamers, and Bruce Stagbrook")
        print("=" * 60)
        print("\n  Type 'help' for commands, 'ladder' for concepts, 'demo' for magic\n")

        while True:
            try:
                line = input("quantum> ").strip()
                if not line or line.startswith("#"):
                    continue
                if not self.dispatch(line):
                    break
            except KeyboardInterrupt:
                print("\n  (Use 'quit' to exit)")
            except Exception as e:
                print(f"  Error: {e}")

    def dispatch(self, line: str) -> bool:
        """Process a command. Returns False to exit."""
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            print("  Farewell, quantum explorer!")
            return False

        if cmd == "help":
            print(COMMANDS_HELP)
        elif cmd == "ladder":
            print(LADDER)
        elif cmd == "demo":
            self.run_demo()
        elif cmd == "basis":
            self.cmd_basis(args)
        elif cmd == "uniform":
            self.cmd_uniform(args)
        elif cmd == "set":
            self.cmd_set(args)
        elif cmd == "add":
            self.cmd_add(args)
        elif cmd == "show":
            self.cmd_show()
        elif cmd == "probs":
            self.cmd_probs()
        elif cmd == "wave":
            self.cmd_wave()
        elif cmd == "pie":
            self.cmd_pie()
        elif cmd == "normalize":
            self.cmd_normalize(args)
        elif cmd == "prune":
            self.cmd_prune(args)
        elif cmd == "hadamard":
            self.cmd_hadamard(args)
        elif cmd == "walsh":
            self.cmd_walsh()
        elif cmd == "mark":
            self.cmd_mark(args)
        elif cmd == "grover":
            self.cmd_grover(args)
        elif cmd == "measure":
            self.cmd_measure(args)
        elif cmd == "sample":
            self.cmd_sample(args)
        elif cmd == "reset":
            self.cmd_reset()
        else:
            print(f"  Unknown command: {cmd}  (type 'help' for commands)")

        return True

    def cmd_basis(self, args: List[str]):
        if not args:
            print("  Usage: basis <item1> <item2> ...")
            return
        self.basis = args
        print(f"  Basis set: {', '.join(self.basis)}")

    def cmd_uniform(self, args: List[str]):
        if not self.basis:
            print("  Error: Set basis first (e.g., 'basis a b c')")
            return
        weight = float(args[0]) if args else 1.0
        self.save_history("uniform")
        self.state = from_shapes(self.basis, weight=weight)
        print(f"  Uniform superposition over {len(self.basis)} items")

    def cmd_set(self, args: List[str]):
        if len(args) != 2:
            print("  Usage: set <item> <weight>")
            return
        item, weight = args[0], float(args[1])
        self.save_history(f"set {item}")
        self.state[item] = weight
        print(f"  {item} = {weight}")

    def cmd_add(self, args: List[str]):
        if len(args) != 2:
            print("  Usage: add <item> <weight>")
            return
        item, weight = args[0], float(args[1])
        self.save_history(f"add {item}")
        self.state[item] = self.state.get(item, 0.0) + weight
        print(f"  {item} now = {self.state[item]}")

    def cmd_show(self):
        if not self.state:
            print("  (empty state - use 'basis' and 'uniform' to start)")
            return
        print("\n  STATE (amplitude and probability):")
        print(visualize_state_bars(self.state))
        print()

    def cmd_probs(self):
        if not self.state:
            print("  (empty state)")
            return
        probs = probabilities(self.state)
        print("\n  PROBABILITIES:")
        for k in sorted(probs.keys()):
            print(f"    {k}: {probs[k]:.1%}")
        print()

    def cmd_wave(self):
        if not self.state:
            print("  (empty state)")
            return
        print()
        print(visualize_state_wave(self.state))
        print()

    def cmd_pie(self):
        if not self.state:
            print("  (empty state)")
            return
        print()
        print(visualize_state_circle(self.state))
        print()

    def cmd_normalize(self, args: List[str]):
        if not args:
            print("  Usage: normalize l1|l2")
            return
        mode = args[0].lower()
        self.save_history(f"normalize {mode}")
        if mode == "l1":
            self.state = normalize_l1(self.state)
        elif mode == "l2":
            self.state = normalize_l2(self.state)
        else:
            print("  Use 'l1' or 'l2'")
            return
        print(f"  Normalized ({mode})")

    def cmd_prune(self, args: List[str]):
        eps = float(args[0]) if args else 1e-12
        self.save_history("prune")
        before = len(self.state)
        self.state = prune(self.state, eps=eps)
        after = len(self.state)
        print(f"  Pruned: {before} -> {after} items (eps={eps})")

    def cmd_hadamard(self, args: List[str]):
        if len(args) != 2:
            print("  Usage: hadamard <item1> <item2>")
            return
        self.save_history(f"hadamard {args[0]} {args[1]}")
        op = hadamard_pair(args[0], args[1])
        self.state = apply_operator(self.state, op)
        print(f"  Hadamard applied to {args[0]}, {args[1]}")

    def cmd_walsh(self):
        if not self.basis:
            print("  Error: Set basis first")
            return
        self.save_history("walsh")
        self.state = walsh_hadamard(self.state, self.basis)
        print("  Walsh-Hadamard transform applied")

    def cmd_mark(self, args: List[str]):
        if len(args) != 1:
            print("  Usage: mark <target>")
            return
        target = args[0]
        self.save_history(f"mark {target}")
        self.state = apply_operator(self.state, mark(lambda s: s == target))
        print(f"  Marked {target} (phase flipped)")

    def cmd_grover(self, args: List[str]):
        if len(args) != 2:
            print("  Usage: grover <target> <steps>")
            return
        target = args[0]
        steps = int(args[1])
        self.save_history(f"grover {target} {steps}")
        for _ in range(steps):
            self.state = grover_step(self.state, target)
        print(f"  Grover search: {steps} iterations for '{target}'")

        # Show result
        probs = probabilities(self.state)
        target_prob = probs.get(target, 0.0)
        print(f"  Target probability now: {target_prob:.1%}")

    def cmd_measure(self, args: List[str]):
        if not self.state:
            print("  (empty state)")
            return
        seed = int(args[0]) if args else random.randint(0, 999999)
        outcome = measure(self.state, rng=random.Random(seed))
        print(f"\n  MEASUREMENT: {outcome}")
        print("  (wave function collapsed!)\n")

    def cmd_sample(self, args: List[str]):
        if not args:
            print("  Usage: sample <n> [seed]")
            return
        n = int(args[0])
        seed = int(args[1]) if len(args) > 1 else random.randint(0, 999999)
        counts = sample_counts(self.state, n=n, seed=seed)

        print(f"\n  SAMPLING: {n} measurements")
        for k in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
            bar = "" * (counts[k] * 40 // n)
            print(f"    {k:12} {bar} ({counts[k]})")
        print()

    def cmd_reset(self):
        self.state = {}
        self.basis = []
        self.history = []
        print("  State cleared. Ready for new exploration!")

    def run_demo(self):
        """The speak-it-see-it-experience-it demo."""
        from speak_see_experience import SpeakSeeExperienceDemo
        demo = SpeakSeeExperienceDemo()
        demo.run()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Affordances CLI for Noobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantum_noob_cli.py              # Interactive mode
  python quantum_noob_cli.py ladder       # Show the affordance ladder
  python quantum_noob_cli.py demo         # Speak-it-see-it experience
        """
    )
    parser.add_argument("command", nargs="?", help="Command to run (ladder, demo)")
    args = parser.parse_args()

    runtime = QuantumNoobRuntime()

    if args.command == "ladder":
        print(LADDER)
    elif args.command == "demo":
        runtime.run_demo()
    elif args.command == "help":
        print(COMMANDS_HELP)
    elif args.command:
        runtime.dispatch(args.command)
    else:
        runtime.run_interactive()


if __name__ == "__main__":
    main()
