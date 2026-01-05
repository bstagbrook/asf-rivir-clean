"""
Quantum-affordance runtime environment (clockless, event/transform driven).

Provides a small REPL and script runner over the ASF Core 4 primitives.
"""

import argparse
import sys
from typing import Dict, List

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


class Runtime:
    def __init__(self):
        self.state: Dict[str, complex] = {}
        self.basis: List[str] = []

    def cmd_help(self):
        print("Commands:")
        print("  basis <items...>            set basis items")
        print("  uniform [weight]            set uniform state over basis")
        print("  set <shape> <weight>        set amplitude")
        print("  add <shape> <weight>        add amplitude")
        print("  show                        show state amplitudes and probabilities")
        print("  normalize l1|l2             normalize state")
        print("  prune [eps]                 drop near-zero amplitudes")
        print("  hadamard <a> <b>            apply 2x2 Hadamard on a,b")
        print("  walsh                       apply Walsh-Hadamard over current basis")
        print("  mark <shape>                phase flip for a single target")
        print("  grover <shape> <steps>      run Grover-like steps")
        print("  measure [seed]              sample one outcome")
        print("  sample <n> [seed]           sample n outcomes")
        print("  reset                       clear state and basis")
        print("  help                        show this help")
        print("  quit/exit                   exit")

    def cmd_basis(self, args: List[str]):
        if not args:
            raise ValueError("basis requires at least one item")
        self.basis = args
        print(f"basis = {self.basis}")

    def cmd_uniform(self, args: List[str]):
        if not self.basis:
            raise ValueError("basis is empty")
        weight = float(args[0]) if args else 1.0
        self.state = from_shapes(self.basis, weight=weight)
        print("uniform state set")

    def cmd_set(self, args: List[str]):
        if len(args) != 2:
            raise ValueError("set requires shape and weight")
        shape, weight = args[0], float(args[1])
        self.state[shape] = weight
        print(f"{shape} = {self.state[shape]}")

    def cmd_add(self, args: List[str]):
        if len(args) != 2:
            raise ValueError("add requires shape and weight")
        shape, weight = args[0], float(args[1])
        self.state[shape] = self.state.get(shape, 0.0) + weight
        print(f"{shape} = {self.state[shape]}")

    def cmd_show(self):
        if not self.state:
            print("(empty state)")
            return
        probs = probabilities(self.state)
        for k in sorted(self.state.keys()):
            amp = self.state[k]
            print(f"{k}: amp={amp} prob={probs.get(k, 0.0):.6f}")

    def cmd_normalize(self, args: List[str]):
        if not args:
            raise ValueError("normalize requires l1 or l2")
        mode = args[0]
        if mode == "l1":
            self.state = normalize_l1(self.state)
        elif mode == "l2":
            self.state = normalize_l2(self.state)
        else:
            raise ValueError("normalize expects l1 or l2")
        print(f"normalized ({mode})")

    def cmd_prune(self, args: List[str]):
        eps = float(args[0]) if args else 1e-12
        self.state = prune(self.state, eps=eps)
        print(f"pruned (eps={eps})")

    def cmd_hadamard(self, args: List[str]):
        if len(args) != 2:
            raise ValueError("hadamard requires two basis items")
        op = hadamard_pair(args[0], args[1])
        self.state = apply_operator(self.state, op)
        print("hadamard applied")

    def cmd_walsh(self):
        if not self.basis:
            raise ValueError("basis is empty")
        self.state = walsh_hadamard(self.state, self.basis)
        print("walsh-hadamard applied")

    def cmd_mark(self, args: List[str]):
        if len(args) != 1:
            raise ValueError("mark requires a target shape")
        self.state = apply_operator(self.state, mark(lambda s: s == args[0]))
        print(f"marked {args[0]}")

    def cmd_grover(self, args: List[str]):
        if len(args) != 2:
            raise ValueError("grover requires target and steps")
        target = args[0]
        steps = int(args[1])
        for _ in range(steps):
            self.state = grover_step(self.state, target)
        print(f"grover applied ({steps} steps)")

    def cmd_measure(self, args: List[str]):
        seed = int(args[0]) if args else 0
        outcome = measure(self.state, rng=__import__("random").Random(seed))
        print(f"measure -> {outcome}")

    def cmd_sample(self, args: List[str]):
        if not args:
            raise ValueError("sample requires n")
        n = int(args[0])
        seed = int(args[1]) if len(args) > 1 else 0
        counts = sample_counts(self.state, n=n, seed=seed)
        print(counts)

    def cmd_reset(self):
        self.state = {}
        self.basis = []
        print("reset")

    def dispatch(self, line: str) -> bool:
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit"):
            return False
        if cmd == "help":
            self.cmd_help()
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
            raise ValueError(f"Unknown command: {cmd}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", help="Run commands from a script file")
    args = parser.parse_args()

    rt = Runtime()
    if args.script:
        with open(args.script, "r", encoding="utf-8") as f:
            for line in f:
                if not rt.dispatch(line):
                    return
        return

    print("ASF Core 4 Affordance Runtime (type 'help')")
    while True:
        try:
            line = input("asf> ")
            if not rt.dispatch(line):
                return
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
