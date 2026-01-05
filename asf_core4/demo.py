"""Small end-to-end demo for ASF Core 4."""

from weighted_state import from_shapes, normalize_l2, prune
from operators import grover_step
from sampling import sample_counts
from transforms import walsh_hadamard


def demo_grover():
    shapes = ["A", "S", "(A)", "(AS)"]
    target = "S"

    state = from_shapes(shapes)
    for _ in range(2):
        state = grover_step(state, target)
        state = normalize_l2(prune(state))

    counts = sample_counts(state, n=2000, seed=7)
    print("Grover-like sampling counts:")
    for k in shapes:
        print(f"  {k}: {counts.get(k, 0)}")


def demo_walsh_hadamard():
    basis = ["00", "01", "10", "11"]
    state = {"00": 1.0}
    transformed = walsh_hadamard(state, basis)

    print("Walsh-Hadamard amplitudes:")
    for b in basis:
        amp = transformed.get(b, 0.0)
        print(f"  {b}: {amp}")


def main():
    demo_grover()
    print()
    demo_walsh_hadamard()


if __name__ == "__main__":
    main()
