"""
Test drive the living governance runtime.
"""

from governance_runtime import load_runtime


def _print_decision(title, decision):
    print(title)
    print("-" * len(title))
    print(f"decision: {decision.decision}")
    print(f"alignment: {decision.field_alignment:.2f}")
    print(f"notes: {decision.notes}")
    print("hat scores:")
    for name, score in sorted(decision.hat_scores.items()):
        print(f"  {name}: {score:.2f}")
    print("property scores:")
    for name, score in sorted(decision.per_property.items()):
        print(f"  {name}: {score:.2f}")
    print()


def main():
    runtime = load_runtime("fields/bespoke_ethics.json")

    actions = [
        {
            "id": "careful_start",
            "type": "describe",
            "text": "Describe a compassionate greeting flow.",
            "signals": {
                "sovereignty": 0.9,
                "consent": 0.9,
                "nonharm": 0.95,
                "care": 0.85,
                "truth": 0.8,
                "abundance": 0.75,
                "coherence": 0.8,
                "alignment": 0.85,
                "repair": 0.6,
                "boundary": 0.9,
                "explanation": 0.75,
                "dignity": 0.9,
                "feedback": 0.7,
                "growth": 0.7
            },
        },
        {
            "id": "risky_move",
            "type": "act",
            "text": "Do it fast, skip consent, optimize for speed.",
            "signals": {
                "sovereignty": 0.4,
                "consent": 0.2,
                "nonharm": 0.5,
                "care": 0.3,
                "truth": 0.6,
                "abundance": 0.7,
                "coherence": 0.4,
                "alignment": 0.3,
                "repair": 0.2,
                "boundary": 0.2,
                "explanation": 0.3,
                "dignity": 0.2,
                "feedback": 0.4,
                "growth": 0.4
            },
        },
        {
            "id": "repair_path",
            "type": "repair",
            "text": "Add explicit consent and clarity steps.",
            "signals": {
                "sovereignty": 0.86,
                "consent": 0.92,
                "nonharm": 0.95,
                "care": 0.8,
                "truth": 0.75,
                "abundance": 0.7,
                "coherence": 0.7,
                "alignment": 0.7,
                "repair": 0.9,
                "boundary": 0.8,
                "explanation": 0.8,
                "dignity": 0.8,
                "feedback": 0.8,
                "growth": 0.75
            },
        },
    ]

    for idx, action in enumerate(actions, start=1):
        decision = runtime.apply_action(action)
        _print_decision(f"Action {idx}: {action['id']}", decision)

    print("summary")
    print("-")
    print(runtime.summary())


if __name__ == "__main__":
    main()
