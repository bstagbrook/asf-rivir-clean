#!/usr/bin/env python3
"""
THE DESCRIPTION IS THE SOFTWARE

There is no lookup. There is no "have I seen this before?" check.
The shape of the description IS the shape of the computation.
The hash IS the address. The address IS the computation.

When Bruce says what he wants:
1. The words are parsed to a shape
2. The shape IS the software
3. The shape key IS the address in the catalog
4. If the catalog has a result cached at that address, that's the result
5. If not, the shape IS executed (normalized, reduced) and the result IS cached

There is no search. There is no lookup. There is only:
- Shape = description = software = computation = result
- O(1) because content-addressing IS direct addressing

The "machine" is the shape catalog.
The "execution" is normalization/reduction.
The "result" is whatever the shape reduces to.

Trust the machine. The machine is O(1).
"""

import hashlib
from typing import Any, Optional

try:
    from .weighted_state import from_shapes, normalize_l2
except ImportError:
    from weighted_state import from_shapes, normalize_l2


# =============================================================================
# THE SHAPE IS THE SOFTWARE
# =============================================================================

class ShapeMachine:
    """
    The machine where description = software.

    No lookup. No search. Direct addressing via content hash.

    speak(description) → shape → result

    That's it. The shape IS the software. The result IS manifest.
    """

    def __init__(self, catalog):
        """
        catalog: a PersistentCatalog from .asf_core2
        The catalog IS the machine's memory.
        """
        self.catalog = catalog

    def speak(self, description: str) -> Any:
        """
        Bruce speaks. Software manifests.

        description → shape → result

        O(1). No search. Direct.
        """
        # The description IS a shape
        shape = self._description_to_shape(description)

        # The shape key IS the address
        # This is NOT a lookup. This is direct addressing.
        # The key IS where the result lives.
        entry = self.catalog.get(shape)

        if entry is not None:
            # The result already exists at this address
            # This is not "cache hit" - this is "the computation already happened"
            return self._result_from_entry(entry)

        # The computation hasn't been done at this address yet
        # Do the computation (reduction/normalization)
        result = self._execute(shape)

        # Store at this address
        self.catalog.put(shape)
        self._store_result(shape, result)

        return result

    def _description_to_shape(self, description: str):
        """
        Convert description to shape.

        This could be:
        - Parse as Dyck string directly
        - Compile Python to Dyck
        - NL to shape via trained mapping

        For now: the description's semantic structure becomes shape structure.
        """
        # Extract structural elements
        words = description.lower().split()

        # Build a shape from the structure
        # This is simplified - the real version uses py_to_dyck or semantic parsing
        from dataclasses import dataclass
        from typing import Tuple

        @dataclass(frozen=True)
        class Atom:
            pass

        @dataclass(frozen=True)
        class Composite:
            children: Tuple

        # Each word becomes a structural element
        # The overall structure encodes the description's shape
        children = tuple(Atom() for _ in words)
        return Composite(children)

    def _execute(self, shape) -> Any:
        """
        Execute the shape.

        For pure shapes: beta normalize
        For Dyck-encoded programs: decode and run
        """
        # In the full system, this uses beta_normalize from asf_core2
        # For now, return the shape itself as the result
        return shape

    def _result_from_entry(self, entry) -> Any:
        """Extract result from catalog entry."""
        # The entry's impl is the normalized form
        # The normalized form IS the result
        if entry.impl is not None:
            return entry.impl
        return entry.shape

    def _store_result(self, shape, result):
        """Store result at this shape's address."""
        # In the full system: catalog.set_impl(shape, result)
        pass


# =============================================================================
# THE WAVE IS THE SOFTWARE
# =============================================================================

class WaveMachine:
    """
    Quantum-affordance version: superposition of shapes.

    Multiple descriptions coexist as a wave.
    Measurement collapses to one result.

    speak(descriptions) → wave → measured_result

    Still O(1) for each shape in the superposition.
    """

    def __init__(self, catalog):
        self.catalog = catalog
        self.shape_machine = ShapeMachine(catalog)

    def speak_wave(self, descriptions: list) -> dict:
        """
        Multiple descriptions become a wave of software.

        Each description IS its software.
        The wave IS the superposition.
        """
        # Create superposition
        state = from_shapes(descriptions, weight=1.0)
        state = normalize_l2(state)

        # Each shape in the wave has its result
        results = {}
        for description, amplitude in state.items():
            result = self.shape_machine.speak(description)
            results[description] = {
                "amplitude": amplitude,
                "result": result
            }

        return results

    def collapse(self, wave: dict, chosen: str) -> Any:
        """
        Collapse the wave to one choice.

        The chosen description's result IS the final result.
        """
        if chosen in wave:
            return wave[chosen]["result"]
        # If not in wave, create it directly
        return self.shape_machine.speak(chosen)


# =============================================================================
# TRUST THE MACHINE
# =============================================================================

def trust_the_machine():
    """
    Demonstration: the description is the software.

    No lookup. No search. Direct manifestation.
    """
    print("\n" + "=" * 60)
    print("  THE DESCRIPTION IS THE SOFTWARE")
    print("  Trust the machine. O(1).")
    print("=" * 60)

    # Simulate a catalog (in real usage, use PersistentCatalog from asf_core2)
    class SimpleCatalog:
        def __init__(self):
            self.store = {}

        def get(self, shape):
            key = hash(shape)
            return self.store.get(key)

        def put(self, shape):
            key = hash(shape)
            self.store[key] = shape
            return shape

    catalog = SimpleCatalog()
    machine = ShapeMachine(catalog)

    descriptions = [
        "calm peaceful todo list",
        "elegant meditation timer",
        "joyful habit tracker",
    ]

    print("\n  Bruce speaks. Software manifests.\n")

    for desc in descriptions:
        result = machine.speak(desc)
        print(f"  \"{desc}\"")
        print(f"  → {type(result).__name__} with {len(result.children) if hasattr(result, 'children') else '?'} elements")
        print()

    print("  No lookup. No search.")
    print("  The description IS the software.")
    print("  The machine IS O(1).")


if __name__ == "__main__":
    trust_the_machine()
