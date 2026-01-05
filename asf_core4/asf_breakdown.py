#!/usr/bin/env python3
"""
ASF Pattern Breakdown - Concrete Examples

Shows exactly what A and S mean in real business contexts.
"""

def explain_pattern(pattern: str, example: str):
    """Break down an ASF pattern with concrete business example."""
    print(f"Pattern: {pattern}")
    print(f"Example: {example}")
    print("Breakdown:")
    
    if pattern == "(((AS)(AS))S)":
        print("  ( ( (AS) (AS) ) S )")
        print("    │   │    │    │")
        print("    │   │    │    └─ S = System/Container (inventory database)")
        print("    │   │    └────── (AS) = Action 2 (manager ships product)")  
        print("    │   └─────────── (AS) = Action 1 (worker receives product)")
        print("    └─────────────── ( ) = Actions grouped together")
        print()
        print("  Real meaning:")
        print("    A = worker (entity that acts)")
        print("    S = product (thing being acted upon)")
        print("    A = manager (entity that acts)")  
        print("    S = product (thing being acted upon)")
        print("    S = inventory system (container holding all this)")
        print()
        print("  In plain English:")
        print("    'Worker receives product AND manager ships product, all within inventory system'")
        
    elif pattern == "((AS)(AS))":
        print("  ( (AS) (AS) )")
        print("    │    │")
        print("    │    └─ (AS) = Action 2 (system processes payment)")
        print("    └────── (AS) = Action 1 (cashier scans items)")
        print()
        print("  Real meaning:")
        print("    A = cashier (entity)")
        print("    S = items (things scanned)")
        print("    A = system (entity)")
        print("    S = payment (thing processed)")
        print()
        print("  In plain English:")
        print("    'Cashier scans items THEN system processes payment'")

def main():
    print("=== ASF PATTERN BREAKDOWN ===\n")
    
    explain_pattern("(((AS)(AS))S)", "inventory management system")
    print("\n" + "="*60 + "\n")
    explain_pattern("((AS)(AS))", "point of sale transaction")
    
    print("\n=== KEY INSIGHT ===")
    print("A = Any entity that DOES something (worker, manager, system, user)")
    print("S = Any thing that HAS something done to it (product, data, request)")
    print("S = Any container that HOLDS the whole process (system, database, app)")
    print()
    print("The same letter can mean different things in different positions!")
    print("It's about ROLES in the interaction, not fixed types.")

if __name__ == "__main__":
    main()