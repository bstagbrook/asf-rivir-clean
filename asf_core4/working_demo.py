#!/usr/bin/env python3
"""
Working Demo: Python→ASF→Component Database

Shows the full pipeline with simple Python programs that transpile successfully.
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

try:
    from .py_to_dyck import compile_source, run_source
except ImportError:
    from py_to_dyck import compile_source, run_source

# Load ASF Core 2
import importlib.util
def load_asf_core2():
    path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@dataclass
class ComponentPattern:
    name: str
    asf_pattern: str
    description: str
    frequency: int = 1

class SimpleDecomposer:
    def __init__(self):
        self.core2 = load_asf_core2()
        self.catalog = self.core2.PersistentCatalog("demo_components.db")
        self.patterns: Dict[str, ComponentPattern] = {}
    
    def analyze_program(self, name: str, source: str):
        """Analyze a simple Python program."""
        print(f"\n=== ANALYZING: {name} ===")
        print(f"Source:\n{source}")
        
        try:
            # Transpile to Dyck/ASF
            dyck = compile_source(source)
            print(f"Dyck: {dyck[:100]}...")
            
            # Parse to ASF shape
            shape = self.core2.parse_dyck(dyck)
            asf_repr = self.core2.serialize(shape).decode()
            print(f"ASF: {asf_repr}")
            
            # Store in catalog
            entry = self.catalog.put(shape)
            print(f"Shape key: {self.core2.key(shape)[:16]}...")
            
            # Extract semantic patterns
            patterns = self._extract_patterns(name, source, asf_repr)
            for pattern in patterns:
                self._record_pattern(pattern)
                print(f"Component: {pattern.name} → {pattern.asf_pattern}")
            
            # Test execution
            result = run_source(source)
            print(f"Execution result: {result}")
            
            return True
            
        except Exception as e:
            print(f"Failed: {e}")
            return False
    
    def _extract_patterns(self, name: str, source: str, asf: str) -> List[ComponentPattern]:
        """Extract reusable patterns from the program."""
        patterns = []
        
        # Basic computation pattern
        if any(op in source for op in ['+', '-', '*', '/']):
            patterns.append(ComponentPattern(
                name=f"{name}_arithmetic",
                asf_pattern="(AS)",
                description="Basic arithmetic operation"
            ))
        
        # Conditional logic pattern
        if 'if ' in source:
            patterns.append(ComponentPattern(
                name=f"{name}_conditional",
                asf_pattern="((AS)(AS))",
                description="Conditional branching logic"
            ))
        
        # Function definition pattern
        if 'def ' in source:
            patterns.append(ComponentPattern(
                name=f"{name}_function",
                asf_pattern="(((AS)(AS))S)",
                description="Function definition with logic"
            ))
        
        # Overall program pattern based on ASF complexity
        complexity = asf.count('(')
        if complexity <= 3:
            patterns.append(ComponentPattern(
                name=f"{name}_simple_program",
                asf_pattern=asf,
                description="Simple single-purpose program"
            ))
        else:
            patterns.append(ComponentPattern(
                name=f"{name}_complex_program", 
                asf_pattern=asf,
                description="Multi-step program with logic"
            ))
        
        return patterns
    
    def _record_pattern(self, pattern: ComponentPattern):
        """Record pattern in database."""
        if pattern.name in self.patterns:
            self.patterns[pattern.name].frequency += 1
        else:
            self.patterns[pattern.name] = pattern
    
    def show_component_database(self):
        """Show the accumulated component database."""
        print(f"\n=== COMPONENT DATABASE ===")
        print(f"Total patterns: {len(self.patterns)}")
        
        sorted_patterns = sorted(self.patterns.values(), 
                               key=lambda p: p.frequency, reverse=True)
        
        for pattern in sorted_patterns:
            print(f"{pattern.frequency}x {pattern.asf_pattern:20s} {pattern.description}")

def main():
    decomposer = SimpleDecomposer()
    
    # Simple programs that the transpiler can handle
    programs = [
        ("add_function", """
def add(x):
    return x + 1

add(5)
"""),
        
        ("conditional", """
def check(x):
    return 10 if x > 5 else 0

check(7)
"""),
        
        ("calculator", """
def calc(x):
    return x * 2 + 1

calc(3)
"""),
        
        ("nested_logic", """
def process(x):
    return (x + 1) * 2 if x > 0 else x - 1

process(4)
"""),
        
        ("simple_math", """
def compute(a):
    return a + a * 2

compute(3)
""")
    ]
    
    print("=== PYTHON→ASF COMPONENT EXTRACTION DEMO ===")
    
    successful = 0
    for name, source in programs:
        if decomposer.analyze_program(name, source):
            successful += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Programs analyzed: {len(programs)}")
    print(f"Successful transpilations: {successful}")
    
    decomposer.show_component_database()
    
    # Show how this would work in speak-see-experience
    print(f"\n=== SPEAK-SEE-EXPERIENCE INTEGRATION ===")
    print("User says: 'I want a simple calculator'")
    print("System looks up: calculator_* patterns")
    print("Found patterns:")
    
    calc_patterns = [p for p in decomposer.patterns.values() 
                    if 'calc' in p.name or 'arithmetic' in p.description]
    
    for pattern in calc_patterns:
        print(f"  - {pattern.name}: {pattern.asf_pattern}")
    
    print("System generates working calculator from these ASF patterns")
    print("User demos the actual calculator and rates their experience!")
    
    # Export for integration
    export_data = {
        "patterns": {name: asdict(pattern) for name, pattern in decomposer.patterns.items()},
        "catalog_stats": decomposer.catalog.stats()
    }
    
    with open("component_database.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nComponent database exported to component_database.json")

if __name__ == "__main__":
    main()