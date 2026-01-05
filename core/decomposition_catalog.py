#!/usr/bin/env python3
"""
ASF Decomposition Catalog

Self-requests for common software, then decomposes into reusable ASF fragments.
Demonstrates how functional decomposition accelerates the flywheel.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import json

@dataclass
class ASFFragment:
    pattern: str
    description: str
    frequency: int = 0

class DecompositionCatalog:
    def __init__(self):
        self.fragments: Dict[str, ASFFragment] = {}
        self.decompositions: List[Dict] = []
        
    def request_and_decompose(self, description: str) -> str:
        """Make a software request and decompose it into ASF fragments."""
        
        # Generate ASF for the request
        asf = self._generate_asf(description)
        
        # Extract fragments
        fragments = self._extract_fragments(asf)
        
        # Update catalog
        for frag in fragments:
            if frag not in self.fragments:
                self.fragments[frag] = ASFFragment(frag, self._describe_fragment(frag))
            self.fragments[frag].frequency += 1
        
        # Record decomposition
        self.decompositions.append({
            "request": description,
            "asf": asf,
            "fragments": fragments
        })
        
        return asf
    
    def _generate_asf(self, description: str) -> str:
        """Generate ASF for any software description."""
        desc = description.lower()
        
        # Consumer software patterns
        if "browser" in desc or "chrome" in desc:
            return "((((AS)(AS))(AS))S)"  # user navigates/searches/bookmarks, browser contains
        elif "text editor" in desc or "notepad" in desc:
            return "(((AS)(AS))S)"  # user types/saves, editor contains
        elif "media player" in desc or "music" in desc or "video" in desc:
            return "(((AS)(AS))S)"  # user plays/pauses, player contains
        elif "photo editor" in desc or "image" in desc:
            return "((((AS)(AS))(AS))S)"  # user opens/edits/saves, editor contains
        elif "game" in desc:
            return "(((AS)(AS))S)"  # player moves/acts, game contains
        elif "calculator" in desc:
            return "((AS)S)"  # user inputs, calculator computes
        elif "file manager" in desc or "explorer" in desc:
            return "(((AS)(AS))S)"  # user browses/copies, manager contains
        elif "email client" in desc or "mail" in desc:
            return "(((AS)(AS))S)"  # user reads/sends, client contains
        elif "chat" in desc or "messenger" in desc:
            return "(((AS)(AS))S)"  # user sends/receives, chat contains
        elif "terminal" in desc or "command" in desc:
            return "((AS)S)"  # user types, terminal executes
        
        # Business software patterns
        elif "crm" in desc or "customer" in desc:
            return "((((AS)(AS))(AS))S)"  # track/update/report customers, system contains
        elif "inventory" in desc or "warehouse" in desc:
            return "(((AS)(AS))S)"  # receive/ship items, system tracks
        elif "payroll" in desc or "hr" in desc:
            return "((((AS)(AS))(AS))S)"  # enter/calculate/distribute pay, system contains
        elif "accounting" in desc or "invoice" in desc:
            return "(((AS)(AS))S)"  # create/send invoices, system tracks
        elif "scheduling" in desc or "appointment" in desc:
            return "(((AS)(AS))S)"  # book/cancel appointments, calendar contains
        elif "pos" in desc or "point of sale" in desc:
            return "((AS)(AS))"  # scan items, process payment
        elif "erp" in desc or "enterprise" in desc:
            return "(((((AS)(AS))(AS))(AS))S)"  # complex multi-module system
        elif "reporting" in desc or "dashboard" in desc:
            return "((AS)S)"  # query data, display results
        elif "workflow" in desc or "approval" in desc:
            return "(((AS)(AS))S)"  # submit/approve requests, system routes
        elif "timesheet" in desc or "time tracking" in desc:
            return "(((AS)(AS))S)"  # clock in/out, system calculates
        elif "expense" in desc:
            return "(((AS)(AS))S)"  # submit/approve expenses, system tracks
        elif "project management" in desc:
            return "((((AS)(AS))(AS))S)"  # create/assign/track tasks, project contains
        elif "help desk" in desc or "ticketing" in desc:
            return "(((AS)(AS))S)"  # create/resolve tickets, system tracks
        elif "document management" in desc:
            return "(((AS)(AS))S)"  # upload/organize documents, system stores
        elif "compliance" in desc or "audit" in desc:
            return "((AS)S)"  # generate reports, system validates
        else:
            return "((AS)S)"
    
    def _extract_fragments(self, asf: str) -> List[str]:
        """Extract reusable fragments from ASF string."""
        fragments = []
        
        # Add the full pattern
        fragments.append(asf)
        
        # Extract common sub-patterns
        if "AS" in asf:
            fragments.append("(AS)")  # Basic action
        if "((AS)(AS))" in asf:
            fragments.append("((AS)(AS))")  # Action sequence
        if "S)" in asf:
            fragments.append("S")  # Container
        if "A" in asf:
            fragments.append("A")  # Entity
        
        return list(set(fragments))  # Remove duplicates
    
    def _describe_fragment(self, fragment: str) -> str:
        """Describe what an ASF fragment represents."""
        descriptions = {
            "A": "Entity (user, system, player, file)",
            "S": "Container (app, database, browser, game state)", 
            "(AS)": "Basic Action (click, type, play, save)",
            "((AS)(AS))": "User Workflow (navigate then search, open then edit)",
            "(((AS)(AS))S)": "Interactive App (browser, editor, player, chat)",
            "((AS)S)": "Simple Tool (calculator, terminal, viewer)",
            "((((AS)(AS))(AS))S)": "Complex App (photo editor, browser with extensions)",
            "(((((AS)(AS))(AS))(AS))S)": "Enterprise System (ERP, multi-module)"
        }
        return descriptions.get(fragment, f"Business Pattern: {fragment}")

def main():
    catalog = DecompositionCatalog()
    
    # Mixed consumer and business software requests
    requests = [
        "chrome web browser clone",
        "text editor like notepad",
        "music player app",
        "photo editing software",
        "simple calculator",
        "file manager",
        "email client",
        "chat messenger",
        "terminal emulator",
        "CRM system for sales team",
        "inventory management system",
        "payroll software",
        "project management tool",
        "help desk system",
        "accounting software"
    ]
    
    print("=== CONSUMER & BUSINESS SOFTWARE DECOMPOSITION ===\n")
    
    # Process each request
    for req in requests:
        asf = catalog.request_and_decompose(req)
        print(f"Request: {req}")
        print(f"ASF: {asf}")
        print()
    
    # Show fragment frequency (flywheel effect)
    print("=== FRAGMENT FREQUENCY (Flywheel Effect) ===\n")
    sorted_fragments = sorted(catalog.fragments.items(), 
                            key=lambda x: x[1].frequency, reverse=True)
    
    for pattern, fragment in sorted_fragments:
        print(f"{fragment.frequency:2d}x  {pattern:15s}  {fragment.description}")
    
    # Show cache hit simulation
    print(f"\n=== CACHE HIT SIMULATION ===\n")
    total_fragments = sum(f.frequency for f in catalog.fragments.values())
    reusable_fragments = sum(f.frequency for f in catalog.fragments.values() if f.frequency > 1)
    
    cache_hit_rate = (reusable_fragments / total_fragments) * 100
    print(f"Total fragments used: {total_fragments}")
    print(f"Reusable fragments: {reusable_fragments}")
    print(f"Cache hit rate: {cache_hit_rate:.1f}%")
    
    # Show most valuable fragments for caching
    print(f"\n=== HIGH-VALUE CACHE TARGETS ===\n")
    high_value = [(p, f) for p, f in sorted_fragments if f.frequency >= 3]
    
    for pattern, fragment in high_value:
        savings = fragment.frequency - 1  # First use is cache miss
        print(f"{pattern:15s} → {savings} cache hits saved")
    
    # Export catalog
    catalog_data = {
        "fragments": {p: {"description": f.description, "frequency": f.frequency} 
                     for p, f in catalog.fragments.items()},
        "decompositions": catalog.decompositions
    }
    
    with open("asf_decomposition_catalog.json", "w") as f:
        json.dump(catalog_data, f, indent=2)
    
    print(f"\nCatalog exported to asf_decomposition_catalog.json")

if __name__ == "__main__":
    main()