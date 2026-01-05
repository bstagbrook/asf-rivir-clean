#!/usr/bin/env python3
"""
LLM-Powered Python→ASF Decomposition System

Takes real Python programs (brew/pip packages), transpiles to ASF2,
then uses LLM to decompose step-by-step with semantic naming.
Builds comprehensive database of real-world software patterns.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import time

try:
    from .complete_py_to_dyck import compile_source
    from .shape_runtime import run_python_source
except ImportError:
    from complete_py_to_dyck import compile_source
    from shape_runtime import run_python_source

# Load ASF Core 2
import importlib.util
def load_asf_core2():
    path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@dataclass
class DecomposedComponent:
    """A named component extracted from ASF."""
    name: str
    asf_pattern: str
    python_source: str
    semantic_role: str
    complexity_level: int
    dependencies: List[str]
    frequency: int = 1

@dataclass
class ProgramAnalysis:
    """Complete analysis of a Python program."""
    program_name: str
    source_code: str
    asf_representation: str
    components: List[DecomposedComponent]
    total_complexity: int
    program_type: str  # "cli_tool", "web_app", "library", etc.

class LLMDecomposer:
    """Uses LLM to semantically decompose ASF patterns."""
    
    def __init__(self, db_path: str = "asf_decomposition.db"):
        self.core2 = load_asf_core2()
        self.catalog = self.core2.PersistentCatalog(db_path)
        self.component_db: Dict[str, DecomposedComponent] = {}
        
    def analyze_python_program(self, source_code: str, program_name: str) -> ProgramAnalysis:
        """Complete analysis pipeline for a Python program."""
        
        # Step 1: Transpile to ASF
        try:
            dyck = compile_source(source_code)
            asf_shape = self.core2.parse_dyck(dyck)
            asf_repr = self.core2.serialize(asf_shape).decode()
        except Exception as e:
            print(f"Transpilation failed for {program_name}: {e}")
            return None
            
        # Step 2: Store in catalog
        entry = self.catalog.put(asf_shape)
        
        # Step 3: LLM decomposition (simulated for now)
        components = self._llm_decompose(source_code, asf_repr, program_name)
        
        # Step 4: Classify program type
        program_type = self._classify_program_type(source_code)
        
        # Step 5: Calculate complexity
        complexity = self._calculate_complexity(asf_repr)
        
        analysis = ProgramAnalysis(
            program_name=program_name,
            source_code=source_code,
            asf_representation=asf_repr,
            components=components,
            total_complexity=complexity,
            program_type=program_type
        )
        
        # Step 6: Update component database
        self._update_component_db(components)
        
        return analysis
    
    def _llm_decompose(self, source: str, asf: str, name: str) -> List[DecomposedComponent]:
        """Simulate LLM decomposition of ASF into named components."""
        
        # This would be a real LLM call in production
        # For now, we'll use heuristic decomposition
        
        components = []
        
        # Extract imports as dependencies
        dependencies = []
        for line in source.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                dep = line.strip().split()[1].split('.')[0]
                dependencies.append(dep)
        
        # Analyze source structure
        if 'def main(' in source or 'if __name__' in source:
            components.append(DecomposedComponent(
                name=f"{name}_main_entry",
                asf_pattern="(AS)",
                python_source="main() function",
                semantic_role="program_entry_point",
                complexity_level=1,
                dependencies=dependencies
            ))
        
        if 'class ' in source:
            class_count = source.count('class ')
            components.append(DecomposedComponent(
                name=f"{name}_class_definitions",
                asf_pattern="((AS)S)",
                python_source=f"{class_count} class definitions",
                semantic_role="data_structures",
                complexity_level=2,
                dependencies=dependencies
            ))
        
        if 'def ' in source:
            func_count = source.count('def ') - (1 if 'def main(' in source else 0)
            if func_count > 0:
                components.append(DecomposedComponent(
                    name=f"{name}_functions",
                    asf_pattern="(((AS)(AS))S)",
                    python_source=f"{func_count} function definitions",
                    semantic_role="business_logic",
                    complexity_level=3,
                    dependencies=dependencies
                ))
        
        if any(keyword in source for keyword in ['input(', 'print(', 'sys.argv']):
            components.append(DecomposedComponent(
                name=f"{name}_io_operations",
                asf_pattern="((AS)(AS))",
                python_source="I/O operations",
                semantic_role="user_interface",
                complexity_level=2,
                dependencies=dependencies
            ))
        
        if any(keyword in source for keyword in ['open(', 'with open', 'file']):
            components.append(DecomposedComponent(
                name=f"{name}_file_operations",
                asf_pattern="(((AS)(AS))S)",
                python_source="File operations",
                semantic_role="data_persistence",
                complexity_level=3,
                dependencies=dependencies
            ))
        
        return components
    
    def _classify_program_type(self, source: str) -> str:
        """Classify the type of Python program."""
        if 'flask' in source.lower() or 'django' in source.lower():
            return "web_app"
        elif 'argparse' in source or 'sys.argv' in source:
            return "cli_tool"
        elif 'class ' in source and 'def ' in source:
            return "library"
        elif 'import requests' in source or 'urllib' in source:
            return "network_client"
        elif 'sqlite3' in source or 'database' in source.lower():
            return "data_processor"
        else:
            return "script"
    
    def _calculate_complexity(self, asf: str) -> int:
        """Calculate complexity based on ASF nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for char in asf:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
                
        return max_depth
    
    def _update_component_db(self, components: List[DecomposedComponent]):
        """Update the component frequency database."""
        for comp in components:
            key = f"{comp.semantic_role}:{comp.asf_pattern}"
            if key in self.component_db:
                self.component_db[key].frequency += 1
            else:
                self.component_db[key] = comp

class RealProgramHarvester:
    """Harvests real Python programs from brew/pip packages."""
    
    def __init__(self):
        self.decomposer = LLMDecomposer()
        
    def harvest_brew_programs(self) -> List[str]:
        """Get list of Python programs installed via brew."""
        try:
            # Find Python scripts in common brew locations
            brew_paths = [
                "/opt/homebrew/bin",
                "/usr/local/bin"
            ]
            
            programs = []
            for path in brew_paths:
                if Path(path).exists():
                    for file in Path(path).iterdir():
                        if file.is_file() and file.suffix == '.py':
                            programs.append(str(file))
                        elif file.is_file():
                            # Check if it's a Python script with shebang
                            try:
                                with open(file, 'r') as f:
                                    first_line = f.readline()
                                    if 'python' in first_line:
                                        programs.append(str(file))
                            except:
                                pass
            
            return programs[:10]  # Limit for demo
            
        except Exception as e:
            print(f"Error harvesting brew programs: {e}")
            return []
    
    def harvest_pip_packages(self) -> List[Tuple[str, str]]:
        """Get source code from popular pip packages."""
        
        # Sample programs to analyze
        sample_programs = [
            ("simple_calculator", """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def main():
    while True:
        try:
            a = float(input("First number: "))
            op = input("Operation (+, -, *, /): ")
            b = float(input("Second number: "))
            
            if op == '+':
                result = add(a, b)
            elif op == '-':
                result = subtract(a, b)
            elif op == '*':
                result = a * b
            elif op == '/':
                result = a / b if b != 0 else "Error: Division by zero"
            else:
                result = "Invalid operation"
                
            print(f"Result: {result}")
            
            if input("Continue? (y/n): ").lower() != 'y':
                break
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    main()
"""),
            
            ("file_organizer", """
import os
import shutil
from pathlib import Path

class FileOrganizer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        
    def organize_by_extension(self):
        for file in self.source_dir.iterdir():
            if file.is_file():
                ext = file.suffix.lower()
                if ext:
                    dest_dir = self.source_dir / ext[1:]  # Remove the dot
                    dest_dir.mkdir(exist_ok=True)
                    shutil.move(str(file), str(dest_dir / file.name))
                    
    def organize_by_date(self):
        for file in self.source_dir.iterdir():
            if file.is_file():
                mtime = file.stat().st_mtime
                date_str = time.strftime("%Y-%m", time.localtime(mtime))
                dest_dir = self.source_dir / date_str
                dest_dir.mkdir(exist_ok=True)
                shutil.move(str(file), str(dest_dir / file.name))

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: file_organizer.py <directory>")
        sys.exit(1)
        
    organizer = FileOrganizer(sys.argv[1])
    
    choice = input("Organize by (e)xtension or (d)ate? ")
    if choice.lower() == 'e':
        organizer.organize_by_extension()
    elif choice.lower() == 'd':
        organizer.organize_by_date()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
"""),
            
            ("web_scraper", """
import requests
from bs4 import BeautifulSoup
import json
import time

class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        
    def scrape_page(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
            
    def extract_links(self, soup):
        if soup:
            return [a.get('href') for a in soup.find_all('a', href=True)]
        return []
        
    def save_data(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    scraper = WebScraper("https://example.com")
    
    urls = input("Enter URLs (comma-separated): ").split(',')
    all_links = []
    
    for url in urls:
        url = url.strip()
        print(f"Scraping {url}...")
        soup = scraper.scrape_page(url)
        links = scraper.extract_links(soup)
        all_links.extend(links)
        time.sleep(1)  # Be respectful
        
    scraper.save_data(all_links, 'scraped_links.json')
    print(f"Saved {len(all_links)} links to scraped_links.json")

if __name__ == "__main__":
    main()
""")
        ]
        
        return sample_programs
    
    def analyze_all_programs(self):
        """Analyze all harvested programs."""
        print("=== REAL PROGRAM ASF DECOMPOSITION ===\n")
        
        # Get sample programs (in production, would harvest real ones)
        programs = self.harvest_pip_packages()
        
        analyses = []
        for name, source in programs:
            print(f"Analyzing: {name}")
            analysis = self.decomposer.analyze_python_program(source, name)
            if analysis:
                analyses.append(analysis)
                self._print_analysis(analysis)
                print()
        
        # Show component frequency analysis
        self._show_component_frequency()
        
        return analyses
    
    def _print_analysis(self, analysis: ProgramAnalysis):
        """Print analysis results."""
        print(f"  Program: {analysis.program_name}")
        print(f"  Type: {analysis.program_type}")
        print(f"  Complexity: {analysis.total_complexity}")
        print(f"  ASF: {analysis.asf_representation[:50]}...")
        print(f"  Components:")
        for comp in analysis.components:
            print(f"    - {comp.name}: {comp.asf_pattern} ({comp.semantic_role})")
    
    def _show_component_frequency(self):
        """Show most frequent components across all programs."""
        print("=== COMPONENT FREQUENCY ANALYSIS ===\n")
        
        sorted_components = sorted(
            self.decomposer.component_db.items(),
            key=lambda x: x[1].frequency,
            reverse=True
        )
        
        for key, comp in sorted_components:
            print(f"{comp.frequency}x {comp.asf_pattern:15s} {comp.semantic_role}")
        
        print(f"\nTotal unique components: {len(self.decomposer.component_db)}")

def main():
    harvester = RealProgramHarvester()
    analyses = harvester.analyze_all_programs()
    
    # Export results
    export_data = {
        "analyses": [asdict(a) for a in analyses],
        "components": {k: asdict(v) for k, v in harvester.decomposer.component_db.items()}
    }
    
    with open("real_program_analysis.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nExported analysis to real_program_analysis.json")

if __name__ == "__main__":
    main()