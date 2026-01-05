#!/usr/bin/env python3
"""
NL→ASF Direct Compiler
The description IS the software. O(1) manifestation.
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from asf1 import parse_sexpr, encode_sexpr, serialize_dyck
from py_to_dyck import compile_source

@dataclass
class Intent:
    action: str
    target: str
    modifiers: List[str]
    context: Dict[str, str]

class NLToASF:
    def __init__(self):
        # Intent patterns
        self.patterns = {
            'create': r'(?:create|make|build|generate)\s+(?:a\s+)?(.+)',
            'list': r'(?:list|show|display)\s+(.+)',
            'track': r'(?:track|monitor|log)\s+(.+)',
            'manage': r'(?:manage|organize|handle)\s+(.+)',
            'calculate': r'(?:calculate|compute|find)\s+(.+)',
        }
        
        # Style hints
        self.styles = {
            'simple': ['simple', 'minimal', 'clean', 'basic'],
            'elegant': ['elegant', 'beautiful', 'graceful', 'refined'],
            'calm': ['calm', 'peaceful', 'serene', 'quiet'],
            'focused': ['focused', 'concentrated', 'direct'],
        }

    def parse_intent(self, nl: str) -> Intent:
        """Extract structured intent from natural language"""
        nl = nl.lower().strip()
        
        # Find action
        action = 'create'  # default
        for act, pattern in self.patterns.items():
            if re.search(pattern, nl):
                action = act
                break
        
        # Extract target
        target = re.sub(r'(?:create|make|build|generate|list|show|display|track|monitor|log|manage|organize|handle|calculate|compute|find)\s+(?:a\s+)?', '', nl)
        target = re.sub(r'\s+(?:that|which|with).*', '', target)
        
        # Extract modifiers
        modifiers = []
        for style, keywords in self.styles.items():
            if any(kw in nl for kw in keywords):
                modifiers.append(style)
        
        # Context
        context = {}
        if 'todo' in nl or 'task' in nl:
            context['type'] = 'productivity'
        elif 'note' in nl or 'journal' in nl:
            context['type'] = 'writing'
        elif 'track' in nl or 'log' in nl:
            context['type'] = 'tracking'
            
        return Intent(action, target, modifiers, context)

    def to_asf1(self, nl: str) -> str:
        """NL → ASF1 S-expression → Dyck"""
        intent = self.parse_intent(nl)
        
        # Build S-expression based on intent
        if intent.action == 'create':
            if 'todo' in intent.target or 'task' in intent.target:
                sexpr_str = '(app (create todo-list) (with simple elegant))'
            elif 'note' in intent.target:
                sexpr_str = '(app (create note-app) (with minimal focused))'
            elif 'track' in intent.target:
                sexpr_str = '(app (create tracker) (with data logging))'
            else:
                sexpr_str = f'(app (create {intent.target.replace(" ", "-")}) (with {" ".join(intent.modifiers)}))'
        else:
            sexpr_str = f'({intent.action} {intent.target.replace(" ", "-")})'
        
        # Convert to Dyck
        sexpr = parse_sexpr(sexpr_str)
        shape = encode_sexpr(sexpr)
        dyck = serialize_dyck(shape)
        return dyck

    def to_asf2(self, nl: str) -> str:
        """NL → ASF2 Python AST → Dyck"""
        intent = self.parse_intent(nl)
        
        # Generate Python code based on intent
        if intent.action == 'create':
            if 'todo' in intent.target:
                python_code = '''
def todo_app():
    tasks = []
    def add_task(task):
        tasks.append(task)
    return add_task'''
            elif 'note' in intent.target:
                python_code = '''
def note_app():
    notes = []
    def add_note(note):
        notes.append(note)
    return add_note'''
            else:
                python_code = f'''
def {intent.target.replace(" ", "_")}():
    return "{intent.target}"'''
        else:
            python_code = f'''
def {intent.action}():
    return "{intent.target}"'''
        
        # Convert to Dyck via compile_source
        dyck = compile_source(python_code.strip())
        return dyck

    def manifest(self, nl: str, prefer_asf2: bool = True) -> Tuple[str, str]:
        """The description IS the software. Direct manifestation."""
        # Content hash of the description
        desc_hash = hashlib.sha256(nl.encode()).hexdigest()[:16]
        
        # Direct compilation
        if prefer_asf2:
            dyck = self.to_asf2(nl)
        else:
            dyck = self.to_asf1(nl)
        
        return desc_hash, dyck

if __name__ == "__main__":
    compiler = NLToASF()
    
    test_cases = [
        "Create a simple todo list",
        "Make a calm meditation timer", 
        "Build an elegant note-taking app",
        "Track my daily habits",
    ]
    
    print("=== NL→ASF Direct Compilation ===")
    for nl in test_cases:
        print(f"\nInput: {nl}")
        hash_id, dyck = compiler.manifest(nl)
        print(f"Hash:  {hash_id}")
        print(f"Dyck:  {dyck}")