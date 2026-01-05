#!/usr/bin/env python3
"""
ASF Language Mapping Trainer

Teaches LLMs that natural language descriptions map naturally to ASF parens-soup
because ASF primitives (is-thing, has-thing) mirror linguistic structure.

The key insight: Software descriptions "fall like a deck of cards" into parens
because ASF is fundamentally linguistic, not mathematical.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

@dataclass
class LanguageMapping:
    """A natural language → ASF mapping example."""
    description: str
    asf_parens: str
    explanation: str
    linguistic_breakdown: Dict[str, str]

class ASFLanguageTrainer:
    """Generates training examples showing natural language → ASF alignment."""
    
    def __init__(self):
        self.examples = []
        self._generate_core_examples()
    
    def _generate_core_examples(self):
        """Generate fundamental examples showing the linguistic alignment."""
        
        # Basic is-thing examples
        self.examples.extend([
            LanguageMapping(
                description="a button",
                asf_parens="A",
                explanation="'a button' IS a thing - pure declaration from nothing",
                linguistic_breakdown={"a button": "A (is-thing)"}
            ),
            LanguageMapping(
                description="the user",
                asf_parens="A", 
                explanation="'the user' IS a thing - entity declaration",
                linguistic_breakdown={"the user": "A (is-thing)"}
            ),
            LanguageMapping(
                description="data",
                asf_parens="A",
                explanation="'data' IS a thing - abstract entity",
                linguistic_breakdown={"data": "A (is-thing)"}
            )
        ])
        
        # Basic has-thing examples  
        self.examples.extend([
            LanguageMapping(
                description="a container",
                asf_parens="S",
                explanation="'a container' HAS distinction - it contains/holds",
                linguistic_breakdown={"a container": "S (has-thing)"}
            ),
            LanguageMapping(
                description="a list",
                asf_parens="S",
                explanation="'a list' HAS items - structural container",
                linguistic_breakdown={"a list": "S (has-thing)"}
            ),
            LanguageMapping(
                description="memory",
                asf_parens="S", 
                explanation="'memory' HAS storage capacity - contains state",
                linguistic_breakdown={"memory": "S (has-thing)"}
            )
        ])
        
        # Basic does-thing (compound) examples
        self.examples.extend([
            LanguageMapping(
                description="button clicks",
                asf_parens="(AS)",
                explanation="'button clicks' = button(A) performs action on something(S)",
                linguistic_breakdown={
                    "button": "A (is-thing)",
                    "clicks": "compound action", 
                    "clicks something": "S (has-effect)"
                }
            ),
            LanguageMapping(
                description="user enters data",
                asf_parens="(AS)",
                explanation="'user enters data' = user(A) acts upon data-container(S)",
                linguistic_breakdown={
                    "user": "A (is-thing)",
                    "enters": "compound action",
                    "data": "S (has-content)"
                }
            ),
            LanguageMapping(
                description="system processes request",
                asf_parens="(AS)",
                explanation="'system processes request' = system(A) operates on request(S)",
                linguistic_breakdown={
                    "system": "A (is-thing)",
                    "processes": "compound action",
                    "request": "S (has-payload)"
                }
            )
        ])
        
        # Nested structure examples
        self.examples.extend([
            LanguageMapping(
                description="user clicks button to save data",
                asf_parens="((AS)(AS))",
                explanation="Two actions: user-clicks-button AND system-saves-data",
                linguistic_breakdown={
                    "user clicks button": "(AS) - first action",
                    "to save data": "(AS) - resulting action", 
                    "compound": "((AS)(AS)) - action sequence"
                }
            ),
            LanguageMapping(
                description="form validates input and shows error",
                asf_parens="((AS)(AS))",
                explanation="Two actions: form-validates-input AND form-shows-error",
                linguistic_breakdown={
                    "form validates input": "(AS) - validation action",
                    "shows error": "(AS) - display action",
                    "and": "sequence connector"
                }
            ),
            LanguageMapping(
                description="API receives request, processes data, returns response",
                asf_parens="(((AS)(AS))(AS))",
                explanation="Nested: (receive+process) then return",
                linguistic_breakdown={
                    "receives request": "(AS)",
                    "processes data": "(AS)", 
                    "receives+processes": "((AS)(AS))",
                    "returns response": "(AS)",
                    "full sequence": "(((AS)(AS))(AS))"
                }
            )
        ])
        
        # Complex software descriptions
        self.examples.extend([
            LanguageMapping(
                description="todo app where user adds tasks and marks them complete",
                asf_parens="(((AS)(AS))S)",
                explanation="App(S) contains user-actions: add-task AND mark-complete",
                linguistic_breakdown={
                    "user adds tasks": "(AS)",
                    "marks them complete": "(AS)",
                    "user actions": "((AS)(AS))",
                    "todo app contains": "(((AS)(AS))S)"
                }
            ),
            LanguageMapping(
                description="chat system where users send messages and receive notifications",
                asf_parens="(((AS)(AS))S)",
                explanation="System(S) contains user-actions: send-message AND receive-notification",
                linguistic_breakdown={
                    "users send messages": "(AS)",
                    "receive notifications": "(AS)", 
                    "user interactions": "((AS)(AS))",
                    "chat system contains": "(((AS)(AS))S)"
                }
            )
        ])

    def generate_training_prompt(self) -> str:
        """Generate a training prompt for LLMs."""
        prompt = """# ASF Language Mapping Training

## Core Insight
Natural language software descriptions map directly to ASF parens-soup because ASF mirrors linguistic structure:

- **A** = "is-thing" (entities, nouns, declarations)  
- **S** = "has-thing" (containers, holders, state)
- **(...)** = "does-thing" (actions, compounds, relationships)

Software descriptions "fall like a deck of cards" into parens because they're already linguistic.

## Training Examples

"""
        
        for i, example in enumerate(self.examples, 1):
            prompt += f"### Example {i}\n"
            prompt += f"**Description:** {example.description}\n"
            prompt += f"**ASF:** `{example.asf_parens}`\n"
            prompt += f"**Why:** {example.explanation}\n"
            prompt += "**Breakdown:**\n"
            for phrase, mapping in example.linguistic_breakdown.items():
                prompt += f"  - '{phrase}' → {mapping}\n"
            prompt += "\n"
        
        prompt += """## Pattern Recognition Rules

1. **Nouns/Entities** → `A` (is-thing)
   - "user", "button", "data", "system", "file"

2. **Containers/Holders** → `S` (has-thing)  
   - "list", "database", "memory", "app", "container"

3. **Actions/Verbs** → `(AS)` (does-thing)
   - "clicks", "saves", "processes", "validates", "sends"

4. **Sequences** → `((AS)(AS))` (multiple actions)
   - "and", "then", "after", "when"

5. **Containment** → `(...S)` (thing contains actions)
   - "app where...", "system that...", "program which..."

## The Magic
This isn't translation - it's recognition. Natural language IS already structured like ASF because both reflect how humans think about entities, containers, and actions.

The parens-soup emerges naturally because software descriptions are inherently compositional, just like language itself.
"""
        
        return prompt

    def generate_practice_problems(self, n: int = 10) -> List[str]:
        """Generate practice problems for the LLM to solve."""
        problems = [
            "calculator that adds numbers",
            "user uploads file to server", 
            "email client checks for new messages",
            "game where player moves character and collects items",
            "database stores records and handles queries",
            "web form validates input and submits data",
            "timer counts down and triggers alarm",
            "shopping cart holds items and calculates total",
            "text editor loads file, allows editing, and saves changes",
            "weather app fetches data and displays forecast",
            "login system authenticates user and grants access",
            "photo gallery displays images and allows filtering",
            "music player loads playlist and controls playback",
            "backup system copies files and verifies integrity",
            "search engine indexes pages and returns results"
        ]
        return problems[:n]

    def export_training_data(self, filename: str):
        """Export training data as JSON for fine-tuning."""
        training_data = []
        
        for example in self.examples:
            training_data.append({
                "input": f"Convert to ASF: {example.description}",
                "output": example.asf_parens,
                "explanation": example.explanation,
                "breakdown": example.linguistic_breakdown
            })
        
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=2)

def main():
    trainer = ASFLanguageTrainer()
    
    # Generate training prompt
    prompt = trainer.generate_training_prompt()
    print(prompt)
    
    # Generate practice problems
    print("\n## Practice Problems")
    print("Convert these descriptions to ASF parens-soup:\n")
    for i, problem in enumerate(trainer.generate_practice_problems(5), 1):
        print(f"{i}. {problem}")
    
    # Export training data
    trainer.export_training_data("asf_language_training.json")
    print(f"\nTraining data exported to asf_language_training.json")

if __name__ == "__main__":
    main()