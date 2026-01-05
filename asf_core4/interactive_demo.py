#!/usr/bin/env python3
"""
Interactive Demo Module for Speak-See-Experience

Lets users actually demo the generated software and rate their experience.
"""

import time
import random
from typing import Dict, List, Optional

class InteractiveDemo:
    """Simulates software demos for user experience rating."""
    
    def __init__(self):
        self.demo_state = {}
        
    def demo_calculator(self) -> Dict:
        """Demo a calculator app."""
        print("\n=== CALCULATOR DEMO ===")
        print("Try the calculator! Type expressions like '5 + 3' or 'quit' to exit.")
        
        interactions = 0
        start_time = time.time()
        
        while True:
            try:
                expr = input("calc> ").strip()
                if expr.lower() in ('quit', 'exit', 'q'):
                    break
                    
                # Simple calculator simulation
                if any(op in expr for op in ['+', '-', '*', '/']):
                    result = eval(expr)  # Unsafe but demo only
                    print(f"Result: {result}")
                    interactions += 1
                else:
                    print("Enter a math expression (e.g., 5 + 3)")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        duration = time.time() - start_time
        return {
            "interactions": interactions,
            "duration": duration,
            "completed": interactions > 0
        }
    
    def demo_text_editor(self) -> Dict:
        """Demo a text editor."""
        print("\n=== TEXT EDITOR DEMO ===")
        print("Type some text, then 'save' to save or 'quit' to exit.")
        
        text_buffer = []
        interactions = 0
        start_time = time.time()
        saved = False
        
        while True:
            line = input("editor> ").strip()
            if line.lower() == 'quit':
                break
            elif line.lower() == 'save':
                print(f"Saved {len(' '.join(text_buffer))} characters to file.txt")
                saved = True
                interactions += 1
            else:
                text_buffer.append(line)
                interactions += 1
                print(f"Added line. Buffer: {len(text_buffer)} lines")
        
        duration = time.time() - start_time
        return {
            "interactions": interactions,
            "duration": duration,
            "completed": saved,
            "content_created": len(' '.join(text_buffer))
        }
    
    def demo_browser(self) -> Dict:
        """Demo a web browser."""
        print("\n=== BROWSER DEMO ===")
        print("Navigate the web! Type URLs like 'google.com' or 'quit' to exit.")
        
        bookmarks = []
        history = []
        interactions = 0
        start_time = time.time()
        
        while True:
            url = input("browser> ").strip()
            if url.lower() in ('quit', 'exit', 'q'):
                break
            elif url.lower().startswith('bookmark'):
                if history:
                    bookmarks.append(history[-1])
                    print(f"Bookmarked: {history[-1]}")
                    interactions += 1
                else:
                    print("No page to bookmark")
            else:
                # Simulate loading page
                print(f"Loading {url}...")
                time.sleep(0.5)
                print(f"✓ Loaded {url}")
                history.append(url)
                interactions += 1
                
                if len(history) > 1:
                    print("Tip: Type 'bookmark' to save this page")
        
        duration = time.time() - start_time
        return {
            "interactions": interactions,
            "duration": duration,
            "completed": len(bookmarks) > 0,
            "pages_visited": len(history),
            "bookmarks_saved": len(bookmarks)
        }
    
    def demo_chat(self) -> Dict:
        """Demo a chat messenger."""
        print("\n=== CHAT DEMO ===")
        print("Chat with AI! Type messages or 'quit' to exit.")
        
        messages = []
        interactions = 0
        start_time = time.time()
        
        responses = [
            "That's interesting!",
            "Tell me more about that.",
            "I see what you mean.",
            "How does that make you feel?",
            "That's a great point!",
            "I hadn't thought of it that way."
        ]
        
        while True:
            msg = input("you> ").strip()
            if msg.lower() in ('quit', 'exit', 'q'):
                break
                
            messages.append(("user", msg))
            interactions += 1
            
            # Simulate AI response
            time.sleep(0.3)
            response = random.choice(responses)
            print(f"ai> {response}")
            messages.append(("ai", response))
        
        duration = time.time() - start_time
        return {
            "interactions": interactions,
            "duration": duration,
            "completed": len(messages) > 2,
            "messages_sent": len([m for m in messages if m[0] == "user"])
        }
    
    def run_demo(self, software_type: str) -> Dict:
        """Run appropriate demo based on software type."""
        demos = {
            "calculator": self.demo_calculator,
            "text editor": self.demo_text_editor,
            "browser": self.demo_browser,
            "chat": self.demo_chat
        }
        
        # Find matching demo
        demo_func = None
        for key, func in demos.items():
            if key in software_type.lower():
                demo_func = func
                break
        
        if not demo_func:
            return self.demo_generic(software_type)
        
        return demo_func()
    
    def demo_generic(self, software_type: str) -> Dict:
        """Generic demo for unknown software types."""
        print(f"\n=== {software_type.upper()} DEMO ===")
        print("This is a simulated demo. Press Enter to interact, 'quit' to exit.")
        
        interactions = 0
        start_time = time.time()
        
        actions = [
            "Clicked main button",
            "Opened menu",
            "Selected option",
            "Saved changes",
            "Refreshed view",
            "Closed dialog"
        ]
        
        while True:
            action = input("demo> ").strip()
            if action.lower() in ('quit', 'exit', 'q'):
                break
            
            # Simulate action
            simulated = random.choice(actions)
            print(f"✓ {simulated}")
            interactions += 1
            
            if interactions >= 5:
                print("Demo completed successfully!")
                break
        
        duration = time.time() - start_time
        return {
            "interactions": interactions,
            "duration": duration,
            "completed": interactions >= 3
        }

def rate_experience(demo_results: Dict, software_description: str) -> float:
    """Get user rating of their demo experience."""
    print(f"\n=== EXPERIENCE RATING ===")
    print(f"You just demoed: {software_description}")
    print(f"Demo stats:")
    print(f"  - Interactions: {demo_results['interactions']}")
    print(f"  - Duration: {demo_results['duration']:.1f} seconds")
    print(f"  - Completed: {'Yes' if demo_results['completed'] else 'No'}")
    
    while True:
        try:
            rating = input("\nRate your experience (0-10): ").strip()
            score = float(rating)
            if 0 <= score <= 10:
                return score / 10.0  # Normalize to 0-1
            else:
                print("Please enter a number between 0 and 10")
        except ValueError:
            print("Please enter a valid number")

def main():
    """Demo the interactive demo system."""
    demo = InteractiveDemo()
    
    software_types = [
        "simple calculator",
        "text editor like notepad", 
        "chrome web browser clone",
        "chat messenger"
    ]
    
    print("=== INTERACTIVE DEMO SYSTEM ===\n")
    
    for software in software_types:
        print(f"\nTesting: {software}")
        results = demo.run_demo(software)
        rating = rate_experience(results, software)
        
        print(f"Final rating: {rating:.1%}")
        
        # This would integrate with structural intelligence
        print(f"→ Recording satisfaction: {rating:.1%} for '{software}'")
        
        proceed = input("\nTry next demo? (y/n): ").strip().lower()
        if proceed != 'y':
            break
    
    print("\nDemo system complete!")

if __name__ == "__main__":
    main()