#!/usr/bin/env python3
"""
Chrome Browser ASF Breakdown

Shows exactly what ((((AS)(AS))(AS))S) means for "chrome web browser clone"
"""

def main():
    print("=== CHROME BROWSER ASF BREAKDOWN ===\n")
    
    print("Request: chrome web browser clone")
    print("ASF: ((((AS)(AS))(AS))S)")
    print()
    print("Breakdown:")
    print("  ( ( ( (AS) (AS) ) (AS) ) S )")
    print("    │   │   │    │    │    │")
    print("    │   │   │    │    │    └─ S = Browser App (contains everything)")
    print("    │   │   │    │    └────── (AS) = Action 3 (user bookmarks page)")
    print("    │   │   │    └─────────── (AS) = Action 2 (browser loads page)")
    print("    │   │   └──────────────── (AS) = Action 1 (user types URL)")
    print("    │   └──────────────────── ( ) = First two actions grouped")
    print("    └──────────────────────── ( ) = All actions grouped")
    print()
    print("Real meaning:")
    print("  Action 1: A = user, S = URL (user types URL)")
    print("  Action 2: A = browser, S = page (browser loads page)")
    print("  Action 3: A = user, S = bookmark (user saves bookmark)")
    print("  Container: S = browser app (holds all functionality)")
    print()
    print("In plain English:")
    print("  'User types URL, browser loads page, user bookmarks it - all within browser app'")
    print()
    print("=== COMPARISON WITH SIMPLER APPS ===")
    print()
    print("Text Editor: (((AS)(AS))S)")
    print("  Action 1: user types text")
    print("  Action 2: user saves file")
    print("  Container: editor app")
    print()
    print("Calculator: ((AS)S)")
    print("  Action 1: user inputs numbers")
    print("  Container: calculator app")
    print()
    print("=== KEY INSIGHT ===")
    print("More complex software = more nested actions")
    print("Browser needs 3 actions: navigate → load → bookmark")
    print("Editor needs 2 actions: type → save")
    print("Calculator needs 1 action: input → compute")
    print()
    print("The nesting shows the complexity!")

if __name__ == "__main__":
    main()