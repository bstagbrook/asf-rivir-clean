#!/usr/bin/env python3
"""
NL→Shape Direct Compiler using YOUR py_to_dyck
"""
import sys
sys.path.append('/Volumes/StagbrookField/stagbrook_field/asf_core4')
from py_to_dyck import compile_source, run_dyck

def nl_to_shape(nl_request: str) -> tuple[str, str]:
    """Convert NL to executable Python then to Dyck shape"""
    # Simple NL→Python mapping
    if 'double' in nl_request.lower():
        python_code = 'lambda x: x * 2'
    elif 'triple' in nl_request.lower():
        python_code = 'lambda x: x * 3'
    elif 'increment' in nl_request.lower():
        python_code = 'lambda x: x + 1'
    elif 'todo' in nl_request.lower():
        python_code = 'lambda x: []'
    else:
        python_code = 'lambda x: x'
    
    dyck = compile_source(python_code)
    return python_code, dyck

def execute_shape(dyck: str):
    """Execute shape using YOUR runtime"""
    return run_dyck(dyck)

if __name__ == "__main__":
    nl = "I want a function that doubles numbers"
    python, dyck = nl_to_shape(nl)
    result = execute_shape(dyck)
    print(f"NL: {nl}")
    print(f"Python: {python}")
    print(f"Dyck: {dyck[:50]}...")
    print(f"Result: {result}")